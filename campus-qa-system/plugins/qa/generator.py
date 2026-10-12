"""generator：答案生成插件 —— LLM 优先（可配置），抽取式兜底，全部带来源标注。

设计依据（Rubric「Prompt 与 RAG 工程」+ 概念卡 3/6/8）
------------------------------------------------------
- **Prompt 结构**：系统指令（角色与规则）+ 上下文（编号的检索段落）+ 用户问题
  + 输出格式要求（答案 + 来源段落编号），与概念卡 3 一一对应。
- **拒答前置**：本插件不做拒答判定——调用方（fusion_qa）已在生成之前完成
  三态判定（decision.py）。Prompt 里的「资料不足时明确说不确定」只是第二道
  软防线（概念卡 8：指令是软约束，阈值才是硬约束）。
- **双后端**：
    - ``llm``   配置了 OpenAI 兼容接口（base_url/api_key/model）时使用，
                仅用标准库 urllib，无新增依赖；
    - ``extractive`` 无 LLM 时的兜底：从 top 检索块中抽取与问句重叠度最高
                的句子组装答案（零依赖、可复现，评测与冒烟用它跑）。
    - ``auto``（默认）有 LLM 配置走 LLM，调用失败自动回落抽取式。
- **溯源**：无论哪个后端，答案末尾必须标注来源段落（评分表「回答准确性与
  溯源」要求标注来源段落，抽查可溯源）。

方法签名
--------
``generate`` 参数：``{"question": str, "hits": [...], "triples"?: [...]}``，
返回 ``{"answer": str, "mode": "llm"|"extractive", "citations": [...]}``。
"""

from __future__ import annotations

import json
import re
import urllib.request
from typing import Any, Dict, List, Optional, Sequence, Set

from base.sdk.base import Plugin, PluginMeta

from .decision import TAU  # noqa: F401  (阈值统一从 decision 取，避免两处不一致)

_STOP_CHARS = set("的了呢吗啊吧么是在有和与及或者对于从被把请问您我你他她它们这那" " \t，。？！？!,:：；;、（）()【】\"'")

# LLM Prompt —— 概念卡 3 的四段结构。证据块按 [n] 编号，要求答案末尾标注来源。
SYSTEM_PROMPT = (
    "你是《操作系统》课程的助教，负责回答学生的课程问题。严格遵守以下规则：\n"
    "1. 只能依据下面给出的【课程资料】回答，不得使用资料之外的知识；\n"
    "2. 如果资料不足以回答问题，必须明确说“课程资料中未找到相关内容”，"
    "不要猜测、不要编造；\n"
    "3. 回答要简洁准确，先给结论，再给要点；\n"
    "4. 回答的最后一行必须标注来源段落编号，格式为：来源：[1] 或 来源：[1][2]。"
)


class GeneratorPlugin(Plugin):
    meta = PluginMeta(
        name="generator",
        version="0.1.0",
        description="答案生成：LLM优先（OpenAI兼容接口），抽取式兜底，全部答案带来源标注",
        group="qa",
    )

    def __init__(self) -> None:
        self._mode = "auto"
        self._llm: Dict[str, Any] = {}
        self._max_evidence = 3
        self._max_sentences = 3

    # ------------------------------------------------------------------
    # Plugin 契约
    # ------------------------------------------------------------------
    def initialize(self, config: Dict[str, Any] | None = None) -> None:
        cfg = config or {}
        self._mode = str(cfg.get("mode", "auto"))          # auto | llm | extractive
        self._llm = dict(cfg.get("llm") or {})
        self._max_evidence = int(cfg.get("max_evidence", 3))
        self._max_sentences = int(cfg.get("max_sentences", 3))

    def methods(self) -> List[str]:
        return ["generate", "info"]

    def call(self, method: str, params: Dict[str, Any]) -> Any:
        if method == "info":
            return self.info()
        if method == "generate":
            return self.generate(
                params.get("question", ""),
                params.get("hits") or [],
                params.get("triples") or [],
            )
        raise KeyError(method)

    # ------------------------------------------------------------------
    # 证据组装（两个后端共用）
    # ------------------------------------------------------------------
    @staticmethod
    def _hit_label(h: Dict[str, Any], idx: int) -> str:
        title = (h.get("title") or "").strip()
        src = h.get("source") or h.get("filename") or ""
        label = title or src or f"chunk{h.get('chunk_id', idx)}"
        return f"[{idx}] {label}"

    def _build_context(self, hits: Sequence[Dict[str, Any]],
                       triples: Sequence[Any]) -> str:
        lines: List[str] = ["【课程资料】"]
        for i, h in enumerate(hits[: self._max_evidence], 1):
            content = re.sub(r"\s+", " ", (h.get("content") or "")).strip()
            lines.append(f"{self._hit_label(h, i)} {content}")
        if triples:
            lines.append("")
            lines.append("【知识图谱】（与问题实体相关的关系，可作佐证）")
            for t in triples[:6]:
                if isinstance(t, dict):
                    lines.append(
                        f"- {t.get('source')} --{t.get('rel')}--> {t.get('target')}"
                    )
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # LLM 后端（OpenAI 兼容 /chat/completions，标准库实现）
    # ------------------------------------------------------------------
    def _llm_available(self) -> bool:
        return bool(self._llm.get("base_url") and self._llm.get("api_key"))

    def _llm_generate(self, question: str, context: str) -> Optional[str]:
        base = str(self._llm["base_url"]).rstrip("/")
        url = f"{base}/chat/completions"
        payload = {
            "model": self._llm.get("model", "gpt-4o-mini"),
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"{context}\n\n【问题】\n{question}"},
            ],
            "temperature": self._llm.get("temperature", 0.1),
        }
        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self._llm['api_key']}",
            },
            method="POST",
        )
        timeout = float(self._llm.get("timeout", 30))
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                body = json.loads(resp.read().decode("utf-8"))
            text = body["choices"][0]["message"]["content"]
            return (text or "").strip() or None
        except Exception:
            return None

    # ------------------------------------------------------------------
    # 抽取式后端（零依赖兜底）
    # ------------------------------------------------------------------
    @staticmethod
    def _query_terms(question: str) -> Set[str]:
        return {c for c in question if c not in _STOP_CHARS}

    def _extractive_generate(self, question: str,
                             hits: Sequence[Dict[str, Any]]) -> str:
        """以「节」为单位的抽取式组装（防幻觉的下限实现）。

        heading 分块下「一块 = 一节」，超长节会被再切成多块（同 title）。
        组装规则：
          1. 主体 = top-1 命中所在**节**的全部块（按 title 合并，跨块枚举不丢要点）；
          2. 第二节（不同 title）：分数接近 top1 时整节并入（跨节问题），
             否则补一句与问句重叠最高的句子（≥6 字，过滤被切断的边界碎片）；
          3. 不跨块拼接、不改写 —— 零编造空间。
        """
        terms = self._query_terms(question)
        body_parts: List[str] = []
        citations: List[str] = []

        def _content(h: Dict[str, Any]) -> str:
            return re.sub(r"\s+", " ", (h.get("content") or "")).strip()

        top = hits[0]
        top_title = (top.get("title") or "").strip()
        top_score = float(top.get("score") or 0.0)

        # 1) top-1 所在节的全部块
        for i, h in enumerate(hits[: self._max_evidence + 2], 1):
            if (h.get("title") or "").strip() != top_title:
                continue
            c = _content(h)
            if c:
                body_parts.append(c)
        if body_parts:
            citations.append(self._hit_label(top, 1))

        # 2) 第二节：整节 or 单句
        used_titles = {top_title}
        for h in hits[: self._max_evidence + 2]:
            title = (h.get("title") or "").strip()
            if not title or title in used_titles:
                continue
            used_titles.add(title)
            c = _content(h)
            if not c:
                continue
            score = float(h.get("score") or 0.0)
            if top_score > 0 and score >= 0.92 * top_score:
                body_parts.append(c)
                citations.append(self._hit_label(h, 2))
            else:
                sentences = [s.strip() for s in re.split(r"[。；;]+", c) if len(s.strip()) >= 6]
                scored = sorted(
                    ((sum(ch in terms for ch in s) / max(len(s), 1), s) for s in sentences),
                    key=lambda x: x[0],
                    reverse=True,
                )
                if scored and scored[0][0] > 0:
                    body_parts.append(scored[0][1])
                    citations.append(self._hit_label(h, 2))
            break  # 只补充一节

        if not body_parts:
            # 字面完全无重叠的极端情况：返回 top-1 前缀，保证有内容、可溯源
            body_parts = [_content(top)[:160] or "（该段落无正文）"]
            if not citations:
                citations = [self._hit_label(top, 1)]

        body = "\n".join(body_parts)
        return f"根据课程资料：\n{body}\n来源：{' '.join(citations)}"

    # ------------------------------------------------------------------
    # 主入口
    # ------------------------------------------------------------------
    def generate(self, question: str,
                 hits: Sequence[Dict[str, Any]],
                 triples: Optional[Sequence[Any]] = None) -> Dict[str, Any]:
        question = (question or "").strip()
        citations = [
            self._hit_label(h, i)
            for i, h in enumerate(hits[: self._max_evidence], 1)
        ]
        if not question or not hits:
            return {
                "answer": "课程资料中未找到相关内容。",
                "mode": "extractive",
                "citations": [],
            }

        use_llm = self._mode in ("auto", "llm") and self._llm_available()
        if use_llm:
            context = self._build_context(hits, triples or [])
            text = self._llm_generate(question, context)
            if text or self._mode == "llm":
                if text:
                    return {"answer": text, "mode": "llm", "citations": citations}
                return {
                    "answer": "生成服务暂不可用，请稍后重试或改用 extractive 模式。",
                    "mode": "llm",
                    "citations": [],
                }
        answer = self._extractive_generate(question, hits)
        return {"answer": answer, "mode": "extractive", "citations": citations}
