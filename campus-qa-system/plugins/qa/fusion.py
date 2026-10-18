"""RAG + KG 融合问答：三态编排（作答 / 澄清 / 拒答）+ 溯源回答。

v0.2.0（2026-10-12，qa 组交付）
--------------------------------
在 v0.1.0 模板拼接版基础上补齐课设过关硬条件：

1. **拒答判定**（分数闸门）：top-1 分数 < τ=0.185 → 拒答，话术清晰。
   库外问题拒答率是任务书硬性过关条件（≥70%，Rubric 优档 100%）。
2. **澄清判定**（三态口径）：移植 vector_db 组的参考实现
   （evals/vector_db/check_clarify_rule.py），判据见 decision.py。
3. **诱导题兜底**（可支撑性检查）：问句问具体数值/实现/步骤而证据里没有
   → 拒答。vector_db 实测诱导题分数拦不住（§3.3），此为编排层补的闸。
4. **generator 生成**：LLM（OpenAI 兼容接口，可配置）优先、抽取式兜底，
   所有作答带来源段落标注（Rubric「回答准确性与溯源」）。

判定顺序（全部在生成之前，硬约束）：

    拒答   top1 < τ
    澄清   过闸 + 无先行词指示代词 + 锚不到章节标题
    兜底   过闸 + 问细节而证据无细节
    作答   其余 → RAG chunk + KG 三元组 → generator 生成 + 溯源
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Set

from base.sdk.base import Plugin, PluginMeta

from . import decision
from .decision import TAU, clarify_message, refuse_message

_ENTITY_HINT = re.compile(
    r"([一-鿿A-Za-z0-9]{2,12})(?:专业|课程|老师|教师|教授|学分|先修|毕业)"
)


class FusionQaPlugin(Plugin):
    meta = PluginMeta(
        name="fusion_qa",
        version="0.2.0",
        description="RAG+KG融合问答：三态判定（作答/澄清/拒答）+ 生成 + 溯源",
        group="qa",
    )

    def __init__(self) -> None:
        self._retriever = None
        self._kg_query = None
        self._generator = None
        self._top_k = 3
        self._max_triples = 8
        self._tau = TAU

    # ------------------------------------------------------------------
    # 装配
    # ------------------------------------------------------------------
    def initialize(self, config: Dict[str, Any] | None = None) -> None:
        cfg = config or {}
        self._retriever = cfg.get("retriever")
        self._kg_query = cfg.get("kg_query")
        self._generator = cfg.get("generator")
        self._top_k = int(cfg.get("top_k", 3))
        self._max_triples = int(cfg.get("max_triples", 8))
        self._tau = float(cfg.get("tau", TAU))

        # 多进程部署：跨节点依赖一律写地址，不写死在代码里（交接文档 §6.6）
        if self._retriever is None and cfg.get("retriever_url"):
            from base.rpc.client import RemotePluginProxy

            proxy = RemotePluginProxy("rag_retriever", str(cfg["retriever_url"]))
            proxy.initialize()
            self._retriever = proxy
        if self._kg_query is None and cfg.get("kg_query_url"):
            from base.rpc.client import RemotePluginProxy

            proxy = RemotePluginProxy("kg_query", str(cfg["kg_query_url"]))
            proxy.initialize()
            self._kg_query = proxy

        if self._generator is None:
            from .generator import GeneratorPlugin

            gen = GeneratorPlugin()
            gen.initialize(cfg.get("generator_config") or {})
            self._generator = gen

    def methods(self) -> List[str]:
        return ["ask", "info"]

    def call(self, method: str, params: Dict[str, Any]) -> Any:
        if method == "info":
            return self.info()
        if method == "ask":
            return self.ask(params.get("question", ""), int(params.get("top_k", self._top_k)))
        raise KeyError(method)

    # ------------------------------------------------------------------
    # 辅助
    # ------------------------------------------------------------------
    def _titles(self) -> List[str]:
        """全部章节标题（澄清锚定用）。检索缺失时返回空表（澄清降级为不触发）。"""
        if self._retriever is None:
            return []
        try:
            return list(self._retriever.call("stats", {}).get("titles") or [])
        except Exception:
            return []

    def _guess_entities(self, question: str) -> List[str]:
        found: List[str] = []

        def _add(token: str) -> None:
            token = re.sub(r"(的|之)?(先修|课程|专业|老师|教师|教授)$", "", token).strip("的 ")
            if token and len(token) >= 2 and token not in found:
                found.append(token)

        if self._kg_query is not None:
            for m in re.finditer(r"[一-鿿A-Za-z0-9]{2,12}", question):
                token = m.group(0)
                hits = self._kg_query.call("search", {"entity": token, "limit": 1})
                nodes = hits.get("nodes") or []
                if nodes:
                    _add(nodes[0].get("label") or token)
                else:
                    _add(token)
        for m in _ENTITY_HINT.finditer(question):
            _add(m.group(1))
        stop = {"什么", "如何", "怎么", "要求", "专业", "哪个", "哪些", "属于", "属于哪个", "是什么"}
        return [e for e in found if e not in stop][:5]

    # ------------------------------------------------------------------
    # 主流程：三态判定 → （作答时）RAG+KG 证据 → 生成
    # ------------------------------------------------------------------
    def ask(self, question: str, top_k: Optional[int] = None) -> Dict[str, Any]:
        k = top_k or self._top_k
        question = (question or "").strip()
        base: Dict[str, Any] = {
            "question": question,
            "state": decision.STATE_ANSWER,
            "mode": "rag+kg",
            "answer": "",
            "reason": "",
            "entities": [],
            "rag_hits": [],
            "kg": {"entities": [], "subgraphs": [], "triples": []},
            "sources": [],
            "confidence": 0.0,
            "decision": {},
        }
        if not question:
            base.update({
                "state": decision.STATE_REFUSE,
                "answer": "请输入问题。",
                "reason": "空问题",
            })
            return base

        rag = (
            self._retriever.call("retrieve", {"query": question, "top_k": k})
            if self._retriever is not None
            else {"hits": []}
        )
        hits = rag.get("hits") or []
        titles = self._titles()
        base["rag_hits"] = hits

        # ---- 三态判定（生成之前，硬约束）----
        verdict = decision.decide(question, hits, titles, tau=self._tau)
        base["decision"] = {
            **verdict,
            "tau": self._tau,
            "titles_available": bool(titles),
        }

        if verdict["state"] == decision.STATE_REFUSE:
            base.update({
                "state": decision.STATE_REFUSE,
                "answer": refuse_message(verdict["reason"]),
                "reason": verdict["reason"],
            })
            return base

        if verdict["state"] == decision.STATE_CLARIFY:
            base.update({
                "state": decision.STATE_CLARIFY,
                "answer": clarify_message(question, titles),
                "reason": verdict["reason"],
            })
            return base

        # ---- 作答路径：RAG 证据 + KG 佐证 ----
        entities = self._guess_entities(question)
        base["entities"] = entities
        kg_block: Dict[str, Any] = {"entities": entities, "subgraphs": [], "triples": []}
        if self._kg_query is not None and entities:
            for ent in entities:
                sg = self._kg_query.call("subgraph", {"entity": ent, "depth": 1})
                if sg.get("found"):
                    kg_block["subgraphs"].append(
                        {
                            "entity": ent,
                            "center_label": sg.get("center_label"),
                            "center_type": sg.get("center_type"),
                            "nodes": sg.get("nodes", []),
                            "links": sg.get("links", []),
                        }
                    )
            tr = self._kg_query.call(
                "triples", {"labels": entities, "limit": self._max_triples}
            )
            kg_block["triples"] = tr.get("triples") or []
        base["kg"] = kg_block

        generated = self._generator.call(
            "generate",
            {"question": question, "hits": hits, "triples": kg_block["triples"]},
        )
        base["answer"] = generated.get("answer") or ""
        base["generator_mode"] = generated.get("mode")

        # ---- 溯源 ----
        sources: List[Dict[str, Any]] = []
        seen: Set[str] = set()
        for h in hits:
            # 按（文件, 章节标题, chunk_id）去重：同一文件的不同章节是不同来源，
            # 按文件名去重会把多章节来源压成一条，溯源就失真了。
            src = h.get("source") or h.get("filename")
            key = f"{src}|{h.get('title') or ''}|{h.get('chunk_id')}"
            if key in seen:
                continue
            seen.add(key)
            sources.append(
                {
                    "type": "chunk",
                    "source": src,
                    "title": h.get("title"),
                    "chunk_id": h.get("chunk_id"),
                    "score": h.get("score"),
                    "preview": (h.get("content") or "")[:80],
                }
            )
        for t in kg_block.get("triples") or []:
            sources.append(
                {"type": "kg", "source": "knowledge_graph", "triple": t}
            )
        base["sources"] = sources
        base["confidence"] = self._confidence(hits, kg_block)
        return base

    @staticmethod
    def _confidence(hits: List[Dict[str, Any]], kg_block: Dict[str, Any]) -> float:
        score = 0.0
        if hits:
            score += min(0.45, 0.2 + 0.1 * len(hits))
        if kg_block.get("triples"):
            score += min(0.4, 0.15 + 0.05 * len(kg_block["triples"]))
        if kg_block.get("subgraphs"):
            score += 0.1
        return round(min(score, 0.95), 3)
