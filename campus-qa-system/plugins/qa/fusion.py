"""RAG + KG 融合问答：检索 chunk + 图谱三元组 → 可溯源回答。"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Set

from base.sdk.base import Plugin, PluginMeta

_ENTITY_HINT = re.compile(
    r"([一-鿿A-Za-z0-9]{2,12})(?:专业|课程|老师|教师|教授|学分|先修|毕业)"
)


class FusionQaPlugin(Plugin):
    meta = PluginMeta(
        name="fusion_qa",
        version="0.1.0",
        description="RAG+KG融合问答：chunk 证据 + 图谱三元组 + 溯源",
        group="qa",
    )

    def __init__(self) -> None:
        self._retriever = None
        self._kg_query = None
        self._top_k = 3
        self._max_triples = 8

    def initialize(self, config: Dict[str, Any] | None = None) -> None:
        cfg = config or {}
        self._retriever = cfg.get("retriever")
        self._kg_query = cfg.get("kg_query")
        self._top_k = int(cfg.get("top_k", 3))
        self._max_triples = int(cfg.get("max_triples", 8))

    def methods(self) -> List[str]:
        return ["ask", "info"]

    def call(self, method: str, params: Dict[str, Any]) -> Any:
        if method == "info":
            return self.info()
        if method == "ask":
            return self.ask(params.get("question", ""), int(params.get("top_k", self._top_k)))
        raise KeyError(method)

    def _guess_entities(self, question: str) -> List[str]:
        found: List[str] = []

        def _add(token: str) -> None:
            token = re.sub(r"(的|之)?(先修|课程|专业|老师|教师|教授)$", "", token).strip("的 ")
            if token and len(token) >= 2 and token not in found:
                found.append(token)

        # 先在图谱词表里找问句子串命中
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
        # 去掉过泛的词
        stop = {"什么", "如何", "怎么", "要求", "专业", "哪个", "哪些", "属于", "属于哪个", "是什么"}
        return [e for e in found if e not in stop][:5]

    def ask(self, question: str, top_k: Optional[int] = None) -> Dict[str, Any]:
        k = top_k or self._top_k
        question = (question or "").strip()
        if not question:
            return {"answer": "请输入问题。", "sources": [], "rag_hits": [], "kg": {}}

        rag = (
            self._retriever.call("retrieve", {"query": question, "top_k": k})
            if self._retriever is not None
            else {"hits": []}
        )
        hits = rag.get("hits") or []

        entities = self._guess_entities(question)
        kg_block: Dict[str, Any] = {"entities": entities, "subgraphs": [], "triples": []}
        if self._kg_query is not None and entities:
            triple_rows: List[str] = []
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
            triple_rows = tr.get("triples") or []
            kg_block["triples"] = triple_rows
        elif self._kg_query is not None:
            # 兜底：用 chunk 文本里的课程名试查
            labels = []
            for h in hits[:2]:
                labels.extend(re.findall(r"[一-鿿A-Za-z0-9]{2,12}课程", h.get("content", "")))
            if labels:
                tr = self._kg_query.call("triples", {"labels": labels, "limit": self._max_triples})
                kg_block["triples"] = tr.get("triples") or []

        answer = self._compose_answer(question, hits, kg_block)
        sources = []
        seen: Set[str] = set()
        for h in hits:
            src = h.get("source") or h.get("filename")
            if src and src not in seen:
                seen.add(src)
                sources.append(
                    {
                        "type": "chunk",
                        "source": src,
                        "chunk_id": h.get("chunk_id"),
                        "score": h.get("score"),
                        "preview": (h.get("content") or "")[:80],
                    }
                )
        for t in kg_block.get("triples") or []:
            sources.append(
                {
                    "type": "kg",
                    "source": "knowledge_graph",
                    "triple": t,
                }
            )

        return {
            "question": question,
            "answer": answer,
            "mode": "rag+kg",
            "entities": entities,
            "rag_hits": hits,
            "kg": kg_block,
            "sources": sources,
            "confidence": self._confidence(hits, kg_block),
        }

    def _compose_answer(
        self, question: str, hits: List[Dict[str, Any]], kg_block: Dict[str, Any]
    ) -> str:
        parts: List[str] = []
        triples = kg_block.get("triples") or []
        if triples:
            parts.append("【知识图谱】")
            for t in triples[:5]:
                parts.append(
                    f"- {t['source']}（{t['source_type']}） --{t['rel']}--> {t['target']}（{t['target_type']}）"
                )
        if hits:
            parts.append("【文档证据】")
            for i, h in enumerate(hits, 1):
                content = re.sub(r"\s+", " ", h.get("content") or "").strip()
                parts.append(f"[{i}] ({h.get('source')}) {content[:160]}")
        if not parts:
            parts.append("根据现有资料无法回答：未检索到相关文档，也未命中知识图谱实体。")
        else:
            parts.append("【综合】")
            if triples and hits:
                parts.append(
                    f"结合图谱关系与文档片段，针对「{question}」的要点如上；完整表述以文档原文为准。"
                )
            elif triples:
                parts.append(f"图谱中与「{question}」相关的关系如上，可结合文档进一步核对。")
            else:
                parts.append(f"以下文档片段与「{question}」相关：")
        return "\n".join(parts)

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
