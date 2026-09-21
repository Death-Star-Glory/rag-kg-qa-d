"""RAG 检索插件：关键词/字符 n-gram 打分，chunk 池可来自 file_mgmt。"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from base.sdk.base import Plugin, PluginMeta

_TOKEN = re.compile(r"[一-鿿A-Za-z0-9]+")


def _tokens(text: str) -> List[str]:
    toks = _TOKEN.findall(text.lower())
    # 中文按字 + 英文/数字词
    out: List[str] = []
    for t in toks:
        if re.match(r"^[a-z0-9]+$", t):
            out.append(t)
        else:
            out.extend(list(t))
            if len(t) >= 2:
                out.append(t)
    return out


class RagRetrieverPlugin(Plugin):
    meta = PluginMeta(
        name="rag_retriever",
        version="0.1.0",
        description="RAG检索：chunk 关键词打分召回",
        group="qa",
    )

    def __init__(self) -> None:
        self._chunks: List[Dict[str, Any]] = []
        self._doc_parser = None
        self._file_store = None

    def initialize(self, config: Dict[str, Any] | None = None) -> None:
        cfg = config or {}
        self._doc_parser = cfg.get("doc_parser")
        self._file_store = cfg.get("file_store")
        self._chunks = list(cfg.get("chunks") or [])

    def methods(self) -> List[str]:
        return ["index_files", "index_chunks", "retrieve", "stats", "info"]

    def call(self, method: str, params: Dict[str, Any]) -> Any:
        if method == "info":
            return self.info()
        if method == "index_files":
            return self.index_files(params.get("filenames"))
        if method == "index_chunks":
            return self.index_chunks(params.get("chunks") or [])
        if method == "retrieve":
            return self.retrieve(
                params.get("query", ""),
                int(params.get("top_k", 3)),
            )
        if method == "stats":
            return {"chunk_count": len(self._chunks)}
        raise KeyError(method)

    def index_chunks(self, chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        self._chunks = list(chunks)
        return {"chunk_count": len(self._chunks)}

    def index_files(self, filenames: Optional[List[str]] = None) -> Dict[str, Any]:
        if self._file_store is None or self._doc_parser is None:
            raise RuntimeError("rag_retriever 未注入 file_store/doc_parser")
        names = filenames or [x["filename"] for x in self._file_store.list_files()]
        all_chunks: List[Dict[str, Any]] = []
        for name in names:
            split = self._doc_parser.call(
                "split_file", {"filename": name, "persist": False}
            )
            all_chunks.extend(split["chunks"])
        self._chunks = all_chunks
        return {"chunk_count": len(self._chunks), "files": names}

    def retrieve(self, query: str, top_k: int = 3) -> Dict[str, Any]:
        q_tokens = set(_tokens(query))
        if not q_tokens:
            return {"query": query, "hits": [], "top_k": top_k}
        scored: List[Dict[str, Any]] = []
        for ch in self._chunks:
            c_tokens = _tokens(ch.get("content", ""))
            if not c_tokens:
                continue
            c_set = set(c_tokens)
            inter = q_tokens & c_set
            if not inter:
                continue
            # 简单重叠得分 + 完整词命中加权
            score = len(inter) / (len(q_tokens) + 1)
            for tok in q_tokens:
                if len(tok) >= 2 and tok in ch.get("content", ""):
                    score += 0.35
            scored.append({**ch, "score": round(score, 4)})
        scored.sort(key=lambda x: x["score"], reverse=True)
        hits = scored[: max(1, top_k)]
        return {"query": query, "hits": hits, "top_k": top_k, "indexed": len(self._chunks)}
