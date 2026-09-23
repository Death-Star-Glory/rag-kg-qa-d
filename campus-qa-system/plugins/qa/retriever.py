"""RAG 检索插件 —— 对外接口不变，内部委托 vector_store。

改造说明（2026-09-22，方案 C 薄封装）
--------------------------------------
本插件原先自实现了一套关键词打分（`交集比 + 0.35 × 完整词命中数`），
与 `plugins/vector_db` 的职责完全重叠。经同条件对比
（见 `evals/vector_db/compare_retriever.py`）：

    | 指标     | 原实现 | vector_store |
    |---------|--------|--------------|
    | Hit@1   | 95.0%  | 95.0%        |
    | Hit@3   | 100.0% | 95.0%        |
    | MRR     | 0.967  | 0.950        |
    | 分数上界  | 无（实测最大 1.0722） | 1.0（归一化）|

**检索质量上原实现并不差**（Hit@3 与 MRR 更高），但它有一个硬伤：
分数无上界，第二项随问句长度线性增长 —— 无法作为拒答阈值，
而「库外拒答率 ≥70%」是项目硬性过关条件。

故改为薄封装：

    - 对外方法签名与返回结构**完全不变** → qa 组（fusion）零改动
    - 内部委托 `vector_store` → 检索逻辑归位到 vector_db 组
    - 分数统一归一化到 [0,1] → 编排层可写统一的拒答判定

后端注入方式（按优先级）：

    1. ``config["vector_store"]``     本地实例（单进程 demo，gateway 用）
    2. ``config["vector_store_url"]`` 远程地址（多进程，走 RemotePluginProxy）
    3. 都没传                          自动创建本地 vector_store（默认，保证不崩）

返回结构保持兼容，并**新增** ``score_scale`` 字段透传分数尺度说明，
供编排层做拒答判定时读取（老调用方忽略该字段即可）。
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from base.sdk.base import Plugin, PluginMeta


class RagRetrieverPlugin(Plugin):
    meta = PluginMeta(
        name="rag_retriever",
        version="0.2.0",
        description="RAG检索：对外接口不变，内部委托 vector_store（薄封装）",
        group="qa",
    )

    def __init__(self) -> None:
        self._store = None
        self._doc_parser = None
        self._file_store = None

    # ------------------------------------------------------------------
    # 装配
    # ------------------------------------------------------------------
    def _make_local_store(self):
        """自动创建本地 vector_store 实例。

        存在的意义：单进程 demo（gateway.build_demo_registry）不会显式注入后端，
        若此处直接报错会连带把现有 demo 打挂。自动兜底让改造对调用方透明。
        """
        from plugins.vector_db import EmbedderPlugin, VectorStorePlugin

        embedder = EmbedderPlugin()
        embedder.initialize({"dim": 4096})
        store = VectorStorePlugin()
        store.initialize({"embedder": embedder, "dim": 4096})
        return store

    def initialize(self, config: Dict[str, Any] | None = None) -> None:
        cfg = config or {}
        self._doc_parser = cfg.get("doc_parser")
        self._file_store = cfg.get("file_store")

        store = cfg.get("vector_store")
        if store is None:
            url = cfg.get("vector_store_url")
            if url:
                from base.rpc.client import RemotePluginProxy

                proxy = RemotePluginProxy("vector_store", str(url))
                proxy.initialize()
                store = proxy
        if store is None:
            store = self._make_local_store()
        self._store = store

        if cfg.get("chunks"):
            self.index_chunks(cfg["chunks"])

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
            return self.stats()
        raise KeyError(method)

    # ------------------------------------------------------------------
    # 索引
    # ------------------------------------------------------------------
    def index_chunks(self, chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        result = self._store.call("index", {"chunks": chunks})
        return {"chunk_count": result.get("indexed", len(chunks))}

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
        result = self._store.call("index", {"chunks": all_chunks})
        return {
            "chunk_count": result.get("indexed", len(all_chunks)),
            "files": names,
        }

    # ------------------------------------------------------------------
    # 检索
    # ------------------------------------------------------------------
    def retrieve(self, query: str, top_k: int = 3) -> Dict[str, Any]:
        """检索。返回结构与改造前一致，额外透传 score_scale。"""
        r = self._store.call("search", {"query": query, "top_k": top_k})
        return {
            "query": r.get("query", query),
            "hits": r.get("hits") or [],
            "top_k": top_k,
            "indexed": r.get("indexed", 0),
            # 新增字段：编排层做拒答判定前应读它。
            # 分数尺度由底层 embedding 决定，换实现后阈值必须重新标定。
            "score_scale": r.get("score_scale"),
        }

    def stats(self) -> Dict[str, Any]:
        st = self._store.call("stats", {})
        # chunk_count 是改造前的字段名，保留以兼容既有调用方
        return {**st, "chunk_count": st.get("indexed", 0)}
