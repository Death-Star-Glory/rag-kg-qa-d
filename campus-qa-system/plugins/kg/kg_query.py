"""kg_query：实体检索与子图查询，供 API/融合使用。"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from base.sdk.base import Plugin, PluginMeta
from .kg_store import KnowledgeGraphStore


class KgQueryPlugin(Plugin):
    meta = PluginMeta(
        name="kg_query",
        version="0.1.0",
        description="知识图谱查询：entity/subgraph/triples",
        group="kg",
    )

    def __init__(self) -> None:
        self._store = KnowledgeGraphStore()
        self._max_nodes = 50

    def initialize(self, config: Dict[str, Any] | None = None) -> None:
        cfg = config or {}
        self._store = cfg.get("store") or KnowledgeGraphStore()
        persist = cfg.get("persist_path")
        if persist:
            self._store.load(persist)
        self._max_nodes = int(cfg.get("max_nodes", 50))

    @property
    def store(self) -> KnowledgeGraphStore:
        return self._store

    def methods(self) -> List[str]:
        return ["search", "subgraph", "triples", "stats", "info"]

    def call(self, method: str, params: Dict[str, Any]) -> Any:
        if method == "info":
            return self.info()
        if method == "search":
            return self.search(params.get("entity", ""), int(params.get("limit", 10)))
        if method == "subgraph":
            return self.subgraph(
                params.get("entity", ""),
                int(params.get("depth", 1)),
                params.get("type"),
            )
        if method == "triples":
            labels = params.get("labels") or ([params["entity"]] if params.get("entity") else [])
            return self.triples(labels, int(params.get("limit", 30)))
        if method == "stats":
            return self._store.stats()
        raise KeyError(method)

    def search(self, entity: str, limit: int = 10) -> Dict[str, Any]:
        nodes = self._store.find_nodes(entity)[: max(1, limit)]
        return {"entity": entity, "nodes": nodes, "count": len(nodes)}

    def subgraph(
        self, entity: str, depth: int = 1, ntype: Optional[str] = None
    ) -> Dict[str, Any]:
        node = self._store.get_node(entity, ntype)
        if node is None:
            hits = self._store.find_nodes(entity)
            node = hits[0] if hits else None
        if node is None:
            return {
                "center": None,
                "nodes": [],
                "links": [],
                "entity": entity,
                "found": False,
            }
        graph = self._store.neighbors(node["id"], depth=depth)
        if len(graph["nodes"]) > self._max_nodes:
            graph["nodes"] = graph["nodes"][: self._max_nodes]
            keep_ids = {n["id"] for n in graph["nodes"]}
            graph["links"] = [e for e in graph["links"] if e["source"] in keep_ids and e["target"] in keep_ids]
        graph["entity"] = entity
        graph["found"] = True
        graph["center_label"] = node["label"]
        graph["center_type"] = node["type"]
        return graph

    def triples(self, labels: List[str], limit: int = 30) -> Dict[str, Any]:
        rows = self._store.triples_for_labels(labels, limit=limit)
        return {"labels": labels, "triples": rows, "count": len(rows)}

    def stats(self) -> Dict[str, Any]:
        return self._store.stats()
