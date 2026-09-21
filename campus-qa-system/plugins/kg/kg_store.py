"""内存知识图谱：节点 + 有向边，可序列化。"""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple


class KnowledgeGraphStore:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        # id -> {id, type, label, props}
        self._nodes: Dict[str, Dict[str, Any]] = {}
        # list of {source, target, type, props}
        self._edges: List[Dict[str, Any]] = []
        self._edge_keys: Set[Tuple[str, str, str]] = set()

    @staticmethod
    def node_id(label: str, ntype: str) -> str:
        return f"{ntype}::{label}"

    def upsert_node(self, label: str, ntype: str, **props: Any) -> str:
        nid = self.node_id(label, ntype)
        with self._lock:
            node = self._nodes.get(nid)
            if node is None:
                node = {"id": nid, "type": ntype, "label": label, "props": {}}
                self._nodes[nid] = node
            node["props"].update(props)
            return nid

    def upsert_edge(
        self,
        source_label: str,
        source_type: str,
        target_label: str,
        target_type: str,
        rel_type: str,
        **props: Any,
    ) -> None:
        sid = self.upsert_node(source_label, source_type)
        tid = self.upsert_node(target_label, target_type)
        key = (sid, tid, rel_type)
        with self._lock:
            if key in self._edge_keys:
                for e in self._edges:
                    if (e["source"], e["target"], e["type"]) == key:
                        e["props"].update(props)
                        return
            self._edge_keys.add(key)
            self._edges.append(
                {"source": sid, "target": tid, "type": rel_type, "props": props}
            )

    def find_nodes(self, keyword: str) -> List[Dict[str, Any]]:
        kw = keyword.strip()
        with self._lock:
            hits = []
            for node in self._nodes.values():
                if kw and (kw in node["label"] or node["label"] in kw):
                    hits.append(dict(node))
            return hits

    def get_node(self, label: str, ntype: Optional[str] = None) -> Optional[Dict[str, Any]]:
        with self._lock:
            if ntype:
                return self._nodes.get(self.node_id(label, ntype))
            for node in self._nodes.values():
                if node["label"] == label:
                    return dict(node)
            return None

    def neighbors(self, node_id: str, depth: int = 1) -> Dict[str, Any]:
        """返回 center + nodes + links（可视化友好，id/label/type）。"""
        depth = max(1, min(int(depth), 3))
        with self._lock:
            if node_id not in self._nodes:
                return {"center": None, "nodes": [], "links": []}
            seen_nodes: Dict[str, Dict[str, Any]] = {node_id: self._nodes[node_id]}
            links: List[Dict[str, Any]] = []
            frontier = {node_id}
            for _ in range(depth):
                nxt: Set[str] = set()
                for e in self._edges:
                    if e["source"] in frontier or e["target"] in frontier:
                        links.append(
                            {
                                "source": e["source"],
                                "target": e["target"],
                                "type": e["type"],
                                "props": e["props"],
                            }
                        )
                        for nid in (e["source"], e["target"]):
                            if nid not in seen_nodes and nid in self._nodes:
                                seen_nodes[nid] = self._nodes[nid]
                                nxt.add(nid)
                frontier = nxt
                if not frontier:
                    break
            return {
                "center": node_id,
                "nodes": [
                    {
                        "id": n["id"],
                        "label": n["label"],
                        "type": n["type"],
                        "props": n["props"],
                    }
                    for n in seen_nodes.values()
                ],
                "links": links,
            }

    def triples_for_labels(self, labels: Iterable[str], limit: int = 30) -> List[Dict[str, Any]]:
        """按标签过滤三元组，返回结构化 dict（供 API/融合层使用）。"""
        label_set = {x.strip() for x in labels if x and x.strip()}
        out: List[Dict[str, Any]] = []
        with self._lock:
            id_to = {n["id"]: n for n in self._nodes.values()}
            for e in self._edges:
                s_n = id_to.get(e["source"], {})
                t_n = id_to.get(e["target"], {})
                s_label = s_n.get("label", e["source"])
                t_label = t_n.get("label", e["target"])
                hit = (
                    s_label in label_set
                    or t_label in label_set
                    or e["source"] in label_set
                    or e["target"] in label_set
                    or any(lab in s_label or lab in t_label for lab in label_set)
                )
                if not hit:
                    continue
                out.append(
                    {
                        "source": s_label,
                        "source_type": s_n.get("type", ""),
                        "target": t_label,
                        "target_type": t_n.get("type", ""),
                        "rel": e["type"],
                        "props": e["props"],
                    }
                )
                if len(out) >= limit:
                    break
        return out

    def stats(self) -> Dict[str, int]:
        with self._lock:
            type_count: Dict[str, int] = {}
            for n in self._nodes.values():
                type_count[n["type"]] = type_count.get(n["type"], 0) + 1
            return {
                "node_count": len(self._nodes),
                "edge_count": len(self._edges),
                "type_counts": type_count,  # type: ignore[dict-item]
            }

    def all_triples(self) -> List[Dict[str, Any]]:
        with self._lock:
            id_to = {n["id"]: n for n in self._nodes.values()}
            rows = []
            for e in self._edges:
                s, t = id_to.get(e["source"], {}), id_to.get(e["target"], {})
                rows.append(
                    {
                        "source": s.get("label", e["source"]),
                        "source_type": s.get("type", ""),
                        "target": t.get("label", e["target"]),
                        "target_type": t.get("type", ""),
                        "rel": e["type"],
                        "props": e["props"],
                    }
                )
            return rows

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with self._lock:
            payload = {"nodes": list(self._nodes.values()), "edges": self._edges}
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def load(self, path: str | Path) -> bool:
        path = Path(path)
        if not path.exists():
            return False
        payload = json.loads(path.read_text(encoding="utf-8"))
        with self._lock:
            self._nodes = {n["id"]: n for n in payload.get("nodes", [])}
            self._edges = list(payload.get("edges", []))
            self._edge_keys = {
                (e["source"], e["target"], e["type"]) for e in self._edges
            }
        return True
