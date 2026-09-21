"""kg_builder：从文档/chunk 规则抽取实体关系，写入图存储。"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from base.sdk.base import Plugin, PluginMeta
from .kg_store import KnowledgeGraphStore


COURSE_LINE = re.compile(
    r"(?P<course>[一-鿿A-Za-z0-9]{2,20})\s*[（(]?(?P<credits>\d+(?:\.\d+)?)\s*学分[)）]?"
)
PREREQ = re.compile(
    r"(?P<course>[一-鿿A-Za-z0-9]{2,20})\s*的?先修(?:课|课程)?(?:为|是|:|：)\s*(?P<pre>[^。；;\n]+)"
)
MAJOR_REQ = re.compile(
    r"(?P<major>[一-鿿A-Za-z0-9]{2,16}?)专业\s*(?:要求|需)\s*完成\s*(?P<credits>\d+)\s*学分"
)
BELONGS = re.compile(
    r"(?P<course>[一-鿿A-Za-z0-9]{2,20})\s*(?:属于|隶属|开设于)\s*(?P<major>[一-鿿A-Za-z0-9]{2,16}?)专业"
)
TEACHES = re.compile(
    r"(?P<teacher>[一-鿿]{2,4})(?:老师|教师|教授)\s*(?:讲授|主讲|任教)\s*(?P<course>[一-鿿A-Za-z0-9]{2,20})"
)


def _clean_label(raw: str) -> str:
    s = re.sub(r"[（(].*?[)）]", "", raw or "").strip()
    s = re.sub(r"(的|之)?(先修|课程|专业|老师|教师|教授)$", "", s).strip()
    s = s.strip("的、，, ")
    return s


class KgBuilderPlugin(Plugin):
    meta = PluginMeta(
        name="kg_builder",
        version="0.1.0",
        description="知识图谱构建：规则抽取实体关系",
        group="kg",
    )

    def __init__(self) -> None:
        self._store = KnowledgeGraphStore()
        self._persist_path: Optional[Path] = None
        self._file_store = None
        self._doc_parser = None
        self._status: Dict[str, Any] = {
            "state": "idle",
            "built_docs": 0,
            "chunks": 0,
            "message": "",
        }

    def initialize(self, config: Dict[str, Any] | None = None) -> None:
        cfg = config or {}
        persist = cfg.get("persist_path")
        self._persist_path = Path(persist) if persist else None
        self._store = KnowledgeGraphStore()
        if self._persist_path and self._persist_path.exists():
            self._store.load(self._persist_path)
        self._file_store = cfg.get("file_store")
        self._doc_parser = cfg.get("doc_parser")
        self._status["state"] = "ready"

    @property
    def store(self) -> KnowledgeGraphStore:
        return self._store

    def methods(self) -> List[str]:
        return ["build", "build_text", "status", "stats", "info"]

    def call(self, method: str, params: Dict[str, Any]) -> Any:
        if method == "info":
            return self.info()
        if method == "build":
            return self.build(params.get("filenames") or params.get("doc_ids") or [])
        if method == "build_text":
            return self.build_text(params.get("text", ""), params.get("source", "inline"))
        if method == "status":
            return dict(self._status)
        if method == "stats":
            return self._store.stats()
        raise KeyError(method)

    def _extract_from_text(self, text: str, source: str) -> Dict[str, int]:
        nodes = edges = 0
        for m in MAJOR_REQ.finditer(text):
            major = _clean_label(m.group("major"))
            if len(major) < 2:
                continue
            credits = int(m.group("credits"))
            req_label = f"{major}毕业要求"
            self._store.upsert_node(major, "专业", source=source)
            self._store.upsert_node(req_label, "要求", credits=credits, source=source)
            self._store.upsert_edge(
                major, "专业", req_label, "要求", "要求学分", credits=credits, source=source
            )
            nodes += 2
            edges += 1
        for m in COURSE_LINE.finditer(text):
            course = _clean_label(m.group("course"))
            if len(course) < 2 or course in {"毕业要求", "转专业申请", "专业要求", "课程与先修"}:
                continue
            credits = float(m.group("credits"))
            self._store.upsert_node(course, "课程", credits=credits, source=source)
            nodes += 1
        for m in PREREQ.finditer(text):
            course = _clean_label(m.group("course"))
            if len(course) < 2:
                continue
            pre_raw = m.group("pre")
            self._store.upsert_node(course, "课程", source=source)
            nodes += 1
            for pre in re.split(r"[、,，和及]", pre_raw):
                pre = _clean_label(pre)
                if not pre or len(pre) < 2:
                    continue
                self._store.upsert_node(pre, "课程", source=source)
                self._store.upsert_edge(pre, "课程", course, "课程", "先修", source=source)
                nodes += 1
                edges += 1
        for m in BELONGS.finditer(text):
            course = _clean_label(m.group("course"))
            major = _clean_label(m.group("major"))
            if len(course) < 2 or len(major) < 2:
                continue
            self._store.upsert_node(course, "课程", source=source)
            self._store.upsert_node(major, "专业", source=source)
            self._store.upsert_edge(course, "课程", major, "专业", "属于", source=source)
            nodes += 2
            edges += 1
        for m in TEACHES.finditer(text):
            teacher = _clean_label(m.group("teacher"))
            course = _clean_label(m.group("course"))
            if len(teacher) < 2 or len(course) < 2:
                continue
            self._store.upsert_node(teacher, "教师", source=source)
            self._store.upsert_node(course, "课程", source=source)
            self._store.upsert_edge(teacher, "教师", course, "课程", "讲授", source=source)
            nodes += 2
            edges += 1
        return {"nodes_touched": nodes, "edges_touched": edges}

    def build_text(self, text: str, source: str = "inline") -> Dict[str, Any]:
        stats = self._extract_from_text(text, source)
        if self._persist_path:
            self._store.save(self._persist_path)
        self._status.update(
            {
                "state": "ready",
                "built_docs": int(self._status.get("built_docs", 0)) + 1,
                "message": f"built from {source}",
            }
        )
        return {**stats, "source": source, "graph": self._store.stats()}

    def build(self, filenames: Optional[List[str]] = None) -> Dict[str, Any]:
        self._status.update({"state": "building", "message": "start"})
        try:
            sources: List[Dict[str, str]] = []
            if self._file_store is not None:
                names = filenames or [x["filename"] for x in self._file_store.list_files()]
                for name in names:
                    doc = self._file_store.load(name)
                    if self._doc_parser is not None:
                        chunks = self._doc_parser.call(
                            "split_file", {"filename": name, "persist": False}
                        )["chunks"]
                        text = "\n".join(c["content"] for c in chunks)
                    else:
                        text = doc["content"]
                    sources.append({"source": name, "text": text})
            else:
                raise RuntimeError("kg_builder 未注入 file_store，无法 build 文件")

            total = {"nodes_touched": 0, "edges_touched": 0}
            for item in sources:
                s = self._extract_from_text(item["text"], item["source"])
                total["nodes_touched"] += s["nodes_touched"]
                total["edges_touched"] += s["edges_touched"]
            if self._persist_path:
                self._store.save(self._persist_path)
            self._status.update(
                {
                    "state": "ready",
                    "built_docs": len(sources),
                    "chunks": len(sources),
                    "message": "ok",
                }
            )
            return {
                **total,
                "docs": [s["source"] for s in sources],
                "graph": self._store.stats(),
            }
        except Exception as e:  # noqa: BLE001
            self._status.update({"state": "error", "message": str(e)})
            raise
