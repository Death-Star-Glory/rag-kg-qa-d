"""doc-parser：定长 + 分隔符分块，输出 chunk 列表。"""

from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from base.sdk.base import Plugin, PluginMeta

DEFAULT_SEPS = ["\n\n", "\n", "。", "！", "？", "；", "!", "?", ";", ".", " "]


class DocParserPlugin(Plugin):
    meta = PluginMeta(
        name="doc_parser",
        version="0.1.0",
        description="文档拆解：分块与元数据",
        group="file_mgmt",
    )

    def __init__(self) -> None:
        self._chunk_size = 160
        self._chunk_overlap = 20
        self._seps = list(DEFAULT_SEPS)
        self._chunk_dir = Path("data/chunks")
        self._file_store = None

    def initialize(self, config: Dict[str, Any] | None = None) -> None:
        cfg = config or {}
        self._chunk_size = int(cfg.get("chunk_size", 160))
        self._chunk_overlap = int(cfg.get("chunk_overlap", 20))
        if cfg.get("separators"):
            self._seps = list(cfg["separators"])
        self._chunk_dir = Path(cfg.get("chunk_dir", "data/chunks"))
        self._chunk_dir.mkdir(parents=True, exist_ok=True)
        self._file_store = cfg.get("file_store")

    def methods(self) -> List[str]:
        return ["split", "split_file", "info"]

    def call(self, method: str, params: Dict[str, Any]) -> Any:
        if method == "info":
            return self.info()
        if method == "split":
            return self.split(
                params["text"], params.get("filename"), bool(params.get("persist", False))
            )
        if method == "split_file":
            return self.split_file(params["filename"], bool(params.get("persist", True)))
        raise KeyError(method)

    def _segment(self, text: str) -> List[str]:
        parts = [text]
        for sep in self._seps:
            nxt: List[str] = []
            for part in parts:
                if not part:
                    continue
                subs = part.split(sep)
                for i, sub in enumerate(subs):
                    if i < len(subs) - 1:
                        nxt.append(sub + sep)
                    elif sub:
                        nxt.append(sub)
            parts = nxt
        out: List[str] = []
        for part in parts:
            if len(part) <= self._chunk_size:
                out.append(part)
            else:
                for i in range(0, len(part), self._chunk_size):
                    out.append(part[i : i + self._chunk_size])
        return out

    def _split_raw(self, text: str) -> List[str]:
        size = max(1, self._chunk_size)
        overlap = max(0, min(self._chunk_overlap, size - 1))
        chunks: List[str] = []
        buf = ""

        def flush() -> None:
            nonlocal buf
            if buf.strip():
                chunks.append(buf.strip())
            buf = ""

        for piece in self._segment(text):
            if not buf:
                buf = piece
            elif len(buf) + 1 + len(piece) <= size:
                buf = f"{buf}{piece}" if buf.endswith(("\n", "。")) else f"{buf} {piece}"
            else:
                flush()
                buf = (chunks[-1][-overlap:] if overlap and chunks else "") + piece
            while len(buf) > size:
                head = buf[:size]
                rest = buf[size - overlap :] if overlap else buf[size:]
                chunks.append(head.strip())
                buf = rest
        flush()
        return chunks or ([text.strip()] if text.strip() else [])

    def split(self, text: str, filename: Optional[str] = None, persist: bool = False) -> Dict[str, Any]:
        doc_id = uuid.uuid4().hex[:12]
        source = filename or "inline"
        chunks = [
            {
                "chunk_id": f"{doc_id}_{i:04d}",
                "doc_id": doc_id,
                "source": source,
                "chunk_index": i,
                "content": c,
                "char_count": len(c),
            }
            for i, c in enumerate(self._split_raw(text))
        ]
        result = {
            "doc_id": doc_id,
            "source": source,
            "chunk_count": len(chunks),
            "chunk_size": self._chunk_size,
            "chunk_overlap": self._chunk_overlap,
            "chunks": chunks,
        }
        if persist:
            path = self._chunk_dir / f"{doc_id}.json"
            path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
            result["chunk_path"] = str(path)
        return result

    def split_file(self, filename: str, persist: bool = True) -> Dict[str, Any]:
        if self._file_store is None:
            raise RuntimeError("doc_parser 未注入 file_store")
        doc = self._file_store.load(filename)
        result = self.split(doc["content"], filename, persist)
        result["source_meta"] = {
            "filename": doc.get("filename"),
            "source": doc.get("source"),
            "size": doc.get("size"),
        }
        return result
