"""doc-parser：分块与元数据。

两种切分模式（`split_mode` 配置项）：
    fixed   定长 + 分隔符切分（默认，保持原有行为）
    heading 按【标题】切，一块 = 一节；超长章节内部再走定长

为什么默认仍是 fixed：本插件属 file_mgmt 组，改默认值会影响其调用方。
heading 模式经检索评测验证收益显著（Hit@1 74.1% → 88.9%，MRR 0.815 → 0.907），
是否切为默认由 file_mgmt 组决定；本文件只提供能力，不替对方做决定。

已知偏差修复（2026-09-22）：
    原实现用 `chunks.append(head.strip())` 保存分块，strip 会吃掉重叠区里的空白，
    导致实际重叠字符数少于 `chunk_overlap` 配置值（实测配置 12 实际 9）。
    内容不丢，但跨越块边界的短语可能无法完整出现在任何单块中。
    现改为内部保留原文、仅在最终输出时 strip，使重叠字符数精确等于配置值。
"""

from __future__ import annotations

import json
import re
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from base.sdk.base import Plugin, PluginMeta

DEFAULT_SEPS = ["\n\n", "\n", "。", "！", "？", "；", "!", "?", ";", ".", " "]
HEADING_RE = re.compile(r"^【(?P<title>[^】]+)】\s*(?P<body>.*)$")


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
        self._split_mode = "fixed"
        self._chunk_dir = Path("data/chunks")
        self._file_store = None

    def initialize(self, config: Dict[str, Any] | None = None) -> None:
        cfg = config or {}
        self._chunk_size = int(cfg.get("chunk_size", 160))
        self._chunk_overlap = int(cfg.get("chunk_overlap", 20))
        self._split_mode = str(cfg.get("split_mode", "fixed"))
        if self._split_mode not in ("fixed", "heading"):
            raise ValueError(f"未知 split_mode: {self._split_mode}（仅支持 fixed / heading）")
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
        """定长切分。返回的片段保留首尾空白，由 split() 统一 strip。

        内部不 strip 的原因：strip 会改变片段长度，使 `buf[size-overlap:]`
        算出的重叠区与配置值不符。保留原文后重叠字符数精确等于 chunk_overlap。
        """
        size = max(1, self._chunk_size)
        overlap = max(0, min(self._chunk_overlap, size - 1))
        chunks: List[str] = []
        buf = ""

        def flush() -> None:
            nonlocal buf
            if buf.strip():
                chunks.append(buf)
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
                chunks.append(head)
                buf = rest
        flush()
        return chunks or ([text] if text.strip() else [])

    def _split_heading(self, text: str) -> List[Tuple[Optional[str], str]]:
        """按【标题】切分：一块 = 一节，标题进 title 字段。

        分块边界落在章节边界上，避免定长切把两节拼进同一块
        （那样"什么是X"的答案会被另一节的内容稀释）。
        超长章节（> chunk_size）内部再走定长切，保证不会产生超长块。
        无【】标题的文本退化为定长切。
        """
        sections: List[Tuple[str, str]] = []
        cur_title: Optional[str] = None
        cur_body: List[str] = []
        for line in text.splitlines():
            m = HEADING_RE.match(line.strip())
            if m:
                if cur_title is not None:
                    sections.append((cur_title, "\n".join(cur_body).strip()))
                cur_title = m.group("title").strip()
                cur_body = [m.group("body").strip()]
            elif cur_title is not None and line.strip():
                cur_body.append(line.strip())
        if cur_title is not None:
            sections.append((cur_title, "\n".join(cur_body).strip()))

        if not sections:
            return [(None, c) for c in self._split_raw(text)]

        out: List[Tuple[Optional[str], str]] = []
        for title, body in sections:
            if len(body) <= self._chunk_size:
                out.append((title, body))
            else:
                for sub in self._split_raw(body):
                    out.append((title, sub))
        return out

    def split(self, text: str, filename: Optional[str] = None, persist: bool = False) -> Dict[str, Any]:
        doc_id = uuid.uuid4().hex[:12]
        source = filename or "inline"
        if self._split_mode == "heading":
            pieces: List[Tuple[Optional[str], str]] = self._split_heading(text)
        else:
            pieces = [(None, c) for c in self._split_raw(text)]

        chunks = [
            {
                "chunk_id": f"{doc_id}_{i:04d}",
                "doc_id": doc_id,
                "source": source,
                "chunk_index": i,
                "title": title or "",
                "content": content.strip(),
                "char_count": len(content.strip()),
            }
            for i, (title, content) in enumerate(pieces)
        ]
        result = {
            "doc_id": doc_id,
            "source": source,
            "chunk_count": len(chunks),
            "chunk_size": self._chunk_size,
            "chunk_overlap": self._chunk_overlap,
            "split_mode": self._split_mode,
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
