"""file-store：原文与元数据，不做检索/图谱。"""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

from base.sdk.base import Plugin, PluginMeta

_SAFE = re.compile(r"^[\w\-.一-鿿]+$")


class FileStorePlugin(Plugin):
    meta = PluginMeta(
        name="file_store",
        version="0.1.0",
        description="文件存储：save/load/list/delete",
        group="file_mgmt",
    )

    def __init__(self) -> None:
        self._root = Path("data/files")
        self._meta_path = Path("data/files_meta.json")
        self._index: Dict[str, Dict[str, Any]] = {}

    def initialize(self, config: Dict[str, Any] | None = None) -> None:
        cfg = config or {}
        self._root = Path(cfg.get("root", "data/files"))
        self._meta_path = Path(cfg.get("meta_path", str(self._root) + "_meta.json"))
        self._root.mkdir(parents=True, exist_ok=True)
        if self._meta_path.exists():
            self._index = json.loads(self._meta_path.read_text(encoding="utf-8"))
        else:
            self._index = {}

    def methods(self) -> List[str]:
        return ["save", "load", "list", "delete", "info"]

    def call(self, method: str, params: Dict[str, Any]) -> Any:
        if method == "info":
            return self.info()
        if method == "save":
            return self.save(params["filename"], params["content"], params.get("source"))
        if method == "load":
            return self.load(params["filename"])
        if method == "list":
            return self.list_files()
        if method == "delete":
            return self.delete(params["filename"])
        raise KeyError(f"未知方法: {method}")

    def _name(self, filename: str) -> str:
        name = Path(filename).name
        if not name or not _SAFE.match(name):
            raise ValueError(f"非法文件名: {filename}")
        return name

    def _flush(self) -> None:
        self._meta_path.parent.mkdir(parents=True, exist_ok=True)
        self._meta_path.write_text(
            json.dumps(self._index, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    def save(self, filename: str, content: str, source: str | None = None) -> Dict[str, Any]:
        name = self._name(filename)
        path = self._root / name
        path.write_text(content, encoding="utf-8")
        rec = {
            "filename": name,
            "path": str(path),
            "size": len(content.encode("utf-8")),
            "source": source or name,
            "saved_at": datetime.now(timezone.utc).isoformat(),
        }
        self._index[name] = rec
        self._flush()
        return rec

    def load(self, filename: str) -> Dict[str, Any]:
        name = self._name(filename)
        path = self._root / name
        if not path.exists():
            raise FileNotFoundError(f"文件不存在: {name}")
        meta = self._index.get(name, {"filename": name, "source": name})
        return {**meta, "content": path.read_text(encoding="utf-8")}

    def list_files(self) -> List[Dict[str, Any]]:
        out = []
        for path in sorted(self._root.iterdir()) if self._root.exists() else []:
            if not path.is_file():
                continue
            meta = self._index.get(path.name, {})
            out.append(
                {
                    "filename": path.name,
                    "size": path.stat().st_size,
                    "source": meta.get("source", path.name),
                    "saved_at": meta.get("saved_at"),
                }
            )
        return out

    def delete(self, filename: str) -> Dict[str, Any]:
        name = self._name(filename)
        path = self._root / name
        if path.exists():
            path.unlink()
        self._index.pop(name, None)
        self._flush()
        return {"deleted": name, "ok": True}
