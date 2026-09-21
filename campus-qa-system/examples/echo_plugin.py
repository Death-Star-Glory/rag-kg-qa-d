"""回声插件：基座连通性验证，无业务语义。"""

from __future__ import annotations

import time
from typing import Any, Dict, List

from base.sdk.base import Plugin, PluginMeta


class EchoPlugin(Plugin):
    meta = PluginMeta(
        name="echo",
        version="0.1.0",
        description="基座示例：原样返回参数，用于验证插件加载与 RPC",
        group="framework",
    )

    def __init__(self) -> None:
        self._greeting = "pong"
        self._started_at: float | None = None

    def initialize(self, config: Dict[str, Any] | None = None) -> None:
        cfg = config or {}
        self._greeting = str(cfg.get("greeting", "pong"))
        self._started_at = time.time()

    def methods(self) -> List[str]:
        return ["ping", "echo"]

    def call(self, method: str, params: Dict[str, Any]) -> Any:
        if method == "info":
            return self.info()
        if method == "ping":
            return {
                "message": self._greeting,
                "uptime_sec": (time.time() - (self._started_at or time.time())),
            }
        if method == "echo":
            return {"echo": params.get("message", ""), "params": params}
        raise KeyError(f"未知方法: {method}")
