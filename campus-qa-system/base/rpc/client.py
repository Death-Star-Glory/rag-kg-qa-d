"""HTTP RPC 客户端与远程插件代理。"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
import uuid
from typing import Any, Dict, List, Optional

from ..sdk.base import Plugin, PluginMeta


class RpcClient:
    def __init__(self, base_url: str, timeout: float = 10.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def call(self, method: str, params: Optional[Dict[str, Any]] = None) -> Any:
        payload = {
            "id": uuid.uuid4().hex,
            "method": method,
            "params": params or {},
        }
        req = urllib.request.Request(
            url=f"{self.base_url}/rpc",
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                body = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"RPC HTTP {e.code}: {detail}") from e
        except urllib.error.URLError as e:
            raise RuntimeError(f"RPC 连接失败 {self.base_url}: {e.reason}") from e

        if body.get("error"):
            err = body["error"]
            raise RuntimeError(f"RPC [{err.get('code')}]: {err.get('message')}")
        return body.get("result")

    def health(self) -> Dict[str, Any]:
        req = urllib.request.Request(url=f"{self.base_url}/health", method="GET")
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))


class RemotePluginProxy(Plugin):
    """远程插件代理：调用方无需区分本地/远程。"""

    def __init__(
        self,
        name: str,
        base_url: str,
        meta: Optional[PluginMeta] = None,
        timeout: float = 10.0,
    ) -> None:
        self.meta = meta or PluginMeta(name=name, description=f"remote:{name}")
        self._client = RpcClient(base_url, timeout=timeout)
        self._methods: List[str] = []
        self._remote = True

    def initialize(self, config: Dict[str, Any] | None = None) -> None:
        info = self._client.call(f"{self.meta.name}.info")
        self._methods = list(info.get("methods") or [])
        self.meta = PluginMeta(
            name=info.get("name", self.meta.name),
            version=info.get("version", self.meta.version),
            description=info.get("description", self.meta.description),
            group=info.get("group", self.meta.group),
        )

    def methods(self) -> List[str]:
        return list(self._methods)

    def call(self, method: str, params: Dict[str, Any]) -> Any:
        return self._client.call(f"{self.meta.name}.{method}", params)

    def info(self) -> Dict[str, Any]:
        return {**self.meta.to_dict(), "methods": self.methods(), "remote": True}
