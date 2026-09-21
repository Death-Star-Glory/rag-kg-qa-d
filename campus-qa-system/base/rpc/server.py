"""节点 HTTP 服务：/health /plugins /rpc。"""

from typing import Any, Dict, Optional

from ..sdk.registry import PluginRegistry
from . import protocol as p
from .protocol import RpcRequest, RpcResponse

try:
    from fastapi import Body, FastAPI, Request
    FASTAPI_AVAILABLE = True
    FASTAPI_IMPORT_ERROR: Optional[Exception] = None
except ImportError as _e:  # pragma: no cover
    FASTAPI_AVAILABLE = False
    FASTAPI_IMPORT_ERROR = _e
    Body = FastAPI = Request = None  # type: ignore


def create_node_app(registry: PluginRegistry, node_name: str = "node"):
    if not FASTAPI_AVAILABLE:
        raise RuntimeError(
            "节点需要 fastapi/uvicorn: pip install fastapi uvicorn pyyaml"
            + (f" ({FASTAPI_IMPORT_ERROR})" if FASTAPI_IMPORT_ERROR else "")
        )

    app = FastAPI(title=f"campus-qa:{node_name}")

    @app.get("/health")
    def health() -> Dict[str, Any]:
        return {
            "status": "ok",
            "node": node_name,
            "plugins": registry.names(),
        }

    @app.get("/plugins")
    def plugins() -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        for name in registry.names():
            try:
                out[name] = registry.get(name).info()
            except KeyError:
                out[name] = {"name": name, "status": "registered_not_ready"}
        return out

    # 注意：本文件不要用 from __future__ import annotations
    # 否则 FastAPI 无法在模块全局解析 Request/Dict 等注解
    @app.post("/rpc")
    async def rpc(request: Request, payload: Dict[str, Any] = Body(default=None)) -> Dict[str, Any]:
        import uuid

        data = payload
        if data is None:
            try:
                data = await request.json()
            except Exception:
                data = None

        if not isinstance(data, dict) or "method" not in data:
            rid = str((data or {}).get("id") or "") if isinstance(data, dict) else ""
            return RpcResponse.failure(
                rid or uuid.uuid4().hex,
                p.ERR_INVALID_REQUEST,
                "请求体须为 JSON 且包含 method 字段",
            ).to_dict()

        params = data.get("params") or {}
        if not isinstance(params, dict):
            return RpcResponse.failure(
                str(data.get("id") or uuid.uuid4().hex),
                p.ERR_INVALID_REQUEST,
                "params 必须是对象",
            ).to_dict()

        req = RpcRequest(
            method=str(data["method"]),
            params=params,
            id=str(data.get("id") or uuid.uuid4().hex),
        )

        if "." not in req.method:
            return RpcResponse.failure(
                req.id, p.ERR_INVALID_REQUEST, f"非法 method: {req.method}"
            ).to_dict()

        plugin_name, method = req.method.split(".", 1)
        if not registry.has(plugin_name):
            return RpcResponse.failure(
                req.id, p.ERR_PLUGIN_NOT_FOUND, f"插件不存在: {plugin_name}"
            ).to_dict()

        try:
            plugin = registry.create(plugin_name)
            allowed = set(plugin.methods()) | {"info"}
            if method not in allowed:
                return RpcResponse.failure(
                    req.id,
                    p.ERR_METHOD_NOT_FOUND,
                    f"方法不存在: {req.method}",
                    data=plugin.methods(),
                ).to_dict()
            result = plugin.call(method, req.params)
            return RpcResponse.success(req.id, result).to_dict()
        except KeyError as e:
            return RpcResponse.failure(req.id, p.ERR_METHOD_NOT_FOUND, str(e)).to_dict()
        except Exception as e:  # noqa: BLE001
            return RpcResponse.failure(req.id, p.ERR_INTERNAL, str(e)).to_dict()

    return app
