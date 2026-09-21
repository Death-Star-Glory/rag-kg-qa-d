"""节点间统一 JSON-RPC 信封。"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class RpcRequest:
    method: str  # "{plugin_name}.{method}"
    params: Dict[str, Any] = field(default_factory=dict)
    id: str = field(default_factory=lambda: uuid.uuid4().hex)

    def to_dict(self) -> Dict[str, Any]:
        return {"id": self.id, "method": self.method, "params": self.params}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RpcRequest":
        return cls(
            method=data["method"],
            params=data.get("params") or {},
            id=data.get("id") or uuid.uuid4().hex,
        )


@dataclass
class RpcError:
    code: int
    message: str
    data: Optional[Any] = None

    def to_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {"code": self.code, "message": self.message}
        if self.data is not None:
            out["data"] = self.data
        return out


@dataclass
class RpcResponse:
    id: str
    result: Any = None
    error: Optional[RpcError] = None

    @property
    def ok(self) -> bool:
        return self.error is None

    def to_dict(self) -> Dict[str, Any]:
        if self.error is not None:
            return {"id": self.id, "error": self.error.to_dict()}
        return {"id": self.id, "result": self.result}

    @classmethod
    def success(cls, id: str, result: Any) -> "RpcResponse":
        return cls(id=id, result=result)

    @classmethod
    def failure(
        cls, id: str, code: int, message: str, data: Any = None
    ) -> "RpcResponse":
        return cls(id=id, error=RpcError(code=code, message=message, data=data))


# 错误码约定
ERR_INVALID_REQUEST = -32600
ERR_PLUGIN_NOT_FOUND = -32601
ERR_METHOD_NOT_FOUND = -32602
ERR_INTERNAL = -32000
