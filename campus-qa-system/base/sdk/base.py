"""插件契约：本地实现与远程代理共用同一接口。"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from typing import Any, Dict, List


@dataclass(frozen=True)
class PluginMeta:
    """插件元信息。name 在节点内唯一，建议与插件目录/RPC 前缀一致。"""

    name: str
    version: str = "0.1.0"
    description: str = ""
    group: str = ""  # file_mgmt / vector_db / kg / qa / framework

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class Plugin(ABC):
    """所有业务插件的最小基类。"""

    meta: PluginMeta

    @abstractmethod
    def initialize(self, config: Dict[str, Any] | None = None) -> None:
        """注入配置并准备资源。节点启动时调用一次。"""

    @abstractmethod
    def methods(self) -> List[str]:
        """对外可 RPC 调用的方法名（不含插件前缀）。必须包含业务方法。"""

    @abstractmethod
    def call(self, method: str, params: Dict[str, Any]) -> Any:
        """按方法名分发。未知名方法应抛出 KeyError。"""

    def shutdown(self) -> None:
        """释放资源，默认无操作。"""

    def info(self) -> Dict[str, Any]:
        return {**self.meta.to_dict(), "methods": self.methods()}


# RPC method 命名约定：f"{plugin.meta.name}.{method}"
# 例如 doc_parser.split、vector_store.search
RPC_METHOD_SEP = "."
