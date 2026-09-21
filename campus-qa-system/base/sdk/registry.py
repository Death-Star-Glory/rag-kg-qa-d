"""本地插件注册表：名称 → 工厂/实例。"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Type

from .base import Plugin


class PluginRegistry:
    def __init__(self) -> None:
        self._factories: Dict[str, Callable[..., Plugin]] = {}
        self._instances: Dict[str, Plugin] = {}

    def register(self, name: str, factory: Callable[..., Plugin]) -> None:
        if name in self._factories:
            raise ValueError(f"插件名重复注册: {name}")
        self._factories[name] = factory

    def create(self, name: str, config: Optional[Dict[str, Any]] = None) -> Plugin:
        if name in self._instances:
            return self._instances[name]
        if name not in self._factories:
            raise KeyError(f"插件未注册: {name}")
        plugin = self._factories[name]()
        plugin.initialize(config or {})
        self._instances[name] = plugin
        return plugin

    def get(self, name: str) -> Plugin:
        if name not in self._instances:
            raise KeyError(f"插件未实例化: {name}")
        return self._instances[name]

    def has(self, name: str) -> bool:
        return name in self._instances or name in self._factories

    def names(self) -> List[str]:
        return sorted(set(self._factories) | set(self._instances))

    def shutdown_all(self) -> None:
        for plugin in list(self._instances.values()):
            plugin.shutdown()
        self._instances.clear()


def register_plugin(registry: PluginRegistry, name: str):
    """装饰器：将插件类注册到 registry。"""

    def decorator(cls: Type[Plugin]) -> Type[Plugin]:
        registry.register(name, cls)
        return cls

    return decorator
