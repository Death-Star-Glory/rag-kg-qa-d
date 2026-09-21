"""按配置动态加载插件类。

配置项 entry 支持两种写法：
1. "examples.echo_plugin:EchoPlugin"
2. "base.plugins.builtin:EchoPlugin"  （包内路径）
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Type

from .base import Plugin
from .registry import PluginRegistry

# 项目根目录（campus-qa-system），保证 examples.* 可导入
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def load_plugin_class(entry: str) -> Type[Plugin]:
    if ":" not in entry:
        raise ValueError(f"插件 entry 格式应为 package.module:ClassName，收到: {entry}")
    module_name, class_name = entry.split(":", 1)
    module = importlib.import_module(module_name)
    cls = getattr(module, class_name, None)
    if cls is None:
        raise ImportError(f"模块 {module_name} 中不存在类 {class_name}")
    if not (isinstance(cls, type) and issubclass(cls, Plugin)):
        raise TypeError(f"{entry} 不是 Plugin 子类")
    return cls


def create_plugin_instance(entry: str, config: Optional[Dict[str, Any]] = None) -> Plugin:
    cls = load_plugin_class(entry)
    plugin = cls()
    plugin.initialize(config or {})
    return plugin


def load_plugin_entries(
    registry: PluginRegistry,
    plugin_configs: Dict[str, Dict[str, Any]],
) -> None:
    """
    将节点配置中的 plugins 段装载进 registry。

    plugin_configs 结构:
      {
        "echo": {"entry": "examples.echo_plugin:EchoPlugin", "config": {...}},
        ...
      }
    """
    for name, spec in plugin_configs.items():
        entry = spec.get("entry") or spec.get("class")
        if not entry:
            raise KeyError(f"插件 {name} 缺少 entry 配置")
        cfg = dict(spec.get("config") or {})
        # 业务插件可在 config 里声明协作插件名，bootstrap 阶段统一注入
        cls = load_plugin_class(entry)
        registry.register(name, cls)
        registry.create(name, cfg)
