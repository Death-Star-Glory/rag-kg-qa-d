"""Gateway：课设演示用 REST，聚合 file_mgmt / kg / qa 插件。"""

from .app import create_app, build_demo_registry

__all__ = ["create_app", "build_demo_registry"]
