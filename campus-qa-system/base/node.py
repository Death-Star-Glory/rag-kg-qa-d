"""节点运行时：读拓扑 → 装插件 → 暴露 RPC。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from base.config_loader import get_node_config, load_topology  # noqa: E402
from base.sdk.loader import load_plugin_entries  # noqa: E402
from base.sdk.registry import PluginRegistry  # noqa: E402
from base.rpc.server import create_node_app  # noqa: E402


def build_registry(node_cfg: Dict[str, Any]) -> PluginRegistry:
    registry = PluginRegistry()
    plugins_cfg = node_cfg.get("plugins") or {}
    load_plugin_entries(registry, plugins_cfg)
    return registry


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="campus-qa-system 节点基座")
    parser.add_argument(
        "--config",
        default=str(PROJECT_ROOT / "config" / "topology.yaml"),
        help="拓扑配置路径",
    )
    parser.add_argument("--node", required=True, help="topology.nodes 中的节点名")
    parser.add_argument("--host", default=None)
    parser.add_argument("--port", type=int, default=None)
    args = parser.parse_args(argv)

    topology = load_topology(args.config)
    node_cfg = get_node_config(topology, args.node)
    host = args.host or node_cfg.get("host", "127.0.0.1")
    port = args.port or int(node_cfg.get("port", 8200))
    group = node_cfg.get("group", "")

    registry = build_registry(node_cfg)
    app = create_node_app(registry, node_name=args.node)

    print(f"[campus-qa] node={args.node} group={group or '-'} http://{host}:{port}")
    print(f"[campus-qa] plugins={registry.names()}")

    try:
        import uvicorn
    except ImportError:
        raise SystemExit("请安装: pip install fastapi uvicorn pyyaml")

    uvicorn.run(app, host=host, port=port, log_level="info")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
