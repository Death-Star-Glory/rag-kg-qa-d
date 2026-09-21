"""基座自检：插件加载 + 本地 call；--rpc 时再验远程。"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from base.config_loader import get_node_config, load_topology  # noqa: E402
from base.node import build_registry  # noqa: E402


def run_local() -> None:
    topology = load_topology(ROOT / "config" / "topology.yaml")
    node_cfg = get_node_config(topology, "echo_node")
    registry = build_registry(node_cfg)
    echo = registry.get("echo")

    print("== local echo.ping ==")
    pong = echo.call("ping", {})
    print(pong)
    assert pong.get("message") == "pong", pong

    print("== local echo.echo ==")
    echoed = echo.call("echo", {"message": "campus-qa"})
    print(echoed)
    assert echoed["echo"] == "campus-qa"

    print("== local plugin.info ==")
    print(echo.info())
    print("本地基座 OK")


def run_rpc(base_url: str) -> None:
    from base.rpc.client import RemotePluginProxy
    from base.sdk.base import PluginMeta

    print(f"\n== RPC {base_url} ==")
    proxy = RemotePluginProxy("echo", base_url, PluginMeta(name="echo"))
    proxy.initialize()
    print("remote info:", proxy.info())
    result = proxy.call("echo", {"message": "from-rpc"})
    print(result)
    assert result["echo"] == "from-rpc"
    print("RPC 基座 OK")


if __name__ == "__main__":
    run_local()
    if len(sys.argv) > 1 and sys.argv[1].startswith("http"):
        run_rpc(sys.argv[1])
        print("\n全部通过")
    elif "--rpc" in sys.argv:
        run_rpc("http://127.0.0.1:8200")
        print("\n全部通过")
    else:
        print("\n本地通过。远程验证：")
        print("  python -m base.node --node echo_node")
        print("  python smoke_base.py --rpc")
