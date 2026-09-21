"""拓扑配置加载：优先 PyYAML，缺失时用内置极简解析。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

DEFAULT_TOPOLOGY = Path(__file__).resolve().parents[1] / "config" / "topology.yaml"


def load_topology(path: str | Path | None = None) -> Dict[str, Any]:
    raw = Path(path) if path else DEFAULT_TOPOLOGY
    if not raw.is_absolute():
        # 相对路径：先相对 cwd，再相对项目根
        cand = Path.cwd() / raw
        if not cand.exists():
            cand = Path(__file__).resolve().parents[1] / raw
        raw = cand
    if not raw.exists():
        raise FileNotFoundError(f"拓扑配置不存在: {path} (resolved={raw})")

    text = raw.read_text(encoding="utf-8")
    if raw.suffix.lower() == ".json":
        return json.loads(text)
    try:
        import yaml  # type: ignore

        data = yaml.safe_load(text) or {}
    except ImportError:
        data = _parse_simple_yaml(text)
    if "nodes" not in data:
        raise ValueError(f"拓扑配置缺少 nodes 键: {raw}")
    return data


def get_node_config(topology: Dict[str, Any], node_name: str) -> Dict[str, Any]:
    nodes = topology.get("nodes") or {}
    if node_name not in nodes:
        raise KeyError(f"节点不存在: {node_name}，可用: {', '.join(nodes)}")
    return nodes[node_name] or {}


def _parse_simple_yaml(text: str) -> Dict[str, Any]:
    """仅支持本项目 topology 的 2 空格缩进子集。"""
    root: Dict[str, Any] = {}
    stack: list[tuple[int, Dict[str, Any]]] = [(-1, root)]

    for raw_line in text.splitlines():
        if not raw_line.strip() or raw_line.lstrip().startswith("#"):
            continue
        indent = len(raw_line) - len(raw_line.lstrip(" "))
        line = raw_line.strip()
        if ":" not in line:
            continue
        key, _, rest = line.partition(":")
        key = key.strip()
        rest = rest.strip()

        while len(stack) > 1 and indent <= stack[-1][0]:
            stack.pop()
        parent = stack[-1][1]

        if rest == "":
            child: Dict[str, Any] = {}
            parent[key] = child
            stack.append((indent, child))
            continue

        parent[key] = _scalar(rest)
    return root


def _scalar(token: str) -> Any:
    if token.lower() in {"true", "false"}:
        return token.lower() == "true"
    if token.lower() in {"null", "~"}:
        return None
    if (token.startswith('"') and token.endswith('"')) or (
        token.startswith("'") and token.endswith("'")
    ):
        return token[1:-1]
    try:
        return int(token)
    except ValueError:
        pass
    try:
        return float(token)
    except ValueError:
        return token
