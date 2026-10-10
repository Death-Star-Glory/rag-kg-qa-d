"""vector_db 接入基座验证：起真实节点进程，用 RemotePluginProxy 远程调用。

验证的是交接文档第 10 节的硬要求：
    新插件必须能被 RemotePluginProxy 调用，否则视为未接入基座。

与 smoke_vector_db.py 的区别：
    smoke_vector_db.py   本地直连插件（PluginRegistry.create），验证业务逻辑
    本脚本               起 vector_db_node 进程走 HTTP RPC，验证接入基座

跑法（需 fastapi/uvicorn，用装了依赖的解释器）：
    python smoke_rpc_vector_db.py

依赖：fastapi / uvicorn / pyyaml（见 requirements.txt）
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from base.rpc.client import RemotePluginProxy, RpcClient  # noqa: E402

NODE = "vector_db_node"
HOST, PORT = "127.0.0.1", 8102
BASE = f"http://{HOST}:{PORT}"
CORPUS = ROOT / "evals" / "vector_db" / "corpus" / "os_course.txt"

IN_SCOPE = ["进程有哪三种基本状态？", "产生死锁的四个必要条件是什么？", "线程和进程有什么区别？"]
OUT_SCOPE = ["今天广州天气怎么样？", "帮我写一段 Python 快速排序代码。"]


def wait_ready(timeout: float = 25.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(f"{BASE}/health", timeout=1.5) as resp:
                json.loads(resp.read().decode("utf-8"))
                return True
        except Exception:  # noqa: BLE001
            time.sleep(0.4)
    return False


def load_chunks() -> list:
    """本地分块后通过 RPC 灌给远程向量库 —— 演示编排层的数据流。

    真实链路里 chunks 由 file_mgmt_node 的 doc_parser 产出，
    这里为减少启动节点数，直接本地调用同一插件。
    """
    from plugins.file_mgmt import DocParserPlugin

    text = "".join(
        l for l in CORPUS.read_text(encoding="utf-8").splitlines(keepends=True)
        if not l.lstrip().startswith("#")
    ).strip()
    parser = DocParserPlugin()
    parser.initialize({
        "chunk_size": 160,
        "chunk_overlap": 24,
        "split_mode": "heading",
        "chunk_dir": str(ROOT / "data" / "rpc_smoke" / "chunks"),
    })
    return parser.call("split", {"text": text, "filename": CORPUS.name, "persist": False})["chunks"]


def main() -> int:
    try:
        import fastapi  # noqa: F401
        import uvicorn  # noqa: F401
    except ImportError as e:
        print(f"[skip] 本脚本需要 fastapi/uvicorn/yaml：{e}")
        print("       安装：pip install -r requirements.txt")
        print("       注：本脚本用 sys.executable 起子进程，"
              "故必须用装了依赖的那个解释器来运行本脚本。")
        return 2

    if not CORPUS.exists():
        print(f"找不到语料 {CORPUS}")
        return 1

    log = open(os.devnull, "w")
    proc = subprocess.Popen(
        [sys.executable, "-m", "base.node", "--node", NODE],
        cwd=str(ROOT),
        stdout=log,
        stderr=subprocess.STDOUT,
    )
    try:
        if not wait_ready():
            print(f"[FAIL] 节点 {NODE} 未在超时内就绪（端口 {PORT} 可能被占用）")
            return 1
        print(f"== 节点就绪 == {BASE}")

        health = RpcClient(BASE).health()
        print(f"  node={health.get('node')}  plugins={health.get('plugins')}")

        # ---------- 1. RemotePluginProxy 能否初始化 ----------
        print("\n== 远程代理初始化 ==")
        embedder = RemotePluginProxy("embedder", BASE)
        embedder.initialize()
        store = RemotePluginProxy("vector_store", BASE)
        store.initialize()
        print(f"  embedder.methods()      = {embedder.methods()}")
        print(f"  vector_store.methods()  = {store.methods()}")
        assert "search" in store.methods(), "vector_store 未暴露 search"
        assert "analyze" in store.methods(), "vector_store 未暴露 analyze"

        # ---------- 2. 远程建索引 ----------
        print("\n== 远程建索引 ==")
        chunks = load_chunks()
        idx = store.call("index", {"chunks": chunks})
        print(f"  {idx}")

        # ---------- 3. 远程检索 ----------
        print("\n== 远程检索 ==")
        scores = {"库内": [], "库外": []}
        for group, questions in (("库内", IN_SCOPE), ("库外", OUT_SCOPE)):
            for q in questions:
                r = store.call("search", {"query": q, "top_k": 3})
                top = r["hits"][0]
                scores[group].append(top["score"])
                title = top.get("title") or "-"
                print(f"  [{group}] {top['score']:.4f}  [{title}]  {q}")

        # ---------- 4. 分数尺度与可答性信号 ----------
        print("\n== 分数尺度 ==")
        scale = store.call("score_scale", {})
        print(f"  formula={scale['formula']}  range={scale['range']}")

        print("\n== 可答性信号 analyze ==")
        for group, questions in (("库内", IN_SCOPE[:1]), ("库外", OUT_SCOPE)):
            for q in questions:
                a = store.call("analyze", {"query": q})
                print(f"  [{group}] oov={a['oov_ratio']:.2f}  {q}")

        # ---------- 5. 断言 ----------
        print("\n== 校验 ==")
        all_scores = scores["库内"] + scores["库外"]
        oob = [s for s in all_scores if not (0.0 <= s <= 1.0)]
        in_min, out_max = min(scores["库内"]), max(scores["库外"])
        gap = in_min - out_max
        checks = [
            ("全部分数落在 [0,1]", not oob),
            ("库内最低分 > 库外最高分（存在分隔带）", gap > 0),
        ]
        for name, ok in checks:
            print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
        print(f"  库内最低分={in_min:.4f}  库外最高分={out_max:.4f}  分隔带={gap:+.4f}")

        if not all(ok for _, ok in checks):
            return 1

        print("\nvector_db 接入基座验证通过（RemotePluginProxy 可调用）")
        return 0
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=8)
        except subprocess.TimeoutExpired:
            proc.kill()
        log.close()


if __name__ == "__main__":
    raise SystemExit(main() or 0)
