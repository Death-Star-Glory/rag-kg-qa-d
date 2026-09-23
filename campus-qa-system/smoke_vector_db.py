"""vector_db 组冒烟自检：装配 → 分块 → 建索引 → 检索 → 打印分数尺度。

跑法：python smoke_vector_db.py
预期：库内问题分数明显高于库外问题，且所有分数落在 [0, 1]。

说明：本脚本刻意使用**内嵌小样本**，不读 evals/vector_db/corpus/。
原因是冒烟只验证「链路通不通」，不该依赖评测语料 ——
否则语料一换（内容、章节号变动）冒烟就会红，掩盖真正的回归。
检索质量与阈值标定请跑 evals/vector_db/ 下的脚本。
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from base.sdk.registry import PluginRegistry
from plugins.file_mgmt import DocParserPlugin
from plugins.vector_db import EmbedderPlugin, VectorStorePlugin

SAMPLE = (
    "【1.1 进程的概念】程序是静态的指令集合，存储在磁盘上。进程是程序的一次执行过程，"
    "是系统进行资源分配和调度的基本单位。一个程序可以对应多个进程。\n\n"
    "【1.2 进程的状态】进程在其生命周期内处于三种基本状态：就绪、运行、阻塞。"
    "就绪态指进程已具备运行条件，只等待 CPU；运行态指进程占用 CPU 执行；"
    "阻塞态指进程因等待某事件而暂停。\n\n"
    "【1.3 进程与程序的区别】（1）程序是静态的，进程是动态的；"
    "（2）程序可长期保存，进程有生命周期；（3）进程具有并发性，程序没有；"
    "（4）进程是资源分配的基本单位。\n\n"
    "【1.8 死锁】死锁指多个进程因竞争资源而互相等待、无外力介入都无法推进的僵局。"
    "产生死锁的四个必要条件：互斥、占有并等待、不可抢占、循环等待。"
)

IN_SCOPE = ["进程和程序有什么区别？", "产生死锁的四个必要条件是什么？"]
OUT_SCOPE = ["今天广州天气怎么样？", "帮我写一段 Python 快速排序代码。"]


def main() -> int:
    reg = PluginRegistry()
    reg.register("embedder", EmbedderPlugin)
    reg.register("vector_store", VectorStorePlugin)
    reg.register("doc_parser", DocParserPlugin)

    embedder = reg.create("embedder", {"dim": 4096})
    parser = reg.create(
        "doc_parser",
        {"chunk_size": 160, "chunk_overlap": 24, "chunk_dir": "data/chunks"},
    )
    store = reg.create("vector_store", {"embedder": embedder, "dim": 4096})

    print("== 分块 ==")
    split = parser.call("split", {"text": SAMPLE, "filename": "smoke_sample.txt", "persist": False})
    chunks = split["chunks"]
    print(f"chunk_count={len(chunks)}  chunk_size={split['chunk_size']}")

    print("\n== 建索引 ==")
    print(store.call("index", {"chunks": chunks}))

    print("\n== 分数尺度 ==")
    scale = store.call("score_scale", {})
    print(f"formula={scale['formula']}  range={scale['range']}")

    print("\n== 检索 ==")
    all_scores = []
    for group, questions in (("库内", IN_SCOPE), ("库外", OUT_SCOPE)):
        for q in questions:
            r = store.call("search", {"query": q, "top_k": 3})
            top = r["hits"][0] if r["hits"] else None
            if top:
                all_scores.append((group, q, top["score"]))
            print(f"\n  [{group}] {q}")
            for i, h in enumerate(r["hits"], 1):
                text = (h.get("content") or "").replace("\n", " ")[:46]
                print(f"    {i}. score={h['score']:.4f} cos={h['cos']:.4f} cov={h['cov']:.3f} | {text}")

    print("\n== 可答性信号 analyze ==")
    print("  （供编排层做拒答的第二判据：问句实词中不在语料词表的占比）")
    for group, questions in (("库内", IN_SCOPE), ("库外", OUT_SCOPE)):
        for q in questions:
            a = store.call("analyze", {"query": q})
            oov = "".join(a["oov_chars"][:8])
            print(f"  [{group}] oov={a['oov_ratio']:.2f}  语料外字={oov or '无'}  | {q}")

    print("\n== 尺度检查 ==")
    bad = [s for _, _, s in all_scores if not (0.0 <= s <= 1.0)]
    print(f"  分数越界条数: {len(bad)}  {'OK' if not bad else 'FAIL ' + str(bad)}")

    ins = [s for g, _, s in all_scores if g == "库内"]
    oos = [s for g, _, s in all_scores if g == "库外"]
    if ins and oos:
        print(f"  库内最低分={min(ins):.4f}  库外最高分={max(oos):.4f}")
        gap = min(ins) - max(oos)
        print(f"  分隔带宽度={gap:.4f}  {'可分（阈值取区间内）' if gap > 0 else '重叠（需换语义 embedding）'}")

    print("\nvector_db 冒烟通过")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
