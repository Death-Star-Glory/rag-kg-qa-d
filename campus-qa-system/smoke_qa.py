"""qa 模块冒烟：三态判定 + 生成 + 溯源，端到端跑 5 类问题。

覆盖课设过关硬条件的每一种形态：
    1. 库内知识题        → 作答，且答案带来源段落
    2. 库外无关题        → 拒答（硬性条件：库外拒答率 ≥70%）
    3. 诱导题（问数值）   → 拒答（可支撑性检查兜底）
    4. 模糊题（无先行词） → 澄清
    5. 模糊题（可锚定）   → 作答

用法：
    python smoke_qa.py            # 零第三方依赖
退出码：0 = 通过；1 = 有断言失败。
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from plugins.file_mgmt.doc_parser import DocParserPlugin  # noqa: E402
from plugins.kg.kg_builder import KgBuilderPlugin  # noqa: E402
from plugins.kg.kg_query import KgQueryPlugin  # noqa: E402
from plugins.qa.fusion import FusionQaPlugin  # noqa: E402
from plugins.qa.generator import GeneratorPlugin  # noqa: E402
from plugins.qa.retriever import RagRetrieverPlugin  # noqa: E402
from plugins.vector_db.embedder import EmbedderPlugin  # noqa: E402
from plugins.vector_db.vector_store import VectorStorePlugin  # noqa: E402

CORPUS = ROOT / "evals" / "vector_db" / "corpus" / "os_course.txt"


def build_fusion():
    """单进程装配：heading 分块（有 title，澄清判据可用）→ 向量索引 → 三态问答。"""
    text = CORPUS.read_text(encoding="utf-8")

    parser = DocParserPlugin()
    parser.initialize({"split_mode": "heading"})
    chunks = parser.call(
        "split", {"text": text, "filename": "os_course.txt"}
    )["chunks"]

    embedder = EmbedderPlugin()
    embedder.initialize({"dim": 4096})
    store = VectorStorePlugin()
    store.initialize({"embedder": embedder, "dim": 4096})
    store.call("index", {"chunks": chunks})

    retriever = RagRetrieverPlugin()
    retriever.initialize({"vector_store": store})

    kg_builder = KgBuilderPlugin()
    kg_builder.initialize({})
    kg_builder.call("build_text", {"text": text, "source": "os_course.txt"})
    kg_query = KgQueryPlugin()
    kg_query.initialize({"store": kg_builder.store, "max_nodes": 50})

    generator = GeneratorPlugin()
    generator.initialize({"mode": "extractive"})

    fusion = FusionQaPlugin()
    fusion.initialize(
        {
            "retriever": retriever,
            "kg_query": kg_query,
            "generator": generator,
            "top_k": 3,
            "tau": 0.185,
        }
    )
    return fusion, store


def main() -> int:
    fusion, store = build_fusion()
    stats = store.call("stats", {})
    print(f"索引：{stats['indexed']} 块，章节标题 {len(stats.get('titles') or [])} 个")

    cases = [
        # (问题, 期望 state, 说明)
        ("什么是进程？", "answered", "库内知识题 → 作答 + 来源"),
        ("进程和程序有什么区别？", "answered", "库内知识题 → 作答 + 来源"),
        ("明天广州的天气怎么样？", "refused", "库外无关题 → 拒答"),
        ("帮我写一段 Python 快速排序代码。", "refused", "库外无关题 → 拒答"),
        ("时间片轮转算法的时间片通常设为多少毫秒？", "refused", "诱导题（问数值）→ 可支撑性检查兜底拒答"),
        ("进程控制块 PCB 在 Linux 内核源码中对应哪个结构体？", "refused", "诱导题（问实现）→ 可支撑性检查兜底拒答"),
        ("操作系统是怎么管理这些东西的？", "clarify", "模糊题（“这些”无先行词，锚不到章节）→ 澄清"),
        ("讲讲进程。", "answered", "模糊题（可锚定）→ 作答"),
    ]

    failed = 0
    for question, expect_state, desc in cases:
        result = fusion.call("ask", {"question": question, "top_k": 3})
        state = result["state"]
        ok = state == expect_state
        if ok and state == "answered":
            ok = "来源" in result["answer"] and bool(result["sources"])
        mark = "PASS" if ok else "FAIL"
        if not ok:
            failed += 1
        print(f"\n[{mark}] {desc}")
        print(f"  问：{question}")
        print(f"  态：{state}（期望 {expect_state}）  top1={result['decision'].get('top1_score')}")
        print(f"  答：{result['answer'][:120]}{'…' if len(result['answer']) > 120 else ''}")
        if result["sources"]:
            print(f"  溯源：{[s.get('title') or s.get('source') for s in result['sources'][:3]]}")

    print()
    if failed:
        print(f"qa 冒烟未通过（{failed} 条失败）")
        return 1
    print("qa 冒烟通过（三态判定 / 诱导题兜底 / 生成溯源 全部符合预期）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
