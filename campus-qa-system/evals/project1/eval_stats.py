"""统计摘要脚本（《14_评测脚手架》第四节模板 + 组内口径扩展）。

用法：
    python evals/project1/eval_stats.py                # 读 results.jsonl
    python evals/project1/eval_stats.py results.jsonl
"""

from __future__ import annotations

import json
import os
import sys
from collections import Counter

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

HERE = os.path.dirname(os.path.abspath(__file__))


def main() -> int:
    path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "results.jsonl")
    rows = [json.loads(l) for l in open(path, encoding="utf-8")]
    c = Counter(r["result"] for r in rows)

    answered = c["correct"] + c["wrong"] + c["hallucinated"]
    acc = c["correct"] / answered if answered else 0
    refuse_total = c["refuse_correct"] + c["refuse_wrong"]
    refuse_rate = c["refuse_correct"] / refuse_total if refuse_total else 0

    by_type: Counter = Counter()
    for r in rows:
        by_type[f"{r['type']}/{r['result']}"] += 1

    print(f"总条数: {len(rows)}")
    print(f"准确率: {acc:.1%}  (correct={c['correct']}, wrong={c['wrong']}, hallucinated={c['hallucinated']})")
    print(f"拒答正确率: {refuse_rate:.1%}  (拒对={c['refuse_correct']}, 拒错={c['refuse_wrong']})")
    print(f"待澄清（单列，不计对错）: {c['clarify_ok']}")
    print(f"题型 x 判定分布: {dict(sorted(by_type.items()))}")
    if c["hallucinated"] >= 3:
        print("警告: hallucinated ≥ 3 条，复盘文档须逐条归因（脚手架第五节）")
    ok = refuse_rate >= 0.7 and len(rows) >= 20
    print("达标判定（拒答率≥70% 且 评测集≥20 条）:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
