# -*- coding: utf-8 -*-
"""「模糊题该澄清」判据的参考实现与校验。

背景
----
评测集里 `vague` 档共 4 条，逐条判口径后是 **3 答 1 澄清**：

    q37 讲讲进程。                    → 应作答（指向 1.1/1.2/1.3）
    q39 调度相关的东西。               → 应作答（指向 1.6）
    q40 内存那一块。                   → 应作答（指向第 2 章）
    q38 操作系统是怎么管理这些东西的？    → **应澄清**（"这些东西"无先行词）

**为什么不能用分数解决**：q38 的 top1=0.2483，比可答题 q39 的 0.2082 还高。
抬 τ 会先把 q39(0.208) 和 q17(0.2427) 误伤掉。所以判据必须是**字面的**。

判据设计
--------
一条问句要同时满足三点才判「澄清」：

    1. 分数过了闸（≥ τ）—— 否则是拒答，不是澄清
    2. 含无先行词的指示代词（"这些""那一块""它"…）
    3. **问句锚不到任何章节标题** —— 即语料里没有哪一节是在讲它问的东西

第 3 点为什么锚定「全部章节标题」而不是「top-k 命中的标题」：
    检索失败时 top-k 里根本没有正确章节。例如 q40「内存那一块。」的 top-3 是
    1.4 PCB / 3.3 目录 / 3.2 文件物理结构 —— 全是错的。若按 top-k 锚定，
    q40 会被误判成「澄清」，把**检索失败**伪装成**口径问题**，两件事就混了。
    按全量标题锚定，「内存」能对上 2.1 内存管理的基本功能 → q40 走作答路径，
    它召回失败的事实如实留在 Hit@k 指标里，不被掩盖。

本文件做两件事：
    1. 给出 `need_clarify()` 参考实现（qa 组可直接复制到编排层）
    2. 在 40 题上校验：该规则**只**命中 q38，对其余 39 题零影响

用法：
    python evals/vector_db/check_clarify_rule.py
    python evals/vector_db/check_clarify_rule.py --split fixed
退出码：0 = 校验通过；1 = 规则误伤（改规则，不要改评测集口径）
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from typing import Any, Dict, List

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

EVAL_SET = os.path.join(HERE, "eval_set.jsonl")

TAU = 0.185          # 来自 calibration_heading.json

# ----------------------------------------------------------------------
# 参考实现：qa 组可直接复制这一段到编排层
# ----------------------------------------------------------------------
# 无先行词的指示代词。命中它只是「可疑」，还要看能否锚定章节。
PRON_RE = re.compile(r"这个|那个|这些|那些|它|它们|这一块|那一块|上面|上述|刚才|前面说的")

# 泛词：出现在标题里但不指向具体章节，锚定时必须排除。
# 不排掉它们，"操作系统是怎么管理这些东西的"会因为"管理"命中
# 「内存管理的基本功能」而被误判成可锚定。
GENERIC = {"操作", "系统", "管理", "功能", "方式", "结构", "关系", "区别",
           "概念", "基本", "主要", "常见", "什么", "哪些", "怎么", "如何",
           "存储", "分配", "控制", "状态", "文件"}


def anchor_terms(titles: List[str]) -> List[str]:
    """从全部章节标题里抽「可锚定词」：去编号、去英文，切 2-gram，滤掉泛词。"""
    out = set()
    for t in titles or []:
        t = re.sub(r"^\d+(\.\d+)*\s*", "", t or "")      # 去 "1.4 " 编号
        t = re.sub(r"[A-Za-z]+", "", t)                   # 去 PCB / TLB 等缩写
        t = re.sub(r"[\s（）()、，,。.【】]+", "", t)
        out.update(t[i:i + 2] for i in range(len(t) - 1))
    return [g for g in out if g not in GENERIC]


def need_clarify(query: str, top1_score: float,
                 titles: List[str], tau: float = TAU) -> bool:
    """是否该走「澄清」而不是「作答 / 拒答」。

    三条同时成立才澄清：
        1. 分数过了闸（否则是拒答，不是澄清）
        2. 问句含无先行词的指示代词
        3. 问句锚不到任何章节标题（定位不到具体章节）

    参数：
        titles —— 索引里的全部章节标题。本地调用取 `store.call("stats", {})["titles"]`，
                  远程同理（走 RPC 拿 stats）。
    """
    if top1_score < tau:
        return False                                # 不过闸 → 拒答路径
    if not PRON_RE.search(query):
        return False                                # 无指示代词 → 正常作答
    terms = anchor_terms(titles)
    return not any(w in query for w in terms)       # 锚不到 → 澄清


# ----------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["heading", "fixed"], default="heading")
    ap.add_argument("--tau", type=float, default=TAU)
    args = ap.parse_args()

    rp = os.path.join(HERE, f"results_{args.split}.jsonl")
    if not os.path.exists(rp):
        print(f"找不到 {rp}")
        print(f"请先执行：python evals/vector_db/run_eval.py --split {args.split}")
        return 1

    with open(rp, encoding="utf-8") as f:
        rows = [json.loads(l) for l in f if l.strip()]
    with open(EVAL_SET, encoding="utf-8") as f:
        expect = {json.loads(l)["id"]: json.loads(l) for l in f if l.strip()}

    # 章节标题从索引本体取，保证与线上一致（不在这里另抄一份标题表）
    from evals.vector_db.run_eval import load_corpus, split_via_parser
    chunks = split_via_parser(load_corpus(), args.split)
    titles: List[str] = []
    for c in chunks:
        t = (c.get("title") or "").strip()
        if t and t not in titles:
            titles.append(t)
    terms = anchor_terms(titles)

    print("=" * 72)
    print(f"澄清判据校验 · split={args.split} · τ={args.tau:.3f} · 共 {len(rows)} 题")
    print("=" * 72)

    if not titles:
        print("\n[跳过] 该分块模式下没有任何章节标题（fixed 模式不产出 title 字段），")
        print("       而澄清判据依赖「标题锚定」，故在此模式下不适用。")
        print("       请在 heading 模式下运行本校验：")
        print("         python evals/vector_db/run_eval.py --split heading")
        print("         python evals/vector_db/check_clarify_rule.py --split heading")
        return 0

    print(f"可锚定词 {len(terms)} 个（从 {len(titles)} 个章节标题抽取，已滤掉泛词）")

    fired, wrong = [], []
    for r in rows:
        qid = r["id"]
        got = need_clarify(r["question"], r["top1_score"], titles, args.tau)
        want = expect[qid].get("expected") == "clarify_ok"
        if got:
            fired.append((qid, r["question"], want, r["top1_score"], r["top1_title"]))
        if got != want:
            wrong.append((qid, got, want, r["question"]))

    print("\n[1] 规则命中的题")
    if not fired:
        print("    无")
    for qid, q, want, sc, title in fired:
        mark = "✅ 符合口径" if want else "❌ 误伤"
        print(f"    {qid} top1={sc:.4f} [{title}]  {q}   {mark}")

    print("\n[2] 与 expected 的差异")
    if not wrong:
        print("    无 —— 规则与逐条口径完全一致")
    for qid, got, want, q in wrong:
        print(f"    {qid} 规则={'澄清' if got else '非澄清'} / "
              f"口径={'澄清' if want else '非澄清'}  {q}")

    passed = [r for r in rows if r["top1_score"] >= args.tau]
    print("\n[3] 影响面")
    print(f"    分数过闸（≥τ）的题：{len(passed)} 条 —— 只有这些题可能被规则改成「澄清」")
    print(f"    未过闸（<τ）的题：{len(rows) - len(passed)} 条 —— 一律走拒答，规则不参与")
    print(f"    实际被改成澄清：{len(fired)} 条")

    print("\n[4] 校验结论")
    if not wrong:
        print(f"    [PASS] 规则只命中 {len(fired)} 条，与 expected 口径一致，其余题零影响")
        print("           qa 组可把 need_clarify() 直接复制进编排层")
        return 0
    print("    [FAIL] 规则与口径不一致 —— 请改规则，不要改评测集口径")
    return 1


if __name__ == "__main__":
    raise SystemExit(main() or 0)
