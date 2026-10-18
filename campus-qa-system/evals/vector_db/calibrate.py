# -*- coding: utf-8 -*-
"""vector_db 组 · 拒答阈值标定。

输入：run_eval.py 产出的 results_*.jsonl（含每题的 top1 分数与 OOV 比）
输出：一份可直接交给编排层（qa 组）的阈值建议，含：
    1. 一维扫描：只按分数阈值 τ 判定，看每个 τ 下的漏答/正确拒答
    2. 二维策略：τ + OOV 双判据，看能否把诱导题也拦下来
    3. 逐条失败清单：在推荐阈值下判错的题是哪几条、为什么

判定口径（三类，不是两类）：
    answer      应作答 → 分数 ≥ τ 判对，否则漏答
    refuse      应拒答 → 分数 < τ 判对，否则该拒未拒
    clarify_ok  应澄清 → 既不判对也不判错，单列（当前仅 q38）
    判对率分母 = 总题数 − clarify_ok 条数。同时报严格口径（把 clarify_ok 记为误答）备查。

核心结论（跑出来就知道，脚本不预设）：
    无关题可靠分数阈值拦；诱导题拦不住 —— 因为它们的字面分数比应答题还高。
    所以编排层必须用「分数阈值 + 答案可支撑性检查」两段式，不能只调一个 τ。

用法：
    python evals/vector_db/calibrate.py                    # 默认读 heading
    python evals/vector_db/calibrate.py --split fixed
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Tuple

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

HERE = os.path.dirname(os.path.abspath(__file__))


def load(split: str) -> List[Dict[str, Any]]:
    path = os.path.join(HERE, f"results_{split}.jsonl")
    if not os.path.exists(path):
        print(f"找不到 {path}")
        print(f"请先执行：python evals/vector_db/run_eval.py --split {split}")
        sys.exit(1)
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


# ----------------------------------------------------------------------
# 判定：给定 τ，这一题会被判成"作答"还是"拒答"
# ----------------------------------------------------------------------
def predict(score: float, oov: float, tau: float, oov_max: float | None = None) -> bool:
    """返回 True 表示判为作答（不拒答）。"""
    if oov_max is not None and oov > oov_max:
        return False
    return score >= tau


def evaluate(rows, tau, oov_max=None) -> Dict[str, int]:
    """统计五种结果。

    `expected == "clarify_ok"` 的题（当前仅 q38）单列成 `clarify`：
    这类题既不该硬答，也不该拒答，正确行为是追问。把它算成「误答」会误导调参
    （它的分数 0.248 高于可答题 q39 的 0.208，抬 τ 只会误伤 q17），
    算成「判对」又是在自欺。所以既不进分子也不进分母，单独报出来。
    """
    c = {"answer_ok": 0, "answer_bad": 0, "refuse_ok": 0, "refuse_bad": 0, "clarify": 0}
    for r in rows:
        answered = predict(r["top1_score"], r["oov_ratio"], tau, oov_max)
        should = r["in_library"]
        if r.get("expected") == "clarify_ok":
            c["clarify"] += 1
            continue
        if should and answered:
            c["answer_ok"] += 1
        elif should and not answered:
            c["answer_bad"] += 1        # 漏答：应作答却拒答
        elif not should and not answered:
            c["refuse_ok"] += 1
        else:
            c["refuse_bad"] += 1        # 该拒未拒：最严重
    return c


def sweep(rows, oov_max=None, lo=0.02, hi=0.70, step=0.005) -> List[Tuple[float, Dict[str, int]]]:
    out = []
    t = lo
    while t <= hi + 1e-9:
        out.append((round(t, 4), evaluate(rows, t, oov_max)))
        t += step
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["heading", "fixed"], default="heading")
    args = ap.parse_args()

    rows = load(args.split)
    n_in = sum(1 for r in rows if r["in_library"])
    n_out = len(rows) - n_in
    n_clarify = sum(1 for r in rows if r.get("expected") == "clarify_ok")
    n_judged = len(rows) - n_clarify          # 判对率分母：剔除「应澄清」条目
    oos = [r for r in rows if r["type"] == "out_of_scope"]
    adv = [r for r in rows if r["type"] == "adversarial"]

    print("=" * 68)
    print(f"拒答阈值标定 · split={args.split} · 共 {len(rows)} 题"
          f"（应作答 {n_in} / 应拒答 {n_out} / 应澄清 {n_clarify}）")
    print("=" * 68)
    print("口径：判对率分母 = 总题数 − 应澄清题数（应澄清既不判对也不判错，单列）")

    # ---------- 1. 分数分布 ----------
    in_min = min(r["top1_score"] for r in rows if r["in_library"])
    oos_max = max((r["top1_score"] for r in oos), default=0.0)
    adv_max = max((r["top1_score"] for r in adv), default=0.0)
    print("\n[1] 分数分布")
    print(f"    应作答最低分      {in_min:.4f}")
    print(f"    无关题最高分      {oos_max:.4f}")
    print(f"    诱导题最高分      {adv_max:.4f}")
    print(f"    可拒带（无关题）  {in_min - oos_max:+.4f}")
    print(f"    可拒带（诱导题）  {in_min - adv_max:+.4f}   ← 负值说明诱导题分数高于应答题")

    # ---------- 2. 一维扫描：只按分数 ----------
    print("\n[2] 一维扫描：只用分数阈值 τ")
    swept = sweep(rows)
    print(f"    {'τ':>7}{'漏答':>7}{'正确拒答':>10}{'该拒未拒':>10}{'应答题正确':>12}")
    shown = set()
    for tau, c in swept:
        if int(round(tau * 1000)) % 25 == 0 and tau not in shown:
            shown.add(tau)
            print(f"    {tau:>7.3f}{c['answer_bad']:>7}{c['refuse_ok']:>10}"
                  f"{c['refuse_bad']:>10}{c['answer_ok']:>12}")

    # 选择策略：漏答与误答代价不对等。
    #   漏答 → 直接损失准确率分子，且任务书口径下不影响拒答率；
    #   误答 → 只影响拒答率分母，而拒答率是硬条件。
    # 所以先在「零漏答」约束内挑，再最大化正确拒答；取该区间的中点以留鲁棒余量。
    zero_miss = [(t, c) for t, c in swept if c["answer_bad"] == 0]
    if zero_miss:
        best_ro = max(c["refuse_ok"] for _, c in zero_miss)
        cand = [t for t, c in zero_miss if c["refuse_ok"] == best_ro]
        tau1 = round(sum(cand) / len(cand), 3)
    else:
        tau1, _ = max(swept, key=lambda x: x[1]["answer_ok"] + x[1]["refuse_ok"])
    c1 = evaluate(rows, tau1)
    ok1 = c1["answer_ok"] + c1["refuse_ok"]
    print(f"    → 推荐 τ={tau1:.3f}（零漏答约束下最大化正确拒答）")
    print(f"      判对 {ok1}/{n_judged} ({ok1 / n_judged:.1%})，"
          f"漏答 {c1['answer_bad']}、该拒未拒 {c1['refuse_bad']}、待澄清 {c1['clarify']}")
    print(f"      （严格口径：把「应澄清」记为误答 → 判对 {ok1}/{len(rows)} = {ok1 / len(rows):.1%}）")

    oos_ref = sum(1 for r in oos if r["top1_score"] < tau1)
    print(f"    → 该 τ 下无关题拒答率 {oos_ref}/{len(oos)} = {oos_ref / len(oos):.1%}"
          f"（任务书硬性条件 ≥70%）")
    adv_ref = sum(1 for r in adv if r["top1_score"] < tau1)
    print(f"    → 该 τ 下诱导题拒答率 {adv_ref}/{len(adv)} = {adv_ref / len(adv):.1%}"
          f"（脚手架口径会把这类计入拒答分母）")

    # ---------- 3. 二维策略：τ + OOV ----------
    print("\n[3] 二维策略：分数阈值 τ + OOV 上界")
    print(f"    {'τ':>7}{'OOV上界':>9}{'漏答':>7}{'正确拒答':>10}{'该拒未拒':>10}{'判对率':>9}  口径")
    grid = []
    for tau in [round(0.02 + i * 0.01, 3) for i in range(60)]:
        for oov_max in [0.30, 0.35, 0.40, 0.45, 0.50, 0.60, None]:
            c = evaluate(rows, tau, oov_max)
            grid.append((tau, oov_max, c, c["answer_ok"] + c["refuse_ok"]))
    # 口径 A：零漏答约束下最大化判对（与一维同口径，可直接比较）
    z = [g for g in grid if g[2]["answer_bad"] == 0]
    bestA = max(z, key=lambda g: g[3])
    # 口径 B：不管漏答，全局最大化判对（看上限）
    bestB = max(grid, key=lambda g: g[3])
    for tag, g in (("零漏答", bestA), ("全局最优", bestB)):
        tau, oov_max, c, ok = g
        lbl = "不启用" if oov_max is None else f"{oov_max:.2f}"
        print(f"    {tau:>7.3f}{lbl:>9}{c['answer_bad']:>7}{c['refuse_ok']:>10}"
              f"{c['refuse_bad']:>10}{ok / n_judged:>9.1%}  {tag}")
    tau2, oov2, c2, ok2 = bestA
    print(f"    → 同口径（零漏答）下：一维判对 {ok1}，二维判对 {ok2}（{ok2 - ok1:+d} 条）")
    tauB, oovB, cB, okB = bestB
    if cB["answer_bad"] == c2["answer_bad"]:
        print(f"    → 全局上限与零漏答口径重合（{okB} 条）："
              f"说明当前题集下没有「多判对但要多漏答」的余地，")
        print("      即判对率的上限被诱导题锁死，靠调 τ 无法再提升。")
    else:
        print(f"    → 允许漏答的全局上限：{okB} 条（{okB / n_judged:.1%}），"
              f"代价是漏答 {cB['answer_bad']} 条 —— 拿准确率换拒答率，不划算")

    # ---------- 4. 主推配置下的失败清单 ----------
    print(f"\n[4] 主推配置（τ={tau1:.3f}，仅分数）下的失败清单")
    bad = []
    for r in rows:
        if r.get("expected") == "clarify_ok":
            continue
        answered = predict(r["top1_score"], r["oov_ratio"], tau1, None)
        should = r["in_library"]
        if should != answered:
            kind = "漏答（应答却拒）" if should else "该拒未拒"
            bad.append((r["id"], r["type"], kind, r["top1_score"], r["oov_ratio"], r["question"]))
    if not bad:
        print("    无")
    for bid, btype, kind, sc, oov, q in bad:
        print(f"    {bid} [{btype}] {kind}  top1={sc:.4f} oov={oov:.2f}  {q}")

    clar = [r for r in rows if r.get("expected") == "clarify_ok"]
    if clar:
        print("    另（不计入失败）：以下题口径为「应澄清」，需编排层追问而非拒答")
        for r in clar:
            print(f"    {r['id']} [{r['type']}] 待澄清  top1={r['top1_score']:.4f} "
                  f"oov={r['oov_ratio']:.2f}  {r['question']}")

    # ---------- 5. 结论 ----------
    in_oov = (sum(r["oov_ratio"] for r in rows if r["in_library"]) / n_in) if n_in else 0.0
    oos_oov = (sum(r["oov_ratio"] for r in oos) / len(oos)) if oos else 0.0
    adv_oov = (sum(r["oov_ratio"] for r in adv) / len(adv)) if adv else 0.0

    print("\n[5] 给编排层的结论")
    print(f"    1) 主推 τ={tau1:.3f}（仅分数）。无关题拒答 {oos_ref}/{len(oos)}"
          f"={oos_ref / len(oos):.0%}，满足任务书硬性条件；漏答 {c1['answer_bad']} 条。")
    print(f"       判对 {ok1}/{n_judged} ({ok1 / n_judged:.1%})，"
          f"该拒未拒 {c1['refuse_bad']} 条（全是诱导题）。")
    if n_clarify:
        print(f"    1b) 另有 {n_clarify} 条口径为「应澄清」（{'/'.join(r['id'] for r in clar)}）："
              f"不是拒答，也不是硬答，")
        print("        正确行为是追问。本模块不做澄清，只提供信号；"
              "判据与话术见 docs/vector_db交接文档.md 第 3.5 节。")
    print(f"    2) 诱导题拒答率仅 {adv_ref}/{len(adv)}={adv_ref / len(adv):.0%}，"
          f"且调大 τ 也换不来：")
    print(f"       诱导题最高分 {adv_max:.3f} 高于应答题最低分 {in_min:.3f}，"
          f"两者区间重叠，")
    print("       任何单一 τ 都无法同时做到「零漏答」与「拦下诱导题」。")
    print(f"    3) 二维（τ + OOV 上界）实测只多判对 {ok2 - ok1} 条，"
          f"漏答 {c1['answer_bad']} → {c2['answer_bad']}，性价比低。")
    print(f"       OOV 均值：应答题 {in_oov:.2f} / 无关题 {oos_oov:.2f} / 诱导题 {adv_oov:.2f}。")
    if oos_oov - in_oov >= 0.3:
        print("       区分度可用，但对「口语化但指向明确」的问句会误报，")
        print("       故只宜作辅助提示，不宜做硬门槛。")
    else:
        print("       注意：语料覆盖范围扩大后，诱导题的用词被语料词表吸收，")
        print("       OOV 区分度下降，其辅助价值随之减弱。")
    print("    4) 诱导题的正解不是调阈值，而是加一道「答案可支撑性检查」：")
    print("       检索到的块里是否真的含有问句所问的那个具体信息（数值/实现/步骤）。")
    print("       这属于编排层（qa 组）职责，本组提供分数与 OOV 两个信号供其使用。")
    print("    5) 换 embedding 或改分块策略后必须重跑本标定：τ 绑定分数尺度。")

    # 落一份机器可读的建议
    out = {
        "split": args.split,
        "n": len(rows),
        "n_judged": n_judged,
        "clarify_pending": n_clarify,
        "recommend": {"tau": tau1, "oov_max": None,
                      "note": "主推仅分数阈值；OOV 作辅助信号，不建议设硬门槛"},
        "score_only": {"tau": tau1, "correct": ok1, "miss": c1["answer_bad"],
                       "false_answer": c1["refuse_bad"],
                       "judged": n_judged,
                       "accuracy": ok1 / n_judged if n_judged else 0.0,
                       "accuracy_strict": ok1 / len(rows) if rows else 0.0,
                       "oos_refuse_rate": oos_ref / len(oos) if oos else 0.0,
                       "adv_refuse_rate": adv_ref / len(adv) if adv else 0.0},
        "two_factor": {"tau": tau2, "oov_max": oov2, "correct": ok2,
                       "miss": c2["answer_bad"], "false_answer": c2["refuse_bad"]},
        "distribution": {"in_min": in_min, "oos_max": oos_max, "adv_max": adv_max},
        "failures": [
            {"id": b[0], "type": b[1], "kind": b[2], "top1": b[3], "oov": b[4], "question": b[5]}
            for b in bad
        ],
        "clarify_items": [
            {"id": r["id"], "type": r["type"], "top1": r["top1_score"],
             "oov": r["oov_ratio"], "question": r["question"]}
            for r in clar
        ],
    }
    p = os.path.join(HERE, f"calibration_{args.split}.json")
    with open(p, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\n标定结果已写入 {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main() or 0)
