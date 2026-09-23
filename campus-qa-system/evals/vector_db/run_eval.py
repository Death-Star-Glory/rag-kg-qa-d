# -*- coding: utf-8 -*-
"""vector_db 组 · 检索层评测。

为什么单独做检索层评测（而不是只做端到端）：
    端到端答错时，错因混了「检索没召回 / Prompt 没约束 / LLM 不听话」三段，无法定位。
    本组负责向量检索，故只测一件事：**该命中的块有没有进 top-k**，
    并输出分数分布，供编排层标定拒答阈值。

分块模式：
    heading  按【】标题切，与《13_样例语料与标准分块》的标准一致（推荐）
    fixed    走现有 doc_parser 定长 160 切分（对照组，用于暴露切分缺陷）
    两者跑同一套题，差异即分块策略带来的损失。

用法：
    python evals/vector_db/run_eval.py                # 默认 heading
    python evals/vector_db/run_eval.py --split both   # 两种都跑并对比
    python evals/vector_db/run_eval.py --top-k 3 --dim 4096

产出：
    evals/vector_db/results_heading.jsonl
    evals/vector_db/results_fixed.jsonl
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
ROOT = os.path.dirname(os.path.dirname(HERE))          # campus-qa-system/
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

CORPUS = os.path.join(HERE, "corpus", "os_course.txt")
EVAL_SET = os.path.join(HERE, "eval_set.jsonl")

HEADING_RE = re.compile(r"^【(?P<title>[^】]+)】\s*(?P<body>.*)$")


# ----------------------------------------------------------------------
# 语料与分块
# ----------------------------------------------------------------------
def load_corpus(path: str = CORPUS) -> str:
    """读语料，# 开头的行是注释，丢弃。"""
    lines = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.lstrip().startswith("#"):
                continue
            lines.append(line)
    return "".join(lines).strip()


def split_via_parser(text: str, mode: str, source: str = "os_course.txt") -> List[Dict[str, Any]]:
    """两种模式都走 doc_parser 本体，保证评测对象与线上一致。

    不在评测脚本里另写一份切分逻辑 —— 那样测的是评测脚本，不是插件。
    """
    from plugins.file_mgmt import DocParserPlugin

    parser = DocParserPlugin()
    parser.initialize({
        "chunk_size": 160,
        "chunk_overlap": 24,
        "split_mode": mode,
        "chunk_dir": os.path.join(HERE, "_chunks_tmp"),
    })
    res = parser.call("split", {"text": text, "filename": source, "persist": False})
    return res["chunks"]


def split_heading(text: str, source: str = "os_course.txt") -> List[Dict[str, Any]]:
    return split_via_parser(text, "heading", source)


def split_fixed(text: str, source: str = "os_course.txt") -> List[Dict[str, Any]]:
    return split_via_parser(text, "fixed", source)


# ----------------------------------------------------------------------
# 命中判定
# ----------------------------------------------------------------------
def match(chunk: Dict[str, Any], item: Dict[str, Any]) -> bool:
    """分块是否命中该题的期望章节。

    判定顺序（**不要随意放宽**）：

    1. 有 `expect_chunk` 且分块带 `title`（heading 模式）
       → **只按标题前缀判**。命中即 True，不命中即 False，**不再退到 must_terms**。

       为什么不能退：`must_terms` 判的是"词有没有出现在正文里"，太松。
       实测踩坑：q40「内存那一块。」的 top-1 是「1.4 进程控制块 PCB」，
       只因为该块正文里有"内存指针"三个字，就被判成命中了第 2 章 —— 纯属误报，
       会让 Hit@1 虚高。正文提一句某词 ≠ 这一节在讲该主题。

    2. 有 `expect_chunk` 但分块无 `title`（fixed 模式）
       → 无标题可用，只能退到 `must_terms`。此路径偏松，仅作对照组，
         结论只用于「heading vs fixed 的相对比较」，不用于绝对值。

    3. 无 `expect_chunk`（库外题）→ 不看命中，返回 False。
    """
    expect = item.get("expect_chunk") or []
    title = (chunk.get("title") or "").strip()

    if expect and title:
        return any(title.startswith(e) for e in expect)

    if expect and not title:
        terms = item.get("must_terms") or []
        content = chunk.get("content") or ""
        return bool(terms) and all(t in content for t in terms)

    return False


# ----------------------------------------------------------------------
# 主流程
# ----------------------------------------------------------------------
def run(split_mode: str, top_k: int, dim: int) -> Dict[str, Any]:
    from plugins.vector_db import EmbedderPlugin, VectorStorePlugin

    text = load_corpus()
    chunks = split_heading(text) if split_mode == "heading" else split_fixed(text)

    embedder = EmbedderPlugin()
    embedder.initialize({"dim": dim})
    store = VectorStorePlugin()
    store.initialize({"embedder": embedder, "dim": dim})
    idx = store.call("index", {"chunks": chunks})

    with open(EVAL_SET, encoding="utf-8") as f:
        items = [json.loads(l) for l in f if l.strip()]

    rows: List[Dict[str, Any]] = []
    for it in items:
        r = store.call("search", {"query": it["question"], "top_k": top_k})
        hits = r["hits"]
        top = hits[0] if hits else None
        ana = store.call("analyze", {"query": it["question"]})

        rank = 0
        for pos, h in enumerate(hits, 1):
            if match(h, it):
                rank = pos
                break

        rows.append({
            "id": it["id"],
            "question": it["question"],
            "type": it["type"],
            "in_library": it["in_library"],
            "expected": it.get("expected") or ("answer" if it["in_library"] else "refuse"),
            "expect_chunk": it.get("expect_chunk") or [],
            "split": split_mode,
            "top1_score": top["score"] if top else 0.0,
            "top1_cos": top["cos"] if top else 0.0,
            "top1_cov": top["cov"] if top else 0.0,
            "top1_title": (top.get("title") if top else "") or "",
            "top1_head": ((top.get("content") or "")[:60].replace("\n", " ") if top else ""),
            "oov_ratio": ana["oov_ratio"],
            "oov_chars": ana["oov_chars"],
            "rank": rank,
            "hit1": rank == 1,
            "hit_k": rank >= 1,
            "hits": [
                {
                    "rank": i,
                    "score": h["score"],
                    "cos": h["cos"],
                    "cov": h["cov"],
                    "title": h.get("title") or "",
                    "head": (h.get("content") or "")[:50].replace("\n", " "),
                }
                for i, h in enumerate(hits, 1)
            ],
        })

    out_path = os.path.join(HERE, f"results_{split_mode}.jsonl")
    with open(out_path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    summary = _summarize(rows, idx, split_mode, top_k, dim)
    summary["out_path"] = out_path
    return summary


def _pct(vals: List[float], q: float) -> float:
    """线性插值分位数。用 P10 而非 min，避免单条异常题拉低整条下界。"""
    if not vals:
        return 0.0
    xs = sorted(vals)
    if len(xs) == 1:
        return xs[0]
    pos = q * (len(xs) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (pos - lo)


def _summarize(rows, idx, split_mode, top_k, dim) -> Dict[str, Any]:
    """分层报告。

    关键：库外必须拆成 out_of_scope 与 adversarial 两档。
        out_of_scope —— 与语料零字面重叠，分数天然低，可靠阈值拦下；
        adversarial  —— 与语料字面高度重叠但语料无该内容，字面向量必然给高分。
    两者混算会得出「没有分隔带」的错误结论，掩盖真正可用的那半段。
    """
    ins = [r for r in rows if r["in_library"]]
    oos = [r for r in rows if r["type"] == "out_of_scope"]
    adv = [r for r in rows if r["type"] == "adversarial"]
    vg = [r for r in rows if r["type"] == "vague"]

    hit1 = sum(1 for r in ins if r["hit1"]) / len(ins) if ins else 0.0
    hitk = sum(1 for r in ins if r["hit_k"]) / len(ins) if ins else 0.0
    mrr = (sum(1 / r["rank"] for r in ins if r["rank"] > 0) / len(ins)) if ins else 0.0

    in_scores = [r["top1_score"] for r in ins]
    in_min = min(in_scores) if in_scores else 0.0
    in_p10 = _pct(in_scores, 0.10)
    oos_max = max((r["top1_score"] for r in oos), default=0.0)
    adv_max = max((r["top1_score"] for r in adv), default=0.0)
    vg_max = max((r["top1_score"] for r in vg), default=0.0)

    gap_oos = in_min - oos_max          # 真正的可拒带
    gap_adv = in_min - adv_max          # 负值即字面向量硬上限
    # 粗估阈值：取无关题最高分之上、分隔带 40% 处，偏向库内侧。
    # 注意这只是粗估（只用了 in_min / oos_max 两个点），
    # 正式阈值由 calibrate.py 按「零漏答约束下最大化正确拒答」扫描得出，两者可能差 0.005 左右。
    tau = oos_max + (gap_oos * 0.4 if gap_oos > 0 else 0.0)

    print("=" * 60)
    print(f"检索层评测 · split={split_mode}  dim={dim}  top_k={top_k}")
    print("=" * 60)
    print(f"语料分块数        : {idx['indexed']}  词表={idx['vocab_size']}")
    print(f"评测题数          : {len(rows)}  (库内 {len(ins)} / 库外 {len(oos) + len(adv)})")
    print("-" * 60)
    print(f"Hit@1             : {hit1:.1%}")
    print(f"Hit@{top_k}             : {hitk:.1%}")
    print(f"MRR               : {mrr:.3f}")
    print("-" * 60)
    print(f"库内 top1 最低分  : {in_min:.4f}   P10={in_p10:.4f}")
    print(f"无关题(out) 最高分: {oos_max:.4f}")
    print(f"  → 分隔带 1      : {gap_oos:+.4f}  {'可分，粗估 τ≈' + format(tau, '.3f') + '（正式值看 calibrate.py）' if gap_oos > 0 else '重叠'}")
    print(f"诱导题(adv) 最高分: {adv_max:.4f}")
    print(f"  → 分隔带 2      : {gap_adv:+.4f}  {'可分' if gap_adv > 0 else '重叠 —— 字面向量硬上限，纯分数阈值拦不住'}")
    print(f"模糊题(vague)最高分: {vg_max:.4f}")
    print("-" * 60)

    for t in ("in_scope", "out_of_scope", "adversarial", "vague"):
        sub = [r for r in rows if r["type"] == t]
        if not sub:
            continue
        avg = sum(r["top1_score"] for r in sub) / len(sub)
        mx = max(r["top1_score"] for r in sub)
        oov = sum(r["oov_ratio"] for r in sub) / len(sub)
        oov_mx = max(r["oov_ratio"] for r in sub)
        print(f"  {t:<14} n={len(sub):<3} 平均top1={avg:.3f} 最高top1={mx:.3f}"
              f" | OOV均值={oov:.2f} 最高={oov_mx:.2f}")

    miss = [r["id"] for r in ins if not r["hit_k"]]
    if miss:
        print("-" * 60)
        print(f"未召回（top{top_k} 内无期望章节）: {miss}")
        for r in ins:
            if not r["hit_k"]:
                print(f"  {r['id']} {r['question']}  → top1={r['top1_score']:.4f} [{r['top1_title']}]")

    # 「应澄清」单列：这类题不是该拒答，也不该硬答，正确行为是追问。
    # 不计入判对/判错的分母，但必须打印出来，否则口径会被误读。
    clarify = [r for r in rows if r["expected"] == "clarify_ok"]
    if clarify:
        print("-" * 60)
        print(f"应澄清（不计入判对/判错，口径见交接文档第 3.5 节）：{len(clarify)} 条")
        for r in clarify:
            print(f"  {r['id']} {r['question']}  → top1={r['top1_score']:.4f} [{r['top1_title']}]")

    print("-" * 60)
    print("评分表硬性条件自检：")
    print(f"  [{'PASS' if len(rows) >= 20 else 'FAIL'}] 评测集 ≥20 条（当前 {len(rows)}）")
    print(f"  [{'PASS' if hitk >= 0.8 else 'WARN'}] 库内 Hit@{top_k} ≥80%（当前 {hitk:.1%}）")
    print(f"  [{'PASS' if gap_oos > 0 else 'WARN'}] 无关题存在分数分隔带（当前 {gap_oos:+.4f}）")

    return {
        "split": split_mode, "dim": dim, "top_k": top_k,
        "indexed": idx["indexed"], "n": len(rows), "n_in": len(ins),
        "n_out": len(oos) + len(adv), "n_clarify": len(clarify),
        "hit1": hit1, "hitk": hitk, "mrr": mrr,
        "gap": gap_oos, "gap_adv": gap_adv, "tau": tau,
        "in_min": in_min, "in_p10": in_p10,
        "oos_max": oos_max, "adv_max": adv_max, "miss": miss,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["heading", "fixed", "both"], default="heading")
    ap.add_argument("--top-k", type=int, default=3)
    ap.add_argument("--dim", type=int, default=4096)
    args = ap.parse_args()

    modes = ["heading", "fixed"] if args.split == "both" else [args.split]
    summaries = [run(m, args.top_k, args.dim) for m in modes]

    if len(summaries) == 2:
        a, b = summaries
        print()
        print("=" * 60)
        print("分块策略对比（同题同参数）")
        print("=" * 60)
        print(f"{'指标':<18}{'heading':>12}{'fixed':>12}")
        for k, label in (("indexed", "分块数"), ("hit1", "Hit@1"), ("hitk", f"Hit@{args.top_k}"),
                         ("mrr", "MRR"), ("gap", "分隔带"), ("in_min", "库内最低分")):
            va, vb = a[k], b[k]
            fmt = (lambda x: f"{x:.1%}") if isinstance(va, float) and k in ("hit1", "hitk") else \
                  (lambda x: f"{x:.3f}") if isinstance(va, float) else (lambda x: str(x))
            print(f"{label:<18}{fmt(va):>12}{fmt(vb):>12}")
        print()
        print(f"结论：heading 相比 fixed，Hit@1 {a['hit1'] - b['hit1']:+.1%}，"
              f"MRR {a['mrr'] - b['mrr']:+.3f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main() or 0)
