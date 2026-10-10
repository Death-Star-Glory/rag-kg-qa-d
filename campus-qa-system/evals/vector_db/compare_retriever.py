# -*- coding: utf-8 -*-
"""薄封装一致性校验：plugins.qa.retriever 与 plugins.vector_db.vector_store。

背景
----
`plugins/qa/retriever.py` 原先自实现了一套关键词打分，与 vector_db 职责重叠。
经同条件对比后组内决定采用**方案 C 薄封装** —— retriever 对外接口不变，
内部转发给 vector_store。本脚本用于验证改造生效、且未被改坏。

预期结果
--------
两者指标**完全一致**（走的是同一套底层逻辑）。
若出现不一致，说明薄封装失效或有人恢复了旧实现。

改造前的对比数据（作为方案选择的依据，保留备查）
------------------------------------------------
口径：旧语料 1 章 8 块 / 30 题。**与当前 3 章 29 块 / 40 题不可直接比较。**

    | 指标     | 原实现 | vector_store |
    |---------|--------|--------------|
    | Hit@1   | 95.0%  | 95.0%        |
    | Hit@3   | 100.0% | 95.0%        |
    | MRR     | 0.967  | 0.950        |
    | 分数上界  | 无（实测最大 1.0722） | 1.0（归一化）|
    | 分隔带   | +0.0666 | +0.0304     |

当时结论：原实现检索质量不差甚至略优，但分数无上界、无法作拒答阈值。
详见 `docs/vector_db交接文档.md` 第 7 节。

用法
----
    python evals/vector_db/compare_retriever.py
"""

from __future__ import annotations

import json
import os
import sys
from typing import Any, Dict, List

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from evals.vector_db.run_eval import (  # noqa: E402
    EVAL_SET,
    load_corpus,
    match,
    split_via_parser,
)

TOP_K = 3


# ----------------------------------------------------------------------
# 两个实现各自的装配与检索
# ----------------------------------------------------------------------
def build_retriever(chunks: List[Dict[str, Any]]):
    from plugins.qa.retriever import RagRetrieverPlugin

    p = RagRetrieverPlugin()
    p.initialize({})
    p.call("index_chunks", {"chunks": chunks})
    return p


def build_vector_store(chunks: List[Dict[str, Any]]):
    from plugins.vector_db import EmbedderPlugin, VectorStorePlugin

    emb = EmbedderPlugin()
    emb.initialize({"dim": 4096})
    store = VectorStorePlugin()
    store.initialize({"embedder": emb, "dim": 4096})
    store.call("index", {"chunks": chunks})
    return store


def retrieve_retriever(p, query: str) -> Dict[str, Any]:
    return p.call("retrieve", {"query": query, "top_k": TOP_K})


def retrieve_store(p, query: str) -> Dict[str, Any]:
    return p.call("search", {"query": query, "top_k": TOP_K})


# ----------------------------------------------------------------------
def run_one(name: str, items, retrieve_fn) -> Dict[str, Any]:
    rows = []
    for it in items:
        r = retrieve_fn(it["question"])
        hits = r.get("hits") or []
        top = hits[0] if hits else None
        rank = 0
        for pos, h in enumerate(hits, 1):
            if match(h, it):
                rank = pos
                break
        rows.append({
            "id": it["id"],
            "type": it["type"],
            "in_library": it["in_library"],
            "score": top["score"] if top else None,
            "rank": rank,
            "hit1": rank == 1,
            "hitk": rank >= 1,
            "empty": not hits,
        })

    ins = [r for r in rows if r["in_library"]]
    outs = [r for r in rows if not r["in_library"]]
    oos = [r for r in outs if r["type"] == "out_of_scope"]

    scored = [r["score"] for r in rows if r["score"] is not None]
    in_scores = [r["score"] for r in ins if r["score"] is not None]
    oos_scores = [r["score"] for r in oos if r["score"] is not None]

    return {
        "name": name,
        "rows": rows,
        "hit1": sum(1 for r in ins if r["hit1"]) / len(ins) if ins else 0.0,
        "hitk": sum(1 for r in ins if r["hitk"]) / len(ins) if ins else 0.0,
        "mrr": (sum(1 / r["rank"] for r in ins if r["rank"] > 0) / len(ins)) if ins else 0.0,
        "oob": sum(1 for s in scored if not (0.0 <= s <= 1.0)),
        "score_max": max(scored) if scored else 0.0,
        "in_min": min(in_scores) if in_scores else None,
        "oos_max": max(oos_scores) if oos_scores else None,
        "empty_out": sum(1 for r in outs if r["empty"]),
        "miss": [r["id"] for r in ins if not r["hitk"]],
    }


def main() -> int:
    text = load_corpus()
    chunks = split_via_parser(text, "heading")
    with open(EVAL_SET, encoding="utf-8") as f:
        items = [json.loads(l) for l in f if l.strip()]

    retriever = build_retriever(chunks)
    store = build_vector_store(chunks)

    a = run_one("rag_retriever", items, lambda q: retrieve_retriever(retriever, q))
    b = run_one("vector_store", items, lambda q: retrieve_store(store, q))

    print("=" * 74)
    print(f"检索实现对比 · 语料 {len(chunks)} 块（heading 分块）· 评测集 {len(items)} 题 · top_k={TOP_K}")
    print("=" * 74)
    print(f"{'指标':<26}{'rag_retriever (qa组)':>22}{'vector_store (本组)':>22}")
    print("-" * 74)

    def row(label, va, vb, fmt="{}"):
        print(f"{label:<26}{fmt.format(va):>22}{fmt.format(vb):>22}")

    row("Hit@1", a["hit1"], b["hit1"], "{:.1%}")
    row("Hit@3", a["hitk"], b["hitk"], "{:.1%}")
    row("MRR", a["mrr"], b["mrr"], "{:.3f}")
    print("-" * 74)
    row("分数最大值", a["score_max"], b["score_max"], "{:.4f}")
    row("分数越界条数(>1 或 <0)", a["oob"], b["oob"])
    row("能否用作拒答阈值", "否" if a["oob"] else "是", "否" if b["oob"] else "是")
    print("-" * 74)
    row("库内最低分", a["in_min"] if a["in_min"] is not None else "-",
        b["in_min"] if b["in_min"] is not None else "-", "{:.4f}")
    row("无关题最高分", a["oos_max"] if a["oos_max"] is not None else "-",
        b["oos_max"] if b["oos_max"] is not None else "-", "{:.4f}")
    gap_a = (a["in_min"] - a["oos_max"]) if (a["in_min"] is not None and a["oos_max"] is not None) else None
    gap_b = (b["in_min"] - b["oos_max"]) if (b["in_min"] is not None and b["oos_max"] is not None) else None
    row("分隔带", f"{gap_a:+.4f}" if gap_a is not None else "-",
        f"{gap_b:+.4f}" if gap_b is not None else "-")
    row("无关题返回空结果数", a["empty_out"], b["empty_out"])
    print("-" * 74)
    row("未召回题数", len(a["miss"]), len(b["miss"]))
    print(f"{'':<26}{str(a['miss']):>22}{str(b['miss']):>22}")

    print("\n" + "=" * 74)
    print("薄封装一致性校验")
    print("=" * 74)
    print(f"  Hit@1   {a['hit1']:>8.1%} vs {b['hit1']:<8.1%}")
    print(f"  Hit@3   {a['hitk']:>8.1%} vs {b['hitk']:<8.1%}")
    print(f"  MRR     {a['mrr']:>8.3f} vs {b['mrr']:<8.3f}")
    print(f"  最大分    {a['score_max']:>8.4f} vs {b['score_max']:<8.4f}")
    print(f"  越界条数  {a['oob']:>8} vs {b['oob']:<8}")

    same = (
        abs(a["hit1"] - b["hit1"]) < 1e-9
        and abs(a["hitk"] - b["hitk"]) < 1e-9
        and abs(a["mrr"] - b["mrr"]) < 1e-9
        and abs(a["score_max"] - b["score_max"]) < 1e-9
        and a["oob"] == b["oob"]
    )
    print()
    if same:
        print("  [PASS] 两者完全一致 —— 薄封装生效")
        print("         rag_retriever 已统一走 vector_store，qa 组无需改动调用点")
        print("         分数统一归一化到 [0,1]，编排层可写统一的拒答判定")
        print("         代价：改造前 rag_retriever（旧语料 1 章 8 块 / 30 题）Hit@3=100.0%，")
        print("               与当前 3 章 29 块 / 40 题口径不可直接比较；当前口径下两者一致")
        return 0
    print("  [FAIL] 两者不一致 —— 薄封装失效")
    print("         检查 plugins/qa/retriever.py 是否仍委托 self._store 转发")
    print("         若有人恢复了旧的关键词实现，分数尺度会重新变得无上界")
    return 1


if __name__ == "__main__":
    raise SystemExit(main() or 0)
