# -*- coding: utf-8 -*-
"""分块完整性检查：验证分块后原文是否被完整保留（无丢字）。

为什么需要它：
    分块是检索的上游。分块丢字 → 检索永远召不回那段内容 → 端到端答错，
    而错因看起来像"LLM 不行"。所以分块完整性必须在检索评测之前先验证。

判定方法（v2）：
    把原文与「所有分块拼接结果」都去掉空白后，用 difflib.SequenceMatcher
    求原文中**未被覆盖的片段**。为空即无丢字。

    为什么不用「滑动窗口是否出现」：窗口法会把"重叠量不足"误报成"丢字"。
    块间重叠按原始字符计（含空白），去空白后有效重叠必然小于配置值，
    于是长度=overlap 的窗口必然跨不过边界 —— 但内容其实没丢。
    差分法直接回答"有没有内容缺失"，不受重叠量影响。

用法：
    python evals/vector_db/check_split.py
    python evals/vector_db/check_split.py --mode heading
"""

from __future__ import annotations

import argparse
import difflib
import os
import re
import sys
from typing import Any, Dict, List, Tuple

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

DEFAULT_CORPUS = os.path.join(HERE, "corpus", "os_course.txt")


def norm(s: str) -> str:
    """去空白 + 去【】标题括号。

    括号是格式标记不是内容，且 heading 模式会把标题拆进 title 字段
    （不带括号），不剔掉会被误报成缺失。
    """
    return re.sub(r"\s+", "", s).replace("【", "").replace("】", "")


def load_body(path: str) -> str:
    lines = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.lstrip().startswith("#"):
                continue
            if line.strip():
                lines.append(line.strip())
    return "\n".join(lines)


def missing_segments(a: str, b: str) -> List[Tuple[int, str]]:
    """返回原文 a 中未被 b 覆盖的连续片段 [(起始下标, 片段)]。

    用 SequenceMatcher 求 a、b 的匹配块，未匹配部分即缺失内容。
    注意 b 中允许重复（块间 overlap 会带来重复），不影响判定。
    """
    sm = difflib.SequenceMatcher(None, a, b, autojunk=False)
    matched = [False] * len(a)
    for blk in sm.get_matching_blocks():
        for i in range(blk.a, blk.a + blk.size):
            matched[i] = True
    out: List[Tuple[int, str]] = []
    i = 0
    while i < len(a):
        if not matched[i]:
            j = i
            while j < len(a) and not matched[j]:
                j += 1
            out.append((i, a[i:j]))
            i = j
        else:
            i += 1
    return out


def overlap_between(x: str, y: str) -> int:
    """相邻块的最长后缀-前缀重叠（原始字符）。"""
    m = 0
    for L in range(1, min(len(x), len(y)) + 1):
        if x[-L:] == y[:L]:
            m = L
    return m


def run_split(body: str, size: int, overlap: int, mode: str) -> List[Dict[str, Any]]:
    from plugins.file_mgmt import DocParserPlugin

    p = DocParserPlugin()
    p.initialize({
        "chunk_size": size,
        "chunk_overlap": overlap,
        "split_mode": mode,
        "chunk_dir": os.path.join(HERE, "_chunks_tmp"),
    })
    return p.call("split", {"text": body, "filename": "t.txt", "persist": False})["chunks"]


def check(body: str, size: int, overlap: int, mode: str) -> Dict[str, Any]:
    chunks = run_split(body, size, overlap, mode)
    a = norm(body)
    # heading 模式把【标题】拆进了 title 字段，比对时须还原回去，
    # 否则会把标题误报成"缺失内容"。
    if mode == "heading":
        b = norm("".join((c.get("title") or "") + c["content"] for c in chunks))
    else:
        b = norm("".join(c["content"] for c in chunks))
    miss = missing_segments(a, b)
    raw = [c["content"] for c in chunks]
    ovs = [overlap_between(raw[k], raw[k + 1]) for k in range(len(raw) - 1)]
    return {
        "chunks": len(chunks),
        "total": len(a),
        "missing": miss,
        # heading 按节切，节间本就不重叠，0 是预期值
        "min_overlap": (min(ovs) if ovs else 0) if mode == "fixed" else -1,
        "max_len": max((len(x) for x in raw), default=0),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default=DEFAULT_CORPUS)
    ap.add_argument("--mode", default="fixed", choices=["fixed", "heading"])
    ap.add_argument("--params", default="160/24,80/16,60/12,300/50,120/20",
                    help="待测参数，格式 size/overlap，逗号分隔")
    args = ap.parse_args()

    body = load_body(args.corpus)
    print("=" * 70)
    print(f"分块完整性检查 · {os.path.basename(args.corpus)} · mode={args.mode}")
    print("=" * 70)
    print(f"{'size':>6}{'overlap':>9}{'块数':>7}{'原文字数':>10}{'缺失片段':>10}{'最小块间重叠':>14}  结论")

    bad = []
    for spec in args.params.split(","):
        size, ov = (int(x) for x in spec.split("/"))
        r = check(body, size, ov, args.mode)
        verdict = "OK" if not r["missing"] else f"丢字 {len(r['missing'])} 处"
        ov_txt = "n/a（按节切）" if r["min_overlap"] < 0 else str(r["min_overlap"])
        print(f"{size:>6}{ov:>9}{r['chunks']:>7}{r['total']:>10}"
              f"{len(r['missing']):>10}{ov_txt:>14}  {verdict}")
        if r["missing"]:
            bad.append((size, ov, r))

    for size, ov, r in bad:
        print(f"\n[定位] size={size} overlap={ov}")
        for off, seg in r["missing"][:5]:
            print(f"    offset={off:>5}  缺失 {len(seg)} 字: {seg[:60]}")

    print("\n" + "-" * 70)
    if bad:
        print(f"发现 {len(bad)} 组参数存在内容缺失，需检查 doc_parser._split_raw。")
    else:
        print("全部参数通过：分块未丢失任何内容。")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main() or 0)

