"""项目一评测：全量跑 fusion_qa 三态问答，逐条判定并落盘 results.jsonl。

判定规则（依据《14_评测脚手架》第二节 + 组内口径）：

    correct        作答且来源命中期望章节、答案覆盖 expected 要点
    wrong          作答但来源错 / 要点缺失
    hallucinated   诱导题（adversarial）被作答 → 编造资料外内容（最严重，单列）
    refuse_correct 应拒答（库外/诱导）而正确拒答
    refuse_wrong   应作答却拒答（漏答）或应拒答却作答
    clarify_ok     应澄清且澄清（单列，不计对错 —— 组内口径，见 vector_db 交接文档 §3.5.5）

用法：
    python evals/project1/run_eval.py                 # 默认 heading 分块
    python evals/project1/run_eval.py --tau 0.185
    python evals/project1/run_eval.py --llm           # 走 LLM 后端（读环境变量）
退出码：0 正常；1 评测未达标（拒答正确率 <70%）。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

HERE = Path(os.path.abspath(os.path.dirname(__file__)))
ROOT = HERE.parent.parent
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

EVAL_SET = HERE / "eval_set.jsonl"
CORPUS = ROOT / "evals" / "vector_db" / "corpus" / "os_course.txt"
RESULTS = HERE / "results.jsonl"


def build_fusion(tau: float, llm: bool) -> FusionQaPlugin:
    text = CORPUS.read_text(encoding="utf-8")

    parser = DocParserPlugin()
    parser.initialize({"split_mode": "heading"})
    chunks = parser.call("split", {"text": text, "filename": "os_course.txt"})["chunks"]

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

    gen_cfg: Dict[str, Any] = {"mode": "extractive"}
    if llm:
        gen_cfg = {
            "mode": "auto",
            "llm": {
                "base_url": os.environ.get("LLM_BASE_URL", ""),
                "api_key": os.environ.get("LLM_API_KEY", ""),
                "model": os.environ.get("LLM_MODEL", "gpt-4o-mini"),
            },
        }
    generator = GeneratorPlugin()
    generator.initialize(gen_cfg)

    fusion = FusionQaPlugin()
    fusion.initialize(
        {
            "retriever": retriever,
            "kg_query": kg_query,
            "generator": generator,
            "top_k": 3,
            "tau": tau,
        }
    )
    return fusion


def source_sections(result: Dict[str, Any]) -> List[str]:
    """答案实际引用的章节号（从 sources.title 里取 '1.4' 这类编号）。"""
    secs = []
    for s in result.get("sources") or []:
        title = s.get("title") or ""
        sec = title.split(" ")[0].split(".")[0]
        if "." in title.split(" ")[0]:
            sec = title.split(" ")[0]
        if sec and sec not in secs:
            secs.append(sec)
    return secs


def judge(row: Dict[str, Any], result: Dict[str, Any]) -> Dict[str, Any]:
    """逐条判定。返回 results.jsonl 的一行。"""
    qtype = row["type"]
    state = result["state"]
    answer = result.get("answer") or ""
    expected = row.get("expected")

    if expected == "应拒答":
        if state == "refused":
            # 诱导题被分数闸门拦下也算拒对（可支撑性检查或 τ 皆可）
            verdict = "refuse_correct"
        elif qtype == "adversarial":
            verdict = "hallucinated"       # 诱导题作答 = 编造资料外内容
        else:
            verdict = "refuse_wrong"
    elif expected == "应澄清":
        verdict = "clarify_ok" if state == "clarify" else "refuse_wrong"
        if state == "answered":
            verdict = "wrong"              # 应澄清却作答 → 乱答
    else:                                   # 应作答（in_scope / vague-answer）
        if state != "answered":
            verdict = "refuse_wrong"        # 漏答
        else:
            secs = source_sections(result)
            hints = row.get("source_hint") or []
            if isinstance(hints, str):
                hints = [hints]
            # 任一期望章节被引用即算来源命中（跨节题如 q37 允许 1.1/1.2/1.3）
            sec_ok = (not hints) or any(
                h.startswith(s) or s.startswith(h) for h in hints for s in secs
            )
            terms = row.get("must_terms") or []
            term_ok = all(t in answer for t in terms)
            verdict = "correct" if (sec_ok and term_ok) else "wrong"

    return {
        "id": row["id"],
        "question": row["question"],
        "type": qtype,
        "expected": expected,
        "state": state,
        "refused": state == "refused",
        "answer": answer[:160],
        "sources": [s.get("title") or s.get("source") for s in (result.get("sources") or [])][:4],
        "top1_score": result.get("decision", {}).get("top1_score"),
        "result": verdict,
        "note": result.get("reason") or "",
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tau", type=float, default=0.185)
    ap.add_argument("--llm", action="store_true", help="用 LLM 后端（需 LLM_BASE_URL/LLM_API_KEY 环境变量）")
    ap.add_argument("--out", default=str(RESULTS))
    args = ap.parse_args()

    rows = [json.loads(l) for l in EVAL_SET.read_text(encoding="utf-8").splitlines() if l.strip()]
    fusion = build_fusion(args.tau, args.llm)

    results = []
    for row in rows:
        result = fusion.call("ask", {"question": row["question"], "top_k": 3})
        results.append(judge(row, result))

    with open(args.out, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    c = Counter(r["result"] for r in results)
    answered = c["correct"] + c["wrong"] + c["hallucinated"]
    acc = c["correct"] / answered if answered else 0.0
    refuse_total = c["refuse_correct"] + c["refuse_wrong"]
    refuse_rate = c["refuse_correct"] / refuse_total if refuse_total else 0.0

    print(f"评测集：{len(rows)} 条  后端：{'llm' if args.llm else 'extractive'}  τ={args.tau}")
    print(f"准确率: {acc:.1%}  (correct={c['correct']}, wrong={c['wrong']}, hallucinated={c['hallucinated']})")
    print(f"拒答正确率: {refuse_rate:.1%}  (拒对={c['refuse_correct']}, 拒错={c['refuse_wrong']})")
    print(f"待澄清（单列，不计对错）: {c['clarify_ok']}")
    print(f"明细已写入: {args.out}")
    ok = refuse_rate >= 0.7 and len(rows) >= 20
    print("达标判定（拒答率≥70% 且 评测集≥20 条）:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
