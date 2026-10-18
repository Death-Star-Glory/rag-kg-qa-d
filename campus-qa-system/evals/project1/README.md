# evals/project1 —— 项目一评测（qa 组）

依据《14_评测脚手架》建立：评测集、逐条判定、统计摘要。

## 文件

| 文件 | 说明 |
|---|---|
| `eval_set.jsonl` | 评测集 40 条（教师格式：id / question / type / expected / source_hint） |
| `run_eval.py` | 全量跑 `fusion_qa` 三态问答，逐条判定，写 `results.jsonl` |
| `eval_stats.py` | 统计摘要（教师脚手架第四节模板 + 题型分布） |
| `results.jsonl` | 最近一轮逐条结果（`--llm` 跑 LLM 后端时另存 `results_llm.jsonl`） |

## 题型配比（硬性要求）

| type | 条数 | 占比 | 要求 |
|---|---|---|---|
| in_scope | 24 | 60% | ≥60% ✅ |
| out_of_scope | 8 | 20% | ≥20% ✅ |
| adversarial | 4 | 10% | ≥10% ✅ |
| vague | 4 | 10% | ≥10% ✅ |

题目源自 vector_db 组评测集（`evals/vector_db/eval_set.jsonl`，2026-09-22 标定），
按教师格式转换；两者共用同一语料 `evals/vector_db/corpus/os_course.txt`，
检索层指标（Hit@k / τ 标定）以 vector_db 的评测脚本为准。

## 判定规则

`correct / wrong / hallucinated / refuse_correct / refuse_wrong` 按脚手架第二节；
`clarify_ok` 为组内口径（vector_db 交接文档 §3.5.5：应澄清题单列，不计对错）。
诱导题（adversarial）被作答记 `hallucinated`（编造资料外内容，最严重）。

## 实测结果（extractive 后端 / heading 分块 / τ=0.185 / 2026-10-12）

| 指标 | 值 | 过关线 |
|---|---|---|
| 准确率 | **92.6%**（25/27） | ≥70% 为良 ✅ |
| 拒答正确率 | **100%**（12/12） | ≥70% 硬性条件 ✅ |
| 库外题（8 条）拒答 | 8/8 = 100% | Rubric 优档 ✅ |
| 诱导题（4 条）拒答 | 4/4 = 100% | 原 0%，可支撑性检查补齐 ✅ |
| 幻觉条数 | 0 | — |
| 待澄清 | 1（q38，口径内） | — |

## 跑法

```bash
cd campus-qa-system
python evals/project1/run_eval.py                # extractive 后端（默认，可复现）
python evals/project1/run_eval.py --llm          # LLM 后端（需 LLM_BASE_URL / LLM_API_KEY / LLM_MODEL）
python evals/project1/eval_stats.py              # 统计摘要
```

## 已知失败案例（复盘用，详见 docs/qa交接文档.md 第 6 节）

1. **q01「什么是进程？」** — 1.1 未进 top-3（字面向量对“什么是X”定义题召回上限）。
2. **q40「内存那一块。」** — 第 2 章未进 top-3（同上，vector_db 交接文档 §3.5.3 已记录）。
3. 历史问题：**超长节切分丢枚举要点** — 已通过生成器“同节合并引用”修复（A/B 见交接文档）。
