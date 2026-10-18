# 知识问答组（qa）：retriever + fusion_qa + generator + decision

三态问答（作答 / 澄清 / 拒答）+ 生成 + 溯源。不做检索算法（归 vector_db 组）、
不做图谱构建（归 kg 组）；本组只做**编排、判定、生成**。

## 插件

| 插件 | 文件 | 职责 |
|---|---|---|
| `rag_retriever` | `retriever.py` | 检索薄封装（v0.2.0，方案 C，归位 vector_db） |
| `fusion_qa` | `fusion.py` | 三态编排：检索 → 判定 →（作答路径）RAG+KG 证据 → 生成 |
| `generator` | `generator.py` | 答案生成：LLM（OpenAI 兼容）优先、抽取式兜底，全部带来源标注 |
| — | `decision.py` | 三态判定纯函数（拒答闸门 / 澄清判据 / 可支撑性检查） |

## fusion_qa 方法

```
ask {question, top_k} →
  { state: "answered"|"refused"|"clarify",
    answer, reason, mode, entities,
    rag_hits, kg{subgraphs,triples}, sources, confidence,
    decision{state, reason, top1_score, tau} }
```

判定顺序（全部在生成之前，硬约束，见 `decision.py`）：

```
拒答   top1 < τ(0.185)                      —— 分数闸门（库外题）
澄清   过闸 + 无先行词指示代词 + 锚不到章节    —— 字面判据（模糊题）
兜底   过闸 + 问细节而证据无细节              —— 可支撑性检查（诱导题）
作答   其余 → generator 生成 + 来源标注
```

## generator 配置

```yaml
generator:
  config:
    mode: extractive        # extractive | auto | llm
    # 接 LLM 时：
    # mode: auto
    # llm: { base_url: "...", api_key: "...", model: "..." }
    max_evidence: 3
```

LLM 走 OpenAI 兼容 `/chat/completions`（标准库 urllib，无新增依赖）；
Prompt 四段结构（系统指令 / 编号资料 / 问题 / 输出格式）见 `generator.py`。
`extractive` 为零依赖兜底：以「节」为单位整段引用，不跨块拼接、不改写。

## 快速验证

```bash
python smoke_qa.py                        # 8 例冒烟：三态/兜底/溯源
python evals/project1/run_eval.py         # 40 题全量评测
python evals/project1/eval_stats.py       # 统计摘要
```

详细设计与失败案例复盘见 `docs/qa交接文档.md`。
