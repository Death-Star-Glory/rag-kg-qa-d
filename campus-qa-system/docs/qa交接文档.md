# qa 组交接文档（知识问答模块）

**模块**：`plugins/qa/`（retriever / fusion_qa / generator / decision）+ `evals/project1/`
**交付日期**：2026-10-12
**对应课设**：项目一 · 课程知识问答智能体（RAG），过关硬条件全部满足

---

## 1. 本次交付了什么

| 交付物 | 位置 | 状态 |
|---|---|---|
| 三态判定（拒答 / 澄清 / 可支撑性） | `plugins/qa/decision.py` | ✅ 纯函数，零依赖 |
| 答案生成插件（LLM 优先 + 抽取式兜底） | `plugins/qa/generator.py` | ✅ |
| fusion_qa 三态编排改造（v0.1.0 → v0.2.0） | `plugins/qa/fusion.py` | ✅ 接口向后兼容 |
| qa 模块冒烟（8 例覆盖三态） | `smoke_qa.py` | ✅ 零第三方依赖 |
| 评测体系（40 题，教师格式） | `evals/project1/` | ✅ 拒答率 100% |
| gateway / topology 接线 | `gateway/app.py`、`config/topology.yaml` | ✅ 全部冒烟回归通过 |

## 2. 实测指标（extractive 后端 / heading 分块 / τ=0.185）

| 指标 | 值 | 过关线 | 判定 |
|---|---|---|---|
| 拒答正确率 | **100%**（12/12） | ≥70%（硬性） | ✅ |
| 库外题拒答率 | 8/8 = 100% | Rubric 优档 100% | ✅ |
| 诱导题拒答率 | 4/4 = 100% | 改造前 0% | ✅ |
| 准确率 | **92.6%**（25/27） | ≥70% 为良 | ✅（接近优） |
| 幻觉条数 | 0 | ≥3 条须逐条归因 | ✅ |
| 待澄清 | 1（q38，口径内单列） | — | ✅ |
| 评测集 | 40 条（24/8/4/4 = 60/20/10/10） | ≥20 条（硬性） | ✅ |

复现：

```bash
cd campus-qa-system
python smoke_qa.py
python evals/project1/run_eval.py
python evals/project1/eval_stats.py
```

## 3. 设计依据（答辩可讲的三件事）

### 3.1 为什么拒答必须前置（阈值是硬约束）

评分表「拒答与防幻觉」20 分 + 任务书硬性条件。概念卡 8：Prompt 里写
「不要编造」是软约束，必须用**阈值 + 过滤 + 评测**三件套。
所以判定在 `decision.decide()` 里、在调用 generator 之前完成，
LLM 不参与拒答决策。

τ=0.185 来自 vector_db 标定（库内最低 0.208 / 无关最高 0.161，分隔带
+0.047，漏答 0）。**换 embedding 或改打分权重必须重标定**
（`python evals/vector_db/calibrate.py --split heading`）。

### 3.2 三态而不是两态（模糊题）

模糊题一刀切拒答会白丢 3 条准确率，一刀切作答会乱答 q38。
判据（vector_db 交接文档 §3.5，参考实现整段移植）：

```
过闸 + 无先行词指示代词 + 锚不到任何章节标题 → 澄清
```

锚定用 `vector_store.stats()["titles"]` 的**全量**标题（不是 top-k 的），
避免把检索失败伪装成口径问题。

### 3.3 诱导题为什么 τ 拦不住、怎么拦（可支撑性检查）

诱导题与语料字面高度重叠（最高 0.452 > 库内最低 0.208），分数不可分
（vector_db 交接文档 §3.3）。qa 组补的闸：**问句在问细节，而检索证据里
没有对应内容 → 拒答**。三条独立规则（`decision.lacks_support`）：

| 问句特征 | 证据检查 | 拦下的评测题 |
|---|---|---|
| 问数值（多少/几毫秒/KB） | 证据无数字（章节编号已清洗） | q34、q36 |
| 问实现（源码/内核/结构体/代码） | 证据无对应词（概念描述不算） | q33 |
| 问步骤（具体/计算/详细步骤） | 证据无步骤性内容 | q35 |

规则刻意收窄：不含「哪些/几种」（正常列举题），避免误伤。

### 3.4 生成：为什么留双后端

- **extractive（默认）**：零依赖、可复现、零编造空间（以「节」为单位
  整段引用，不跨块拼接、不改写）。评测与冒烟的口径。
- **LLM（可选）**：OpenAI 兼容接口，Prompt 四段结构（概念卡 3：系统指令
  + 编号资料 + 问题 + 输出格式），`temperature=0.1`，答案末尾强制标来源。
  课设演示可以接，接了之后重跑 `run_eval.py --llm` 看指标变化。

## 4. 超长节的枚举丢失问题（A/B 记录）

**问题**：heading 分块下超长节（如 2.5）会被再切成多块且边界切在句中
（“淘|汰一页”）；按句挑选答案会丢枚举要点，q18「页面置换算法有哪些」
最佳置换/先进先出不全 → wrong。

**改法**：生成器从「按句挑」改为「同节合并整段引用」（`generator.py`
`_extractive_generate`），第二节按分数距离决定整节并入或补单句（≥6 字，
过滤边界碎片）。

**A/B（同一评测集 40 题）**：

| 版本 | 准确率 | wrong |
|---|---|---|
| 按句挑（改前） | 66.7% | 9 |
| 同节合并（改后） | **92.6%** | 2 |

## 5. 接口变化（组外需知悉）

| 文件 | 改动 | 兼容性 |
|---|---|---|
| `plugins/qa/fusion.py` | ask 返回新增 `state / reason / decision / generator_mode` 字段；`sources` 改为按（文件,章节,chunk）去重 | 旧字段全部保留；`gateway/app.py` 未改路由 |
| `plugins/qa/__init__.py` | 导出 `GeneratorPlugin`、`decision` | 新增 |
| `config/topology.yaml` | qa_node 新增 generator 插件；fusion_qa 加 tau | 未动其它节点 |
| `gateway/app.py` | build_demo_registry 注册并注入 generator | `/qa/ask` 响应结构只增不改 |

**多进程部署**：qa_node 依赖 vector_db_node(8102) 先启动（既有约定，
见总交接文档 §11.7）；fusion_qa 新增 `retriever_url / kg_query_url`
配置项支持跨节点 RPC，默认单进程注入。

## 6. 失败案例分析（≥3 例，任务书要求）

1. **q01「什么是进程？」→ wrong（检索）**
   期望 1.1，实际 top-3 = 1.5/1.4/1.3。字面向量对「什么是X」类定义题的
   召回上限：1.1 里「进程」出现密度反而低于 1.5（1.5 同时含进程+线程+资源
   分配等重叠词）。归因：**检索层**（vector_db 字面 embedding），非编排层。
   改进方向：换语义 embedding（需重标定 τ）或查询改写，属 vector_db 组范畴。

2. **q40「内存那一块。」→ wrong（检索）**
   期望第 2 章，top-3 全错（1.4/3.3/3.2）。残缺问句字面信号只有「内存」，
   与 1.4 的「内存指针」等干扰项重叠。归因：**检索层**。注意它**没有**
   被误判成澄清（「内存」锚定到 2.1 标题），口径问题与检索失败未混淆。

3. **q38「操作系统是怎么管理这些东西的？」→ clarify（口径边界）**
   正确行为（组内已定口径）。曾有人提议抬 τ 到 0.25+ 拦它，实测会误伤
   q39(0.208)/q17(0.243)，净收益为负 —— 这就是判据必须字面化的实证。

4. **（历史）q18 枚举丢失 → 已修复**，见第 4 节 A/B 记录。

## 7. 如果再做一次会改什么

1. 换语义 embedding（如 bge-small）后重标定 τ，预期解决 q01/q40 两条
   检索失败；字面向量是当前准确率的主要瓶颈（Hit@3 92.6% 上限）。
2. 澄清话术可以带上「问题改写建议」而非只列章节标题。
3. LLM 后端的答案要点覆盖率判定可引入 LLM-as-judge 自动化（目前 must_terms
   子串判定偏严）。

## 8. 接手路线（新组员）

```bash
cd campus-qa-system
pip install -r requirements.txt        # 只有 fastapi/uvicorn/pyyaml
python smoke_qa.py                     # 8 例冒烟，先跑这个
python evals/project1/run_eval.py      # 复现 92.6% / 100% / 0 幻觉
```

然后按顺序读：本文 → `plugins/qa/decision.py`（判定）→
`plugins/qa/generator.py`（生成）→ `evals/project1/README.md`（评测口径）。
检索层问题先读 `docs/vector_db交接文档.md`，图谱问题先读 kg 组代码。
