# vector_db 组 · 交接文档

> 更新：2026-09-22 · 负责人：（你的名字）· 模块：`plugins/vector_db/`

本文档写给**组内其他人**。目的是：你不需要问我，也能用上我的东西、知道边界在哪、知道哪里还没做完。

---

## 1. 我交付了什么

| 文件                                  | 作用                                   | 状态        |
| ----------------------------------- | ------------------------------------ | --------- |
| `plugins/vector_db/embedder.py`     | 文本 → 4096 维向量（字符 n-gram 哈希 + TF-IDF） | ✅         |
| `plugins/vector_db/vector_store.py` | 索引 / 检索 / 分数尺度 / 可答性信号               | ✅         |
| `plugins/vector_db/__init__.py`     | 导出两个插件类                              | ✅         |
| `plugins/vector_db/README.md`       | 模块自述（接口表 + 调用示例 + 关键数字）              | ✅         |
| `smoke_vector_db.py`                | 冒烟自检，装配→分块→建索引→检索（本地直连）              | ✅ 通过      |
| `smoke_rpc_vector_db.py`            | **接入基座验证**，起真实节点走 HTTP RPC           | ✅ 通过      |
| `evals/vector_db/`                  | 语料 + 40 题评测集 + 评测/标定/分块完整性/一致性/澄清判据校验脚本 | ✅         |
| `docs/vector_db设计文档.md`             | 面向答辩的技术文档                            | ✅         |
| `plugins/file_mgmt/doc_parser.py`   | **改动**：重叠修复 + 标题切分模式 + `title` 字段    | ⚠️ 见第 5 节 |
| `config/topology.yaml`              | **改动**：新增 `vector_db_node`(8102)     | ⚠️ 见第 5 节 |
| `plugins/qa/retriever.py`           | **改动**：按方案 C 改为薄封装，转发本模块             | ⚠️ 见第 5 节 |

**改动原则**：只加不改。`doc_parser` 默认行为保持原样，新能力靠配置项开启；  
`retriever` 对外接口签名与返回结构一字未动。

---

## 2. 接口契约（qa 组重点看）

### 2.1 建索引

```python
store.call("index", {"chunks": chunks})
# → {"indexed": 29, "dim": 4096, "vocab_size": 2158}
```

`chunks` 来自 `doc_parser.split()` 的输出。**若 chunk 带 `title` 字段，检索时会自动加权**  
（标题是分块的语义摘要，纳入向量后"什么是X"更容易命中对应章节）。

### 2.2 检索

```python
r = store.call("search", {"query": "进程有哪三种基本状态？", "top_k": 3})
```

返回（真实输出）：

```json
{
  "query": "进程有哪三种基本状态？",
  "top_k": 3,
  "indexed": 29,
  "hits": [
    {
      "chunk_id": "0940beca30e9_0001",
      "chunk_index": 1,
      "title": "1.2 进程的状态",
      "content": "进程在其生命周期内处于三种基本状态：就绪、运行、阻塞……",
      "char_count": 104,
      "score": 0.4259,      // 综合分，[0,1]，拒答判定用这个
      "cos": 0.2716,        // 余弦分项
      "cov": 0.889          // 关键词覆盖率分项
    }
  ],
  "score_scale": { "formula": "0.75 * cosine + 0.25 * keyword_coverage", "range": [0.0, 1.0] }
}
```

`hits` 已按 score 降序。`top_k` 只是截断，**打分是全量算的**，所以可以放心取 top-1 做阈值判定。

### 2.3 分数尺度

```python
store.call("score_scale", {})
```

**编排层做拒答判定前必须读这个。** 分数尺度由 embedding 实现决定，  
更换 embedding 或调整 `w_cos`/`w_wov` 权重后，**阈值必须重新标定**，否则全拒或全不拒。

### 2.4 可答性信号（给拒答用的第二判据）

```python
a = store.call("analyze", {"query": "时间片通常设置为多少毫秒？"})
# → {"oov_ratio": 0.1818, "oov_chars": ["毫", "秒"], "total_chars": 11, "indexed": 29}
```

`oov_ratio` = 问句实词中**不在语料词表**的单字占比。

实测分布（40 题）：

| 类别           | OOV 均值   | 最高   |
| ------------ | -------- | ---- |
| in_scope     | 0.02     | 0.12 |
| out_of_scope | **0.65** | 1.00 |
| adversarial  | 0.14     | 0.19 |
| vague        | 0.15     | 0.33 |

⚠️ **语料从 1 章扩到 3 章后，诱导题的 OOV 从 0.31 掉到 0.14** ——  
"毫秒""结构体"这类词被新语料吸收了。无关题仍可分（0.65 vs 0.02），  
但诱导题与应答题的差距被压缩，**这个信号对诱导题基本失效了**。

**已知局限**：对"口语化但指向明确"的问句会误报 ——  
`q39`「调度相关的东西。」是**库内可答**的模糊题（期望召回 1.6 节），  
但 OOV=0.33，几乎与诱导题同一水位。所以它**只能作辅助提示，不要设成硬门槛** ——  
实测作硬门槛收益为 +0。

---

## 3. 给 qa 组：拒答怎么接

### 3.1 直接用我的阈值

```python
TAU = 0.185   # 来自 evals/vector_db/calibration_heading.json

def should_refuse(store, query):
    r = store.call("search", {"query": query, "top_k": 3})
    top = r["hits"][0] if r["hits"] else None
    if top is None or top["score"] < TAU:
        return True, "检索结果与知识库内容相关性不足"
    return False, ""
```

判定必须在**调用 LLM 之前**完成，用硬约束，不要写进 Prompt 让 LLM 自己判断  
（评分表明确要求阈值前置）。

### 3.2 标定数据（`calibration_heading.json`）

| 指标     | 值                              |
| ------ | ------------------------------ |
| 推荐 τ   | **0.185**                      |
| 判对     | 35/39 = **89.7%**（待澄清 1 条单列） |
| 漏答     | 0                              |
| 无关题拒答率 | 8/8 = **100%**（任务书硬性条件 ≥70% ✅） |
| 诱导题拒答率 | 0/4 = 0%                       |
| 待澄清    | 1（q38）                        |

**判对率有两个口径，我都报出来**：

| 口径   | 算式                            | 结果           |
| ---- | ----------------------------- | ------------ |
| 主口径  | 分母 = 总题数 − 应澄清题数 = 39         | 35/39 = **89.7%** |
| 严格口径 | 把「应澄清」记为误答，分母 = 40            | 35/40 = **87.5%** |

**口径提醒**：标定用的「库内最低分 0.208」是 `in_library=True` 的 27 条  
（24 条 in_scope + 3 条模糊题）里的最小值，来自 `q39`「调度相关的东西。」。  
若只算严格 in_scope，最低分是 `q17`（分段和分页有什么区别）= 0.243。  
两个口径下分隔带分别为 **+0.047** 与 **+0.082**，τ=0.185 都成立且偏保守。

### 3.3 ⚠️ 必须知道的一条：诱导题拦不住

```
库内最低分   0.208（严格 in_scope 为 0.243）
无关题最高分 0.161  → 分隔带 +0.047  可分 ✅
诱导题最高分 0.452  → 分隔带 −0.243  不可分 ❌
```

**诱导题（如"PCB 在 Linux 内核源码中对应哪个结构体"）的分数比应答题还高。**  
原因是它与语料字面高度重叠（"进程控制块 PCB"），但语料没有它真正要的东西  
（具体内核实现）。这是**字面向量的表达上限，调 τ 解决不了**。

实测扫描：τ=0.26 能拦下 2 条诱导题（q35/q36），但会引入 1 条漏答（q17，0.243），  
净收益仅 +1；τ≥0.29 后漏答增长快于拦截，净收益归零或为负。**不值得为诱导题抬高阈值。**

**正解**是加一道「答案可支撑性检查」，属编排层职责。可参考的思路：

```python
# 问句在问"具体数值/实现/步骤"时，检索块里是否真的有这类内容？
DETAIL_RE = re.compile(r"多少|几毫秒|具体|源码|结构体|步骤|代码|怎么实现")
def lacks_support(query, top_chunk):
    if DETAIL_RE.search(query):
        # 语料若只给了名称/机理而没给数值/步骤，就判定为不可支撑
        return not re.search(r"\d|例如|如下|第一步", top_chunk["content"])
    return False
```

上面只是示意，具体规则你们定。我的职责是**提供分数和 OOV 两个信号**，  
判定逻辑归你们。

### 3.4 换了 embedding 怎么办

1. 重跑 `python evals/vector_db/run_eval.py --split heading`
2. 重跑 `python evals/vector_db/calibrate.py --split heading`
3. 读新的 `calibration_heading.json` 里的 `recommend.tau`

### 3.5 模糊题怎么接：三态，不是两态（**组内已定，照做即可**）

这是最容易接错的一块，单独讲清楚。

#### 3.5.1 先看现象：模糊题不能一刀切

评测集 `vague` 档 4 条，逐条判口径后是 **3 答 1 澄清**：

| 题   | 问句                  | 正确行为         | top-1  | top-1 章节     |
| --- | ------------------- | ------------ | ------ | ------------ |
| q37 | 讲讲进程。               | 作答（指向 1.1/1.2/1.3） | 0.427  | 1.3 进程与程序的区别 |
| q39 | 调度相关的东西。            | 作答（指向 1.6）    | 0.208  | 1.6 进程调度算法   |
| q40 | 内存那一块。              | 作答（指向第 2 章）   | 0.280  | 1.4 进程控制块 PCB |
| q38 | 操作系统是怎么管理这些东西的？     | **澄清**       | 0.248  | 1.4 进程控制块 PCB |

**所以「模糊题一律拒答」是错的**（会白丢 3 条准确率），
**「模糊题一律作答」也是错的**（q38 会乱答）。

#### 3.5.2 为什么不能靠调 τ 解决

q38 的 0.248 **高于**可答题 q39 的 0.208。  
想拦 q38 就得把 τ 抬到 0.25 以上，那会先把 q39(0.208) 和 q17(0.2427) 误伤掉 ——  
**净收益为负**。所以判据必须是**字面的**，不是分数的。

#### 3.5.3 判据：三条同时成立才澄清

```
1. 分数过闸（top1 ≥ τ）        —— 否则是拒答，不是澄清
2. 问句含无先行词的指示代词      —— "这些""那一块""它""上面"…
3. 问句锚不到任何章节标题        —— 语料里没有哪一节在讲它问的东西
```

**第 3 点为什么锚定「全部章节标题」而不是「top-k 命中的标题」**：  
检索失败时 top-k 里根本没有正确章节。例如 q40「内存那一块。」的 top-3 是
1.4 PCB / 3.3 目录 / 3.2 文件物理结构 —— 全是错的。  
若按 top-k 锚定，q40 会被误判成"澄清"，把**检索失败**伪装成**口径问题**，
两件事就混了，指标也失真。  
按全量标题锚定，「内存」能对上 2.1 内存管理的基本功能 → q40 走作答路径，
它召回失败的事实如实留在 Hit@k 里，不被掩盖。

为此 `vector_store.stats()` 新增了 `titles` 字段（去重保序的章节标题表）：

```python
titles = store.call("stats", {})["titles"]     # → ["1.1 进程的概念", "1.2 进程的状态", ...]
```

#### 3.5.4 可直接复制的参考实现

完整版（含校验脚本）在 **`evals/vector_db/check_clarify_rule.py`**，
已实测「只命中 q38，其余 39 题零影响」。核心就这 30 行：

```python
import re

TAU = 0.185
PRON_RE = re.compile(r"这个|那个|这些|那些|它|它们|这一块|那一块|上面|上述|刚才|前面说的")

# 泛词：出现在标题里但不指向具体章节，锚定时必须排除。
# 不排掉，"操作系统是怎么管理这些东西的"会因为"管理"命中
# 「内存管理的基本功能」而被误判成可锚定。
GENERIC = {"操作", "系统", "管理", "功能", "方式", "结构", "关系", "区别",
           "概念", "基本", "主要", "常见", "什么", "哪些", "怎么", "如何",
           "存储", "分配", "控制", "状态", "文件"}


def anchor_terms(titles):
    """从全部章节标题里抽可锚定词：去编号、去英文，切 2-gram，滤掉泛词。"""
    out = set()
    for t in titles or []:
        t = re.sub(r"^\d+(\.\d+)*\s*", "", t or "")
        t = re.sub(r"[A-Za-z]+", "", t)
        t = re.sub(r"[\s（）()、，,。.【】]+", "", t)
        out.update(t[i:i + 2] for i in range(len(t) - 1))
    return [g for g in out if g not in GENERIC]


def need_clarify(query, top1_score, titles, tau=TAU):
    if top1_score < tau:
        return False                                # 不过闸 → 拒答路径
    if not PRON_RE.search(query):
        return False                                # 无指示代词 → 正常作答
    return not any(w in query for w in anchor_terms(titles))   # 锚不到 → 澄清
```

编排层的三态流程：

```python
r = store.call("search", {"query": q, "top_k": 3})
titles = store.call("stats", {})["titles"]
top1 = r["hits"][0]["score"] if r["hits"] else 0.0

if top1 < TAU:
    return REFUSE_TEXT                      # 拒答
if need_clarify(q, top1, titles):
    return "你是想问哪一部分？"                 # 澄清
return answer_with(r["hits"])                # 作答
```

#### 3.5.5 边界与注意事项

1. **本模块不做澄清判定。** `need_clarify()` 是**参考实现**，放在 `evals/` 里而不是
   `plugins/vector_db/` 里 —— 因为判定属编排层职责。复制到 qa 组代码里即可。
2. **澄清话术由 qa 组定。** 上面那句"你是想问哪一部分？"只是占位。
3. **澄清依赖标题。** `fixed` 分块模式不产出 `title` 字段，此时 `titles` 为空，
   澄清判据不适用（`check_clarify_rule.py --split fixed` 会直接跳过并说明）。
   **这也是建议启用 `heading` 模式的又一个理由。**
4. **评测口径**：`eval_set.jsonl` 里 q38 的 `expected` 标为 `clarify_ok`，
   标定时**既不判对也不判错**，单列。同时报严格口径（记为误答）备查。
   判对率分母 = 40 − 1 = 39。

---

## 4. 复现命令

```bash
cd campus-qa-system

# 冒烟
python smoke_vector_db.py          # 本地直连插件（无需第三方依赖）
python smoke_rpc_vector_db.py      # 接入基座，走 HTTP RPC（需 fastapi/uvicorn）

# 分块完整性（改动 doc_parser 后必跑）
python evals/vector_db/check_split.py --mode fixed
python evals/vector_db/check_split.py --mode heading

# 检索评测（两种分块对比）
python evals/vector_db/run_eval.py --split both

# 阈值标定
python evals/vector_db/calibrate.py --split heading
python evals/vector_db/calibrate.py --split fixed

# 薄封装一致性回归校验（应输出 [PASS] 两者完全一致）
python evals/vector_db/compare_retriever.py

# 澄清判据校验（应输出 [PASS] 只命中 1 条）
python evals/vector_db/check_clarify_rule.py
```

除 `smoke_rpc_vector_db.py` 外均零第三方依赖，纯标准库。  
`smoke_rpc_vector_db.py` 用 `sys.executable` 起子进程，**必须用装了依赖的解释器运行**，  
否则会打印 skip 提示（返回码 2）。

---

## 5. ⚠️ 我改了组外的哪些文件（请相关组确认）

改了**三个文件**（`doc_parser.py` / `topology.yaml` / `qa/retriever.py`），  
另有一个文件**故意没改**（`gateway/app.py`）。逐条列出。

### 改动 1：修复块间重叠不足（bug fix）

**问题**：原实现 `chunks.append(head.strip())` 在保存分块时 strip 掉了重叠区里的空白，  
导致实际重叠字符数少于 `chunk_overlap` 配置值。

**实测**（配置 overlap=24，size=160）：修复后最小块间重叠为 21；size=60/overlap=12 时最小为 9。  
修复前这个数只会更小。

**影响**：内容不丢（已用差分法验证），但跨越块边界的短语可能无法完整出现在任何单块中，  
影响检索召回。

**修复**：内部保留原文，仅在最终输出时 strip。

**验证**：`python evals/vector_db/check_split.py --mode fixed`  
修复后 5 组参数全部「缺失片段 0」。

### 改动 2：新增标题切分模式（新能力，默认关闭）

```python
parser.initialize({"split_mode": "heading"})   # 默认仍是 "fixed"
```

- `heading`：按 `【标题】` 切，一块 = 一节，标题进 `title` 字段；超长章节内部再走定长切
- `fixed`：原有行为，不变

**为什么值得开**（实测数据，语料 3 章 3084 字 / 评测集 40 题）：

| 指标    | heading   | fixed |
| ----- | --------- | ----- |
| 分块数   | 29        | 28    |
| Hit@1 | **88.9%** | 74.1% |
| Hit@3 | **92.6%** | 88.9% |
| MRR   | **0.907** | 0.815 |

heading 模式下 q07（调度算法）、q20（文件物理结构）都召回了，fixed 下没有 ——  
定长切把清单式内容拆到了相邻两块。

**还有一条额外理由**：`title` 字段是「澄清判定」的锚定依据（见第 3.5 节），
`fixed` 模式不产出 title，澄清判据在该模式下不适用。所以向量库链路上建议启用 heading。

**是否切成默认值，由 file_mgmt 组决定。** 我只提供能力，不替你们改默认值——  
那会影响你们的调用方。建议至少在向量库这条链路上启用。

### 改动 3：chunk 结构新增 `title` 字段

```python
{"chunk_id": ..., "doc_id": ..., "source": ..., "chunk_index": ...,
 "title": "1.2 进程的状态",   # 新增，fixed 模式下为 ""
 "content": ..., "char_count": ...}
```

加字段是向后兼容的，但如果你们有严格 schema 校验，需要同步。

---


### 改动 4：`config/topology.yaml` 新增 `vector_db_node`（请组长/全体确认）

**背景**：交接文档第 2 节的分工表里列了 `vector_db_node`(8102)，  
但 `topology.yaml` 里**没有这个节点** —— 文档承诺与实现不一致。  
同时交接文档第 10 节要求「新插件必须能被 `RemotePluginProxy` 调用，否则视为未接入基座」，  
不补节点就无法验证接入。

**新增内容**（纯新增，不动其他节点）：

```yaml
  vector_db_node:
    group: vector_db
    host: 127.0.0.1
    port: 8102
    data_dir: data/vector_db
    plugins:
      embedder:
        entry: plugins.vector_db.embedder:EmbedderPlugin
        config:
          dim: 4096
      vector_store:
        entry: plugins.vector_db.vector_store:VectorStorePlugin
        config:
          dim: 4096
          w_cos: 0.75
          w_cov: 0.25
```

**验证（实测通过）**：

```bash
python -c "from base.config_loader import *; from base.node import build_registry; \
  print(build_registry(get_node_config(load_topology('config/topology.yaml'), 'vector_db_node')).names())"
# → ['embedder', 'vector_store']

python smoke_rpc_vector_db.py
# → 节点起在 8102，RemotePluginProxy 初始化成功，远程 index/search/analyze 全部可用
# → 库内最低分=0.2850  库外最高分=0.1608  分隔带=+0.1242
# → vector_db 接入基座验证通过
```

`smoke_rpc_vector_db.py` 起的是**真实节点进程**（`python -m base.node --node vector_db_node`），  
不是 TestClient 假装配 —— 验证的是完整 HTTP RPC 链路。

**端口 8102 的归属**：按交接文档第 2 节，8101=file_mgmt、8102=vector_db、  
8103=kg、8104=qa。如果组长已另有安排，告诉我改。

**⚠️ 连带影响：`qa_node` 现在依赖 `vector_db_node` 先启动**

改动 5 里给 `rag_retriever` 配了 `vector_store_url`，`initialize` 会去调 8102 的  
`vector_store.info`。实测：

```bash
# 场景：只起 qa_node，vector_db_node 未启动
python -m base.node --node qa_node
# → urllib.error.HTTPError: HTTP Error 502
# → Traceback: base/node.py build_registry → base/sdk/loader.py registry.create
#              → plugin.initialize
# → 整个 qa_node 启动失败（不是降级，是起不来）
```

**启动顺序**：

```bash
python -m base.node --node vector_db_node    # 8102 先起
python -m base.node --node qa_node           # 8104 后起
```

**要恢复 qa_node 独立启动**：把 `topology.yaml` 里  
`rag_retriever.config.vector_store_url` 那一行注释掉，会自动创建本地 `vector_store` 实例。

**为什么没加 try/except 自动兜底**：静默 fallback 会造出「以为在用远程向量库、  
实际用了本地空索引」这种极难排查的 bug。分布式部署下节点依赖显式化更安全。  
如果 qa 组觉得该加容错，说一声，我改。

**gateway 单进程 demo 不受影响**：`build_demo_registry()` 没传 `vector_store_url`，  
走第 3 级兜底自动建本地 store。已实测 `smoke_fusion.py` / `smoke_api.py` 通过。

---


### 改动 5：`plugins/qa/retriever.py` 改为薄封装（方案 C，已落地）

**这是本次最重要的组外改动，请 qa 组重点确认。**

**改了什么**：`rag_retriever` 的**对外接口一字未动** ——  
`initialize` / `methods` / `index_files` / `index_chunks` / `retrieve` 的  
方法名、参数、返回结构全部保持原样。内部实现从「自写关键词打分」改为  
**转发给 `vector_store`**。

**qa 组需要做什么**：**什么都不用做。** `fusion.py` 里调的还是  
`retriever.retrieve(query, top_k)`，返回结构不变。

**改动的实际内容**：

```python
def retrieve(self, query: str, top_k: int = 3) -> Dict[str, Any]:
    r = self._store.call("search", {"query": query, "top_k": top_k})
    return {"query": r.get("query", query), "hits": r.get("hits") or [],
            "top_k": top_k, "indexed": r.get("indexed", 0),
            "score_scale": r.get("score_scale")}     # 新增字段
```

后端注入优先级（三级兜底，保证 demo 不会因为少配一个 URL 就崩）：

| 顺序 | 配置项                          | 用途                                              |
| -- | ---------------------------- | ----------------------------------------------- |
| 1  | `config["vector_store"]`     | 传入本地实例（单进程 / 测试）                                |
| 2  | `config["vector_store_url"]` | 走 `RemotePluginProxy` 远程调用（`topology.yaml` 里已配） |
| 3  | 都不传                          | 自动创建本地 `EmbedderPlugin + VectorStorePlugin` 兜底  |

**需要 qa 组注意的两点**：

1. **分数尺度变了**。改造前 `retrieve()` 的分数无上界（实测最大 1.0722），  
   现在是 `[0,1]` 归一化。**如果 qa 组在别处对分数做过硬编码假设，需要同步。**  
   目前 `fusion.py` 只做排序融合，不受影响。
2. **返回多了 `score_scale` 字段**。是新增，不是替换，旧代码忽略即可。

**回归校验**：`python evals/vector_db/compare_retriever.py`  
输出 `[PASS] 两者完全一致` 即说明薄封装生效。  
若输出 `[FAIL]`，说明有人恢复了旧的关键词实现，分数尺度会重新变得无上界，  
**编排层的拒答阈值会随之失效** —— 请立刻告知我。

**为什么这么做**：两个检索实现并存，qa 组写拒答判定时就没有统一的分数尺度可用。  
归属依据是分工表（`docs/交接文档.md` 第 2 节把「向量检索」划归 vector_db 组），  
不是性能数据 —— 客观数据见**本文第 7 节**。

---

### 改动 6：`gateway/app.py` **未改，且按组内决定不改**（说明）

`build_demo_registry()` 里没有注册 `vector_store`。**组内已定：不关 vector_db 组的事，不动它。**

#### 6.1 不改会不会影响 demo？—— 不会，已实测

`build_demo_registry()` 创建 `rag_retriever` 时传的是
`{"file_store": ..., "doc_parser": ...}`，**既没有 `vector_store` 也没有 `vector_store_url`**，
所以会命中薄封装的**第 3 级兜底**：自动创建一个本地 `vector_store`。

实测（用 `build_demo_registry()` 起真实装配，灌一份语料）：

```
doc_parser split_mode = fixed
index_files -> {'chunk_count': 28, 'files': ['os_course.txt']}
stats       -> {'indexed': 28, 'dim': 4096, 'vocab_size': 2164,
                'weights': {'cos': 0.75, 'cov': 0.25}, 'chunk_count': 28}

  0.3964 [(无title)]  进程有哪三种基本状态？
  0.5099 [(无title)]  产生死锁的四个必要条件是什么？
  0.0381 [(无title)]  明天广州的天气怎么样？

score_scale 透传 = True
hits 字段名 = ['char_count', 'chunk_id', 'chunk_index', 'content', 'cos', 'cov',
              'doc_id', 'score', 'source', 'title']
```

**结论**：`index_files` / `retrieve` / `stats` 三个方法都正常，字段向后兼容
（`chunk_count` 保留、`title` 与 `score_scale` 是新增），
`smoke_api.py` 也通过。**demo 功能不受影响，不会崩。**

#### 6.2 但 demo 目前跑的是 `fixed` 模式（下一手注意这一条）

`build_demo_registry()` 创建 `doc_parser` 时没传 `split_mode`，所以走默认 `fixed`：

| 影响 | 说明 |
|---|---|
| 分块 | 28 块，**无 `title` 字段** |
| 检索质量 | 相当于 Hit@1 **74.1%**（heading 是 88.9%） |
| 澄清判据 | **不适用** —— 判据依赖标题锚定，无 title 时 `stats()["titles"]` 为空 |
| 拒答阈值 | τ 仍可用（分数尺度与分块模式无关，heading/fixed 都是 0.185 / 0.182 量级） |

**这是「demo 能用但没跑在最优配置上」，不是故障。** 谁负责 demo 谁决定要不要切。

#### 6.3 如果要切（一行的事，留给后续）

**切 heading**：在 `build_demo_registry()` 的 doc_parser 配置里加一项：

```python
doc_parser = registry.create(
    "doc_parser",
    {
        "chunk_size": 160,
        "chunk_overlap": 24,
        "chunk_dir": str(root / "chunks"),
        "file_store": file_store,
        "split_mode": "heading",          # ← 只加这一行
    },
)
```

**共用远程节点**（若以后 vector_db 独立进程跑）：给 retriever 配置加一项：

```python
retriever = registry.create(
    "rag_retriever",
    {"file_store": file_store, "doc_parser": doc_parser,
     "vector_store_url": "http://127.0.0.1:8102"},     # ← 只加这一行
)
```

**要在 demo 里直接调 `vector_store`**（而不是经 retriever 转发），才需要注册插件：

```python
from plugins.vector_db import EmbedderPlugin, VectorStorePlugin

# registry.register(...) 区块
registry.register("embedder", EmbedderPlugin)
registry.register("vector_store", VectorStorePlugin)

# 实例化区块（doc_parser 之后）
embedder = registry.create("embedder", {"dim": 4096})
vector_store = registry.create("vector_store", {"dim": 4096, "embedder": embedder})
```

再在 `/kg/build` 端点里灌索引：

```python
chunks = []
for name in (filenames or [f["filename"] for f in reg.get("file_store").call("list", {})]):
    doc = reg.get("file_store").call("load", {"filename": name})
    chunks += reg.get("doc_parser").call(
        "split", {"text": doc["content"], "filename": name, "persist": False})["chunks"]
reg.get("vector_store").call("index", {"chunks": chunks})
```

**以上都是「如果」** —— 不做也完全没问题，当前兜底路径已经能跑。

---

## 6. 还没做完的（诚实清单）

| # | 事项                                                 | 状态 / 卡点                                  | 谁做   |
| - | -------------------------------------------------- | ---------------------------------------- | ---- |
| 1 | `plugins/qa/retriever.py` 改调 `vector_store.search` | ✅ **已完成**（方案 C 薄封装，见改动 5）                | 我    |
| 2 | 语料补齐（≥3 章）                                          | ✅ **已完成**：3 章 23 节 3084 字（见第 8 节第 4 条说明） | 我    |
| 3 | 换语料后重跑评测与标定                                        | ✅ **已完成**：40 题、τ=0.185、全部脚本重跑            | 我    |
| 4 | `gateway/app.py` 注册向量库（demo 用）                     | ✅ **已定：不改**（不关本组的事，且兜底路径已能跑，见改动 6） | — |
| 5 | 语料换成教师指定课件                                         | ⏸ 教师未再发放完整版，当前为公开教材整理版                   | 全组   |

**已完成**：插件实现、评测集（40 题）、评测脚本、阈值标定、分块完整性检查、  
一致性回归校验、澄清判据校验、`topology.yaml` 加 `vector_db_node`、设计文档、交接文档、模块 README。

**组内已确认的两件事（2026-09-22）**：

1. **语料来源**：1.1–1.3 节教师样例原文 + 其余 20 节公开教材整理，**组内接受**。
   来源说明写在语料文件头部，不藏。教师不再发放完整版，按现状交付。
2. **模糊题口径**：定为**三态**（作答 / 澄清 / 拒答），q38 应澄清。
   判据与参考实现见第 3.5 节，`eval_set.jsonl` 的 `expected` 字段已落地。

**第 5 项是唯一的实质性遗留**：若后续拿到教师指定课件，
按设计文档第 11 节的四步重跑即可，脚本现成。

**另需注意**：向量索引是**内存态**，节点重启即丢。demo 流程里每次 `/kg/build`  
重建索引即可；若要做成常驻服务，需要给 `vector_store` 加持久化（当前未做，  
不在项目一最低要求内）。

---

## 7. 检索实现归属：已定案（方案 C）

**状态：已按方案 C 落地。** 本节保留决策依据，供答辩与评审时引用。

**跑法**：`python evals/vector_db/compare_retriever.py`

### 7.1 改造前：两者曾客观对比过

口径：旧语料 1 章 8 块 / 30 题。**与当前 3 章 29 块 / 40 题不可直接比较。**

| 指标        | 原 `rag_retriever`（qa 组） | `vector_store`（本组） |
| --------- | ----------------------- | ------------------ |
| Hit@1     | 95.0%                   | 95.0%              |
| **Hit@3** | **100.0%**              | 95.0%              |
| **MRR**   | **0.967**               | 0.950              |
| 未召回题      | **0**                   | 1（q01）             |
| 分数最大值     | 1.0722                  | 0.5698             |
| 分数越界条数    | 2                       | **0**              |
| 库内最低分     | 0.3333                  | 0.2071             |
| 无关题最高分    | 0.2667                  | 0.1767             |
| 分隔带       | **+0.0666**             | +0.0304            |

### 7.2 诚实结论（不美化）

**1. 检索质量上，原 `rag_retriever` 不差，甚至略优。**

Hit@1 打平，Hit@3 与 MRR 它更高。原因是它的关键词直接匹配对短问句很有效，  
而 `vector_store` 的 IDF 压制反而让 `q01`（"什么是进程？"）丢了分。

**所以「归位」的论据不能是「我们的检索更准」——事实并非如此。**

**2. 尺度稳定性两者都不理想。**

同一语义、不同问句长度的 top-1 分数：

| 问句                  | 原 rag_retriever | vector_store |
| ------------------- | --------------- | ------------ |
| 进程                  | 0.8500          | 0.3870       |
| 进程的状态               | 0.7143          | 0.4384       |
| 进程有哪三种基本状态          | 0.6667          | 0.4766       |
| ……以及它们如何转换          | 0.5500          | 0.4968       |
| ……以及这些状态在什么条件下会发生转换 | 0.6667          | 0.6295       |
| **分数跨度**            | 0.3000          | 0.2425       |

语义相同，分数却漂移 0.24~0.30。`vector_store` 略稳，但**优势不大，不宜夸大**。

**3. 原 `rag_retriever` 确有一个可验证的缺陷**：分数无上界。  
`score = 交集比 + 0.35 × 完整词命中数`，第二项随问句长度线性增长，  
实测 2 条分数 > 1（最大 1.0722）。不过本语料下它仍有分隔带，  
**所以准确说法是「尺度不封顶、阈值不可移植」，不是「完全不能用」**。

### 7.3 三个方案与最终选择

| 方案              | 做法                                          | 代价                |
| --------------- | ------------------------------------------- | ----------------- |
| A 归位            | vector_db 接手检索，`rag_retriever` 删除           | qa 组要改 fusion 调用点 |
| B 保留            | 两个实现并存，vector_db 只交付「评测集 + 阈值标定 + 分块策略」     | 职责重叠未解决           |
| **C 薄封装 ✅ 已采用** | `rag_retriever` 对外接口不变，内部转发给 `vector_store` | 内部约 40 行改写        |

**为什么选 C**：

- qa 组**零改动** —— `fusion.py` 里调的还是 `retriever.retrieve(...)`
- 职责归位 —— 检索逻辑回到 vector_db 组
- 评测集与阈值标定**直接可用**（底层已换成 `vector_store`）
- 分数统一到 `[0,1]`，qa 组写拒答逻辑时不用管尺度

**两个实现并存是必须避免的**：调用方不知道用哪个，两套分数尺度让 qa 组  
无法写统一的拒答判定，还要维护两份同类代码。

**归属依据是分工表，不是性能数据。** 交接文档第 2 节把「向量检索」划归  
vector_db 组，第 10 节写明「越界代码在评审时打回」。

### 7.4 落地后的现状

| 指标     | `rag_retriever` | `vector_store` |
| ------ | --------------- | -------------- |
| Hit@1  | 88.9%           | 88.9%          |
| Hit@3  | 92.6%           | 92.6%          |
| MRR    | 0.907           | 0.907          |
| 分数最大值  | 0.5808          | 0.5808         |
| 分数越界条数 | 0               | 0              |
| 未召回题   | q01、q40         | q01、q40        |

```
[PASS] 两者完全一致 —— 薄封装生效
```

`compare_retriever.py` 现在是**回归校验**：出现不一致即说明薄封装被破坏，  
分数尺度会重新变得无上界，编排层的拒答阈值随之失效。

---

## 8. 已知限制（写进设计文档，别被面试问穿）

1. **字面向量有天花板**。哈希 n-gram 无法理解同义改写（"进程怎么切换" vs "进程状态转换"）。  
   实测 Hit@1 88.9% 已是这套方案的上限，剩余错误集中在需要语义泛化的题上。  
   升级路径：接课程统一 embedding 接口，`embedder.vectorize()` 内部替换即可，其余代码不动。
2. **两条未召回（q01、q40）都是「问句过短」**，共同特征是只给了 6 字以内的信息。  
   - `q01`「什么是进程？」top-1 命中 1.5 节（0.300）——"进程"在 29 块里的 12 块出现，  
     IDF=1.84 不足以把它拉到 1.1 节。  
   - `q40`「内存那一块。」top-3 全错（1.4 PCB / 3.3 目录 / 3.2 文件物理结构）——  
     只有一个词"内存"作线索，被泛词挤掉。  
   这是**有效信号不足**，不是检索实现的问题。
3. **OOV 信号对诱导题已基本失效**（语料扩大后被稀释），见 2.4。
4. **评测语料不是教师发放材料**。`corpus/os_course.txt` 共 3 章 23 节 3084 字，  
   其中 **1.1 / 1.2 / 1.3 三节是教师样例原文**，其余 20 节是依据公开操作系统教材  
   通用知识点整理的。教师明确不再发放完整版。  
   **来源说明已写在语料文件头部，不藏；组内已确认接受。**
5. **指标口径自查过一次误报**。初版命中判定在标题不匹配时会退到"关键词是否出现在正文"，
   导致 q40 因 1.4 块含"内存指针"被判成命中第 2 章，Hit@1 虚高到 92.6%。  
   已修（heading 模式只按标题判），修正后 Hit@1 = 88.9%、MRR = 0.907。  
   **这一条要主动讲，别让人以为 88.9% 是退步** —— 它是把虚高的数改成了真数。

---

## 9. 一句话总结

向量检索这一环**已交付完毕**：可运行、可复现、有量化指标、有失败归因，  
已接入基座（`vector_db_node` 8102，`RemotePluginProxy` 可调用），  
归属问题已按方案 C（薄封装）落地 —— `rag_retriever` 转发 `vector_store`，两者指标完全一致。

**qa 组要做三件事**：

1. 读 `calibration_heading.json` 里的 `tau=0.185`，在调 LLM 之前用硬阈值判拒答（第 3.1 节）。
2. 加一道「答案可支撑性检查」拦 4 条诱导题 —— 阈值拦不住（第 3.3 节）。
3. 按三态口径接模糊题：q38 那种"锚不到章节"的追问澄清，不要拒答（**第 3.5 节，含可复制代码**）。

本模块提供三个信号：`search` 的分数、`analyze` 的 OOV、`stats` 的章节标题表。

**组内已确认**：语料来源接受（1.1–1.3 教师原文 + 20 节公开教材整理，来源写在文件头）；
模糊题口径定为三态。

**提醒**：向量索引是内存态，节点重启即丢；demo 每次建图时重建。

---

## 10. 出问题怎么退回去

本模块的交付**全部在 `vector-db` 分支**，`main` 仍是基线 `11c043f`，**未动**。
⚠️ 仓库的**默认分支是 `master`**（比 `main` 新 1 个提交），分支现状与 PR 目标待确认，
见 `docs/交接文档.md` 第 12 节。
所以最坏情况下「不 merge」就等于没上线。

**完整回退方案（含 6 种场景、3 个本机环境坑、回退后验证命令）见
`docs/交接文档.md` 第 12 节**，已逐条实测。这里只放与本模块直接相关的两条：

| 想退掉什么 | 命令 |
|---|---|
| 只退 `plugins/qa/retriever.py`（薄封装改回原实现） | `git checkout 11c043f -- campus-qa-system/plugins/qa/retriever.py` |
| 只退 `plugins/file_mgmt/doc_parser.py`（分块改动） | `git checkout 11c043f -- campus-qa-system/plugins/file_mgmt/doc_parser.py` |

⚠️ 上面两条命令**会连暂存区一起改**，之后想复原必须用：

```bash
git restore --source=HEAD --staged --worktree <文件路径>
```

单用 `git restore <文件>` **无效**（实测过，见交接文档 12.2 坑 2）。

**如果只是想让 `qa_node` 恢复独立启动**，不用回退任何代码 ——
把 `topology.yaml` 里 `rag_retriever.config.vector_store_url` 那一行注释掉即可（第 5 节改动 4）。
