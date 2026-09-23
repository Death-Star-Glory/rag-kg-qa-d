# plugins/vector_db — 向量化与相似检索

**职责边界**（交接文档第 2 节）：只做向量化、建索引、相似检索。
不解析文档（→ file_mgmt）、不生成答案、**不做拒答判定**（→ qa 编排层）。

本模块只回答一个问题：**问句和哪些分块最相关，相关到什么程度。**

---

## 两个插件

### `embedder` — 文本 → 定长稠密向量

| 方法 | 参数 | 返回 |
|---|---|---|
| `embed` | `{"text": str}` | `{"dim": 4096, "vector": [...]}` |
| `dim` | `{}` | `{"dim": 4096}` |
| `info` | `{}` | 插件元信息 |

离线实现：字符 unigram + bigram 哈希 + TF 加权，4096 维，零第三方依赖。
接课程统一 embedding 接口时**只替换 `vectorize()` 内部**，其余代码不动。

### `vector_store` — 索引与检索

| 方法 | 参数 | 返回 |
|---|---|---|
| `index` | `{"chunks": [...]}` | `{"indexed": n, "dim": 4096, "vocab_size": m}` |
| `search` | `{"query": str, "top_k": 3}` | 见下 |
| `analyze` | `{"query": str}` | `{"oov_ratio": float, "oov_chars": [...]}` |
| `score_scale` | `{}` | 分数尺度说明（编排层做拒答判定前必读） |
| `stats` | `{}` | `{"indexed", "dim", "vocab_size", "weights", "titles"}` |
| `info` | `{}` | 插件元信息 |

`stats()["titles"]` 是索引里的全部章节标题（去重保序）——
编排层做「澄清判定」时用它锚定问句，见 `docs/vector_db交接文档.md` 第 3.5 节。

`search` 返回：

```json
{
  "query": "...", "top_k": 3, "indexed": 29,
  "hits": [
    {"chunk_id": "...", "title": "1.2 进程的状态", "content": "...",
     "score": 0.4712, "cos": 0.3810, "cov": 0.742}
  ],
  "score_scale": {"formula": "0.75 * cosine + 0.25 * keyword_coverage", "range": [0.0, 1.0]}
}
```

打分公式：`score = w_cos × cosine + w_cov × keyword_coverage`，两分项与加权和均落在 `[0,1]`。
**归一化是刻意的** —— 拒答阈值需要固定尺度才可用。

---

## 调用方式

### 本地（单进程）

```python
from plugins.vector_db import EmbedderPlugin, VectorStorePlugin

embedder = EmbedderPlugin(); embedder.initialize({"dim": 4096})
store = VectorStorePlugin();  store.initialize({"dim": 4096, "embedder": embedder})
store.call("index", {"chunks": chunks})
store.call("search", {"query": "进程有哪些状态", "top_k": 3})
```

### 远程（多进程，走基座）

节点已在 `config/topology.yaml` 注册为 `vector_db_node`(8102)：

```bash
python -m base.node --node vector_db_node
```

```python
from base.rpc.client import RemotePluginProxy
store = RemotePluginProxy("vector_store", "http://127.0.0.1:8102")
store.initialize()
store.call("search", {"query": "进程有哪些状态", "top_k": 3})
```

RPC 信封：`POST {base_url}/rpc`，`{"id": ..., "method": "vector_store.search", "params": {...}}`。

**注意**：索引是**内存态**，节点重启即丢。demo 流程每次重建索引即可；
若要常驻服务需另加持久化（当前未做）。

---

## 跑法

```bash
python smoke_vector_db.py          # 本地冒烟（零依赖）
python smoke_rpc_vector_db.py      # 接入基座验证（需 fastapi/uvicorn）
```

评测与阈值标定见 `evals/vector_db/README.md`。
设计依据见 `docs/vector_db设计文档.md`，协作说明见 `docs/vector_db交接文档.md`。

---

## 关键数字（2026-09-22 实测）

**口径**：语料 `evals/vector_db/corpus/os_course.txt`（操作系统 3 章 23 节，3084 字，29 块）
· 评测集 40 题 · top_k=3。

| 项 | 值 |
|---|---|
| 库内 Hit@1（标题切分块） | **88.9%** |
| 库内 Hit@1（定长切分块） | 74.1% |
| 库内 Hit@3（标题切分块） | **92.6%** |
| MRR（标题切分块） | **0.907** |
| 推荐拒答阈值 τ | **0.185**（heading）/ 0.182（fixed） |
| 判对 | 35/39 = 89.7%（1 条口径为「应澄清」，单列） |
| 无关题拒答率 | 100%（8/8） |
| 诱导题拒答率 | 0%（字面向量硬上限，需编排层加可支撑性检查） |
| 分数范围 | 恒在 `[0,1]`，实测最大 0.5808，越界 0 条 |

`plugins/qa/retriever.py` 已按方案 C 改为薄封装，指标与本模块**完全一致**
（`python evals/vector_db/compare_retriever.py` 校验）。
