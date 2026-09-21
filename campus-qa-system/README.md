# 课设：高校智能问答系统（基座 + RAG/KG 融合）

插件化 + 远程通讯（分布式）最小基座，已接入 **知识图谱 API** 与 **RAG+KG 融合问答**。

| 业务组 | 键名 | 职责边界 |
|--------|------|----------|
| 文件管理 | `file_mgmt` | 原文存储、解析拆解 |
| 向量数据库 | `vector_db` | 向量化、入库与检索（占位/可扩展） |
| 知识图谱 | `kg` | 实体关系构建与子图查询 |
| 知识问答 | `qa` | RAG 检索 + KG 融合问答 |

## 快速开始

```bash
cd campus-qa-system
pip install fastapi uvicorn pyyaml

# 基座自检
python smoke_base.py

# RAG+KG 融合链路
python smoke_fusion.py

# REST API（TestClient）
python smoke_api.py

# 启动网关（浏览器/前端对接）
python -m gateway.app --port 8300
```

## Gateway REST

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/health` | 存活与插件列表 |
| GET | `/system/status` | 文档/图谱/检索状态 |
| POST | `/documents/save` | `{filename, content, source?}` |
| GET | `/documents/list` | 文档列表 |
| POST | `/kg/build` | `{filenames?: []}` 建图并重建 RAG 索引 |
| GET | `/kg/status` | 建图状态 + 图谱统计 |
| GET | `/kg/query?entity=` | 实体检索 |
| POST | `/kg/subgraph` | `{entity, depth?}` 子图 nodes/links |
| POST | `/qa/ask` | `{question, top_k?}` RAG+KG 融合回答 |

统一响应：`{code, message, data}`。

## 融合说明（PPT 可用）

```text
问题 → RAG(chunk 召回) + KG(实体/三元组/子图) → 组装证据 → 答案+溯源+置信度
```

- 图谱抽取：规则匹配（学分/先修/属于/讲授/专业要求）
- 检索：字符/关键词打分（原型；可换向量库插件）
- 子图 JSON：`nodes[{id,label,type}]` + `links[{source,target,type}]`，前端可直接画图

## 目录

```
campus-qa-system/
├── base/                 # 插件 SDK / RPC / 节点
├── plugins/
│   ├── file_mgmt/        # file_store, doc_parser
│   ├── kg/               # kg_builder, kg_query, kg_store
│   ├── qa/               # rag_retriever, fusion_qa
│   └── vector_db/        # 占位
├── gateway/app.py        # REST 对接层
├── config/topology.yaml
├── data/sample_docs/     # 样例教务文档
├── docs/交接文档.md
└── smoke_*.py
```

## 节点 / RPC 基座

```bash
python smoke_base.py
python -m base.node --node echo_node
python smoke_base.py --rpc
```

业务插件入口见 `config/topology.yaml`；演示网关当前单进程装配插件，接口已按组拆分，可改为 `RemotePluginProxy` 分布式部署。

详细交接见 [docs/交接文档.md](docs/交接文档.md)。
