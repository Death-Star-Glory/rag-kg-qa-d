# 课设：高校智能问答系统（基座）

插件化 + 远程通讯（分布式）最小基座。业务按四组扩展：

| 业务组 | 键名 | 职责边界 |
|--------|------|----------|
| 文件管理 | `file_mgmt` | 原文存储、解析拆解 |
| 向量数据库 | `vector_db` | 向量化、入库与检索 |
| 知识图谱 | `kg` | 实体关系构建与查询 |
| 知识问答 | `qa` | 检索编排与答案生成 |

本仓库当前交付：**框架基座 + 交接文档 + echo 示例插件**。业务插件由后续按组接入。

## 快速开始

```bash
cd campus-qa-system
pip install fastapi uvicorn pyyaml

# 基座自检（无需网络）
python smoke_base.py

# 启动示例节点（加载 echo 插件）
python -m base.node --config config/topology.yaml --node echo_node

# 另开终端验证 RPC
python smoke_base.py --rpc http://127.0.0.1:8200
```

## 目录

```
campus-qa-system/
├── base/                 # 基座：SDK / RPC / 节点 / 配置
├── plugins/              # 业务插件（四组占位）
├── examples/echo_plugin.py
├── config/topology.yaml
├── docs/交接文档.md
└── smoke_base.py
```

详细设计与交接说明见 [docs/交接文档.md](docs/交接文档.md)。
