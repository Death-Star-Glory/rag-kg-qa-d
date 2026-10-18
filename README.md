# RAG + KG 高校智能问答（课设）

基于 RAG 与知识图谱融合的高校智能问答系统 — **课程设计仓库**。

## 本次交付：插件基座 + KG API + RAG/KG 融合 + 三态问答（qa 组）

正式开发入口见 **[`campus-qa-system/`](campus-qa-system/)**：

- 插件契约 / 注册表 / 配置驱动 / JSON-RPC 节点
- 知识图谱：`kg_builder` 规则建图、`kg_query` 子图查询
- 问答融合：`rag_retriever` + `fusion_qa`（chunk 证据 + 图谱三元组 + 溯源）
- qa 组（2026-10-12）：三态判定（作答/澄清/拒答）+ 生成（LLM 优先/抽取式兜底）+ 溯源，
  评测 40 题：准确率 92.6%、拒答正确率 100%、幻觉 0（[`docs/qa交接文档.md`](campus-qa-system/docs/qa交接文档.md)）
- Gateway REST：`/documents/*` `/kg/*` `/qa/ask`
- 交接文档：[`campus-qa-system/docs/交接文档.md`](campus-qa-system/docs/交接文档.md)

```bash
cd campus-qa-system
pip install -r requirements.txt
python smoke_base.py
python smoke_qa.py        # qa 三态冒烟
python smoke_fusion.py
python smoke_api.py
python evals/project1/run_eval.py   # 项目一评测
python -m gateway.app --port 8300
```

历史 demo 的 Docker / 脚本仍在仓库根目录，供对照；课设主入口是 `campus-qa-system/`。

## 项目简介

这是一个基于检索增强生成（RAG）技术的智能问答系统演示项目，用于课程设计。项目采用模块化架构，支持配置驱动和插件化扩展。
