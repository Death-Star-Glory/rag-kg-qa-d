# RAG QA Demo

基于RAG的智能问答系统演示（课设项目）

## 项目简介

这是一个基于检索增强生成（RAG）技术的智能问答系统演示项目，用于课程设计。项目采用模块化架构，支持配置驱动和插件化扩展。

## 核心特性

### 1. 混合检索
- 向量语义检索（ChromaDB）
- BM25关键词检索
- 动态权重调整
- 交叉编码器重排序

### 2. 答案生成
- 支持OpenAI和Ollama
- 提示词工程
- 答案引用标注
- 置信度评估

### 3. 文档处理
- 支持PDF、TXT、DOCX格式
- 智能分块策略
- 向量化存储

### 4. 模块化架构
- 配置驱动
- 插件化设计
- 依赖注入
- 接口抽象

## 技术栈

| 类别 | 技术选型 | 说明 |
|------|----------|------|
| 后端框架 | FastAPI | 高性能异步API框架 |
| 前端框架 | Streamlit | 快速构建数据应用界面 |
| LLM模型 | Qwen2.5-7B / OpenAI API | 语义理解与生成 |
| 向量数据库 | ChromaDB | 轻量级向量存储 |
| RAG框架 | LangChain | 文档处理与检索管道 |
| 嵌入模型 | text2vec-base-chinese | 中文文本向量化 |
| 重排序 | cross-encoder/ms-marco | 检索结果精排 |

## 快速开始

### 1. 环境准备

```bash
# 克隆项目
git clone <repository-url>
cd rag-qa-demo

# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或
venv\Scripts\activate  # Windows

# 安装依赖
pip install -r requirements.txt
```

### 2. 配置系统

```bash
# 复制配置模板
cp config/default.yaml config/development.yaml

# 编辑配置文件
# config/development.yaml
```

### 3. 启动服务

```bash
# 启动后端
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# 启动前端（新终端）
cd frontend
streamlit run app.py
```

### 4. 访问系统

- 前端界面：http://localhost:8501
- API文档：http://localhost:8000/docs
- 系统状态：http://localhost:8000/system/status

## 项目结构

```
rag-qa-demo/
├── app/                          # 后端应用
│   ├── core/                     # 核心模块
│   │   ├── interfaces/           # 接口定义
│   │   ├── config.py             # 配置类
│   │   ├── config_manager.py     # 配置管理器
│   │   ├── registry.py           # 模块注册表
│   │   ├── container.py          # 依赖注入容器
│   │   └── ...                   # 其他模块
│   └── main.py                   # FastAPI入口
├── frontend/                     # 前端应用
│   └── app.py                    # Streamlit入口
├── config/                       # 配置文件
│   ├── default.yaml              # 默认配置
│   ├── development.yaml          # 开发环境配置
│   └── modules/                  # 模块配置
├── data/                         # 数据目录
│   ├── documents/                # 原始文档
│   ├── vector_store/             # 向量数据库
│   └── test_data/                # 测试数据
├── tests/                        # 测试文件
├── requirements.txt              # 依赖清单
└── README.md                     # 项目说明
```

## 配置说明

### 配置文件结构

```yaml
# config/default.yaml
app:
  name: "RAG问答演示"
  version: "1.0.0"
  debug: false
  environment: "development"

server:
  host: "0.0.0.0"
  port: 8000
  reload: false
  workers: 1

# ... 更多配置
```

### 环境变量覆盖

支持通过环境变量覆盖配置，前缀为 `APP_`：

```bash
export APP_SERVER_PORT=9000
export APP_LLM_PROVIDER=openai
export APP_DEBUG=true
```

## API接口

### 文档管理

- `POST /documents/upload` - 上传文档
- `GET /documents/list` - 列出文档
- `DELETE /documents/{filename}` - 删除文档

### 智能问答

- `POST /qa/ask` - 提问

### 系统状态

- `GET /system/status` - 获取系统状态
- `GET /health` - 健康检查

## 扩展开发

### 添加新模块

1. **定义接口**：在 `app/core/interfaces/` 中添加抽象基类
2. **实现模块**：创建具体实现类
3. **注册模块**：使用装饰器注册到注册表
4. **配置模块**：在 `config/modules/` 中添加配置文件

### 示例：自定义检索器

```python
from app.core.interfaces.retriever import BaseRetriever
from app.core.registry import ModuleRegistry

@ModuleRegistry.register("custom_retriever")
class CustomRetriever(BaseRetriever):
    def _do_initialize(self):
        # 初始化逻辑
        pass
    
    def retrieve(self, query, top_k=5):
        # 检索逻辑
        pass
```

## 测试

```bash
# 运行所有测试
pytest tests/

# 运行特定测试
pytest tests/test_config.py

# 运行带覆盖率的测试
pytest --cov=app tests/
```

## 使用示例

### 1. 上传文档

```python
import requests

# 上传文档
with open("document.pdf", "rb") as f:
    response = requests.post(
        "http://localhost:8000/documents/upload",
        files={"file": f}
    )
print(response.json())
```

### 2. 提问

```python
import requests

# 提问
response = requests.post(
    "http://localhost:8000/qa/ask",
    json={"question": "计算机科学专业的毕业要求是什么？"}
)
print(response.json())
```

## 课程设计要求

### 1. 功能要求

- [x] 文档上传与处理
- [x] 智能问答功能
- [x] 答案引用标注
- [x] 混合检索策略
- [x] 模块化架构

### 2. 技术要求

- [x] 使用FastAPI框架
- [x] 使用Streamlit前端
- [x] 使用LangChain框架
- [x] 使用ChromaDB向量数据库
- [x] 支持多种LLM

### 3. 文档要求

- [x] 系统设计文档
- [x] API接口文档
- [x] 用户使用手册
- [x] 测试报告

## 常见问题

### 1. 模型下载失败

```bash
# 设置 HuggingFace 镜像
export HF_ENDPOINT=https://hf-mirror.com
```

### 2. 内存不足

- 减小 `chunk_size` 配置
- 使用更小的嵌入模型
- 启用模型量化

### 3. 连接数据库失败

- 检查数据库服务是否启动
- 检查配置文件中的连接参数
- 检查网络连接

## 贡献指南

1. Fork 项目
2. 创建功能分支
3. 提交更改
4. 推送到分支
5. 创建 Pull Request

## 许可证

MIT License

## 联系方式

- 项目主页：<repository-url>
- 问题反馈：<issues-url>
- 邮箱：<email>