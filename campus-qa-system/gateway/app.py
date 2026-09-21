"""FastAPI Gateway：文档 / 图谱 / 融合问答 API。

注意：本文件不要使用 from __future__ import annotations，
否则 FastAPI/Pydantic 无法解析嵌套在函数内的请求模型注解。
"""

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from base.sdk.registry import PluginRegistry
from plugins.file_mgmt import DocParserPlugin, FileStorePlugin
from plugins.kg import KgBuilderPlugin, KgQueryPlugin
from plugins.qa import FusionQaPlugin, RagRetrieverPlugin

try:
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel, Field
    FASTAPI_OK = True
except ImportError:
    FASTAPI_OK = False
    BaseModel = object  # type: ignore
    Field = None  # type: ignore
    FastAPI = HTTPException = None  # type: ignore


def _ok(data: Any, message: str = "ok") -> Dict[str, Any]:
    return {"code": 0, "message": message, "data": data}


def _err(message: str, code: int = 1, data: Any = None) -> Dict[str, Any]:
    return {"code": code, "message": message, "data": data}


class DocIn(BaseModel):
    filename: str
    content: str
    source: Optional[str] = None


class AskIn(BaseModel):
    question: str
    top_k: int = 3


class BuildIn(BaseModel):
    filenames: Optional[List[str]] = None


class SubgraphIn(BaseModel):
    entity: str
    depth: int = 1
    type: Optional[str] = None


def build_demo_registry(data_dir: Optional[Path] = None) -> PluginRegistry:
    """单进程演示：本地装配四组插件（仍保持插件接口，可改为 RPC）。"""
    root = Path(data_dir) if data_dir else (PROJECT_ROOT / "data" / "demo")
    root.mkdir(parents=True, exist_ok=True)

    registry = PluginRegistry()
    registry.register("file_store", FileStorePlugin)
    registry.register("doc_parser", DocParserPlugin)
    registry.register("kg_builder", KgBuilderPlugin)
    registry.register("kg_query", KgQueryPlugin)
    registry.register("rag_retriever", RagRetrieverPlugin)
    registry.register("fusion_qa", FusionQaPlugin)

    file_store = registry.create(
        "file_store",
        {"root": str(root / "files"), "meta_path": str(root / "files_meta.json")},
    )
    doc_parser = registry.create(
        "doc_parser",
        {
            "chunk_size": 160,
            "chunk_overlap": 24,
            "chunk_dir": str(root / "chunks"),
            "file_store": file_store,
        },
    )
    kg_builder = registry.create(
        "kg_builder",
        {
            "persist_path": str(root / "kg.json"),
            "file_store": file_store,
            "doc_parser": doc_parser,
        },
    )
    kg_query = registry.create(
        "kg_query",
        {"store": kg_builder.store, "persist_path": str(root / "kg.json"), "max_nodes": 50},
    )
    retriever = registry.create(
        "rag_retriever",
        {"file_store": file_store, "doc_parser": doc_parser},
    )
    registry.create(
        "fusion_qa",
        {"retriever": retriever, "kg_query": kg_query, "top_k": 3},
    )
    return registry


def create_app(registry: Optional[PluginRegistry] = None):
    if not FASTAPI_OK:
        raise RuntimeError("gateway 需要 fastapi: pip install fastapi uvicorn pyyaml")

    reg = registry or build_demo_registry()
    app = FastAPI(title="campus-qa gateway", version="0.1.0")

    @app.get("/health")
    def health():
        return _ok({"plugins": reg.names(), "status": "ok"})

    @app.get("/system/status")
    def system_status():
        kg_stats = reg.get("kg_query").call("stats", {})
        rag_stats = reg.get("rag_retriever").call("stats", {})
        kg_status = reg.get("kg_builder").call("status", {})
        return _ok(
            {
                "plugins": reg.names(),
                "documents": reg.get("file_store").call("list", {}),
                "kg": {"stats": kg_stats, "build": kg_status},
                "rag": rag_stats,
            }
        )

    @app.post("/documents/save")
    def save_document(body: DocIn):
        rec = reg.get("file_store").call(
            "save",
            {"filename": body.filename, "content": body.content, "source": body.source},
        )
        return _ok(rec, "saved")

    @app.get("/documents/list")
    def list_documents():
        return _ok(reg.get("file_store").call("list", {}))

    @app.post("/kg/build")
    def kg_build(body: BuildIn = None):  # type: ignore
        filenames = body.filenames if body is not None else None
        result = reg.get("kg_builder").call("build", {"filenames": filenames})
        reg.get("rag_retriever").call("index_files", {})
        return _ok(result, "built")

    @app.get("/kg/status")
    def kg_status():
        builder = reg.get("kg_builder").call("status", {})
        stats = reg.get("kg_query").call("stats", {})
        return _ok({"build": builder, "stats": stats})

    @app.get("/kg/query")
    def kg_query(entity: str, limit: int = 10):
        return _ok(reg.get("kg_query").call("search", {"entity": entity, "limit": limit}))

    @app.post("/kg/subgraph")
    def kg_subgraph(body: SubgraphIn):
        data = reg.get("kg_query").call(
            "subgraph",
            {"entity": body.entity, "depth": body.depth, "type": body.type},
        )
        return _ok(data)

    @app.post("/qa/ask")
    def qa_ask(body: AskIn):
        try:
            data = reg.get("fusion_qa").call(
                "ask", {"question": body.question, "top_k": body.top_k}
            )
        except Exception as e:  # noqa: BLE001
            raise HTTPException(status_code=500, detail=str(e)) from e
        return _ok(data, "ok")

    return app


def main(argv: Optional[List[str]] = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="campus-qa gateway")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8300)
    parser.add_argument("--data-dir", default=str(PROJECT_ROOT / "data" / "demo"))
    args = parser.parse_args(argv)

    registry = build_demo_registry(Path(args.data_dir))
    app = create_app(registry)
    print(f"[gateway] http://{args.host}:{args.port} plugins={registry.names()}")
    import uvicorn

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
