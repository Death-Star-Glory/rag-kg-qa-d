"""铻嶅悎閾捐矾鍐掔儫锛氭枃妗?鈫?寤哄浘 鈫?RAG+KG 闂瓟銆?""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from gateway.app import build_demo_registry  # noqa: E402


def main() -> None:
    data_dir = ROOT / "data" / "smoke_demo"
    if data_dir.exists():
        shutil.rmtree(data_dir)
    sample = ROOT / "data" / "sample_docs" / "cs_curriculum.txt"
    text = sample.read_text(encoding="utf-8")

    reg = build_demo_registry(data_dir)
    store = reg.get("file_store")
    builder = reg.get("kg_builder")
    retriever = reg.get("rag_retriever")
    fusion = reg.get("fusion_qa")
    kg_query = reg.get("kg_query")

    print("== save doc ==")
    print(store.call("save", {"filename": "cs_curriculum.txt", "content": text, "source": "sample"}))

    print("\n== kg build ==")
    built = builder.call("build", {"filenames": ["cs_curriculum.txt"]})
    print(built)
    assert built["graph"]["node_count"] >= 3, built
    assert built["graph"]["edge_count"] >= 2, built

    print("\n== rag index ==")
    print(retriever.call("index_files", {}))

    print("\n== kg subgraph 鎿嶄綔绯荤粺 ==")
    sg = kg_query.call("subgraph", {"entity": "鎿嶄綔绯荤粺", "depth": 1})
    print({"found": sg["found"], "center": sg.get("center_label"), "nodes": len(sg["nodes"]), "links": len(sg["links"])})
    assert sg["found"] is True
    assert len(sg["links"]) >= 1

    print("\n== fusion ask ==")
    q = "鎿嶄綔绯荤粺鐨勫厛淇鏄粈涔堬紵灞炰簬鍝釜涓撲笟锛?
    result = fusion.call("ask", {"question": q, "top_k": 3})
    print(result["answer"])
    print("entities:", result["entities"])
    print("triples:", len(result["kg"]["triples"]), "hits:", len(result["rag_hits"]))
    print("confidence:", result["confidence"])
    assert result["rag_hits"] or result["kg"]["triples"]
    assert "鏉ユ簮" in result["answer"] or "鏂囨。" in result["answer"] or "鍥捐氨" in result["answer"]

    print("\nRAG+KG 铻嶅悎鍐掔儫閫氳繃")


if __name__ == "__main__":
    main()

