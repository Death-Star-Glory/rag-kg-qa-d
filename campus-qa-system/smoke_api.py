"""API 灞傛祴璇曪細浣跨敤 FastAPI TestClient锛堟棤闇€璧疯繘绋嬶級銆?""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from fastapi.testclient import TestClient  # noqa: E402

from gateway.app import build_demo_registry, create_app  # noqa: E402


def main() -> None:
    data_dir = ROOT / "data" / "api_smoke"
    if data_dir.exists():
        import shutil

        shutil.rmtree(data_dir)
    sample = (ROOT / "data" / "sample_docs" / "cs_curriculum.txt").read_text(encoding="utf-8")
    client = TestClient(create_app(build_demo_registry(data_dir)))

    h = client.get("/health")
    assert h.status_code == 200 and h.json()["code"] == 0

    r = client.post(
        "/documents/save",
        json={"filename": "cs_curriculum.txt", "content": sample, "source": "sample"},
    )
    assert r.status_code == 200 and r.json()["code"] == 0

    r = client.post("/kg/build", json={})
    body = r.json()
    assert r.status_code == 200 and body["code"] == 0
    print("build:", body["data"]["graph"])

    r = client.get("/kg/query", params={"entity": "鏁版嵁缁撴瀯"})
    print("query:", r.json()["data"])
    assert r.json()["data"]["count"] >= 1

    r = client.post("/kg/subgraph", json={"entity": "鏁版嵁缁撴瀯", "depth": 1})
    sg = r.json()["data"]
    print("subgraph nodes/links:", len(sg["nodes"]), len(sg["links"]))
    assert sg["found"] is True

    r = client.post("/qa/ask", json={"question": "鏁版嵁缁撴瀯鐨勫厛淇鏄粈涔堬紵", "top_k": 3})
    ans = r.json()["data"]
    print("answer:\n", ans["answer"])
    print("confidence:", ans["confidence"], "sources:", len(ans["sources"]))
    assert ans["sources"]

    print("API 鍐掔儫閫氳繃")


if __name__ == "__main__":
    main()

