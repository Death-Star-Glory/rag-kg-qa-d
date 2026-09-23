"""向量数据库组：embedder、vector_store。

只做向量化与相似检索，不解析文档、不生成答案（交接文档第 2 节边界表）。
"""

from plugins.vector_db.embedder import EmbedderPlugin
from plugins.vector_db.vector_store import VectorStorePlugin

__all__ = ["EmbedderPlugin", "VectorStorePlugin"]
