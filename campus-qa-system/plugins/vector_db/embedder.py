"""embedder 插件：文本 → 定长稠密向量。

离线实现：字符 unigram + bigram 哈希 + TF 加权，零第三方依赖。
接入课程统一 embedding 接口时，只替换 vectorize() 内部，其余代码不动。

注：模块级函数（tokenize / raw_vector / normalize / cosine）供 vector_store 直接复用，
避免把 4096 维向量塞进 RPC 信封来回传。
"""

from __future__ import annotations

import hashlib
import math
import re
from typing import Any, Dict, List, Optional, Set

from base.sdk.base import Plugin, PluginMeta

DEFAULT_DIM = 4096
# 需剔除的噪声：空白、数字、中英文标点。标点必须剔干净，
# 否则 analyze() 会把"？"当成语料外的字，污染 OOV 比。
_CLEAN_RE = re.compile(
    r"[\s0-9，。；：、（）()【】《》〈〉\[\]\{\}\-—－_/\\·.\"'“”‘’？?！!～~…]+"
)
CLEAN_RE = _CLEAN_RE


def tokenize(text: str) -> List[str]:
    """中文按字 + 相邻二元组；数字标点已剔除。"""
    clean = _CLEAN_RE.sub("", text)
    if not clean:
        return []
    return list(clean) + [clean[i : i + 2] for i in range(len(clean) - 1)]


def _bucket(token: str, dim: int) -> int:
    h = hashlib.md5(token.encode("utf-8")).digest()
    return int.from_bytes(h[:4], "big") % dim


def raw_vector(text: str, dim: int = DEFAULT_DIM, vocab: Optional[Set[str]] = None) -> List[float]:
    """未归一化的 TF 向量。

    vocab 非空时只保留语料词表内的 token：问句里的疑问词与未登录词在语料中
    不存在，不过滤会把有效信号稀释掉。
    """
    vec = [0.0] * dim
    toks = tokenize(text)
    if vocab is not None:
        toks = [t for t in toks if t in vocab]
    if not toks:
        return vec
    counts: Dict[str, int] = {}
    for t in toks:
        counts[t] = counts.get(t, 0) + 1
    for t, c in counts.items():
        # bigram 权重高于单字：bigram 携带词序信息，区分度更高
        w = 2.0 if len(t) == 2 else 1.0
        vec[_bucket(t, dim)] += w * (1.0 + math.log(c))
    return vec


def normalize(vec: List[float]) -> List[float]:
    """L2 归一化。归一化后点积即余弦相似度。"""
    norm = math.sqrt(sum(v * v for v in vec))
    if norm <= 0:
        return vec
    return [v / norm for v in vec]


def cosine(a: List[float], b: List[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def vectorize(
    text: str,
    dim: int = DEFAULT_DIM,
    idf: Optional[List[float]] = None,
    vocab: Optional[Set[str]] = None,
) -> List[float]:
    raw = raw_vector(text, dim, vocab)
    if idf is not None:
        raw = [x * idf[i] for i, x in enumerate(raw)]
    return normalize(raw)


class EmbedderPlugin(Plugin):
    meta = PluginMeta(
        name="embedder",
        version="0.1.0",
        description="文本向量化：字符 n-gram 哈希 + TF（离线实现，可换课程统一接口）",
        group="vector_db",
    )

    def __init__(self) -> None:
        self._dim = DEFAULT_DIM

    def initialize(self, config: Dict[str, Any] | None = None) -> None:
        cfg = config or {}
        self._dim = int(cfg.get("dim", DEFAULT_DIM))

    def methods(self) -> List[str]:
        return ["embed", "dim", "info"]

    def call(self, method: str, params: Dict[str, Any]) -> Any:
        if method == "info":
            return self.info()
        if method == "dim":
            return {"dim": self._dim}
        if method == "embed":
            text = params.get("text", "")
            return {"dim": self._dim, "vector": vectorize(text, self._dim)}
        raise KeyError(method)
