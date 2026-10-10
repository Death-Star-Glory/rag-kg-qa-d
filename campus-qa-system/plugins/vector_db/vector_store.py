"""vector_store 插件：向量索引与相似检索。

职责边界（交接文档第 2 节）：只做向量化、入库、相似检索；
不解析文档、不生成答案。拒答判定属编排层（qa），本插件只提供分数与尺度说明。
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Set

from base.sdk.base import Plugin, PluginMeta
from plugins.vector_db.embedder import (
    CLEAN_RE,
    DEFAULT_DIM,
    cosine,
    normalize,
    raw_vector,
    tokenize,
)

STOPWORDS: Set[str] = set(
    "的了吗呢吧啊是和与或及在有为对以从到把被这那有请帮我讲讲一下什么是哪些哪个"
    "怎么怎么样如何为什么能否可以需要它他她你们我以及其中就是还要会能"
)


class VectorStorePlugin(Plugin):
    meta = PluginMeta(
        name="vector_store",
        version="0.1.0",
        description="向量索引与相似检索：TF-IDF 余弦 + 关键词覆盖混合打分",
        group="vector_db",
    )

    def __init__(self) -> None:
        self._dim = DEFAULT_DIM
        self._chunks: List[Dict[str, Any]] = []
        self._idf: Optional[List[float]] = None
        self._vocab: Optional[Set[str]] = None
        self._vocab_chars: Set[str] = set()
        self._w_cos = 0.75
        self._w_cov = 0.25
        self._embedder = None

    def initialize(self, config: Dict[str, Any] | None = None) -> None:
        cfg = config or {}
        self._dim = int(cfg.get("dim", DEFAULT_DIM))
        self._w_cos = float(cfg.get("w_cos", 0.75))
        self._w_cov = float(cfg.get("w_cov", 0.25))
        self._embedder = cfg.get("embedder")
        self._vocab_chars: Set[str] = set()
        if cfg.get("chunks"):
            self.index(cfg["chunks"])

    def methods(self) -> List[str]:
        return ["index", "search", "stats", "score_scale", "analyze", "info"]

    def call(self, method: str, params: Dict[str, Any]) -> Any:
        if method == "info":
            return self.info()
        if method == "index":
            return self.index(params.get("chunks") or [])
        if method == "search":
            return self.search(
                params.get("query", ""), int(params.get("top_k", 3))
            )
        if method == "stats":
            return self.stats()
        if method == "score_scale":
            return self.score_scale()
        if method == "analyze":
            return self.analyze(params.get("query", ""))
        raise KeyError(method)

    # ------------------------------------------------------------------
    # 索引
    # ------------------------------------------------------------------
    @staticmethod
    def _index_text(chunk: Dict[str, Any]) -> str:
        """索引文本 = 标题 + 正文。

        标题是分块的语义摘要，纳入向量后"什么是X"这类问句更容易命中对应章节。
        现有 doc_parser 不产出 title 字段，缺省时退化为纯正文。
        """
        title = (chunk.get("title") or "").strip()
        content = chunk.get("content") or ""
        return f"{title}。{content}" if title else content

    def index(self, chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """建索引：统计语料 IDF 与词表，把每个分块转成归一化 TF-IDF 向量。"""
        self._chunks = list(chunks)
        texts = [self._index_text(c) for c in self._chunks]
        raws = [raw_vector(t, self._dim) for t in texts]
        n = len(raws)

        df = [0] * self._dim
        for v in raws:
            for i, x in enumerate(v):
                if x > 0:
                    df[i] += 1
        # 平滑 IDF：只出现在少数分块里的词权重更高，
        # 用来压制"进程"这类在所有分块都出现的高频泛词
        idf = [math.log((n + 1) / (d + 1)) + 1.0 for d in df]

        for ch, v in zip(self._chunks, raws):
            ch["_vec"] = normalize([x * idf[i] for i, x in enumerate(v)])
        self._idf = idf

        vocab: Set[str] = set()
        for t in texts:
            vocab.update(tokenize(t))
        self._vocab = vocab
        self._vocab_chars = {c for c in vocab if len(c) == 1}

        return {"indexed": n, "dim": self._dim, "vocab_size": len(vocab)}

    # ------------------------------------------------------------------
    # 检索
    # ------------------------------------------------------------------
    def search(self, query: str, top_k: int = 3) -> Dict[str, Any]:
        if not self._chunks or self._idf is None:
            return {
                "query": query,
                "top_k": top_k,
                "hits": [],
                "indexed": 0,
                "score_scale": self.score_scale(),
            }

        qv = normalize(
            [
                x * self._idf[i]
                for i, x in enumerate(raw_vector(query, self._dim, self._vocab))
            ]
        )
        q_terms = {
            c for c in (query or "") if c not in STOPWORDS and not c.isspace()
        }

        hits: List[Dict[str, Any]] = []
        for ch in self._chunks:
            cos = cosine(qv, ch["_vec"])
            cov = 0.0
            if q_terms:
                content = ch.get("content") or ""
                cov = sum(1 for t in q_terms if t in content) / len(q_terms)
            score = self._w_cos * cos + self._w_cov * cov
            public = {k: v for k, v in ch.items() if not k.startswith("_")}
            public["score"] = round(score, 4)
            public["cos"] = round(cos, 4)
            public["cov"] = round(cov, 3)
            hits.append(public)

        hits.sort(key=lambda x: x["score"], reverse=True)
        return {
            "query": query,
            "top_k": top_k,
            "hits": hits[: max(1, top_k)],
            "indexed": len(self._chunks),
            "score_scale": self.score_scale(),
        }

    def stats(self) -> Dict[str, Any]:
        """索引统计。

        `titles` 是各分块的章节标题（去重、保序）—— 给编排层做「澄清判定」用：
        问句能否锚定到某个章节标题，决定它该作答还是该追问。详见
        evals/vector_db/check_clarify_rule.py 与 docs/vector_db交接文档.md 第 3.5 节。
        """
        titles: List[str] = []
        for c in self._chunks:
            t = (c.get("title") or "").strip()
            if t and t not in titles:
                titles.append(t)
        return {
            "indexed": len(self._chunks),
            "dim": self._dim,
            "vocab_size": len(self._vocab) if self._vocab else 0,
            "weights": {"cos": self._w_cos, "cov": self._w_cov},
            "titles": titles,
        }

    def analyze(self, query: str) -> Dict[str, Any]:
        """问句可答性信号 —— 供编排层做拒答的第二判据。

        Why 需要它：纯分数阈值拦不住诱导题。问句与语料字面高度重叠
        （"进程控制块 PCB"），但语料没有问句真正要的东西（"Linux 内核源码对应哪个结构体"），
        字面向量照样给高分（实测 0.452，高于库内最低分 0.208）。

        本方法给出一个与分数正交的信号：
            oov_ratio —— 问句实词中不在语料词表的单字占比。
                         语料外的实体/术语越多，越可能是问语料没写的内容
                         （如"毫秒""内核""步骤"）。
        已知局限（实测数据说话，不美化）：
            对「残缺但指向明确」的口语问句（如"调度相关的东西"）会误报，
            因为"东西""相关"这类词本就不在技术语料里。故它只能作为
            辅助判据，不能单独决定拒答。
        """
        chars = [
            c
            for c in CLEAN_RE.sub("", query or "")
            if c not in STOPWORDS
        ]
        if not chars:
            return {"oov_ratio": 0.0, "oov_chars": [], "total_chars": 0, "indexed": len(self._chunks)}
        if not self._vocab_chars:
            return {"oov_ratio": 1.0, "oov_chars": sorted(set(chars)), "total_chars": len(chars), "indexed": 0}
        oov = sorted({c for c in chars if c not in self._vocab_chars})
        return {
            "oov_ratio": round(len([c for c in chars if c not in self._vocab_chars]) / len(chars), 4),
            "oov_chars": oov,
            "total_chars": len(chars),
            "indexed": len(self._chunks),
        }

    def score_scale(self) -> Dict[str, Any]:
        """分数尺度说明。编排层做拒答判定前必须读这个。"""
        return {
            "formula": f"{self._w_cos} * cosine + {self._w_cov} * keyword_coverage",
            "range": [0.0, 1.0],
            "note": (
                "分数尺度由本插件的 embedding 实现决定。更换 embedding 或调整权重后，"
                "编排层的拒答阈值必须重新标定，否则会出现全拒或全不拒。"
            ),
        }
