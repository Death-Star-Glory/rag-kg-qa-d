"""三态判定（作答 / 澄清 / 拒答）—— qa 编排层的硬约束，全部为纯函数。

设计依据
--------
1. **拒答（分数闸门）**：检索 top-1 分数 < τ（0.185，来自
   `evals/vector_db/calibrate.py` 标定）→ 拒答。判定发生在任何生成动作之前，
   是硬约束（评分表「拒答与防幻觉」要求），不写进 Prompt 让 LLM 自判。
2. **澄清（字面判据）**：从 `evals/vector_db/check_clarify_rule.py` 的
   参考实现整段移植（vector_db 交接文档 §3.5.4 明确允许复制）。
   三条同时成立才澄清：分数过闸 + 含无先行词指示代词 + 锚不到任何章节标题。
3. **可支撑性检查（诱导题兜底）**：vector_db 实测诱导题分数比应答题还高
   （最高 0.452 vs 库内最低 0.208，见交接文档 §3.3），τ 拦不住。
   问句在问「具体数值 / 代码实现 / 具体步骤」而检索证据里没有对应内容时，
   判定不可支撑 → 拒答。规则刻意收窄（多少/源码/结构体/具体步骤等），
   避免误伤「哪些内容」这类正常列举题。

三态流程（组内已定口径，见 vector_db 交接文档 §3.5）：

    拒答   top1 < τ
    澄清   过闸 + 指示代词 + 锚不到章节标题
    兜底   过闸 + 问细节而证据无细节（诱导题）
    作答   其余情况

换 embedding / 改打分权重后：先重跑 vector_db 标定，再改 TAU（§3.4）。
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

# 拒答阈值：evals/vector_db/calibrate.py 标定结果（heading 分块口径）。
# 分数由 vector_store 归一化到 [0,1]，库内最低 0.208 / 无关最高 0.161，
# 分隔带 +0.047，τ=0.185 偏保守（漏答 0）。
TAU = 0.185

# ----------------------------------------------------------------------
# 澄清判据（自 evals/vector_db/check_clarify_rule.py 移植，勿改口径）
# ----------------------------------------------------------------------
# 无先行词的指示代词。命中它只是「可疑」，还要看能否锚定章节。
PRON_RE = re.compile(r"这个|那个|这些|那些|它|它们|这一块|那一块|上面|上述|刚才|前面说的")

# 泛词：出现在标题里但不指向具体章节，锚定时必须排除。
GENERIC = {"操作", "系统", "管理", "功能", "方式", "结构", "关系", "区别",
           "概念", "基本", "主要", "常见", "什么", "哪些", "怎么", "如何",
           "存储", "分配", "控制", "状态", "文件"}


def anchor_terms(titles: List[str]) -> List[str]:
    """从全部章节标题里抽「可锚定词」：去编号、去英文，切 2-gram，滤掉泛词。"""
    out = set()
    for t in titles or []:
        t = re.sub(r"^\d+(\.\d+)*\s*", "", t or "")
        t = re.sub(r"[A-Za-z]+", "", t)
        t = re.sub(r"[\s（）()、，,。.【】]+", "", t)
        out.update(t[i:i + 2] for i in range(len(t) - 1))
    return [g for g in out if g not in GENERIC]


def need_clarify(query: str, top1_score: float,
                 titles: List[str], tau: float = TAU) -> bool:
    """是否该走「澄清」而不是「作答 / 拒答」（三条同时成立才澄清）。"""
    if top1_score < tau:
        return False                                # 不过闸 → 拒答路径
    if not PRON_RE.search(query):
        return False                                # 无指示代词 → 正常作答
    terms = anchor_terms(titles)
    return not any(w in query for w in terms)       # 锚不到 → 澄清


# ----------------------------------------------------------------------
# 可支撑性检查（诱导题兜底，qa 组规则）
# ----------------------------------------------------------------------
# 问「具体数值」：多少 / 几毫秒 / 多大 等。刻意不含「哪些 / 几种」——
# 后者是正常列举题（q03「进程有哪三种基本状态」），拦了就漏答。
NUMERIC_ASK = re.compile(r"多少|几\s*(毫秒|秒|次|字节|KB|MB)|多大|通常是.{0,6}(毫秒|KB|MB|字节)")

# 问「代码 / 内核实现」：语料（os_course.txt）是概念性教材文本，
# 没有任何源码 / 结构体 / 函数内容。
IMPL_ASK = re.compile(r"源码|内核|结构体名|哪个结构体|代码实现|怎么实现|如何实现|实现细节|函数名")

# 问「具体步骤」。
STEP_ASK = re.compile(r"具体.{0,8}步骤|计算步骤|详细步骤|逐步.{0,4}(过程|流程)")

# 证据里的「数值信号」。
_NUMBER_RE = re.compile(r"\d")

# 证据里的「实现信号」。
_IMPL_SIGNAL = re.compile(r"源码|代码|结构体|函数|内核")

# 证据里的「步骤信号」。
_STEP_SIGNAL = re.compile(r"步骤|第[一二三四五六七八九十]步|[①②③④⑤⑥]")

# 章节编号（如 "1.4"、"2.3.1"）不算数值证据，清洗掉再查数字。
_SECTION_NO_RE = re.compile(r"\d+(?:\.\d+)+|\d+\s*[、.．]\s*|[（(]\d+[)）]")


def _evidence_text(hits: Sequence[Dict[str, Any]]) -> str:
    """拼接 top-k 检索块正文，并清洗章节编号（编号里的数字不是数值证据）。"""
    parts = []
    for h in hits or []:
        title = (h.get("title") or "").strip()
        content = h.get("content") or ""
        parts.append(f"{title}。{content}" if title else content)
    return _SECTION_NO_RE.sub("", "\n".join(parts))


def lacks_support(query: str,
                  hits: Sequence[Dict[str, Any]]) -> Tuple[bool, str]:
    """诱导题兜底：问句在问细节，而检索证据里没有对应内容 → 不可支撑。

    返回 (是否不可支撑, 原因)。三条规则独立，命中任一即拒答：
        1. 问数值    而证据无数字；
        2. 问代码实现 而证据无代码/结构体/函数（概念描述不算）；
        3. 问具体步骤 而证据无步骤性内容。
    """
    ev = _evidence_text(hits)
    if not ev:
        return True, "无检索证据"
    if IMPL_ASK.search(query) and not _IMPL_SIGNAL.search(ev):
        return True, "问句在问源码/内核/结构体等实现细节，课程资料只有概念性描述"
    if NUMERIC_ASK.search(query) and not _NUMBER_RE.search(ev):
        return True, "问句在问具体数值，课程资料中没有给出任何数值"
    if STEP_ASK.search(query) and not _STEP_SIGNAL.search(ev):
        return True, "问句在问具体步骤，课程资料中没有步骤性内容"
    return False, ""


# ----------------------------------------------------------------------
# 三态主入口
# ----------------------------------------------------------------------
STATE_ANSWER = "answered"
STATE_REFUSE = "refused"
STATE_CLARIFY = "clarify"


def decide(query: str,
           hits: List[Dict[str, Any]],
           titles: Optional[List[str]] = None,
           tau: float = TAU) -> Dict[str, Any]:
    """三态判定主入口。

    返回 {"state": "answered"|"refused"|"clarify", "reason": str, "top1_score": float}。
    调用方（fusion_qa）依据 state 走作答 / 澄清 / 拒答话术，判定全部在
    生成之前完成（硬约束）。
    """
    top1 = float(hits[0]["score"]) if hits else 0.0

    if not hits or top1 < tau:
        return {
            "state": STATE_REFUSE,
            "reason": f"检索最高分 {top1:.3f} 低于阈值 {tau}，知识库内无相关内容",
            "top1_score": top1,
        }

    if need_clarify(query, top1, titles or [], tau=tau):
        return {
            "state": STATE_CLARIFY,
            "reason": "问句含无先行词的指示代词，且锚定不到具体章节",
            "top1_score": top1,
        }

    unsupported, why = lacks_support(query, hits)
    if unsupported:
        return {
            "state": STATE_REFUSE,
            "reason": f"不可支撑：{why}",
            "top1_score": top1,
        }

    return {"state": STATE_ANSWER, "reason": "ok", "top1_score": top1}


# ----------------------------------------------------------------------
# 话术（评分表要求「拒答话术清晰」）
# ----------------------------------------------------------------------
def refuse_message(reason: str) -> str:
    return (
        "抱歉，课程资料中未找到足以回答该问题的内容，为避免编造，我不作答。\n"
        f"（判定依据：{reason}）\n"
        "建议：换一种问法，或确认问题是否属于《操作系统》课程资料覆盖的范围。"
    )


def clarify_message(query: str, titles: List[str]) -> str:
    heads = "、".join((titles or [])[:6])
    return (
        f"「{query}」中的指示对象（如“这些 / 那一块 / 它”）指代不明确，"
        "而课程资料中有多节内容可能相关，无法确定你想问哪一部分。\n"
        f"可参考的章节：{heads or '（无章节信息）'}\n"
        "请补充具体对象后再问，例如“进程是怎么管理的”或“内存是怎么管理的”。"
    )
