"""指纹特征工程：把一段文本转成可解释的风格特征向量。

设计原则
--------
1. **一切按长度归一化。** 文本长度在 40~250 词间随机变化，绝对计数会退化成
   「长度探测器」。公开解析实测：只用长度判断作者，balanced accuracy 约 0.096，
   而 18 选 1 的随机期望才 0.056 —— 长度确实有信息，但绝不该成为主导。
   所以这里字符级信号按「每千字符」计，词级信号按「每千词」计。
2. **只保留跨 prompt 稳定的信号。** 测试集的 prompt 全是新的，任何依赖话题词、
   专有名词的特征都会在榜上失效。这里只放与主题无关的书写习惯。
3. **全部可解释。** 每个特征名都能指到一个具体的书写习惯，便于做错误分析。
   特征名统一用 `组名.特征名` 格式，可按前缀分组统计重要性。
"""

from __future__ import annotations

import math
import re

import numpy as np

# ---------------------------------------------------------------- 特殊字符
CURLY_APOS = "\u2019"   # ’
STRAIGHT_APOS = "'"     # '
EM_DASH = "\u2014"      # —
EN_DASH = "\u2013"      # –
MINUS_SIGN = "\u2212"   # −
CURLY_LQ = "\u201c"     # “
CURLY_RQ = "\u201d"     # ”
STRAIGHT_Q = '"'
ELLIPSIS = "\u2026"     # …

# 通用标点，用于统计标点总体风格
GENERIC_PUNCT = list(";:,.?!()[]{}/%&*#`><-_+=~|")

# ---------------------------------------------------------------- 词表
# 功能词：这些词的频率几乎与主题无关，是最经典的作者归属特征。
FUNCTION_WORDS = [
    # 冠词 / 限定词
    "the", "a", "an", "this", "that", "these", "those", "each", "every",
    "all", "some", "any", "both", "few", "many", "much", "more", "most",
    "other", "another", "such", "own", "same",
    # 介词
    "of", "in", "on", "at", "to", "for", "with", "from", "by", "as",
    "into", "through", "during", "before", "after", "above", "below",
    "between", "under", "over", "about", "against", "among", "upon",
    "within", "without", "beyond", "beneath", "toward", "towards",
    # 连词
    "and", "or", "but", "if", "while", "because", "so", "than", "though",
    "although", "unless", "whereas", "whether", "nor", "yet",
    # be / have / do
    "is", "are", "was", "were", "be", "been", "being", "am",
    "has", "have", "had", "does", "do", "did",
    # 助动词与情态动词
    "will", "would", "shall", "should", "can", "could", "may", "might", "must",
    # 代词
    "i", "you", "he", "she", "it", "we", "they", "me", "him", "her", "us",
    "them", "my", "your", "his", "its", "our", "their", "myself", "itself",
    # 副词 / 话语标记
    "not", "no", "only", "also", "very", "just", "even", "still", "too",
    "then", "there", "here", "however", "therefore", "thus", "hence",
    "moreover", "furthermore", "nevertheless", "instead", "rather",
]

# 情态强度与口语化程度
MODALS = ["can", "may", "might", "could", "would", "should", "must", "will", "shall"]

# 立场副词：不同模型的"语气"偏好差异明显
HEDGES = ["perhaps", "arguably", "possibly", "probably", "likely", "seemingly",
          "apparently", "somewhat", "fairly", "relatively", "roughly", "nearly"]
BOOSTERS = ["clearly", "certainly", "definitely", "undoubtedly", "absolutely",
            "essentially", "fundamentally", "ultimately", "indeed", "remarkably",
            "notably", "importantly", "crucially", "precisely", "simply", "merely"]

# 书卷词 / "AI 腔" 词汇。注意：这些是措辞偏好，不是话题词，跨 prompt 相对稳定。
LITERARY = ["profound", "tapestry", "nuance", "nuanced", "interplay", "landscape",
            "realm", "intricate", "meticulous", "vibrant", "compelling", "subtle",
            "stark", "quiet", "silent", "echo", "resonance", "resonate", "delve",
            "embrace", "journey", "woven", "weave", "thread", "threads", "lens",
            "fabric", "canvas", "architecture", "scaffold", "paradox", "tension"]

# 句子开头的转折/顺承词
SENT_STARTERS = ["and", "but", "so", "yet", "however", "moreover", "thus",
                 "therefore", "instead", "meanwhile", "still", "then"]

# 非 ASCII 语言特有的重音字符：可暴露训练语料里的多语种倾向
ACCENTED = "öéàçüñèáöőíóúâêîôûäëïÿœæ"

_WORD_RE = re.compile(r"[A-Za-z]+(?:['\u2019][A-Za-z]+)*")
_SENT_RE = re.compile(r"[^.!?]*[.!?]+|[^.!?]+$")


def tokenize(text: str) -> list[str]:
    return _WORD_RE.findall(text)


def split_sentences(text: str) -> list[str]:
    parts = [s.strip() for s in _SENT_RE.findall(text)]
    return [s for s in parts if s]


def _char_rate(count: int, n_chars: int, scale: float = 1000.0) -> float:
    return scale * count / max(n_chars, 1)


def _word_rate(count: int, n_words: int, scale: float = 1000.0) -> float:
    return scale * count / max(n_words, 1)


def _safe_ratio(num: float, den: float) -> float:
    return num / den if den else 0.0


def features_one(text: str) -> dict[str, float]:
    """单条文本 -> 特征字典。"""
    n_chars = len(text)
    tokens = tokenize(text)
    lower = [t.lower() for t in tokens]
    n_words = len(tokens)

    sentences = split_sentences(text)
    sent_lens = [len(tokenize(s)) for s in sentences] or [0]
    n_sent = len(sentences)

    f: dict[str, float] = {}

    # ---------------------------------------------------- 标点（每千字符）
    special = {
        "curly_apos": CURLY_APOS,
        "straight_apos": STRAIGHT_APOS,
        "em_dash": EM_DASH,
        "en_dash": EN_DASH,
        "minus_sign": MINUS_SIGN,
        "curly_lq": CURLY_LQ,
        "curly_rq": CURLY_RQ,
        "straight_dq": STRAIGHT_Q,
        "ellipsis_char": ELLIPSIS,
        "nbsp": "\u00a0",
    }
    for name, ch in special.items():
        f[f"punct.{name}"] = _char_rate(text.count(ch), n_chars)

    f["punct.triple_dot"] = _char_rate(text.count("..."), n_chars)
    for ch in GENERIC_PUNCT:
        f[f"punct.g_{ch}"] = _char_rate(text.count(ch), n_chars)

    punct_total = sum(text.count(c) for c in GENERIC_PUNCT) + sum(
        text.count(c) for c in special.values()
    )
    f["punct.total_rate"] = _char_rate(punct_total, n_chars)
    f["punct.punct_to_word"] = _safe_ratio(punct_total, n_words)

    # 引号风格：弯引号占全部双引号的比例（能直接把 OpenAI 摘出来）
    curly_dq = text.count(CURLY_LQ) + text.count(CURLY_RQ)
    straight_dq = text.count(STRAIGHT_Q)
    f["punct.curly_dq_share"] = _safe_ratio(curly_dq, curly_dq + straight_dq)
    curly_ap = text.count(CURLY_APOS)
    straight_ap = text.count(STRAIGHT_APOS)
    f["punct.curly_apos_share"] = _safe_ratio(curly_ap, curly_ap + straight_ap)
    f["punct.has_curly_dq"] = 1.0 if curly_dq else 0.0
    f["punct.has_curly_apos"] = 1.0 if curly_ap else 0.0
    f["punct.has_em_dash"] = 1.0 if EM_DASH in text else 0.0

    # 排版小习惯
    f["space.double_space"] = _char_rate(len(re.findall(r"  +", text)), n_chars)
    f["space.space_before_punct"] = _char_rate(
        len(re.findall(r"\s[,.;:!?]", text)), n_chars
    )
    f["space.tabs"] = float(text.count("\t"))
    f["space.newlines"] = float(text.count("\n"))

    # ---------------------------------------------------- 词级（每千词）
    lens = [len(t) for t in tokens] or [0]
    f["word.avg_len"] = float(np.mean(lens))
    f["word.median_len"] = float(np.median(lens))
    f["word.std_len"] = float(np.std(lens))
    f["word.long8_rate"] = _word_rate(sum(1 for x in lens if x >= 8), n_words)
    f["word.long12_rate"] = _word_rate(sum(1 for x in lens if x >= 12), n_words)
    f["word.short3_rate"] = _word_rate(sum(1 for x in lens if x <= 3), n_words)
    f["word.types"] = float(len(set(lower)))
    f["word.ttr"] = _safe_ratio(len(set(lower)), n_words)
    f["word.n_words"] = float(n_words)
    f["word.n_chars"] = float(n_chars)
    f["word.avg_chars_per_word"] = _safe_ratio(n_chars, n_words)

    f["caps.titlecase_rate"] = _word_rate(
        sum(1 for t in tokens if t[:1].isupper()), n_words
    )
    f["caps.allcaps_rate"] = _word_rate(
        sum(1 for t in tokens if len(t) > 1 and t.isupper()), n_words
    )
    f["morph.accented_rate"] = _word_rate(
        sum(1 for ch in text if ch in ACCENTED), n_words
    )
    f["morph.hyphen_word_rate"] = _word_rate(
        len(re.findall(r"[A-Za-z]+-[A-Za-z]+", text)), n_words
    )
    f["morph.contraction_rate"] = _word_rate(
        sum(1 for t in lower if "'" in t or CURLY_APOS in t), n_words
    )
    f["morph.numeral_rate"] = _word_rate(
        sum(1 for t in tokens if any(c.isdigit() for c in t)), n_words
    )
    f["morph.digit_char_rate"] = _char_rate(
        sum(1 for c in text if c.isdigit()), n_chars
    )

    # ---------------------------------------------------- 句级
    f["sent.n_sent"] = float(n_sent)
    f["sent.avg_words"] = float(np.mean(sent_lens))
    f["sent.std_words"] = float(np.std(sent_lens))
    f["sent.max_words"] = float(np.max(sent_lens))
    f["sent.min_words"] = float(np.min(sent_lens))
    f["sent.per_100w"] = _word_rate(n_sent, n_words, scale=100.0)
    long_sent = sum(1 for x in sent_lens if x >= 30)
    f["sent.long30_rate"] = _safe_ratio(long_sent, n_sent)
    f["sent.question_rate"] = _safe_ratio(text.count("?"), n_sent)
    f["sent.exclaim_rate"] = _safe_ratio(text.count("!"), n_sent)
    f["sent.colon_rate"] = _safe_ratio(text.count(":"), n_sent)
    f["sent.semicolon_rate"] = _safe_ratio(text.count(";"), n_sent)

    starters = [tokenize(s)[0].lower() for s in sentences if tokenize(s)]
    if starters:
        for w in SENT_STARTERS:
            f[f"sent.start_{w}"] = _safe_ratio(sum(1 for s in starters if s == w), len(starters))
    else:
        for w in SENT_STARTERS:
            f[f"sent.start_{w}"] = 0.0

    # ---------------------------------------------------- 虚词 / 立场词（每千词）
    from collections import Counter

    freq = Counter(lower)
    for w in FUNCTION_WORDS:
        f[f"fw.{w}"] = _word_rate(freq.get(w, 0), n_words)
    for w in MODALS:
        f[f"modal.{w}"] = _word_rate(freq.get(w, 0), n_words)
    for w in HEDGES:
        f[f"hedge.{w}"] = _word_rate(freq.get(w, 0), n_words)
    for w in BOOSTERS:
        f[f"boost.{w}"] = _word_rate(freq.get(w, 0), n_words)
    for w in LITERARY:
        f[f"lit.{w}"] = _word_rate(freq.get(w, 0), n_words)

    f["hedge.total"] = sum(f[f"hedge.{w}"] for w in HEDGES)
    f["boost.total"] = sum(f[f"boost.{w}"] for w in BOOSTERS)
    f["lit.total"] = sum(f[f"lit.{w}"] for w in LITERARY)
    f["fw.total"] = sum(f[f"fw.{w}"] for w in FUNCTION_WORDS)

    # ---------------------------------------------------- Markdown / 排版残留
    f["md.bold_pairs"] = _char_rate(len(re.findall(r"\*\*", text)) / 2.0, n_chars)
    f["md.heading"] = _char_rate(len(re.findall(r"(?m)^#{1,6}\s", text)), n_chars)
    f["md.bullet"] = _char_rate(len(re.findall(r"(?m)^\s*[-*+]\s", text)), n_chars)
    f["md.numbered"] = _char_rate(len(re.findall(r"(?m)^\s*\d+[.)]\s", text)), n_chars)
    f["md.blockquote"] = _char_rate(len(re.findall(r"(?m)^\s*>\s", text)), n_chars)
    f["md.backtick"] = _char_rate(text.count("`"), n_chars)
    f["md.star_count"] = _char_rate(text.count("*"), n_chars)
    f["md.hash_count"] = _char_rate(text.count("#"), n_chars)
    f["md.dash_bullet_inline"] = _char_rate(len(re.findall(r"\s-\s", text)), n_chars)

    # 首尾习惯：是否直接给结论、最后是否有总结句
    f["shape.starts_upper"] = 1.0 if text[:1].isupper() else 0.0
    f["shape.ends_period"] = 1.0 if text.rstrip().endswith(".") else 0.0
    f["shape.ends_question"] = 1.0 if text.rstrip().endswith("?") else 0.0
    f["shape.ends_quote"] = 1.0 if text.rstrip()[-1:] in {"\u201d", '"'} else 0.0
    f["shape.log_words"] = math.log1p(n_words)

    return f


def extract_features(texts, show_progress: bool = False):
    """文本列表 -> (X, feature_names)。

    特征名顺序在所有文本间严格一致，因此可以安全地按列索引。
    """
    rows: list[dict[str, float]] = []
    for i, t in enumerate(texts):
        rows.append(features_one(t))
        if show_progress and (i + 1) % 500 == 0:
            print(f"  已处理 {i + 1}/{len(texts)}")

    names = list(rows[0].keys()) if rows else []
    X = np.asarray([[r[n] for n in names] for r in rows], dtype=np.float64)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    return X, names


def group_of(feature_name: str) -> str:
    return feature_name.split(".", 1)[0]
