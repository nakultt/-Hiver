"""Surface-overlap metrics.

These are computed and reported, but they are NOT the accuracy metric. They are
here to be argued with -- `replybench validate-metric` shows them failing the
paraphrase test (a meaning-preserving rewrite of the gold reply scores badly on
ROUGE-L while the rubric holds steady) and rewarding the nearest-neighbour
baseline that a human would not send.

Keeping them in the report is the honest move. "ROUGE is bad for this" is a
claim, and the repo should contain the evidence for it rather than the assertion.
"""
from __future__ import annotations

import re

from ..schemas import LexicalScores

_WORD = re.compile(r"[a-z0-9']+")


def toks(text: str) -> list[str]:
    return _WORD.findall(text.lower())


def lcs_length(a: list[str], b: list[str]) -> int:
    if not a or not b:
        return 0
    prev = [0] * (len(b) + 1)
    for x in a:
        cur = [0]
        for j, y in enumerate(b):
            cur.append(prev[j] + 1 if x == y else max(cur[j], prev[j + 1]))
        prev = cur
    return prev[-1]


def rouge_l(candidate: str, reference: str, beta: float = 1.2) -> float:
    """LCS-based F-measure, the standard summarisation ROUGE-L."""
    c, r = toks(candidate), toks(reference)
    if not c or not r:
        return 0.0
    l = lcs_length(c, r)
    if l == 0:
        return 0.0
    p, rec = l / len(c), l / len(r)
    return ((1 + beta**2) * p * rec) / (rec + beta**2 * p)


def token_f1(candidate: str, reference: str) -> float:
    """Bag-of-words F1 with multiplicity (SQuAD-style)."""
    from collections import Counter

    c, r = Counter(toks(candidate)), Counter(toks(reference))
    overlap = sum((c & r).values())
    if overlap == 0:
        return 0.0
    p, rec = overlap / sum(c.values()), overlap / sum(r.values())
    return 2 * p * rec / (p + rec)


def edit_similarity(candidate: str, reference: str, cap: int = 3000) -> float:
    """1 - normalised Levenshtein, as a crude 'how much would I retype' proxy.

    Character-level, capped for cost. Reported but not scored: a reply can be a
    total rewrite of the reference and still be correct.
    """
    a, b = candidate[:cap], reference[:cap]
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return 1.0 - prev[-1] / max(len(a), len(b))


def score(candidate: str, reference: str) -> LexicalScores:
    c, r = toks(candidate), toks(reference)
    return LexicalScores(
        rouge_l=round(rouge_l(candidate, reference), 4),
        token_f1=round(token_f1(candidate, reference), 4),
        length_ratio=round(len(c) / len(r), 3) if r else 0.0,
        char_edit_similarity=round(edit_similarity(candidate, reference), 4),
    )
