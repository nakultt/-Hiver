"""Tier 0: checks that need no model at all.

These are free, instant, perfectly reproducible, and they catch the failures that
are *embarrassing* rather than subtle -- an unfilled `[customer name]`, a leaked
internal policy line, "As an AI language model", a fact ID printed at the
customer. An LLM judge will usually catch these too, but paying a model to find a
literal "TODO" is a bad trade, and a deterministic check cannot have an off day.

The numeric audit is the interesting one. Every figure in a support reply -- a
price, a percentage, a refund window, a rate limit -- is checkable against the
knowledge base with a regex. That gives a hallucination detector for the most
expensive class of error with no model in the loop, which in turn gives us
something to validate the LLM judge's factuality score *against*.
"""
from __future__ import annotations

import re

from .. import kb

PLACEHOLDER = re.compile(r"(\[[A-Za-z _/]{2,30}\]|\{\{?[a-z_]{2,30}\}?\}|<[a-z_]{2,20}>|\bTODO\b|\bXXX\b|\bTBD\b)")
META = re.compile(
    r"(as an ai|as a language model|i'm an ai|here('s| is) (the|a) (draft|reply|response)|"
    r"i hope this helps!?$|certainly!|sure thing!|below is the)",
    re.I,
)
FACT_ID = re.compile(r"\b(?:PRICE|TRIAL|REF|CANCEL|SLA|SEC|INT|DATA|API|ONB|MOB|BILL|AUTH)-\d{2}\b")
URL = re.compile(r"https?://[^\s)>\]]+")
SIGNOFF = re.compile(r"(kestrel support|priya|regards|best,|thanks,|cheers)", re.I)

# Money, percentages, and bare counts with a unit that matters in this domain.
NUMERIC = re.compile(
    r"(?:[$£€]\s?\d[\d,]*(?:\.\d+)?)"
    r"|(?:\d+(?:\.\d+)?\s?%)"
    r"|(?:\b\d[\d,]*(?:\.\d+)?\s?(?:days?|day|hours?|months?|business days?|"
    r"seats?|users?|rows?|requests?/minute|requests? per minute|minutes?)\b)",
    re.I,
)

BANNED = (
    "we apologise for any inconvenience caused",
    "we apologize for any inconvenience caused",
    "please do not hesitate to contact us",
    "valued customer",
)


def _norm_num(s: str) -> str:
    return re.sub(r"[\s,]", "", s.lower()).replace("£", "").replace("$", "").replace("€", "")


def _canon(raw: str) -> str:
    """Canonical form of a number so 5,000 / 5000 / 5000.0 compare equal."""
    try:
        v = float(raw.replace(",", "").replace("$", "").replace("£", "").replace("€", "").strip())
    except ValueError:
        return raw.strip()
    return str(int(v)) if v == int(v) else str(v)


def _kb_numeric_vocabulary() -> set[str]:
    vocab: set[str] = set()
    for f in kb.FACTS:
        for m in re.findall(r"\d[\d,]*(?:\.\d+)?", f.statement):
            vocab.add(_canon(m))
    return vocab


_KB_NUMS = _kb_numeric_vocabulary()


def unsupported_numbers(reply: str, incoming: str) -> list[str]:
    """Figures in the reply that appear neither in the KB nor in the customer's email.

    Deliberately permissive: dates, times of day, ordinals and small integers are
    ignored, because "by Thursday" and "3 quick questions" are not factual claims
    about policy. What it catches is "$450", "60 days", "99.99%".
    """
    src = {_canon(m) for m in re.findall(r"\d[\d,]*(?:\.\d+)?", incoming)}
    bad = []
    for m in NUMERIC.findall(reply):
        raw = re.sub(r"[^\d.,]", "", m)
        if not raw:
            continue
        canon = _canon(raw)
        try:
            val = float(canon)
        except ValueError:
            continue
        if val <= 5 and "%" not in m:
            continue  # "3 questions", "2 sites" -- not policy claims
        # Exact match only. Prefix matching would let "99.99%" pass because the
        # KB contains "99.9", which is precisely the hallucination this exists
        # to catch.
        if canon in _KB_NUMS or canon in src:
            continue
        bad.append(m.strip())
    return sorted(set(bad))


def run(reply: str, incoming: str) -> tuple[list[str], list[str]]:
    """Return (hard_failures, warnings).

    A hard failure means "do not put this in front of a customer", and it caps
    the composite score regardless of how well the reply scores elsewhere. A
    fluent, warm, perfectly-toned reply that leaks an internal policy line is not
    a 0.85 with a note; it is unsendable.
    """
    hard: list[str] = []
    warn: list[str] = []
    body = reply.strip()

    if not body:
        return (["empty_reply"], [])
    if len(body) < 40:
        hard.append("reply_too_short")
    if len(body) > 4000:
        warn.append("reply_very_long")

    if PLACEHOLDER.search(body):
        hard.append("unresolved_placeholder:" + PLACEHOLDER.search(body).group(0)[:40])
    if META.search(body):
        hard.append("meta_commentary:" + META.search(body).group(0)[:40])
    if FACT_ID.search(body):
        hard.append("leaked_fact_id:" + FACT_ID.search(body).group(0))

    low = body.lower()
    for fact in kb.INTERNAL_FACTS:
        frag = fact.statement.lower()[:45]
        if frag in low:
            hard.append("leaked_internal_policy:" + fact.id)
            break

    for url in URL.findall(body):
        if "kestrelhq.com" not in url:
            hard.append("external_url:" + url[:60])

    nums = unsupported_numbers(body, incoming)
    if nums:
        warn.append("unsupported_numbers:" + ",".join(nums[:5]))

    for phrase in BANNED:
        if phrase in low:
            warn.append("banned_phrase:" + phrase)

    if not SIGNOFF.search(body):
        warn.append("no_signoff")

    sentences = [s.strip() for s in re.split(r"[.!?]\s+", body) if len(s.strip()) > 25]
    if len(sentences) != len(set(sentences)):
        warn.append("repeated_sentence")

    return hard, warn
