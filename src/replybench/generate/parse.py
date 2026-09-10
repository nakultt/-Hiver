"""Get the actual message out of a raw email body.

Deterministic and cheap on purpose: this runs before retrieval, and if you index
a 200-word legal disclaimer you retrieve on the disclaimer. It is also the piece
that decides whether the `buried-ask-top-posted` hard case is answerable at all.

Kept rule-based rather than LLM-based because it is called on every email, must
be fast, and is trivially unit-testable -- see tests/test_parse.py.
"""
from __future__ import annotations

import re

QUOTE_HEADER = re.compile(
    r"^\s*(On .{5,120}wrote:|-{2,}\s*Original Message\s*-{2,}|_{5,}|"
    r"-{3,}\s*Forwarded message\s*-{3,}|From:\s.+)\s*$",
    re.I | re.M,
)

SIG_MARKERS = (
    re.compile(r"^\s*--\s*$", re.M),
    re.compile(r"^\s*Sent from my \w+", re.I | re.M),
    re.compile(r"^\s*(Get|Sent from) Outlook", re.I | re.M),
    re.compile(r"^\s*sent from mobile", re.I | re.M),
)

DISCLAIMER = re.compile(
    r"(this email and any attachments are confidential|the information contained in this "
    r"communication|please consider the environment|accepts no liability|if you have "
    r"received this in error)",
    re.I,
)

CONTACT_LINE = re.compile(r"^\s*(M|T|Tel|Mob|Phone|D|E)\s*[:.]\s*\+?[\d\s()-]{7,}$", re.I | re.M)

GREETING = re.compile(r"^\s*(hi|hello|hey|dear|good (morning|afternoon|evening))\b[^\n]{0,40}$", re.I)


def strip_quotes(body: str) -> str:
    """Drop quoted history.

    Note the ordering problem this solves: in a top-posted thread the *new*
    content is above the quote, in a bottom-posted one it is below. We keep
    everything before the first quote header and drop '>'-prefixed lines
    wherever they appear, which handles both.
    """
    m = QUOTE_HEADER.search(body)
    head = body[: m.start()] if m else body
    lines = [ln for ln in head.splitlines() if not ln.lstrip().startswith(">")]
    return "\n".join(lines)


def strip_signature(body: str) -> str:
    cut = len(body)
    for marker in SIG_MARKERS:
        m = marker.search(body)
        if m:
            cut = min(cut, m.start())
    d = DISCLAIMER.search(body)
    if d:
        line_start = body.rfind("\n", 0, d.start()) + 1
        cut = min(cut, line_start)
    body = body[:cut]
    body = CONTACT_LINE.sub("", body)
    return body


def clean(body: str) -> str:
    """Full clean: quotes out, signature out, whitespace normalised."""
    out = strip_signature(strip_quotes(body))
    out = re.sub(r"\n{3,}", "\n\n", out)
    return out.strip()


def clean_for_index(body: str) -> str:
    """What the retriever indexes: cleaned, minus the greeting/sign-off scaffolding.

    Greetings and names are pure noise for lexical matching -- every email has
    them, so they inflate similarity between unrelated messages.
    """
    text = clean(body)
    lines = [ln for ln in text.splitlines() if not GREETING.match(ln)]
    text = "\n".join(lines)
    text = re.sub(
        r"^\s*(thanks|thank you|cheers|regards|best|kind regards|many thanks|best wishes)[,!.]?\s*$",
        "", text, flags=re.I | re.M,
    )
    return re.sub(r"\s+", " ", text).strip()


def looks_truncated(text: str) -> bool:
    t = text.rstrip()
    return bool(t) and t[-1] not in ".!?)\"'" and not t.endswith("Support")
