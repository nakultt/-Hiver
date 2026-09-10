"""Hybrid retrieval over past (email, reply) pairs and over the knowledge base.

Three rankers, fused
--------------------
  * BM25 over the cleaned, quote-stripped body -- rewards rare-term hits.
  * TF-IDF cosine -- rewards overall topical overlap, and disagrees with BM25
    usefully.
  * Dense embeddings -- the only one that survives vocabulary mismatch, where a
    customer writes "single sign on" and the corpus says "SAML SSO".

Fused with Reciprocal Rank Fusion rather than score averaging: the three live on
different scales and normalising them well is fiddly, while RRF only needs the
ordering.

Dense retrieval is an *addition*, not a replacement, because it has the opposite
failure mode to the lexical rankers -- it will happily return something topically
adjacent when the exact term was the whole point (an "annual" refund is not a
"monthly" one). It also degrades gracefully: if the embedding call fails or no
provider is configured, `self.dense` stays None and the lexical pair carries the
retrieval on its own.

The `no_retrieval` ablation measures what all of this is actually buying, which
is the honest way to defend any of it.
"""
from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Iterable, Sequence

from .. import kb
from ..schemas import Example
from .parse import clean_for_index

STOP = frozenset("""
a an and are as at be been but by can could do does did for from had has have he her his
how i if in into is it its me my no not of on or our out so than that the their them then
there these they this to too us was we were what when where which who will with would you
your ll re ve don t s hi hello hey thanks thank regards best cheers please just get got
""".split())

TOKEN = re.compile(r"[a-z0-9][a-z0-9'&.-]*")


def tokenize(text: str) -> list[str]:
    out: list[str] = []
    for raw in TOKEN.findall(text.lower()):
        w = raw.strip(".-'")
        if len(w) < 2 or w in STOP:
            continue
        # Crude suffix folding. A real stemmer is overkill for 60 documents and
        # introduces its own errors ("billing" -> "bill" -> matches "bill pay").
        for suf in ("ing", "ies", "es", "ed", "s"):
            if len(w) > 4 and w.endswith(suf):
                w = w[: -len(suf)]
                break
        out.append(w)
    return out


# ---------------------------------------------------------------------------
@dataclass
class _Doc:
    key: str
    tokens: list[str]
    tf: Counter = field(default_factory=Counter)

    def __post_init__(self) -> None:
        self.tf = Counter(self.tokens)


class BM25:
    """Okapi BM25. ~40 lines, no dependency, and fully inspectable."""

    def __init__(self, docs: Sequence[_Doc], k1: float = 1.4, b: float = 0.72):
        self.docs = list(docs)
        self.k1, self.b = k1, b
        self.n = len(self.docs) or 1
        self.avglen = sum(len(d.tokens) for d in self.docs) / self.n if self.docs else 1.0
        df: Counter = Counter()
        for d in self.docs:
            df.update(set(d.tokens))
        self.idf = {
            t: math.log(1 + (self.n - c + 0.5) / (c + 0.5)) for t, c in df.items()
        }

    def scores(self, query: Iterable[str]) -> list[float]:
        q = list(query)
        out = []
        for d in self.docs:
            dl = len(d.tokens) or 1
            s = 0.0
            for t in q:
                f = d.tf.get(t, 0)
                if not f:
                    continue
                s += self.idf.get(t, 0.0) * f * (self.k1 + 1) / (
                    f + self.k1 * (1 - self.b + self.b * dl / self.avglen)
                )
            out.append(s)
        return out


class TfidfCosine:
    """Length-normalised TF-IDF cosine, as an independent second ranker.

    It disagrees with BM25 usefully: BM25 rewards rare-term hits, cosine rewards
    overall topical overlap. Fusing them is more robust than tuning either.
    """

    def __init__(self, docs: Sequence[_Doc]):
        self.docs = list(docs)
        n = len(self.docs) or 1
        df: Counter = Counter()
        for d in self.docs:
            df.update(set(d.tokens))
        self.idf = {t: math.log(n / (1 + c)) + 1.0 for t, c in df.items()}
        self.vecs = [self._vec(d.tf) for d in self.docs]

    def _vec(self, tf: Counter) -> dict[str, float]:
        v = {t: (1 + math.log(f)) * self.idf.get(t, 1.0) for t, f in tf.items()}
        norm = math.sqrt(sum(x * x for x in v.values())) or 1.0
        return {t: x / norm for t, x in v.items()}

    def scores(self, query: Iterable[str]) -> list[float]:
        qv = self._vec(Counter(query))
        return [sum(w * dv.get(t, 0.0) for t, w in qv.items()) for dv in self.vecs]


def rrf(rankings: Sequence[Sequence[int]], k: int = 60) -> dict[int, float]:
    """Reciprocal Rank Fusion.

    Chosen over score averaging because BM25 and cosine live on different scales
    and normalising them well is fiddly; RRF only needs the ordering.
    """
    fused: dict[int, float] = {}
    for ranking in rankings:
        for rank, idx in enumerate(ranking):
            fused[idx] = fused.get(idx, 0.0) + 1.0 / (k + rank + 1)
    return fused


def cosine(a: Sequence[float], b: Sequence[float]) -> float:
    if not a or not b:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a)) or 1.0
    nb = math.sqrt(sum(y * y for y in b)) or 1.0
    return dot / (na * nb)


class DenseIndex:
    """Embedding retrieval over the exemplar corpus.

    This is the ranker that survives vocabulary mismatch: a customer writing
    "single sign on" and a corpus saying "SAML SSO" share no tokens, so BM25 and
    TF-IDF both score them near zero while cosine similarity does not. It is
    added as a THIRD ranking in the RRF fusion rather than replacing the lexical
    ones, because dense retrieval has the opposite failure mode -- it happily
    returns something topically adjacent when the exact term was the point
    (an "annual" refund vs a "monthly" one).

    Built lazily and cached on disk, so it costs one embedding call per corpus
    document once, then nothing.
    """

    def __init__(self, keys: Sequence[str], vectors: Sequence[Sequence[float]]):
        self.keys = list(keys)
        self.vectors = [list(v) for v in vectors]

    def scores(self, query_vec: Sequence[float]) -> list[float]:
        return [cosine(query_vec, v) for v in self.vectors]

    @property
    def ready(self) -> bool:
        return bool(self.vectors) and all(self.vectors)


# ---------------------------------------------------------------------------
class Retriever:
    """Retrieves exemplar (email, reply) pairs and grounding facts."""

    def __init__(self, corpus: Sequence[Example], dense: "DenseIndex | None" = None):
        self.corpus = list(corpus)
        self.dense = dense
        docs = []
        for ex in self.corpus:
            # Field weighting: the subject is short and high-signal, so it is
            # repeated to raise its effective term frequency.
            text = (ex.incoming.subject + " ") * 3 + clean_for_index(ex.incoming.body)
            docs.append(_Doc(ex.id, tokenize(text)))
        self.docs = docs
        self.bm25 = BM25(docs)
        self.cos = TfidfCosine(docs)

        fact_docs = [_Doc(f.id, tokenize(f.area + " " + f.statement)) for f in kb.PUBLIC_FACTS]
        self.fact_bm25 = BM25(fact_docs)
        self.fact_ids = [f.id for f in kb.PUBLIC_FACTS]

    # ------------------------------------------------------------- exemplars
    def search(
        self, subject: str, body: str, k: int = 4, query_vec: Sequence[float] | None = None
    ) -> list[tuple[Example, float]]:
        if not self.corpus:
            return []
        q = tokenize((subject + " ") * 3 + clean_for_index(body))
        if not q and query_vec is None:
            return []
        b = self.bm25.scores(q) if q else [0.0] * len(self.corpus)
        c = self.cos.scores(q) if q else [0.0] * len(self.corpus)
        rankings = [sorted(range(len(b)), key=lambda i: -b[i]),
                    sorted(range(len(c)), key=lambda i: -c[i])]

        d: list[float] | None = None
        if query_vec is not None and self.dense and self.dense.ready:
            d = self.dense.scores(query_vec)
            rankings.append(sorted(range(len(d)), key=lambda i: -d[i]))

        fused = rrf(rankings)
        best = sorted(fused.items(), key=lambda kv: -kv[1])[:k]
        return [(self.corpus[i], s) for i, s in best
                if b[i] > 0 or c[i] > 0 or (d is not None and d[i] > 0.3)]

    async def build_dense(self, llm) -> None:
        """Embed the corpus once. Safe to call repeatedly; cached on disk."""
        if not self.corpus:
            return
        texts = [e.incoming.subject + "\n" + clean_for_index(e.incoming.body)
                 for e in self.corpus]
        try:
            vecs = await llm.embed(texts)
            self.dense = DenseIndex([e.id for e in self.corpus], vecs)
        except Exception:  # noqa: BLE001 - dense is an enhancement, not a dependency
            self.dense = None

    async def embed_query(self, llm, subject: str, body: str):
        if self.dense is None:
            return None
        try:
            return (await llm.embed([subject + "\n" + clean_for_index(body)]))[0]
        except Exception:  # noqa: BLE001
            return None

    # ------------------------------------------------------------------ facts
    def facts(
        self, subject: str, body: str, exemplars: Sequence[Example], k: int = 14
    ) -> list[kb.Fact]:
        """Facts to ground on: lexical hits, plus whatever similar past replies cited.

        The second half matters more than the first. "What did we cite the last
        three times someone asked this?" is a better grounding signal than term
        overlap, because the customer's words rarely match policy wording.
        """
        q = tokenize(subject + " " + clean_for_index(body))
        scores = self.fact_bm25.scores(q) if q else [0.0] * len(self.fact_ids)
        ranked = sorted(range(len(self.fact_ids)), key=lambda i: -scores[i])
        chosen: list[str] = [self.fact_ids[i] for i in ranked[: max(0, k - 6)] if scores[i] > 0]

        for ex in exemplars:
            for fid in ex.reply.cited_facts + ex.key_facts:
                if fid not in chosen:
                    chosen.append(fid)

        # Pricing and plan facts are near-universally relevant in this domain and
        # cheap to include; omitting them causes plan-tier mistakes.
        for fid in ("PRICE-01", "PRICE-03", "SLA-04"):
            if fid not in chosen:
                chosen.append(fid)

        return [kb.BY_ID[f] for f in chosen if f in kb.BY_ID][: k + 8]
