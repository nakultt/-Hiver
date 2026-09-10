"""Hybrid retrieval over past (email, reply) pairs and over the knowledge base.

Why not embeddings?
-------------------
The obvious move is a dense vector store. I did not use one, and the reason is
worth stating because it is a trade-off rather than a shortcut:

  * The Gemini free tier gives no embedding quota worth using, and adding a
    second provider for embeddings would make the repo harder to run, which is
    the one thing a take-home must not be.
  * At this corpus size (~60 exemplars) lexical retrieval is genuinely
    competitive. Dense retrieval earns its keep at 10^4-10^6 documents, where
    vocabulary mismatch dominates; at 10^1-10^2 it mostly adds latency.
  * The failure mode that actually matters here -- a customer saying "single
    sign on" when the corpus says "SAML SSO" -- is handled by the *field-weighted*
    BM25 below plus an intent prior, not by cosine distance.

So: BM25 over the cleaned email body, TF-IDF cosine as a second opinion, fused
with Reciprocal Rank Fusion. `EmbeddingBackend` is left as a stub with the
interface it would need, so swapping one in is a contained change rather than a
rewrite. The ablation `--system no_retrieval` measures what retrieval is
actually buying, which is the honest way to defend any of this.
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


class EmbeddingBackend:
    """Interface a dense retriever would implement. Deliberately not implemented.

    Documented rather than half-built so the trade-off in this module's docstring
    is verifiable: swapping in dense retrieval means implementing `encode` and
    adding one more ranking to the RRF fusion, nothing else.
    """

    def encode(self, texts: Sequence[str]) -> list[list[float]]:
        raise NotImplementedError("no embedding provider configured; see module docstring")


# ---------------------------------------------------------------------------
class Retriever:
    """Retrieves exemplar (email, reply) pairs and grounding facts."""

    def __init__(self, corpus: Sequence[Example]):
        self.corpus = list(corpus)
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
    def search(self, subject: str, body: str, k: int = 4) -> list[tuple[Example, float]]:
        if not self.corpus:
            return []
        q = tokenize((subject + " ") * 3 + clean_for_index(body))
        if not q:
            return []
        b = self.bm25.scores(q)
        c = self.cos.scores(q)
        rank_b = sorted(range(len(b)), key=lambda i: -b[i])
        rank_c = sorted(range(len(c)), key=lambda i: -c[i])
        fused = rrf([rank_b, rank_c])
        best = sorted(fused.items(), key=lambda kv: -kv[1])[:k]
        return [(self.corpus[i], s) for i, s in best if b[i] > 0 or c[i] > 0]

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
