"""System-level reporting: overall scores, uncertainty, and what actually broke."""
from __future__ import annotations

import random
from collections import Counter
from statistics import mean as _mean


def mean(vals):
    """Mean that returns 0.0 for an empty sequence.

    Failed generations legitimately produce empty dimension lists, and a
    reporting crash is a worse outcome than a zero.
    """
    vals = list(vals)
    return _mean(vals) if vals else 0.0
from typing import Sequence

from ..schemas import Example, ResponseScore
from ..taxonomy import DIMENSIONS


def bootstrap_ci(
    values: Sequence[float], *, iters: int = 2000, alpha: float = 0.05, seed: int = 7
) -> tuple[float, float]:
    """Percentile bootstrap CI for the mean.

    Reported on every headline number because at n=18 the difference between two
    systems scoring 0.71 and 0.74 is noise, and a benchmark that hides that is
    worse than no benchmark.
    """
    if not values:
        return (0.0, 0.0)
    rng = random.Random(seed)
    n = len(values)
    means = []
    for _ in range(iters):
        means.append(mean(rng.choices(list(values), k=n)))
    means.sort()
    lo = means[int(alpha / 2 * iters)]
    hi = means[min(iters - 1, int((1 - alpha / 2) * iters))]
    return (round(lo, 4), round(hi, 4))


def spearman(xs: Sequence[float], ys: Sequence[float]) -> float:
    """Rank correlation, no scipy."""
    n = len(xs)
    if n < 3:
        return 0.0

    def ranks(v: Sequence[float]) -> list[float]:
        order = sorted(range(n), key=lambda i: v[i])
        r = [0.0] * n
        i = 0
        while i < n:
            j = i
            while j + 1 < n and v[order[j + 1]] == v[order[i]]:
                j += 1
            avg = (i + j) / 2 + 1
            for k in range(i, j + 1):
                r[order[k]] = avg
            i = j + 1
        return r

    rx, ry = ranks(xs), ranks(ys)
    mx, my = mean(rx), mean(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = (sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry)) ** 0.5
    return round(num / den, 4) if den else 0.0


def summarise_system(scores: Sequence[ResponseScore]) -> dict:
    if not scores:
        return {}
    comps = [s.composite for s in scores]
    out = {
        "n": len(scores),
        "composite": round(mean(comps), 4),
        "composite_ci95": bootstrap_ci(comps),
        "dimensions": {
            d: round(mean([s.dimensions[d].score for s in scores if d in s.dimensions]), 4)
            for d in DIMENSIONS
            if any(d in s.dimensions for s in scores)
        },
        "readiness": dict(Counter(s.readiness for s in scores)),
        "sendable_rate": round(
            sum(1 for s in scores if s.readiness in ("send_as_is", "light_edit")) / len(scores), 4),
        "hard_failure_rate": round(sum(1 for s in scores if s.hard_failures) / len(scores), 4),
        "breach_rate": round(sum(1 for s in scores if s.breaches) / len(scores), 4),
        "consequential_error_rate": round(
            sum(1 for s in scores if s.consequential_errors) / len(scores), 4),
        "rouge_l": round(mean([s.lexical.rouge_l for s in scores if s.lexical]), 4),
    }
    return out


def failure_taxonomy(scores: Sequence[ResponseScore], examples: Sequence[Example]) -> dict:
    """What broke, ranked. The part a team would actually act on."""
    by_id = {e.id: e for e in examples}
    hard: Counter = Counter()
    warn: Counter = Counter()
    traps: Counter = Counter()
    missing: Counter = Counter()
    extra: Counter = Counter()
    ignored_asks = 0
    total_asks = 0

    for s in scores:
        for h in s.hard_failures:
            hard[h.split(":")[0]] += 1
        for w in s.warnings:
            warn[w.split(":")[0]] += 1
        for a in s.missing_actions:
            missing[a] += 1
        for a in s.extra_actions:
            extra[a] += 1
        for a in s.asks:
            total_asks += 1
            if a.status == "ignored":
                ignored_asks += 1
        ex = by_id.get(s.example_id)
        if ex and ex.trap and s.composite < 0.65:
            traps[ex.trap] += 1

    return {
        "hard_failures": hard.most_common(8),
        "warnings": warn.most_common(8),
        "most_missed_actions": missing.most_common(8),
        "most_over_claimed_actions": extra.most_common(8),
        "traps_fallen_for": traps.most_common(10),
        "ignored_ask_rate": round(ignored_asks / total_asks, 4) if total_asks else 0.0,
    }


def full_report(scores: Sequence[ResponseScore], examples: Sequence[Example]) -> dict:
    by_id = {e.id: e for e in examples}
    systems = sorted({s.system for s in scores})
    report: dict = {"systems": {}, "by_split": {}, "metric_diagnostics": {}}

    for sysname in systems:
        rows = [s for s in scores if s.system == sysname]
        report["systems"][sysname] = summarise_system(rows)
        report["systems"][sysname]["failures"] = failure_taxonomy(rows, examples)

    for split in ("test", "hard"):
        report["by_split"][split] = {
            sysname: summarise_system(
                [s for s in scores if s.system == sysname
                 and by_id.get(s.example_id) and by_id[s.example_id].split == split])
            for sysname in systems
        }

    # Does the cheap surface metric agree with the real one? (Spoiler in README.)
    main = [s for s in scores if s.system == "main" and s.lexical]
    allrows = [s for s in scores if s.lexical]
    report["metric_diagnostics"] = {
        "spearman_rouge_vs_composite_main": spearman(
            [s.lexical.rouge_l for s in main], [s.composite for s in main]),
        "spearman_rouge_vs_composite_all": spearman(
            [s.lexical.rouge_l for s in allrows], [s.composite for s in allrows]),
        "spearman_length_vs_composite": spearman(
            [s.lexical.length_ratio for s in allrows], [s.composite for s in allrows]),
        "note": (
            "length_vs_composite is a verbosity-bias probe: a judge that rewards long "
            "replies shows up here as a strong positive rank correlation."
        ),
    }

    # Human-written gold (the hand-authored hard split) vs model-written gold
    # (test split). A large gap is evidence the synthetic references are biased.
    for sysname in systems:
        t = report["by_split"]["test"].get(sysname, {}).get("composite")
        h = report["by_split"]["hard"].get(sysname, {}).get("composite")
        if t is not None and h is not None:
            report["systems"][sysname]["model_gold_minus_human_gold"] = round(t - h, 4)

    return report
