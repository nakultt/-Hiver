"""Does the metric agree with a person?

Three questions, in increasing order of how much they matter:

  1. **Correlation.** Does the composite move with the human score? Spearman and
     Kendall, because we care about ordering, not absolute calibration -- nobody
     needs the metric to output 0.55 when a human says 0.55, they need it to put
     the 0.55 reply below the 0.7 one.
  2. **Pairwise ranking accuracy.** Of all pairs of replies a human ranked
     differently, how often does the metric agree? This is the number that maps
     onto the actual product decision ("show suggestion A or B"), and it is
     robust to any monotone rescaling of either side.
  3. **Are the weights right?** The six weights were a judgement call. Here they
     are re-fitted to maximise agreement with the human labels, and compared
     against the hand-set priors. If the fitted weights are wildly different, the
     priors were wrong and the README should say so.

Caveat that cannot be repeated too often: n is small and there is exactly one
annotator. This tells you whether the metric is *broken*. It does not tell you
it is *right*.
"""
from __future__ import annotations

import itertools
import random
from statistics import mean
from typing import Sequence

from ..evaluate.aggregate import spearman
from ..evaluate.audit import audit as run_audit
from ..evaluate.judge import judge as run_judge
from ..evaluate.score import build_score
from ..llm import LLM
from ..schemas import Example, GenerationRecord, HumanLabel, SuggestedReply
from ..taxonomy import DEFAULT_WEIGHTS


def kendall_tau(xs: Sequence[float], ys: Sequence[float]) -> float:
    """Tau-b, handling ties (which hand-assigned scores have plenty of)."""
    n = len(xs)
    conc = disc = tx = ty = 0
    for i, j in itertools.combinations(range(n), 2):
        dx, dy = xs[i] - xs[j], ys[i] - ys[j]
        if dx == 0 and dy == 0:
            tx += 1; ty += 1; continue
        if dx == 0:
            tx += 1; continue
        if dy == 0:
            ty += 1; continue
        if (dx > 0) == (dy > 0):
            conc += 1
        else:
            disc += 1
    denom = ((conc + disc + tx) * (conc + disc + ty)) ** 0.5
    return round((conc - disc) / denom, 4) if denom else 0.0


def pairwise_accuracy(human: Sequence[float], metric: Sequence[float], eps: float = 0.05) -> dict:
    """Of pairs a human clearly ranked, how often does the metric agree?

    `eps` skips pairs the human scored within a hair of each other; those carry
    no signal and would just dilute the number in whichever direction noise fell.
    """
    agree = total = 0
    for i, j in itertools.combinations(range(len(human)), 2):
        dh = human[i] - human[j]
        if abs(dh) < eps:
            continue
        total += 1
        dm = metric[i] - metric[j]
        if (dh > 0) == (dm > 0) and dm != 0:
            agree += 1
    return {"pairs_compared": total,
            "accuracy": round(agree / total, 4) if total else 0.0,
            "note": "chance is 0.50; below ~0.65 the metric is not usable for ranking"}


def fit_weights(
    per_dim: Sequence[dict[str, float]], human: Sequence[float], *, iters: int = 4000, seed: int = 3
) -> dict:
    """Random search over the simplex for weights maximising Spearman with human.

    Random search rather than a closed-form fit because the objective is a rank
    correlation (non-differentiable) and there are six parameters -- this is
    seconds of compute and needs no optimiser dependency. With n this small the
    fitted weights are themselves noisy, which is exactly why the comparison to
    the prior matters more than the fitted values.
    """
    keys = list(DEFAULT_WEIGHTS)
    rng = random.Random(seed)

    def score_with(w: dict[str, float]) -> float:
        vals = [sum(d.get(k, 0.0) * w[k] for k in keys) for d in per_dim]
        return spearman(vals, list(human))

    best_w = dict(DEFAULT_WEIGHTS)
    best = score_with(best_w)
    prior = best
    for _ in range(iters):
        raw = [rng.random() for _ in keys]
        tot = sum(raw) or 1.0
        w = {k: v / tot for k, v in zip(keys, raw)}
        s = score_with(w)
        if s > best:
            best, best_w = s, w

    return {
        "spearman_with_default_weights": round(prior, 4),
        "spearman_with_fitted_weights": round(best, 4),
        "fitted_weights": {k: round(v, 3) for k, v in sorted(best_w.items(), key=lambda kv: -kv[1])},
        "default_weights": DEFAULT_WEIGHTS,
        "improvement": round(best - prior, 4),
        "verdict": (
            "priors are close to optimal on this set"
            if best - prior < 0.08 else
            "fitted weights beat the priors materially -- treat the defaults as suspect"
        ),
    }


async def run(
    llm: LLM, examples: Sequence[Example], labels: Sequence[HumanLabel], *, judge_model: str | None = None
) -> dict:
    """Score every hand-labelled candidate with the full metric, then compare."""
    by_id = {e.id: e for e in examples}
    rows: list[dict] = []

    for lab in labels:
        ex = by_id.get(lab.example_id)
        if ex is None:
            continue
        rec = GenerationRecord(example_id=ex.id, system="human_label:" + lab.defect,
                               reply=SuggestedReply(body=lab.candidate))
        try:
            a = await run_audit(llm, ex, rec.reply, model=judge_model)
            v = await run_judge(llm, ex, rec.reply, model=judge_model)
        except Exception as exc:  # noqa: BLE001
            rows.append({"id": lab.id, "error": str(exc)[:150]})
            continue
        s = build_score(ex, rec, a, v)
        rows.append({
            "id": lab.id,
            "example_id": ex.id,
            "defect": lab.defect,
            "human": lab.human_overall,
            "metric": s.composite,
            "readiness": s.readiness,
            "dims": {k: d.score for k, d in s.dimensions.items()},
            "rationale": lab.rationale,
        })

    good = [r for r in rows if "error" not in r]
    if len(good) < 4:
        return {"error": "not enough scored labels", "rows": rows}

    human = [r["human"] for r in good]
    metric = [r["metric"] for r in good]

    # Does the metric separate the three quality bands a human assigned?
    bands: dict[str, list[float]] = {}
    for r in good:
        bands.setdefault(r["defect"], []).append(r["metric"])

    per_dim_corr = {}
    for d in DEFAULT_WEIGHTS:
        per_dim_corr[d] = spearman([r["dims"].get(d, 0.0) for r in good], human)

    return {
        "n": len(good),
        "spearman": spearman(metric, human),
        "kendall_tau": kendall_tau(metric, human),
        "pairwise": pairwise_accuracy(human, metric),
        "mean_absolute_error": round(mean(abs(a - b) for a, b in zip(metric, human)), 4),
        "metric_mean_by_human_band": {
            k: round(mean(v), 4) for k, v in sorted(bands.items())
        },
        "per_dimension_spearman_with_human_overall": per_dim_corr,
        "weight_fit": fit_weights([r["dims"] for r in good], human),
        "rows": sorted(good, key=lambda r: r["human"]),
        "caveat": (
            "One annotator, n=" + str(len(good)) + ", labels written before the metric "
            "was run on them. Detects a broken metric; does not certify a correct one."
        ),
    }
