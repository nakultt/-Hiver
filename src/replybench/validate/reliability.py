"""Is the judge stable, and is it just flattering its own model?

Two failure modes that a single run cannot see:

  1. **Noise.** Ask the same judge the same question k times at a non-zero
     temperature. If the scores wander by 0.2, then a 0.05 gap between two
     systems is meaningless and every comparison in the report is theatre. We
     measure the spread and report it next to the between-system gaps, so the
     reader can see whether the signal exceeds the noise.

  2. **Self-preference.** Generator and judge here are the same model family --
     forced by the free tier, and the single biggest threat to validity in this
     repo. If a model systematically prefers its own output, the headline score
     is inflated and no amount of rubric design fixes it. The check: re-judge a
     subsample with a *different* model and compare the two rankings. High rank
     correlation means the ranking is not an artefact of who is judging; low
     correlation means it is, and the report should be read as unreliable.
"""
from __future__ import annotations

from statistics import mean, pstdev
from typing import Sequence

from ..evaluate.aggregate import spearman
from ..evaluate.audit import audit as run_audit
from ..evaluate.judge import judge as run_judge
from ..evaluate.score import build_score
from ..llm import LLM
from ..schemas import Example, GenerationRecord


async def self_consistency(
    llm: LLM,
    examples: Sequence[Example],
    records: Sequence[GenerationRecord],
    *,
    k: int = 3,
    temperature: float = 0.7,
    limit: int = 8,
) -> dict:
    """Judge the same drafts k times and report the spread."""
    by_id = {e.id: e for e in examples}
    sample = [r for r in records if r.reply.body.strip()][:limit]
    rows = []

    for rec in sample:
        ex = by_id.get(rec.example_id)
        if ex is None:
            continue
        scores = []
        for i in range(k):
            try:
                v = await run_judge(llm, ex, rec.reply, temperature=temperature, variant=i + 1)
            except Exception:  # noqa: BLE001
                continue
            scores.append(v.holistic.score)
        if len(scores) >= 2:
            rows.append({
                "example_id": rec.example_id,
                "system": rec.system,
                "samples": scores,
                "mean": round(mean(scores), 4),
                "sd": round(pstdev(scores), 4),
                "range": round(max(scores) - min(scores), 4),
            })

    if not rows:
        return {"error": "no samples scored"}
    sds = [r["sd"] for r in rows]
    ranges = [r["range"] for r in rows]
    return {
        "n_drafts": len(rows),
        "k": k,
        "temperature": temperature,
        "mean_sd": round(mean(sds), 4),
        "max_range": round(max(ranges), 4),
        "rows": rows,
        "interpretation": (
            "Treat any between-system gap smaller than ~2x mean_sd as noise. "
            "This is the resolution limit of the judge."
        ),
    }


async def cross_model_judge(
    llm: LLM,
    examples: Sequence[Example],
    records: Sequence[GenerationRecord],
    *,
    alt_model: str | None = None,
    limit: int = 12,
) -> dict:
    """Re-score a subsample with a different judge model; compare rankings."""
    by_id = {e.id: e for e in examples}
    alt = alt_model or llm.s.strong_model
    sample = [r for r in records if r.reply.body.strip()][:limit]

    primary: list[float] = []
    secondary: list[float] = []
    rows = []

    for rec in sample:
        ex = by_id.get(rec.example_id)
        if ex is None:
            continue
        try:
            a1 = await run_audit(llm, ex, rec.reply, model=llm.s.judge_model)
            v1 = await run_judge(llm, ex, rec.reply, model=llm.s.judge_model)
            a2 = await run_audit(llm, ex, rec.reply, model=alt)
            v2 = await run_judge(llm, ex, rec.reply, model=alt)
        except Exception as exc:  # noqa: BLE001
            rows.append({"example_id": rec.example_id, "error": str(exc)[:150]})
            continue
        s1 = build_score(ex, rec, a1, v1).composite
        s2 = build_score(ex, rec, a2, v2).composite
        primary.append(s1)
        secondary.append(s2)
        rows.append({"example_id": rec.example_id, "system": rec.system,
                     "primary": s1, "secondary": s2, "delta": round(s2 - s1, 4)})

    if len(primary) < 3:
        return {"error": "not enough cross-judged rows", "rows": rows}

    bias = mean(s - p for p, s in zip(primary, secondary))
    return {
        "n": len(primary),
        "primary_judge": llm.s.judge_model,
        "secondary_judge": alt,
        "spearman_between_judges": spearman(primary, secondary),
        "mean_shift": round(bias, 4),
        "primary_mean": round(mean(primary), 4),
        "secondary_mean": round(mean(secondary), 4),
        "rows": rows,
        "interpretation": (
            "spearman_between_judges is the number that matters: it says whether the "
            "RANKING survives changing the judge. mean_shift is a calibration offset "
            "and is largely harmless -- a judge that is uniformly stricter still ranks "
            "systems the same way. A high shift with a high correlation means "
            "'stricter but agrees'; a low correlation means the ranking is an artefact "
            "of the judge and the report should not be trusted."
        ),
    }
