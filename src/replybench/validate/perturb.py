"""Unit tests for the metric itself.

The hard question in this challenge is not "what score did the system get" but
"why should anyone believe the score". Human agreement studies are the usual
answer and they need annotators. This is the answer that does not:

    Take a reply we know is good -- the gold reply a human sent. Break it in ONE
    specific, known way. A metric that is measuring what it claims to measure
    must (a) notice, and (b) notice in the RIGHT dimension.

That gives two numbers per defect class:

    sensitivity  did the targeted dimension actually drop?
    specificity  did the untargeted dimensions stay put?

A metric that drops everything whenever anything is wrong is not diagnostic --
it is a mood. Specificity is what makes the per-dimension report actionable.

And one control, which is the point of the whole exercise:

    paraphrase   rewrite the gold reply so it means the same thing in different
                 words. Every score MUST hold. This is the case that kills
                 ROUGE-L, and we report both side by side so the claim
                 "surface overlap is the wrong metric" is evidence, not opinion.
"""
from __future__ import annotations

import random
import re
from statistics import mean
from typing import Sequence

from ..llm import LLM
from ..schemas import Example, GenerationRecord, PerturbationCase, SuggestedReply
from .. import kb
from ..evaluate.audit import audit as run_audit
from ..evaluate.judge import judge as run_judge
from ..evaluate.score import build_score

# defect kind -> (dimensions that must drop, human-readable note)
EXPECTATIONS: dict[str, tuple[list[str], str]] = {
    "fact_swap": (["factuality"], "a policy number replaced with a wrong one"),
    "drop_answer": (["completeness"], "the paragraph answering one ask deleted"),
    "rude": (["tone_fit"], "same content, hostile register"),
    "over_promise": (["policy_safety"], "an unauthorised refund promise appended"),
    "topic_swap": (["action_match", "completeness"], "a reply to a different email entirely"),
    "padding": (["clarity"], "80 words of corporate filler inserted"),
    "paraphrase": ([], "meaning preserved, words changed -- nothing should drop"),
}

RUDE_OPENER = "Look, "
RUDE_LINES = [
    "I'm not sure what you expected here.",
    "This is fairly basic and it's covered in the docs.",
    "If you'd read the previous email you wouldn't need to ask.",
]
FILLER = (
    "We would like to take this opportunity to thank you for your continued patience and "
    "for being a valued customer of ours. Please rest assured that your satisfaction "
    "remains our highest priority at all times, and that we are committed to delivering a "
    "best-in-class experience across every one of your interactions with our team. We do "
    "sincerely apologise for any inconvenience caused in this matter."
)
OVER_PROMISE = (
    "\n\nI've gone ahead and processed a full refund of the remaining annual term back to "
    "your card, and waived next month's invoice as a goodwill gesture. No approval needed."
)

_NUM = re.compile(r"\b(\d[\d,]*(?:\.\d+)?)\s?(%|days?|hours?|months?|business days?|seats?|rows?)\b", re.I)


def _fact_swap(text: str, rng: random.Random) -> str | None:
    """Replace a figure with a plausible but wrong one."""
    matches = list(_NUM.finditer(text))
    if not matches:
        return None
    m = rng.choice(matches)
    try:
        val = float(m.group(1).replace(",", ""))
    except ValueError:
        return None
    wrong = int(val * rng.choice([2, 3])) or 99
    return text[: m.start(1)] + str(wrong) + text[m.end(1) :]


def _drop_answer(text: str) -> str | None:
    paras = [p for p in text.split("\n\n") if p.strip()]
    if len(paras) < 4:
        return None
    # Drop a middle paragraph -- keeps greeting and sign-off so the reply still
    # looks complete, which is exactly the failure we want the metric to catch.
    del paras[len(paras) // 2]
    return "\n\n".join(paras)


def _rude(text: str, rng: random.Random) -> str:
    lines = text.split("\n")
    body = "\n".join(lines[1:]) if len(lines) > 1 else text
    return RUDE_OPENER + body.lstrip().replace("\n\n", "\n\n" + rng.choice(RUDE_LINES) + " ", 1)


def _padding(text: str) -> str:
    paras = text.split("\n\n")
    if len(paras) < 2:
        return text + "\n\n" + FILLER
    paras.insert(1, FILLER)
    return "\n\n".join(paras)


PARAPHRASE_SYS = """Rewrite this support reply so it means EXACTLY the same thing in
different words. Keep every fact, every number, every commitment and every question
identical. Change sentence structure and word choice. Do not add anything. Do not remove
anything. Do not change the tone. Return ONLY the rewritten reply."""


async def build_cases(
    llm: LLM, examples: Sequence[Example], *, n: int = 5, seed: int = 11
) -> list[PerturbationCase]:
    rng = random.Random(seed)
    picked = [e for e in examples if len(e.reply.body) > 300][:n]
    cases: list[PerturbationCase] = []

    for ex in picked:
        gold = ex.reply.body
        cases.append(PerturbationCase(
            id=ex.id + "::control", example_id=ex.id, kind="control", text=gold,
            expect_stable=True, note="the gold reply itself -- the ceiling"))

        for kind, maker in (
            ("fact_swap", lambda t: _fact_swap(t, rng)),
            ("drop_answer", _drop_answer),
            ("rude", lambda t: _rude(t, rng)),
            ("padding", _padding),
            ("over_promise", lambda t: t + OVER_PROMISE),
        ):
            out = maker(gold)
            if out and out != gold:
                cases.append(PerturbationCase(
                    id=ex.id + "::" + kind, example_id=ex.id, kind=kind, text=out,
                    expect_drop_in=EXPECTATIONS[kind][0], note=EXPECTATIONS[kind][1]))

        other = rng.choice([e for e in picked if e.id != ex.id] or [ex])
        if other.id != ex.id:
            cases.append(PerturbationCase(
                id=ex.id + "::topic_swap", example_id=ex.id, kind="topic_swap",
                text=other.reply.body, expect_drop_in=EXPECTATIONS["topic_swap"][0],
                note=EXPECTATIONS["topic_swap"][1]))

    # The control that matters most: a real paraphrase, which needs a model.
    for ex in picked:
        try:
            para = await llm.chat(
                [{"role": "system", "content": PARAPHRASE_SYS},
                 {"role": "user", "content": ex.reply.body}],
                model=llm.s.gen_model, temperature=0.7, max_tokens=1200)
            if len(para.strip()) > 100:
                cases.append(PerturbationCase(
                    id=ex.id + "::paraphrase", example_id=ex.id, kind="paraphrase",
                    text=para.strip(), expect_stable=True, note=EXPECTATIONS["paraphrase"][1]))
        except Exception:  # noqa: BLE001 - a missing paraphrase is not fatal
            pass

    return cases


async def run_suite(
    llm: LLM, examples: Sequence[Example], cases: Sequence[PerturbationCase]
) -> dict:
    by_id = {e.id: e for e in examples}
    scored: dict[str, dict] = {}

    for case in cases:
        ex = by_id[case.example_id]
        rec = GenerationRecord(
            example_id=ex.id, system="perturb:" + case.kind,
            reply=SuggestedReply(body=case.text))
        try:
            a = await run_audit(llm, ex, rec.reply)
            v = await run_judge(llm, ex, rec.reply)
        except Exception as exc:  # noqa: BLE001
            scored[case.id] = {"error": str(exc)[:150]}
            continue
        s = build_score(ex, rec, a, v)
        scored[case.id] = {
            "kind": case.kind,
            "example_id": ex.id,
            "composite": s.composite,
            "dims": {k: d.score for k, d in s.dimensions.items()},
            "rouge_l": s.lexical.rouge_l if s.lexical else 0.0,
        }

    # ---- compare every defect against its own example's control ----
    controls = {v["example_id"]: v for v in scored.values()
                if isinstance(v, dict) and v.get("kind") == "control"}
    results: dict[str, dict] = {}
    for kind in [k for k in EXPECTATIONS] + ["control"]:
        rows = [v for v in scored.values()
                if isinstance(v, dict) and v.get("kind") == kind and v["example_id"] in controls]
        if not rows:
            continue
        deltas: dict[str, list[float]] = {}
        rouge_delta = []
        comp_delta = []
        for r in rows:
            c = controls[r["example_id"]]
            for d, val in r["dims"].items():
                deltas.setdefault(d, []).append(val - c["dims"].get(d, 0.0))
            rouge_delta.append(r["rouge_l"] - c["rouge_l"])
            comp_delta.append(r["composite"] - c["composite"])

        targeted = EXPECTATIONS.get(kind, ([], ""))[0]
        mean_deltas = {d: round(mean(v), 4) for d, v in deltas.items()}
        detected = all(mean_deltas.get(d, 0.0) <= -0.08 for d in targeted) if targeted else None
        collateral = [d for d, v in mean_deltas.items()
                      if d not in targeted and v <= -0.15]

        results[kind] = {
            "n": len(rows),
            "targeted_dimensions": targeted,
            "mean_delta_by_dimension": mean_deltas,
            "mean_delta_composite": round(mean(comp_delta), 4),
            "mean_delta_rouge_l": round(mean(rouge_delta), 4),
            "detected": detected,
            "collateral_damage": collateral,
            "note": EXPECTATIONS.get(kind, ([], "unperturbed gold reply"))[1],
        }

    fired = [k for k, v in results.items() if v["detected"] is True]
    targeted_kinds = [k for k, v in results.items() if v["detected"] is not None]
    para = results.get("paraphrase", {})
    results["_summary"] = {
        "sensitivity": (str(len(fired)) + "/" + str(len(targeted_kinds))
                        + " defect classes detected in the targeted dimension"),
        "clean_detections": [k for k in fired if not results[k]["collateral_damage"]],
        "paraphrase_composite_delta": para.get("mean_delta_composite"),
        "paraphrase_rouge_delta": para.get("mean_delta_rouge_l"),
        "headline": (
            "If paraphrase_rouge_delta is strongly negative while "
            "paraphrase_composite_delta is near zero, surface-overlap metrics are "
            "penalising a reply that is just as good, and the rubric is not."
        ),
    }
    return results
