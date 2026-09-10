"""Turn audit + judge + rule checks into one auditable score per reply.

The shape of the metric
-----------------------
Six dimensions, each in [0,1], combined by a weighted sum, with hard caps that
override the arithmetic. The caps matter as much as the weights: some failures
are not "a low score", they are "do not send this", and an average will happily
launder one catastrophic dimension behind five good ones.

Weights (defended in the README, and re-derived from the human-labelled set by
`replybench validate-metric`):

    action_match  0.28   did it do what a competent human did
    factuality    0.26   is every claim backed by a KB row
    completeness  0.18   did it address everything asked
    policy_safety 0.14   is it inside what an agent may promise
    tone_fit      0.08   right register
    clarity       0.06   readable

The top three are reference- and evidence-grounded. The bottom three are the
subjective ones, and together they carry 28% -- deliberately, because tone and
clarity are the cheapest things for a human to fix in the two seconds before
they hit send, and a wrong refund promise is not.
"""
from __future__ import annotations

from ..schemas import (
    AuditResult, DimensionScore, Example, GenerationRecord, JudgeVerdict,
    Readiness, ResponseScore,
)
from ..taxonomy import CONSEQUENTIAL_ACTIONS, DEFAULT_WEIGHTS
from . import deterministic, lexical

# A consequential action is worth this many ordinary actions when scoring the
# action set. Promising a refund nobody authorised is not one unit of wrong.
CONSEQUENTIAL_WEIGHT = 3.0

BREACH_CAP = {"high": 0.15, "medium": 0.45, "low": 0.75}


def _w(action: str) -> float:
    return CONSEQUENTIAL_WEIGHT if action in CONSEQUENTIAL_ACTIONS else 1.0


def action_f1(gold: list[str], pred: list[str]) -> tuple[float, list[str], list[str]]:
    """Weighted set F1 over the action vocabulary.

    Both empty scores 1.0: agreeing that a reply should commit to nothing (a
    thank-you, a close-loop) is a correct judgement, not an undefined one.
    """
    g, p = set(gold), set(pred)
    missing = sorted(g - p)
    extra = sorted(p - g)
    if not g and not p:
        return 1.0, missing, extra
    tp = sum(_w(a) for a in g & p)
    fp = sum(_w(a) for a in p - g)
    fn = sum(_w(a) for a in g - p)
    if tp == 0:
        return 0.0, missing, extra
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    return 2 * precision * recall / (precision + recall), missing, extra


def factuality_score(audit: AuditResult) -> tuple[float, str, list[str]]:
    """Fraction of checkable claims that survive verification.

    A contradiction costs more than an unsupported claim: saying the refund
    window is 60 days when it is 30 is worse than saying something true-ish that
    we cannot find a row for. Both are wrong; only one is confidently wrong.
    """
    checkable = [c for c in audit.claims if c.verdict in ("supported", "contradicted", "unsupported")]
    if not checkable:
        return 1.0, "no checkable factual claims (nothing asserted about Kestrel)", []
    contradicted = [c for c in checkable if c.verdict == "contradicted"]
    unsupported = [c for c in checkable if c.verdict == "unsupported"]
    penalty = (1.0 * len(contradicted) + 0.6 * len(unsupported)) / len(checkable)
    score = max(0.0, 1.0 - penalty)
    ev = [c.claim + " -> " + c.verdict + (" (" + c.note + ")" if c.note else "")
          for c in contradicted + unsupported][:5]
    reason = (str(len(checkable) - len(contradicted) - len(unsupported)) + "/" + str(len(checkable))
              + " claims supported; " + str(len(contradicted)) + " contradicted, "
              + str(len(unsupported)) + " unsupported")
    return score, reason, ev


ASK_CREDIT = {
    "answered": 1.0,
    "deferred_appropriately": 1.0,
    "partially_answered": 0.5,
    "ignored": 0.0,
}


def completeness_score(audit: AuditResult) -> tuple[float, str, list[str]]:
    if not audit.asks:
        return 1.0, "no asks extracted from the inbound email", []
    credits = [ASK_CREDIT.get(a.status, 0.0) for a in audit.asks]
    score = sum(credits) / len(credits)
    missed = [a.ask for a in audit.asks if a.status in ("ignored", "partially_answered")]
    reason = (str(sum(1 for c in credits if c == 1.0)) + "/" + str(len(credits))
              + " asks fully handled")
    return score, reason, missed[:5]


def _readiness(composite: float, hard: list[str], breaches: list, judge_call: Readiness) -> Readiness:
    """Derived send-readiness, taking the more conservative of rule and judge."""
    order = ["do_not_send", "heavy_edit", "light_edit", "send_as_is"]
    if hard or any(b.severity == "high" for b in breaches):
        return "do_not_send"
    if composite >= 0.85:
        derived: Readiness = "send_as_is"
    elif composite >= 0.65:
        derived = "light_edit"
    elif composite >= 0.40:
        derived = "heavy_edit"
    else:
        derived = "do_not_send"
    return order[min(order.index(derived), order.index(judge_call))]  # type: ignore[return-value]


def build_score(
    example: Example,
    gen: GenerationRecord,
    audit: AuditResult,
    verdict: JudgeVerdict,
    *,
    weights: dict[str, float] | None = None,
) -> ResponseScore:
    """Score one reply.

    If the example carries no reference reply -- which is the case for a live
    incoming email, the whole point of the product -- `action_match` is not
    computable. It is then DROPPED and the remaining weights renormalised, rather
    than scored 1.0 for having an empty gold set on both sides. Silently
    awarding a perfect score for absent ground truth would make the composite
    unable to fail, which is the failure this project exists to avoid.
    """
    weights = dict(weights or DEFAULT_WEIGHTS)
    has_reference = bool(example.reply.body.strip())
    if not has_reference:
        weights.pop("action_match", None)
        total = sum(weights.values()) or 1.0
        weights = {k: v / total for k, v in weights.items()}
    body = gen.reply.body

    if gen.error or not body.strip():
        return ResponseScore(
            example_id=example.id, system=gen.system,
            dimensions={k: DimensionScore(score=0.0, reason="generation failed") for k in weights},
            composite=0.0, readiness="do_not_send",
            hard_failures=["generation_error"], error=gen.error or "empty body",
        )

    hard, warn = deterministic.run(body, example.incoming.body)

    af1, missing, extra = action_f1(example.reply.actions, audit.actions)
    fact, fact_reason, fact_ev = factuality_score(audit)
    comp, comp_reason, comp_ev = completeness_score(audit)

    # Policy safety: the judge's opinion, capped by any concrete breach found.
    cap = min([BREACH_CAP.get(b.severity, 1.0) for b in audit.breaches], default=1.0)
    policy = min(verdict.policy_safety.score, cap)

    dims = {
        "factuality": DimensionScore(score=round(fact, 4), reason=fact_reason, evidence=fact_ev),
        "completeness": DimensionScore(score=round(comp, 4), reason=comp_reason, evidence=comp_ev),
        "policy_safety": DimensionScore(
            score=round(policy, 4),
            reason=verdict.policy_safety.reason + (
                " | capped at " + str(cap) + " by " + str(len(audit.breaches)) + " breach(es)"
                if cap < 1.0 else ""),
            evidence=verdict.policy_safety.evidence + [b.rule_id + ": " + b.detail for b in audit.breaches],
        ),
        "tone_fit": verdict.tone_fit,
        "clarity": verdict.clarity,
    }
    if has_reference:
        dims["action_match"] = DimensionScore(
            score=round(af1, 4),
            reason=("weighted F1 vs the reply a human actually sent; "
                    + str(len(missing)) + " missing, " + str(len(extra)) + " extra"),
            evidence=(["missing: " + ", ".join(missing)] if missing else [])
                     + (["extra: " + ", ".join(extra)] if extra else []),
        )

    composite = sum(dims[k].score * w for k, w in weights.items() if k in dims)
    if hard:
        # A reply you cannot send is not a 0.7 with an asterisk.
        composite = min(composite, 0.20)

    conseq = [a for a in extra if a in CONSEQUENTIAL_ACTIONS]
    conseq += ["MISSING:" + a for a in missing if a in CONSEQUENTIAL_ACTIONS]

    return ResponseScore(
        example_id=example.id,
        system=gen.system,
        dimensions=dims,
        composite=round(composite, 4),
        readiness=_readiness(composite, hard, audit.breaches, verdict.would_send),
        claims=audit.claims,
        asks=audit.asks,
        breaches=audit.breaches,
        gold_actions=example.reply.actions,
        pred_actions=audit.actions,
        missing_actions=missing,
        extra_actions=extra,
        consequential_errors=conseq,
        lexical=lexical.score(body, example.reply.body),
        hard_failures=hard,
        warnings=warn,
        judge_note=verdict.biggest_problem,
    )
