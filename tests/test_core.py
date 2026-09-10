"""Tests for the deterministic parts of the metric.

These need no API key. They cover the pieces where a silent bug would corrupt
every score in the report: action-set F1 edge cases, the numeric hallucination
detector, quote stripping, and the paraphrase property that motivates the whole
design (a rewrite that means the same thing must not be punished by the
dimensions, even though ROUGE-L falls).
"""
from __future__ import annotations

from replybench.evaluate import deterministic, lexical
from replybench.evaluate.score import action_f1, completeness_score, factuality_score
from replybench.generate.parse import clean, clean_for_index
from replybench.generate.retrieve import Retriever, tokenize
from replybench.schemas import AskCheck, AuditResult, ClaimCheck
from replybench.taxonomy import ACTIONS, CONSEQUENTIAL_ACTIONS, DEFAULT_WEIGHTS


# --------------------------------------------------------------- taxonomy
def test_weights_sum_to_one():
    assert abs(sum(DEFAULT_WEIGHTS.values()) - 1.0) < 1e-9


def test_consequential_actions_are_real_actions():
    assert CONSEQUENTIAL_ACTIONS <= set(ACTIONS)


# ------------------------------------------------------------- action F1
def test_both_empty_is_perfect_agreement():
    # A close-loop reply that commits to nothing is a correct judgement, not
    # an undefined one. If this returns 0.0, every thank-you scores zero.
    score, missing, extra = action_f1([], [])
    assert score == 1.0 and missing == [] and extra == []


def test_exact_match():
    assert action_f1(["apologize", "issue_refund"], ["issue_refund", "apologize"])[0] == 1.0


def test_disjoint_sets_score_zero():
    assert action_f1(["apologize"], ["state_pricing"])[0] == 0.0


def test_consequential_false_positive_hurts_more_than_a_trivial_one():
    gold = ["state_pricing"]
    trivial = action_f1(gold, ["state_pricing", "send_documentation_link"])[0]
    severe = action_f1(gold, ["state_pricing", "issue_refund"])[0]
    assert severe < trivial, "promising a refund must cost more than adding a doc link"


def test_missing_and_extra_are_reported():
    _, missing, extra = action_f1(["a_gold", "apologize"], ["apologize", "state_pricing"])
    assert missing == ["a_gold"] and extra == ["state_pricing"]


# ------------------------------------------------------------ factuality
def test_factuality_perfect_when_nothing_asserted():
    score, reason, _ = factuality_score(AuditResult(claims=[]))
    assert score == 1.0 and "no checkable" in reason


def test_contradiction_costs_more_than_unsupported():
    contra = factuality_score(AuditResult(claims=[
        ClaimCheck(claim="x", verdict="contradicted", source_id="REF-01")]))[0]
    unsup = factuality_score(AuditResult(claims=[
        ClaimCheck(claim="x", verdict="unsupported")]))[0]
    assert contra < unsup


def test_not_checkable_claims_are_excluded():
    # "I've escalated this" is a commitment, not a factual claim. If these
    # counted, every reply would be penalised for being helpful.
    score, _, _ = factuality_score(AuditResult(claims=[
        ClaimCheck(claim="I've escalated this", verdict="not_checkable"),
        ClaimCheck(claim="Refunds take 5-10 days", verdict="supported", source_id="REF-03"),
    ]))
    assert score == 1.0


# ----------------------------------------------------------- completeness
def test_appropriate_deferral_counts_as_handled():
    score, _, _ = completeness_score(AuditResult(asks=[
        AskCheck(ask="refund me", status="deferred_appropriately")]))
    assert score == 1.0


def test_ignored_ask_scores_zero_and_is_surfaced():
    score, _, missed = completeness_score(AuditResult(asks=[
        AskCheck(ask="answered one", status="answered"),
        AskCheck(ask="dropped one", status="ignored"),
    ]))
    assert score == 0.5 and missed == ["dropped one"]


# ---------------------------------------------------------- deterministic
def test_unfilled_placeholder_is_a_hard_failure():
    hard, _ = deterministic.run("Hi [customer name],\n\nAll sorted.\n\nPriya", "hello")
    assert any(h.startswith("unresolved_placeholder") for h in hard)


def test_leaked_fact_id_is_a_hard_failure():
    hard, _ = deterministic.run(
        "Hi,\n\nPer REF-01 you can have a refund within 30 days.\n\nPriya\nKestrel Support", "x")
    assert any(h.startswith("leaked_fact_id") for h in hard)


def test_meta_commentary_is_a_hard_failure():
    hard, _ = deterministic.run(
        "Here is the draft reply:\n\nHi, all sorted.\n\nPriya\nKestrel Support", "x")
    assert any(h.startswith("meta_commentary") for h in hard)


def test_numeric_hallucination_detected():
    bad = deterministic.unsupported_numbers(
        "Our refund window is 60 days and uptime is 99.99%.", "when do refunds expire?")
    assert bad, "60 days / 99.99% appear in neither the KB nor the customer email"


def test_kb_numbers_are_not_flagged():
    # 30 days (REF-01) and 99.9% (SLA-01) are true; flagging them would make the
    # detector useless noise.
    assert not deterministic.unsupported_numbers(
        "Refunds are within 30 days and uptime is 99.9%.", "tell me about refunds")


def test_small_counts_are_not_treated_as_policy_claims():
    assert not deterministic.unsupported_numbers("Three quick questions, 2 sites.", "hi")


# ------------------------------------------------------------------ parse
QUOTED = """Does the CSV import have a row limit? We have 6,200 staff.

On Wed, 12 Mar 2025 at 14:22, Priya Raman <support@kestrelhq.com> wrote:
> No problem at all - the migration team will pick that up.
> > Happy to help.
"""


def test_quote_stripping_keeps_the_new_question():
    out = clean(QUOTED)
    assert "row limit" in out and "migration team" not in out


def test_signature_and_disclaimer_stripped():
    out = clean("Please send the invoice.\n\n--\nJane Doe\nCFO\n"
                "This email and any attachments are confidential.")
    assert "invoice" in out and "confidential" not in out


def test_mobile_footer_stripped():
    assert "iPhone" not in clean("where is the export button?\n\nSent from my iPhone")


def test_index_form_drops_greeting_noise():
    idx = clean_for_index("Hi there,\n\nHow do I export timesheets?\n\nThanks,\nJo")
    assert "export" in idx and "Hi there" not in idx


# ---------------------------------------------------------------- lexical
def test_rouge_is_one_for_identical_text():
    assert lexical.rouge_l("the refund window is 30 days", "the refund window is 30 days") == 1.0


def test_paraphrase_tanks_rouge():
    """The motivating failure for this entire repo.

    Two replies that say the same true thing, one scored near zero by surface
    overlap. This is why `action_match` + `factuality` exist.
    """
    a = "You can have a full refund within 30 days of the charge."
    b = "Refunds are available in full up to thirty days after we bill you."
    assert lexical.rouge_l(a, b) < 0.5


def test_wrong_fact_barely_moves_rouge():
    """And the mirror image: a reply that is *wrong* scores ~1.0 on ROUGE."""
    right = "You can have a full refund within 30 days of the charge."
    wrong = "You can have a full refund within 90 days of the charge."
    assert lexical.rouge_l(wrong, right) > 0.9


# --------------------------------------------------------------- retrieval
def test_tokenizer_drops_stopwords_and_folds_suffixes():
    t = tokenize("The refunds are processing quickly")
    assert "the" not in t and "are" not in t and "refund" in t


def test_retriever_survives_an_empty_corpus():
    assert Retriever([]).search("subject", "body") == []


# ------------------------------------------------- scoring without a reference
def _blank_verdict(v: float = 0.7):
    from replybench.schemas import DimensionScore, JudgeVerdict
    d = DimensionScore(score=v, reason="t", evidence=["q"])
    return JudgeVerdict(tone_fit=d, clarity=d, policy_safety=d, holistic=d,
                        would_send="light_edit")


def _example(reference: str):
    from replybench.schemas import Email, Example, Reply
    return Example(
        id="t", split="test", intent="answer_question", difficulty="routine",
        incoming=Email(subject="s", body="Can I have a refund?", sender_name="A",
                       sender_email="a@b.com"),
        reply=Reply(body=reference, actions=["issue_refund"]),
    )


def _gen():
    from replybench.schemas import GenerationRecord, SuggestedReply
    # Long enough to clear the 40-char `reply_too_short` hard failure, which
    # otherwise caps the composite at 0.20 and masks what these tests check.
    return GenerationRecord(
        example_id="t", system="main",
        reply=SuggestedReply(body="Hi there,\n\nYes, that refund is on its way to you "
                                  "now.\n\nPriya Raman\nKestrel Support"))


def test_action_match_dropped_when_there_is_no_reference():
    """A live email has no gold reply, so action_match is not computable.

    It must be DROPPED, not scored 1.0 for having two empty sets. Scoring it 1.0
    made `suggest --explain` report 0.92 for a placeholder reply -- a composite
    that could not fail, which is the exact failure this project is about.
    """
    from replybench.evaluate.score import build_score
    s = build_score(_example(""), _gen(), AuditResult(), _blank_verdict())
    assert "action_match" not in s.dimensions


def test_action_match_present_when_there_is_a_reference():
    from replybench.evaluate.score import build_score
    s = build_score(_example("Yes, refunded."), _gen(), AuditResult(), _blank_verdict())
    assert "action_match" in s.dimensions


def test_weights_renormalise_so_a_perfect_reply_can_still_reach_one():
    """Dropping a dimension must not cap the composite below 1.0."""
    from replybench.schemas import ClaimCheck
    from replybench.evaluate.score import build_score
    audit = AuditResult(claims=[ClaimCheck(claim="c", verdict="supported", source_id="REF-01")])
    s = build_score(_example(""), _gen(), audit, _blank_verdict(1.0))
    assert s.composite > 0.99
