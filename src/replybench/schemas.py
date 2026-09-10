"""Typed contracts for every artefact that crosses a module or a file boundary."""
from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, Field

Verdict = Literal["supported", "contradicted", "unsupported", "not_checkable"]
AskStatus = Literal["answered", "partially_answered", "deferred_appropriately", "ignored"]
Readiness = Literal["send_as_is", "light_edit", "heavy_edit", "do_not_send"]


# ===========================================================================
# Dataset
# ===========================================================================
class Email(BaseModel):
    """An inbound message, as it would arrive in a shared inbox."""

    subject: str
    body: str = Field(description="Raw body, including any quoted trail and signature.")
    sender_name: str
    sender_email: str
    sender_role: str = ""
    account_plan: str = ""
    received_at: str = ""


class Reply(BaseModel):
    """The reply a human agent actually sent."""

    body: str
    author: str = ""
    actions: list[str] = Field(default_factory=list)
    cited_facts: list[str] = Field(default_factory=list)


class Example(BaseModel):
    """One dataset row: an incoming email and the reply that was sent."""

    id: str
    split: Literal["train", "test", "hard"]
    intent: str
    difficulty: str
    tags: list[str] = Field(default_factory=list)
    incoming: Email
    reply: Reply
    customer_asks: list[str] = Field(
        default_factory=list,
        description="Atomic requests/questions the customer made. Gold labels for completeness.",
    )
    key_facts: list[str] = Field(
        default_factory=list,
        description="KB fact IDs a correct reply must get right.",
    )
    trap: str = Field("", description="For hard cases: the mistake this row is designed to catch.")
    provenance: dict[str, Any] = Field(default_factory=dict)


class Dataset(BaseModel):
    examples: list[Example]

    def split(self, name: str) -> list[Example]:
        return [e for e in self.examples if e.split == name]


# ===========================================================================
# Generation
# ===========================================================================
class SuggestedReply(BaseModel):
    """What the generator returns. Structured so it is auditable, not just prose."""

    body: str
    intent: str = ""
    actions: list[str] = Field(default_factory=list)
    cited_facts: list[str] = Field(default_factory=list)
    confidence: float = 0.5
    needs_human_review: bool = False
    open_questions: list[str] = Field(default_factory=list)


class GenerationRecord(BaseModel):
    example_id: str
    system: str = Field(description="Which generator produced this (main, ablation, baseline).")
    reply: SuggestedReply
    retrieved_ids: list[str] = Field(default_factory=list)
    retrieved_facts: list[str] = Field(default_factory=list)
    latency_s: float = 0.0
    error: str = ""


# ===========================================================================
# Evaluation
# ===========================================================================
class ClaimCheck(BaseModel):
    claim: str
    verdict: Verdict
    source_id: str = Field("", description="KB fact ID, 'incoming', or 'reference'.")
    note: str = ""


class AskCheck(BaseModel):
    ask: str
    status: AskStatus
    evidence: str = ""


class PolicyBreach(BaseModel):
    rule_id: str
    severity: Literal["low", "medium", "high"]
    detail: str
    quote: str = ""


class DimensionScore(BaseModel):
    """A 0-1 score plus the reason it is that number.

    Every dimension must carry `evidence`. A score with no evidence is a number
    we cannot audit, and an evaluation you cannot audit is not an evaluation.
    """

    score: float
    reason: str
    evidence: list[str] = Field(default_factory=list)


class JudgeVerdict(BaseModel):
    """Raw rubric output from the LLM judge (the subjective dimensions)."""

    tone_fit: DimensionScore
    clarity: DimensionScore
    policy_safety: DimensionScore
    holistic: DimensionScore
    would_send: Readiness
    biggest_problem: str = ""


class LexicalScores(BaseModel):
    rouge_l: float
    token_f1: float
    length_ratio: float
    char_edit_similarity: float


class ResponseScore(BaseModel):
    """The full accuracy report for one suggested reply."""

    example_id: str
    system: str

    dimensions: dict[str, DimensionScore]
    composite: float
    readiness: Readiness

    claims: list[ClaimCheck] = Field(default_factory=list)
    asks: list[AskCheck] = Field(default_factory=list)
    breaches: list[PolicyBreach] = Field(default_factory=list)

    gold_actions: list[str] = Field(default_factory=list)
    pred_actions: list[str] = Field(default_factory=list)
    missing_actions: list[str] = Field(default_factory=list)
    extra_actions: list[str] = Field(default_factory=list)
    consequential_errors: list[str] = Field(default_factory=list)

    lexical: Optional[LexicalScores] = None
    hard_failures: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    judge_note: str = ""
    error: str = ""


# ===========================================================================
# Structured LLM sub-tasks (each is one small, checkable question)
# ===========================================================================
class ClaimExtraction(BaseModel):
    claims: list[str] = Field(description="Atomic, independently checkable assertions.")


class ClaimVerification(BaseModel):
    results: list[ClaimCheck]


class AskExtraction(BaseModel):
    asks: list[str]


class AskCoverage(BaseModel):
    results: list[AskCheck]


class ActionLabels(BaseModel):
    intent: str = ""
    actions: list[str]


class BreachReport(BaseModel):
    breaches: list[PolicyBreach] = Field(default_factory=list)


class AuditResult(BaseModel):
    """Output of the grounded audit pass.

    Four sub-tasks in one call. Ideally they would be four independent calls --
    independence is what stops one judgement contaminating another -- but the
    free-tier request budget does not stretch that far. The compromise is that
    the *grounded* work (this) stays separate from the *stylistic* work
    (JudgeVerdict), which is where halo effects do the most damage.
    """

    intent: str = ""
    actions: list[str] = Field(default_factory=list)
    claims: list[ClaimCheck] = Field(default_factory=list)
    asks: list[AskCheck] = Field(default_factory=list)
    breaches: list[PolicyBreach] = Field(default_factory=list)


# ===========================================================================
# Meta-evaluation (validating the metric itself)
# ===========================================================================
class HumanLabel(BaseModel):
    """A hand-assigned quality label used to check the automatic metric."""

    id: str
    example_id: str
    candidate: str
    human_overall: float = Field(description="0-1 overall quality, hand-assigned.")
    human_dimensions: dict[str, float] = Field(default_factory=dict)
    rationale: str = ""
    defect: str = ""


class PerturbationCase(BaseModel):
    """A reference reply with one deliberate, known defect injected."""

    id: str
    example_id: str
    kind: str
    text: str
    expect_drop_in: list[str] = Field(default_factory=list)
    expect_stable: bool = False
    note: str = ""
