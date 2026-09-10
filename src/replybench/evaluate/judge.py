"""The rubric judge: the part that is genuinely a matter of taste.

Scope is deliberately narrow. The judge is asked only about things a human
reviewer would also have to eyeball -- register, readability, whether it reads as
sendable -- plus a second, independent opinion on policy safety. Facts, actions
and coverage are handled by the grounded audit, because handing a single model
one big "score this reply out of 10" prompt produces a number that correlates
with length and confidence far more than with quality.

Three anti-bias measures, all measured rather than assumed by
`replybench validate-metric`:

  * Every score must carry a quoted fragment as evidence. Forcing the judge to
    point at the text before scoring suppresses vibes-based grading.
  * Anchored rubric: each band is defined by what it looks like, not by an
    adjective. "0.6" alone means nothing across raters or runs.
  * The judge never sees which system produced the draft, and never sees the
    reference reply. It cannot reward the house favourite.
"""
from __future__ import annotations

from .. import kb
from ..llm import LLM
from ..schemas import Example, JudgeVerdict, SuggestedReply

JUDGE_SYS = """You review draft replies for a support team at Kestrel. You do not know
which system wrote the draft, and you will not be shown a 'correct' answer. Score only
what is in front of you.

COMPANY VOICE (the target register)
{voice}

Score four things from 0.0 to 1.0. Use the anchors -- do not drift toward the middle.

TONE_FIT -- is this the right register for THIS customer in THIS situation?
  1.0  Pitched exactly right. An angry customer is met with directness and ownership,
       not grovelling. A routine question gets a brisk answer, not effusiveness.
  0.7  Acceptable but slightly off -- a shade too formal, or one apology too many.
  0.4  Noticeably wrong: corporate padding, mismatched warmth, or emotional tone-deafness
       (cheerful about an outage; defensive with a reasonable complaint).
  0.1  Would damage the relationship. Cold, robotic, sarcastic, or matching hostility.

CLARITY -- can a busy person read this once and know what happens next?
  1.0  Lead with the answer. Scannable. Every sentence earns its place.
  0.7  Clear but padded, or the answer is buried in the second paragraph.
  0.4  Rambling, repetitive, or structurally confusing.
  0.1  Hard to follow, or so vague the customer learns nothing.

POLICY_SAFETY -- would sending this create a problem for the company?
  1.0  Commits only to what a support agent can actually deliver. Routes what it should.
  0.7  Slightly loose phrasing that could be read as a stronger promise than intended.
  0.4  A real over-promise, an implied guarantee, or an unrequested speculative diagnosis.
  0.1  Commits money, entitlements, dates or confidential information it must not.

HOLISTIC -- forget the sub-scores. Would a good agent send this, and would the customer
be well served? This is your overall judgement and it is allowed to disagree with the
parts.

Then WOULD_SEND, one of:
  "send_as_is"   ready to go untouched
  "light_edit"   a word or two, under 30 seconds of work
  "heavy_edit"   substantially rewritten before sending
  "do_not_send"  wrong, unsafe, or would have to be started over

RULES
- `evidence` must contain at least one VERBATIM fragment quoted from the draft. If you
  cannot quote something that supports your score, your score is wrong.
- Do not reward length. A three-line reply that answers the question fully beats a
  fifteen-line one that also answers it.
- Do not reward confidence. A draft that states something firmly is not thereby better.
- `biggest_problem`: one sentence, the single thing you would fix first. "" if nothing.

Return JSON with keys: tone_fit, clarity, policy_safety, holistic (each an object with
score, reason, evidence), plus would_send and biggest_problem."""

JUDGE_USER = """CUSTOMER'S EMAIL
From: {name} ({role}), plan: {plan}
Subject: {subject}
---
{incoming}
---

DRAFT REPLY
---
{draft}
---

Score it."""


async def judge(
    llm: LLM,
    example: Example,
    reply: SuggestedReply,
    *,
    model: str | None = None,
    temperature: float = 0.0,
    variant: int = 0,
) -> JudgeVerdict:
    return await llm.structured(
        [
            {"role": "system", "content": JUDGE_SYS.format(voice=kb.COMPANY["voice"])},
            {"role": "user", "content": JUDGE_USER.format(
                name=example.incoming.sender_name,
                role=example.incoming.sender_role or "customer",
                plan=example.incoming.account_plan or "unknown",
                subject=example.incoming.subject,
                incoming=example.incoming.body,
                draft=reply.body,
            )},
        ],
        JudgeVerdict,
        model=model or llm.s.judge_model,
        temperature=temperature,
        max_tokens=1800,
        variant=variant,
    )
