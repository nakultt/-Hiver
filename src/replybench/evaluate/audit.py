"""The grounded audit pass: what did this reply claim, commit to, and miss?

This is the half of the metric that does not depend on taste. Every question it
asks has a defensible answer that a second person could check:

  * Which actions does this reply actually take? (closed vocabulary)
  * What factual assertions does it make, and is each one backed by a specific
    knowledge-base row? (fact IDs, quotable)
  * Of the things the customer asked for, which were addressed? (the asks were
    extracted from the inbound email *before* any reply existed)
  * Does it promise anything an agent is not allowed to promise?

Crucially the audit never sees the reference reply. If it did, "unsupported
claim" would collapse into "differs from the reference", and we would be back to
measuring similarity while calling it accuracy.
"""
from __future__ import annotations

from .. import kb
from ..llm import LLM
from ..schemas import AuditResult, Example, SuggestedReply
from ..taxonomy import action_menu, intent_menu, validate_actions

AUDIT_SYS = """You audit a draft support reply. You are strict, specific, and you never
guess. You will NOT be shown any 'correct' reply -- judge only against the knowledge base
and the customer's own email.

=========================== KNOWLEDGE BASE ===========================
These are the ONLY facts about Kestrel that are true. Anything the draft asserts about
Kestrel that is not entailed here is not supported, however plausible it sounds.

{facts}

=================== WHAT AN AGENT MAY PROMISE (internal) ==================
{authority}

============================ YOUR FOUR TASKS ============================

1. ACTIONS -- label what the draft actually does, using only these keys:
{actions}
   An action counts only if the draft genuinely commits to or performs it. Mentioning a
   topic is not taking an action. Also give the single dominant intent from:
{intents}

2. CLAIMS -- extract every factual assertion the draft makes about Kestrel (policies,
   prices, capabilities, limits, certifications, timelines), as atomic statements. Then
   assign each a verdict:
     "supported"     -- entailed by a specific KB row. Give its ID in source_id.
     "contradicted"  -- conflicts with a specific KB row. Give its ID. Quote the conflict
                        in `note` (e.g. 'draft says 60 days, PRICE-01 says 30').
     "unsupported"   -- a factual claim about Kestrel with no KB row behind it.
     "not_checkable" -- not a factual claim about Kestrel. Pleasantries, apologies,
                        questions, and the agent's own commitments ("I've escalated
                        this", "I'll come back to you Thursday") are NOT checkable
                        facts -- they are actions. Use source_id "incoming" for anything
                        that merely repeats what the customer told us.
   Do not invent claims the draft did not make. If it asserts nothing, return [].

3. ASK COVERAGE -- for each listed customer ask, decide:
     "answered"                -- the draft gives a real answer or performs it
     "partially_answered"      -- touched but left materially incomplete
     "deferred_appropriately"  -- not answered, but the draft says why and what happens
                                  next (routing, needing info, needing approval). This
                                  COUNTS as handled.
     "ignored"                 -- not addressed at all
   Quote the relevant fragment of the draft in `evidence`, or "" if ignored.

4. BREACHES -- list anything the draft promises beyond agent authority, or that leaks
   internal policy, discloses other customers, commits to a ship date, quotes Scale
   pricing, or diagnoses a possible incident speculatively. Give the AUTH id as rule_id,
   a severity of low/medium/high, and quote the offending text.
   Return [] if there are none. Do not manufacture breaches.

Return JSON with keys: intent, actions, claims, asks, breaches."""

AUDIT_USER = """CUSTOMER'S EMAIL
From: {name} ({role}), plan: {plan}
Subject: {subject}
---
{incoming}
---

THE CUSTOMER'S ASKS (extracted from the email above before any reply existed):
{asks}

DRAFT REPLY UNDER AUDIT
---
{draft}
---

Audit it."""


async def audit(
    llm: LLM,
    example: Example,
    reply: SuggestedReply,
    *,
    model: str | None = None,
    variant: int = 0,
) -> AuditResult:
    asks = example.customer_asks or ["(none extracted)"]
    numbered = "\n".join(str(i) + ". " + a for i, a in enumerate(asks, 1))

    result = await llm.structured(
        [
            {"role": "system", "content": AUDIT_SYS.format(
                facts=kb.render_public(),
                authority=kb.render_authority(),
                actions=action_menu(),
                intents=intent_menu(),
            )},
            {"role": "user", "content": AUDIT_USER.format(
                name=example.incoming.sender_name,
                role=example.incoming.sender_role or "customer",
                plan=example.incoming.account_plan or "unknown",
                subject=example.incoming.subject,
                incoming=example.incoming.body,
                asks=numbered,
                draft=reply.body,
            )},
        ],
        AuditResult,
        model=model or llm.s.judge_model,
        temperature=0.0,
        max_tokens=3000,
        variant=variant,
    )
    result.actions = validate_actions(result.actions)
    return result
