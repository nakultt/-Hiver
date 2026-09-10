"""Suggested-reply generation, plus the ablations and baselines it is measured against.

Design choice: retrieval-augmented prompting, not fine-tuning
------------------------------------------------------------
Fine-tuning on the corpus would bake in the house voice more strongly, and at
tens of thousands of pairs it would probably win. It was the wrong call here:

  * Policy changes weekly in a real support org. A fine-tune has to be re-run to
    learn that the refund window moved; a retrieval system just needs the KB row
    edited. Grounding facts in a prompt is the difference between a stale model
    and a stale row.
  * Fine-tuning gives no citation trail. This whole project turns on being able
    to say *why* a reply is wrong, and "the weights preferred it" is not a why.
  * ~60 exemplars is nowhere near fine-tuning scale, and pretending otherwise
    would be the dishonest option.

So the corpus is used two ways: as few-shot exemplars retrieved per email (voice
and shape), and as the source of which KB facts past replies actually cited
(grounding). The `no_retrieval` and `no_kb` ablations exist to show what each of
those is actually worth rather than asserting it.
"""
from __future__ import annotations

import time
from typing import Sequence

from .. import kb
from ..llm import LLM
from ..schemas import Email, Example, GenerationRecord, SuggestedReply
from ..taxonomy import ACTIONS, INTENTS, action_menu, validate_actions
from .parse import clean
from .retrieve import Retriever

SYSTEMS = ("main", "no_retrieval", "no_kb", "nearest_neighbour", "template")

LLM_SYSTEMS = ("main", "no_retrieval", "no_kb")

SYSTEM_DESCRIPTIONS = {
    "main": "Retrieved exemplars + retrieved KB facts + authority rules.",
    "no_retrieval": "KB facts + authority rules, no exemplars. Isolates what retrieval buys.",
    "no_kb": "Exemplars only, no KB facts. Isolates what grounding buys.",
    "nearest_neighbour": "Returns the most similar past reply verbatim. No LLM.",
    "template": "One fixed acknowledgement for every email. No LLM. The floor.",
}

TEMPLATE_REPLY = """Hi there,

Thanks for getting in touch with Kestrel Support, and apologies for any inconvenience.

We've received your message and a member of our team will review it and get back to you
as soon as possible. If your query is urgent, please don't hesitate to let us know.

We appreciate your patience and value you as a customer.

Best regards,
Kestrel Support"""


BASE_SYS = """You draft the suggested reply that a human support agent at Kestrel sees
pre-filled in their inbox. They will read it, maybe edit it, and send it. Write what should
actually be sent.

COMPANY VOICE
{voice}

{facts_block}{authority_block}RULES
- Answer EVERY question the customer asked. If one cannot be answered, say why.
- Never assert a fact you have not been given. If you do not know, say so or ask.
- Never promise something beyond your authority. Route it and say you are routing it.
- Do not pad. Three lines is the right length when three lines is the answer.
- Never invent an order number, a date, a name, or a ship date.
- Sign off as Priya Raman / Kestrel Support.

OUTPUT -- return JSON with exactly these keys:
  body                 the reply text, ready to send (use \\n for line breaks)
  intent               one of: {intent_keys}
  actions              list of action keys you took, from the vocabulary below
  cited_facts          list of fact IDs (e.g. "REF-01") you relied on; [] if none
  confidence           0.0-1.0, how sure you are this is send-ready
  needs_human_review   true if a human must check before sending
  open_questions       anything you could not resolve

ACTION VOCABULARY
{actions}"""

FACTS_BLOCK = """WHAT IS TRUE (the only facts you may assert)
{facts}

"""

AUTHORITY_BLOCK = """WHAT YOU MAY PROMISE (internal -- never quote verbatim to a customer)
{authority}

"""

EXEMPLAR_BLOCK = """Here are past emails from this inbox and the replies our team sent.
Match their voice, structure and length. Do NOT copy their facts -- those belonged to a
different customer.

{exemplars}

============================================================
"""

USER_TMPL = """{exemplar_block}NEW EMAIL TO REPLY TO

From: {name} <{email}>
Role: {role}
Their plan: {plan}
Subject: {subject}

---
{body}
---

Draft the suggested reply."""


def _format_exemplars(pairs: Sequence[tuple[Example, float]]) -> str:
    out = []
    for i, (ex, _score) in enumerate(pairs, 1):
        out.append(
            "--- PAST EXAMPLE " + str(i) + " ---\n"
            "Subject: " + ex.incoming.subject + "\n"
            "Customer wrote:\n" + clean(ex.incoming.body)[:900] + "\n\n"
            "We replied:\n" + ex.reply.body[:1200]
        )
    return "\n\n".join(out)


async def generate_one(
    llm: LLM,
    example: Example,
    retriever: Retriever,
    system: str = "main",
    *,
    model: str | None = None,
    k: int = 3,
) -> GenerationRecord:
    started = time.time()
    inc: Email = example.incoming

    # ---------------------------------------------------- non-LLM baselines
    if system == "template":
        return GenerationRecord(
            example_id=example.id, system=system,
            reply=SuggestedReply(body=TEMPLATE_REPLY, intent="acknowledge_and_hold",
                                 confidence=0.0, needs_human_review=True),
            latency_s=0.0,
        )

    if system == "nearest_neighbour":
        hits = retriever.search(inc.subject, inc.body, k=1)
        body = hits[0][0].reply.body if hits else TEMPLATE_REPLY
        return GenerationRecord(
            example_id=example.id, system=system,
            reply=SuggestedReply(body=body, confidence=0.0, needs_human_review=True),
            retrieved_ids=[hits[0][0].id] if hits else [],
            latency_s=0.0,
        )

    # ------------------------------------------------------- LLM generators
    use_exemplars = system in ("main", "no_kb")
    use_facts = system in ("main", "no_retrieval")

    hits = retriever.search(inc.subject, inc.body, k=k) if use_exemplars else []
    facts = retriever.facts(inc.subject, inc.body, [e for e, _ in hits]) if use_facts else []

    facts_block = FACTS_BLOCK.format(facts=kb.render(facts)) if use_facts else ""
    authority_block = AUTHORITY_BLOCK.format(authority=kb.render_authority()) if use_facts else ""
    exemplar_block = EXEMPLAR_BLOCK.format(exemplars=_format_exemplars(hits)) if hits else ""

    sys_prompt = BASE_SYS.format(
        voice=kb.COMPANY["voice"],
        facts_block=facts_block,
        authority_block=authority_block,
        intent_keys=", ".join(sorted(INTENTS)),
        actions=action_menu(),
    )
    user_prompt = USER_TMPL.format(
        exemplar_block=exemplar_block,
        name=inc.sender_name, email=inc.sender_email, role=inc.sender_role or "customer",
        plan=inc.account_plan or "unknown", subject=inc.subject, body=inc.body,
    )

    try:
        reply = await llm.structured(
            [{"role": "system", "content": sys_prompt}, {"role": "user", "content": user_prompt}],
            SuggestedReply, model=model or llm.s.gen_model, temperature=0.3, max_tokens=1400,
        )
    except Exception as exc:  # noqa: BLE001 - one bad row must not kill a run
        return GenerationRecord(
            example_id=example.id, system=system,
            reply=SuggestedReply(body="", needs_human_review=True),
            error=str(exc)[:300], latency_s=time.time() - started,
        )

    reply.actions = validate_actions(reply.actions)
    reply.cited_facts = kb.valid_ids(reply.cited_facts)
    return GenerationRecord(
        example_id=example.id, system=system, reply=reply,
        retrieved_ids=[e.id for e, _ in hits],
        retrieved_facts=[f.id for f in facts],
        latency_s=time.time() - started,
    )
