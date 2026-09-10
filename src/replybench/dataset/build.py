"""Build the corpus.

Pipeline per row (four separate model calls, on purpose):

  1. CUSTOMER writes the inbound email. It is given a persona, an emotional
     register and a one-line situation -- but *not* the knowledge base. If the
     customer knew our policies it would phrase questions in our vocabulary and
     retrieval would look magical for the wrong reason.
  2. Mess is injected mechanically (quoted trails, mobile footers, typos).
  3. AGENT writes the reply. It gets the KB, the authority rules and the voice
     guide, and it sees the *noisy* email -- the same thing the generator will
     see at inference.
  4. ANNOTATOR labels the reply's intent and actions, seeing only the reply and
     the closed action vocabulary. Labels describe what the reply *does*, not
     what step 3 was told to aim for; the gap between the two is reported by
     `dataset-report` as a dataset-quality signal.
  5. ASK-EXTRACTOR lists the customer's atomic requests from the inbound email
     ALONE.

Step 5 must not see the reply. If it did, it would only ever list asks the reply
happened to address, and `completeness` would score ~1.0 for every system
forever. This is the single easiest way to accidentally build a metric that
cannot fail, and it is why the call is separate.
"""
from __future__ import annotations

import asyncio
import json
import random
import re
from typing import Any

from .. import kb
from ..config import DATA_DIR
from ..llm import LLM, gather_capped
from ..schemas import AskExtraction, ActionLabels, Email, Example, Reply
from ..taxonomy import action_menu, intent_menu, validate_actions
from .handwritten import HARD_CASES
from .noise import apply_noise
from .scenarios import NOISE_PROFILES, PERSONAS, PLANS, REGISTERS, SITUATIONS

# Situations reserved for the test split only. They probe whether the system
# generalises to a topic it has never retrieved an exemplar for.
OOD_SITUATIONS = {
    "vat_reverse_charge",
    "audit_log_request",
    "forecasting_accuracy",
    "locked_out_owner",
    "renewal_price_increase",
}

COMPANIES = [
    ("Fold & Fern", "foldandfern.co.uk"), ("Saltmarsh Kitchens", "saltmarshkitchens.com"),
    ("The Holly Bush Group", "hollybushgroup.co.uk"), ("Northgate Hospitality", "northgatehosp.com"),
    ("Brightmoor Retail", "brightmoor.co"), ("Wildwood Inns", "wildwoodinns.com"),
    ("Corrigan Bakeries", "corriganbakeries.co.uk"), ("Meridian Leisure", "meridianleisure.com"),
    ("Arbor Meadow", "arbormeadowgroup.com"), ("Lockkeeper Group", "lockkeepergroup.com"),
    ("Hartley & Co", "hartleyandco.com"), ("West Quay Retail", "westquayretail.com"),
    ("Fenchurch Kitchens", "fenchurchkitchens.com"), ("Caldera Retail", "caldera-retail.com"),
    ("Stonebridge Inns", "stonebridgeinns.co.uk"), ("Pelham Coffee", "pelhamcoffee.com"),
    ("Cobblestone Group", "cobblestone-group.com"), ("Harlow & Vine", "harlowandvine.co.uk"),
    ("Ravensworth Leisure", "ravensworthleisure.com"), ("Tidewater Hospitality", "tidewaterhosp.com"),
]

FIRST = ["Aisha", "Tom", "Nadia", "Rob", "Elena", "Yusuf", "Caroline", "Mo", "Jules", "Priti",
         "Callum", "Sofia", "Hugh", "Deborah", "Marek", "Grace", "Owen", "Farida", "Neil", "Bea",
         "Tomas", "Lucy", "Rashid", "Hannah", "Gavin", "Iris", "Dmitri", "Nell", "Sean", "Ava"]
LAST = ["Okafor", "Whitfield", "Ellery", "Danes", "Sarkis", "Rahman", "Feld", "Farouk", "Amara",
        "Shah", "Beck", "Corrigan", "Bellamy", "Voss", "Kowalski", "Adeyemi", "Pryce", "Haddad",
        "Brannigan", "Lindqvist", "Petrie", "Moss", "Iqbal", "Doyle", "Traynor", "Nkemdirim"]


# ===========================================================================
# Prompts
# ===========================================================================
CUSTOMER_SYS = """You write realistic inbound emails to a B2B SaaS support inbox.

You are a CUSTOMER or PROSPECT. You do NOT work at the vendor and you do not know
their internal policies, plan names beyond what you already pay for, or their
support vocabulary. Write the way a real person in this role writes when they
have a problem and limited time.

Hard rules:
- Write ONLY the email body. No subject line inside the body, no commentary.
- Do NOT sound like marketing copy or like a support ticket template.
- Do NOT state the vendor's policies back to them; you are asking, not reciting.
- Vary structure. Real emails are often one paragraph. Some have no greeting.
  Some open with two lines of context nobody needed.
- Never mention that this is synthetic, an example, or a scenario."""

CUSTOMER_USER = """Write the email body.

WHO YOU ARE: {who}
YOUR WRITING STYLE: {style}
YOUR MOOD: {register_desc}
YOUR COMPANY: {company}
YOUR RELATIONSHIP TO THE VENDOR (Kestrel, staff scheduling software): {plan_desc}

WHAT PROMPTED YOU TO WRITE:
{seed}

LENGTH: about {words} words. {shape}

Write only the body text."""

SUBJECT_SYS = """You write the subject line a real person would put on an email they just wrote.
Return ONLY the subject line, nothing else. Match their register: rushed people write
lowercase fragments, formal people write full noun phrases, angry people sometimes shout.
Never exceed 9 words."""

AGENT_SYS = """You are an experienced support agent at Kestrel, writing the reply that
actually gets sent. Your reply becomes the gold standard in a training corpus, so it must
be what a strong agent would really send -- not a maximal, hedged, everything-covered essay.

COMPANY VOICE
{voice}

WHAT IS TRUE (the only facts you may assert)
{public_facts}

WHAT YOU ARE ALLOWED TO PROMISE (internal -- never quote these verbatim to a customer)
{authority}

RULES
- Never assert a fact that is not in the list above. If you do not know, say so, or ask.
- Never exceed your authority. If something needs approval or another team, say that plainly
  and route it -- do not promise it.
- Answer EVERY question the customer asked. If you cannot answer one, say why.
- Do not pad. A three-line reply is correct when three lines is the answer.
- Sign off as {agent} / Kestrel Support.
- Write ONLY the reply body. No subject line, no "Here is the reply", no commentary."""

AGENT_USER = """Reply to this email.

From: {name} <{email}>
Role: {role}
Their plan: {plan}
Subject: {subject}

---
{body}
---

Write the reply body only."""

ANNOTATE_SYS = """You label support replies against a fixed vocabulary. You see ONLY the
reply. Label what the reply actually DOES, not what it should have done.

INTENTS (choose exactly one -- the reply's dominant purpose):
{intents}

ACTIONS (choose every one the reply actually takes; an action counts only if the reply
genuinely commits to or performs it, not if it merely alludes to the topic):
{actions}

Return JSON: {{"intent": "<intent key>", "actions": ["<action key>", ...]}}
Use only keys from the lists. An empty action list is valid for a pure acknowledgement."""

ASKS_SYS = """You extract what a customer actually asked for.

You see ONLY the inbound email -- never a reply. List every distinct request, question or
expectation the sender is raising, as short standalone statements from the sender's point
of view. Split compound sentences into separate asks. Include implicit but clear
expectations (a deadline, a demand not to be put on a call). Ignore pleasantries,
signatures, disclaimers and anything inside a quoted (">") trail unless the sender is
explicitly re-raising it.

Return JSON: {{"asks": ["...", "..."]}} -- typically 1 to 5 items."""


# ===========================================================================
# Sampling
# ===========================================================================
def _plan_desc(plan: str) -> str:
    return {
        "Core": "You are an existing customer on their cheapest paid plan (Core).",
        "Growth": "You are an existing customer on their mid plan (Growth).",
        "Scale": "You are a large existing customer on their enterprise plan (Scale).",
        "trial": "You are in the middle of a free trial.",
        "prospect": "You are evaluating them and are not yet a customer.",
    }[plan]


def _shape(rng: random.Random) -> str:
    return rng.choice([
        "Use a greeting and a sign-off.",
        "No greeting -- dive straight in.",
        "Sign off with just your first name.",
        "Include one irrelevant sentence of context before the real question.",
        "Ask the main thing first, then add a smaller second question at the end.",
        "Write it as a single unbroken paragraph.",
        "Use a short bulleted or numbered list.",
    ])


def _cells(n: int, *, seed: int, pool: list[Any]) -> list[dict[str, Any]]:
    """Deterministically sample n grid cells, cycling situations for balance."""
    rng = random.Random(seed)
    out: list[dict[str, Any]] = []
    for i in range(n):
        sit = pool[i % len(pool)]
        persona = PERSONAS[rng.randrange(len(PERSONAS))]
        register = REGISTERS[rng.randrange(len(REGISTERS))]
        plan = rng.choice(PLANS)
        # Keep plan coherent with the situation where it obviously matters.
        if sit.key in ("trial_expiry_extension", "trial_data_after_expiry"):
            plan = "trial"
        if sit.key in ("pricing_small_team", "demo_request", "seat_minimum_pushback"):
            plan = "prospect"
        if sit.key in ("api_on_core", "sso_availability", "geofence_request", "audit_log_request"):
            plan = "Core"
        first, last = rng.choice(FIRST), rng.choice(LAST)
        company, domain = rng.choice(COMPANIES)
        out.append({
            "situation": sit,
            "persona": persona,
            "register": register,
            "plan": plan,
            "noise": rng.choice(NOISE_PROFILES),
            "words": rng.choice([35, 45, 60, 60, 80, 80, 110, 150]),
            "shape": _shape(rng),
            "name": first + " " + last,
            "email": first.lower() + "." + last.lower() + "@" + domain,
            "company": company,
            "agent": rng.choice(["Priya Raman", "Tom Whitfield", "Aoife Byrne", "Marcus Hale"]),
            "seed": rng.randrange(10**6),
        })
    return out


# ===========================================================================
# Row construction
# ===========================================================================
async def _build_row(
    llm: LLM, cell: dict[str, Any], idx: int, split: str, reply_model: str
) -> Example | None:
    """Build one row.

    `reply_model` differs by split on purpose. Train replies are *exemplars* --
    diversity matters more than polish, so they use the cheap model. Test replies
    are *references* that everything downstream is scored against, so they get
    the better model and a scarcer quota.
    """
    sit = cell["situation"]
    rng = random.Random(cell["seed"])
    dm = llm.s.data_model

    # 1 -- inbound email, written without knowledge of the KB
    body = await llm.chat(
        [
            {"role": "system", "content": CUSTOMER_SYS},
            {"role": "user", "content": CUSTOMER_USER.format(
                who=cell["persona"]["who"], style=cell["persona"]["style"],
                register_desc=cell["register"]["desc"], company=cell["company"],
                plan_desc=_plan_desc(cell["plan"]), seed=sit.seed,
                words=cell["words"], shape=cell["shape"],
            )},
        ],
        model=llm.s.gen_model, temperature=1.0, max_tokens=700, variant=idx,
    )
    body = body.strip()
    if len(body) < 40:
        return None

    subject = (await llm.chat(
        [
            {"role": "system", "content": SUBJECT_SYS},
            {"role": "user", "content": "Mood: " + cell["register"]["key"] + "\n\nEmail:\n" + body},
        ],
        model=llm.s.gen_model, temperature=0.9, max_tokens=40, variant=idx,
    )).strip().strip('"').split("\n")[0][:120]

    # 2 -- mechanical mess
    noisy = apply_noise(
        body, cell["noise"], rng=rng, name=cell["name"], email=cell["email"],
        role=cell["persona"]["who"], company=cell["company"],
    )

    # 3 -- the gold reply
    relevant = kb.facts_for_areas(sit.areas) or kb.PUBLIC_FACTS
    public = kb.render([f for f in relevant if not f.internal]) + "\n" + kb.render(
        [f for f in kb.PUBLIC_FACTS if f.area in ("pricing", "sla") and f not in relevant][:6]
    )
    reply_body = await llm.chat(
        [
            {"role": "system", "content": AGENT_SYS.format(
                voice=kb.COMPANY["voice"], public_facts=public,
                authority=kb.render_authority(), agent=cell["agent"],
            )},
            {"role": "user", "content": AGENT_USER.format(
                name=cell["name"], email=cell["email"], role=cell["persona"]["who"],
                plan=cell["plan"], subject=subject, body=noisy,
            )},
        ],
        model=reply_model, temperature=0.6, max_tokens=1100, variant=idx,
    )
    reply_body = reply_body.strip()
    if len(reply_body) < 40:
        return None

    # 4 + 5 -- labels, and asks extracted from the EMAIL ONLY (never the reply)
    labels, asks = await asyncio.gather(
        llm.structured(
            [
                {"role": "system", "content": ANNOTATE_SYS.format(
                    intents=intent_menu(), actions=action_menu())},
                {"role": "user", "content": "REPLY:\n---\n" + reply_body + "\n---"},
            ],
            ActionLabels, model=dm, temperature=0.0, max_tokens=700, variant=idx,
        ),
        llm.structured(
            [
                {"role": "system", "content": ASKS_SYS},
                {"role": "user", "content": "Subject: " + subject + "\n\n---\n" + noisy + "\n---"},
            ],
            AskExtraction, model=dm, temperature=0.0, max_tokens=700, variant=idx,
        ),
    )

    cited = kb.valid_ids(re.findall(r"\b[A-Z]{3,6}-\d{2}\b", reply_body))
    tags = [sit.key, cell["persona"]["key"], cell["register"]["key"], cell["noise"], cell["plan"]]
    if sit.key in OOD_SITUATIONS:
        tags.append("ood")

    return Example(
        id=split + "-" + format(idx, "04d") + "-" + sit.key,
        split=split,  # type: ignore[arg-type]
        intent=labels.intent if labels.intent else sit.intent,
        difficulty=sit.difficulty,
        tags=tags,
        incoming=Email(
            subject=subject, body=noisy, sender_name=cell["name"],
            sender_email=cell["email"], sender_role=cell["persona"]["who"],
            account_plan=cell["plan"],
        ),
        reply=Reply(
            body=reply_body, author=cell["agent"],
            actions=validate_actions(labels.actions), cited_facts=cited,
        ),
        customer_asks=[a.strip() for a in asks.asks if a.strip()][:8],
        key_facts=sit.key_facts,
        provenance={
            "source": "synthetic",
            "situation": sit.key,
            "intended_actions": sit.expected_actions,
            "customer_model": llm.s.gen_model,
            "agent_model": reply_model,
            "noise_profile": cell["noise"],
            "register": cell["register"]["key"],
        },
    )


# ===========================================================================
# Near-duplicate control
# ===========================================================================
def _shingles(text: str, k: int = 5) -> set[str]:
    words = re.findall(r"[a-z0-9]+", text.lower())
    return {" ".join(words[i : i + k]) for i in range(max(0, len(words) - k + 1))}


def jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def drop_near_duplicates(
    examples: list[Example], *, within: float = 0.5, against: list[Example] | None = None,
    against_thresh: float = 0.4,
) -> tuple[list[Example], list[str]]:
    """Remove rows too similar to each other, or to the training set.

    Test-train leakage is the classic way to make a retrieval system look
    brilliant: if a test email is a paraphrase of a training email, the model is
    copying, not generalising. We measure it and cut it rather than hoping.
    """
    ref = [_shingles(e.incoming.body) for e in (against or [])]
    kept: list[Example] = []
    kept_sh: list[set[str]] = []
    dropped: list[str] = []
    for ex in examples:
        sh = _shingles(ex.incoming.body)
        if any(jaccard(sh, r) > against_thresh for r in ref):
            dropped.append(ex.id + " (leak vs train)")
            continue
        if any(jaccard(sh, k) > within for k in kept_sh):
            dropped.append(ex.id + " (dup within split)")
            continue
        kept.append(ex)
        kept_sh.append(sh)
    return kept, dropped


# ===========================================================================
# Entry point
# ===========================================================================
async def build_dataset(n_train: int = 60, n_test: int = 16, seed: int = 20250911) -> dict[str, Any]:
    train_pool = [s for s in SITUATIONS if s.key not in OOD_SITUATIONS]
    test_pool = SITUATIONS  # test sees held-out situations too

    async with LLM() as llm:
        if not llm.s.live:
            print("!! LLM_BACKEND=mock: this will produce placeholder text, not a usable corpus.")

        train_cells = _cells(n_train, seed=seed, pool=train_pool)
        test_cells = _cells(n_test, seed=seed + 977, pool=test_pool)

        print("Building train split (" + str(n_train) + " rows, 5 calls each)...")
        train_rows = await gather_capped(
            [_build_row(llm, c, i, "train", llm.s.data_model)
             for i, c in enumerate(train_cells)], desc="train")
        print("Building test split (" + str(n_test) + " rows)...")
        test_rows = await gather_capped(
            [_build_row(llm, c, i, "test", llm.s.strong_model)
             for i, c in enumerate(test_cells)], desc="test")

        usage = llm.usage.as_dict()

    train = [r for r in train_rows if r]
    test = [r for r in test_rows if r]

    train, dropped_tr = drop_near_duplicates(train)
    test, dropped_te = drop_near_duplicates(test, against=train)

    examples = train + test + HARD_CASES
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    path = DATA_DIR / "dataset.jsonl"
    with path.open("w", encoding="utf-8") as fh:
        for ex in examples:
            fh.write(json.dumps(ex.model_dump(), ensure_ascii=False) + "\n")

    meta = {
        "seed": seed,
        "counts": {"train": len(train), "test": len(test), "hard": len(HARD_CASES)},
        "dropped_near_duplicates": dropped_tr + dropped_te,
        "ood_situations": sorted(OOD_SITUATIONS),
        "llm_usage": usage,
        "models": {"customer": "gen_model", "train_reply": "data_model", "test_reply": "strong_model"},
    }
    (DATA_DIR / "dataset_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta


def load_dataset(path: Any = None) -> list[Example]:
    p = path or (DATA_DIR / "dataset.jsonl")
    rows = []
    with open(p, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(Example.model_validate_json(line))
    return rows
