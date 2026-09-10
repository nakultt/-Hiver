"""Realistic mess.

A model asked for "a support email" writes a tidy three-paragraph note with a
greeting and a sign-off. Real shared-inbox mail is not like that: it has quoted
trails, mobile footers, forwarded headers, disclaimers nobody reads, and the
actual question buried under four lines of "see below".

This matters for evaluation, not just realism. A generator that only ever sees
clean input looks better than it is, and a *retriever* that indexes 400 lines of
legal disclaimer retrieves garbage. Injecting the mess here is what lets us
claim the parser in `generate/parse.py` is load-bearing.
"""
from __future__ import annotations

import random

MOBILE_SIGS = [
    "\n\nSent from my iPhone",
    "\n\nSent from Outlook for Android",
    "\n\nGet Outlook for iOS",
    "\n\nsent from mobile, excuse typos",
]

LONG_SIGS = [
    (
        "\n\n--\n{name}\n{role}\n{company}\nM: +44 7{d7}\n{company_domain}\n\n"
        "This email and any attachments are confidential and intended solely for the addressee. "
        "If you have received this in error please notify the sender and delete it. "
        "{company} accepts no liability for any damage caused by any virus transmitted by this email."
    ),
    (
        "\n\nBest regards,\n{name}\n{role} | {company}\nT: +44 20 7{d6}  |  {company_domain}\n"
        "Please consider the environment before printing this email."
    ),
]

DISCLAIMER = (
    "\n\n________________________________\n"
    "The information contained in this communication from the sender is confidential. "
    "It is intended solely for use by the recipient and others authorised to receive it."
)


def _digits(rng: random.Random, n: int) -> str:
    return "".join(str(rng.randint(0, 9)) for _ in range(n))


def _typo(body: str, rng: random.Random) -> str:
    swaps = {
        "the": "teh", "and": "adn", "you": "yuo", "with": "wiht",
        "because": "becuase", "receive": "recieve", "definitely": "definately",
        "our": "our", "just": "jsut",
    }
    words = body.split(" ")
    hits = 0
    for i, w in enumerate(words):
        low = w.lower().strip(".,!?")
        if low in swaps and rng.random() < 0.5 and hits < 4:
            words[i] = w.lower().replace(low, swaps[low])
            hits += 1
    out = " ".join(words)
    if rng.random() < 0.5:
        out = out.replace(". ", ". ").replace("I ", "i ")
    return out


def _quoted_trail(body: str, rng: random.Random, agent: str, company: str) -> str:
    prior = rng.choice([
        "Thanks for getting back to me. Let me check with the team and revert.",
        "No problem at all — just let me know once you've had a look.",
        "That's great, thank you. One more thing though (below).",
    ])
    date = "Mon, " + str(rng.randint(1, 28)) + " " + rng.choice(["Jan", "Feb", "Mar", "Apr", "May", "Jun"]) + " 2025 at " + str(rng.randint(9, 17)) + ":" + _digits(rng, 2)
    return (
        body
        + "\n\nOn " + date + ", " + agent + " <support@kestrelhq.com> wrote:\n"
        + "> " + prior + "\n"
        + ">\n> Best,\n> " + agent + "\n> " + company + " Support\n"
    )


def _forwarded(body: str, rng: random.Random, name: str, email: str) -> str:
    return (
        "Hi — forwarding this on, can you help?\n\n"
        "---------- Forwarded message ---------\n"
        "From: " + name + " <" + email + ">\n"
        "Date: " + rng.choice(["Tue", "Wed", "Thu"]) + ", " + str(rng.randint(1, 28)) + " Mar 2025\n"
        "Subject: Re: Kestrel\n"
        "To: ops@" + email.split("@", 1)[1] + "\n\n"
        + body
    )


def _top_posted(body: str, rng: random.Random, agent: str) -> str:
    """The question is at the TOP, under a wall of quoted history.

    This is the case that breaks naive 'take the last paragraph' parsers.
    """
    history = []
    for i in range(rng.randint(2, 4)):
        history.append(
            "> " + rng.choice([
                "Understood, thanks for confirming.",
                "Ok that makes sense. I'll check with the store managers.",
                "Sorry for the slow reply, been a busy week.",
                "Appreciate you looking into this.",
                "Perfect, that's what I needed.",
            ])
            + "\n>\n> On an earlier note, " + agent + " wrote:\n> > "
            + rng.choice([
                "Happy to help — let me know how you get on.",
                "I've made that change for you now.",
                "Here's the guide: https://docs.kestrelhq.com/getting-started",
            ])
            + "\n>"
        )
    return body + "\n\n" + "\n".join(history)


def apply_noise(
    body: str,
    profile: str,
    *,
    rng: random.Random,
    name: str,
    email: str,
    role: str,
    company: str,
) -> str:
    """Return `body` roughed up according to `profile`."""
    domain = email.split("@", 1)[1] if "@" in email else "example.com"
    agent = rng.choice(["Priya Raman", "Tom Whitfield", "Aoife Byrne", "Marcus Hale"])

    if profile == "clean":
        return body
    if profile == "mobile_signature":
        return body + rng.choice(MOBILE_SIGS)
    if profile == "long_signature":
        tpl = rng.choice(LONG_SIGS)
        sig = tpl.format(
            name=name, role=role, company=company, company_domain=domain,
            d7=_digits(rng, 7), d6=_digits(rng, 6),
        )
        return body + sig + (DISCLAIMER if rng.random() < 0.4 else "")
    if profile == "quoted_trail":
        return _quoted_trail(body, rng, agent, company)
    if profile == "forwarded":
        return _forwarded(body, rng, name, email)
    if profile == "typos":
        return _typo(body, rng)
    if profile == "top_posted_thread":
        return _top_posted(body, rng, agent)
    return body
