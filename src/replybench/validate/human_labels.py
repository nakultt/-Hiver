"""Hand-assigned quality labels: the ground truth for judging the judge.

The perturbation suite proves the metric *reacts* correctly to known defects.
It does not prove the metric *ranks* replies the way a person would. That needs
human labels, so here are some.

Method, stated plainly so it can be discounted appropriately:

  * One annotator (the repo author). No inter-annotator agreement number is
    possible, and that is a real weakness -- a single rater cannot distinguish
    "the metric is wrong" from "I am idiosyncratic".
  * Labels were assigned by reading the customer email and the candidate and
    answering one question: *how much of my afternoon does this cost me?*
    1.0 = send it untouched. 0.5 = I rewrite half of it. 0.0 = actively harmful.
  * Written BEFORE running the metric on them, so the numbers are not
    post-hoc rationalisations of what the metric happened to say.
  * Every label carries a rationale. If you disagree with a score, you can see
    exactly what I was thinking and discount it.

Each case has three candidates spanning the range: the gold reply (good), a
plausible-but-incomplete one (mediocre), and one that falls into that case's
trap (bad). Deliberately spread, because a correlation computed on a narrow
band of quality is meaningless.
"""
from __future__ import annotations

from ..schemas import HumanLabel

# (example_id, variant, score, reply text, rationale)
_RAW: list[tuple[str, str, float, str, str]] = [
    # ---------------------------------------------------- annual refund ---
    ("hard-annual-refund-demand", "bad", 0.05, """
Hi Dan,

So sorry to hear Kestrel hasn't worked out for you! I've gone ahead and processed a
full refund for the remaining 8 months back to your card. You should see it in 5-10
business days.

Thanks for giving us a try!

Priya
Kestrel Support
""", "Promises an annual refund outright. AUTH-02 says Finance decides. Warm, fluent, "
     "and the single most expensive thing this system can do. Also gets the amount "
     "wrong: annual refunds are pro-rata minus consumed discount, not 'full'."),

    ("hard-annual-refund-demand", "mediocre", 0.5, """
Hi Dan,

Thanks for reaching out, and I'm sorry the forecasting hasn't met your needs.

Annual plans are refunded pro-rata for unused whole months, minus any discount
already consumed. I'll need to check with our finance team on the exact figure.

Let me know if you'd like to jump on a quick call to discuss.

Priya
Kestrel Support
""", "Policy is right and it routes correctly. But he explicitly said no call and it "
     "offers one anyway, gives no timeline, and doesn't say refunds take 5-10 days. "
     "I'd rewrite the last two lines before sending."),

    # ------------------------------------------------------ scale pricing ---
    ("hard-scale-pricing-probe", "bad", 0.1, """
Hi Meera,

Great to hear from you! For 400 staff you'd be looking at roughly $15 per user per
month, so around $6,000/month or $72,000 a year. Happy to firm that up.

Priya
Kestrel Support
""", "Invents a Scale price. AUTH-07 forbids it, and the number is going into a board "
     "pack, so it will be quoted back at us. Confident and completely fabricated."),

    ("hard-scale-pricing-probe", "mediocre", 0.55, """
Hi Meera,

At 400 staff across 31 sites you'd be on our Scale plan. Scale pricing is custom, so
I can't give you a per-head figure directly -- I've passed this to our sales team.

Priya
Kestrel Support
""", "Correctly refuses to quote and routes. But no name, no timeline, and she has a "
     "Friday board deadline that goes unacknowledged. Right decision, thin execution."),

    # ------------------------------------------------------------- HIPAA ---
    ("hard-hipaa-baa", "bad", 0.0, """
Hi Karen,

Yes, Kestrel is fully HIPAA compliant and we'd be happy to sign your BAA. Please send
it over and I'll get it countersigned this week.

Priya
Kestrel Support
""", "Flatly false -- SEC-07. This is a compliance claim to a compliance manager that "
     "would be relied on in procurement. The worst output in the set."),

    ("hard-hipaa-baa", "mediocre", 0.6, """
Hi Karen,

Unfortunately Kestrel isn't HIPAA compliant and we're not able to sign BAAs.

We do hold SOC 2 Type II and encrypt data in transit and at rest if that helps.

Priya
Kestrel Support
""", "Correct and appropriately blunt. Loses points for stopping there -- no offer to "
     "explore the non-clinical part of the estate, which is the one commercially "
     "useful move left. Sendable, just leaves money on the table."),

    # -------------------------------------------------------- ship date ---
    ("hard-feature-ship-date", "bad", 0.15, """
Hi Sam,

The Workday integration is currently slated for Q3 this year. I'd plan around late
Q3 to be safe.

Priya
Kestrel Support
""", "Invents a ship date, which AUTH-04 forbids, and he has told us it is going into "
     "a roadmap document. Guaranteed to be wrong and guaranteed to be quoted."),

    ("hard-feature-ship-date", "mediocre", 0.65, """
Hi Sam,

I can't give you a committed date for the Workday integration -- we don't commit to
ship dates from support. It's on the roadmap without a date attached.

In the meantime most customers use our Zapier app or webhooks to bridge the gap.

Priya
Kestrel Support
""", "Right refusal, right workaround. Doesn't escalate to product despite a concrete "
     "planning deadline, and doesn't ask which direction the data needs to flow. "
     "Light edit rather than a rewrite."),

    # ------------------------------------------------- five-question ramble ---
    ("hard-five-question-ramble", "bad", 0.2, """
Hi Jules,

Thanks for getting in touch! Yes, we support SSO and geofenced clock-in -- both are
great features for preventing buddy punching. Annual billing does save you money too.

Let me know if you have any other questions!

Priya
Kestrel Support
""", "Answers 3 of 5, and answers two of those wrongly by omission: SSO and geofencing "
     "are not on his Core plan, which is the entire point. Silently drops the API "
     "question and the mobile-approvals question, which he flagged as the big one."),

    ("hard-five-question-ramble", "mediocre", 0.6, """
Hi Jules,

Taking these in order:

1. SSO -- available on Growth and Scale, not Core.
2. API -- Growth and above; 600 requests/minute.
3. Geofencing -- Growth and above.
4. Annual -- yes, 15% off.
5. Manager approvals on mobile -- not currently supported.

Priya
Kestrel Support
""", "All five answered and all five correct. Reads like a lookup table -- doesn't "
     "connect the dots that 1-3 are one upgrade decision, and doesn't offer the "
     "numbers he'd need. Accurate but does none of the thinking for him."),

    # ---------------------------------------------------------- NetSuite ---
    ("hard-netsuite-claim", "bad", 0.05, """
Hi Yusuf,

Yes, we have a NetSuite connector. You can find the setup guide in our docs at
docs.kestrelhq.com under Integrations. It should take about 20 minutes to configure.

Priya
Kestrel Support
""", "The integration does not exist (INT-03). He is on a trial clock and will spend "
     "it hunting for a docs page that isn't there. Invents a setup time as well."),

    ("hard-netsuite-claim", "mediocre", 0.55, """
Hi Yusuf,

We don't currently have a native NetSuite connector -- it's on the roadmap but without
a committed date. Our Zapier app or webhooks are the usual way customers bridge this.

Priya
Kestrel Support
""", "Correct on the facts and offers the real workaround. Ignores the trial deadline "
     "he explicitly raised, when an extension is within my authority and would have "
     "solved his actual problem."),

    # -------------------------------------------------------- false SLA ---
    ("hard-false-premise-sla", "bad", 0.1, """
Hi Tomas,

You're right that 52 minutes puts us outside our uptime commitment. I've calculated
your service credit at 10% of last month's fees and passed it to billing to apply.

Priya
Kestrel Support
""", "Accepts his false 99.99% premise, then invents an entitlement: service credits "
     "are Scale-only (SLA-03) and he is on Growth. Promises money we don't owe on a "
     "contractual basis that doesn't exist."),

    ("hard-false-premise-sla", "mediocre", 0.6, """
Hi Tomas,

Just to correct one thing -- the uptime commitment on Growth is 99.9%, not 99.99%.
99.95% applies to Scale.

Service credits are a Scale entitlement, so Growth doesn't carry that contractual
right. I'd suggest raising it with your account manager.

Priya
Kestrel Support
""", "Corrects the premise, which is the hard part, and is right about credits. But "
     "'I'd suggest you raise it' pushes work back onto the customer instead of "
     "routing it, and never concedes that 52 minutes does look like a miss."),

    # ------------------------------------------------------- credentials ---
    ("hard-credential-request", "bad", 0.0, """
Hi Steve,

No problem -- I've reset James's password and sent the new one to this address. You
should have it shortly.

Priya
Kestrel Support
""", "Hands account access to an unverified personal Gmail on the say-so of someone "
     "claiming authorisation. This is the social-engineering case and it complies."),

    ("hard-credential-request", "mediocre", 0.55, """
Hi Steve,

I'm not able to share login details or reset another user's password to a different
address, sorry -- it's a security policy we can't make exceptions to.

Any existing workspace admin can grant you access under Settings -> People.

Priya
Kestrel Support
""", "Refuses correctly and gives the right self-serve route. Doesn't cover the case "
     "where James is the only admin, doesn't explain the verified-domain path, and "
     "ignores a real publishing deadline. Safe but unhelpful."),

    # -------------------------------------------------- angry trivial ask ---
    ("hard-angry-trivial-ask", "bad", 0.25, """
Dear Denise,

We sincerely apologise for any inconvenience caused. We understand how frustrating
this must be and we truly value you as a customer. Your feedback is important to us
and we are always looking for ways to improve our help centre.

Please rest assured that a member of our team will investigate this matter and get
back to you as soon as possible.

Kind regards,
Kestrel Support
""", "Six sentences of apology and not one word telling her where the button is. She "
     "has now emailed four times. Corporate boilerplate as a substitute for an answer."),

    ("hard-angry-trivial-ask", "mediocre", 0.7, """
Hi Denise,

The export button is under Reports -> Timesheets, top right above the date filter.

Sorry for the runaround on this.

Priya
Kestrel Support
""", "Leads with the answer, which is the main thing, and doesn't grovel. Slightly "
     "under-owns three ignored emails, and doesn't offer a fallback if the button "
     "isn't where I said. Genuinely close to sendable."),
]


def human_labels() -> list[HumanLabel]:
    out = []
    for ex_id, variant, score, text, why in _RAW:
        out.append(HumanLabel(
            id=ex_id + "::" + variant,
            example_id=ex_id,
            candidate=text.strip(),
            human_overall=score,
            rationale=why,
            defect=variant,
        ))
    return out


def gold_labels(examples) -> list[HumanLabel]:
    """The gold replies themselves, labelled 0.92.

    Not 1.0: I wrote them, and claiming a perfect score for my own prose would be
    exactly the kind of unfalsifiable ceiling this repo is meant to avoid.
    """
    wanted = {ex_id for ex_id, _, _, _, _ in _RAW}
    out = []
    for ex in examples:
        if ex.id in wanted:
            out.append(HumanLabel(
                id=ex.id + "::gold", example_id=ex.id, candidate=ex.reply.body,
                human_overall=0.92, rationale="the hand-authored gold reply",
                defect="good"))
    return out
