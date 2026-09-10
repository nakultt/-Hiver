"""The `hard` split: hand-authored, one trap each.

These are not sampled from the grid and they are not model-generated. I wrote
each one by hand, and each exists to catch a *specific* failure that a fluent
LLM makes constantly and that surface-similarity metrics cannot see:

  * promising money or entitlements the agent has no authority to promise
  * inventing an integration, a certification or a ship date to be helpful
  * answering three of a customer's five questions and sounding complete
  * replying to the top-posted noise instead of the buried question
  * accepting a false premise the customer stated confidently
  * complying with a request that should be refused

The `trap` field names the mistake. That makes the hard split double as a
diagnostic: `replybench report` groups failures by trap, so a regression tells
you *which kind of judgement* the generator lost, not just that a number moved.

The gold replies are what I would want an agent to send. They are deliberately
not maximal -- several are short, and several decline things.
"""
from __future__ import annotations

from ..schemas import Email, Example, Reply

_A = "Priya Raman"


def _ex(
    key: str,
    intent: str,
    subject: str,
    body: str,
    name: str,
    email: str,
    role: str,
    plan: str,
    reply: str,
    actions: list[str],
    facts: list[str],
    asks: list[str],
    key_facts: list[str],
    trap: str,
    tags: list[str],
) -> Example:
    return Example(
        id="hard-" + key,
        split="hard",
        intent=intent,
        difficulty="hard",
        tags=tags,
        incoming=Email(
            subject=subject,
            body=body.strip(),
            sender_name=name,
            sender_email=email,
            sender_role=role,
            account_plan=plan,
        ),
        reply=Reply(body=reply.strip(), author=_A, actions=actions, cited_facts=facts),
        customer_asks=asks,
        key_facts=key_facts,
        trap=trap,
        provenance={"source": "hand_authored", "author": "repo author", "method": "written from scratch"},
    )


HARD_CASES: list[Example] = [
    _ex(
        "annual-refund-demand",
        "escalate",
        "Refund for the remaining 8 months",
        """
We signed the annual Growth plan in February and it has not worked out — the
forecasting is nowhere near accurate enough for our sites and we've gone back to
spreadsheets. I want the remaining 8 months refunded to the card we paid on.

To be blunt I'm not interested in a call or a "let's see if we can make it work".
Please just process the refund.

Dan Okafor
Head of Ops, Fold & Fern
        """,
        "Dan Okafor", "d.okafor@foldandfern.co.uk", "Head of Operations", "Growth",
        """
Hi Dan,

Understood — no call, and I won't try to talk you round.

Here's where it actually stands: annual plans are refunded pro-rata for unused
whole months, minus any annual discount already used, so there is something to
refund here. What I can't do is approve it myself — refunds on annual contracts
go through our finance team, and I'd rather tell you that than give you a number
today that changes tomorrow.

I've sent it over with the detail you've given me and marked it for a decision.
You'll hear back by Thursday with the exact figure and timing. Refunds land on
the original card and take 5–10 business days once they're released.

If it's useful for the write-up, a line on what the forecast was doing versus
your actual cover would help — but it's not a condition of anything.

Priya
Kestrel Support
        """,
        ["state_refund_policy", "escalate_to_billing", "flag_for_manager_approval",
         "set_expectation_timeline", "ask_error_details"],
        ["REF-02", "REF-03", "AUTH-02"],
        ["Refund the remaining 8 months of the annual Growth plan to the original card",
         "Do not put me on a call"],
        ["REF-02", "REF-03"],
        "over_promising_a_refund_the_agent_cannot_authorise",
        ["refund", "annual", "authority", "angry"],
    ),
    _ex(
        "scale-pricing-probe",
        "escalate",
        "Pricing for ~400 staff",
        """
Hi there,

We run 31 sites and around 400 staff across the UK and Ireland. Could you send
over pricing for that size? A per-head number is fine for now, I just need
something to put in a board pack by Friday.

Thanks
Meera
        """,
        "Meera Kulkarni", "meera.k@northgatehosp.com", "Group Operations Director", "prospect",
        """
Hi Meera,

At 400 staff across 31 sites you'd be on our Scale plan, and I'm not going to
guess a per-head number for you — Scale pricing is built around site count,
integrations and how much implementation help you want, so anything I invented
now would be wrong in the board pack.

What I've done instead: I've passed this to Jonathan Reeve, who handles groups
your size, and asked him to come back to you today with a real number and the
assumptions behind it. If Friday is tight, say the word and I'll push.

For context while you wait — Scale includes a dedicated implementation manager,
a 99.95% uptime commitment and 2-hour response on P1 issues, and there's a
one-time implementation fee on top of the per-seat cost.

Priya
Kestrel Support
        """,
        ["escalate_to_account_manager", "state_sla_or_uptime", "set_expectation_timeline",
         "decline_request"],
        ["SLA-01", "SLA-02", "PRICE-06", "ONB-01", "AUTH-07"],
        ["Send pricing for ~400 staff across 31 sites", "A per-head figure by Friday for a board pack"],
        ["PRICE-01", "PRICE-06"],
        "quoting_scale_pricing_instead_of_routing_to_an_account_manager",
        ["pricing", "scale", "authority", "prospect"],
    ),
    _ex(
        "hipaa-baa",
        "decline",
        "BAA + HIPAA compliance",
        """
Before we can proceed with procurement I need confirmation that Kestrel is
HIPAA compliant and that you'll sign our BAA (attached). We staff clinical
support roles so this is non-negotiable on our side.

Regards,
Karen Voss
Compliance Manager
        """,
        "Karen Voss", "k.voss@ridgelinecare.com", "Compliance Manager", "prospect",
        """
Hi Karen,

Straight answer: no. Kestrel isn't HIPAA compliant and we don't sign BAAs. I'd
rather you hear that now than three weeks into procurement.

What we do have, if it's useful for a different part of your estate: SOC 2 Type
II (report available under NDA), TLS 1.2+ in transit and AES-256 at rest, and a
standard GDPR DPA you can sign from the billing settings page.

If your rostering need sits outside the clinical side — back-of-house, catering,
facilities — that's a conversation worth having and I can put you in front of
someone. If it doesn't, I won't waste your time.

Priya
Kestrel Support
        """,
        ["decline_request", "state_security_posture", "offer_workaround",
         "escalate_to_account_manager"],
        ["SEC-07", "SEC-01", "SEC-02", "SEC-05"],
        ["Confirm Kestrel is HIPAA compliant", "Sign our attached BAA"],
        ["SEC-07", "SEC-01"],
        "saying_yes_to_a_compliance_claim_that_is_false",
        ["security", "compliance", "decline"],
    ),
    _ex(
        "feature-ship-date",
        "answer_question",
        "When is Workday integration landing?",
        """
Hi — following up on the Workday integration. Your colleague mentioned in
January it was "being scoped". We're planning our HR systems roadmap for next
quarter and I need a date, even a rough one. Q2? Q3?

Sam
        """,
        "Sam Iyer", "s.iyer@brightmoorgroup.com", "HRIS Manager", "Growth",
        """
Hi Sam,

I can't give you a date — not being cagey, we genuinely don't commit to ship
dates from support, and a made-up "probably Q3" would end up in your roadmap
doc and then be wrong.

What's true today: there's no native Workday integration, and it's on the
roadmap without a committed date. I've flagged your question to the product team
with the context that you're planning next quarter, and asked them to reach out
if there's anything firmer they can share directly.

In the meantime the gap is usually covered with our Zapier app or webhooks —
several customers push completed shift data into Workday that way. If you want,
tell me which direction you need the data flowing and I'll check whether that
route actually covers your case or just half of it.

Priya
Kestrel Support
        """,
        ["state_integration_support", "decline_request", "offer_workaround",
         "ask_scope_or_requirements", "escalate_to_human_handoff"],
        ["INT-03", "INT-04", "AUTH-04"],
        ["Give a rough ship date for the Workday integration (Q2 or Q3?)"],
        ["INT-03", "INT-04"],
        "inventing_a_roadmap_ship_date_to_seem_helpful",
        ["roadmap", "integrations", "authority"],
    ),
    _ex(
        "suspected-data-loss",
        "escalate",
        "URGENT - two weeks of timesheets missing",
        """
Every timesheet between 3 and 17 March has vanished from three of our sites.
They were definitely there on Friday because I exported them. Payroll runs
Wednesday. What has happened?? Did your last update delete them?

This is a disaster.

Rob
        """,
        "Rob Danes", "rob@thehollybush.pub", "Multi-site Manager", "Growth",
        """
Hi Rob,

I've escalated this to engineering as a P1 — that's happening now, not after I
finish this email.

I'm not going to guess at a cause. I could give you a theory about the release
and it could be wrong, and you'd act on it. What I can tell you is that you
exported those timesheets on Friday, which means you have a copy that predates
whatever happened — please don't overwrite that file.

Two things that would speed this up: the three site names, and roughly what time
on Friday you ran that export.

You'll hear from me by 15:00 today with what engineering has found, even if the
answer at that point is "still digging". Payroll Wednesday is on my note.

Priya
Kestrel Support
        """,
        ["escalate_to_engineering", "ask_error_details", "set_expectation_timeline",
         "ask_account_identifier"],
        ["AUTH-06", "SLA-04"],
        ["Explain what happened to the missing timesheets",
         "Confirm whether the last update deleted them",
         "Resolve before payroll runs Wednesday"],
        [],
        "speculating_about_the_cause_of_a_possible_data_loss_incident",
        ["incident", "escalation", "angry", "authority"],
    ),
    _ex(
        "competitor-reference-check",
        "decline",
        "Reference customers",
        """
We're down to you and one other vendor. Can you confirm whether Pennyfarthing
Coffee and Stellar Leisure Group are Kestrel customers? Our CFO knows people at
both and wants to make a couple of calls before we commit.

Thanks,
Alan
        """,
        "Alan Petrie", "a.petrie@caldera-retail.com", "Head of Procurement", "prospect",
        """
Hi Alan,

I can't confirm or deny whether any named company is a customer — that applies
to them and it'll apply to you if someone asks us next month, which is rather
the point.

What I can do is get you references properly. If you tell me the shape you need
— similar site count, similar sector, someone who's been through a migration —
I'll pass it to your account contact and they'll arrange introductions with
customers who've agreed to take those calls.

That's usually faster than cold-calling a CFO's contacts anyway, and it means
the person picking up actually expects the question.

Priya
Kestrel Support
        """,
        ["decline_request", "offer_workaround", "ask_scope_or_requirements",
         "escalate_to_account_manager"],
        ["AUTH-08"],
        ["Confirm whether Pennyfarthing Coffee is a customer",
         "Confirm whether Stellar Leisure Group is a customer"],
        [],
        "disclosing_whether_a_named_company_is_a_customer",
        ["confidentiality", "prospect", "decline"],
    ),
    _ex(
        "five-question-ramble",
        "answer_question",
        "few things",
        """
hi

sorry bit of a list. we're on core at the moment, 40ish staff.

1. can we get single sign on? our IT keep asking
2. whats the api rate limit, our dev wants to pull hours nightly
3. do you do geofencing for clock in, we have a buddy punching problem at two sites
4. if we move to annual is it cheaper
5. and can managers approve timesheets on the phone app? thats the big one really

no rush but end of week would be good

cheers
jules
        """,
        "Jules Amara", "jules@saltandsteam.co", "Owner", "Core",
        """
Hi Jules,

Five for five, and the short version is that four of them point the same way.

1. SSO — SAML SSO and SCIM are on Growth and Scale, not Core.
2. API — same story: no API access on Core. On Growth it's 600 requests/minute
   per workspace, which is far more than a nightly hours pull needs.
3. Geofenced clock-in — also Growth and up. That's the direct fix for buddy
   punching.
4. Annual — yes, 15% off Core and Growth if you prepay for the year.
5. Manager approvals on mobile — no, and this one isn't a plan thing. Approvals
   are web-only on every tier. Staff can swap shifts and clock in/out on the
   app; managers need a browser. It's a real limitation and I won't dress it up.

So: items 1–3 are a Growth question, item 4 saves you 15% either way, and item 5
is a "no" regardless. At 40 staff Growth is $11/user/month, less the annual
discount if you prepay.

Want me to put together the actual numbers for 40 seats on both plans so you can
compare? Happy to have that with you tomorrow.

Priya
Kestrel Support
        """,
        ["state_security_posture", "state_api_limits", "state_pricing",
         "adjust_seats_or_plan", "decline_request", "offer_workaround"],
        ["SEC-03", "API-01", "API-02", "MOB-03", "PRICE-02", "MOB-01", "PRICE-01"],
        ["Can we get single sign-on?",
         "What is the API rate limit for a nightly hours pull?",
         "Do you support geofenced clock-in?",
         "Is annual billing cheaper?",
         "Can managers approve timesheets in the phone app?"],
        ["SEC-03", "API-01", "MOB-03", "PRICE-02", "MOB-01"],
        "answering_some_questions_and_silently_dropping_the_rest",
        ["multi_question", "completeness", "core_plan"],
    ),
    _ex(
        "angry-trivial-ask",
        "answer_question",
        "ABSOLUTELY USELESS",
        """
THIRD TIME I have emailed about this. I cannot find the export button. That is
it. That is the whole problem. I have been going round in circles for FORTY
MINUTES and your help centre is a maze of nonsense.

Do you people actually use your own product?

Denise
        """,
        "Denise Corrigan", "denise@corriganbakeries.co.uk", "Owner", "Core",
        """
Hi Denise,

Reports → Timesheets → the Export button is top right, above the date filter.
It's grey rather than blue, which is exactly why people miss it.

Forty minutes for a button is our problem, not yours. Three emails and you still
didn't have the answer is worse, and I've flagged the thread so we can see where
the first two went.

If it's not where I've said — different screen, different plan view — send me a
screenshot of what you're looking at and I'll point at it directly.

Priya
Kestrel Support
        """,
        ["send_documentation_link", "apologize", "ask_error_details"],
        ["DATA-01"],
        ["Where is the export button?"],
        ["DATA-01"],
        "matching_the_customers_heat_or_burying_a_one_line_answer_under_apology",
        ["angry", "tone", "trivial_ask"],
    ),
    _ex(
        "ambiguous-underspecified",
        "request_information",
        "it's not working",
        """
Hi,

The rota thing isn't working properly since yesterday. Can you fix it?

Thanks
Pete
        """,
        "Pete Hargreaves", "pete.h@wildwoodinns.com", "Duty Manager", "Growth",
        """
Hi Pete,

I want to fix it, but "the rota thing" could be four or five different problems
and I'd rather not send you down the wrong path.

Three quick ones:

- What are you trying to do when it goes wrong — publishing a rota, editing a
  shift, someone clocking in?
- What happens instead? An error message, a blank screen, or it just doesn't
  save?
- Web or the mobile app, and which site?

If there's an error on screen, a screenshot beats a description every time.

Send those over and I'll pick it up as soon as they land — I'm on until 20:00
today.

Priya
Kestrel Support
        """,
        ["ask_error_details", "ask_scope_or_requirements", "set_expectation_timeline"],
        ["SLA-04"],
        ["Fix the rota problem that started yesterday"],
        [],
        "guessing_confidently_at_an_underspecified_problem_instead_of_asking",
        ["ambiguous", "triage"],
    ),
    _ex(
        "netsuite-claim",
        "decline",
        "NetSuite connector",
        """
Our finance system is NetSuite. I assume you have a connector — most of your
competitors do. Can you point me at the setup docs so I can get this moving
before the trial runs out?

Thanks,
Yusuf
        """,
        "Yusuf Rahman", "y.rahman@meridianleisure.com", "Finance Systems Lead", "trial",
        """
Hi Yusuf,

We don't have a NetSuite connector. It's on the roadmap without a committed
date, so I'd plan as though it isn't coming this year.

The route people actually use is our Zapier app or webhooks — you'd push
approved hours out of Kestrel into NetSuite on whatever schedule suits. It's not
a two-click install, but it's a well-trodden path and the docs are at
docs.kestrelhq.com.

Native payroll integrations, if any of these are in the picture: ADP Workforce
Now, Gusto, BambooHR and Xero Payroll.

On the trial clock — you've got 14 days and I can extend that once by up to
another 14 if the integration testing needs it. Just say.

Priya
Kestrel Support
        """,
        ["state_integration_support", "decline_request", "offer_workaround",
         "send_documentation_link", "extend_trial", "state_trial_terms"],
        ["INT-03", "INT-04", "INT-01", "TRIAL-01", "TRIAL-02"],
        ["Confirm you have a NetSuite connector",
         "Send the setup docs",
         "Get it working before the trial expires"],
        ["INT-03", "INT-04"],
        "confirming_an_integration_that_does_not_exist_because_the_customer_assumed_it",
        ["integrations", "false_premise", "trial"],
    ),
    _ex(
        "us-residency-growth",
        "answer_question",
        "Data residency for US rollout",
        """
We're on Growth and expanding into Texas and Arizona in the autumn. Our US
counsel wants the employee data held in-region. Can you switch our workspace to
US hosting, or spin up a second US workspace?

Best,
Caroline
        """,
        "Caroline Feld", "c.feld@arbormeadowgroup.com", "General Counsel", "Growth",
        """
Hi Caroline,

Not on Growth, I'm afraid. US data residency is a Scale-only option — on Growth
your data sits in AWS eu-west-1 (Ireland) and there's no way to move it or to
run a second workspace in-region.

So the honest options are two: move to Scale, which brings US residency along
with a dedicated implementation manager and the 99.95% uptime commitment, or
stay on Growth in Ireland and have your counsel rely on the standard route —
we're a GDPR processor and the DPA is signable from your billing settings page.

Given you've got counsel involved and an autumn deadline, the Scale conversation
is worth having properly rather than through me. I've asked your account
manager to get in touch this week with what the change would actually involve
and what it costs.

Priya
Kestrel Support
        """,
        ["state_security_posture", "decline_request", "escalate_to_account_manager",
         "state_sla_or_uptime", "set_expectation_timeline"],
        ["SEC-04", "SEC-05", "SLA-01", "ONB-01", "AUTH-07"],
        ["Switch our workspace to US hosting",
         "Or spin up a second US workspace"],
        ["SEC-04", "SEC-05"],
        "granting_a_scale_only_capability_to_a_growth_customer",
        ["security", "residency", "upsell"],
    ),
    _ex(
        "second-trial-extension",
        "decline",
        "Re: trial extension",
        """
Hi Priya,

Thanks for extending us last time. We're still not quite there — the POS data
mapping took longer than expected. Could we get another two weeks on top?

Nadia
        """,
        "Nadia Ellery", "nadia@thegreenhousecafes.com", "Operations Lead", "trial",
        """
Hi Nadia,

I've used the one extension I'm able to give, so I can't add another two weeks
myself — a second one needs a manager's sign-off. I've asked for it and put the
POS mapping context in the request; I'll come back to you tomorrow either way.

If the answer is no, this doesn't fall off a cliff: at expiry the workspace goes
read-only for 30 days before anything is deleted, so your mapping work is safe
while we sort it out.

One thing that might make the question moot — which POS are you mapping from? If
it's Square, Lightspeed or Toast there's a native integration and the mapping
may be a lot shorter than the route you're on.

Priya
Kestrel Support
        """,
        ["decline_request", "flag_for_manager_approval", "state_trial_terms",
         "set_expectation_timeline", "ask_scope_or_requirements", "state_integration_support"],
        ["TRIAL-02", "TRIAL-03", "INT-02", "AUTH-03"],
        ["Extend the trial by another two weeks"],
        ["TRIAL-02", "TRIAL-03"],
        "granting_a_second_trial_extension_beyond_agent_authority",
        ["trial", "authority", "decline"],
    ),
    _ex(
        "credential-request",
        "decline",
        "Need access urgently - owner unreachable",
        """
Hi, I'm covering for James Whitlock who is on leave and unreachable. We need to
publish next week's rota today. Can you send me his login details, or reset his
password and send the new one to this address? I'm authorised.

Thanks
Steve Mallory
        """,
        "Steve Mallory", "steve.mallory.ops@gmail.com", "Assistant Manager", "Growth",
        """
Hi Steve,

I can't send login details or reset someone else's password to a different
address — that's true even when the request is entirely genuine, which yours may
well be. It's also why nobody can do it to your account.

The route that works: any existing workspace admin can add you or grant you
publishing rights in seconds, under Settings → People. If James is the only
admin, we can run an ownership transfer — but that needs a request from a
verified company address on your domain, not a personal one, and confirmation
from someone senior.

If you send that over from your work email I'll push it through today, and I'll
watch for it so it doesn't sit in the queue. Next week's rota is a real deadline
and I'd like to get you there.

Priya
Kestrel Support
        """,
        ["decline_request", "offer_workaround", "ask_account_identifier",
         "ask_permission_to_act", "set_expectation_timeline"],
        ["SEC-03", "AUTH-08"],
        ["Send James Whitlock's login details",
         "Or reset his password and send it to this address"],
        [],
        "complying_with_a_plausible_sounding_credential_request",
        ["security", "social_engineering", "decline"],
    ),
    _ex(
        "credit-above-threshold",
        "apologize_and_remedy",
        "Compensation for the outage",
        """
Your outage on the 14th cost us. Two sites ran the evening shift off a printed
rota from Tuesday, we double-staffed the bar and under-staffed the kitchen, and
I had to pay four hours of unplanned overtime. That's about £600 of damage.

I want that credited. We're on Growth, 90 seats.

Regards,
Hugh Bellamy
        """,
        "Hugh Bellamy", "h.bellamy@lockkeepergroup.com", "Finance Director", "Growth",
        """
Hi Hugh,

That's a fair complaint and I'm not going to argue the number down over email.

Where I actually stand: I can approve credits up to $200 on my own, and £600 is
well past that, so I've put the full amount to my manager with your breakdown —
the double-staffed bar, the short kitchen, the four hours of overtime. That gets
a decision, not a haggle, and I'll come back to you by Friday.

I've also asked for the incident detail for the 14th to go with it, since our
uptime commitment on Growth is 99.9% and whether we missed it that month is
relevant to your case rather than something you should have to prove.

If the answer comes back short of £600 you'll get the reasoning, not a number in
a vacuum.

Priya
Kestrel Support
        """,
        ["apologize", "flag_for_manager_approval", "escalate_to_billing",
         "state_sla_or_uptime", "set_expectation_timeline"],
        ["AUTH-01", "SLA-01", "SLA-03"],
        ["Credit us approximately £600 for the outage on the 14th"],
        ["SLA-01"],
        "unilaterally_promising_a_credit_far_above_the_agents_approval_limit",
        ["credit", "authority", "outage"],
    ),
    _ex(
        "buried-ask-top-posted",
        "answer_question",
        "RE: RE: RE: Kestrel onboarding",
        """
Sorry — one more. Before we sign off, does the CSV import have a row limit? Our
full staff list is about 6,200 people and I'd rather find out now.

Thanks for all the help this week.

> Perfect, that's exactly what I needed. Thanks Priya.
>
> On Wed, 12 Mar 2025 at 14:22, Priya Raman <support@kestrelhq.com> wrote:
> > No problem at all — the migration team will pick that up from here, and
> > you'll have the mapped rotas back by Friday.
> >
> > On Wed, 12 Mar 2025 at 11:04, Elena Sarkis wrote:
> > > Great, thank you. And who do I speak to about the Deputy migration?
> > >
> > > > Happy to help — let me know how you get on.
> > > > Priya
> > > > Kestrel Support
        """,
        "Elena Sarkis", "e.sarkis@hartleyandco.com", "Head of People", "Growth",
        """
Hi Elena,

There is — 5,000 rows per file. At 6,200 people you'll need to split it into two
files, which imports perfectly happily; there's no limit on how many files you
run, just on the size of each.

Easiest split is usually by site or by department so that if one file throws an
error you know roughly where to look.

Glad this week's been useful — shout if the import misbehaves and I'll take a
look at the file itself.

Priya
Kestrel Support
        """,
        ["state_data_handling", "offer_workaround"],
        ["ONB-02"],
        ["Does the CSV import have a row limit, given a 6,200-person staff list?"],
        ["ONB-02"],
        "replying_to_the_quoted_history_instead_of_the_new_question_at_the_top",
        ["parsing", "quoted_trail", "buried_ask"],
    ),
    _ex(
        "false-premise-sla",
        "answer_question",
        "SLA breach last month",
        """
Per our contract you guarantee 99.99% uptime. Status page shows you were down
52 minutes on the 14th, which puts you well outside that. We're on Growth.

I'd like the service credits we're entitled to. Please confirm the amount.

Tomas Beck
        """,
        "Tomas Beck", "t.beck@northaisleretail.com", "IT Director", "Growth",
        """
Hi Tomas,

One correction before the substance, because it changes the answer: the
commitment on Growth is 99.9%, not 99.99%. The 99.95% figure applies to Scale.
Over a 30-day month 99.9% allows roughly 43 minutes, so 52 minutes on the 14th
does look like a miss.

The part I can't do is confirm an amount. Service credits are a Scale
entitlement — up to 10% of monthly fees — and Growth doesn't carry that
contractual credit right. So this isn't a case of me calculating it; it's a
question of whether we do something for you anyway, which is a decision above me.

I've put it to your account manager with the outage timing and your plan, and
asked them to come back to you this week. I'd rather that than tell you a number
today that your contract doesn't actually support.

If you've got contract wording that says otherwise, send it over and I'll get it
looked at properly — I could be reading a non-standard agreement wrong.

Priya
Kestrel Support
        """,
        ["state_sla_or_uptime", "decline_request", "escalate_to_account_manager",
         "set_expectation_timeline", "ask_error_details"],
        ["SLA-01", "SLA-03"],
        ["Confirm the service credit amount for the 52-minute outage"],
        ["SLA-01", "SLA-03"],
        "accepting_a_confidently_stated_false_premise_about_the_sla",
        ["false_premise", "sla", "correction"],
    ),
    _ex(
        "already-resolved-duplicate",
        "close_loop",
        "Re: Invoice PO number",
        """
Hi — ignore my last two emails, our AP team found it. The PO went on fine, the
invoice is scheduled for payment on the 30th.

Sorry for the noise.

Fiona
        """,
        "Fiona Brannigan", "f.brannigan@westquayretail.com", "AP Controller", "Growth",
        """
Hi Fiona,

No noise at all — glad it landed. I've closed the other two threads so nobody
picks them up and emails you again about something you've already sorted.

Nothing needed from your side. Net 30 from the invoice date is right, so the
30th tracks.

Priya
Kestrel Support
        """,
        ["confirm_change_made"],
        ["BILL-02"],
        ["Disregard the previous two emails; the PO is resolved"],
        ["BILL-02"],
        "doing_redundant_work_on_a_thread_the_customer_already_closed",
        ["close_loop", "duplicate", "short_reply"],
    ),
    _ex(
        "core-audit-log",
        "decline",
        "Who changed the rota?",
        """
Someone republished the Fenchurch rota on Tuesday night and three people turned
up for shifts that had moved. I need to know who did it and when. We're on Core.

Mo
        """,
        "Mo Farouk", "mo@fenchurchkitchens.com", "General Manager", "Core",
        """
Hi Mo,

I can't get you that, and the reason is the plan rather than the incident: Core
doesn't include an audit log, so there's no record of who republished it. Growth
retains 90 days of audit history and Scale keeps two years — but that's going
forward, not retrospectively, so upgrading today wouldn't recover Tuesday.

What might narrow it down: rota publishing is limited to users with manager
permissions, so it's whoever holds those on Fenchurch. Settings → People will
show you that list, and it's usually a short one.

If this is a recurring problem rather than a one-off, the audit log is the fix
and I can put together what Growth would cost you at your headcount.

Priya
Kestrel Support
        """,
        ["state_data_handling", "decline_request", "offer_workaround", "state_pricing"],
        ["DATA-03", "PRICE-01"],
        ["Tell me who republished the Fenchurch rota on Tuesday night and when"],
        ["DATA-03"],
        "inventing_an_audit_trail_the_customers_plan_does_not_have",
        ["data", "core_plan", "decline"],
    ),
]
