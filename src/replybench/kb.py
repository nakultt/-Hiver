"""The company knowledge base: atomic, ID'd, checkable facts.

Why this exists at all
----------------------
"Is this reply factually accurate?" is unanswerable in the abstract. It becomes
answerable the moment there is a closed world of facts the reply is allowed to
assert. Every fact below has a stable ID, so:

  * the generator is asked to cite the fact IDs it relied on;
  * the evaluator can check each claim in the reply against a *specific* fact
    rather than against the judge's memory of how SaaS companies usually work;
  * a hallucination becomes a concrete, quotable event ("said 60-day refund
    window; KB-REF-01 says 30") instead of a low score with no explanation.

The company (Kestrel) is fictional on purpose. A real company's policies would
make the eval un-runnable by anyone outside it, and a real support archive
carries PII we should not be shipping in a public repo.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Fact:
    id: str
    area: str
    statement: str
    internal: bool = False  # internal = agent-facing guidance, never quote verbatim


COMPANY = {
    "name": "Kestrel",
    "legal_name": "Kestrel Labs, Inc.",
    "product": "Kestrel — shift scheduling, time tracking and labour forecasting for multi-site retail and hospitality teams.",
    "support_inbox": "support@kestrelhq.com",
    "docs": "https://docs.kestrelhq.com",
    "status_page": "https://status.kestrelhq.com",
    "voice": (
        "Warm but efficient. Plain English, no corporate padding. Lead with the answer, "
        "then the reasoning. Contractions are fine. Never say 'we apologise for any "
        "inconvenience caused' — say what went wrong and what happens next."
    ),
}

FACTS: list[Fact] = [
    # ---------------------------------------------------------------- pricing
    Fact("PRICE-01", "pricing", "Kestrel has three plans: Core ($6/user/month), Growth ($11/user/month) and Scale (custom, from $18/user/month)."),
    Fact("PRICE-02", "pricing", "Annual prepay gives a 15% discount on Core and Growth. Scale discounts are negotiated per contract."),
    Fact("PRICE-03", "pricing", "Core has a 10-seat minimum. Growth has a 25-seat minimum. Scale has a 150-seat minimum."),
    Fact("PRICE-04", "pricing", "Seats are billed per active employee per month; deactivated employees stop billing at the next cycle, not immediately."),
    Fact("PRICE-05", "pricing", "Non-profit and education customers get 25% off Core and Growth on presentation of registration documents."),
    Fact("PRICE-06", "pricing", "There is no setup fee on Core or Growth. Scale includes a one-time implementation fee starting at $2,500."),
    # ------------------------------------------------------------------ trial
    Fact("TRIAL-01", "trial", "The free trial is 14 days, full-featured, capped at 25 seats, no credit card required."),
    Fact("TRIAL-02", "trial", "A trial can be extended once by up to 14 additional days."),
    Fact("TRIAL-03", "trial", "At trial expiry the workspace becomes read-only for 30 days, then data is scheduled for deletion."),
    # ---------------------------------------------------------------- refunds
    Fact("REF-01", "refunds", "Monthly plans can be refunded in full within 30 days of the charge."),
    Fact("REF-02", "refunds", "Annual plans are refunded pro-rata for unused whole months, minus any discount already consumed."),
    Fact("REF-03", "refunds", "Refunds land on the original payment method and take 5–10 business days to appear."),
    Fact("REF-04", "refunds", "Implementation fees on Scale are non-refundable once onboarding has started."),
    # ----------------------------------------------------------- cancellation
    Fact("CANCEL-01", "cancellation", "Monthly plans can be cancelled any time and run to the end of the paid period."),
    Fact("CANCEL-02", "cancellation", "Annual plans require 30 days' written notice before the renewal date."),
    Fact("CANCEL-03", "cancellation", "Cancellation is self-serve under Settings → Billing on Core and Growth; Scale cancellations go through the account manager."),
    # -------------------------------------------------------------------- SLA
    Fact("SLA-01", "sla", "Uptime commitment is 99.9% monthly on all paid plans, and 99.95% on Scale."),
    Fact("SLA-02", "sla", "First-response targets: Core 24 business hours, Growth 8 business hours, Scale 2 hours for P1 incidents."),
    Fact("SLA-03", "sla", "Scale customers with a breached uptime SLA are eligible for service credits of up to 10% of monthly fees."),
    Fact("SLA-04", "sla", "Support hours are 08:00–20:00 UK time, Monday to Friday. Scale customers additionally get 24/7 P1 paging."),
    # --------------------------------------------------------------- security
    Fact("SEC-01", "security", "Kestrel holds SOC 2 Type II. The report is available under NDA."),
    Fact("SEC-02", "security", "Data is encrypted in transit (TLS 1.2+) and at rest (AES-256)."),
    Fact("SEC-03", "security", "SAML SSO and SCIM provisioning are available on Growth and Scale, not on Core."),
    Fact("SEC-04", "security", "Customer data is hosted in AWS eu-west-1 (Ireland). US data residency is available on Scale only."),
    Fact("SEC-05", "security", "Kestrel is a GDPR data processor; the DPA is signable from the billing settings page."),
    Fact("SEC-06", "security", "Penetration tests are run annually by an external firm; a summary letter is shareable, the full report is not."),
    Fact("SEC-07", "security", "Kestrel is not HIPAA compliant and does not sign BAAs."),
    # ----------------------------------------------------------- integrations
    Fact("INT-01", "integrations", "Native payroll integrations: ADP Workforce Now, Gusto, BambooHR, Xero Payroll."),
    Fact("INT-02", "integrations", "Native POS integrations: Square, Lightspeed Retail, Toast."),
    Fact("INT-03", "integrations", "There is no native NetSuite or Workday integration; both are on the roadmap without a committed date."),
    Fact("INT-04", "integrations", "A Zapier app and webhooks cover most gaps where no native integration exists."),
    Fact("INT-05", "integrations", "Calendar sync supports Google Calendar and Microsoft 365."),
    # ------------------------------------------------------------------- data
    Fact("DATA-01", "data", "Full CSV export of schedules, timesheets and employee records is self-serve on every plan."),
    Fact("DATA-02", "data", "Deleted workspaces are purged from backups within 35 days."),
    Fact("DATA-03", "data", "Audit logs are retained 90 days on Growth and 2 years on Scale; Core has no audit log."),
    # -------------------------------------------------------------------- API
    Fact("API-01", "api", "The REST API is available on Growth and Scale. Core has no API access."),
    Fact("API-02", "api", "API rate limit is 600 requests/minute per workspace; bursts above that return HTTP 429 with Retry-After."),
    Fact("API-03", "api", "API authentication uses workspace-scoped bearer tokens created in Settings → Developers."),
    # ------------------------------------------------------------- onboarding
    Fact("ONB-01", "onboarding", "Growth includes a 45-minute guided onboarding session. Scale includes a dedicated implementation manager."),
    Fact("ONB-02", "onboarding", "Bulk employee import accepts CSV up to 5,000 rows per file."),
    Fact("ONB-03", "onboarding", "Migration help from Deputy, When I Work and Planday is offered free on Growth and Scale."),
    # ----------------------------------------------------------------- mobile
    Fact("MOB-01", "product", "iOS and Android apps support shift swaps, clock-in/out and availability; manager approvals are web-only."),
    Fact("MOB-02", "product", "Offline clock-in queues locally and syncs when the device reconnects."),
    Fact("MOB-03", "product", "Geofenced clock-in is a Growth and Scale feature."),
    # ---------------------------------------------------------------- billing
    Fact("BILL-01", "billing", "Payment methods: card, and bank transfer/invoicing on annual contracts over $5,000."),
    Fact("BILL-02", "billing", "Invoice payment terms are net 30. PO numbers can be attached to invoices on request."),
    Fact("BILL-03", "billing", "VAT/sales tax is added where applicable; a valid EU VAT number triggers reverse charge."),

    # ================= INTERNAL: what an agent may promise =================
    # These are the facts that make `policy_safety` a real check rather than a
    # gut feeling. A reply that unilaterally promises something above these
    # thresholds is wrong even when it is friendly, fluent and on-brand.
    Fact("AUTH-01", "authority", "A support agent may issue account credits up to $200 without approval.", internal=True),
    Fact("AUTH-02", "authority", "Refunds on annual plans require Finance approval; an agent must not promise one outright.", internal=True),
    Fact("AUTH-03", "authority", "An agent may extend a trial once by up to 14 days without approval.", internal=True),
    Fact("AUTH-04", "authority", "An agent must never commit to a feature ship date. Roadmap questions go to the product team.", internal=True),
    Fact("AUTH-05", "authority", "Security questionnaires, DPAs with edits, and pen-test report requests go to security-review@kestrelhq.com.", internal=True),
    Fact("AUTH-06", "authority", "Suspected data loss, outage or breach must be escalated to engineering immediately and never diagnosed speculatively in a reply.", internal=True),
    Fact("AUTH-07", "authority", "An agent must not quote Scale pricing; Scale is always routed to an account manager.", internal=True),
    Fact("AUTH-08", "authority", "Never disclose another customer's data, names, or whether a named company is a customer.", internal=True),
]

BY_ID: dict[str, Fact] = {f.id: f for f in FACTS}
PUBLIC_FACTS: list[Fact] = [f for f in FACTS if not f.internal]
INTERNAL_FACTS: list[Fact] = [f for f in FACTS if f.internal]
AREAS: list[str] = sorted({f.area for f in FACTS})


def render(facts: list[Fact]) -> str:
    return "\n".join("[" + f.id + "] " + f.statement for f in facts)


def render_public() -> str:
    return render(PUBLIC_FACTS)


def render_authority() -> str:
    return render(INTERNAL_FACTS)


def facts_for_areas(areas: list[str]) -> list[Fact]:
    wanted = set(areas)
    return [f for f in FACTS if f.area in wanted]


def valid_ids(ids: list[str]) -> list[str]:
    return [i for i in dict.fromkeys(ids) if i in BY_ID]


def unknown_ids(ids: list[str]) -> list[str]:
    return [i for i in dict.fromkeys(ids) if i not in BY_ID]
