"""The scenario grid the synthetic corpus is sampled from.

The point of a grid
-------------------
If you just ask a model for "200 support emails" you get 200 emails about
password resets and billing, written in one voice, with a suspiciously uniform
length. That corpus will flatter any system you evaluate on it.

So the corpus is sampled from an explicit cross-product instead:

    situation  x  persona  x  register  x  plan  x  noise profile

Coverage is then a property we control and can report, rather than a property we
hope the generator stumbled into. `replybench dataset-report` prints the
realised balance of every axis so the claim is checkable.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Situation:
    key: str
    intent: str
    difficulty: str
    areas: list[str]
    key_facts: list[str]
    seed: str
    expected_actions: list[str] = field(default_factory=list)


SITUATIONS: list[Situation] = [
    Situation(
        "trial_expiry_extension", "confirm_action", "routine", ["trial"], ["TRIAL-01", "TRIAL-02"],
        "Trial ends in two days; they have not finished importing staff and ask for more time.",
        ["extend_trial", "state_trial_terms", "set_expectation_timeline"],
    ),
    Situation(
        "trial_data_after_expiry", "answer_question", "moderate", ["trial", "data"], ["TRIAL-03", "DATA-01"],
        "Trial lapsed last week; they want to know whether their rota data is gone.",
        ["state_trial_terms", "state_data_handling", "offer_workaround"],
    ),
    Situation(
        "pricing_small_team", "quote_pricing", "routine", ["pricing"], ["PRICE-01", "PRICE-03"],
        "A 12-person cafe group asks what Kestrel would cost them.",
        ["state_pricing", "ask_scope_or_requirements"],
    ),
    Situation(
        "pricing_annual_discount", "quote_pricing", "routine", ["pricing"], ["PRICE-01", "PRICE-02"],
        "Existing monthly customer asks whether paying yearly is cheaper.",
        ["state_pricing", "adjust_seats_or_plan"],
    ),
    Situation(
        "nonprofit_discount", "answer_question", "routine", ["pricing"], ["PRICE-05"],
        "A charity asks whether there is a non-profit rate.",
        ["state_pricing", "ask_account_identifier"],
    ),
    Situation(
        "seat_billing_confusion", "answer_question", "moderate", ["pricing", "billing"], ["PRICE-04"],
        "They deactivated six seasonal staff but the invoice did not drop; they think it is a bug.",
        ["state_pricing", "escalate_to_billing"],
    ),
    Situation(
        "refund_monthly_recent", "confirm_action", "routine", ["refunds"], ["REF-01", "REF-03"],
        "Charged 9 days ago for a monthly plan they never rolled out; they want the money back.",
        ["issue_refund", "state_refund_policy", "set_expectation_timeline"],
    ),
    Situation(
        "refund_out_of_window", "decline", "moderate", ["refunds"], ["REF-01"],
        "Asking for a refund on a monthly charge from four months ago.",
        ["decline_request", "state_refund_policy", "offer_workaround"],
    ),
    Situation(
        "cancel_annual_notice", "answer_question", "moderate", ["cancellation"], ["CANCEL-02", "CANCEL-03"],
        "Annual customer wants to cancel; renewal is in three weeks.",
        ["state_cancellation_terms", "escalate_to_billing"],
    ),
    Situation(
        "cancel_selfserve", "answer_question", "routine", ["cancellation"], ["CANCEL-01", "CANCEL-03"],
        "Monthly customer asks how to cancel.",
        ["state_cancellation_terms", "send_documentation_link"],
    ),
    Situation(
        "sso_availability", "answer_question", "routine", ["security"], ["SEC-03"],
        "Core-plan IT lead asks how to turn on SAML SSO.",
        ["state_security_posture", "state_pricing", "adjust_seats_or_plan"],
    ),
    Situation(
        "security_questionnaire", "escalate", "moderate", ["security"], ["SEC-01", "SEC-06"],
        "Procurement sends a 60-question security questionnaire and asks for the pen-test report.",
        ["escalate_to_security_review", "state_security_posture", "set_expectation_timeline"],
    ),
    Situation(
        "data_residency_eu", "answer_question", "moderate", ["security"], ["SEC-04", "SEC-05"],
        "German customer asks where data is stored and whether there is a DPA.",
        ["state_security_posture", "state_data_handling", "send_documentation_link"],
    ),
    Situation(
        "gdpr_deletion", "answer_question", "moderate", ["security", "data"], ["SEC-05", "DATA-02"],
        "Ex-employee has asked the customer to erase their record; they ask how deletion works.",
        ["state_data_handling", "send_documentation_link"],
    ),
    Situation(
        "payroll_integration_adp", "answer_question", "routine", ["integrations"], ["INT-01"],
        "Asks whether Kestrel pushes hours into ADP.",
        ["state_integration_support", "send_documentation_link"],
    ),
    Situation(
        "integration_missing", "decline", "moderate", ["integrations"], ["INT-03", "INT-04"],
        "Asks for a Workday integration before they will sign.",
        ["state_integration_support", "offer_workaround", "escalate_to_account_manager"],
    ),
    Situation(
        "pos_toast_sync", "answer_question", "routine", ["integrations"], ["INT-02"],
        "Restaurant group asks if sales data from Toast can drive labour forecasting.",
        ["state_integration_support", "ask_scope_or_requirements"],
    ),
    Situation(
        "api_rate_limit", "answer_question", "moderate", ["api"], ["API-02", "API-03"],
        "Developer is getting 429s from a nightly sync job.",
        ["state_api_limits", "offer_workaround", "send_documentation_link"],
    ),
    Situation(
        "api_on_core", "decline", "moderate", ["api", "pricing"], ["API-01", "PRICE-01"],
        "Core customer wants an API token and cannot find the Developers page.",
        ["state_api_limits", "decline_request", "state_pricing"],
    ),
    Situation(
        "clock_in_offline", "answer_question", "routine", ["product"], ["MOB-02"],
        "Site has patchy wifi; staff worry clock-ins are being lost.",
        ["state_integration_support", "offer_workaround"],
    ),
    Situation(
        "geofence_request", "answer_question", "routine", ["product", "pricing"], ["MOB-03", "PRICE-01"],
        "Core customer wants geofenced clock-in to stop buddy-punching.",
        ["state_pricing", "adjust_seats_or_plan"],
    ),
    Situation(
        "manager_approval_mobile", "answer_question", "routine", ["product"], ["MOB-01"],
        "District manager wants to approve timesheets from their phone.",
        ["state_integration_support", "offer_workaround", "decline_request"],
    ),
    Situation(
        "bulk_import_failing", "request_information", "moderate", ["onboarding"], ["ONB-02"],
        "A staff CSV import keeps failing partway through with no clear error.",
        ["ask_error_details", "state_data_handling", "set_expectation_timeline"],
    ),
    Situation(
        "migration_from_deputy", "answer_question", "routine", ["onboarding"], ["ONB-03", "ONB-01"],
        "Switching from Deputy and worried about rebuilding two years of rotas.",
        ["state_integration_support", "propose_meeting_time"],
    ),
    Situation(
        "outage_reported", "escalate", "hard", ["sla"], ["SLA-01", "SLA-04"],
        "Managers cannot load the schedule view this morning; shift starts in an hour.",
        ["escalate_to_engineering", "apologize", "set_expectation_timeline", "offer_workaround"],
    ),
    Situation(
        "slow_first_response", "apologize_and_remedy", "moderate", ["sla"], ["SLA-02", "SLA-04"],
        "Growth customer waited three days for a reply and is annoyed.",
        ["apologize", "state_sla_or_uptime", "apply_account_credit", "escalate_to_account_manager"],
    ),
    Situation(
        "sla_credit_request", "escalate", "hard", ["sla"], ["SLA-01", "SLA-03"],
        "Scale customer claims last month missed the uptime SLA and wants credits.",
        ["escalate_to_account_manager", "state_sla_or_uptime", "set_expectation_timeline"],
    ),
    Situation(
        "invoice_po_number", "confirm_action", "routine", ["billing"], ["BILL-02"],
        "Finance needs a PO number added to the last invoice before they can pay.",
        ["confirm_change_made", "escalate_to_billing"],
    ),
    Situation(
        "vat_reverse_charge", "answer_question", "moderate", ["billing"], ["BILL-03"],
        "Irish customer was charged VAT and believes reverse charge should apply.",
        ["state_pricing", "escalate_to_billing", "ask_account_identifier"],
    ),
    Situation(
        "bank_transfer_request", "answer_question", "routine", ["billing"], ["BILL-01", "BILL-02"],
        "Wants to pay by bank transfer rather than card.",
        ["state_pricing", "escalate_to_billing"],
    ),
    Situation(
        "audit_log_request", "answer_question", "moderate", ["data", "pricing"], ["DATA-03"],
        "Wants to see who changed a published rota last Tuesday.",
        ["state_data_handling", "state_pricing"],
    ),
    Situation(
        "export_before_leaving", "answer_question", "routine", ["data"], ["DATA-01", "DATA-02"],
        "Leaving for a competitor; wants everything exported and then deleted.",
        ["state_data_handling", "send_documentation_link", "escalate_to_account_manager"],
    ),
    Situation(
        "demo_request", "schedule", "routine", ["pricing"], ["PRICE-01", "ONB-01"],
        "Ops director for a 40-site chain wants to see the product.",
        ["propose_meeting_time", "ask_scope_or_requirements", "escalate_to_account_manager"],
    ),
    Situation(
        "locked_out_owner", "request_information", "moderate", ["security"], ["SEC-03"],
        "Workspace owner left the company and nobody can get admin access.",
        ["ask_account_identifier", "ask_permission_to_act", "escalate_to_human_handoff"],
    ),
    Situation(
        "double_charge", "apologize_and_remedy", "moderate", ["billing", "refunds"], ["REF-01", "REF-03"],
        "Card was charged twice in the same week.",
        ["apologize", "issue_refund", "escalate_to_billing", "set_expectation_timeline"],
    ),
    Situation(
        "shift_swap_confusion", "answer_question", "routine", ["product"], ["MOB-01"],
        "Staff swaps are showing on the app but not on the published rota.",
        ["ask_error_details", "offer_workaround"],
    ),
    Situation(
        "forecasting_accuracy", "answer_question", "hard", ["product", "integrations"], ["INT-02"],
        "Labour forecast is consistently 20% over actual footfall; they want to know why.",
        ["ask_error_details", "escalate_to_engineering", "offer_workaround"],
    ),
    Situation(
        "seat_minimum_pushback", "decline", "moderate", ["pricing"], ["PRICE-03", "PRICE-01"],
        "Eight-person business wants Core but is under the seat minimum.",
        ["state_pricing", "decline_request", "offer_workaround"],
    ),
    Situation(
        "renewal_price_increase", "escalate", "hard", ["pricing", "cancellation"], ["PRICE-02", "CANCEL-02"],
        "Renewal quote came in higher than last year and they are threatening to leave.",
        ["escalate_to_account_manager", "apologize", "set_expectation_timeline"],
    ),
    Situation(
        "thanks_after_fix", "close_loop", "routine", ["sla"], [],
        "Following up to say the fix worked and to thank the agent.",
        [],  # a pure close-loop reply legitimately commits to nothing; empty gold sets are real
    ),
]

# ---------------------------------------------------------------- other axes
PERSONAS: list[dict[str, str]] = [
    {"key": "ops_manager", "who": "Operations manager at a 6-site coffee chain", "style": "busy, practical, short sentences, writes from phone sometimes"},
    {"key": "hr_lead", "who": "HR lead at a 200-person hospitality group", "style": "careful, process-minded, uses full paragraphs and correct punctuation"},
    {"key": "it_admin", "who": "IT administrator", "style": "technical, terse, includes error strings and version numbers"},
    {"key": "finance", "who": "Finance/AP controller", "style": "formal, references invoice numbers and payment terms"},
    {"key": "franchise_owner", "who": "Owner-operator of two franchise stores", "style": "informal, lowercase, typos, no greeting"},
    {"key": "procurement", "who": "Procurement analyst at a large retailer", "style": "template-heavy, numbered questions, corporate boilerplate"},
    {"key": "store_manager", "who": "Individual store manager", "style": "plain, slightly apologetic, not very technical"},
    {"key": "founder", "who": "Founder of a fast-growing restaurant group", "style": "direct, impatient, expects speed"},
    {"key": "developer", "who": "Backend developer integrating our API", "style": "precise, code snippets, HTTP status codes"},
    {"key": "exec_assistant", "who": "Executive assistant coordinating on behalf of a director", "style": "polite, third-person, scheduling-focused"},
]

REGISTERS: list[dict[str, str]] = [
    {"key": "neutral", "desc": "Matter-of-fact. No strong emotion."},
    {"key": "warm", "desc": "Friendly, chatty, includes a bit of small talk."},
    {"key": "rushed", "desc": "Clipped and urgent. Fragments. Sent between meetings."},
    {"key": "frustrated", "desc": "Clearly annoyed but still civil. Some pointed remarks."},
    {"key": "angry", "desc": "Openly angry. Capitals, threats to leave, mentions of wasted time."},
    {"key": "confused", "desc": "Unsure what they even need. Rambling, self-correcting."},
    {"key": "formal", "desc": "Stiff and businesslike. 'Dear Sir/Madam' energy."},
]

PLANS = ["Core", "Core", "Growth", "Growth", "Growth", "Scale", "trial", "prospect"]

# How messy the raw email body is. Real shared inboxes are not clean.
NOISE_PROFILES = [
    "clean",
    "clean",
    "mobile_signature",
    "quoted_trail",
    "quoted_trail",
    "forwarded",
    "typos",
    "long_signature",
    "top_posted_thread",
]
