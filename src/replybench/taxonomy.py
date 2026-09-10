"""Shared label vocabularies.

Both the dataset and the evaluator draw from these lists. That is deliberate:
"did the suggested reply do the same thing the human did?" is only a meaningful
question if both sides are described in the same closed vocabulary. Free-text
comparison of two replies collapses into vibes; set comparison over a fixed
action space is measurable.
"""
from __future__ import annotations

# --------------------------------------------------------------------------
# Intent: the single dominant purpose of the reply. One per reply.
# --------------------------------------------------------------------------
INTENTS: dict[str, str] = {
    "answer_question": "Directly answers what was asked using known policy or product facts.",
    "request_information": "Cannot proceed without something from the customer; asks for it.",
    "confirm_action": "Tells the customer something has been done or changed on our side.",
    "decline": "Says no to the request, or says it is out of scope, with a reason.",
    "apologize_and_remedy": "Owns a failure on our side and offers a concrete remedy.",
    "escalate": "Hands off to a named specialist team and sets expectations.",
    "schedule": "Proposes or confirms a call, demo, or meeting.",
    "quote_pricing": "Gives commercial terms: price, tier, discount, invoice mechanics.",
    "acknowledge_and_hold": "Buys time: acknowledges receipt, promises a real answer later.",
    "close_loop": "Wraps up a resolved thread; thanks, confirms, invites follow-up.",
}

# --------------------------------------------------------------------------
# Actions: the commitments and moves a reply makes. Many per reply.
#
# This is the backbone of the accuracy metric. A suggested reply is "accurate"
# primarily when it takes the same *actions* a competent human took -- not when
# it uses the same words.
# --------------------------------------------------------------------------
ACTIONS: dict[str, str] = {
    # --- asking for something ---
    "ask_account_identifier": "Requests account email, workspace name, or customer ID.",
    "ask_invoice_or_order_ref": "Requests an invoice number, order ref, or payment ID.",
    "ask_error_details": "Requests error text, screenshot, logs, device/browser, or timestamps.",
    "ask_scope_or_requirements": "Requests headcount, locations, use case, or requirements.",
    "ask_availability": "Requests times/timezone for a call.",
    "ask_permission_to_act": "Asks the customer to confirm before we change something.",
    # --- stating policy / facts ---
    "state_pricing": "States a price, tier cost, discount, or minimum.",
    "state_refund_policy": "States refund eligibility rules or windows.",
    "state_cancellation_terms": "States notice period or end-of-term behaviour.",
    "state_trial_terms": "States trial length, limits, or what happens at expiry.",
    "state_sla_or_uptime": "States response-time or uptime commitments.",
    "state_security_posture": "States certifications, encryption, SSO, or data residency.",
    "state_integration_support": "States whether an integration exists or is planned.",
    "state_data_handling": "States export, retention, or deletion behaviour.",
    "state_api_limits": "States API rate limits or auth mechanics.",
    # --- doing something ---
    "issue_refund": "Commits to refunding money.",
    "apply_account_credit": "Commits to a credit on the account.",
    "waive_fee": "Commits to waiving or discounting a charge.",
    "extend_trial": "Commits to extending a trial.",
    "adjust_seats_or_plan": "Commits to changing seat count, tier, or billing cycle.",
    "confirm_change_made": "Confirms a change is already applied.",
    "reset_or_reissue_access": "Resets a password, reissues an invite, unlocks an account.",
    # --- routing ---
    "escalate_to_engineering": "Routes a suspected bug or outage to engineering.",
    "escalate_to_billing": "Routes to the billing/finance team.",
    "escalate_to_account_manager": "Routes to a named AM or sales owner.",
    "escalate_to_security_review": "Routes a security/compliance questionnaire onward.",
    "flag_for_manager_approval": "Says the requested concession needs internal approval.",
    # --- shaping the conversation ---
    "offer_workaround": "Gives an interim way to get unblocked.",
    "send_documentation_link": "Points to docs, a help-centre article, or a guide.",
    "propose_meeting_time": "Offers specific slots or a booking link.",
    "set_expectation_timeline": "Gives a concrete when-you-will-hear-back.",
    "decline_request": "Explicitly refuses or rules something out.",
    "apologize": "Offers an apology for an experience or failure.",
    "escalate_to_human_handoff": "States a human teammate will take over.",
}

# Actions that move money or entitlements. Getting these wrong is expensive, so
# the evaluator treats a false-positive here far more harshly than a missing
# pleasantry: suggesting "we'll refund you" when the human did not is the single
# worst thing this system can do.
CONSEQUENTIAL_ACTIONS: frozenset[str] = frozenset({
    "issue_refund",
    "apply_account_credit",
    "waive_fee",
    "extend_trial",
    "adjust_seats_or_plan",
    "confirm_change_made",
    "reset_or_reissue_access",
    "decline_request",
})

DIFFICULTIES = ("routine", "moderate", "hard")

SPLITS = ("train", "test", "hard")

# The six scored dimensions. Weights are defended in the README and re-derived
# empirically in `replybench validate-metric` (weight-fitting section).
DIMENSIONS: dict[str, str] = {
    "action_match": "Did it commit to the same actions a competent human took?",
    "factuality": "Is every checkable claim supported by the knowledge base or the email?",
    "completeness": "Was every question/request in the incoming email addressed?",
    "policy_safety": "Does it stay inside what an agent is allowed to promise?",
    "tone_fit": "Right register for this customer, this situation, this company voice.",
    "clarity": "Readable, concise, well-structured, no filler.",
}

DEFAULT_WEIGHTS: dict[str, float] = {
    "action_match": 0.28,
    "factuality": 0.26,
    "completeness": 0.18,
    "policy_safety": 0.14,
    "tone_fit": 0.08,
    "clarity": 0.06,
}


def validate_actions(actions: list[str]) -> list[str]:
    """Drop anything outside the vocabulary; models occasionally invent labels."""
    return [a for a in dict.fromkeys(actions) if a in ACTIONS]


def action_menu() -> str:
    return "\n".join("  " + k + ": " + v for k, v in ACTIONS.items())


def intent_menu() -> str:
    return "\n".join("  " + k + ": " + v for k, v in INTENTS.items())
