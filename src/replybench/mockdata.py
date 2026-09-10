"""Deterministic offline stand-in for the model.

This exists so that `pytest` and a first-time `--smoke` run work with no API key
and no network. It is plumbing-only: it returns schema-valid shapes with filler
content. Any score computed against it is meaningless, and the CLI says so.
"""
from __future__ import annotations

import hashlib
import json
from typing import Any


def _h(text: str, n: int) -> int:
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest(), 16) % n


def mock_for(prompt: str, *, json_mode: bool, variant: int = 0) -> str:
    p = prompt

    if "You extract what a customer actually asked for" in p:
        return json.dumps({"asks": ["Mock ask one", "Mock ask two"]})

    if "You label support replies against a fixed vocabulary" in p:
        return json.dumps({"intent": "answer_question",
                           "actions": ["state_pricing", "send_documentation_link"]})

    if "atomic, independently checkable" in p.lower() or "CLAIM EXTRACTION" in p:
        return json.dumps({"claims": ["Mock claim about pricing.", "Mock claim about support hours."]})

    if "CLAIM VERIFICATION" in p:
        return json.dumps({"results": [
            {"claim": "Mock claim about pricing.", "verdict": "supported",
             "source_id": "PRICE-01", "note": "mock"},
            {"claim": "Mock claim about support hours.", "verdict": "unsupported",
             "source_id": "", "note": "mock"},
        ]})

    if "ASK COVERAGE" in p:
        return json.dumps({"results": [
            {"ask": "Mock ask one", "status": "answered", "evidence": "mock"},
            {"ask": "Mock ask two", "status": "ignored", "evidence": "mock"},
        ]})

    if "POLICY BREACH" in p:
        return json.dumps({"breaches": []})

    if "RUBRIC JUDGE" in p:
        base = 0.6 + (_h(p + str(variant), 25) / 100.0)
        def dim(label: str) -> dict[str, Any]:
            return {"score": round(min(1.0, base), 3), "reason": "mock " + label,
                    "evidence": ["mock evidence"]}
        return json.dumps({
            "tone_fit": dim("tone"), "clarity": dim("clarity"),
            "policy_safety": dim("policy"), "holistic": dim("holistic"),
            "would_send": "light_edit", "biggest_problem": "mock",
        })

    if json_mode:
        return json.dumps({"mock": True, "variant": variant})

    if "You are an experienced support agent" in p or "SUGGESTED REPLY" in p:
        return (
            "Hi there,\n\nThanks for getting in touch. This is mock output produced with no "
            "API key, so it says nothing useful about your account.\n\nPriya\nKestrel Support"
        )

    if "You write the subject line" in p:
        return "Mock subject line"

    return "Mock response body for offline runs."
