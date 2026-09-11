"""Escalation scorer: weighted signals -> score in [0, 1] -> decision with a stated reason.

Two versions are kept so the fix can be measured on a fresh sample:
  v1 (headline): the incident signal only fires inside billing/account intents.
  v2: the incident signal is intent-independent (a charged-twice family-plan tweet counts).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

import pandas as pd

from . import config as C

# Weights sum to 1.35 and the score is capped at 1.0. Money/security intent alone (a pricing
# question) stays under the 0.45 threshold; with any second signal it crosses.
WEIGHTS = {
    "sensitive_intent": 0.35,
    "money_or_security_incident": 0.25,
    "low_confidence": 0.20,
    "weak_retrieval": 0.20,
    "hostile_or_urgent": 0.20,
    "needs_private_data": 0.10,
    "repeat_contact": 0.05,
}
REASONS = {
    "sensitive_intent": "money or account security is involved",
    "money_or_security_incident": "money already changed hands or the account was compromised",
    "low_confidence": "intent classifier is unsure",
    "weak_retrieval": "no historical resolution matches",
    "hostile_or_urgent": "customer is angry or urgent",
    "needs_private_data": "resolution needs account details",
    "repeat_contact": "customer reports a repeat contact",
    "unhandleable": "no actionable content in the message",
}

# Each word lives in exactly one regex so one token cannot contribute two weights.
HOSTILE = re.compile(
    r"\b(lawyer|legal|sue|unacceptable|ridiculous|disgust(ing|ed)?|furious|asap|urgent|immediately|worst|"
    r"garbage|useless|rip ?off|report you|cancel(l)?ing my|terrible|joke)\b",
    re.I,
)
INCIDENT = re.compile(
    r"\b(charged|charges?|billed|debited|took (my )?money|taking money|paid|payment (was )?taken|refund(ed)?|"
    r"hacked|hijack(ed)?|stolen|stole|compromised|someone (else )?(is )?(using|logged|took|got into|bought|sold)|"
    r"not my (account|email)|(someone|somebody) else'?s? (device|account|music))\b",
    re.I,
)
PRIVATE = re.compile(
    r"\b(email address|e-mail|username|user name|order number|receipt|invoice|(credit|debit) card|card (number|details)|"
    r"bank|account number|transaction)\b",
    re.I,
)
REPEAT = re.compile(
    r"\b(again|still (no|not|waiting|haven't|hasn't|can't|cant|won't|wont)|for (\d+|two|three|four|several|many) (days|weeks|months)|"
    r"(second|third|2nd|3rd|nth) time|already (emailed|dm|dmed|dm'd|contacted|tweeted|messaged)|"
    r"no (one|response|reply)|nobody|ignored|keep (getting|saying)|(\d+|48|72) hours)\b",
    re.I,
)


@dataclass
class EscalationResult:
    score: float
    escalate: bool
    reason: str
    signals: dict = field(default_factory=dict)


def _caps_ratio(text: str) -> float:
    letters = [c for c in text if c.isalpha()]
    return sum(c.isupper() for c in letters) / len(letters) if letters else 0.0


def signals(message: str, intent: str, confidence: float, examples: pd.DataFrame | None, sim_threshold: float, dm_share: float = 0.0, version: str = "v1") -> dict[str, bool]:
    top_sim = 0.0 if examples is None or len(examples) == 0 else float(examples.similarity.max())
    hostile = bool(HOSTILE.search(message)) or _caps_ratio(message) > 0.3 or message.count("!") >= 3
    incident = bool(INCIDENT.search(message))
    return {
        "sensitive_intent": intent in C.SENSITIVE_INTENTS,
        "money_or_security_incident": incident if version == "v2" else (intent in C.SENSITIVE_INTENTS and incident),
        "low_confidence": confidence < 0.5,
        "weak_retrieval": top_sim < sim_threshold,
        "hostile_or_urgent": hostile,
        "needs_private_data": bool(PRIVATE.search(message)) or dm_share >= 0.6,
        "repeat_contact": bool(REPEAT.search(message)),
    }


def score_from_signals(sig: dict[str, bool], threshold: float, drop: str | None = None) -> EscalationResult:
    """Deterministic re-scoring from stored signals; `drop` removes one signal for ablation."""
    active = {k: v for k, v in sig.items() if k != drop}
    s = min(1.0, sum(WEIGHTS[k] for k, v in active.items() if v))
    firing = sorted((k for k, v in active.items() if v), key=lambda k: -WEIGHTS[k])
    if s >= threshold:
        reason = "Escalate: " + " + ".join(REASONS[k] for k in firing[:2])
    else:
        reason = "Auto-handle: " + ("no escalation signals fired" if not firing else "only " + " + ".join(REASONS[k] for k in firing[:2]) + f" (score {s:.2f} < {threshold:.2f})")
    return EscalationResult(round(s, 3), s >= threshold, reason, sig)


def score(message: str, intent: str, confidence: float, examples: pd.DataFrame | None, threshold: float, sim_threshold: float = 0.55, dm_share: float = 0.0, version: str = "v1") -> EscalationResult:
    if intent == "unhandleable":
        return EscalationResult(1.0, True, "Escalate: " + REASONS["unhandleable"], {"unhandleable": True})
    return score_from_signals(signals(message, intent, confidence, examples, sim_threshold, dm_share, version), threshold)
