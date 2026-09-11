"""End-to-end agent: classify -> retrieve -> draft -> escalate. One call, one dict."""
from __future__ import annotations

import json
import os

from . import config as C
from .classifier import IntentClassifier
from .escalation import score as escalation_score
from .reply import draft
from .retrieval import Retriever
from .text import clean_customer

PARAMS_PATH = C.MODELS / "params.json"
DEFAULT_PARAMS = {"escalation_threshold": 0.5, "sim_threshold": 0.55, "k": 5}
VERSION = os.environ.get("SYSTEM_VERSION", "v1")  # v1 = headline; v2 = post-hoc escalation/unhandleable fixes


def load_params() -> dict:
    if PARAMS_PATH.exists():
        return {**DEFAULT_PARAMS, **json.loads(PARAMS_PATH.read_text())}
    return dict(DEFAULT_PARAMS)


class SupportAgent:
    def __init__(self, classifier: IntentClassifier | None = None, retriever: Retriever | None = None, params: dict | None = None, version: str = VERSION):
        self.classifier = classifier or IntentClassifier.load()
        self.retriever = retriever or Retriever.load()
        self.params = params or load_params()
        self.version = version

    def handle(self, raw_message: str, threshold: float | None = None, draft_reply: bool = True) -> dict:
        thr = self.params["escalation_threshold"] if threshold is None else threshold
        text = clean_customer(raw_message)
        intent, conf = self.classifier.predict([text], version=self.version)[0]
        examples = None
        dm_share = 0.0
        questions: list[str] = []
        reply = {"reply": None, "mode": "none", "used_examples": [], "cache_miss": False}
        if intent != "unhandleable":
            examples = self.retriever.search(text, k=self.params["k"], intent=intent)
            questions = self.retriever.questions_for(intent)
            dm_share = self.retriever.dm_share(text)
            if draft_reply:
                reply = draft(text, intent, examples, questions)
        esc = escalation_score(text, intent, conf, examples, thr, self.params["sim_threshold"], dm_share, version=self.version)
        return {
            "_examples": examples,
            "message": text,
            "intent": intent,
            "confidence": round(conf, 3),
            "reply": reply.get("reply"),
            "reply_mode": reply.get("mode"),
            "reply_cache_miss": reply.get("cache_miss", False),
            "groundedness": reply.get("groundedness"),
            "checks": reply.get("checks"),
            "usage": reply.get("usage"),
            "seconds": reply.get("seconds"),
            "evidence": [] if examples is None else examples.brand_text.tolist(),
            "escalate": esc.escalate,
            "escalation_score": esc.score,
            "reason": esc.reason,
            "signals": esc.signals,
            "retrieved": None if examples is None else examples[["brand_text", "similarity", "category"]].to_dict("records"),
            "questions": questions,
        }

    def draft_for(self, result: dict) -> dict:
        """Fill the reply fields of a result produced with draft_reply=False (used for parallel drafting)."""
        ex = result.get("_examples")
        if result["intent"] == "unhandleable" or ex is None:
            return result
        reply = draft(result["message"], result["intent"], ex, result["questions"])
        result.update({"reply": reply.get("reply"), "reply_mode": reply.get("mode"), "reply_cache_miss": reply.get("cache_miss", False),
                       "groundedness": reply.get("groundedness"), "checks": reply.get("checks"), "usage": reply.get("usage"), "seconds": reply.get("seconds")})
        return result
