"""Two baselines the system must beat.

trivial : majority intent, one canned template, never escalate.
simple  : TF-IDF + logistic regression, nearest-neighbour historical reply verbatim,
          keyword escalation.
"""
from __future__ import annotations

import re
from collections import Counter

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

from . import config as C
from .retrieval import Retriever
from .text import clean_customer, is_unhandleable

SIMPLE_PATH = C.MODELS / "baseline_tfidf_lr.joblib"
TEMPLATE = "Hey there! Sorry to hear that. Can you DM us your account's email address or username? We'll take a look [link:ldFdZRiNAt]"
KEYWORD_ESCALATE = re.compile(r"\b(refund|charged|hacked|fraud|scam|lawyer|cancel|unacceptable|password)\b", re.I)


class TrivialBaseline:
    name = "trivial"

    def __init__(self, majority_intent: str):
        self.majority_intent = majority_intent

    def handle(self, raw_message: str, **_) -> dict:
        text = clean_customer(raw_message)
        return {"message": text, "intent": self.majority_intent, "confidence": 1.0, "reply": TEMPLATE, "reply_mode": "template",
                "escalate": False, "escalation_score": 0.0, "reason": "Auto-handle: baseline never escalates", "reply_cache_miss": False,
                "evidence": [TEMPLATE]}  # the template is itself a canonical historical reply


class SimpleBaseline:
    name = "simple"

    def __init__(self, pipeline, retriever: Retriever):
        self.pipeline = pipeline
        self.retriever = retriever

    def handle(self, raw_message: str, **_) -> dict:
        text = clean_customer(raw_message)
        if is_unhandleable(text):
            intent, conf = "unhandleable", 1.0
        else:
            probs = self.pipeline.predict_proba([text])[0]
            k = int(probs.argmax())
            intent, conf = str(self.pipeline.classes_[k]), float(probs[k])
        reply, sim = self.retriever.nearest_any(text)
        neighbours = self.retriever.nearest_any_k(text, k=5)
        esc = bool(KEYWORD_ESCALATE.search(text)) or intent == "unhandleable"
        return {"message": text, "intent": intent, "confidence": round(conf, 3), "reply": reply, "reply_mode": "nearest_neighbour",
                "escalate": esc, "escalation_score": float(esc), "reason": "Escalate: keyword hit" if esc else "Auto-handle: no keyword hit",
                "reply_cache_miss": False, "nn_similarity": round(sim, 3), "evidence": neighbours}


def train_simple(train_texts: list[str], train_labels: list[str]):
    pipe = make_pipeline(
        TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True),
        LogisticRegression(C=3.0, max_iter=2000, class_weight="balanced"),
    )
    pipe.fit(train_texts, train_labels)
    joblib.dump(pipe, SIMPLE_PATH)
    return pipe


def load_baselines(retriever: Retriever | None = None) -> tuple[TrivialBaseline, SimpleBaseline]:
    import pandas as pd

    labels = pd.read_parquet(C.PROCESSED / "weak_labels.parquet")
    majority = Counter(labels[labels.split == "corpus"].weak_intent).most_common(1)[0][0]
    return TrivialBaseline(majority), SimpleBaseline(joblib.load(SIMPLE_PATH), retriever or Retriever.load())


def main() -> None:
    import pandas as pd

    from .split import load_pairs

    pairs = load_pairs()
    labels = pd.read_parquet(C.PROCESSED / "weak_labels.parquet")
    roots = pairs[(pairs.turn_index == 0) & (pairs.split == "corpus")].merge(labels[["thread_id", "weak_intent"]], on="thread_id")
    roots = roots[roots.weak_intent != "unhandleable"]
    train_simple(roots.root_text.tolist(), roots.weak_intent.tolist())
    print(f"simple baseline trained on {len(roots)} -> {SIMPLE_PATH}")


if __name__ == "__main__":
    main()
