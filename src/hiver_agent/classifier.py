"""Intent classifier: sentence embeddings + logistic regression.

Trained on weak labels from the corpus split; C is chosen on dev; golden is never seen.
`unhandleable` is decided by a rule before the classifier runs.
"""
from __future__ import annotations

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

from . import config as C
from .embed import embed
from .text import is_unhandleable

MODEL_PATH = C.MODELS / "intent_lr.joblib"


class IntentClassifier:
    def __init__(self, model: LogisticRegression, classes: list[str]):
        self.model = model
        self.classes = classes

    def predict(self, texts: list[str], version: str = "v1") -> list[tuple[str, float]]:
        out: list[tuple[str, float] | None] = [None] * len(texts)
        todo = [i for i, t in enumerate(texts) if not is_unhandleable(t, version)]
        todo_set = set(todo)
        for i in range(len(texts)):
            if i not in todo_set:
                out[i] = ("unhandleable", 1.0)
        if todo:
            vecs = embed([texts[i] for i in todo])
            probs = self.model.predict_proba(vecs)
            for j, i in enumerate(todo):
                k = int(np.argmax(probs[j]))
                out[i] = (self.classes[k], float(probs[j][k]))
        return out  # type: ignore[return-value]

    def save(self, path=MODEL_PATH) -> None:
        joblib.dump({"model": self.model, "classes": self.classes}, path)

    @classmethod
    def load(cls, path=MODEL_PATH) -> IntentClassifier:
        d = joblib.load(path)
        return cls(d["model"], d["classes"])


def train(train_texts: list[str], train_labels: list[str], dev_texts: list[str], dev_labels: list[str]) -> tuple[IntentClassifier, dict]:
    Xtr = embed(train_texts, cache_name="train")
    Xdv = embed(dev_texts, cache_name="dev")
    best, best_f1, log = None, -1.0, {}
    for c in (0.3, 1.0, 3.0, 10.0):
        lr = LogisticRegression(C=c, max_iter=2000, class_weight="balanced")
        lr.fit(Xtr, train_labels)
        f1 = f1_score(dev_labels, lr.predict(Xdv), average="macro")
        log[c] = round(float(f1), 4)
        if f1 > best_f1:
            best, best_f1 = lr, f1
    clf = IntentClassifier(best, list(best.classes_))
    return clf, {"dev_macro_f1_by_C": log, "chosen_C": float(best.C)}


def main() -> None:
    from .split import load_pairs

    pairs = load_pairs()
    labels = pd.read_parquet(C.PROCESSED / "weak_labels.parquet")
    roots = pairs[pairs.turn_index == 0].merge(labels[["thread_id", "weak_intent"]], on="thread_id")
    roots = roots[roots.weak_intent != "unhandleable"]
    tr = roots[roots.split == "corpus"]
    dv = roots[roots.split == "dev"]
    clf, info = train(tr.root_text.tolist(), tr.weak_intent.tolist(), dv.root_text.tolist(), dv.weak_intent.tolist())
    clf.save()
    print(info, "->", MODEL_PATH)


if __name__ == "__main__":
    main()
