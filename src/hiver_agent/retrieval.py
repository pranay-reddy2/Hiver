"""Retrieval over resolution-bearing historical replies plus a per-intent question bank."""
from __future__ import annotations

import re
from collections import Counter

import joblib
import numpy as np
import pandas as pd

from . import config as C
from .embed import embed
from .filters import RESOLUTION_CATEGORIES, categorise

INDEX_PATH = C.MODELS / "retrieval_index.joblib"


class Retriever:
    def __init__(self, frame: pd.DataFrame, vectors: np.ndarray, question_bank: dict[str, list[str]], all_frame: pd.DataFrame, all_vectors: np.ndarray):
        self.frame = frame.reset_index(drop=True)
        self.vectors = vectors
        self.question_bank = question_bank
        self.all_frame = all_frame.reset_index(drop=True)
        self.all_vectors = all_vectors

    INTENT_BONUS = 0.10  # soft preference for the predicted intent instead of a hard mask
    _GREETING = re.compile(r"^(hey|hi|hello|howdy)[^!.,:]*[!.,:]\s*", re.I)

    def _dedupe_key(self, text: str) -> str:
        return self._GREETING.sub("", text).lower().strip()

    def search(self, text: str, k: int = 5, intent: str | None = None) -> pd.DataFrame:
        """Top-k distinct resolutions. A wrong intent prediction no longer wipes out the evidence:
        the predicted intent gets a similarity bonus rather than an exclusive mask. Near-identical
        replies (same text after the greeting) are collapsed so the drafter sees distinct resolutions."""
        q = embed([text])[0]
        sims = self.vectors @ q
        frame = self.frame
        ranked = sims.copy()
        if intent:
            ranked = ranked + np.where((frame.weak_intent == intent).values, self.INTENT_BONUS, 0.0)
        order = np.argsort(-ranked)[: k * 8]
        seen, keep = set(), []
        for i in order:
            key = self._dedupe_key(frame.iloc[i].brand_text)
            if key in seen:
                continue
            seen.add(key)
            keep.append(i)
            if len(keep) == k:
                break
        out = frame.iloc[keep].copy()
        out["similarity"] = sims[keep]
        return out

    def dm_share(self, text: str, k: int = 5) -> float:
        """Share of the k nearest replies over the whole corpus that are DM redirects."""
        q = embed([text])[0]
        sims = self.all_vectors @ q
        top = np.argsort(-sims)[:k]
        return float((self.all_frame.iloc[top].category == "dm_redirect").mean())

    def nearest_any_k(self, text: str, k: int = 5) -> list[str]:
        q = embed([text])[0]
        sims = self.all_vectors @ q
        return self.all_frame.iloc[np.argsort(-sims)[:k]].brand_text.tolist()

    def nearest_any(self, text: str) -> tuple[str, float]:
        """Nearest neighbour over every historical reply, for the simple baseline."""
        q = embed([text])[0]
        sims = self.all_vectors @ q
        i = int(np.argmax(sims))
        return str(self.all_frame.iloc[i].brand_text), float(sims[i])

    def questions_for(self, intent: str, n: int = 2) -> list[str]:
        return self.question_bank.get(intent, self.question_bank.get("_global", []))[:n]

    def save(self, path=INDEX_PATH) -> None:
        joblib.dump(self, path)

    @classmethod
    def load(cls, path=INDEX_PATH) -> Retriever:
        return joblib.load(path)


def build(pairs: pd.DataFrame, labels: pd.DataFrame) -> Retriever:
    corpus = pairs[pairs.split == "corpus"].copy()
    corpus["category"] = corpus.brand_text.map(categorise)
    corpus = corpus.merge(labels[["thread_id", "weak_intent"]], on="thread_id", how="left")
    corpus["weak_intent"] = corpus.weak_intent.fillna("other")

    res = corpus[corpus.category.isin(RESOLUTION_CATEGORIES)].reset_index(drop=True)
    vec = embed(res.query_text.tolist(), cache_name="corpus_resolutions")
    all_vec = embed(corpus.query_text.tolist(), cache_name="corpus_all")

    bank: dict[str, list[str]] = {}
    diag = corpus[corpus.category == "diagnostic_q"]
    for intent, grp in diag.groupby("weak_intent"):
        counts = Counter(grp.brand_text.str.replace(r"^(Hey|Hi)[^!.,]*[!.,]\s*", "", regex=True))
        bank[intent] = [q for q, _ in counts.most_common(3)]
    bank["_global"] = [q for q, _ in Counter(diag.brand_text).most_common(3)]
    keep_cols = ["thread_id", "query_text", "message_text", "brand_text", "category", "weak_intent", "turn_index"]
    return Retriever(res[keep_cols], vec, bank, corpus[keep_cols], all_vec)


def main() -> None:
    from .split import load_pairs

    pairs = load_pairs()
    labels = pd.read_parquet(C.PROCESSED / "weak_labels.parquet")
    r = build(pairs, labels)
    r.save()
    print(f"resolution corpus={len(r.frame)} all={len(r.all_frame)} banks={list(r.question_bank)} -> {INDEX_PATH}")


if __name__ == "__main__":
    main()
