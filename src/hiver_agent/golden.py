"""Golden set tooling: sample candidates for hand-labelling; compute annotator agreement.

Sampling (200 items from the golden_pool split, never from corpus or dev):
  * 100 stratified over rule-label intent so rare intents are represented.
  * 100 uniformly at random over the pool, reported separately ("random_uniform"; the pool spans
    nine weeks, so this is the traffic mix, not a time series).
The labelling guide is data/golden/README.md. Labels are typed into golden_labels.csv.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from . import config as C
from .labels import rule_label
from .split import load_pairs

CANDIDATES = C.GOLDEN / "golden_candidates.csv"
LABELS = C.GOLDEN / "golden_labels.csv"
ANNOTATOR2 = C.GOLDEN / "annotator2_labels.csv"
REPLY_RATINGS = C.GOLDEN / "human_reply_ratings.csv"
LABEL_COLS = ["intent", "escalate", "escalate_reason", "notes"]


def sample(n_strat: int = 100, n_random: int = 100, seed: int = C.SEED, exclude_thread_ids=None, prefix: str = "g") -> pd.DataFrame:
    pool = load_pairs("golden_pool")
    pool = pool[pool.turn_index == 0].copy()
    if exclude_thread_ids is not None:
        pool = pool[~pool.thread_id.isin(set(int(t) for t in exclude_thread_ids))]
    pool["rule_intent"] = pool.message_text.map(rule_label)
    rng = np.random.default_rng(seed)

    strata = [i for i in C.INTENTS if i != "unhandleable"]
    per = n_strat // len(strata)
    picks = []
    for intent in strata:
        grp = pool[pool.rule_intent == intent]
        take = min(per, len(grp))
        picks.append(grp.sample(take, random_state=int(rng.integers(1 << 31))))
    strat = pd.concat(picks)
    short = n_strat - len(strat)
    if short > 0:
        rest = pool.drop(strat.index)
        strat = pd.concat([strat, rest.sample(short, random_state=seed)])
    strat["sample_strategy"] = "stratified"

    rest = pool.drop(strat.index)
    rand = rest.sample(n_random, random_state=seed + 1)
    rand["sample_strategy"] = "random_uniform"

    out = pd.concat([strat, rand]).sample(frac=1, random_state=seed + 2).reset_index(drop=True)
    out.insert(0, "item_id", [f"{prefix}{i:03d}" for i in range(len(out))])
    cols = ["item_id", "thread_id", "created_at", "sample_strategy", "customer_raw", "message_text", "brand_text"]
    out = out[cols].rename(columns={"message_text": "message", "brand_text": "historical_reply"})
    for c in LABEL_COLS:
        out[c] = ""
    return out


def load_labels(path=LABELS) -> pd.DataFrame:
    df = pd.read_csv(path, dtype=str).fillna("")
    df = df[df.intent.str.strip() != ""].copy()
    bad = set(df.intent) - set(C.INTENTS)
    if bad:
        raise ValueError(f"unknown intents in {path.name}: {sorted(bad)}")
    df["escalate"] = df.escalate.str.strip().str.lower().map({"y": True, "yes": True, "1": True, "true": True, "n": False, "no": False, "0": False, "false": False})
    if df.escalate.isna().any():
        raise ValueError("escalate must be y/n for every labelled row")
    return df


def annotator_agreement(a: pd.DataFrame, b: pd.DataFrame) -> dict:
    from sklearn.metrics import cohen_kappa_score

    m = a.merge(b, on="item_id", suffixes=("_a", "_b"))
    res = {"n": int(len(m))}
    res["intent_kappa"] = round(float(cohen_kappa_score(m.intent_a, m.intent_b)), 3)
    res["intent_agreement"] = round(float((m.intent_a == m.intent_b).mean()), 3)
    no_other = m[(m.intent_a != "other") & (m.intent_b != "other")]
    if len(no_other) > 1:
        res["intent_kappa_excluding_other"] = round(float(cohen_kappa_score(no_other.intent_a, no_other.intent_b)), 3)
    res["escalate_kappa"] = round(float(cohen_kappa_score(m.escalate_a.astype(int), m.escalate_b.astype(int))), 3)
    res["escalate_agreement"] = round(float((m.escalate_a == m.escalate_b).mean()), 3)
    per_class = {}
    for intent in sorted(set(m.intent_a) | set(m.intent_b)):
        mask = (m.intent_a == intent) | (m.intent_b == intent)
        per_class[intent] = {"n": int(mask.sum()), "agreement": round(float((m[mask].intent_a == m[mask].intent_b).mean()), 3)}
    res["per_class_agreement"] = per_class
    return res


def label_status() -> dict:
    """What a human has and has not done yet: label provenance and overturn rate for both golden
    files, the second-annotator sheet, the reply-rating sheet, and the filter precision check."""
    from .evaluate import label_provenance

    out = {}
    for name in ("golden_labels.csv", "golden2_labels.csv"):
        path = C.GOLDEN / name
        if path.exists():
            out[name] = label_provenance(load_labels(path))
    if ANNOTATOR2.exists():
        a2 = pd.read_csv(ANNOTATOR2, dtype=str).fillna("")
        out["annotator2_labels.csv"] = {"n": int((a2.intent.str.strip() != "").sum()), "labeler": a2.labeler.value_counts().to_dict() if "labeler" in a2 else {}}
    if REPLY_RATINGS.exists():
        rr = pd.read_csv(REPLY_RATINGS, dtype=str).fillna("")
        out["human_reply_ratings.csv"] = {"rated": int((rr.resolution_fit.str.strip() != "").sum()), "of": int(len(rr))}
    fp = C.GOLDEN / "filter_precision_check.csv"
    if fp.exists():
        f = pd.read_csv(fp, dtype=str).fillna("")
        out["filter_precision_check.csv"] = {"checked": int((f.is_real_resolution.str.strip() != "").sum()), "of": int(len(f))}
    out["human_evidence_present"] = bool(
        out.get("golden_labels.csv", {}).get("human_labelled") and out.get("human_reply_ratings.csv", {}).get("rated", 0) > 0
    )
    return out


def main(cmd: str = "sample") -> None:
    if cmd == "sample":
        df = sample()
        if CANDIDATES.exists():
            raise SystemExit(f"{CANDIDATES} exists; delete it to resample (labels would be orphaned)")
        df.to_csv(CANDIDATES, index=False)
        df.sample(50, random_state=C.SEED + 3)[["item_id", "customer_raw", "customer_clean", "intent", "escalate", "escalate_reason"]].to_csv(C.GOLDEN / "annotator2_candidates.csv", index=False)
        print(f"{len(df)} candidates -> {CANDIDATES}; 50 -> annotator2_candidates.csv")
        print(df.sample_strategy.value_counts().to_dict())
    elif cmd == "sample2":
        # Fresh 100-item sample, disjoint from the first 200, for grading post-hoc fixes.
        first = pd.read_csv(CANDIDATES)
        df = sample(n_strat=50, n_random=50, seed=C.SEED + 7, exclude_thread_ids=first.thread_id, prefix="h")
        path = C.GOLDEN / "golden2_candidates.csv"
        if path.exists():
            raise SystemExit(f"{path} exists; delete it to resample")
        df.to_csv(path, index=False)
        print(f"{len(df)} fresh candidates -> {path}")
    elif cmd == "status":
        print(json.dumps(label_status(), indent=2))
    elif cmd == "agreement":
        a, b = load_labels(LABELS), load_labels(ANNOTATOR2)
        print(json.dumps(annotator_agreement(a, b), indent=2))


if __name__ == "__main__":
    import sys

    main(sys.argv[1] if len(sys.argv) > 1 else "sample")
