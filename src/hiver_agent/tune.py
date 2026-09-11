"""Tune escalation and retrieval thresholds on dev. Golden is never used here.

Dev has no hand escalation labels, so the proxy target is "the historical reply was a DM
redirect" (the brand itself decided the case needed private handling). This is a proxy and is
called out in the report.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from . import config as C
from .agent import DEFAULT_PARAMS, PARAMS_PATH
from .classifier import IntentClassifier
from .escalation import score as esc_score
from .filters import categorise
from .retrieval import Retriever
from .split import load_pairs


def main(n: int = 400) -> None:
    dev = load_pairs("dev")
    dev = dev[dev.turn_index == 0].sample(min(n, len(dev[dev.turn_index == 0])), random_state=C.SEED)
    dev["proxy_escalate"] = dev.brand_text.map(categorise) == "dm_redirect"
    clf, ret = IntentClassifier.load(), Retriever.load()
    preds = clf.predict(dev.message_text.tolist())
    top_sims, feats = [], []
    for (intent, conf), text in zip(preds, dev.message_text, strict=True):
        ex = ret.search(text, k=5, intent=intent) if intent != "unhandleable" else None
        top_sims.append(float(ex.similarity.max()) if ex is not None else 0.0)
        feats.append((text, intent, conf, ex, ret.dm_share(text) if ex is not None else 0.0))
    sim_thr = float(np.percentile([s for s in top_sims if s > 0], 25))

    best, log = None, {}
    for thr in np.arange(0.2, 0.75, 0.05):
        pred = np.array([esc_score(t, i, c, ex, thr, sim_thr, dm).escalate for t, i, c, ex, dm in feats])
        y = dev.proxy_escalate.values
        tp, fp, fn = (pred & y).sum(), (pred & ~y).sum(), (~pred & y).sum()
        p, r = tp / max(1, tp + fp), tp / max(1, tp + fn)
        f = 2 * p * r / max(1e-9, p + r)
        log[round(float(thr), 2)] = {"precision": round(float(p), 3), "recall": round(float(r), 3), "f1": round(float(f), 3), "rate": round(float(pred.mean()), 3)}
        if best is None or f > log[best]["f1"]:
            best = round(float(thr), 2)
    params = {**DEFAULT_PARAMS, "escalation_threshold": best, "sim_threshold": round(sim_thr, 3), "dev_sweep": log, "proxy": "historical reply was dm_redirect"}
    PARAMS_PATH.write_text(json.dumps(params, indent=2))
    print(json.dumps({k: v for k, v in params.items() if k != "dev_sweep"}, indent=2))
    print(pd.DataFrame(log).T.to_string())


if __name__ == "__main__":
    main()
