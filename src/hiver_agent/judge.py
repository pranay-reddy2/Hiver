"""LLM-as-judge for reply quality. Absolute 1-5 on four axes, anchored rubric in configs/rubric.md."""
from __future__ import annotations

import pandas as pd

from . import config as C
from .llm import CacheMiss, get_llm

AXES = ["groundedness", "resolution_fit", "brand_voice", "safety_scope"]
JUDGE_SCHEMA = {
    "type": "object",
    "properties": {
        **{a: {"type": "integer", "minimum": 1, "maximum": 5} for a in AXES},
        "rationale": {"type": "string"},
    },
    "required": AXES + ["rationale"],
    "additionalProperties": False,
}


def rubric_text() -> str:
    return (C.CONFIGS / "rubric.md").read_text(encoding="utf-8")


def judge_system() -> str:
    return (
        "You are grading a draft public reply from @SpotifyCares to a customer tweet. Score each axis "
        "from 1 to 5 using ONLY the rubric below. Be strict: a 5 requires meeting every condition. "
        "Retrieved historical replies are the only admissible evidence for groundedness.\n\n" + rubric_text()
    )


def judge(message: str, reply: str, evidence: list[str], intent: str, model: str = C.JUDGE_MODEL) -> dict | None:
    llm = get_llm()
    user = (
        f"Customer message: {message}\nPredicted intent: {intent}\n\n"
        "Retrieved historical replies (evidence):\n" + "\n".join(f"- {e}" for e in evidence) + "\n\n"
        f"Draft reply to grade:\n{reply}\n\nScore all four axes."
    )
    try:
        return llm.complete_json(system=judge_system(), user=user, schema=JUDGE_SCHEMA, model=model, tag="judge", effort="low")
    except CacheMiss:
        return None
    except Exception as e:  # noqa: BLE001
        print(f"judge failed: {type(e).__name__}: {str(e)[:100]}", flush=True)
        return None


def agreement(human: pd.DataFrame, machine: pd.DataFrame) -> dict:
    """Spearman and quadratic-weighted kappa per axis between human and judge ratings."""
    from scipy.stats import spearmanr
    from sklearn.metrics import cohen_kappa_score

    m = human.merge(machine, on="item_id", suffixes=("_h", "_j"))
    out = {"n": int(len(m))}
    for a in AXES:
        h, j = m[f"{a}_h"].astype(int), m[f"{a}_j"].astype(int)
        rho = spearmanr(h, j).correlation if h.nunique() > 1 and j.nunique() > 1 else float("nan")
        out[a] = {
            "spearman": round(float(rho), 3),
            "weighted_kappa": round(float(cohen_kappa_score(h, j, weights="quadratic")), 3),
            "exact_agreement": round(float((h == j).mean()), 3),
            "within_1": round(float((abs(h - j) <= 1).mean()), 3),
        }
    return out
