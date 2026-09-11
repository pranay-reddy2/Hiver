"""Weak intent labels for training the classifier.

Two labelers:
  * rule_label: keyword rules from configs/intents.yaml (no API, used as bootstrap and by
    the keyword baseline).
  * llm_label: batched structured-output calls, cached. Overrides rule labels when present.
Neither is ever used on the golden set, which is hand-labelled.
"""
from __future__ import annotations

import re

import pandas as pd

from . import config as C
from .llm import CacheMiss, get_llm
from .text import is_unhandleable

_RULES: dict[str, list[re.Pattern]] = {
    intent: [re.compile(r"\b" + re.escape(k.lower()) + r"\b") for k in spec.get("rules", [])]
    for intent, spec in C.INTENTS_CFG["intents"].items()
}
# Priority when several intents fire: the more specific/actionable one wins.
_PRIORITY = [
    "family_plan",
    "offline_downloads",
    "billing_subscription",
    "account_access",
    "content_availability",
    "playback_app_bug",
    "feature_request_feedback",
]


def rule_label(clean_text: str) -> str:
    if is_unhandleable(clean_text):
        return "unhandleable"
    t = clean_text.lower()
    hits = {i: sum(1 for p in _RULES[i] if p.search(t)) for i in _PRIORITY}
    best = max(hits.values())
    if best == 0:
        return "other"
    for i in _PRIORITY:
        if hits[i] == best:
            return i
    return "other"


_SCHEMA = {
    "type": "object",
    "properties": {
        "labels": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "integer"},
                    "intent": {"type": "string", "enum": C.INTENTS},
                },
                "required": ["id", "intent"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["labels"],
    "additionalProperties": False,
}


def _intent_guide() -> str:
    return "\n".join(f"- {name}: {spec['description'].strip()}" for name, spec in C.INTENTS_CFG["intents"].items())


LABEL_SYSTEM = (
    "You label customer tweets sent to @SpotifyCares with exactly one intent. "
    "Use `unhandleable` only when there is no actionable content (bare mention, image-only, "
    "non-English, under three real words). Use `other` for a real request that fits no intent. "
    "When two intents apply, pick the one that determines what the support agent would do next.\n\n"
    "Intents:\n" + _intent_guide()
)


def llm_label_batch(texts: list[str], model: str = C.LABEL_MODEL) -> list[str]:
    llm = get_llm()
    user = "Label each tweet.\n\n" + "\n".join(f"[{i}] {t}" for i, t in enumerate(texts)) + "\n\nReturn a label for every id."
    out = llm.complete_json(system=LABEL_SYSTEM, user=user, schema=_SCHEMA, model=model, tag="weak_label", effort="low")
    by_id = {int(x["id"]): x["intent"] for x in out.get("labels", [])}
    return [by_id.get(i, rule_label(t)) for i, t in enumerate(texts)]


def weak_labels(df: pd.DataFrame, text_col: str = "root_text", batch: int = 25, use_llm: bool = True, workers: int = 4) -> pd.Series:
    from concurrent.futures import ThreadPoolExecutor

    rules = df[text_col].map(rule_label)
    if not use_llm:
        return rules
    out = rules.copy()
    texts = df[text_col].tolist()
    idx = list(df.index)
    starts = list(range(0, len(texts), batch))

    def run(start: int):
        chunk = texts[start : start + batch]
        try:
            return start, llm_label_batch(chunk)
        except CacheMiss:
            return start, None  # keep rule label; miss is recorded on the LLM object
        except Exception as e:  # noqa: BLE001 - one bad batch keeps its rule labels
            print(f"batch {start // batch}: {type(e).__name__}: {str(e)[:120]}", flush=True)
            return start, None

    done = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for start, labels in pool.map(run, starts):
            done += 1
            if done % 20 == 0:
                print(f"labelled batches {done}/{len(starts)}", flush=True)
            if labels is None:
                continue
            for j, lab in enumerate(labels):
                out.loc[idx[start + j]] = lab
    return out


def main() -> None:
    from .split import load_pairs

    pairs = load_pairs()
    roots = pairs[(pairs.turn_index == 0) & pairs.split.isin(["corpus", "dev"])].copy()
    roots["rule_intent"] = roots.message_text.map(rule_label)
    roots["weak_intent"] = weak_labels(roots, text_col="message_text")
    llm = get_llm()
    llm.fail_on_misses()  # never silently train on rule labels because the cache was cold
    print(llm.report_misses())
    keep = roots[["thread_id", "split", "rule_intent", "weak_intent"]]
    keep.to_parquet(C.PROCESSED / "weak_labels.parquet", index=False)
    print(keep.weak_intent.value_counts().to_dict())
    print("rule/llm agreement:", round((keep.rule_intent == keep.weak_intent).mean(), 3))


if __name__ == "__main__":
    main()
