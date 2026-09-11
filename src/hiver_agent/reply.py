"""Reply drafter: RAG over historical resolutions, with a hard grounding rule."""
from __future__ import annotations

import re

import pandas as pd

from . import config as C
from .llm import CacheMiss, get_llm
from .text import links_in

REPLY_SCHEMA = {
    "type": "object",
    "properties": {
        "reply": {"type": "string"},
        "mode": {"type": "string", "enum": ["fix", "policy", "diagnostic", "none"]},
        "used_examples": {"type": "array", "items": {"type": "integer"}},
        "rationale": {"type": "string"},
    },
    "required": ["reply", "mode", "used_examples", "rationale"],
    "additionalProperties": False,
}

SYSTEM = """You draft public Twitter replies for @SpotifyCares, Spotify's support account.

Voice: friendly and casual, first person plural ("we"), one short greeting, at most one emoji,
under 280 characters, no signature. Typical closers: "Let us know how it goes", "Keep us posted".

Hard rules:
1. Only give steps, facts, or links that appear in the retrieved historical replies. Links must be
   copied verbatim as [link:ID] tokens. Never invent a URL, a policy, a refund, a release date, or an ETA.
2. If a retrieved reply resolves the same problem, give that resolution (mode "fix" or "policy").
3. If no retrieved reply resolves it, ask the provided diagnostic questions instead (mode "diagnostic").
   Do not tell the customer to DM us unless a retrieved reply for the same problem does so.
4. Never ask for passwords, card numbers, or other private data in public.
5. Address the customer's actual message. Do not repeat information they already gave."""


def _fmt_examples(examples: pd.DataFrame) -> str:
    lines = []
    for i, r in enumerate(examples.itertuples(index=False)):
        lines.append(f"[{i}] (sim={r.similarity:.1f}, {r.category})\n  customer: {r.query_text[:300]}\n  reply: {r.brand_text}")
    return "\n".join(lines)


def build_user_prompt(message: str, intent: str, examples: pd.DataFrame, questions: list[str]) -> str:
    return (
        f"Customer message: {message}\n\nPredicted intent: {intent}\n\n"
        f"Retrieved historical replies:\n{_fmt_examples(examples)}\n\n"
        f"Diagnostic questions historically asked for this intent:\n" + "\n".join(f"- {q}" for q in questions) + "\n\n"
        "Draft the reply."
    )


EMOJI = re.compile(r"[\U0001F300-\U0001FAFF\u2600-\u27BF\U0001F900-\U0001F9FF]")
ASKS_DEVICE = re.compile(r"\b(what|which) (device|phone|operating system|os|version)|device.{0,20}(version|os)\b", re.I)
GIVES_DEVICE = re.compile(r"\b(iphone|ipad|android|ios ?\d|windows|mac(os| os)?|desktop|web player|chromecast|sonos|xbox|ps4|version \d|\d+\.\d+\.\d+)\b", re.I)


def groundedness_check(reply: str, examples: pd.DataFrame) -> dict:
    """Cheap automatic check: every link in the draft must appear in a retrieved reply."""
    allowed = set(link for t in examples.brand_text for link in links_in(t))
    used = set(links_in(reply))
    bad_links = sorted(used - allowed)
    raw_urls = re.findall(r"https?://\S+", reply)
    return {"ungrounded_links": bad_links, "raw_urls": raw_urls, "grounded": not bad_links and not raw_urls}


def reply_checks(reply: str, message: str) -> dict:
    """Deterministic format checks that sit beside the judge: length, emoji count, and whether the
    draft asks for device/OS/version the customer already gave."""
    rendered = re.sub(r"\[link:\w+\]", "https://t.co/xxxxxxxxxx", reply)
    return {
        "chars": len(rendered),
        "under_280": len(rendered) <= 280,
        "emoji_count": len(EMOJI.findall(reply)),
        "at_most_one_emoji": len(EMOJI.findall(reply)) <= 1,
        "asks_for_given_info": bool(ASKS_DEVICE.search(reply)) and bool(GIVES_DEVICE.search(message)),
    }


def draft(message: str, intent: str, examples: pd.DataFrame, questions: list[str], model: str = C.GEN_MODEL) -> dict:
    llm = get_llm()
    user = build_user_prompt(message, intent, examples, questions)
    try:
        out = llm.complete_json(system=SYSTEM, user=user, schema=REPLY_SCHEMA, model=model, tag="draft", effort="medium")
    except CacheMiss:
        return {"reply": None, "mode": "none", "used_examples": [], "rationale": "cache miss", "cache_miss": True}
    except Exception as e:  # noqa: BLE001 - quota or transport failure: record, do not kill the run
        print(f"draft failed: {type(e).__name__}: {str(e)[:100]}", flush=True)
        return {"reply": None, "mode": "none", "used_examples": [], "rationale": f"api error: {type(e).__name__}", "cache_miss": True}
    if out.get("_refusal"):
        return {"reply": None, "mode": "none", "used_examples": [], "rationale": "refusal", "cache_miss": False}
    out["cache_miss"] = False
    out["groundedness"] = groundedness_check(out["reply"], examples)
    out["checks"] = reply_checks(out["reply"], message)
    meta = llm.meta_for(system=SYSTEM, user=user, schema=REPLY_SCHEMA, model=model, effort="medium")
    out["usage"] = (meta or {}).get("usage")
    out["seconds"] = (meta or {}).get("seconds")
    return out
