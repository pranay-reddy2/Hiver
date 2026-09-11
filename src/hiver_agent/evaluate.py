"""Evaluation harness. Runs the system and both baselines over a labelled golden set.

Reports intent accuracy / macro-F1 with a confusion matrix, escalation precision / recall / F1 with
per-intent recall and a per-signal ablation, paired-bootstrap CIs for every system-vs-simple gap,
the threshold sweep (re-scored from stored signals), automatic groundedness and format checks, LLM
judge scores where each system is judged against its own evidence, and draft cost / latency.
Writes reports/results.json and reports/results.md (or a named variant).
"""
from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support

from . import config as C
from .agent import SupportAgent, load_params
from .baselines import load_baselines
from .escalation import WEIGHTS, score_from_signals
from .golden import LABELS, load_labels
from .judge import AXES, judge
from .llm import get_llm
from .split import assert_no_leakage, load_pairs

WORKERS = 4


# ---------------------------------------------------------------- statistics
def bootstrap_ci(values: np.ndarray, fn, n: int = 1000, seed: int = C.SEED) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    stats = [fn(values[rng.integers(0, len(values), len(values))]) for _ in range(n)]
    return float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5))


def paired_bootstrap(a: pd.DataFrame, b: pd.DataFrame, stat, n: int = 2000, seed: int = C.SEED) -> dict:
    """CI on stat(a_resampled) - stat(b_resampled) with the same item resample for both systems."""
    rng = np.random.default_rng(seed)
    idx = np.arange(len(a))
    diffs = []
    for _ in range(n):
        s = rng.integers(0, len(idx), len(idx))
        diffs.append(stat(a.iloc[s]) - stat(b.iloc[s]))
    diffs = np.array(diffs)
    return {"diff": round(float(stat(a) - stat(b)), 3), "ci95": [round(float(np.percentile(diffs, 2.5)), 3), round(float(np.percentile(diffs, 97.5)), 3)],
            "p_diff_le_0": round(float((diffs <= 0).mean()), 3)}


def _f1(df: pd.DataFrame) -> float:
    _, _, f, _ = precision_recall_fscore_support(df.escalate_gold.astype(int), df.escalate.astype(int), average="binary", zero_division=0)
    return float(f)


def _acc(df: pd.DataFrame) -> float:
    m = df[df.intent_gold != "unhandleable"]
    return float((m.intent_gold == m.intent).mean()) if len(m) else 0.0


# ---------------------------------------------------------------- metrics
def intent_metrics(gold: pd.Series, pred: pd.Series) -> dict:
    mask = gold != "unhandleable"
    g, p = gold[mask], pred[mask]
    correct = (g.values == p.values).astype(float)
    lo, hi = bootstrap_ci(correct, np.mean)
    labels = [i for i in C.INTENTS if i != "unhandleable"]
    cm = confusion_matrix(g, p, labels=labels)
    return {
        "n": int(mask.sum()),
        "accuracy": round(float(accuracy_score(g, p)), 3),
        "accuracy_ci95": [round(lo, 3), round(hi, 3)],
        "macro_f1": round(float(f1_score(g, p, average="macro")), 3),
        "confusion": {"labels": labels, "rows_gold_cols_pred": cm.tolist()},
        "unhandleable_detection": {
            "gold_n": int((~mask).sum()),
            "recall": round(float((pred[~mask] == "unhandleable").mean()), 3) if (~mask).any() else None,
            "false_positives": int(((pred == "unhandleable") & mask).sum()),
        },
    }


def escalation_metrics(gold: pd.Series, pred: pd.Series) -> dict:
    p, r, f, _ = precision_recall_fscore_support(gold.astype(int), pred.astype(int), average="binary", zero_division=0)
    return {"precision": round(float(p), 3), "recall": round(float(r), 3), "f1": round(float(f), 3), "escalation_rate": round(float(pred.mean()), 3), "gold_rate": round(float(gold.mean()), 3)}


def reference_similarity(replies: pd.Series, references: pd.Series) -> pd.Series:
    """Cosine similarity (MiniLM) between each draft and the reply the brand actually sent to that
    customer. Judge-free and reference-based; a missing draft scores 0."""
    from .embed import embed

    mask = replies.notna() & (replies.astype(str).str.strip() != "")
    out = pd.Series(0.0, index=replies.index)
    if mask.any():
        a = embed(replies[mask].astype(str).tolist())
        b = embed(references[mask].astype(str).tolist())
        out[mask] = (a * b).sum(axis=1)
    return out.round(4)


def label_provenance(gold: pd.DataFrame) -> dict:
    """Who wrote the labels, and how many rows the reviewer changed from the model draft."""
    out = {"labeler": gold.labeler.value_counts().to_dict() if "labeler" in gold else {"unknown": int(len(gold))}}
    if "draft_intent" in gold and "draft_escalate" in gold:
        draft_esc = gold.draft_escalate.str.strip().str.lower().isin(["y", "yes", "1", "true"])
        changed = (gold.intent != gold.draft_intent) | (gold.escalate != draft_esc)
        out["overturned_vs_model_draft"] = {"n": int(changed.sum()), "rate": round(float(changed.mean()), 3),
                                            "intent": int((gold.intent != gold.draft_intent).sum()), "escalate": int((gold.escalate != draft_esc).sum())}
    out["human_labelled"] = not any(("draft" in str(k).lower()) or ("model" in str(k).lower()) or ("gemini" in str(k).lower()) or ("claude" in str(k).lower()) for k in out["labeler"])
    return out


def per_intent_recall(m: pd.DataFrame) -> dict:
    g = m[m.escalate_gold]
    return {k: {"n": int(len(v)), "recall": round(float(v.escalate.mean()), 2)} for k, v in g.groupby("intent_gold")}


def signal_ablation(m: pd.DataFrame, threshold: float) -> dict:
    """Drop one signal at a time and re-score from stored signals. `unhandleable` rows keep their rule.
    Also scores `sensitive_intent_only`: escalate iff the intent is billing/account (or unhandleable),
    the one-rule baseline the six-signal scorer has to beat."""
    out = {}
    only = [bool(s.get("unhandleable") or s.get("sensitive_intent")) for s in m.signals]
    e = escalation_metrics(m.escalate_gold, pd.Series(only, index=m.index))
    out["sensitive_intent_only"] = {k: e[k] for k in ("precision", "recall", "f1", "escalation_rate")}
    for drop in [None] + list(WEIGHTS):
        pred = []
        for sig in m.signals:
            if sig.get("unhandleable"):
                pred.append(True)
            else:
                pred.append(score_from_signals(sig, threshold, drop=drop).escalate)
        e = escalation_metrics(m.escalate_gold, pd.Series(pred, index=m.index))
        out["full" if drop is None else f"without_{drop}"] = {k: e[k] for k in ("precision", "recall", "f1", "escalation_rate")}
    return out


def threshold_sweep(m: pd.DataFrame, base: float) -> dict:
    out = {}
    for delta in (-0.1, 0.0, 0.1):
        thr = round(base + delta, 2)
        pred = [True if s.get("unhandleable") else score_from_signals(s, thr).escalate for s in m.signals]
        out[str(thr)] = escalation_metrics(m.escalate_gold, pd.Series(pred, index=m.index))
    return out


# ---------------------------------------------------------------- running
def run_system(system, rows: pd.DataFrame, draft_reply: bool = True) -> pd.DataFrame:
    is_agent = isinstance(system, SupportAgent)
    out = []
    for r in rows.itertuples(index=False):  # classify + retrieve serially (local models)
        res = system.handle(r.message, draft_reply=False) if is_agent else system.handle(r.message)
        res["item_id"] = r.item_id
        out.append(res)
    if is_agent and draft_reply:  # LLM drafts in parallel
        with ThreadPoolExecutor(max_workers=WORKERS) as pool:
            out = list(pool.map(system.draft_for, out))
    for res in out:
        res.pop("_examples", None)
    return pd.DataFrame(out)


def judge_frame(preds: pd.DataFrame, gold: pd.DataFrame, model: str | None = None) -> pd.DataFrame:
    """Each system is judged against its own evidence (what it retrieved or copied from)."""
    gold_by_id = gold.set_index("item_id")
    kw = {"model": model} if model else {}

    def one(r):
        if not isinstance(r.reply, str) or not r.reply:
            return {"item_id": r.item_id, **{a: None for a in AXES}}
        evidence = list(r.evidence or []) + [f"(historical diagnostic question) {q}" for q in (getattr(r, "questions", None) or [])]
        j = judge(gold_by_id.loc[r.item_id].message, r.reply, evidence, r.intent, **kw)
        return {"item_id": r.item_id, **({a: j[a] for a in AXES} if j else {a: None for a in AXES})}

    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        scores = list(pool.map(one, list(preds.itertuples(index=False))))
    return pd.DataFrame(scores)


def evaluate(with_judge: bool = True, labels_path=LABELS, name: str = "results", version: str | None = None, with_drafts: bool = True) -> dict:
    t0 = time.time()
    gold = load_labels(labels_path)
    pairs = load_pairs()
    assert_no_leakage(pairs, gold.thread_id.astype(int))
    params = load_params()
    agent = SupportAgent(params=params) if version is None else SupportAgent(params=params, version=version)
    trivial, simple = load_baselines(agent.retriever)

    systems = {"trivial": trivial, "simple": simple, "system": agent}
    preds = {n: run_system(s, gold, draft_reply=with_drafts) for n, s in systems.items()}
    merged = {n: gold.merge(df, on="item_id", suffixes=("_gold", "")) for n, df in preds.items()}

    results: dict = {"name": name, "n_gold": int(len(gold)), "params": params, "system_version": agent.version, "label_provenance": label_provenance(gold), "systems": {}}
    for n, m in merged.items():
        entry = {"intent": intent_metrics(m.intent_gold, m.intent), "escalation": escalation_metrics(m.escalate_gold, m.escalate), "by_sample_strategy": {}}
        if with_drafts and "reply" in m:
            m["ref_sim"] = reference_similarity(m.reply, m.historical_reply)
            has = m.reply.notna() & (m.reply.astype(str).str.strip() != "")
            entry["reference_similarity"] = {"n": int(has.sum()), "mean": round(float(m.ref_sim[has].mean()), 3) if has.any() else None, "mean_incl_missing_as_0": round(float(m.ref_sim.mean()), 3)}
        for strat, grp in m.groupby("sample_strategy"):
            entry["by_sample_strategy"][strat] = {"intent": intent_metrics(grp.intent_gold, grp.intent), "escalation": escalation_metrics(grp.escalate_gold, grp.escalate)}
        if n == "system":
            entry["escalation"]["per_intent_recall"] = per_intent_recall(m)
            entry["escalation_threshold_sweep"] = threshold_sweep(m, params["escalation_threshold"])
            entry["signal_ablation"] = signal_ablation(m, params["escalation_threshold"])
            if with_drafts:
                gr = m.groundedness.dropna()
                entry["auto_groundedness"] = {"n": int(len(gr)), "grounded_rate": round(float(np.mean([g["grounded"] for g in gr])), 3) if len(gr) else None}
                ck = pd.DataFrame([c for c in m.checks.dropna()])
                entry["format_checks"] = {k: round(float(ck[k].mean()), 3) for k in ("under_280", "at_most_one_emoji", "asks_for_given_info")} | {"mean_chars": round(float(ck.chars.mean()), 1)} if len(ck) else None
                usage = pd.DataFrame([u for u in m.usage.dropna()])
                secs = m.seconds.dropna()
                entry["draft_cost_latency"] = {
                    "n_live_calls": int(len(secs)),
                    "mean_input_tokens": round(float(usage.input.mean()), 0) if len(usage) else None,
                    "mean_output_tokens": round(float(usage.output.mean()), 0) if len(usage) else None,
                    "mean_thinking_tokens": round(float(usage.thinking.dropna().mean()), 0) if len(usage) and "thinking" in usage and usage.thinking.notna().any() else None,
                    "mean_seconds": round(float(secs.mean()), 2) if len(secs) else None,
                    "p90_seconds": round(float(secs.quantile(0.9)), 2) if len(secs) else None,
                }
                entry["reply_cache_misses"] = int(m.reply_cache_miss.sum())
                entry["reply_modes"] = m.reply_mode.value_counts().to_dict()
        if with_judge and with_drafts:
            jf = judge_frame(preds[n], gold)
            entry["judge"] = {a: (round(float(jf[a].dropna().mean()), 2) if jf[a].notna().any() else None) for a in AXES}
            entry["judge"]["n_scored"] = int(jf[AXES[0]].notna().sum())
            merged[n] = m.merge(jf, on="item_id", suffixes=("_auto", ""))
        results["systems"][n] = entry

    # Paired bootstrap on every system-vs-simple gap, same item resample for both.
    a, b = merged["system"], merged["simple"]
    gaps = {"intent_accuracy": paired_bootstrap(a, b, _acc), "escalation_f1": paired_bootstrap(a, b, _f1)}
    if with_drafts and "ref_sim" in a and "ref_sim" in b:
        gaps["reference_similarity"] = paired_bootstrap(a, b, lambda d: float(d.ref_sim.mean()))
    if with_judge and with_drafts:
        for ax in AXES:
            ja, jb = a[["item_id", ax]].fillna(1), b[["item_id", ax]].fillna(1)  # a missing draft scores 1
            both = ja.merge(jb, on="item_id", suffixes=("_a", "_b"))
            gaps[f"judge_{ax}"] = paired_bootstrap(both[[f"{ax}_a"]].rename(columns={f"{ax}_a": "v"}), both[[f"{ax}_b"]].rename(columns={f"{ax}_b": "v"}), lambda d: float(d.v.mean()))
    results["system_vs_simple_paired_bootstrap"] = gaps

    llm = get_llm()
    results["llm_cache"] = llm.report_misses()
    results["wall_seconds"] = round(time.time() - t0, 1)
    # An offline run with cache misses is incomplete (null drafts, unscored judges). It must never
    # overwrite a complete run, so its outputs go to *_partial and the console says so.
    if llm.misses and not llm.live:
        name = f"{name}_partial"
        results["partial"] = True
        print(f"WARNING: cache misses; writing outputs as {name}.* and leaving the complete run untouched")
    suffix = "" if name == "results" else "_" + name
    for n, m in merged.items():
        m.drop(columns=[c for c in ("retrieved",) if c in m], errors="ignore").to_csv(C.REPORTS / f"preds_{n}{suffix}.csv", index=False)
    (C.REPORTS / f"{name}.json").write_text(json.dumps(results, indent=2))
    (C.REPORTS / f"{name}.md").write_text(render_md(results))
    print(render_md(results))
    return results


# ---------------------------------------------------------------- rendering
def render_md(res: dict) -> str:
    L = [f"# {res['name']}: golden set n={res['n_gold']}, system version {res.get('system_version')}", ""]
    prov = res.get("label_provenance")
    if prov:
        L.append("Label provenance: " + ", ".join(f"{k}: {v}" for k, v in prov["labeler"].items())
                 + ("" if prov.get("human_labelled") else "  **(model-drafted labels; every number below is agreement with a model's reading of the guide)**"))
        if "overturned_vs_model_draft" in prov:
            o = prov["overturned_vs_model_draft"]
            L.append(f"Rows changed from the model draft: {o['n']} of {res['n_gold']} ({o['rate']}); intent {o['intent']}, escalate {o['escalate']}")
        L.append("")
    has_judge = any("judge" in e for e in res["systems"].values())
    L.append("| system | intent acc (95% CI) | macro-F1 | esc. precision | esc. recall | esc. F1 | esc. rate | " + (" | ".join(f"judge {a}" for a in AXES) + " |" if has_judge else ""))
    L.append("|" + "---|" * (7 + (len(AXES) if has_judge else 0)))
    for n, e in res["systems"].items():
        i, s, j = e["intent"], e["escalation"], e.get("judge", {})
        row = f"| {n} | {i['accuracy']} ({i['accuracy_ci95'][0]}–{i['accuracy_ci95'][1]}) | {i['macro_f1']} | {s['precision']} | {s['recall']} | {s['f1']} | {s['escalation_rate']} |"
        if has_judge:
            row += " " + " | ".join(str(j.get(a, "–")) for a in AXES) + " |"
        L.append(row)
    if any("reference_similarity" in e for e in res["systems"].values()):
        L += ["", "Reference similarity (MiniLM cosine between the reply and the reply the brand actually sent; judge-free):", "", "| system | n with reply | mean | mean, missing reply = 0 |", "|---|---|---|---|"]
        for n, e in res["systems"].items():
            r = e.get("reference_similarity")
            if r:
                L.append(f"| {n} | {r['n']} | {r['mean']} | {r['mean_incl_missing_as_0']} |")
    g = res["system_vs_simple_paired_bootstrap"]
    L += ["", "System minus simple baseline, paired bootstrap (2000 resamples):", "", "| metric | diff | 95% CI | P(diff ≤ 0) |", "|---|---|---|---|"]
    for k, v in g.items():
        L.append(f"| {k} | {v['diff']:+.3f} | [{v['ci95'][0]:+.3f}, {v['ci95'][1]:+.3f}] | {v['p_diff_le_0']} |")
    sysr = res["systems"]["system"]
    L += ["", "Escalation threshold sweep (system, re-scored from stored signals):", "", "| threshold | precision | recall | rate |", "|---|---|---|---|"]
    for thr, m in sysr["escalation_threshold_sweep"].items():
        L.append(f"| {thr} | {m['precision']} | {m['recall']} | {m['escalation_rate']} |")
    L += ["", "Escalation recall by gold intent (system):", "", "| gold intent | n | recall |", "|---|---|---|"]
    for k, v in sysr["escalation"]["per_intent_recall"].items():
        L.append(f"| {k} | {v['n']} | {v['recall']} |")
    L += ["", "Signal ablation (system, drop one signal; `sensitive_intent_only` is the one-rule baseline):", "", "| config | precision | recall | F1 | rate |", "|---|---|---|---|---|"]
    for k, v in sysr["signal_ablation"].items():
        L.append(f"| {k} | {v['precision']} | {v['recall']} | {v['f1']} | {v['escalation_rate']} |")
    cm = sysr["intent"]["confusion"]
    L += ["", "Intent confusion (system; rows = gold, cols = predicted):", "", "| gold \\ pred | " + " | ".join(lab[:8] for lab in cm["labels"]) + " |", "|---|" + "---|" * len(cm["labels"])]
    for lab, row in zip(cm["labels"], cm["rows_gold_cols_pred"], strict=True):
        L.append(f"| {lab} | " + " | ".join(str(x) for x in row) + " |")
    ud = sysr["intent"]["unhandleable_detection"]
    L += ["", f"Unhandleable detection by the classifier rule: gold n={ud['gold_n']}, recall={ud['recall']}, false positives={ud['false_positives']} (escalation recall on those rows is in the per-intent table above; other signals can still escalate them)"]
    L += ["", "By sample strategy (intent accuracy / escalation F1):", ""]
    for n, e in res["systems"].items():
        L.append(f"- {n}: " + "; ".join(f"{k}: {v['intent']['accuracy']} / {v['escalation']['f1']}" for k, v in e["by_sample_strategy"].items()))
    if sysr.get("auto_groundedness"):
        L += ["", f"Automatic groundedness (links only): {sysr['auto_groundedness']}", f"Format checks: {sysr.get('format_checks')}", f"Draft cost / latency: {sysr.get('draft_cost_latency')}", f"Reply modes: {sysr.get('reply_modes')}", f"Reply cache misses: {sysr.get('reply_cache_misses')}"]
    L += ["", f"Gold escalation rate: {sysr['escalation']['gold_rate']}", res["llm_cache"], f"Wall time: {res['wall_seconds']}s"]
    return "\n".join(L)
