"""Command-line entry point. `python -m hiver_agent.cli <command>`; see Makefile for the order."""
from __future__ import annotations

import argparse
import json
import sys

import pandas as pd

from . import config as C


def cmd_demo(args):
    from .agent import SupportAgent

    res = SupportAgent().handle(args.message)
    print(json.dumps(res, indent=2, ensure_ascii=False, default=str))


def cmd_filter_check(args):
    """Sample 100 replies the filter admitted as resolutions, for a hand precision check."""
    from .filters import categorise
    from .split import load_pairs

    pairs = load_pairs("corpus")
    pairs["category"] = pairs.brand_text.map(categorise)
    adm = pairs[pairs.category.isin(["fix_steps", "policy_answer"])].sample(100, random_state=C.SEED)
    out = adm[["thread_id", "category", "message_text", "brand_text"]].copy()
    out["is_real_resolution"] = ""
    path = C.GOLDEN / "filter_precision_check.csv"
    if path.exists():
        raise SystemExit(f"{path} exists; delete to resample")
    out.to_csv(path, index=False)
    print(f"100 admitted replies -> {path}. Fill is_real_resolution with y/n, then `make filter-precision`.")


def cmd_filter_precision(args):
    df = pd.read_csv(C.GOLDEN / "filter_precision_check.csv", dtype=str).fillna("")
    df = df[df.is_real_resolution.str.strip() != ""]
    y = df.is_real_resolution.str.strip().str.lower().isin(["y", "yes", "1", "true"])
    print(json.dumps({"n": int(len(df)), "precision": round(float(y.mean()), 3), "by_category": df.assign(y=y).groupby("category").y.mean().round(3).to_dict()}, indent=2))


def cmd_judge_sheet(args):
    """Write 60 system drafts (from preds_system.csv) for human rating on the four axes."""
    from .judge import AXES

    m = pd.read_csv(C.REPORTS / "preds_system.csv")  # already joined with the gold columns
    m = m[m.reply.notna()].sample(min(60, int(m.reply.notna().sum())), random_state=C.SEED)
    sheet = m[["item_id", "message", "intent", "reply", "evidence"]].copy()
    for a in AXES:
        sheet[a] = ""
    path = C.GOLDEN / "human_reply_ratings.csv"
    if path.exists():
        raise SystemExit(f"{path} exists; delete to regenerate")
    sheet.to_csv(path, index=False)
    print(f"{len(sheet)} drafts -> {path}. Score each axis 1-5 per configs/rubric.md, then `make judge-agreement`.")


def cmd_judge_agreement(args):
    from .judge import AXES, agreement

    human = pd.read_csv(C.GOLDEN / "human_reply_ratings.csv", dtype=str).fillna("")
    human = human[human[AXES[0]].str.strip() != ""]
    machine = pd.read_csv(C.REPORTS / "preds_system.csv")[["item_id"] + AXES].dropna()
    print(json.dumps(agreement(human, machine), indent=2))


def cmd_eval(args):
    from .evaluate import evaluate
    from .llm import get_llm

    labels = C.GOLDEN / args.labels if args.labels else None
    kw = {"labels_path": labels} if labels else {}
    evaluate(with_judge=not args.no_judge, name=args.name, version=args.version, with_drafts=not args.no_drafts, **kw)
    llm = get_llm()
    if llm.misses and not llm.live:
        print("\n" + llm.report_misses(), file=sys.stderr)
        sys.exit(2)


def cmd_judge_cross(args):
    """Re-judge the system drafts with a second judge model and report judge-judge agreement."""
    from .evaluate import judge_frame
    from .judge import AXES, agreement
    from .llm import get_llm

    preds = pd.read_csv(C.REPORTS / "preds_system.csv")
    preds["evidence"] = preds.evidence.map(lambda s: eval(s) if isinstance(s, str) else [])
    preds["questions"] = preds.questions.map(lambda s: eval(s) if isinstance(s, str) else [])
    gold = pd.read_csv(C.GOLDEN / "golden_labels.csv", dtype=str)
    other = judge_frame(preds, gold, model=args.model)
    other.to_csv(C.REPORTS / f"judge_{args.model.split(':')[-1]}.csv", index=False)
    primary = preds[["item_id"] + AXES].dropna()
    print(json.dumps({"second_judge": args.model, **agreement(primary, other.dropna())}, indent=2))
    llm = get_llm()
    if llm.misses and not llm.live:
        print(llm.report_misses(), file=sys.stderr)
        sys.exit(2)


def main(argv=None):
    p = argparse.ArgumentParser(prog="hiver_agent")
    sub = p.add_subparsers(dest="cmd", required=True)
    for name, mod in [("prep", "prep"), ("split", "split"), ("labels", "labels"), ("train", "classifier"), ("index", "retrieval"), ("baselines", "baselines"), ("tune", "tune")]:
        sub.add_parser(name).set_defaults(func=lambda a, mod=mod: __import__(f"hiver_agent.{mod}", fromlist=["main"]).main())
    sub.add_parser("golden-sample").set_defaults(func=lambda a: __import__("hiver_agent.golden", fromlist=["main"]).main("sample"))
    sub.add_parser("label-status").set_defaults(func=lambda a: __import__("hiver_agent.golden", fromlist=["main"]).main("status"))
    sub.add_parser("agreement").set_defaults(func=lambda a: __import__("hiver_agent.golden", fromlist=["main"]).main("agreement"))
    sub.add_parser("filter-check").set_defaults(func=cmd_filter_check)
    sub.add_parser("filter-precision").set_defaults(func=cmd_filter_precision)
    sub.add_parser("judge-sheet").set_defaults(func=cmd_judge_sheet)
    sub.add_parser("judge-agreement").set_defaults(func=cmd_judge_agreement)
    e = sub.add_parser("eval")
    e.add_argument("--no-judge", action="store_true")
    e.add_argument("--no-drafts", action="store_true")
    e.add_argument("--labels", default=None, help="labels csv under data/golden (default golden_labels.csv)")
    e.add_argument("--name", default="results")
    e.add_argument("--version", default=None, choices=[None, "v1", "v2"])
    e.set_defaults(func=cmd_eval)
    jc = sub.add_parser("judge-cross")
    jc.add_argument("--model", default="gemini:gemini-3.5-flash")
    jc.set_defaults(func=cmd_judge_cross)
    sub.add_parser("golden-sample2").set_defaults(func=lambda a: __import__("hiver_agent.golden", fromlist=["main"]).main("sample2"))
    d = sub.add_parser("demo")
    d.add_argument("message")
    d.set_defaults(func=cmd_demo)
    m = sub.add_parser("models")
    m.add_argument("--provider", default="gemini")
    m.set_defaults(func=lambda a: print("\n".join(__import__("hiver_agent.llm", fromlist=["list_models"]).list_models(a.provider))))
    args = p.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
