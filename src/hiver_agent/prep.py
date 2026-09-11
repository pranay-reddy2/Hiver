"""Stage 1: raw Kaggle CSV -> customer/brand pairs for one brand.

Output: data/processed/pairs.parquet with one row per (customer tweet, brand reply).
Numbered multi-tweet replies ("1: ...", "2: ...") are joined into one reply.
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

from . import config as C
from .text import clean_brand, clean_customer, is_unhandleable, text_hash

CONTINUATION = re.compile(r"^@\w+\s+\d+\s*[:.)]")
DATE_FMT = "%a %b %d %H:%M:%S %z %Y"


def _split_ids(value) -> list[int]:
    if pd.isna(value):
        return []
    return [int(float(x)) for x in str(value).split(",") if x.strip()]  # float() tolerates "3.0" from NaN-typed columns


def load_brand_frame(path=C.RAW, brand: str = C.BRAND) -> pd.DataFrame:
    df = pd.read_csv(path, dtype={"tweet_id": "int64"}, low_memory=False)
    df["in_response_to_tweet_id"] = df["in_response_to_tweet_id"].astype("Int64")
    by_id = df.set_index("tweet_id")
    keep = set(df.tweet_id[df.author_id == brand])
    # Walk parents up to each thread root so _root_of can resolve depth without the full frame.
    frontier = set(df.loc[df.author_id == brand, "in_response_to_tweet_id"].dropna().astype(int))
    for _ in range(14):
        frontier = {t for t in frontier if t in by_id.index} - keep
        if not frontier:
            break
        keep |= frontier
        frontier = set(by_id.loc[list(frontier), "in_response_to_tweet_id"].dropna().astype(int))
    return df[df.tweet_id.isin(keep)].reset_index(drop=True)


NUMBER = re.compile(r"^@\w+\s+(\d+)\s*[:.)]")


def _part_number(text: str) -> int | None:
    m = NUMBER.match(str(text))
    return int(m.group(1)) if m else None


def _join_continuations(first: pd.Series, by_id: dict[int, pd.Series], brand: str) -> tuple[str, int]:
    """Walk response_tweet_id from a brand reply and append continuation tweets numbered N+1.

    Each part is cleaned separately so per-part signatures ("/JK") are removed before joining.
    """
    parts = [clean_brand(first.text)]
    cur, n = first, _part_number(first.text)
    seen = {int(first.tweet_id)}
    for _ in range(6):
        if n is None:
            break
        nxt = None
        for cid in _split_ids(cur.response_tweet_id):
            child = by_id.get(cid)
            if child is not None and child.author_id == brand and cid not in seen and _part_number(child.text) == n + 1:
                nxt = child
                break
        if nxt is None:
            break
        parts.append(clean_brand(nxt.text))
        seen.add(int(nxt.tweet_id))
        cur, n = nxt, n + 1
    return " ".join(parts), len(parts)


def _root_of(tid: int, by_id: dict[int, pd.Series], max_depth: int = 12) -> tuple[int, int]:
    """Return (root customer tweet id, depth). Depth 0 means the tweet is a thread root."""
    depth = 0
    cur = by_id.get(tid)
    while cur is not None and not pd.isna(cur.in_response_to_tweet_id) and depth < max_depth:
        parent = by_id.get(int(cur.in_response_to_tweet_id))
        if parent is None:
            break
        cur = parent
        depth += 1
    return (int(cur.tweet_id) if cur is not None else tid), depth


def build_pairs(full: pd.DataFrame, brand: str = C.BRAND) -> pd.DataFrame:
    """`full` must contain the brand replies, their parent tweets, and the chain up to each root
    (load_brand_frame keeps exactly that; a full dataframe also works)."""
    by_id = {int(r.tweet_id): r for r in full.itertuples(index=False)}

    rows = []
    for r in full[full.author_id == brand].itertuples(index=False):
        if pd.isna(r.in_response_to_tweet_id):
            continue
        if (_part_number(r.text) or 1) > 1:
            continue  # continuation tweets are absorbed into their first part
        cust = by_id.get(int(r.in_response_to_tweet_id))
        if cust is None or not cust.inbound:
            continue
        reply_text, n_parts = _join_continuations(r, by_id, brand)
        root_id, depth = _root_of(int(cust.tweet_id), by_id)
        root = by_id.get(root_id)
        # Turn index counts the customer's own prior tweets that the brand answered in this thread,
        # not raw depth: customers reply to promos, to other customers, and to themselves.
        same_author_root = root is not None and root.author_id == cust.author_id
        rows.append(
            {
                "thread_id": root_id,
                "customer_tweet_id": int(cust.tweet_id),
                "brand_tweet_id": int(r.tweet_id),
                "depth": depth,
                "same_author_root": bool(same_author_root),
                "created_at": cust.created_at,
                "customer_raw": cust.text,
                "root_raw": root.text if same_author_root else cust.text,
                "brand_raw": reply_text,
                "n_parts": n_parts,
            }
        )
    pairs = pd.DataFrame(rows)
    pairs["created_at"] = pd.to_datetime(pairs.created_at, format=DATE_FMT, errors="coerce", utc=True)
    pairs["customer_text"] = pairs.customer_raw.map(clean_customer)
    pairs["root_text"] = pairs.root_raw.map(clean_customer)
    pairs["brand_text"] = pairs.brand_raw  # already cleaned per part in _join_continuations
    # message_text (set after dedupe) is exactly what the agent sees: the customer's tweet, prefixed
    # by their own earlier root tweet when they are continuing a thread the brand has not answered yet.
    pairs["customer_hash"] = pairs.customer_text.map(text_hash)  # dedupe key is the tweet itself, so the thread sample is stable
    # Dedupe exact-duplicate customer texts (retweets, spam) keeping the earliest.
    pairs = pairs.sort_values("created_at").drop_duplicates("customer_hash", keep="first").reset_index(drop=True)
    # turn_index: 0 for the first brand-answered tweet in the thread, ordered by depth then time.
    # (tweet ids in this dataset are renumbered, not chronological, so never rank by id.)
    order = pairs.sort_values(["thread_id", "depth", "created_at"]).groupby("thread_id").cumcount()
    pairs["turn_index"] = order.reindex(pairs.index)
    continuation = (pairs.turn_index == 0) & pairs.same_author_root & (pairs.root_text != pairs.customer_text)
    pairs["message_text"] = np.where(continuation, pairs.root_text + " || " + pairs.customer_text, pairs.customer_text)
    pairs["query_text"] = np.where(pairs.turn_index == 0, pairs.message_text, pairs.root_text + " || " + pairs.customer_text)
    pairs["unhandleable"] = pairs.message_text.map(is_unhandleable)
    return pairs


def subsample_threads(pairs: pd.DataFrame, max_threads: int, seed: int = C.SEED) -> pd.DataFrame:
    # Eligible threads: the brand answered the root tweet or a customer tweet one hop from it.
    # This is the population the golden sets were drawn from; keep it fixed so the seeded draw is stable.
    roots = pairs[pairs.depth <= 1].thread_id.unique()
    rng = np.random.default_rng(seed)
    if len(roots) > max_threads:
        roots = rng.choice(roots, size=max_threads, replace=False)
    return pairs[pairs.thread_id.isin(set(roots))].reset_index(drop=True)


def main(max_threads: int = C.MAX_THREADS) -> pd.DataFrame:
    df = load_brand_frame()
    pairs = build_pairs(df)
    pairs = subsample_threads(pairs, max_threads)
    out = C.PROCESSED / "pairs.parquet"
    pairs.to_parquet(out, index=False)
    n_root = int((pairs.turn_index == 0).sum())
    print(f"pairs={len(pairs)} threads={pairs.thread_id.nunique()} root_pairs={n_root} -> {out}")
    return pairs


if __name__ == "__main__":
    main()
