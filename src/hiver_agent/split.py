"""Stage 2: deterministic split by thread id. Golden and dev never touch the corpus."""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C


def assign_splits(thread_ids: np.ndarray, seed: int = C.SEED) -> pd.DataFrame:
    ids = np.array(sorted(set(int(t) for t in thread_ids)))
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(ids))
    n = len(ids)
    n_dev = int(n * C.SPLIT_FRACTIONS["dev"])
    n_gold = int(n * C.SPLIT_FRACTIONS["golden_pool"])
    split = np.full(n, "corpus", dtype=object)
    split[perm[:n_dev]] = "dev"
    split[perm[n_dev : n_dev + n_gold]] = "golden_pool"
    return pd.DataFrame({"thread_id": ids, "split": split})


def load_splits() -> pd.DataFrame:
    return pd.read_parquet(C.PROCESSED / "splits.parquet")


def load_pairs(split: str | None = None) -> pd.DataFrame:
    pairs = pd.read_parquet(C.PROCESSED / "pairs.parquet")
    splits = load_splits()
    pairs = pairs.merge(splits, on="thread_id", how="left")
    if split:
        pairs = pairs[pairs.split == split]
    return pairs.reset_index(drop=True)


def assert_no_leakage(pairs: pd.DataFrame, golden_thread_ids) -> None:
    gold = set(int(t) for t in golden_thread_ids)
    bad = pairs[pairs.thread_id.isin(gold) & (pairs.split != "golden_pool")]
    if len(bad):
        raise AssertionError(f"{len(bad)} golden threads leak into {sorted(bad.split.unique())}")


def main() -> None:
    pairs = pd.read_parquet(C.PROCESSED / "pairs.parquet")
    splits = assign_splits(pairs.thread_id.values)
    splits.to_parquet(C.PROCESSED / "splits.parquet", index=False)
    print(splits.split.value_counts().to_dict())


if __name__ == "__main__":
    main()
