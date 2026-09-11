import numpy as np
import pandas as pd
import pytest

from hiver_agent import config as C
from hiver_agent.split import assert_no_leakage, assign_splits


def test_split_is_deterministic_and_disjoint():
    ids = np.arange(1000, 2000)
    a, b = assign_splits(ids), assign_splits(ids)
    assert a.equals(b)
    assert set(a.split) == {"corpus", "dev", "golden_pool"}
    assert a.thread_id.is_unique


def test_leakage_assertion_fires():
    pairs = pd.DataFrame({"thread_id": [1, 2, 3], "split": ["corpus", "dev", "golden_pool"]})
    assert_no_leakage(pairs, [3])
    with pytest.raises(AssertionError):
        assert_no_leakage(pairs, [1])


@pytest.mark.skipif(not (C.PROCESSED / "splits.parquet").exists(), reason="needs built data")
def test_golden_candidates_not_in_corpus_or_dev():
    from hiver_agent.split import load_pairs

    cand = C.GOLDEN / "golden_candidates.csv"
    if not cand.exists():
        pytest.skip("no golden candidates yet")
    gold = pd.read_csv(cand)
    assert_no_leakage(load_pairs(), gold.thread_id.astype(int))
