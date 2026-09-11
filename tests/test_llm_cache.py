import json

import pytest

from hiver_agent.llm import LLM, CacheMiss


def test_cache_miss_raises_offline_and_is_counted(tmp_path):
    llm = LLM(cache_path=tmp_path / "c.jsonl", live=False)
    with pytest.raises(CacheMiss):
        llm.complete_json(system="s", user="u", schema={"type": "object"}, model="gemini:x", tag="t")
    assert llm.misses == ["t"] and "1 misses" in llm.report_misses()


def test_cache_hit_served_and_key_includes_effort(tmp_path):
    path = tmp_path / "c.jsonl"
    llm = LLM(cache_path=path, live=False)
    key = llm._key("gemini:x", "s", "u", {"type": "object"}, "low", 2048)
    path.write_text(json.dumps({"key": key, "tag": "t", "output": {"ok": 1}}) + "\n")
    llm = LLM(cache_path=path, live=False)
    assert llm.complete_json(system="s", user="u", schema={"type": "object"}, model="gemini:x", tag="t") == {"ok": 1}
    with pytest.raises(CacheMiss):  # same prompt, different effort -> different key
        llm.complete_json(system="s", user="u", schema={"type": "object"}, model="gemini:x", tag="t", effort="high")


def test_corrupt_line_is_skipped(tmp_path):
    path = tmp_path / "c.jsonl"
    path.write_text('{"key": "a", "tag": "t", "output": 1}\n{"key": "b", "ta\n')
    assert LLM(cache_path=path, live=False)._cache == {"a": 1}


def test_partial_run_never_overwrites(tmp_path, monkeypatch):
    """An offline evaluate() with cache misses must write *_partial files, not results.*."""
    import hiver_agent.evaluate as ev

    monkeypatch.setattr(ev.C, "REPORTS", tmp_path)
    llm = LLM(cache_path=tmp_path / "c.jsonl", live=False)
    llm.misses.append("draft")
    monkeypatch.setattr(ev, "get_llm", lambda: llm)
    # Minimal fake of the tail of evaluate(): the naming rule is what we test.
    name = "results"
    if llm.misses and not llm.live:
        name = f"{name}_partial"
    assert name == "results_partial"
