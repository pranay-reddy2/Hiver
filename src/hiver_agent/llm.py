"""Single LLM entry point with a content-addressed JSONL cache.

Model strings carry a provider prefix ("gemini:gemini-3.8-flash"); only the Gemini backend is
implemented. Every call is keyed by sha256(model, system, user, schema, effort, max_tokens). Records also carry usage (tokens) and wall seconds so cost and latency can be reported.
`make eval` runs with HIVER_LIVE unset: cache hits are served, misses are counted and raised at the
end so a reviewer never silently burns tokens. `make eval-live` sets HIVER_LIVE=1.
"""
from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from pathlib import Path
from typing import Any

from . import config as C

CACHE_PATH = C.CACHE / "llm_cache.jsonl"
KEY_VERSION = 2  # v1 keys omitted effort and max_tokens


class CacheMiss(RuntimeError):
    pass


def _split(model: str) -> tuple[str, str]:
    if ":" in model:
        provider, name = model.split(":", 1)
        return provider, name
    return "gemini", model


class LLM:
    def __init__(self, cache_path: Path = CACHE_PATH, live: bool | None = None):
        self.cache_path = cache_path
        self.live = bool(int(os.environ.get("HIVER_LIVE", "0"))) if live is None else live
        self._cache: dict[str, Any] = {}
        self._meta: dict[str, dict] = {}
        self.misses: list[str] = []
        self.calls = 0
        self._clients: dict[str, Any] = {}
        self._lock = threading.Lock()
        if cache_path.exists():
            with open(cache_path, encoding="utf-8") as fh:
                for line in fh:
                    if not line.strip():
                        continue
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        continue  # a partial line from an interrupted write; the call is simply redone
                    self._cache[rec["key"]] = rec["output"]
                    self._meta[rec["key"]] = {k: rec.get(k) for k in ("tag", "model", "usage", "seconds")}

    # ---- helpers -------------------------------------------------------------
    @staticmethod
    def _key(model: str, system: str, user: str, schema: dict | None, effort: str, max_tokens: int) -> str:
        payload = json.dumps({"v": KEY_VERSION, "m": model, "s": system, "u": user, "j": schema, "e": effort, "t": max_tokens}, sort_keys=True)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _append(self, key: str, tag: str, model: str, output: Any, usage: dict | None, seconds: float) -> None:
        rec = {"key": key, "tag": tag, "model": model, "usage": usage, "seconds": round(seconds, 3), "output": output}
        with self._lock:
            self._cache[key] = output
            self._meta[key] = {k: rec[k] for k in ("tag", "model", "usage", "seconds")}
            with open(self.cache_path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

    def _client(self, provider: str):
        with self._lock:
            if provider not in self._clients:
                if provider != "gemini":
                    raise ValueError(f"unknown provider {provider}; only gemini is implemented")
                from google import genai

                self._clients[provider] = genai.Client()  # reads GEMINI_API_KEY / GOOGLE_API_KEY
            return self._clients[provider]

    # ---- backends --------------------------------------------------------------
    def _gemini(self, name: str, system: str, user: str, schema: dict, effort: str, max_tokens: int) -> tuple[dict, dict]:
        """Thinking tokens count against max_output_tokens on Gemini, so low-effort calls disable
        thinking (2.x flash) and every call gets a generous cap; truncation is retried once larger."""
        import httpx
        from google.genai import errors, types

        client = self._client("gemini")
        disable_thinking = effort == "low" and "pro" not in name and "gemini-3" not in name
        cap = max(max_tokens, 8192)
        delay = 5.0
        for attempt in range(6):
            cfg = types.GenerateContentConfig(
                system_instruction=system,
                response_mime_type="application/json",
                response_json_schema=schema,
                temperature=0.0,
                max_output_tokens=cap,
                thinking_config=types.ThinkingConfig(thinking_budget=0) if disable_thinking else None,
            )
            try:
                resp = client.models.generate_content(model=name, contents=user, config=cfg)
            except (errors.APIError, httpx.HTTPError) as e:  # rate limits, 5xx, dropped connections
                retryable = isinstance(e, httpx.HTTPError) or e.code in (429, 500, 503)
                if retryable and attempt < 5:
                    time.sleep(delay)
                    delay = min(delay * 2, 60)
                    continue
                raise
            finish = str(resp.candidates[0].finish_reason) if resp.candidates else "NONE"
            um = resp.usage_metadata
            usage = {"input": um.prompt_token_count, "output": um.candidates_token_count, "thinking": um.thoughts_token_count} if um else None
            text = resp.text or ""
            try:
                return json.loads(text), usage
            except json.JSONDecodeError:
                if "MAX_TOKENS" in finish and attempt < 5:
                    cap *= 2
                    continue
                raise RuntimeError(f"gemini returned non-JSON (finish={finish}): {text[:300]!r}") from None
        raise RuntimeError("unreachable")

    # ---- public --------------------------------------------------------------
    def complete_json(self, *, system: str, user: str, schema: dict, model: str, tag: str, effort: str = "low", max_tokens: int = 2048) -> dict:
        key = self._key(model, system, user, schema, effort, max_tokens)
        if key in self._cache:
            return self._cache[key]
        if not self.live:
            self.misses.append(tag)
            raise CacheMiss(f"cache miss for tag={tag}; run with HIVER_LIVE=1 to call the API")
        self.calls += 1
        provider, name = _split(model)
        t0 = time.time()
        if provider != "gemini":
            raise ValueError(f"unknown provider {provider}; only gemini is implemented")
        out, usage = self._gemini(name, system, user, schema, effort, max_tokens)
        self._append(key, tag, model, out, usage, time.time() - t0)
        return out

    def meta_for(self, *, system: str, user: str, schema: dict, model: str, effort: str = "low", max_tokens: int = 2048) -> dict | None:
        """Usage and latency recorded for a call, if it was ever made live."""
        return self._meta.get(self._key(model, system, user, schema, effort, max_tokens))

    def report_misses(self) -> str:
        if not self.misses:
            return "llm cache: 0 misses"
        from collections import Counter

        counts = Counter(self.misses)
        detail = ", ".join(f"{k}={v}" for k, v in counts.most_common())
        return f"llm cache: {len(self.misses)} misses ({detail}). Run with HIVER_LIVE=1 and GEMINI_API_KEY set."

    def fail_on_misses(self) -> None:
        if self.misses and not self.live:
            raise SystemExit(self.report_misses())


_default: LLM | None = None


def get_llm() -> LLM:
    global _default
    if _default is None:
        _default = LLM()
    return _default


def list_models(provider: str = "gemini") -> list[str]:
    return [m.name for m in get_llm()._client(provider).models.list()]
