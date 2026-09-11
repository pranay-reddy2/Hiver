# SpotifyCares support agent

An AI support agent for **@SpotifyCares**, built on the Kaggle *Customer Support on Twitter*
dataset. For each incoming customer tweet it:

1. **Classifies** the intent (7 intents + `other` + `unhandleable`, defined from the data).
2. **Drafts** a public reply grounded only in how @SpotifyCares historically resolved similar tweets.
3. **Decides** auto-handle vs. escalate, with a stated reason.

The report, results, failure analysis and decision log are in [`reports/`](reports/).

## Reproduce the headline numbers (offline, ~1 minute)

```bash
cp .env.example .env   # add GEMINI_API_KEY only if you plan to run `make eval-live`
make setup          # uv venv + pinned deps from uv.lock (or: python -m venv .venv && .venv/bin/pip install -e ".[dev]")
make data           # ~177 MB Kaggle zip, no login required
make build          # prep -> split -> weak labels -> train -> index -> baselines -> tune  (~3 min, CPU)
make eval           # golden-set metrics from the committed LLM cache; no API key needed
```

`make eval` never calls an API. If a prompt is missing from `cache/llm_cache.jsonl` it exits with
code 2 and prints how many calls are missing. `make eval-live` (with `GEMINI_API_KEY` set) makes
those calls and appends them to the cache.

Measured on a clean checkout: see the "Reproduction time" line in `reports/report.md`.

## Try it

```bash
make demo MSG="my downloaded songs keep disappearing from my phone"   # runs live: a new message is never in the cache
```

## Other targets

| target | what |
|---|---|
| `make eval-v2` | the post-hoc escalation / unhandleable fixes on the same golden set (reported, never the headline) |
| `make eval-fresh` | v1 vs v2 on a fresh 100-item sample (`golden2_labels.csv`), decisions only |
| `make judge-cross` | second judge from another model family (needs `ANTHROPIC_API_KEY`); prints judge-judge agreement |
| `make agreement` / `make judge-agreement` / `make filter-precision` | human-agreement numbers once the CSVs under `data/golden/` are filled |
| `make test` / `make lint` | pytest (15 tests) and ruff |

## Layout

| path | what |
|---|---|
| `src/hiver_agent/prep.py` | raw CSV → customer/brand pairs; joins numbered multi-tweet replies, strips signatures |
| `src/hiver_agent/split.py` | thread-id split: corpus 75% / dev 10% / golden pool 15%; leakage assertion |
| `src/hiver_agent/filters.py` | regex filter deciding which replies count as resolutions |
| `src/hiver_agent/labels.py` | weak intent labels (keyword bootstrap, LLM refinement) for the corpus and dev |
| `src/hiver_agent/classifier.py` | MiniLM embeddings + logistic regression, C chosen on dev |
| `src/hiver_agent/retrieval.py` | cosine retrieval over resolution-bearing replies + per-intent question bank |
| `src/hiver_agent/reply.py` | RAG drafter with a hard "only cite retrieved links" rule and an automatic link check |
| `src/hiver_agent/escalation.py` | six weighted signals → score → decision + reason |
| `src/hiver_agent/baselines.py` | trivial (majority / template / never escalate) and simple (TF-IDF+LR / nearest neighbour / keywords) |
| `src/hiver_agent/judge.py` | LLM judge on the anchored rubric in `configs/rubric.md`; human-agreement stats |
| `src/hiver_agent/evaluate.py` | the harness; writes `reports/results.{json,md}` and per-system prediction CSVs |
| `src/hiver_agent/tune.py` | escalation / similarity thresholds swept on dev (proxy target, see report) |
| `src/hiver_agent/golden.py` | golden-set sampling and annotator agreement |
| `data/golden/` | golden candidates, labels, second-annotator sheet, reply ratings, labelling guide |
| `configs/` | intent taxonomy, URL catalogue, rubric |
| `cache/llm_cache.jsonl` | every LLM call, keyed by sha256(model, prompts, schema, effort, max_tokens), with token usage and wall seconds |
| `models/params.json` | the dev-tuned thresholds the report quotes (committed; other model artefacts are rebuilt by `make build`) |

## Models

Generator: `gemini-3.1-pro-preview`. Weak labels and judge: `gemini-2.5-flash` (a different, smaller model
so the judge is not grading its own writing). Embeddings: `all-MiniLM-L6-v2`, run locally.
Model strings are `<provider>:<model>`; the Anthropic backend is also wired, so
`JUDGE_MODEL=anthropic:claude-haiku-4-5 make eval-live` gives a judge from a different family.
`make models` lists what your key can reach. Gemini free-tier rate limits are handled with backoff,
but the ~1.4k calls for a full run may exceed the free daily quota; the cache makes reruns free.

## Borrowed

- Dataset: thoughtvector/customer-support-on-twitter (Kaggle).
- `sentence-transformers/all-MiniLM-L6-v2` for embeddings; scikit-learn for LR and TF-IDF.
- `google-genai` (and optionally the Anthropic SDK) for LLM calls. Everything else was written for this assignment with an AI
  coding assistant (Claude Code); all code was reviewed and is explained in the report.
