PY := .venv/bin/python
RUN := PYTHONPATH=src $(PY) -m hiver_agent.cli

.PHONY: eval-v2 eval-fresh judge-cross golden-sample2 lint models setup data prep split labels train index baselines tune build golden-sample agreement filter-check filter-precision eval eval-live judge-sheet judge-agreement demo test

setup:            ## create venv and install pinned deps
	uv sync --frozen --extra dev

data:             ## download + unzip the Kaggle dataset (~177 MB)
	mkdir -p data/raw && cd data/raw && curl -sSL -o twcs.zip "https://www.kaggle.com/api/v1/datasets/download/thoughtvector/customer-support-on-twitter" && unzip -o -q twcs.zip

prep:      ; $(RUN) prep
split:     ; $(RUN) split
labels:    ; $(RUN) labels
train:     ; $(RUN) train
index:     ; $(RUN) index
baselines: ; $(RUN) baselines
tune:      ; $(RUN) tune
build: prep split labels train index baselines tune   ## rebuild every artefact from raw data

golden-sample:    ; $(RUN) golden-sample
agreement:        ; $(RUN) agreement
filter-check:     ; $(RUN) filter-check
filter-precision: ; $(RUN) filter-precision
judge-sheet:      ; $(RUN) judge-sheet
judge-agreement:  ; $(RUN) judge-agreement

eval:             ## headline numbers, offline from cache/llm_cache.jsonl (fails on cache miss)
	$(RUN) eval
eval-live:        ## same, but allowed to call the Gemini API and extend the cache
	HIVER_LIVE=1 $(RUN) eval

models:           ; $(RUN) models

demo:             ## make demo MSG="my downloads keep disappearing"  (live: a new message is never cached)
	HIVER_LIVE=1 $(RUN) demo "$(MSG)"

eval-v2:          ## post-hoc fixes on the same golden set (reported, not the headline)
	$(RUN) eval --version v2 --name results_v2
eval-fresh:       ## v1 vs v2 on the fresh 100-item sample, decisions only (no drafts, no judge)
	$(RUN) eval --labels golden2_labels.csv --version v1 --name fresh_v1 --no-drafts --no-judge
	$(RUN) eval --labels golden2_labels.csv --version v2 --name fresh_v2 --no-drafts --no-judge
judge-cross:      ## re-judge with a second model (default gemini-3.5-flash); prints judge-judge agreement
	HIVER_LIVE=1 $(RUN) judge-cross
golden-sample2:   ; $(RUN) golden-sample2
lint:             ; .venv/bin/ruff check src tests

test:
	$(PY) -m pytest -q
