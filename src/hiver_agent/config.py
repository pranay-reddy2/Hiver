from __future__ import annotations

import os
from pathlib import Path

import yaml
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env")  # real env vars take precedence over .env
DATA = ROOT / "data"
RAW = DATA / "raw" / "twcs" / "twcs.csv"
PROCESSED = DATA / "processed"
GOLDEN = DATA / "golden"
CACHE = ROOT / "cache"
MODELS = ROOT / "models"
REPORTS = ROOT / "reports"
CONFIGS = ROOT / "configs"

BRAND = "SpotifyCares"
SEED = 20240910
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Model strings are "<provider>:<model>". Generator and judge are different models so the judge
# is not grading its own writing. Set JUDGE_MODEL=anthropic:claude-haiku-4-5 for a different family.
GEN_MODEL = os.environ.get("GEN_MODEL", "gemini:gemini-3.8-flash")
LABEL_MODEL = os.environ.get("LABEL_MODEL", "gemini:gemini-2.5-flash")
JUDGE_MODEL = os.environ.get("JUDGE_MODEL", "gemini:gemini-2.5-flash")

# Split fractions by thread id. Golden examples are sampled from the golden pool.
SPLIT_FRACTIONS = {"corpus": 0.75, "dev": 0.10, "golden_pool": 0.15}

# Max threads pulled from the raw file. Full SpotifyCares is ~23k root threads.
MAX_THREADS = int(os.environ.get("MAX_THREADS", "12000"))


def _load_yaml(name: str) -> dict:
    with open(CONFIGS / name, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


INTENTS_CFG = _load_yaml("intents.yaml")
INTENTS: list[str] = list(INTENTS_CFG["intents"].keys())
CLASSIFIABLE_INTENTS: list[str] = [i for i in INTENTS if i != "unhandleable"]
SENSITIVE_INTENTS: set[str] = set(INTENTS_CFG["sensitive_intents"])
URL_CATALOGUE: dict[str, str] = _load_yaml("url_catalogue.yaml")["catalogue"]

for _p in (PROCESSED, GOLDEN, CACHE, MODELS, REPORTS):
    _p.mkdir(parents=True, exist_ok=True)
