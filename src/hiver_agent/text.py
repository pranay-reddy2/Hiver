"""Text cleaning shared by every stage. Keep it boring and deterministic."""
from __future__ import annotations

import hashlib
import html
import re

HANDLE = re.compile(r"@\w+")
URL = re.compile(r"https?://t\.co/(\w+)")
SIGNATURE = re.compile(r"\s*/[A-Z]{1,3}\s*$")          # "... let us know /JK"
NUMBERED = re.compile(r"^\s*(\d+)\s*[:.)]\s*")           # "1: Thanks! ..."
WS = re.compile(r"\s+")
NON_ENGLISH_HINT = re.compile(
    r"\b(que|não|nao|por favor|para|el|la|los|las|und|ich|nicht|het|een|niet|ang|ako|"
    r"saya|tidak|yang|के|है|मैं)\b",
    re.I,
)


def strip_handles(text: str) -> str:
    return HANDLE.sub(" ", text)


def normalise_urls(text: str) -> str:
    """Replace every t.co link with a stable token `[link:ID]` so it survives cleaning."""
    return URL.sub(lambda m: f"[link:{m.group(1)}]", text)


def clean_customer(text: str) -> str:
    t = html.unescape(text)
    t = HANDLE.sub(" ", t)
    t = URL.sub(" ", t)
    t = re.sub(r"https?://\S+", " ", t)
    return WS.sub(" ", t).strip()


def clean_brand(text: str) -> str:
    t = html.unescape(text)
    t = HANDLE.sub(" ", t)
    t = NUMBERED.sub("", t)
    t = SIGNATURE.sub("", t)
    t = normalise_urls(t)
    return WS.sub(" ", t).strip()


def links_in(text: str) -> list[str]:
    return re.findall(r"\[link:(\w+)\]", text) + URL.findall(text)


def word_count(text: str) -> int:
    return len(re.findall(r"[A-Za-z']{2,}", text))


PROMO_COPY = re.compile(
    r"(premium gives you|get 3 months|3 months of premium|just (99p|\$0\.99|0,99)|try it free|"
    r"student\? get|/mo\b|per maand|we're investigating some payment issues)",
    re.I,
)
BRAND_SIGNATURE = re.compile(r"\s/[A-Z]{2,3}\s*$|^\d+:\s")

_detector = None


def _lang_detector():
    global _detector
    if _detector is None:
        from lingua import Language, LanguageDetectorBuilder

        langs = [Language.ENGLISH, Language.DUTCH, Language.SPANISH, Language.PORTUGUESE, Language.GERMAN, Language.FRENCH,
                 Language.INDONESIAN, Language.TAGALOG, Language.ITALIAN, Language.TURKISH]
        _detector = LanguageDetectorBuilder.from_languages(*langs).with_low_accuracy_mode().build()
    return _detector


def is_non_english(clean_text: str) -> bool:
    """Lingua on the cleaned text; short texts need high confidence so 'where is lemonade' stays English."""
    if word_count(clean_text) < 4:
        return False
    from lingua import Language

    det = _lang_detector()
    top = det.compute_language_confidence_values(clean_text)[0]
    en = det.compute_language_confidence(clean_text, Language.ENGLISH)
    return top.language != Language.ENGLISH and top.value >= 0.8 and en < 0.2


def is_unhandleable(clean_text: str, version: str = "v1") -> bool:
    """Bare mention, image-only, non-English, or too short to act on.

    v1 (headline): word-list language hint. v2: lingua detector plus promo-copy and brand-signature
    rules, added after the golden sample showed the noise is mostly other people's copy.
    """
    if word_count(clean_text) < 3:
        return True
    letters = re.findall(r"[A-Za-z]", clean_text)
    if len(letters) < 0.5 * len(re.sub(r"\s", "", clean_text)):
        return True
    if version == "v2":
        return is_non_english(clean_text) or bool(PROMO_COPY.search(clean_text)) or bool(BRAND_SIGNATURE.search(clean_text))
    hits = len(NON_ENGLISH_HINT.findall(clean_text))
    return hits >= 2 and hits >= 0.15 * max(1, word_count(clean_text))


def text_hash(text: str) -> str:
    return hashlib.sha1(text.lower().encode("utf-8")).hexdigest()[:16]
