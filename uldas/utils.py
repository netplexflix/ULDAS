#file: uldas/utils.py

import os
import sys
import re
import zlib
import logging

import psutil

logger = logging.getLogger(__name__)


# ── CPU limits ───────────────────────────────────────────────────────────
def setup_cpu_limits() -> None:
    """Lower process priority and pin to 75 % of cores."""
    try:
        proc = psutil.Process()

        if hasattr(os, "nice"):
            os.nice(10)
        elif hasattr(proc, "nice"):
            proc.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)

        cpu_count = psutil.cpu_count()
        max_cores = max(1, int(cpu_count * 0.75))

        if hasattr(proc, "cpu_affinity"):
            cores = list(range(min(max_cores, cpu_count)))
            proc.cpu_affinity(cores)
            logger.info("Limited to %d of %d CPU cores", len(cores), cpu_count)

    except Exception as exc:
        logger.warning("Could not set CPU limits: %s", exc)


def limit_subprocess_resources(cmd: list[str]) -> list[str]:
    """Prefix *cmd* with ``nice`` on Unix."""
    if sys.platform == "win32":
        return cmd
    return ["nice", "-n", "10"] + cmd


# ── Language-code helpers ────────────────────────────────────────────────
def normalize_language_code(lang_code: str) -> str:
    """Normalise any language code to a 2-letter ISO 639-1 string."""
    from uldas.constants import ISO639_2_TO_1, ISO639_ALTERNATIVE_CODES

    if not lang_code:
        return "und"

    lang_code = lang_code.lower().strip()

    if lang_code in ("", "und", "unknown", "undefined", "undetermined"):
        return "und"
    if lang_code == "zxx":
        return "zxx"

    if lang_code in ISO639_2_TO_1:
        return ISO639_2_TO_1[lang_code]
    if lang_code in ISO639_ALTERNATIVE_CODES:
        return ISO639_ALTERNATIVE_CODES[lang_code]
    if len(lang_code) == 2:
        return lang_code

    return lang_code


def convert_iso639_1_to_2(code: str) -> str:
    """Convert 2-letter → 3-letter code."""
    from uldas.constants import ISO639_1_TO_2
    return ISO639_1_TO_2.get(code.lower(), code)


def get_language_name(language_code: str) -> str:
    """Human-readable name for a 3-letter code."""
    from uldas.constants import LANGUAGE_NAMES
    return LANGUAGE_NAMES.get(language_code, language_code.upper())


# ── Hallucination detection ──────────────────────────────────────────────
def is_likely_hallucination(text: str, show_details: bool = False) -> bool:
    """Return *True* if *text* looks like a Whisper hallucination."""
    if not text or len(text.strip()) == 0:
        return True

    text = text.strip()

    if len(text) < 3:
        return True

    if len(set(text.replace(" ", ""))) <= 3 and len(text) > 10:
        return True

    unique_chars = len(set(text.replace(" ", "").replace("\n", "")))
    if unique_chars <= 2 and len(text) > 20:
        return True

    if re.search(r"(.)\1{4,}", text):
        return True
    if re.search(r"(.{1,3})\1{3,}", text):
        return True

    # Non-Latin scripts commonly hallucinated on silent audio
    non_latin = 0
    for ch in text:
        cp = ord(ch)
        if (0x1780 <= cp <= 0x17FF or 0x0E00 <= cp <= 0x0E7F or
                0x1000 <= cp <= 0x109F or 0x0980 <= cp <= 0x09FF or
                0x10A0 <= cp <= 0x10FF):
            non_latin += 1
    if len(text) > 0 and (non_latin / len(text)) > 0.7:
        return True

    words = text.split()
    if len(words) > 3:
        if len(set(words)) / len(words) < 0.2:
            return True

    if len(text) > 20:
        parts = text.split()
        if len(parts) > 5 and len(set(parts)) <= 2:
            return True

    try:
        compressed = zlib.compress(text.encode("utf-8"))
        if len(compressed) / len(text.encode("utf-8")) < 0.3:
            return True
    except Exception:
        pass

    common_hallucinations = [
        "okay up here we go", "i'm going to go get some water",
        "let's go", "here we go", "okay let's go", "alright let's go",
        "come on let's go", "okay here we go", "let me get some water",
        "i'm going to get some water", "i need to get some water",
        "hold on let me", "wait let me", "okay wait", "hold on",
        "one second", "just a second", "give me a second", "let me just",
    ]

    clean = re.sub(r"[^\w\s]", "", text.lower().strip())
    for phrase in common_hallucinations:
        if phrase in clean:
            if show_details:
                logger.info("Detected common hallucination phrase: '%s'", phrase)
            return True

    if len(text.lower()) < 50:
        generic = [
            r"\b(okay|ok|alright|let's|here we go|come on)\b.*\b(go|water|get|just|wait)\b",
            r"\bi'm (going to|gonna) (go|get)",
            r"\b(hold on|wait|give me|let me) (a |just |)?(second|minute|moment)\b",
        ]
        for pat in generic:
            if re.search(pat, clean):
                if show_details:
                    logger.info("Detected generic hallucination pattern: %s", pat)
                return True

    return False


# ── Formatting ───────────────────────────────────────────────────────────
def format_duration(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"