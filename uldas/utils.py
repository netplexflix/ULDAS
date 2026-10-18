#file: uldas/utils.py

import os
import sys
import re
import zlib
import logging

import psutil

logger = logging.getLogger(__name__)


# ── CPU limits ───────────────────────────────────────────────────────────
_cpu_limits_applied = False


def setup_cpu_limits() -> None:
    """Lower process priority and pin to 75 % of cores.

    Applied once per process: ``os.nice`` is relative, so repeating it on
    every scheduled run would keep lowering the priority.
    """
    global _cpu_limits_applied
    if _cpu_limits_applied:
        return
    _cpu_limits_applied = True
    try:
        proc = psutil.Process()

        if hasattr(os, "nice"):
            os.nice(10)
        elif hasattr(proc, "nice"):
            proc.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)

        if hasattr(proc, "cpu_affinity"):
            # 75 % of the cores this process may use — in a container
            # pinned to e.g. cores 6-11 that is a subset of 6-11, not of
            # the host's 0-N.
            allowed = sorted(proc.cpu_affinity())
            cores = allowed[:max(1, int(len(allowed) * 0.75))]
            proc.cpu_affinity(cores)
            logger.info("Limited to %d of %d CPU cores", len(cores), len(allowed))

    except Exception as exc:
        logger.warning("Could not set CPU limits: %s", exc)


def cgroup_cpu_quota() -> "float | None":
    """CPU limit of the container (``docker --cpus 4`` → 4.0), or None."""
    try:                                                    # cgroup v2
        with open("/sys/fs/cgroup/cpu.max", encoding="ascii") as fh:
            quota, period = fh.read().split()[:2]
        return None if quota == "max" else int(quota) / int(period)
    except (OSError, ValueError):
        pass
    try:                                                    # cgroup v1
        with open("/sys/fs/cgroup/cpu/cpu.cfs_quota_us", encoding="ascii") as fh:
            quota = int(fh.read())
        with open("/sys/fs/cgroup/cpu/cpu.cfs_period_us", encoding="ascii") as fh:
            period = int(fh.read())
        return quota / period if quota > 0 and period > 0 else None
    except (OSError, ValueError):
        return None


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


def mp4_language_code(code: str) -> str:
    """3-letter code to write into an MP4 ``mdhd`` box: ISO 639-2/T, as the
    ISO base media format specifies (``deu``, not ``ger``)."""
    from uldas.constants import ISO639_2_B_TO_T
    lang3 = convert_iso639_1_to_2(code.split("-")[0].lower())
    return ISO639_2_B_TO_T.get(lang3, lang3)


def get_language_name(language_code: str) -> str:
    """Human-readable name for a language code."""
    from uldas.constants import LANGUAGE_NAMES
    name = LANGUAGE_NAMES.get(language_code)
    if name:
        return name
    if "-" in language_code:
        prefix = language_code.split("-", 1)[0]
        if prefix in LANGUAGE_NAMES:
            return LANGUAGE_NAMES[prefix]
    return language_code.upper()


# ── Hallucination detection ──────────────────────────────────────────────
# Transcripts at least this long are real speech unless they are a
# degenerate loop: genuine dialogue naturally contains stock phrases,
# repeats and a falling type-token ratio.
_LONG_TRANSCRIPT_WORDS = 40

# Scripts Whisper tends to hallucinate on silent audio, with the
# languages that genuinely use them (the check is skipped for those).
_HALLUCINATION_SCRIPTS = (
    (0x1780, 0x17FF, {"km"}),          # Khmer
    (0x0E00, 0x0E7F, {"th"}),          # Thai
    (0x1000, 0x109F, {"my"}),          # Myanmar
    (0x0980, 0x09FF, {"bn", "as"}),    # Bengali / Assamese
    (0x10A0, 0x10FF, {"ka"}),          # Georgian
)

_COMMON_HALLUCINATIONS = [
    "okay up here we go", "i'm going to go get some water",
    "let's go", "here we go", "okay let's go", "alright let's go",
    "come on let's go", "okay here we go", "let me get some water",
    "i'm going to get some water", "i need to get some water",
    "hold on let me", "wait let me", "okay wait", "hold on",
    "one second", "just a second", "give me a second", "let me just",
]

_GENERIC_HALLUCINATIONS = [
    r"\b(okay|ok|alright|lets|here we go|come on)\b.*\b(go|water|get|just|wait)\b",
    r"\bim (going to|gonna) (go|get)",
    r"\b(hold on|wait|give me|let me) (a |just |)?(second|minute|moment)\b",
]


def _clean_words(text: str) -> str:
    """Lowercase, drop punctuation, collapse whitespace."""
    return " ".join(re.sub(r"[^\w\s]", "", text.lower()).split())


_CLEAN_HALLUCINATIONS = [_clean_words(p) for p in _COMMON_HALLUCINATIONS]


def is_likely_hallucination(text: str, language: str = None,
                            show_details: bool = False) -> bool:
    """Return *True* if *text* looks like a Whisper hallucination on
    non-speech audio rather than a transcript of real speech.

    *language* is the language Whisper detected; it keeps genuine Thai,
    Khmer, etc. from being mistaken for script hallucinations.
    """
    if not text or len(text.strip()) == 0:
        return True

    text = text.strip()

    if len(text) < 3:
        return True

    try:
        raw = text.encode("utf-8")
        compression = len(zlib.compress(raw)) / len(raw)
    except Exception:
        compression = 1.0

    words = text.split()
    if len(words) >= _LONG_TRANSCRIPT_WORDS:
        # Only a looping transcript ("Thank you. Thank you. …")
        # compresses this well; real dialogue stays well above it.
        return compression < 0.2

    # Repetition checks look at letters only, so ellipses ("....") and
    # numbers ("1000000") don't count as repeated content.
    letters = re.sub(r"[\W\d_]+", "", text.lower())
    if len(set(letters)) <= 3 and len(letters) > 10:
        return True
    if re.search(r"(.)\1{4,}", letters):
        return True
    if re.search(r"(.{1,3})\1{3,}", letters):
        return True

    lang = (language or "").lower()
    for lo, hi, script_langs in _HALLUCINATION_SCRIPTS:
        if lang in script_langs:
            continue
        in_script = sum(1 for ch in text if lo <= ord(ch) <= hi)
        if in_script / len(text) > 0.7:
            return True

    clean = _clean_words(text)
    clean_words = clean.split()
    if len(clean_words) > 3 and len(set(clean_words)) / len(clean_words) < 0.2:
        return True
    if len(clean_words) > 5 and len(set(clean_words)) <= 2:
        return True

    if compression < 0.3:
        return True

    # Stock phrases only count when they are (nearly) the whole
    # transcript, not when they occur somewhere inside real dialogue.
    for phrase in _CLEAN_HALLUCINATIONS:
        if phrase in clean and len(clean_words) <= len(phrase.split()) + 3:
            if show_details:
                logger.info("Detected common hallucination phrase: '%s'", phrase)
            return True

    if len(clean) < 50:
        for pat in _GENERIC_HALLUCINATIONS:
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