#file: uldas/external_subtitles.py
import logging
import os
import re
from pathlib import Path
from typing import Dict, List, Optional

from uldas.constants import (
    EXTERNAL_SUBTITLE_EXTENSIONS,
    ISO639_1_TO_2,
    ISO639_2_TO_1,
    ISO639_ALTERNATIVE_CODES,
)
from uldas.utils import convert_iso639_1_to_2

logger = logging.getLogger(__name__)

_ALL_LANG_CODES: set[str] = set()
_ALL_LANG_CODES.update(ISO639_2_TO_1.keys())       # 3-letter → 2-letter
_ALL_LANG_CODES.update(ISO639_2_TO_1.values())      # 2-letter codes
_ALL_LANG_CODES.update(ISO639_ALTERNATIVE_CODES.keys())  # alternative 3-letter
_ALL_LANG_CODES.update(ISO639_1_TO_2.keys())        # 2-letter → 3-letter
_ALL_LANG_CODES.update({"zxx", "und"})

# Tags that can appear in subtitle filenames alongside language codes
_KNOWN_FLAGS: set[str] = {"forced", "sdh", "hi", "cc", "default", "full"}


def _is_language_code(token: str) -> bool:
    return token.lower() in _ALL_LANG_CODES


def has_language_tag(subtitle_path: Path) -> bool:
    return get_language_tag(subtitle_path) is not None


def get_language_tag(subtitle_path: Path) -> Optional[str]:
    """Return the language code embedded in the subtitle filename, or *None*.

    Recognises patterns like ``movie.en.srt``, ``movie.eng.srt``,
    ``movie.en.sdh.srt``, etc.
    """
    name = subtitle_path.name
    # Split off the final extension
    parts = name.rsplit(".", 2)
    if len(parts) < 3:
        # Only one dot → no language tag  (e.g. "Movie.srt")
        return None

    # parts[-2] is the candidate language tag (or flag)
    candidate = parts[-2].lower()

    # Check if the candidate itself is a language code
    if _is_language_code(candidate):
        return candidate

    # Check if it's a flag like "sdh" or "forced" — if so, look one
    # level deeper for the language code
    if candidate in _KNOWN_FLAGS:
        deeper_parts = name.rsplit(".", 3)
        if len(deeper_parts) >= 4:
            deeper_candidate = deeper_parts[-3].lower()
            if _is_language_code(deeper_candidate):
                return deeper_candidate

    return None


def find_external_subtitles(
    video_path: Path,
    reprocess_all: bool = False,
) -> List[Path]:
    parent = video_path.parent
    video_stem = video_path.stem.lower()

    results: List[Path] = []

    try:
        for entry in parent.iterdir():
            if not entry.is_file():
                continue
            if entry.suffix.lower() not in EXTERNAL_SUBTITLE_EXTENSIONS:
                continue
            entry_stem_lower = entry.stem.lower()
            if not entry_stem_lower.startswith(video_stem):
                continue

            if reprocess_all:
                results.append(entry)
            else:
                if not has_language_tag(entry):
                    results.append(entry)
    except PermissionError as exc:
        logger.warning("Permission denied scanning for external subtitles: %s", exc)
    except Exception as exc:
        logger.error("Error scanning for external subtitles: %s", exc)

    return results


def detect_external_subtitle_language(
    subtitle_path: Path,
    show_details: bool = False,
) -> Optional[Dict]:
    try:
        ext = subtitle_path.suffix.lower()

        # Read the file content
        try:
            with open(subtitle_path, "r", encoding="utf-8", errors="replace") as fh:
                content = fh.read()
        except Exception as exc:
            logger.error("Could not read subtitle file %s: %s", subtitle_path, exc)
            return None

        if not content or len(content.strip()) < 50:
            if show_details:
                logger.info("Subtitle file too short for detection: %s", subtitle_path.name)
            return {"language_code": "und", "confidence": 0.0}

        # Extract text content (strip formatting/timing)
        text_sample = _extract_text_from_subtitle(content, ext)

        if not text_sample or len(text_sample.strip()) < 50:
            if show_details:
                logger.info("Insufficient text extracted from %s", subtitle_path.name)
            return {"language_code": "und", "confidence": 0.0}

        # Clean the text for language detection (remove SDH noise, etc.)
        cleaned_text = _clean_text_for_language_detection(text_sample)

        if show_details:
            original_len = len(text_sample)
            cleaned_len = len(cleaned_text)
            logger.info("Text cleaning: %d → %d chars (%.0f%% kept) for %s",
                        original_len, cleaned_len,
                        (cleaned_len / original_len * 100) if original_len else 0,
                        subtitle_path.name)

        if not cleaned_text or len(cleaned_text.strip()) < 30:
            if show_details:
                logger.info("Insufficient clean text after filtering for %s",
                            subtitle_path.name)
            # The file is mostly SDH/sound effects — still try with what we have
            if text_sample and len(text_sample.strip()) >= 50:
                cleaned_text = text_sample
            else:
                return {"language_code": "und", "confidence": 0.0}

        # Use langdetect
        try:
            import langdetect
            from langdetect import detect_langs
            langdetect.DetectorFactory.seed = 0
            detected = detect_langs(cleaned_text)
            if detected:
                primary = detected[0]
                code = convert_iso639_1_to_2(primary.lang)
                if show_details:
                    logger.info("External subtitle %s: detected %s (conf: %.2f)",
                                subtitle_path.name, code, primary.prob)
                return {
                    "language_code": code,
                    "confidence": primary.prob,
                }
        except ImportError:
            logger.warning("langdetect not available, falling back to character analysis")
        except Exception as exc:
            if show_details:
                logger.debug("langdetect failed for %s: %s", subtitle_path.name, exc)

        # Fallback: character-based detection
        return _detect_language_by_characters(cleaned_text)

    except Exception as exc:
        logger.error("Error detecting language for external subtitle %s: %s",
                      subtitle_path, exc)
        return None


def _clean_text_for_language_detection(text: str) -> str:
    lines = text.split("\n")
    cleaned_lines = []

    for line in lines:
        line = line.strip()
        if not line:
            continue

        # Remove content in square brackets: [MOUSE SQUEAKS], [PANTING], etc.
        line = re.sub(r"\[[^\]]*\]", "", line)

        # Remove content in parentheses that looks like stage directions:
        # (LAUGHS), (SIGHS), (speaking Spanish), etc.
        line = re.sub(r"\([A-Z][A-Z\s,.'!?-]*\)", "", line)

        # Remove music markers: ♪...♪, ?...? (used as music note substitute)
        line = re.sub(r"♪[^♪]*♪", "", line)
        line = re.sub(r"♫[^♫]*♫", "", line)
        # Handle ? used as music note markers (lines starting and ending with ?)
        if line.startswith("?") and line.endswith("?") and line.count("?") >= 2:
            # This is likely a music/song line using ? as ♪
            line = ""

        # Remove speaker labels at start of line: "MAN:", "FINN:", "CANDY PERSON:", etc.
        line = re.sub(r"^[A-Z][A-Z\s.']+:\s*", "", line)

        # Remove HTML-like tags: <i>, </i>, <b>, etc.
        line = re.sub(r"<[^>]+>", "", line)

        # Remove ASS/SSA override tags: {\b1}, {\an8}, etc.
        line = re.sub(r"\{[^}]*\}", "", line)

        # Clean up whitespace
        line = line.strip()

        if not line:
            continue

        # Skip lines that are purely exclamations/interjections/grunts
        # These are non-linguistic and confuse language detection
        exclamation_pattern = re.compile(
            r"^[\s]*("
            r"[AaOoUuEeHh]+[!.]*"          # Aah!, Oh!, Ugh!, Huh?
            r"|[Hh]([aeiou])+[!.]*"          # Ha!, Heh!, Hyah!
            r"|[Ww]h?[oae]+[!.]*"            # Whoa!, Woo!
            r"|[Gg]r+[aeiou]*[!.]*"          # Grr!, Grrr!
            r"|[Bb]l[a-z]*[!.]*"             # Blblbl, Blah
            r"|[Oo]o+[fmph]*[!.]*"           # Oof!, Oomph!
            r"|[Nn]o+[!.]*"                  # No!, Nooo!
            r"|[Yy]e+[aows]*[!.]*"           # Yeah!, Yeow!
            r"|[Rr]a+h*[!.]*"               # Raah!, Rah!
            r"|[Aa]r[sg]h?[!.]*"             # Arsh!, Argh!
            r"|[Pp]hew[!.]*"                 # Phew!
            r"|[Ss]hh+[!.]*"                 # Shh!
            r"|[Mm]m+[!.]*"                  # Mm, Mmm
            r"|[Hh]mm+[!.]*"                 # Hmm
            r"|[Dd]onk[!.]*"                 # Donk!
            r"|[Ww]h[ou]+p+[!.]*"            # Whoop!, Whupppp!
            r"|[Yy]o[!.]*"                   # Yo!
            r")[\s]*$"
        )
        if exclamation_pattern.match(line):
            continue

        # Skip very short lines (1-2 words) that are likely just
        # interjections or sound effects that slipped through
        words = line.split()
        if len(words) <= 1 and len(line) <= 5:
            # Single very short word — likely an interjection
            continue

        cleaned_lines.append(line)

    return " ".join(cleaned_lines)


def _extract_text_from_subtitle(content: str, ext: str) -> str:
    """Extract plain text from subtitle file content, stripping timing/formatting."""
    if ext in (".srt",):
        return _extract_text_from_srt(content)
    elif ext in (".ass", ".ssa"):
        return _extract_text_from_ass(content)
    elif ext in (".vtt",):
        return _extract_text_from_vtt(content)
    elif ext in (".sub",):
        return _extract_text_from_sub(content)
    else:
        # Generic: strip anything that looks like timing or tags
        text = re.sub(r"<[^>]+>", "", content)
        text = re.sub(r"\{[^}]+\}", "", text)
        text = re.sub(r"\d{1,2}:\d{2}:\d{2}[.,]\d{3}", "", text)
        text = re.sub(r"-->", "", text)
        text = re.sub(r"^\d+\s*$", "", text, flags=re.MULTILINE)
        return text.strip()


def _extract_text_from_srt(content: str) -> str:
    """Extract text lines from SRT content."""
    lines = []
    blocks = content.strip().split("\n\n")
    for block in blocks:
        block_lines = block.strip().split("\n")
        if len(block_lines) >= 3:
            # Skip index line and timing line
            text = "\n".join(block_lines[2:])
            text = re.sub(r"<[^>]+>", "", text)
            text = re.sub(r"\{[^}]+\}", "", text)
            lines.append(text.strip())
    return "\n".join(lines)


def _extract_text_from_ass(content: str) -> str:
    """Extract dialogue text from ASS/SSA content."""
    lines = []
    for line in content.split("\n"):
        line = line.strip()
        if line.lower().startswith("dialogue:"):
            # Dialogue: Layer,Start,End,Style,Name,MarginL,MarginR,MarginV,Effect,Text
            parts = line.split(",", 9)
            if len(parts) >= 10:
                text = parts[9]
                # Remove ASS override tags like {\b1}
                text = re.sub(r"\{[^}]*\}", "", text)
                # Replace \N and \n with newline
                text = text.replace("\\N", "\n").replace("\\n", "\n")
                lines.append(text.strip())
    return "\n".join(lines)


def _extract_text_from_vtt(content: str) -> str:
    """Extract text from WebVTT content."""
    lines = []
    in_cue = False
    for line in content.split("\n"):
        line = line.strip()
        if "-->" in line:
            in_cue = True
            continue
        if in_cue:
            if line == "":
                in_cue = False
                continue
            text = re.sub(r"<[^>]+>", "", line)
            lines.append(text.strip())
    return "\n".join(lines)


def _extract_text_from_sub(content: str) -> str:
    """Extract text from MicroDVD .sub format."""
    lines = []
    for line in content.split("\n"):
        line = line.strip()
        # MicroDVD format: {start}{end}text
        match = re.match(r"\{\d+\}\{\d+\}(.*)", line)
        if match:
            text = match.group(1)
            text = text.replace("|", "\n")
            text = re.sub(r"\{[^}]+\}", "", text)
            lines.append(text.strip())
    return "\n".join(lines)


def _detect_language_by_characters(text: str) -> Optional[Dict]:
    """Fallback character-based language detection."""
    if not text or len(text.strip()) < 10:
        return {"language_code": "und", "confidence": 0.0}

    total = len(text.replace(" ", "").replace("\n", ""))
    if total == 0:
        return {"language_code": "und", "confidence": 0.0}

    cyrillic = sum(1 for c in text if 0x0400 <= ord(c) <= 0x04FF)
    arabic = sum(1 for c in text if 0x0600 <= ord(c) <= 0x06FF)
    cjk = sum(1 for c in text if 0x4E00 <= ord(c) <= 0x9FFF)
    latin = sum(1 for c in text if ord(c) < 0x0250)

    cr, ar, cjr, lr = cyrillic / total, arabic / total, cjk / total, latin / total

    if cr > 0.3:
        return {"language_code": "rus", "confidence": min(0.9, 0.5 + cr * 0.5)}
    if ar > 0.3:
        return {"language_code": "ara", "confidence": min(0.9, 0.5 + ar * 0.5)}
    if cjr > 0.3:
        return {"language_code": "chi", "confidence": min(0.85, 0.45 + cjr * 0.5)}
    if lr > 0.7:
        conf = min(0.65, 0.3 + (lr - 0.7) * 0.3 + min(0.2, len(text) / 5000))
        return {"language_code": "eng", "confidence": conf}

    return {"language_code": "und", "confidence": 0.1}


def detect_sdh_in_external_subtitle(subtitle_path: Path) -> bool:
    try:
        ext = subtitle_path.suffix.lower()

        try:
            with open(subtitle_path, "r", encoding="utf-8", errors="replace") as fh:
                content = fh.read()
        except Exception as exc:
            logger.error("Could not read subtitle file for SDH detection %s: %s",
                         subtitle_path, exc)
            return False

        if not content:
            return False

        # Extract raw text lines (before cleaning) to check for SDH patterns
        raw_text = _extract_text_from_subtitle(content, ext)
        if not raw_text:
            return False

        lines = raw_text.split("\n")
        if not lines:
            return False

        sdh_patterns = [
            r"\[[A-Za-z\s]{3,}\]",
            r"\([A-Za-z\s]{3,}\)",
            r"♪[^♪]+♪",
            r"\*[A-Za-z\s]{3,}\*",
        ]
        sdh_keywords = {
            "narrator", "narrating", "speaking", "whispering", "shouting",
            "yelling", "screaming", "music", "playing", "door", "closes",
            "opens", "phone", "ringing", "rings", "footsteps", "sighs",
            "sigh", "laughs", "laugh", "cries", "cry", "crying", "knocking",
            "knock", "barking", "bark", "meowing", "beeping", "beep",
            "gunshot", "explosion", "thunder", "applause", "cheering",
            "clapping", "breathing", "coughing", "snoring", "groaning",
            "grunting", "chatter", "chattering", "murmuring", "rustling",
            "creaking", "dramatic music", "tense music", "suspenseful music",
            "upbeat music", "in distance", "muffled", "echoing", "faintly",
            "squeaks", "squeaking", "screeches", "panting", "gasps",
            "gasping", "sniffs", "sniffles", "whimpers", "groans",
            "grunts", "hoof beats", "chirping", "chirps", "clanging",
            "creaking", "crunch", "flush", "splat", "thud", "thump",
            "whoosh", "crack", "wenk",
        }

        sdh_count = 0
        total_lines = 0
        for line in lines:
            line = line.strip()
            if not line:
                continue
            total_lines += 1
            found = False
            for pat in sdh_patterns:
                for match in re.findall(pat, line, re.IGNORECASE):
                    inner = re.sub(r"[\[\]\(\)\*♪]", "", match).strip().lower()
                    if any(kw in inner for kw in sdh_keywords):
                        found = True
                        break
                if found:
                    break

            # Also check for speaker labels like "MAN:", "FINN:", etc.
            if not found and re.match(r"^[A-Z][A-Z\s.']+:", line):
                found = True

            if found:
                sdh_count += 1

        if total_lines == 0:
            return False

        ratio = sdh_count / total_lines
        if ratio > 0.10:
            return True

        # Also check for common SDH phrases in the full text
        full_text = " ".join(line.strip().lower() for line in lines if line.strip())
        phrase_pats = [
            r"\bnarrator\b", r"\bspeaking\b", r"\bwhispering\b", r"\bshouting\b",
            r"\bmusic playing\b", r"\bdoor closes\b", r"\bphone ringing\b",
            r"\bfootsteps\b", r"\bsighs\b", r"\blaughs\b", r"\bcries\b",
            r"\bin the distance\b", r"\bmuffled\b", r"\bechoing\b",
            r"\bdramatic music\b", r"\btense music\b",
        ]
        if sum(1 for p in phrase_pats if re.search(p, full_text)) >= 3:
            return True

        return False

    except Exception as exc:
        logger.error("Error detecting SDH in external subtitle %s: %s",
                      subtitle_path, exc)
        return False


def rename_subtitle_with_language(
    subtitle_path: Path,
    language_code: str,
    is_sdh: bool = False,
    dry_run: bool = False,
    show_details: bool = False,
) -> Optional[Path]:
    try:
        # Convert 3-letter code to 2-letter for filename tagging
        from uldas.constants import ISO639_2_TO_1
        short_code = ISO639_2_TO_1.get(language_code, language_code)

        name = subtitle_path.name
        ext = subtitle_path.suffix  # e.g. ".srt"
        stem = subtitle_path.stem   # e.g. "Movie.Name" or "Movie.Name.en"

        # Check if there's already a language tag to replace
        parts = stem.rsplit(".", 1)
        if len(parts) == 2 and _is_language_code(parts[1]):
            # Replace existing language tag
            base_stem = parts[0]
        else:
            base_stem = stem

        # Build new filename with language and optional SDH tag
        if is_sdh:
            new_name = f"{base_stem}.{short_code}.sdh{ext}"
        else:
            new_name = f"{base_stem}.{short_code}{ext}"

        new_path = subtitle_path.parent / new_name

        if new_path == subtitle_path:
            if show_details:
                logger.info("Subtitle already correctly tagged: %s", subtitle_path.name)
            return subtitle_path

        if new_path.exists():
            if show_details:
                logger.warning("Target file already exists, skipping: %s", new_name)
            return None

        if dry_run:
            print(f"[DRY RUN] Would rename {subtitle_path.name} → {new_name}")
            return new_path

        subtitle_path.rename(new_path)

        if show_details:
            logger.info("Renamed: %s → %s", subtitle_path.name, new_name)
        else:
            print(f"  Renamed: {subtitle_path.name} → {new_name}")

        return new_path

    except Exception as exc:
        logger.error("Error renaming subtitle %s: %s", subtitle_path, exc)
        return None