#file: uldas/subtitles.py

import os
import re
import sys
import shutil
import subprocess
import tempfile
import logging
from pathlib import Path
from typing import Optional, Dict, List, Tuple

from uldas.utils import (
    limit_subprocess_resources,
    convert_iso639_1_to_2,
    get_language_name,
)

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
#  Extraction
# ═══════════════════════════════════════════════════════════════════════════

def extract_subtitle_track(
    ffmpeg: str,
    file_path: Path,
    subtitle_track_index: int,
    stream_index: int,
    get_mkv_info_fn,
    show_details: bool = False,
) -> Optional[Path]:
    """Extract a subtitle track to ``.srt`` (text) or ``.sup`` (image)."""
    try:
        info = get_mkv_info_fn(file_path)
        subtitle_codec = None

        if "streams" in info:
            count = 0
            for stream in info["streams"]:
                if stream.get("codec_type") == "subtitle":
                    if count == subtitle_track_index:
                        subtitle_codec = stream.get("codec_name", "").lower()
                        break
                    count += 1

        image_codecs = {
            "hdmv_pgs_subtitle", "pgs", "dvdsub", "dvbsub",
            "dvd_subtitle", "s_hdmv/pgs",
        }
        is_image = (
            subtitle_codec in image_codecs
            or "pgs" in (subtitle_codec or "")
            or "hdmv" in (subtitle_codec or "")
            or "dvd" in (subtitle_codec or "")
        )

        if is_image:
            suffix, copy_codec = ".sup", "copy"
        else:
            suffix, copy_codec = ".srt", "srt"

        tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
        tmp_path = Path(tmp.name)
        tmp.close()

        mappings = [
            f"0:s:{subtitle_track_index}",
            f"0:{stream_index}",
            f"s:{subtitle_track_index}",
        ]

        for mapping in mappings:
            try:
                cmd = [
                    ffmpeg, "-y", "-v", "warning",
                    "-i", str(file_path),
                    "-map", mapping,
                    "-c:s", copy_codec,
                    str(tmp_path),
                ]
                limited = limit_subprocess_resources(cmd)
                subprocess.run(limited, check=True, capture_output=True,
                               text=True, encoding="utf-8", errors="replace")
                if tmp_path.exists() and tmp_path.stat().st_size > 100:
                    return tmp_path
                if tmp_path.exists():
                    tmp_path.unlink()
            except subprocess.CalledProcessError:
                if tmp_path.exists():
                    tmp_path.unlink()

        logger.error("All subtitle extraction attempts failed")
        return None
    except Exception as exc:
        logger.error("Unexpected error during subtitle extraction: %s", exc)
        return None


# ═══════════════════════════════════════════════════════════════════════════
#  SRT parsing
# ═══════════════════════════════════════════════════════════════════════════

def parse_srt_file(srt_path: Path) -> List[Dict]:
    try:
        with open(srt_path, "r", encoding="utf-8", errors="replace") as fh:
            content = fh.read()
        blocks = content.strip().split("\n\n")
        subs: List[Dict] = []
        for block in blocks:
            lines = block.strip().split("\n")
            if len(lines) >= 3:
                try:
                    idx = int(lines[0].strip())
                    timing = lines[1].strip()
                    text = "\n".join(lines[2:])
                    if " --> " in timing:
                        start, end = timing.split(" --> ")
                        subs.append({
                            "index": idx,
                            "start": start.strip(),
                            "end": end.strip(),
                            "text": text.strip(),
                        })
                except (ValueError, IndexError):
                    continue
        return subs
    except Exception as exc:
        logger.error("Error parsing SRT file: %s", exc)
        return []


def parse_srt_time(time_str: str) -> float:
    time_str = time_str.replace(",", ".")
    parts = time_str.split(":")
    return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])


def get_subtitle_text_sample(subtitles: List[Dict], max_chars: int = 5000) -> str:
    parts: list[str] = []
    total = 0
    indices = [0]
    if len(subtitles) > 10:
        indices.append(len(subtitles) // 2)
    if len(subtitles) > 20:
        indices.append(len(subtitles) - 1)

    for idx in indices:
        start = max(0, idx - 5)
        end = min(len(subtitles), idx + 5)
        for sub in subtitles[start:end]:
            text = re.sub(r"<[^>]+>", "", sub["text"])
            text = re.sub(r"\{[^}]+\}", "", text)
            parts.append(text)
            total += len(text)
            if total >= max_chars:
                break
        if total >= max_chars:
            break
    return " ".join(parts)


# ═══════════════════════════════════════════════════════════════════════════
#  Language detection
# ═══════════════════════════════════════════════════════════════════════════

def detect_subtitle_language(
    subtitle_path: Path,
    file_path: Path = None,
    subtitle_track_index: int = None,
    stream_index: int = None,
    ffmpeg: str = None,
    ffprobe: str = None,
    show_details: bool = False,
) -> Optional[Dict]:
    """Detect the language of a subtitle file (text or image-based)."""
    try:
        if subtitle_path.suffix.lower() == ".sup":
            if show_details:
                logger.warning("Image-based subtitle detected (PGS/SUP)")
            try:
                if file_path and subtitle_track_index is not None and stream_index is not None:
                    return _extract_pgs_images_and_ocr(
                        ffmpeg, file_path, subtitle_track_index, stream_index, show_details,
                    )
                return None
            except ImportError:
                return None

        subtitles = parse_srt_file(subtitle_path)
        if not subtitles:
            return None

        text_sample = get_subtitle_text_sample(subtitles)
        if not text_sample or len(text_sample.strip()) < 50:
            return {"language_code": "und", "confidence": 0.0,
                    "subtitle_count": len(subtitles)}

        try:
            import langdetect
            from langdetect import detect_langs
            langdetect.DetectorFactory.seed = 0
            detected = detect_langs(text_sample)
            if detected:
                primary = detected[0]
                code = convert_iso639_1_to_2(primary.lang)
                return {
                    "language_code": code,
                    "confidence": primary.prob,
                    "subtitle_count": len(subtitles),
                }
            return _detect_language_by_characters(text_sample, len(subtitles))
        except ImportError:
            return _detect_language_by_characters(text_sample, len(subtitles))
        except Exception:
            return _detect_language_by_characters(text_sample, len(subtitles))

    except Exception as exc:
        logger.error("Error detecting subtitle language: %s", exc)
        return None


def _detect_language_by_characters(text: str, subtitle_count: int) -> Optional[Dict]:
    if not text or len(text.strip()) < 10:
        return {"language_code": "und", "confidence": 0.0,
                "subtitle_count": subtitle_count}

    total = len(text.replace(" ", "").replace("\n", ""))
    if total == 0:
        return {"language_code": "und", "confidence": 0.0,
                "subtitle_count": subtitle_count}

    cyrillic = sum(1 for c in text if 0x0400 <= ord(c) <= 0x04FF)
    arabic = sum(1 for c in text if 0x0600 <= ord(c) <= 0x06FF)
    cjk = sum(1 for c in text if 0x4E00 <= ord(c) <= 0x9FFF)
    latin = sum(1 for c in text if ord(c) < 0x0250)

    cr, ar, cjr, lr = cyrillic / total, arabic / total, cjk / total, latin / total

    if cr > 0.3:
        return {"language_code": "rus", "confidence": min(0.9, 0.5 + cr * 0.5),
                "subtitle_count": subtitle_count}
    if ar > 0.3:
        return {"language_code": "ara", "confidence": min(0.9, 0.5 + ar * 0.5),
                "subtitle_count": subtitle_count}
    if cjr > 0.3:
        return {"language_code": "chi", "confidence": min(0.85, 0.45 + cjr * 0.5),
                "subtitle_count": subtitle_count}
    if lr > 0.7:
        conf = min(0.65, 0.3 + (lr - 0.7) * 0.3 + min(0.2, len(text) / 5000))
        return {"language_code": "eng", "confidence": conf,
                "subtitle_count": subtitle_count}

    return {"language_code": "und", "confidence": 0.1,
            "subtitle_count": subtitle_count}


# ═══════════════════════════════════════════════════════════════════════════
#  PGS / OCR
# ═══════════════════════════════════════════════════════════════════════════

def _extract_pgs_images_and_ocr(
    ffmpeg: str,
    file_path: Path,
    subtitle_track_index: int,
    stream_index: int,
    show_details: bool = False,
) -> Optional[Dict]:
    """Extract PGS subtitle images and OCR them for language detection."""
    try:
        import pytesseract
        from PIL import Image, ImageEnhance
    except ImportError as exc:
        logger.error("Required library not installed: %s", exc)
        return None

    if sys.platform == "win32":
        for p in (
            r"C:\Program Files\Tesseract-OCR\tesseract.exe",
            r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
            rf"C:\Users\{os.getenv('USERNAME')}\AppData\Local\Programs\Tesseract-OCR\tesseract.exe",
        ):
            if os.path.exists(p):
                pytesseract.pytesseract.tesseract_cmd = p
                break

    try:
        pytesseract.get_tesseract_version()
    except Exception:
        logger.error("Tesseract OCR not found")
        return None

    tmp_dir = tempfile.mkdtemp()
    try:
        # Try several extraction methods
        image_files: list[Path] = []
        for method_cmd in _pgs_extraction_commands(ffmpeg, file_path, subtitle_track_index, tmp_dir):
            try:
                limited = limit_subprocess_resources(method_cmd)
                subprocess.run(limited, capture_output=True, text=True,
                               encoding="utf-8", errors="replace", timeout=120)
            except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
                pass
            image_files = list(Path(tmp_dir).glob("sub_*.png"))
            if image_files:
                break

        if not image_files:
            logger.warning("No subtitle images extracted")
            return None

        ocr_texts: list[str] = []
        for img_file in image_files[:30]:
            try:
                img = Image.open(img_file).convert("L")
                img = ImageEnhance.Contrast(img).enhance(2.0)
                text = pytesseract.image_to_string(img, config="--psm 6")
                if text.strip() and len(text.strip()) > 2:
                    ocr_texts.append(text.strip())
            except Exception:
                continue

        if not ocr_texts:
            return None

        combined = " ".join(ocr_texts)
        if len(combined.strip()) < 50:
            return None

        try:
            import langdetect
            from langdetect import detect_langs
            detected = detect_langs(combined)
            if detected:
                primary = detected[0]
                code = convert_iso639_1_to_2(primary.lang)
                return {
                    "language_code": code,
                    "confidence": primary.prob * 0.75,
                    "subtitle_count": len(ocr_texts),
                }
        except ImportError:
            return _detect_language_by_characters(combined, len(ocr_texts))
        except Exception:
            return _detect_language_by_characters(combined, len(ocr_texts))

        return None
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def _pgs_extraction_commands(ffmpeg, file_path, sub_idx, tmp_dir):
    """Yield ffmpeg commands to try for PGS image extraction."""
    yield [
        ffmpeg, "-y", "-v", "warning",
        "-i", str(file_path),
        "-filter_complex", f"[0:s:{sub_idx}]scale=iw:ih[sub]",
        "-map", "[sub]", "-frames:v", "50", "-vsync", "0",
        f"{tmp_dir}/sub_%04d.png",
    ]
    # Method 2: extract .sup then convert
    sup = os.path.join(tmp_dir, "subtitles.sup")
    yield [
        ffmpeg, "-y", "-v", "warning",
        "-i", str(file_path),
        "-map", f"0:s:{sub_idx}", "-c", "copy", sup,
    ]
    # (if sup exists, caller will re-glob)
    yield [
        ffmpeg, "-y", "-v", "warning",
        "-i", sup, "-frames:v", "50", "-vsync", "0",
        f"{tmp_dir}/sub_%04d.png",
    ]
    # Method 3: overlay
    yield [
        ffmpeg, "-y", "-v", "warning",
        "-i", str(file_path),
        "-filter_complex", f"[0:v][0:s:{sub_idx}]overlay[v]",
        "-map", "[v]", "-frames:v", "50", "-vsync", "0", "-q:v", "2",
        f"{tmp_dir}/sub_%04d.png",
    ]


# ═══════════════════════════════════════════════════════════════════════════
#  Forced subtitle detection
# ═══════════════════════════════════════════════════════════════════════════

def calculate_subtitle_statistics(subtitles: List[Dict], duration: float) -> Dict:
    if not subtitles or duration <= 0:
        return {
            "total_duration": 0.0, "coverage_percent": 0.0, "density": 0.0,
            "count": 0, "avg_duration": 0.0, "gap_variance": 0.0,
            "subtitle_timings": [],
        }

    total_dur = 0.0
    timings: list[tuple] = []
    durations: list[float] = []

    for sub in subtitles:
        try:
            s = parse_srt_time(sub["start"])
            e = parse_srt_time(sub["end"])
            d = e - s
            if d > 0:
                total_dur += d
                timings.append((s, e))
                durations.append(d)
        except Exception:
            continue

    coverage = (total_dur / duration) * 100
    density = len(subtitles) / (duration / 60)
    avg_d = sum(durations) / len(durations) if durations else 0.0

    gaps = []
    for i in range(len(timings) - 1):
        g = timings[i + 1][0] - timings[i][1]
        if g >= 0:
            gaps.append(g)
    gap_var = 0.0
    if len(gaps) > 1:
        mean_g = sum(gaps) / len(gaps)
        gap_var = sum((g - mean_g) ** 2 for g in gaps) / len(gaps)

    return {
        "total_duration": total_dur,
        "coverage_percent": coverage,
        "density": density,
        "count": len(subtitles),
        "avg_duration": avg_d,
        "gap_variance": gap_var,
        "subtitle_timings": timings,
    }


def decide_forced_from_statistics(stats: Dict, duration_minutes: float, config) -> Tuple:
    """Return ``(is_forced | None, reason, confidence_level)``."""
    density = stats["density"]
    coverage = stats["coverage_percent"]
    count = stats["count"]

    # TIER 1 – high confidence
    if density < config.forced_subtitle_low_density_threshold and coverage < config.forced_subtitle_low_coverage_threshold:
        return True, f"Very low density ({density:.1f}) and coverage ({coverage:.1f}%)", 3
    if count < config.forced_subtitle_min_count_threshold:
        return True, f"Very low count ({count})", 3
    if density < 2.0:
        return True, f"Extremely low density ({density:.1f})", 3
    if density > config.forced_subtitle_high_density_threshold and coverage > 30.0:
        return False, f"High density ({density:.1f}) and coverage ({coverage:.1f}%)", 3
    if count > config.forced_subtitle_max_count_threshold and duration_minutes > 30:
        return False, f"High count ({count}) for {duration_minutes:.0f} min", 3
    if density > 10.0:
        return False, f"Very high density ({density:.1f})", 3

    # TIER 2 – medium confidence
    forced_ind = full_ind = 0
    if density < 5.0: forced_ind += 1
    if density > 6.0: full_ind += 1
    if coverage < 30.0: forced_ind += 1
    if coverage > 40.0: full_ind += 1
    if count < 150: forced_ind += 1
    if count > 250: full_ind += 1
    if stats["gap_variance"] > 100.0: forced_ind += 1
    if stats["gap_variance"] < 50.0: full_ind += 1

    if forced_ind >= 2 and full_ind == 0:
        return True, f"Multiple forced indicators (density={density:.1f}, coverage={coverage:.1f}%)", 2
    if full_ind >= 2 and forced_ind == 0:
        return False, f"Multiple full indicators (density={density:.1f}, coverage={coverage:.1f}%)", 2

    # TIER 3 – ambiguous
    reason = f"Ambiguous: density={density:.1f}, coverage={coverage:.1f}%, count={count}"
    if not config.analyze_forced_subtitles:
        return density < 5.5 or coverage < 37.5, reason + " (heuristic)", 1
    return None, reason, 1


def detect_forced_pgs_subtitles(
    ffprobe: str,
    file_path: Path,
    subtitle_track_index: int,
    show_details: bool = False,
) -> bool:
    """Heuristic forced detection for PGS subtitles based on frame count."""
    try:
        from uldas.audio import _get_file_duration
        duration = _get_file_duration(ffprobe, file_path, show_details)
        if duration <= 0:
            return False

        cmd = [
            ffprobe, "-v", "error",
            "-select_streams", f"s:{subtitle_track_index}",
            "-count_packets",
            "-show_entries", "stream=nb_read_packets",
            "-of", "csv=p=0", str(file_path),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True,
                                encoding="utf-8", errors="replace")
        frames = int(result.stdout.strip())
        fpm = frames / (duration / 60)

        if show_details:
            logger.info("PGS frame density: %.1f frames/min (%d frames)", fpm, frames)

        if frames < 100 or fpm < 5:
            return True
        if fpm > 30:
            return False
        return fpm < 15
    except Exception as exc:
        logger.error("Error detecting forced PGS subtitles: %s", exc)
        return False


# ═══════════════════════════════════════════════════════════════════════════
#  SDH detection
# ═══════════════════════════════════════════════════════════════════════════

def detect_sdh_subtitles(subtitle_path: Path) -> bool:
    """Return *True* if the subtitle track appears to be SDH."""
    try:
        subtitles = parse_srt_file(subtitle_path)
        if not subtitles:
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
        }

        sdh_count = 0
        for sub in subtitles:
            text = sub["text"]
            found = False
            for pat in sdh_patterns:
                for match in re.findall(pat, text, re.IGNORECASE):
                    content = re.sub(r"[\[\]\(\)\*♪]", "", match).strip().lower()
                    if any(kw in content for kw in sdh_keywords):
                        found = True
                        break
                if found:
                    break
            if found:
                sdh_count += 1

        ratio = sdh_count / len(subtitles) if subtitles else 0
        if ratio > 0.10:
            return True

        full_text = " ".join(s["text"].lower() for s in subtitles)
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
        logger.error("Error detecting SDH subtitles: %s", exc)
        return False


# ═══════════════════════════════════════════════════════════════════════════
#  Metadata update
# ═══════════════════════════════════════════════════════════════════════════

def update_subtitle_metadata(
    mkvpropedit: str,
    file_path: Path,
    track_index: int,
    language_code: str,
    is_forced: bool = False,
    is_sdh: bool = False,
    dry_run: bool = False,
    show_details: bool = False,
) -> bool:
    name_parts = [get_language_name(language_code)]
    if is_forced:
        name_parts.append("[Forced]")
    if is_sdh:
        name_parts.append("[SDH]")
    track_name = " ".join(name_parts)

    if dry_run:
        print(f"[DRY RUN] Would update subtitle track {track_index}: "
              f"language={language_code}, name='{track_name}'")
        return True

    try:
        cmd = [
            mkvpropedit, str(file_path),
            "--edit", f"track:s{track_index + 1}",
            "--set", f"language={language_code}",
            "--set", f"name={track_name}",
            "--set", f"flag-forced={'1' if is_forced else '0'}",
        ]
        subprocess.run(cmd, capture_output=True, text=True, check=True)
        if show_details:
            logger.info("Updated subtitle track %d: lang=%s name='%s'",
                        track_index, language_code, track_name)
        return True
    except subprocess.CalledProcessError as exc:
        logger.error("Error updating subtitle metadata: %s", exc)
        return False