#file: uldas/audio.py

import subprocess
import tempfile
import logging
import gc
from pathlib import Path
from typing import Optional, Dict, List, Tuple

import numpy as np

from uldas.utils import (
    limit_subprocess_resources,
    normalize_language_code,
    is_likely_hallucination,
)
from uldas.constants import LANGUAGE_CODES

logger = logging.getLogger(__name__)

SAMPLE_RATE = 16000
LID_WINDOW_S = 30          # Whisper's language-ID input length

# ── Detection tuning ─────────────────────────────────────────────────────
# A sample with less speech than this (per VAD) carries no language
# evidence — credits, an action scene, a music cue.  It is skipped; it
# never makes the whole track 'zxx'.
MIN_SAMPLE_SPEECH_S = 3.0
# 'zxx' requires VAD over the *whole* track to find less speech than
# max(ZXX_MAX_SPEECH_S, ZXX_MAX_SPEECH_RATIO × duration), or every bit
# of that "speech" to turn out to be music or singing (see below).
ZXX_MAX_SPEECH_S = 15.0
ZXX_MAX_SPEECH_RATIO = 0.01
# Whisper's language ID answers "Latin" for music and singing.  Real
# audio tracks are practically never in Latin, so audio it calls Latin
# is treated as non-speech rather than as language evidence.
NON_SPEECH_LANGUAGES = {"la"}
# A language no single source was confident about is accepted when its
# speech-weighted pooled probability reaches this and it is the top
# language of at least two sources (and of at least half of them).
POOLED_ACCEPT = 0.75
# Language-ID windows per sample (covers a 60–90 s sample), and windows
# spread over the speech of the whole track in the fallback.
SAMPLE_LID_WINDOWS = 3
FULL_TRACK_LID_WINDOWS = 6

# Returned when there is speech but no confident language: the track
# is left as 'und'.
INCONCLUSIVE = "und"


def _log_memory_usage(context: str = "") -> None:
    try:
        import psutil
        proc = psutil.Process()
        mem = proc.memory_info()
        logger.debug("Memory [%s]: RSS=%.1fMB, VMS=%.1fMB",
                      context, mem.rss / 1024 / 1024, mem.vms / 1024 / 1024)
    except Exception:
        pass


def _cleanup_memory() -> None:
    """Aggressively free memory after transcription.

    CTranslate2 manages its own (CUDA) allocator, so only host memory is
    trimmed here.
    """
    gc.collect()
    try:
        import ctypes
        libc = ctypes.CDLL("libc.so.6")
        libc.malloc_trim(0)
    except Exception:
        pass

_cleanup_cuda_memory = _cleanup_memory


def _unlink(path: Optional[Path]) -> None:
    try:
        if path and path.exists():
            path.unlink()
    except OSError:
        pass


# ── Volume check ─────────────────────────────────────────────────────────
def has_reasonable_volume(ffmpeg: str, audio_path: Path) -> bool:
    try:
        cmd = [
            ffmpeg, "-i", str(audio_path),
            "-af", "volumedetect",
            "-f", "null", "-", "-v", "quiet", "-stats",
        ]
        result = subprocess.run(cmd, capture_output=True, text=True,
                                encoding="utf-8", errors="replace")
        if "mean_volume:" in result.stderr:
            for line in result.stderr.split("\n"):
                if "mean_volume:" in line:
                    try:
                        db = float(line.split("mean_volume:")[1].split("dB")[0].strip())
                        return db > -60.0
                    except (ValueError, IndexError):
                        pass
        return True
    except Exception:
        return True


# ── Decoding / VAD ───────────────────────────────────────────────────────
def _decode_audio(audio_path: Path) -> Optional[np.ndarray]:
    """Decode *audio_path* to 16 kHz mono float32, or None on failure."""
    try:
        from faster_whisper.audio import decode_audio
        return decode_audio(str(audio_path), sampling_rate=SAMPLE_RATE)
    except Exception as exc:
        logger.warning("Could not decode audio %s: %s", audio_path.name, exc)
        return None


def _chunks_seconds(chunks: List[dict]) -> float:
    return sum(c["end"] - c["start"] for c in chunks) / SAMPLE_RATE


def _vad_speech(audio: np.ndarray, config,
                show_details: bool = False) -> Optional[List[dict]]:
    """Run Silero VAD on *audio* and return its speech chunks.

    Returns ``None`` when VAD itself fails — callers must treat that as
    "speech unknown", never as silence.
    """
    try:
        from faster_whisper.vad import VadOptions, get_speech_timestamps
        vad_options = VadOptions(
            min_speech_duration_ms=config.vad_min_speech_duration_ms,
            max_speech_duration_s=float(config.vad_max_speech_duration_s),
        )
        chunks = get_speech_timestamps(audio, vad_options)
    except Exception as exc:
        logger.warning("VAD failed (%s: %s) – speech presence unknown",
                       type(exc).__name__, exc)
        return None
    if show_details:
        logger.info("VAD: %.1fs of speech in %.1fs of audio (%d segment(s))",
                    _chunks_seconds(chunks), len(audio) / SAMPLE_RATE, len(chunks))
    return chunks


def _speech_audio(audio: np.ndarray, chunks: Optional[List[dict]],
                  config) -> np.ndarray:
    """The audio Whisper analyses: only the detected speech when the VAD
    filter is enabled and VAD worked, otherwise the audio as-is."""
    if config.vad_filter and chunks:
        return np.concatenate([audio[c["start"]:c["end"]] for c in chunks])
    return audio


# ── Percentage-based sample extraction ───────────────────────────────────
def extract_audio_sample_percentage_based(
    ffmpeg: str,
    ffprobe: str,
    file_path: Path,
    audio_track_index: int,
    stream_index: int,
    retry_attempt: int = 0,
    show_details: bool = False,
) -> Optional[Path]:
    try:
        duration = _get_file_duration(ffprobe, file_path, show_details)
        if duration <= 0:
            duration = 7200

        if show_details:
            logger.info("File duration: %.1f minutes", duration / 60)

        min_start = max(60, duration * 0.05)
        max_start = duration * 0.85

        # One sample is taken per attempt: the first position that can be
        # extracted.  The first positions of successive attempts are spread
        # over the runtime; the rest are nearby fallbacks.
        if duration > 3600:
            sample_dur = 90
            all_pcts = [
                [0.25, 0.15, 0.35, 0.20, 0.30],
                [0.50, 0.45, 0.55, 0.40, 0.60],
                [0.75, 0.70, 0.80, 0.65, 0.85],
            ]
        elif duration > 1800:
            sample_dur = 75
            all_pcts = [
                [0.25, 0.15, 0.35, 0.30],
                [0.50, 0.40, 0.60, 0.45],
                [0.75, 0.65, 0.85, 0.70],
            ]
        else:
            sample_dur = 60
            all_pcts = [
                [0.25, 0.15, 0.35],
                [0.50, 0.40, 0.60],
                [0.75, 0.65, 0.85],
            ]

        pcts = all_pcts[min(retry_attempt, len(all_pcts) - 1)]
        segments = []
        for p in pcts:
            s = max(min_start, min(max_start, duration * p))
            segments.append((int(s), sample_dur))

        if show_details:
            logger.info(
                "Will try %d samples: %s", len(segments),
                ", ".join(f"{int(s / 60)}m{s % 60:02.0f}s" for s, _ in segments),
            )

        mappings = [
            f"0:a:{audio_track_index}",
            f"0:{stream_index}",
            f"a:{audio_track_index}",
        ]

        for seg_start, seg_dur in segments:
            for mapping in mappings:
                tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
                tmp_path = Path(tmp.name)
                tmp.close()
                try:
                    cmd = [
                        ffmpeg, "-y", "-v", "error",
                        "-ss", str(seg_start),
                        "-i", str(file_path),
                        "-t", str(seg_dur),
                        "-map", mapping,
                        "-ar", "16000", "-ac", "1",
                        "-af", "volume=2.0,highpass=f=80,lowpass=f=8000,dynaudnorm=f=200:g=3",
                        "-f", "wav", str(tmp_path),
                    ]
                    limited = limit_subprocess_resources(cmd)
                    subprocess.run(limited, check=True, capture_output=True,
                                   text=True, encoding="utf-8", errors="replace")

                    if tmp_path.exists() and tmp_path.stat().st_size > 10_000:
                        if has_reasonable_volume(ffmpeg, tmp_path):
                            if show_details:
                                logger.info(
                                    "Extracted audio from %dm%02ds",
                                    seg_start // 60, seg_start % 60,
                                )
                            return tmp_path
                    if tmp_path.exists():
                        tmp_path.unlink()
                except subprocess.CalledProcessError:
                    if tmp_path.exists():
                        tmp_path.unlink()

        logger.error("All percentage-based extraction attempts failed")
        return None
    except Exception as exc:
        logger.error("Unexpected error during audio extraction: %s", exc)
        return None


# ── Full-track extraction ────────────────────────────────────────────────
def extract_full_audio_track(
    ffmpeg: str,
    file_path: Path,
    audio_track_index: int,
    stream_index: int,
    timeout: int = 600,
    show_details: bool = False,
) -> Optional[Path]:
    try:
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        tmp_path = Path(tmp.name)
        tmp.close()

        mappings = [
            f"0:a:{audio_track_index}",
            f"0:{stream_index}",
            f"a:{audio_track_index}",
        ]

        for mapping in mappings:
            try:
                cmd = [
                    ffmpeg, "-y", "-v", "error",
                    "-i", str(file_path),
                    "-map", mapping,
                    "-ar", "16000", "-ac", "1",
                    "-af", "volume=2.0,highpass=f=80,lowpass=f=8000,dynaudnorm=f=200:g=3",
                    "-f", "wav", str(tmp_path),
                ]
                limited = limit_subprocess_resources(cmd)
                if show_details:
                    logger.info("Extracting full audio track %d (timeout: %ds)",
                                audio_track_index, timeout)
                subprocess.run(limited, check=True, capture_output=True,
                               text=True, encoding="utf-8", errors="replace",
                               timeout=timeout)
                if tmp_path.exists() and tmp_path.stat().st_size > 10_000:
                    if show_details:
                        logger.info("Successfully extracted full audio track %d",
                                    audio_track_index)
                    return tmp_path
                if tmp_path.exists():
                    tmp_path.unlink()
            except subprocess.TimeoutExpired:
                logger.error("Full audio extraction timed out after %ds", timeout)
                if tmp_path.exists():
                    tmp_path.unlink()
                return None
            except subprocess.CalledProcessError:
                if tmp_path.exists():
                    tmp_path.unlink()

        logger.error("All full audio extraction attempts failed")
        return None
    except Exception as exc:
        logger.error("Unexpected error during full audio extraction: %s", exc)
        return None


# ── Language identification ──────────────────────────────────────────────
def _to_code(whisper_lang: str) -> str:
    """Whisper language → the code ULDAS writes."""
    lang = whisper_lang.lower()
    if lang == "jw":            # Whisper's legacy code for Javanese
        lang = "jv"
    return normalize_language_code(LANGUAGE_CODES.get(lang, lang))


def _top(probs: Dict[str, float]) -> Tuple[str, float]:
    return max(probs.items(), key=lambda kv: kv[1])


def _fmt_probs(probs: Dict[str, float], n: int = 3) -> str:
    ranked = sorted(probs.items(), key=lambda kv: kv[1], reverse=True)[:n]
    return ", ".join(f"{lang} {p:.2f}" for lang, p in ranked)


def _merge_probs(items: List[Dict]) -> Dict[str, float]:
    """Weighted average of the ``probs`` of *items* (weight = speech seconds)."""
    total = sum(it["weight"] for it in items)
    if total <= 0:
        return {}
    merged: Dict[str, float] = {}
    for it in items:
        for lang, p in it["probs"].items():
            merged[lang] = merged.get(lang, 0.0) + p * it["weight"]
    return {lang: v / total for lang, v in merged.items()}


def _identify_windows(whisper_model, audio: np.ndarray, max_windows: int,
                      spread: bool = False) -> List[Dict]:
    """Whisper language ID on up to *max_windows* 30 s windows of *audio*.

    Windows run consecutively from the start, or are spread evenly over
    all of *audio* when *spread* is set and there is room for that.
    Windows never overlap, so no stretch of audio is counted twice.
    Returns one ``{"weight": seconds, "probs": {lang: p}}`` dict per window.
    """
    win = LID_WINDOW_S * SAMPLE_RATE
    total = len(audio)
    count = min(max_windows, max(1, -(-total // win)))
    if spread and count > 1 and total > count * win:
        starts = [round(i * (total - win) / (count - 1)) for i in range(count)]
    else:
        starts = [i * win for i in range(count)]

    windows = []
    for start in starts:
        chunk = audio[start:start + win]
        if len(chunk) < SAMPLE_RATE:        # under 1 s: no usable evidence
            continue
        _, _, all_probs = whisper_model.detect_language(audio=chunk)
        windows.append({
            "weight": len(chunk) / SAMPLE_RATE,
            "probs": dict(all_probs),
        })
    return windows


def _transcribe_text(whisper_model, audio: np.ndarray, language: str,
                     config, label: str) -> str:
    """Decode *audio* in *language* for the hallucination check.

    Whisper's own VAD stays off: *audio* is already speech-only when the
    VAD filter is enabled, and an empty VAD result crashes CTranslate2.
    """
    if config.show_details:
        logger.info("Transcribing %s (%s)...", label, language)
        _log_memory_usage(f"before_transcription_{label}")

    segments, _ = whisper_model.transcribe(
        audio,
        language=language,
        task="transcribe",
        beam_size=3,
        best_of=2,
        temperature=0.0,
        patience=1.0,
        length_penalty=1.0,
        repetition_penalty=1.2,
        no_repeat_ngram_size=3,
        compression_ratio_threshold=2.0,
        log_prob_threshold=-0.8,
        no_speech_threshold=0.6,
        condition_on_previous_text=False,
        initial_prompt=None,
        word_timestamps=False,
        prepend_punctuations="\"'([{-",
        append_punctuations="\"'.,:!?)]}",
        vad_filter=False,
    )
    text = " ".join(seg.text.strip() for seg in segments).strip()

    if config.show_details:
        logger.info("Sample text: '%s'", text[:150])
        _log_memory_usage(f"after_transcription_{label}")
    return text


def analyse_sample(whisper_model, audio_path: Path, config,
                   label: str) -> Dict:
    """Classify one extracted sample.

    ``kind`` is 'silent' (VAD found too little speech), 'music' (Whisper
    calls it one of :data:`NON_SPEECH_LANGUAGES`), 'noise' (the transcript
    looks like a hallucination), 'speech' or 'error'.  Samples that reach
    language ID carry ``language``, ``probability`` (Whisper's language
    probability, averaged over the sample's windows), ``probs`` and
    ``weight`` (seconds of audio used).
    """
    result: Dict = {"label": label, "kind": "error", "speech_s": None}

    audio = _decode_audio(audio_path)
    if audio is None or len(audio) < SAMPLE_RATE:
        return result

    chunks = _vad_speech(audio, config, config.show_details)
    if chunks is not None:
        result["speech_s"] = _chunks_seconds(chunks)
        if result["speech_s"] < MIN_SAMPLE_SPEECH_S:
            result["kind"] = "silent"
            return result

    speech = _speech_audio(audio, chunks, config)
    try:
        windows = _identify_windows(whisper_model, speech, SAMPLE_LID_WINDOWS)
        if not windows:
            return result
        probs = _merge_probs(windows)
        language, probability = _top(probs)
        text = None
        if language not in NON_SPEECH_LANGUAGES:
            text = _transcribe_text(whisper_model, speech, language, config, label)
    except Exception as exc:
        logger.error("Language analysis failed (%s): %s", label, exc)
        return result
    finally:
        _cleanup_memory()

    result.update(
        language=language,
        probability=probability,
        probs=probs,
        weight=sum(w["weight"] for w in windows),
        text=text,
    )
    if text is None:
        result["kind"] = "music"
    elif is_likely_hallucination(text, language, config.show_details):
        result["kind"] = "noise"
    else:
        result["kind"] = "speech"
    return result


def analyse_full_track(whisper_model, audio_path: Path, config,
                       speech_found: bool = False) -> Dict:
    """Full-track fallback: VAD over the whole track, then language ID on
    windows spread across its speech.

    ``kind`` is 'silent' (the only route to 'zxx': VAD found next to no
    speech, or all of it is music/singing), 'speech' (with one entry per
    usable window in ``sources``) or 'error'.  With *speech_found* (a
    sample already held real speech) the track is never 'silent'.
    """
    result: Dict = {"label": "full track", "kind": "error",
                    "speech_s": None, "sources": []}

    audio = _decode_audio(audio_path)
    if audio is None or len(audio) < SAMPLE_RATE:
        return result
    duration_s = len(audio) / SAMPLE_RATE

    chunks = _vad_speech(audio, config, config.show_details)
    if chunks is not None:
        speech_s = _chunks_seconds(chunks)
        result["speech_s"] = speech_s
        floor = max(ZXX_MAX_SPEECH_S, ZXX_MAX_SPEECH_RATIO * duration_s)
        if speech_s < floor and not speech_found:
            if config.show_details:
                logger.info("Full track: %.1fs of speech in %.0fs of audio "
                            "(below %.1fs) – no linguistic content",
                            speech_s, duration_s, floor)
            result["kind"] = "silent"
            return result

    speech = _speech_audio(audio, chunks, config)
    del audio
    try:
        windows = _identify_windows(whisper_model, speech,
                                    FULL_TRACK_LID_WINDOWS, spread=True)
    except Exception as exc:
        logger.error("Full-track language analysis failed: %s", exc)
        return result
    finally:
        _cleanup_memory()

    music_windows = 0
    for i, w in enumerate(windows, 1):
        language, probability = _top(w["probs"])
        if language in NON_SPEECH_LANGUAGES:
            music_windows += 1
            if config.show_details:
                logger.info("full track window %d: top: %s → music",
                            i, _fmt_probs(w["probs"]))
            continue
        result["sources"].append({
            "label": f"full track window {i}",
            "kind": "speech",
            "speech_s": w["weight"],
            "language": language,
            "probability": probability,
            "probs": w["probs"],
            "weight": w["weight"],
        })
    if result["sources"]:
        result["kind"] = "speech"
    elif music_windows:
        # Only music: no linguistic content, unless a sample heard speech
        # (then the samples alone decide).
        result["kind"] = "speech" if speech_found else "silent"
        if config.show_details and not speech_found:
            logger.info("Full track: the speech VAD found is music/singing – "
                        "no linguistic content")
    return result


def _log_source(res: Dict, config) -> None:
    if not config.show_details:
        return
    speech = ("speech unknown" if res.get("speech_s") is None
              else f"{res['speech_s']:.1f}s speech")
    if "probs" in res:
        logger.info("%s: %s, top: %s → %s",
                    res["label"], speech, _fmt_probs(res["probs"]), res["kind"])
    else:
        logger.info("%s: %s → %s", res["label"], speech, res["kind"])


def _pooled_decision(sources: List[Dict], config) -> Optional[str]:
    """Return the Whisper language backed by several sources, or None."""
    if not sources:
        return None
    pooled = _merge_probs(sources)
    if not pooled:
        return None
    language, probability = _top(pooled)
    votes = sum(1 for s in sources if s["language"] == language)
    if config.show_details:
        logger.info("Pooled evidence from %d source(s): %s (%s top in %d)",
                    len(sources), _fmt_probs(pooled), language, votes)
    if (probability >= POOLED_ACCEPT and votes >= 2
            and votes * 2 >= len(sources)):
        return language
    return None


# ── High-level detection ─────────────────────────────────────────────────
def detect_language_with_retries(
    whisper_model,
    ffmpeg: str,
    ffprobe: str,
    file_path: Path,
    audio_track_index: int,
    stream_index: int,
    config,
    max_retries: int = 3,
) -> Optional[str]:
    """Detect the spoken language of one audio track.

    Returns a language code, ``'zxx'`` (VAD found practically no speech
    in the whole track), :data:`INCONCLUSIVE` (there is speech but no
    confident language) or ``None`` (nothing could be analysed).
    """
    threshold = config.confidence_threshold
    samples: List[Dict] = []

    for attempt in range(max_retries):
        if attempt > 0 and config.show_details:
            logger.info("Sample %d/%d – trying a different position",
                        attempt + 1, max_retries)

        sample = extract_audio_sample_percentage_based(
            ffmpeg, ffprobe, file_path, audio_track_index, stream_index,
            attempt, config.show_details,
        )
        if not sample:
            continue
        try:
            res = analyse_sample(whisper_model, sample, config,
                                 f"sample {attempt + 1}")
        finally:
            _unlink(sample)
        samples.append(res)
        _log_source(res, config)

        if res["kind"] == "speech" and res["probability"] >= threshold:
            code = _to_code(res["language"])
            if config.show_details:
                logger.info("Detected '%s' (probability %.3f) on sample %d",
                            code, res["probability"], attempt + 1)
            return code

    sources = [s for s in samples if s["kind"] == "speech"]
    language = _pooled_decision(sources, config)
    if language:
        return _to_code(language)

    if config.show_details:
        logger.info("No confident language from %d sample(s) – analysing "
                    "full track", len(samples))
    else:
        print(f"Low confidence detected, analysing full audio track "
              f"for track {audio_track_index}...")

    full = extract_full_audio_track(
        ffmpeg, file_path, audio_track_index, stream_index,
        config.operation_timeout_seconds, config.show_details,
    )
    res = None
    if full:
        try:
            res = analyse_full_track(whisper_model, full, config,
                                     speech_found=bool(sources))
        finally:
            _unlink(full)

    if res is None or res["kind"] == "error":
        if not samples:
            logger.warning("All detection attempts failed for track %d",
                           audio_track_index)
            return None
        logger.warning("Full-track analysis unavailable for track %d – "
                       "leaving it undetermined", audio_track_index)
        return INCONCLUSIVE

    if res["kind"] == "silent":
        return "zxx"

    for src in res["sources"]:
        _log_source(src, config)
    all_sources = sources + res["sources"]
    if len(all_sources) == 1 and all_sources[0]["probability"] >= threshold:
        return _to_code(all_sources[0]["language"])
    language = _pooled_decision(all_sources, config)
    if language:
        return _to_code(language)

    logger.info("No confident language for track %d – leaving it 'und'",
                audio_track_index)
    return INCONCLUSIVE


# ── Internal helper ──────────────────────────────────────────────────────
def _get_file_duration(ffprobe: str, file_path: Path, show_details: bool) -> float:
    try:
        cmd = [
            ffprobe, "-v", "quiet",
            "-show_entries", "format=duration",
            "-of", "csv=p=0", str(file_path),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True,
                                encoding="utf-8", errors="replace")
        return float(result.stdout.strip())
    except (subprocess.CalledProcessError, ValueError) as exc:
        if show_details:
            logger.debug("Could not get duration for %s: %s", file_path, exc)
        return 0.0
