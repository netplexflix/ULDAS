#file: uldas/audio.py

import subprocess
import tempfile
import logging
import gc
from pathlib import Path
from typing import Optional, Dict

from uldas.utils import (
    limit_subprocess_resources,
    normalize_language_code,
    is_likely_hallucination,
)
from uldas.constants import LANGUAGE_CODES

logger = logging.getLogger(__name__)


def _log_memory_usage(context: str = "") -> None:
    try:
        import psutil
        proc = psutil.Process()
        mem = proc.memory_info()
        logger.debug("Memory [%s]: RSS=%.1fMB, VMS=%.1fMB",
                      context, mem.rss / 1024 / 1024, mem.vms / 1024 / 1024)
    except Exception:
        pass
    try:
        import torch
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024 / 1024
            reserved = torch.cuda.memory_reserved() / 1024 / 1024
            logger.debug("CUDA Memory [%s]: allocated=%.1fMB, reserved=%.1fMB",
                          context, allocated, reserved)
    except Exception:
        pass


def _cleanup_memory() -> None:
    """Aggressively free memory after transcription."""
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass
    try:
        import ctypes
        libc = ctypes.CDLL("libc.so.6")
        libc.malloc_trim(0)
    except Exception:
        pass

_cleanup_cuda_memory = _cleanup_memory


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


# ── Pre-check: run VAD separately to detect no-speech before Whisper ────
def _vad_has_speech(audio_path: Path, config, show_details: bool = False) -> bool:
    """Run Silero VAD on the audio and return True only if speech is found.

    This prevents calling whisper_model.transcribe(vad_filter=True) on
    audio where VAD would strip everything, which crashes CTranslate2.
    """
    try:
        from faster_whisper.audio import decode_audio
        try:
            from faster_whisper.vad import VadOptions, get_vad_model, get_speech_timestamps
        except ImportError:
            from faster_whisper.vad import (
                VadOptions,
                SileroVADModel,
                get_speech_timestamps,
            )
            get_vad_model = SileroVADModel

        audio = decode_audio(str(audio_path), sampling_rate=16000)

        if show_details:
            logger.info("Pre-VAD check: audio length %.2f seconds", len(audio) / 16000)

        vad_options = VadOptions(
            min_speech_duration_ms=config.vad_min_speech_duration_ms,
            max_speech_duration_s=float(config.vad_max_speech_duration_s),
        )

        model = get_vad_model() if callable(get_vad_model) else get_vad_model
        speech_timestamps = get_speech_timestamps(audio, vad_options)

        has_speech = len(speech_timestamps) > 0
        if show_details:
            logger.info("Pre-VAD check: %d speech segment(s) found", len(speech_timestamps))

        return has_speech

    except Exception as exc:
        if show_details:
            logger.debug("Pre-VAD check failed (%s: %s), will skip VAD transcription",
                         type(exc).__name__, exc)
        # If pre-check fails, safer to skip VAD transcription entirely
        # rather than risk the crash
        return False


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

        if duration > 3600:
            sample_dur = 90
            all_pcts = [
                [0.15, 0.25, 0.35, 0.50, 0.65],
                [0.08, 0.20, 0.45, 0.75, 0.88],
                [0.12, 0.40, 0.60, 0.80, 0.90],
            ]
        elif duration > 1800:
            sample_dur = 75
            all_pcts = [
                [0.15, 0.30, 0.50, 0.70],
                [0.08, 0.40, 0.65, 0.85],
                [0.25, 0.45, 0.75, 0.90],
            ]
        else:
            sample_dur = 60
            all_pcts = [
                [0.20, 0.50, 0.80],
                [0.10, 0.35, 0.75],
                [0.30, 0.60, 0.90],
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


# ── Whisper transcription ────────────────────────────────────────────────
def attempt_transcription(
    whisper_model,
    audio_path: Path,
    use_vad: bool,
    attempt_name: str,
    config,
    ffmpeg: str = None,
) -> Optional[Dict]:
    """Run Whisper on *audio_path*.

    When *use_vad* is True we first run Silero VAD independently.  If
    VAD finds no speech we short-circuit immediately — never calling
    ``transcribe(vad_filter=True)`` which crashes CTranslate2 on
    zero-length output, and never calling ``transcribe(vad_filter=False)``
    which would waste memory on audio that has no speech anyway.
    """
    try:
        if config.show_details:
            logger.info("Starting transcription (%s)...", attempt_name)
            _log_memory_usage(f"before_transcription_{attempt_name}")
            print(f"Transcribing audio ({attempt_name})...", flush=True)

        # ── Pre-VAD: bail out immediately if no speech ───────────────────
        if use_vad:
            if not _vad_has_speech(audio_path, config, config.show_details):
                if config.show_details:
                    logger.info(
                        "Pre-VAD found no speech – returning zxx without "
                        "calling Whisper (%s)", attempt_name,
                    )
                return {
                    "language": "zxx",
                    "confidence": 0.0,
                    "text": "",
                    "text_length": 0,
                    "word_count": 0,
                    "segments_detected": 0,
                    "attempt_name": attempt_name,
                    "vad_removed_all": True,
                    "pre_vad_silent": True,
                }

        vad_options = None
        if use_vad:
            vad_options = {
                "min_speech_duration_ms": config.vad_min_speech_duration_ms,
                "max_speech_duration_s": config.vad_max_speech_duration_s,
            }

        temperature = 0.0 if attempt_name == "with_vad" else 0.2

        segments, info = whisper_model.transcribe(
            str(audio_path),
            language=None,
            task="transcribe",
            beam_size=3,
            best_of=2,
            temperature=temperature,
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
            vad_filter=use_vad,
            vad_parameters=vad_options,
        )

        segments_list = list(segments)
        text_sample = " ".join(seg.text for seg in segments_list).strip()
        vad_removed_all = use_vad and len(segments_list) == 0

        confidence = info.language_probability
        if segments_list:
            seg_confs = []
            for seg in segments_list:
                if hasattr(seg, "avg_logprob") and seg.avg_logprob is not None:
                    seg_confs.append(min(1.0, max(0.0, seg.avg_logprob + 1.0)))
            if seg_confs:
                confidence = max(confidence, sum(seg_confs) / len(seg_confs))

        if config.show_details:
            logger.info("Detected language: %s (confidence: %.2f, method: %s)",
                        info.language, confidence, attempt_name)
            logger.info("Sample text: '%s'", text_sample[:150])
            logger.info("Segments found: %d", len(segments_list))
            _log_memory_usage(f"after_transcription_{attempt_name}")

        return {
            "language": info.language,
            "confidence": confidence,
            "text": text_sample,
            "text_length": len(text_sample),
            "word_count": len(text_sample.split()) if text_sample else 0,
            "segments_detected": len(segments_list),
            "attempt_name": attempt_name,
            "vad_removed_all": vad_removed_all,
        }
    except RuntimeError as exc:
        exc_str = str(exc).lower()
        if "out of memory" in exc_str or "cuda" in exc_str:
            logger.error("CUDA/Runtime error during transcription (%s): %s",
                         attempt_name, exc)
        else:
            logger.error("Runtime error during transcription (%s): %s",
                         attempt_name, exc)
        return None
    except Exception as exc:
        if config.show_details:
            logger.debug("Transcription attempt '%s' failed: %s", attempt_name, exc)
        return None
    finally:
        _cleanup_memory()


def process_transcription_result(result: Dict, config) -> Optional[str]:
    # Pre-VAD already determined no speech
    if result.get("pre_vad_silent"):
        if config.show_details:
            logger.info("Pre-VAD confirmed no speech – marking as 'zxx'")
        return "zxx"

    if result["confidence"] > 0.95 and result["text_length"] > 50:
        code = LANGUAGE_CODES.get(result["language"].lower(), result["language"])
        if result["language"].lower() in ("dutch", "nl"):
            code = "dut"
        return normalize_language_code(code)

    if config.show_details:
        logger.info("Checking transcription quality and hallucination patterns...")
    else:
        print("Analyzing transcription quality...", flush=True)

    if result["text"] and is_likely_hallucination(result["text"], config.show_details):
        if config.show_details:
            logger.info("Detected likely hallucination – marking as 'zxx'")
        return "zxx"

    if result["attempt_name"] == "without_vad" and result.get("vad_removed_all", False):
        has_speech = (
            (result["confidence"] > 0.7 and result["text_length"] > 30
             and result["word_count"] > 5)
            or (result["confidence"] > 0.5 and result["text_length"] > 100
                and result["word_count"] > 20)
        )
        if not has_speech:
            if config.show_details:
                logger.info("VAD removed all audio but transcription produced "
                            "text – likely hallucination")
            return "zxx"
    else:
        has_speech = (
            (result["confidence"] > 0.6 and result["text_length"] > 0)
            or (result["confidence"] > 0.3 and result["text_length"] > 15
                and result["word_count"] > 2)
            or (result["confidence"] > 0.2 and result["text_length"] > 50
                and result["word_count"] > 8)
            or (result["text_length"] > 100 and result["word_count"] > 15)
        )

    if has_speech:
        code = LANGUAGE_CODES.get(result["language"].lower(), result["language"])
        if result["language"].lower() in ("dutch", "nl"):
            code = "dut"
        return normalize_language_code(code)

    if config.show_details:
        logger.info("Insufficient evidence of speech – marking as 'zxx'")
    return "zxx"


# ── High-level detection ─────────────────────────────────────────────────
def detect_language_with_fallback(
    whisper_model, audio_path: Path, config, ffmpeg: str = None,
) -> Optional[str]:
    if not audio_path.exists() or audio_path.stat().st_size < 1000:
        logger.error("Audio file too small or missing: %s", audio_path)
        return None

    vad_removed_all = False

    if config.vad_filter:
        result = attempt_transcription(
            whisper_model, audio_path, True, "with_vad", config, ffmpeg=ffmpeg,
        )
        if result and result.get("pre_vad_silent"):
            # VAD confirmed no speech — skip without_vad entirely
            return process_transcription_result(result, config)
        if result and result["segments_detected"] > 0:
            return process_transcription_result(result, config)
        if result and result["segments_detected"] == 0:
            vad_removed_all = True

    result = attempt_transcription(
        whisper_model, audio_path, False, "without_vad", config, ffmpeg=ffmpeg,
    )
    if result:
        result["vad_removed_all"] = vad_removed_all
        return process_transcription_result(result, config)

    logger.error("Both transcription attempts failed")
    return None


def detect_language_with_confidence(
    whisper_model, audio_path: Path, config, ffmpeg: str = None,
) -> Optional[Dict]:
    if not audio_path.exists() or audio_path.stat().st_size < 1000:
        logger.error("Audio file too small or missing: %s", audio_path)
        return None

    vad_removed_all = False

    if config.vad_filter:
        result = attempt_transcription(
            whisper_model, audio_path, True, "with_vad", config, ffmpeg=ffmpeg,
        )
        if result and result.get("pre_vad_silent"):
            code = process_transcription_result(result, config)
            return {"language_code": code, "confidence": 0.0,
                    "method": "pre_vad_silent"}
        if result and result["segments_detected"] > 0:
            code = process_transcription_result(result, config)
            return {"language_code": code, "confidence": result["confidence"],
                    "method": "with_vad"}
        if result and result["segments_detected"] == 0:
            vad_removed_all = True

    result = attempt_transcription(
        whisper_model, audio_path, False, "without_vad", config, ffmpeg=ffmpeg,
    )
    if result:
        result["vad_removed_all"] = vad_removed_all
        code = process_transcription_result(result, config)
        return {"language_code": code, "confidence": result["confidence"],
                "method": "without_vad"}

    logger.error("Both transcription attempts failed")
    return None


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
    successful = []
    best_confidence = 0.0
    best_result = None
    all_pre_vad_silent = True

    for attempt in range(max_retries):
        if attempt > 0 and config.show_details:
            logger.info("Retry attempt %d/%d – trying different audio samples",
                        attempt + 1, max_retries)

        sample = extract_audio_sample_percentage_based(
            ffmpeg, ffprobe, file_path, audio_track_index, stream_index,
            attempt, config.show_details,
        )
        if not sample:
            continue

        try:
            res = detect_language_with_confidence(
                whisper_model, sample, config, ffmpeg=ffmpeg,
            )
            if sample.exists():
                sample.unlink()
            if res:
                code = res.get("language_code")
                conf = res.get("confidence", 0.0)
                method = res.get("method", "")

                if method != "pre_vad_silent":
                    all_pre_vad_silent = False

                if conf > best_confidence:
                    best_confidence = conf
                    best_result = code
                if code:
                    successful.append(code)
                    if code != "zxx" and conf >= config.confidence_threshold:
                        if config.show_details:
                            logger.info("Detected '%s' (conf %.3f) on attempt %d",
                                        code, conf, attempt + 1)
                        return code

                # If pre-VAD says silent, don't waste retries on same file
                if method == "pre_vad_silent" and code == "zxx":
                    if config.show_details:
                        logger.info("Pre-VAD confirmed no speech on attempt %d "
                                    "– skipping remaining retries", attempt + 1)
                    return "zxx"
        except Exception as exc:
            if config.show_details:
                logger.warning("Retry %d error: %s", attempt + 1, exc)
            if sample and sample.exists():
                sample.unlink()

    if (best_confidence >= config.confidence_threshold
            and best_result and best_result != "zxx"):
        return best_result

    # If all samples were pre-VAD silent, no point doing full track
    if all_pre_vad_silent and successful and all(s == "zxx" for s in successful):
        if config.show_details:
            logger.info("All samples confirmed silent by pre-VAD – marking as 'zxx'")
        return "zxx"

    if config.show_details:
        logger.info("Best confidence (%.3f) below threshold (%.3f) – analysing full track",
                     best_confidence, config.confidence_threshold)
    else:
        print(f"Low confidence detected, analysing full audio track "
              f"for track {audio_track_index}...")

    full = extract_full_audio_track(
        ffmpeg, file_path, audio_track_index, stream_index,
        config.operation_timeout_seconds, config.show_details,
    )
    if full:
        try:
            res = detect_language_with_confidence(
                whisper_model, full, config, ffmpeg=ffmpeg,
            )
            if full.exists():
                full.unlink()
            if res:
                code = res["language_code"]
                conf = res["confidence"]
                if code and code != "zxx" and conf >= config.confidence_threshold:
                    return code
                if code == "zxx":
                    return code
                return "zxx"
        except Exception:
            if full and full.exists():
                full.unlink()

    if successful:
        from collections import Counter
        real = [lang for lang in successful if lang != "zxx"]
        if real:
            return Counter(real).most_common(1)[0][0]
        if all(lang == "zxx" for lang in successful):
            return "zxx"

    logger.warning("All detection attempts failed for track %d", audio_track_index)
    return None


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