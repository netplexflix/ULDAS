#file: uldas/detector.py

import gc
import json
import os
import subprocess
import sys
import time
import threading
import logging
from pathlib import Path
from typing import List, Dict, Optional, Tuple

from faster_whisper import WhisperModel

from uldas.config import Config
from uldas.constants import LANGUAGE_CODES, VIDEO_EXTENSIONS, EXTERNAL_SUBTITLE_EXTENSIONS
from uldas.tracking import ProcessingTracker
from uldas.tools import find_executable
from uldas.utils import (
    setup_cpu_limits,
    limit_subprocess_resources,
    normalize_language_code,
)
from uldas import audio as audio_mod
from uldas import subtitles as sub_mod
from uldas import external_subtitles as ext_sub_mod

logger = logging.getLogger(__name__)


def _flush_all_logs() -> None:
    """Flush all log handlers to ensure output is written before critical operations."""
    for handler in logging.getLogger().handlers:
        try:
            handler.flush()
        except Exception:
            pass
    try:
        sys.stdout.flush()
        sys.stderr.flush()
    except Exception:
        pass


class MKVLanguageDetector:
    """Scans directories, detects languages, and updates MKV metadata."""

    def __init__(self, config: Config):
        setup_cpu_limits()
        self.config = config
        self.deletion_failures: list[dict] = []

        # ── Tracking ─────────────────────────────────────────────────────
        if config.use_tracking:
            self.tracker = ProcessingTracker("config")
            if config.force_reprocess:
                logger.info("Force reprocess enabled – ignoring tracking cache")

        # ── Device / compute ─────────────────────────────────────────────
        device = self._determine_device()
        compute_type = self._determine_compute_type(device)
        cpu_threads = config.cpu_threads if config.cpu_threads > 0 else 0

        if config.show_details:
            logger.info("Initializing faster-whisper: device=%s, compute=%s, model=%s",
                        device, compute_type, config.whisper_model)

        # ── Whisper model ────────────────────────────────────────────────
        self.whisper_model = self._init_whisper(device, compute_type, cpu_threads)

        # ── External tools ───────────────────────────────────────────────
        self.ffmpeg = find_executable("ffmpeg")
        self.ffprobe = find_executable("ffprobe")
        self.mkvpropedit = find_executable("mkvpropedit")
        self.mkvmerge = find_executable("mkvmerge")

        missing = [n for n, v in [("ffmpeg", self.ffmpeg), ("ffprobe", self.ffprobe),
                                   ("mkvpropedit", self.mkvpropedit)] if not v]
        if missing:
            raise RuntimeError(f"Missing executables: {', '.join(missing)}")

        if not self.mkvmerge:
            logger.warning("mkvmerge not found – language detection may be less accurate")

    # ── Whisper init (with fallback) ─────────────────────────────────────
    def _init_whisper(self, device, compute_type, cpu_threads):
        stop = threading.Event()

        def _progress():
            dots = 0
            while not stop.is_set():
                time.sleep(2)
                if not stop.is_set():
                    dots = (dots + 1) % 4
                    logger.info("Still initializing%s", "." * dots)

        if self.config.show_details:
            t = threading.Thread(target=_progress, daemon=True)
            t.start()

        try:
            model = WhisperModel(
                self.config.whisper_model,
                device=device,
                compute_type=compute_type,
                cpu_threads=cpu_threads,
                download_root=None,
                local_files_only=False,
            )
            if self.config.show_details:
                logger.info("✓ WhisperModel initialised successfully")
            return model
        except Exception as exc:
            logger.warning("Primary init failed (%s), falling back to CPU", exc)
            try:
                model = WhisperModel(self.config.whisper_model, device="cpu")
                logger.info("✓ Fallback (CPU) initialisation successful")
                return model
            except Exception as fb_exc:
                raise RuntimeError(f"Failed to initialise Whisper: {fb_exc}") from fb_exc
        finally:
            stop.set()

    # ── Device helpers ───────────────────────────────────────────────────
    def _determine_device(self):
        if self.config.device != "auto":
            return self.config.device
        try:
            import torch
            if torch.cuda.is_available():
                return "cuda"
        except ImportError:
            pass
        return "cpu"

    def _determine_compute_type(self, device):
        if self.config.compute_type != "auto":
            return self.config.compute_type
        return "float16" if device == "cuda" else "int8"

    # ── File discovery ───────────────────────────────────────────────────
    def find_video_files(self, directory: str) -> List[Path]:
        """Walk *directory*, returning only video files that need processing."""
        exts: set[str] = {".mkv"}
        if self.config.remux_to_mkv:
            exts.update(VIDEO_EXTENSIONS)

        use_fast_skip = (
            self.config.use_tracking
            and hasattr(self, "tracker")
            and not self.config.force_reprocess
            and not self.config.reprocess_all
            and not self.config.reprocess_all_subtitles
        )
        tracked_keys: set[str] = set()
        if use_fast_skip:
            tracked_keys = self.tracker.get_fully_processed_keys(
                process_subtitles=self.config.process_subtitles,
            )
        all_files: List[Path] = []
        skipped_in_scan = 0
        dirs_scanned = 0
        last_report = time.monotonic()
        report_interval = 5.0

        if self.config.show_details:
            logger.info("Scanning directory tree: %s for video files (extensions: %s)",
                        directory, ", ".join(sorted(exts)))
        else:
            print(f"Scanning directory tree: {directory} for video files", flush=True)

        try:
            for dirpath, dirnames, filenames in os.walk(directory, followlinks=False):
                dirs_scanned += 1

                for filename in filenames:
                    dot_pos = filename.rfind(".")
                    if dot_pos <= 0:
                        continue
                    if filename[dot_pos:].lower() not in exts:
                        continue

                    full_path_str = os.path.join(dirpath, filename)

                    if use_fast_skip:
                        abs_path_str = os.path.abspath(full_path_str)
                        if abs_path_str in tracked_keys:
                            skipped_in_scan += 1
                            continue

                    all_files.append(Path(full_path_str))

                now = time.monotonic()
                if now - last_report >= report_interval and self.config.show_details:
                    last_report = now
                    if use_fast_skip:
                        logger.info(
                            "Scanning... %d dirs, %d new files, %d skipped (cached)",
                            dirs_scanned, len(all_files), skipped_in_scan,
                        )
                    else:
                        logger.info(
                            "Scanning... %d dirs, %d video files found",
                            dirs_scanned, len(all_files),
                        )

        except PermissionError as exc:
            logger.warning("Permission denied during scan: %s", exc)
        except Exception as exc:
            logger.error("Error during directory scan: %s", exc)

        if self.config.show_details:
            if use_fast_skip:
                logger.info(
                    "Scan complete: %d dirs, %d new files to check, %d skipped (cached)",
                    dirs_scanned, len(all_files), skipped_in_scan,
                )
            else:
                logger.info(
                    "Scan complete: %d dirs, %d video files found",
                    dirs_scanned, len(all_files),
                )
        else:
            if use_fast_skip:
                print(
                    f"\rScan complete: {dirs_scanned} dirs, "
                    f"{len(all_files)} new files, "
                    f"{skipped_in_scan} skipped (already processed)          ",
                )
            else:
                print(
                    f"\rScan complete: {dirs_scanned} dirs, "
                    f"{len(all_files)} video files found          ",
                )

        return all_files

    # ── External subtitle file discovery ─────────────────────────────────
    def find_external_subtitle_files(self, directory: str) -> tuple[List[Path], int]:
        use_tracking = (
            self.config.use_tracking
            and hasattr(self, "tracker")
        )
        use_fast_skip = (
            use_tracking
            and not self.config.force_reprocess
            and not self.config.reprocess_all_subtitles
        )

        all_files: List[Path] = []
        skipped_in_scan = 0
        dirs_scanned = 0
        last_report = time.monotonic()
        report_interval = 5.0

        if self.config.show_details:
            logger.info("Scanning directory tree: %s for subtitle files", directory)
        else:
            print(f"Scanning directory tree: {directory} for subtitle files", flush=True)

        try:
            for dirpath, dirnames, filenames in os.walk(directory, followlinks=False):
                dirs_scanned += 1

                for filename in filenames:
                    dot_pos = filename.rfind(".")
                    if dot_pos <= 0:
                        continue
                    if filename[dot_pos:].lower() not in EXTERNAL_SUBTITLE_EXTENSIONS:
                        continue

                    full_path_str = os.path.join(dirpath, filename)

                    # Skip files that already have a language tag unless
                    # reprocess_all_subtitles is set
                    sub_path = Path(full_path_str)
                    if not self.config.reprocess_all_subtitles:
                        lang_tag = ext_sub_mod.get_language_tag(sub_path)
                        if lang_tag is not None:
                            # Already has a language tag — track it as
                            # no_action_required so the dashboard counts
                            # it, then skip processing.
                            if use_tracking and not self.config.dry_run:
                                self.tracker.mark_external_subtitle_tracked(
                                    sub_path, language_code=lang_tag,
                                )
                            skipped_in_scan += 1
                            continue

                    # Fast skip via tracker (already processed in a prior run)
                    if use_fast_skip:
                        if self.tracker.is_external_subtitle_tracked(sub_path):
                            skipped_in_scan += 1
                            continue

                    all_files.append(sub_path)

                now = time.monotonic()
                if now - last_report >= report_interval and self.config.show_details:
                    last_report = now
                    logger.info(
                        "Scanning subtitles... %d dirs, %d new files, %d skipped",
                        dirs_scanned, len(all_files), skipped_in_scan,
                    )

        except PermissionError as exc:
            logger.warning("Permission denied during subtitle scan: %s", exc)
        except Exception as exc:
            logger.error("Error during subtitle scan: %s", exc)

        # Flush tracked no_action_required entries to disk
        if use_tracking and not self.config.dry_run:
            self.tracker.save_ext_sub_if_dirty()

        total_found = len(all_files) + skipped_in_scan

        if self.config.show_details:
            logger.info(
                "Subtitle scan complete: %d dirs, %d new files, %d skipped",
                dirs_scanned, len(all_files), skipped_in_scan,
            )
        else:
            print(
                f"Subtitle scan complete: {len(all_files)} new subtitle files, "
                f"{skipped_in_scan} skipped",
            )

        return all_files, total_found

    # ── Remux ────────────────────────────────────────────────────────────
    def remux_to_mkv(self, file_path: Path) -> Optional[Path]:
        if file_path.suffix.lower() == ".mkv":
            return file_path
        mkv_path = file_path.with_suffix(".mkv")
        if mkv_path.exists():
            return mkv_path
        if self.config.dry_run:
            print(f"[DRY RUN] Would remux {file_path.name} → {mkv_path.name}")
            return mkv_path

        try:
            print(f"Remuxing {file_path.name} to MKV…")
            strategies = self._build_remux_strategies(file_path, mkv_path)
            for strat in strategies:
                try:
                    limited = limit_subprocess_resources(strat["args"])
                    subprocess.run(limited, check=True, capture_output=True,
                                   text=True, encoding="utf-8", errors="replace")
                    if mkv_path.exists() and mkv_path.stat().st_size > 10_000:
                        verify = [self.ffprobe, "-v", "quiet", "-print_format", "json",
                                  "-show_streams", str(mkv_path)]
                        vr = subprocess.run(verify, check=True, capture_output=True,
                                            text=True, encoding="utf-8", errors="replace")
                        json.loads(vr.stdout)
                        if self.config.show_details:
                            logger.info("Remuxed with strategy '%s'", strat["name"])
                        else:
                            print(f"Successfully remuxed to: {mkv_path.name}")
                        self._remove_original(file_path, mkv_path)
                        return mkv_path
                except (subprocess.CalledProcessError, json.JSONDecodeError):
                    if mkv_path.exists():
                        mkv_path.unlink()
            logger.error("All remux strategies failed for %s", file_path)
            return None
        except Exception as exc:
            logger.error("Unexpected remux error: %s", exc)
            if mkv_path.exists():
                mkv_path.unlink()
            return None

    def _remove_original(self, original: Path, mkv: Path):
        try:
            gc.collect()
            time.sleep(2)
            for attempt in range(3):
                try:
                    original.unlink()
                    if self.config.show_details:
                        logger.info("Removed original: %s", original.name)
                    return
                except (OSError, PermissionError) as exc:
                    if attempt < 2:
                        time.sleep(1.0 * (attempt + 1))
                    else:
                        raise exc
        except Exception as exc:
            self.deletion_failures.append({
                "original_file": str(original),
                "mkv_file": str(mkv),
                "error": str(exc),
            })
            logger.warning("Could not remove original %s: %s", original, exc)

    def _build_remux_strategies(self, src: Path, dst: Path) -> list[dict]:
        """Return a list of ffmpeg remux strategies to try in order."""
        try:
            ar = subprocess.run(
                [self.ffprobe, "-v", "quiet", "-print_format", "json",
                 "-show_streams", str(src)],
                check=True, capture_output=True, text=True,
                encoding="utf-8", errors="replace",
            )
            streams = json.loads(ar.stdout).get("streams", [])
        except Exception:
            streams = []

        is_m2ts = src.suffix.lower() in (".m2ts", ".mts", ".ts")
        has_pcm = any(s.get("codec_name") == "pcm_bluray"
                      for s in streams if s.get("codec_type") == "audio")

        map_args: list[str] = []
        supported_sub = {
            "subrip", "srt", "ass", "ssa", "webvtt", "mov_text",
            "pgs", "dvdsub", "dvbsub", "hdmv_pgs_subtitle",
        }
        for i, s in enumerate(streams):
            ct = s.get("codec_type", "")
            cn = s.get("codec_name", "").lower()
            if ct in ("video", "audio"):
                map_args += ["-map", f"0:{i}"]
            elif ct == "subtitle" and cn in supported_sub:
                map_args += ["-map", f"0:{i}"]
        if not map_args:
            map_args = ["-map", "0:v", "-map", "0:a"]

        strats: list[dict] = []

        if is_m2ts:
            strats.append({
                "name": "m2ts_optimized",
                "args": [
                    self.ffmpeg, "-y", "-v", "warning", "-fflags", "+genpts",
                    "-analyzeduration", "100M", "-probesize", "100M",
                    "-i", str(src),
                    "-map", "0:v", "-c:v", "copy",
                    "-map", "0:a", "-c:a", "flac" if has_pcm else "copy",
                    "-avoid_negative_ts", "make_zero",
                    "-fflags", "+discardcorrupt", "-map_metadata", "0",
                    str(dst),
                ],
            })

        strats += [
            {
                "name": "selective_copy",
                "args": [
                    self.ffmpeg, "-y", "-v", "warning", "-fflags", "+genpts",
                    "-i", str(src), "-c", "copy",
                ] + map_args + [
                    "-avoid_negative_ts", "make_zero", "-map_metadata", "0",
                    str(dst),
                ],
            },
            {
                "name": "no_subtitles",
                "args": [
                    self.ffmpeg, "-y", "-v", "warning", "-fflags", "+genpts",
                    "-i", str(src),
                    "-map", "0:v", "-c:v", "copy",
                    "-map", "0:a", "-c:a", "copy",
                    "-avoid_negative_ts", "make_zero", "-map_metadata", "0",
                    str(dst),
                ],
            },
            {
                "name": "force_remux",
                "args": [
                    self.ffmpeg, "-y", "-v", "warning",
                    "-i", str(src),
                    "-map", "0:v", "-c:v", "libx264", "-preset", "ultrafast", "-crf", "23",
                    "-map", "0:a", "-c:a", "copy",
                    "-avoid_negative_ts", "make_zero",
                    str(dst),
                ],
            },
        ]
        return strats

    # ── MKV info ─────────────────────────────────────────────────────────
    def get_mkv_info(self, file_path: Path) -> Dict:
        mkvmerge = find_executable("mkvmerge")
        if not mkvmerge:
            return self._get_mkv_info_ffprobe(file_path)
        try:
            cmd = [mkvmerge, "-J", str(file_path)]
            r = subprocess.run(cmd, capture_output=True, text=True, check=True,
                               encoding="utf-8", errors="replace")
            data = json.loads(r.stdout)
            return self._convert_mkvmerge_to_ffprobe(data)
        except Exception:
            return self._get_mkv_info_ffprobe(file_path)

    def _convert_mkvmerge_to_ffprobe(self, data: dict) -> dict:
        type_map = {"video": "video", "audio": "audio", "subtitles": "subtitle"}
        info: dict = {"streams": []}
        for track in data.get("tracks", []):
            props = track.get("properties", {})
            s: dict = {
                "index": track.get("id", 0),
                "codec_type": type_map.get(track.get("type", "").lower(), track.get("type", "")),
                "codec_name": props.get("codec_id", ""),
                "tags": {},
            }
            lang = props.get("language", "")
            if lang:
                s["tags"]["language"] = lang
            name = props.get("track_name", "")
            if name:
                s["tags"]["title"] = name
            info["streams"].append(s)
        return info

    def _get_mkv_info_ffprobe(self, file_path: Path) -> Dict:
        try:
            cmd = [self.ffprobe, "-v", "quiet", "-print_format", "json",
                   "-show_streams", str(file_path)]
            r = subprocess.run(cmd, capture_output=True, text=True, check=True,
                               encoding="utf-8", errors="replace")
            return json.loads(r.stdout)
        except Exception:
            return {}

    def _read_format_audio_langs(self, file_path: Path) -> Dict[int, str]:
        """Read audio language tags from format-level metadata (e.g. AVI IAS tags)."""
        from uldas.constants import LANGUAGE_CODES
        try:
            cmd = [self.ffprobe, "-v", "quiet", "-print_format", "json",
                   "-show_format", str(file_path)]
            r = subprocess.run(cmd, capture_output=True, text=True, check=True,
                               encoding="utf-8", errors="replace")
            tags = json.loads(r.stdout).get("format", {}).get("tags", {})
        except Exception:
            return {}
        result: Dict[int, str] = {}
        for key, value in tags.items():
            if not key.upper().startswith("IAS"):
                continue
            try:
                idx = int(key[3:]) - 1          # IAS1 → audio index 0
            except ValueError:
                continue
            code = LANGUAGE_CODES.get(value.lower().strip())
            if code:
                result[idx] = normalize_language_code(code)
        return result

    # ── Track finders ────────────────────────────────────────────────────
    def _find_tracks(self, file_path: Path, codec_type: str,
                     only_undefined: bool) -> list:
        info = self.get_mkv_info(file_path)
        tracks = []
        if "streams" not in info:
            return tracks
        count = 0
        for i, stream in enumerate(info["streams"]):
            if stream.get("codec_type") != codec_type:
                continue
            tags = stream.get("tags", {})
            lang = None
            for k in tags:
                if k.lower() in ("language", "lang"):
                    lang = tags[k].lower().strip()
                    break
            lang = normalize_language_code(lang)
            undef = {"und", "unknown", "undefined", "undetermined", ""}
            if only_undefined and lang and lang not in undef:
                count += 1
                continue
            tracks.append((count, stream, i, lang or "und"))
            count += 1
        return tracks

    def find_undefined_audio_tracks(self, fp): return self._find_tracks(fp, "audio", True)
    def find_all_audio_tracks(self, fp): return self._find_tracks(fp, "audio", False)
    def find_undefined_subtitle_tracks(self, fp): return self._find_tracks(fp, "subtitle", True)
    def find_all_subtitle_tracks(self, fp): return self._find_tracks(fp, "subtitle", False)

    # ── Audio language detection ─────────────────────────────────────────
    def detect_language_with_retries(self, file_path, track_idx, stream_idx, max_retries=3):
        return audio_mod.detect_language_with_retries(
            self.whisper_model, self.ffmpeg, self.ffprobe,
            file_path, track_idx, stream_idx, self.config, max_retries,
        )

    # ── Audio metadata update ────────────────────────────────────────────
    def update_mkv_language(self, file_path: Path, track_index: int,
                            language_code: str, dry_run: bool = False) -> bool:
        if dry_run:
            print(f"[DRY RUN] Would update track {track_index} → {language_code}")
            return True
        try:
            cmd = [
                self.mkvpropedit, str(file_path),
                "--edit", f"track:a{track_index + 1}",
                "--set", f"language={language_code}",
            ]
            subprocess.run(cmd, capture_output=True, text=True, check=True)
            if self.config.show_details:
                logger.info("Updated audio track %d → %s", track_index, language_code)
            return True
        except subprocess.CalledProcessError as exc:
            logger.error("Error updating %s: %s", file_path, exc)
            return False

    # ── Subtitle processing (embedded) ───────────────────────────────────
    def process_subtitle_tracks(self, file_path: Path) -> Dict:
        results = {
            "subtitle_tracks_found": 0,
            "processed_subtitle_tracks": [],
            "failed_subtitle_tracks": [],
            "skipped_subtitle_tracks": [],
            "subtitle_errors": [],
        }
        if not self.config.process_subtitles:
            return results

        if self.config.reprocess_all_subtitles:
            tracks = self.find_all_subtitle_tracks(file_path)
        else:
            tracks = self.find_undefined_subtitle_tracks(file_path)

        results["subtitle_tracks_found"] = len(tracks)
        if not tracks:
            return results

        for sub_idx, stream_info, stream_idx, cur_lang in tracks:
            subtitle_path = None
            try:
                subtitle_path = sub_mod.extract_subtitle_track(
                    self.ffmpeg, file_path, sub_idx, stream_idx,
                    self.get_mkv_info, self.config.show_details,
                )
                if not subtitle_path:
                    results["failed_subtitle_tracks"].append(sub_idx)
                    continue

                lang_result = sub_mod.detect_subtitle_language(
                    subtitle_path, file_path, sub_idx, stream_idx,
                    self.ffmpeg, self.ffprobe, self.config.show_details,
                )
                if not lang_result:
                    results["failed_subtitle_tracks"].append(sub_idx)
                    continue

                code = lang_result["language_code"]
                conf = lang_result["confidence"]

                if conf < self.config.subtitle_confidence_threshold:
                    results["skipped_subtitle_tracks"].append({
                        "track_index": sub_idx,
                        "detected_language": code,
                        "confidence": conf,
                        "reason": "confidence_below_threshold",
                    })
                    continue

                is_forced = False
                if self.config.analyze_forced_subtitles:
                    is_forced = self._detect_forced(file_path, sub_idx, stream_idx, subtitle_path)

                is_sdh = False
                if self.config.detect_sdh_subtitles and subtitle_path.suffix.lower() != ".sup":
                    is_sdh = sub_mod.detect_sdh_subtitles(subtitle_path)

                ok = sub_mod.update_subtitle_metadata(
                    self.mkvpropedit, file_path, sub_idx, code,
                    is_forced, is_sdh, self.config.dry_run, self.config.show_details,
                )
                if ok:
                    results["processed_subtitle_tracks"].append({
                        "track_index": sub_idx,
                        "detected_language": code,
                        "previous_language": cur_lang,
                        "confidence": conf,
                        "is_forced": is_forced,
                        "is_sdh": is_sdh,
                    })
                else:
                    results["failed_subtitle_tracks"].append(sub_idx)
            except Exception as exc:
                logger.error("Error processing subtitle track %d: %s", sub_idx, exc)
                results["failed_subtitle_tracks"].append(sub_idx)
            finally:
                if subtitle_path and subtitle_path.exists():
                    subtitle_path.unlink()

        return results

    # ── External subtitle processing (independent) ───────────────────────
    def process_external_subtitle_file(self, sub_path: Path) -> Dict:
        """Process a single external subtitle file independently.

        Detects language and SDH, then renames the file with the
        appropriate tags.  Returns a result dict.
        """
        result = {
            "original_file": str(sub_path),
            "new_file": None,
            "detected_language": None,
            "confidence": 0.0,
            "is_sdh": False,
            "status": "failed",
            "reason": None,
        }

        try:
            lang_result = ext_sub_mod.detect_external_subtitle_language(
                sub_path, self.config.show_details,
            )
            if not lang_result:
                result["reason"] = "detection_failed"
                return result

            code = lang_result["language_code"]
            conf = lang_result["confidence"]
            result["detected_language"] = code
            result["confidence"] = conf

            if code in ("und", "zxx"):
                result["status"] = "skipped"
                result["reason"] = "undetermined_language"
                return result

            if conf < self.config.subtitle_confidence_threshold:
                result["status"] = "skipped"
                result["reason"] = "confidence_below_threshold"
                return result

            # SDH detection for text-based subtitle files
            is_sdh = False
            if self.config.detect_sdh_subtitles and sub_path.suffix.lower() != ".sup":
                is_sdh = ext_sub_mod.detect_sdh_in_external_subtitle(sub_path)
            result["is_sdh"] = is_sdh

            new_path = ext_sub_mod.rename_subtitle_with_language(
                sub_path, code, is_sdh=is_sdh,
                dry_run=self.config.dry_run,
                show_details=self.config.show_details,
            )
            if new_path:
                result["new_file"] = str(new_path)
                result["status"] = "processed"

                # Track this subtitle file individually
                if (self.config.use_tracking
                        and hasattr(self, "tracker")
                        and not self.config.dry_run):
                    self.tracker.mark_external_subtitle_processed(
                        sub_path, code, new_path,
                    )
            else:
                result["reason"] = "rename_failed"

        except Exception as exc:
            logger.error("Error processing external subtitle %s: %s", sub_path, exc)
            result["reason"] = str(exc)

        return result

    def _detect_forced(self, file_path, sub_idx, stream_idx, subtitle_path):
        """Forced detection dispatcher."""
        if subtitle_path.suffix.lower() == ".sup":
            return sub_mod.detect_forced_pgs_subtitles(
                self.ffprobe, file_path, sub_idx, self.config.show_details,
            )

        subs = sub_mod.parse_srt_file(subtitle_path)
        if not subs:
            return False

        from uldas.audio import _get_file_duration
        duration = _get_file_duration(self.ffprobe, file_path, self.config.show_details)
        if duration <= 0:
            return False

        stats = sub_mod.calculate_subtitle_statistics(subs, duration)
        decision, reason, confidence = sub_mod.decide_forced_from_statistics(
            stats, duration / 60, self.config,
        )

        if self.config.show_details:
            logger.info("Forced decision: %s (confidence=%d, reason=%s)",
                        decision, confidence, reason)

        if confidence >= 2:
            return decision

        # Low confidence → audio analysis
        full_audio = audio_mod.extract_full_audio_track(
            self.ffmpeg, file_path, 0, stream_idx,
            self.config.operation_timeout_seconds, self.config.show_details,
        )
        if not full_audio:
            return stats["density"] < 5.5 or stats["coverage_percent"] < 37.5

        try:
            segments, info = self.whisper_model.transcribe(
                str(full_audio), language=None, task="transcribe",
                beam_size=1, best_of=1, temperature=0.0,
                vad_filter=True,
                vad_parameters={"min_speech_duration_ms": 250, "max_speech_duration_s": 30},
                word_timestamps=True,
            )
            speech = [(s.start, s.end) for s in segments]
            if not speech:
                return True

            total_speech = sum(e - s for s, e in speech)
            overlap = 0.0
            for ss, se in stats["subtitle_timings"]:
                for sp_s, sp_e in speech:
                    os_ = max(ss, sp_s)
                    oe = min(se, sp_e)
                    if os_ < oe:
                        overlap += oe - os_

            pct = (overlap / total_speech * 100) if total_speech > 0 else 0
            return pct < 50
        except Exception:
            return stats["density"] < 5.5 or stats["coverage_percent"] < 37.5
        finally:
            if full_audio and full_audio.exists():
                full_audio.unlink()

    # ── Process single video file ────────────────────────────────────────
    def process_file(self, file_path: Path, _cached_key: str = None) -> Dict:
        """Process a single video file.

        Parameters
        ----------
        file_path : Path
            The video file to process.
        _cached_key : str, optional
            Pre-resolved absolute path string.  Avoids redundant
            ``Path.absolute()`` / ``os.path.abspath()`` calls.
        """
        results: Dict = {
            "original_file": str(file_path),
            "mkv_file": None,
            "was_remuxed": False,
            "undefined_tracks": 0,
            "processed_tracks": [],
            "failed_tracks": [],
            "errors": [],
            "subtitle_results": None,
            "external_subtitle_results": None,
            "skipped_due_to_tracking": False,
        }

        abs_key = _cached_key or os.path.abspath(str(file_path))

        skip_audio = skip_subs = False
        if self.config.use_tracking and hasattr(self, "tracker"):
            bypass = self.config.reprocess_all or self.config.reprocess_all_subtitles
            if not bypass:
                entry = self.tracker.get_entry(file_path, key=abs_key)
                if entry:
                    if entry.get("audio_processed"):
                        skip_audio = True
                    if entry.get("subtitle_processed"):
                        skip_subs = True
                    if skip_audio and (skip_subs or not self.config.process_subtitles):
                        if self.config.show_details:
                            logger.info("Skipping %s (already processed)", file_path.name)
                        results["skipped_due_to_tracking"] = True
                        return results

        # Remux
        mkv_path = file_path
        original_audio_langs: Dict[int, str] = {}
        if self.config.remux_to_mkv and file_path.suffix.lower() != ".mkv":
            # Save defined audio language tags — ffmpeg remux can lose them
            orig_info = self.get_mkv_info(file_path)
            undef_langs = {"und", "unknown", "undefined", "undetermined", ""}
            _aidx = 0
            for _s in orig_info.get("streams", []):
                if _s.get("codec_type") != "audio":
                    continue
                _lang = None
                for _k in _s.get("tags", {}):
                    if _k.lower() in ("language", "lang"):
                        _lang = _s["tags"][_k].lower().strip()
                        break
                if _lang and normalize_language_code(_lang) not in undef_langs:
                    original_audio_langs[_aidx] = _lang
                _aidx += 1

            # Also check format-level tags (AVI IAS metadata)
            fmt_langs = self._read_format_audio_langs(file_path)
            for _idx, _code in fmt_langs.items():
                if _idx not in original_audio_langs:
                    original_audio_langs[_idx] = _code

            if self.config.show_details:
                logger.debug("Original audio languages for %s: %s",
                             file_path.name, original_audio_langs or "(none detected)")

            mkv_path = self.remux_to_mkv(file_path)
            if mkv_path and mkv_path != file_path:
                results["was_remuxed"] = True
                # Re-apply any audio language tags that were lost in remux
                for _aidx, _lang in original_audio_langs.items():
                    try:
                        subprocess.run(
                            [self.mkvpropedit, str(mkv_path),
                             "--edit", f"track:a{_aidx + 1}",
                             "--set", f"language={_lang}"],
                            capture_output=True, text=True, check=True,
                        )
                    except (subprocess.CalledProcessError, Exception):
                        pass
            elif not mkv_path:
                results["errors"].append("Failed to remux")
                return results
        results["mkv_file"] = str(mkv_path)

        audio_ok = subtitle_ok = False

        # ── Audio ────────────────────────────────────────────────────────
        if not skip_audio:
            tracks = (self.find_all_audio_tracks(mkv_path)
                      if self.config.reprocess_all
                      else self.find_undefined_audio_tracks(mkv_path))

            # After remux, tracks that were already labeled in the original
            # file may appear undefined (ffmpeg can lose tags).  Re-apply
            # the known label directly instead of running detection.
            if (results["was_remuxed"] and original_audio_langs
                    and not self.config.reprocess_all):
                genuinely_undefined = []
                for t in tracks:
                    tidx = t[0]
                    if tidx in original_audio_langs:
                        self.update_mkv_language(
                            mkv_path, tidx,
                            original_audio_langs[tidx],
                            self.config.dry_run,
                        )
                    else:
                        genuinely_undefined.append(t)
                tracks = genuinely_undefined

            results["undefined_tracks"] = len(tracks)
            if not tracks:
                audio_ok = True
            else:
                failures = False
                for tidx, _, sidx, cur_lang in tracks:
                    _flush_all_logs()
                    try:
                        code = self.detect_language_with_retries(mkv_path, tidx, sidx)
                        if not code:
                            results["failed_tracks"].append(tidx)
                            failures = True
                            continue
                        if self.update_mkv_language(mkv_path, tidx, code, self.config.dry_run):
                            results["processed_tracks"].append({
                                "track_index": tidx,
                                "detected_language": code,
                                "previous_language": cur_lang,
                            })
                        else:
                            results["failed_tracks"].append(tidx)
                            failures = True
                    except Exception as exc:
                        logger.error("Error on track %d: %s", tidx, exc)
                        results["failed_tracks"].append(tidx)
                        failures = True
                audio_ok = not failures
        else:
            audio_ok = True

        # ── Embedded Subtitles ───────────────────────────────────────────
        if self.config.process_subtitles and not skip_subs:
            if self.config.show_details:
                logger.info("Processing subtitles for: %s", mkv_path.name)
            sub_res = self.process_subtitle_tracks(mkv_path)
            results["subtitle_results"] = sub_res

            subtitle_ok = (
                not sub_res["failed_subtitle_tracks"]
                and not sub_res["subtitle_errors"]
                and not sub_res["skipped_subtitle_tracks"]
            )
        elif skip_subs:
            subtitle_ok = True

        # ── Tracking ─────────────────────────────────────────────────────
        if (self.config.use_tracking and hasattr(self, "tracker")
                and not self.config.dry_run):
            flags = []
            if results["was_remuxed"]:
                flags.append("remuxed")
                # Use remuxed file path as tracking key so the next run
                # recognises the .mkv instead of creating a duplicate entry.
                old_key = abs_key
                abs_key = os.path.abspath(str(mkv_path))
                if old_key != abs_key and old_key in self.tracker.data:
                    del self.tracker.data[old_key]
                    self.tracker._dirty = True
            audio_count = 0
            if results["processed_tracks"]:
                # Only flag "audio_labeled" for genuinely undefined tracks,
                # not for labels that were merely restored after remux.
                genuinely_detected = [
                    t for t in results["processed_tracks"]
                    if t["track_index"] not in original_audio_langs
                ]
                audio_count = len(genuinely_detected)
                if audio_count:
                    flags.append("audio_labeled")
            subtitle_count = 0
            sub_res = results.get("subtitle_results")
            if sub_res and sub_res.get("processed_subtitle_tracks"):
                subtitle_count = len(sub_res["processed_subtitle_tracks"])
                flags.append("subtitle_labeled")
            if not flags:
                flags.append("no_action_required")
            self.tracker.mark_processed(
                mkv_path, audio_ok, subtitle_ok,
                key=abs_key, flags=flags,
                audio_tracks_labeled=audio_count,
                subtitle_tracks_labeled=subtitle_count,
            )

        # Clean up memory between files
        audio_mod._cleanup_cuda_memory()

        return results

    # ── Process directory ────────────────────────────────────────────────
    def process_directory(self, directory: str) -> Tuple[List[Dict], List[Dict], int]:
        video_results = []
        ext_sub_results = []
        total_ext_subs_found = 0

        # Prune tracker entries for files that no longer exist in this dir
        if self.config.use_tracking and hasattr(self, "tracker"):
            pruned = self.tracker.prune_missing_files(directory)
            if pruned and self.config.show_details:
                logger.info("Pruned %d orphaned tracker entries", pruned)

        # ── Video files ──────────────────────────────────────────────────
        video_files = self.find_video_files(directory)

        if not video_files:
            if self.config.show_details:
                logger.info("No new video files found in: %s", directory)
            else:
                print(f"No new video files found in: {directory}")
        else:
            video_results = self._process_video_files(video_files)

        # ── External subtitle files (independent scan) ───────────────────
        if self.config.process_external_subtitles:
            ext_sub_files, total_ext_subs_found = self.find_external_subtitle_files(directory)

            if not ext_sub_files:
                if self.config.show_details:
                    logger.info("No new external subtitle files found in: %s", directory)
                else:
                    print(f"No new external subtitle files found in: {directory}")
            else:
                ext_sub_results = self._process_ext_sub_files(ext_sub_files)

        return video_results, ext_sub_results, total_ext_subs_found

    def _process_video_files(self, video_files: List[Path]) -> List[Dict]:
        """Batch-validate and process video files."""
        results = []
        total = len(video_files)

        actionable = video_files
        skipped_files: list[Path] = []
        key_cache: dict[str, str] = {}

        if (self.config.use_tracking
                and hasattr(self, "tracker")
                and not self.config.force_reprocess):
            bypass = self.config.reprocess_all or self.config.reprocess_all_subtitles
            if not bypass and len(video_files) > 0:
                if self.config.show_details:
                    logger.info("Validating %d candidate files against tracker...", total)

                check_start = time.monotonic()
                actionable, skipped_files, key_cache = self.tracker.check_files_batch(
                    video_files,
                    process_subtitles=self.config.process_subtitles,
                )
                check_elapsed = time.monotonic() - check_start

                for fp in skipped_files:
                    results.append({
                        "original_file": str(fp),
                        "mkv_file": None,
                        "was_remuxed": False,
                        "undefined_tracks": 0,
                        "processed_tracks": [],
                        "failed_tracks": [],
                        "errors": [],
                        "subtitle_results": None,
                        "external_subtitle_results": None,
                        "skipped_due_to_tracking": True,
                    })

                if self.config.show_details:
                    logger.info(
                        "Validation complete in %.1fs: %d to process, %d skipped",
                        check_elapsed, len(actionable), len(skipped_files),
                    )
                elif skipped_files:
                    print(
                        f"Validation complete: {len(actionable)} to process, "
                        f"{len(skipped_files)} already done (skipped)"
                    )

        if not actionable:
            if self.config.show_details:
                logger.info("All video files already processed")
            else:
                print("All video files already processed – nothing to do")
            return results

        action_total = len(actionable)
        for action_idx, fp in enumerate(actionable, 1):
            try:
                if self.config.show_details:
                    logger.info("[%d/%d] Processing: %s",
                                action_idx, action_total, fp.name)
                else:
                    print(f"[{action_idx}/{action_total}] Processing: {fp.name}")

                cached_key = key_cache.get(str(fp))
                results.append(self.process_file(fp, _cached_key=cached_key))
            except Exception as exc:
                logger.error("Error processing %s: %s", fp, exc)
                results.append({
                    "original_file": str(fp), "mkv_file": None,
                    "was_remuxed": False, "undefined_tracks": 0,
                    "processed_tracks": [], "failed_tracks": [],
                    "errors": [str(exc)],
                    "subtitle_results": None,
                    "external_subtitle_results": None,
                    "skipped_due_to_tracking": False,
                })

        return results

    def _process_ext_sub_files(self, ext_sub_files: List[Path]) -> List[Dict]:
        """Process a list of external subtitle files."""
        results = []
        total = len(ext_sub_files)

        for idx, sub_path in enumerate(ext_sub_files, 1):
            try:
                if self.config.show_details:
                    logger.info("[%d/%d] Processing subtitle: %s",
                                idx, total, sub_path.name)
                else:
                    print(f"[{idx}/{total}] Processing subtitle: {sub_path.name}")

                result = self.process_external_subtitle_file(sub_path)
                results.append(result)
            except Exception as exc:
                logger.error("Error processing subtitle %s: %s", sub_path, exc)
                results.append({
                    "original_file": str(sub_path),
                    "new_file": None,
                    "detected_language": None,
                    "confidence": 0.0,
                    "is_sdh": False,
                    "status": "failed",
                    "reason": str(exc),
                })

        return results