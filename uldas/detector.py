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
from typing import Callable, List, Dict, Optional, Tuple

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

    def __init__(self, config: Config,
                 cancel_check: Optional[Callable[[], bool]] = None):
        setup_cpu_limits()
        self.config = config
        self.deletion_failures: list[dict] = []
        self._cancel_check = cancel_check
        self._cancel_logged = False
        self._reprocess_lang_force_all_audio = False

        # Per-file metadata caches, cleared at the start of each
        # process_file() call.  ffprobe/mkvmerge are expensive enough
        # that memoizing within a file's processing lifetime is worth
        # it — several code paths query the same metadata.
        self._mkv_info_cache: Dict[str, Dict] = {}
        self._duration_cache: Dict[str, float] = {}
        self._format_lang_cache: Dict[str, Dict[int, str]] = {}

        # ── Tracking ─────────────────────────────────────────────────────
        if config.use_tracking:
            # Dry-run uses a read-only tracker so scan/prune/reader side
            # effects never hit disk — the dashboard's All Time badges
            # must not move during a preview.
            self.tracker = ProcessingTracker("config", read_only=config.dry_run)
            if config.force_reprocess:
                logger.info("Force reprocess enabled – ignoring tracking cache")
            if config.reprocess_language:
                logger.info(
                    "Reprocess-by-language mode: scanning every file for audio "
                    "tracks currently tagged '%s'; subtitle processing disabled "
                    "for this run",
                    config.reprocess_language,
                )

        # ── Language index (incremental) ─────────────────────────────────
        from uldas.language_index import LanguageIndex
        self.language_index = LanguageIndex(
            config_dir="config", read_only=config.dry_run,
        )

        removed = self.language_index.prune_ignored_tags(config.ignore_tags)
        if removed["files"] or removed["ext_subs"]:
            logger.info(
                "Pruned %d video + %d ext-sub language-index entries "
                "matching ignore_tags",
                removed["files"], removed["ext_subs"],
            )
            self.language_index.save_if_dirty()

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

    # ── Cancellation ─────────────────────────────────────────────────────
    def _should_cancel(self) -> bool:
        """Check whether a graceful cancel has been requested."""
        try:
            if self._cancel_check is not None and self._cancel_check():
                if not self._cancel_logged:
                    logger.info("Cancellation requested — stopping after current file")
                    self._cancel_logged = True
                return True
        except Exception:
            pass
        return False

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

    # ── Unified directory scan ───────────────────────────────────────────
    def _scan_tree(self, directory: str) -> tuple[list[Path], list[Path], int, int, int, int]:
        """Walk *directory* once, collecting both video and external
        subtitle files in a single pass.

        Returns
        -------
        (video_files, subtitle_files, video_skipped, sub_skipped,
         new_already_labeled, dirs_scanned)

        Also populates ``self._last_scan_seen_paths`` — the set of
        absolute paths of **all** video and subtitle files observed in
        *directory*, before any fast-skip filtering.  This set is used
        by ``prune_missing_files`` to drop stale tracker entries without
        issuing any extra filesystem calls.
        """
        video_exts: set[str] = {".mkv"}
        if self.config.remux_to_mkv:
            video_exts.update(VIDEO_EXTENSIONS)
        sub_exts = EXTERNAL_SUBTITLE_EXTENSIONS

        scan_subs = self.config.process_external_subtitles

        use_tracking = self.config.use_tracking and hasattr(self, "tracker")
        use_fast_skip_video = (
            use_tracking
            and not self.config.force_reprocess
            and not self.config.reprocess_all
            and not self.config.reprocess_all_subtitles
            and not self.config.reprocess_language
        )
        use_fast_skip_sub = (
            use_tracking
            and not self.config.force_reprocess
            and not self.config.reprocess_all_subtitles
        )

        tracked_video_keys: set[str] = set()
        if use_fast_skip_video:
            tracked_video_keys = self.tracker.get_fully_processed_keys(
                process_subtitles=self.config.process_subtitles,
            )

        video_files: list[Path] = []
        sub_files: list[Path] = []
        seen_paths: set[str] = set()

        video_skipped = 0
        sub_skipped = 0
        new_already_labeled = 0
        dirs_scanned = 0
        last_report = time.monotonic()
        report_interval = 5.0

        if self.config.show_details:
            logger.info(
                "Scanning directory tree: %s (video exts: %s)",
                directory, ", ".join(sorted(video_exts)),
            )
        else:
            print(f"Scanning directory tree: {directory}", flush=True)

        try:
            for dirpath, dirnames, filenames in os.walk(directory, followlinks=False):
                dirs_scanned += 1

                for filename in filenames:
                    dot_pos = filename.rfind(".")
                    if dot_pos <= 0:
                        continue
                    ext = filename[dot_pos:].lower()

                    is_video = ext in video_exts
                    is_sub = scan_subs and ext in sub_exts
                    if not (is_video or is_sub):
                        continue

                    if self.config.ignore_tags:
                        stem_lower = filename[:dot_pos].lower()
                        if any(tag and tag.lower() in stem_lower
                               for tag in self.config.ignore_tags):
                            if is_video:
                                video_skipped += 1
                            else:
                                sub_skipped += 1
                            continue

                    full_path_str = os.path.join(dirpath, filename)
                    abs_path_str = os.path.abspath(full_path_str)
                    seen_paths.add(abs_path_str)

                    if is_video:
                        if use_fast_skip_video and abs_path_str in tracked_video_keys:
                            video_skipped += 1
                            continue
                        video_files.append(Path(full_path_str))
                    else:  # is_sub
                        sub_path = Path(full_path_str)
                        if use_fast_skip_sub and self.tracker.is_external_subtitle_tracked(sub_path):
                            sub_skipped += 1
                            continue
                        if not self.config.reprocess_all_subtitles:
                            lang_tag = ext_sub_mod.get_language_tag(sub_path)
                            if lang_tag is not None:
                                if use_tracking and not self.config.dry_run:
                                    self.tracker.mark_external_subtitle_tracked(
                                        sub_path, language_code=lang_tag,
                                    )
                                # Already-tagged external subs never reach
                                # process_external_subtitle_file, so record
                                # them in the language index here.
                                if (not self.config.dry_run
                                        and hasattr(self, "language_index")):
                                    self.language_index.update_ext_sub(
                                        sub_path, lang_tag,
                                    )
                                new_already_labeled += 1
                                continue
                        sub_files.append(sub_path)

                now = time.monotonic()
                if now - last_report >= report_interval and self.config.show_details:
                    last_report = now
                    logger.info(
                        "Scanning... %d dirs, %d new videos, %d skipped, "
                        "%d new subs, %d sub-skipped",
                        dirs_scanned, len(video_files), video_skipped,
                        len(sub_files) + new_already_labeled, sub_skipped,
                    )

        except PermissionError as exc:
            logger.warning("Permission denied during scan: %s", exc)
        except Exception as exc:
            logger.error("Error during directory scan: %s", exc)

        # Flush tracked no_action_required entries (only on dirty; batched)
        if use_tracking and not self.config.dry_run:
            self.tracker.save_ext_sub_if_dirty()
        if not self.config.dry_run and hasattr(self, "language_index"):
            self.language_index.save_if_dirty()

        # Stash the seen-set so prune_missing_files can skip redundant I/O.
        self._last_scan_seen_paths = seen_paths

        if self.config.show_details:
            logger.info(
                "Scan complete: %d dirs, %d new videos (%d skipped), "
                "%d new subs (%d skipped, %d already labeled)",
                dirs_scanned, len(video_files), video_skipped,
                len(sub_files), sub_skipped, new_already_labeled,
            )
        else:
            parts = [
                f"{dirs_scanned} dirs",
                f"{len(video_files)} new videos",
            ]
            if video_skipped:
                parts.append(f"{video_skipped} skipped")
            if scan_subs:
                parts.append(f"{len(sub_files) + new_already_labeled} new subs")
                if sub_skipped:
                    parts.append(f"{sub_skipped} sub-skipped")
            print("\rScan complete: " + ", ".join(parts) + "          ")

        return (
            video_files, sub_files,
            video_skipped, sub_skipped,
            new_already_labeled, dirs_scanned,
        )

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
        cache_key = str(file_path)
        cached = self._mkv_info_cache.get(cache_key)
        if cached is not None:
            return cached

        mkvmerge = find_executable("mkvmerge")
        if not mkvmerge:
            info = self._get_mkv_info_ffprobe(file_path)
        else:
            try:
                cmd = [mkvmerge, "-J", str(file_path)]
                r = subprocess.run(cmd, capture_output=True, text=True, check=True,
                                   encoding="utf-8", errors="replace")
                data = json.loads(r.stdout)
                info = self._convert_mkvmerge_to_ffprobe(data)
            except Exception:
                info = self._get_mkv_info_ffprobe(file_path)

        self._mkv_info_cache[cache_key] = info
        return info

    def get_file_duration(self, file_path: Path) -> float:
        """Return duration in seconds, memoized per file."""
        cache_key = str(file_path)
        cached = self._duration_cache.get(cache_key)
        if cached is not None:
            return cached
        duration = audio_mod._get_file_duration(
            self.ffprobe, file_path, self.config.show_details,
        )
        self._duration_cache[cache_key] = duration
        return duration

    def _invalidate_file_caches(self, *paths: Path) -> None:
        """Drop cached metadata for *paths*.  Called at the end of
        process_file() and after remuxing, when a file's metadata
        is no longer needed or when the file has been replaced."""
        for p in paths:
            if p is None:
                continue
            key = str(p)
            self._mkv_info_cache.pop(key, None)
            self._duration_cache.pop(key, None)
            self._format_lang_cache.pop(key, None)

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
        cache_key = str(file_path)
        cached = self._format_lang_cache.get(cache_key)
        if cached is not None:
            return cached

        from uldas.constants import LANGUAGE_CODES
        try:
            cmd = [self.ffprobe, "-v", "quiet", "-print_format", "json",
                   "-show_format", str(file_path)]
            r = subprocess.run(cmd, capture_output=True, text=True, check=True,
                               encoding="utf-8", errors="replace")
            tags = json.loads(r.stdout).get("format", {}).get("tags", {})
        except Exception:
            self._format_lang_cache[cache_key] = {}
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
        self._format_lang_cache[cache_key] = result
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
    def find_audio_tracks_by_language(self, fp, code):
        return [t for t in self._find_tracks(fp, "audio", False) if t[3] == code]
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
                # Keep the language index in sync with the rename.
                if not self.config.dry_run and hasattr(self, "language_index"):
                    self.language_index.rename_ext_sub(sub_path, new_path, code)
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

        duration = self.get_file_duration(file_path)
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
            bypass = (
                self.config.reprocess_all
                or self.config.reprocess_all_subtitles
                or bool(self.config.reprocess_language)
            )
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
                # Original file is gone/remuxed — its cached metadata
                # no longer applies.
                self._invalidate_file_caches(file_path)
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
        if self.config.reprocess_language:
            # One-shot mode: only look at audio tracks currently tagged
            # with the requested language code, and leave subtitle tracks
            # untouched regardless of the "Process Subtitles" setting.
            skip_subs = True
        if not skip_audio:
            if self.config.reprocess_language:
                if self._reprocess_lang_force_all_audio:
                    # Index-driven fast path: files are already targetted, so reprocess all given files
                    tracks = self.find_all_audio_tracks(mkv_path)
                else:
                    tracks = self.find_audio_tracks_by_language(
                        mkv_path, self.config.reprocess_language,
                    )
            elif self.config.reprocess_all:
                tracks = self.find_all_audio_tracks(mkv_path)
            else:
                tracks = self.find_undefined_audio_tracks(mkv_path)

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
            if self.config.reprocess_language:
                touched = bool(results["processed_tracks"]
                               or results["failed_tracks"])
                if touched:
                    self._update_tracker_relabel(mkv_path, abs_key,
                                                 audio_ok, results)
            else:
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
                    if any(t["detected_language"] == "zxx" for t in results["processed_tracks"]):
                        flags.append("silent_content")
                subtitle_count = 0
                sub_res = results.get("subtitle_results")
                if sub_res and sub_res.get("processed_subtitle_tracks"):
                    subtitle_count = len(sub_res["processed_subtitle_tracks"])
                    flags.append("subtitle_labeled")
                if not flags:
                    flags.append("no_action_required")
                track_details = results["processed_tracks"] or None
                subtitle_details = None
                if sub_res and sub_res.get("processed_subtitle_tracks"):
                    subtitle_details = sub_res["processed_subtitle_tracks"]
                original_format = file_path.suffix if results["was_remuxed"] else None
                self.tracker.mark_processed(
                    mkv_path, audio_ok, subtitle_ok,
                    key=abs_key, flags=flags,
                    audio_tracks_labeled=audio_count,
                    subtitle_tracks_labeled=subtitle_count,
                    track_details=track_details,
                    subtitle_details=subtitle_details,
                    original_format=original_format,
                )

        # ── Language index (incremental) ────────────────────────────────
        if not self.config.dry_run and hasattr(self, "language_index"):
            try:
                # Invalidate the per-file ffprobe cache so the re-probe
                # reflects any language tags we just wrote.
                self._invalidate_file_caches(mkv_path)
                from uldas.language_index import _probe_track_langs
                audio_codes, sub_codes = _probe_track_langs(
                    self.ffprobe, mkv_path, mkvmerge=self.mkvmerge,
                )
                self.language_index.update_file(mkv_path, audio_codes, sub_codes)
                if results["was_remuxed"] and str(file_path) != str(mkv_path):
                    # Original file no longer exists after a successful remux.
                    self.language_index.remove_file(file_path)
            except Exception:
                logger.debug("Language-index update failed for %s",
                             mkv_path, exc_info=True)

        # Clean up memory between files
        audio_mod._cleanup_cuda_memory()

        # Drop per-file metadata caches so they don't grow unbounded.
        self._invalidate_file_caches(file_path, mkv_path)

        return results

    # ── Tracker update for the by-language reprocess mode ────────────────
    def _update_tracker_relabel(self, mkv_path: Path, abs_key: str,
                                audio_ok: bool, results: Dict) -> None:
        """Merge a by-language re-detection result into the existing entry."""
        existing = self.tracker.get_entry(mkv_path, key=abs_key) or {}

        old_details = list(existing.get("track_details") or [])
        new_details = results["processed_tracks"] or []
        new_idxs = {t["track_index"] for t in new_details}
        merged_details = [t for t in old_details
                          if t.get("track_index") not in new_idxs]
        merged_details.extend(new_details)
        merged_details.sort(key=lambda t: t.get("track_index", 0))

        prior_flags = list(existing.get("flags") or [])
        # Strip silent_content / no_action_required; we recompute those.
        flags = [f for f in prior_flags
                 if f not in ("silent_content", "no_action_required")]
        if any(t.get("detected_language") == "zxx" for t in merged_details):
            if "silent_content" not in flags:
                flags.append("silent_content")
        if not flags:
            flags.append("no_action_required")

        # Preserve prior counts: relabels do not add to the lifetime total.
        audio_count = int(existing.get("audio_tracks_labeled", 0))
        if audio_count == 0 and "audio_labeled" not in prior_flags and new_details:
            audio_count = sum(
                1 for t in new_details if t.get("detected_language") != "zxx"
            )
            if audio_count and "audio_labeled" not in flags:
                flags.append("audio_labeled")
        subtitle_count = int(existing.get("subtitle_tracks_labeled", 0))

        original_format = existing.get("original_format")
        subtitle_details = existing.get("subtitle_details")
        prior_audio = bool(existing.get("audio_processed", False))
        prior_sub = bool(existing.get("subtitle_processed", False))
        audio_success = prior_audio or audio_ok

        self.tracker.mark_processed(
            mkv_path,
            audio_success,
            prior_sub,
            key=abs_key,
            flags=flags,
            audio_tracks_labeled=audio_count,
            subtitle_tracks_labeled=subtitle_count,
            track_details=merged_details or None,
            subtitle_details=subtitle_details,
            original_format=original_format,
        )

    # ── Process explicit file list (no directory walk) ───────────────────
    def process_files(self, file_paths: List[Path]) -> List[Dict]:
        """Process a pre-built list of video files, bypassing the
        directory walk entirely.
        Used by the reprocess-by-language fast path.
        """
        results: List[Dict] = []

        if self._should_cancel() or not file_paths:
            return results

        ignore_tags = [t.lower() for t in (self.config.ignore_tags or [])
                       if isinstance(t, str) and t]

        actionable: List[Path] = []
        missing = 0
        ignored = 0
        for fp in file_paths:
            try:
                if not fp.exists():
                    missing += 1
                    continue
            except OSError:
                missing += 1
                continue
            if ignore_tags:
                stem = fp.stem.lower()
                if any(tag in stem for tag in ignore_tags):
                    ignored += 1
                    continue
            actionable.append(fp)

        if missing or ignored:
            logger.info(
                "process_files filtering: %d missing, %d ignored (ignore_tags), "
                "%d to process",
                missing, ignored, len(actionable),
            )

        prev_force = self._reprocess_lang_force_all_audio
        if self.config.reprocess_language:
            self._reprocess_lang_force_all_audio = True
        try:
            if actionable:
                results = self._process_video_files(actionable)
        finally:
            self._reprocess_lang_force_all_audio = prev_force
            if self.config.use_tracking and hasattr(self, "tracker"):
                self.tracker.save_if_dirty()
            if hasattr(self, "language_index"):
                self.language_index.save_if_dirty()

        return results

    # ── Process directory ────────────────────────────────────────────────
    def process_directory(self, directory: str) -> Tuple[List[Dict], List[Dict], int, int]:
        video_results = []
        ext_sub_results = []
        total_ext_subs_found = 0
        new_ext_subs = 0

        if self._should_cancel():
            return video_results, ext_sub_results, total_ext_subs_found, new_ext_subs

        try:
            # ── Unified scan (videos + external subtitles in one walk) ──
            (
                video_files, ext_sub_files,
                _video_skipped, sub_skipped,
                new_already_labeled, _dirs,
            ) = self._scan_tree(directory)

            total_ext_subs_found = len(ext_sub_files) + new_already_labeled + sub_skipped
            new_ext_subs = len(ext_sub_files) + new_already_labeled

            # Prune tracker entries for files that no longer exist
            if (self.config.use_tracking and hasattr(self, "tracker")
                    and not self.config.dry_run):
                on_vid_remove = None
                on_sub_remove = None
                if hasattr(self, "language_index"):
                    on_vid_remove = self.language_index.remove_file
                    on_sub_remove = self.language_index.remove_ext_sub
                pruned = self.tracker.prune_missing_files(
                    directory,
                    seen_paths=getattr(self, "_last_scan_seen_paths", None),
                    on_remove_video=on_vid_remove,
                    on_remove_ext_sub=on_sub_remove,
                )
                if pruned and self.config.show_details:
                    logger.info("Pruned %d orphaned tracker entries", pruned)

            if not video_files:
                if self.config.show_details:
                    logger.info("No new video files found in: %s", directory)
                else:
                    print(f"No new video files found in: {directory}")
            else:
                video_results = self._process_video_files(video_files)

            # ── External subtitle files ──────────────────────────────────
            if self.config.process_external_subtitles:
                if not ext_sub_files:
                    if self.config.show_details:
                        logger.info("No new external subtitle files found in: %s", directory)
                    else:
                        print(f"No new external subtitle files found in: {directory}")
                else:
                    ext_sub_results = self._process_ext_sub_files(ext_sub_files)
        finally:
            # Flush any pending tracker changes for this directory.
            # Writes are batched per-directory for performance; this
            # also runs on exceptions so partial progress is saved.
            if self.config.use_tracking and hasattr(self, "tracker"):
                self.tracker.save_if_dirty()
            if hasattr(self, "language_index"):
                self.language_index.save_if_dirty()

        return video_results, ext_sub_results, total_ext_subs_found, new_ext_subs

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
            bypass = (
                self.config.reprocess_all
                or self.config.reprocess_all_subtitles
                or bool(self.config.reprocess_language)
            )
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
        checkpoint_every = 25  # flush tracker to disk every N files
        for action_idx, fp in enumerate(actionable, 1):
            if self._should_cancel():
                break
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

            if action_idx % checkpoint_every == 0:
                if self.config.use_tracking and hasattr(self, "tracker"):
                    self.tracker.save_if_dirty()
                if hasattr(self, "language_index"):
                    self.language_index.save_if_dirty()

        return results

    def _process_ext_sub_files(self, ext_sub_files: List[Path]) -> List[Dict]:
        """Process a list of external subtitle files."""
        results = []
        total = len(ext_sub_files)

        for idx, sub_path in enumerate(ext_sub_files, 1):
            if self._should_cancel():
                break
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