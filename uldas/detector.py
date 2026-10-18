#file: uldas/detector.py

import gc
import json
import math
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
from uldas.constants import (
    LANGUAGE_CODES, EXTERNAL_SUBTITLE_EXTENSIONS, MP4_EXTENSIONS,
    DISC_STRUCTURE_DIRS, in_disc_structure, scan_video_extensions,
)
from uldas.tracking import ProcessingTracker
from uldas.tools import find_executable
from uldas.utils import (
    cgroup_cpu_quota,
    setup_cpu_limits,
    limit_subprocess_resources,
    normalize_language_code,
    mp4_language_code,
)
from uldas import audio as audio_mod
from uldas import subtitles as sub_mod
from uldas import external_subtitles as ext_sub_mod
from uldas import mp4 as mp4_mod

logger = logging.getLogger(__name__)

# ── Remux stream handling (ffprobe codec names) ──────────────────────────
_TS_SUFFIXES = (".m2ts", ".mts", ".ts")
# In-progress remux output; renamed to .mkv once verified.  Never scanned.
PARTIAL_SUFFIX = ".uldas-partial"
# Subtitle codecs Matroska stores as-is.
_MKV_COPY_SUBS = {
    "subrip", "srt", "ass", "ssa", "webvtt", "text",
    "hdmv_pgs_subtitle", "dvd_subtitle", "dvb_subtitle",
}
# Text subtitles Matroska can't hold natively; converted to SRT.
_MKV_SRT_SUBS = {"mov_text"}
# Disc PCM flavours Matroska can't hold; converted losslessly to FLAC.
_MKV_FLAC_AUDIO = {"pcm_bluray", "pcm_dvd"}


def _available_cpu_count() -> int:
    """Cores this process may run on (after ``setup_cpu_limits``), capped
    at the container's CPU quota so threads aren't throttled."""
    try:
        count = len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        try:
            import psutil
            count = len(psutil.Process().cpu_affinity())
        except Exception:
            count = os.cpu_count() or 4
    quota = cgroup_cpu_quota()
    if quota:
        count = min(count, math.ceil(quota))
    return max(1, count)


def _cuda_device_name(index: int = 0) -> Optional[str]:
    """Name of CUDA device *index* (e.g. "NVIDIA GeForce RTX 2060"), or
    None if it can't be read.  Asks the CUDA driver CTranslate2 itself
    uses, so numbering matches it (CUDA_VISIBLE_DEVICES included)."""
    try:
        import ctypes
        lib = (ctypes.WinDLL("nvcuda.dll") if sys.platform == "win32"
               else ctypes.CDLL("libcuda.so.1"))
        dev = ctypes.c_int()
        name = ctypes.create_string_buffer(256)
        if (lib.cuInit(0) or lib.cuDeviceGet(ctypes.byref(dev), index)
                or lib.cuDeviceGetName(name, len(name), dev)):
            return None
        return name.value.decode(errors="replace").strip() or None
    except Exception:
        return None


def _unlink_quiet(path: Optional[Path]) -> None:
    try:
        if path is not None and path.exists():
            path.unlink()
    except OSError:
        pass


def _audio_track_details(results: Dict) -> list:
    """Tracker ``track_details`` for one processed file: labeled tracks
    plus inconclusive ones (kept as 'und' / their previous tag)."""
    details = list(results["processed_tracks"])
    details += [
        {
            "track_index": t["track_index"],
            "detected_language": "und",
            "previous_language": t["previous_language"],
            "inconclusive": True,
        }
        for t in results.get("inconclusive_tracks", [])
    ]
    details.sort(key=lambda t: t["track_index"])
    return details


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
        self.last_remux_error: Optional[str] = None
        self._cancel_check = cancel_check
        self._cancel_logged = False

        # Per-file metadata caches, cleared at the start of each
        # process_file() call.  ffprobe/mkvmerge are expensive enough
        # that memoizing within a file's processing lifetime is worth
        # it — several code paths query the same metadata.
        self._mkv_info_cache: Dict[str, Dict] = {}
        self._duration_cache: Dict[str, float] = {}
        self._format_lang_cache: Dict[str, Dict[int, str]] = {}
        self._speech_cache: Dict[str, Optional[List[Tuple[float, float]]]] = {}

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

        removed = self.language_index.prune_ignored_tags(
            config.ignore_tags,
            match_dirs=bool(config.ignore_tags_match_dirs),
            roots=list(config.path or []),
        )
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
        # faster-whisper treats 0 as "4 threads", not "all cores".
        cpu_threads = (config.cpu_threads if config.cpu_threads > 0
                       else _available_cpu_count())
        self._cpu_threads = cpu_threads

        if config.show_details:
            logger.info("Initializing faster-whisper: device=%s, compute=%s, "
                        "model=%s, cpu_threads=%d",
                        device, compute_type, config.whisper_model, cpu_threads)

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
            try:
                return self._load_whisper(device, compute_type, cpu_threads)
            except Exception as exc:
                # A cached model.bin that is really a symlink stub (see
                # _find_stub_model_cache) makes every load attempt fail
                # with a nonsense binary version.  Wipe it and re-download
                # once before giving up.
                if not self._repair_stub_model_cache(exc):
                    raise
                return self._load_whisper(device, compute_type, cpu_threads)
        except Exception as exc:
            raise RuntimeError(f"Failed to initialise Whisper: {exc}") from exc
        finally:
            stop.set()

    def _load_whisper(self, device, compute_type, cpu_threads):
        # Requested setup first; on a GPU failure try CUDA int8 (works on
        # cards without float16 support) before falling back to CPU int8.
        attempts = [(device, compute_type)]
        if device == "cuda" and compute_type != "int8":
            attempts.append(("cuda", "int8"))
        if (device, compute_type) != ("cpu", "int8"):
            attempts.append(("cpu", "int8"))

        last_exc = None
        for n, (dev, ct) in enumerate(attempts):
            if n:
                logger.warning("Whisper init on %s/%s failed (%s) – trying %s/%s",
                               *attempts[n - 1], last_exc, dev, ct)
            try:
                model = WhisperModel(
                    self.config.whisper_model,
                    device=dev,
                    compute_type=ct,
                    cpu_threads=cpu_threads,
                    download_root=None,
                    local_files_only=False,
                )
            except Exception as exc:
                last_exc = exc
                continue
            msg = f"Whisper model '{self.config.whisper_model}' running on {dev} ({ct})"
            if dev == "cuda":
                gpu = _cuda_device_name()
                if gpu:
                    msg += f" – {gpu}"
            if self.config.show_details:
                logger.info(msg)
            else:
                print(msg, flush=True)
            return model
        raise last_exc

    def _find_stub_model_cache(self) -> Optional[Path]:
        """Return the Hugging Face cache dir of the configured model if its
        ``model.bin`` is an ``XSym`` stub instead of real weights.

        The HF cache links ``snapshots/<rev>/model.bin`` to ``blobs/<sha>``.
        On a bind mount that can't hold symlinks Docker writes a small text
        file starting with ``XSym`` in its place; read back through a
        different mount that stub is served as a plain file and CTranslate2
        reports an absurd "model binary version".
        """
        name = self.config.whisper_model
        if os.path.isdir(name):
            return None            # local model directory, not the HF cache
        try:
            from faster_whisper.utils import _MODELS
            repo_id = _MODELS.get(name, name)
        except Exception:
            repo_id = name
        if "/" not in repo_id:
            repo_id = f"Systran/faster-whisper-{repo_id}"
        try:
            from huggingface_hub.constants import HF_HUB_CACHE
            cache_root = Path(HF_HUB_CACHE)
        except Exception:
            cache_root = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")) / "hub"
        model_dir = cache_root / f"models--{repo_id.replace('/', '--')}"
        for model_bin in model_dir.glob("snapshots/*/model.bin"):
            try:
                with open(model_bin, "rb") as fh:
                    if fh.read(4) == b"XSym":
                        return model_dir
            except OSError:
                continue
        return None

    def _repair_stub_model_cache(self, exc: Exception) -> bool:
        """If *exc* came from a stub ``model.bin``, delete that model's
        cache dir so the next load re-downloads it.  Returns True when a
        retry is worthwhile."""
        if "model binary version" not in str(exc):
            return False
        model_dir = self._find_stub_model_cache()
        if model_dir is None:
            return False
        logger.warning(
            "Cached Whisper model at %s is a symlink stub (XSym), not real "
            "weights — the cache volume was written through a mount that "
            "can't store symlinks. Deleting it and re-downloading. If this "
            "repeats, set HF_HUB_DISABLE_SYMLINKS=1 in the container "
            "environment.", model_dir,
        )
        try:
            import shutil
            shutil.rmtree(model_dir)
        except OSError as rm_exc:
            logger.error("Could not delete %s: %s", model_dir, rm_exc)
            return False
        return True

    # ── Device helpers ───────────────────────────────────────────────────
    def _determine_device(self):
        if self.config.device != "auto":
            return self.config.device
        # CTranslate2 is the inference backend (CPU / CUDA only), so ask it
        # directly. Any failure (missing driver libs, no NVIDIA runtime) → CPU.
        try:
            import ctranslate2
            if ctranslate2.get_cuda_device_count() > 0:
                return "cuda"
        except Exception:
            pass
        return "cpu"

    def _determine_compute_type(self, device):
        if self.config.compute_type != "auto":
            return self.config.compute_type
        if device != "cuda":
            return "int8"
        # float16 needs compute capability >= 7.0; Pascal cards (GTX 10xx,
        # Quadro P-series) reject it but run int8 fine.
        try:
            import ctranslate2
            supported = ctranslate2.get_supported_compute_types("cuda")
            for ct in ("float16", "int8_float16", "int8", "float32"):
                if ct in supported:
                    return ct
        except Exception:
            pass
        return "float16"

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
        video_exts = scan_video_extensions(
            self.config.remux_to_mkv, self.config.mp4_support,
        )
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

        # Lowercase the ignore-tag list once; matched against filename
        # stems and (when enabled) directory names in the walk below.
        tags_lower = [t.lower() for t in (self.config.ignore_tags or [])
                      if isinstance(t, str) and t]
        match_dirs = bool(self.config.ignore_tags_match_dirs and tags_lower)

        video_files: list[Path] = []
        sub_files: list[Path] = []
        seen_paths: set[str] = set()

        video_skipped = 0
        sub_skipped = 0
        new_already_labeled = 0
        dirs_scanned = 0
        dirs_skipped = 0
        last_report = time.monotonic()
        report_interval = 5.0

        if self.config.show_details:
            logger.info(
                "Scanning directory tree: %s (video exts: %s)",
                directory, ", ".join(sorted(video_exts)),
            )
        else:
            print(f"Scanning directory tree: {directory}", flush=True)

        # os.walk silently skips directories it can't list; record that so
        # the files under them aren't mistaken for deleted ones.
        walk_errors: list = []

        def _walk_error(exc: OSError) -> None:
            walk_errors.append(exc)
            logger.warning("Could not list %s: %s",
                           getattr(exc, "filename", "?"), exc)

        try:
            for dirpath, dirnames, filenames in os.walk(
                    directory, followlinks=False, onerror=_walk_error):
                dirs_scanned += 1

                # Prune ignored directories (and Blu-ray / DVD folder
                # structures) in place so os.walk never descends into them.
                keep = [d for d in dirnames
                        if d.lower() not in DISC_STRUCTURE_DIRS
                        and not (match_dirs
                                 and any(tag in d.lower() for tag in tags_lower))]
                dirs_skipped += len(dirnames) - len(keep)
                dirnames[:] = keep

                for filename in filenames:
                    dot_pos = filename.rfind(".")
                    if dot_pos <= 0:
                        continue
                    ext = filename[dot_pos:].lower()

                    is_video = ext in video_exts
                    is_sub = scan_subs and ext in sub_exts
                    if not (is_video or is_sub):
                        continue

                    if tags_lower:
                        stem_lower = filename[:dot_pos].lower()
                        if any(tag in stem_lower for tag in tags_lower):
                            if is_video:
                                video_skipped += 1
                            else:
                                sub_skipped += 1
                            continue

                    full_path_str = os.path.join(dirpath, filename)
                    abs_path_str = os.path.abspath(full_path_str)
                    seen_paths.add(abs_path_str)

                    if is_video:
                        if (use_fast_skip_video
                                and abs_path_str in tracked_video_keys
                                and self.tracker.is_unchanged(abs_path_str)):
                            video_skipped += 1
                            continue
                        # New, or replaced since it was tracked: the batch
                        # check drops the stale entry and requeues it.
                        video_files.append(Path(full_path_str))
                    else:  # is_sub
                        sub_path = Path(full_path_str)
                        if (use_fast_skip_sub
                                and abs_path_str in self.tracker.ext_sub_data
                                and ext_sub_mod.get_language_tag(sub_path) is not None):
                            # Tracked and tagged: skipped whether or not it
                            # changed, so don't spend a stat() finding out.
                            sub_skipped += 1
                            continue
                        if ext == ".sub" and ext_sub_mod.is_vobsub_data(sub_path):
                            # Binary half of a VobSub pair: handled (and
                            # renamed) together with its .idx.
                            continue
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

        except Exception as exc:
            logger.error("Error during directory scan: %s", exc)
            scan_complete = False
        else:
            scan_complete = not walk_errors

        # Flush tracked no_action_required entries (only on dirty; batched)
        if use_tracking and not self.config.dry_run:
            self.tracker.save_ext_sub_if_dirty()
        if not self.config.dry_run and hasattr(self, "language_index"):
            self.language_index.save_if_dirty()

        # Stash the seen-set (and which file types it covers) so
        # prune_missing_files can skip redundant I/O.
        self._last_scan_seen_paths = seen_paths
        self._last_scan_exts = set(video_exts) | (set(sub_exts) if scan_subs else set())
        self._last_scan_complete = scan_complete

        if self.config.show_details:
            logger.info(
                "Scan complete: %d dirs (%d skipped), %d new videos (%d skipped), "
                "%d new subs (%d skipped, %d already labeled)",
                dirs_scanned, dirs_skipped, len(video_files), video_skipped,
                len(sub_files), sub_skipped, new_already_labeled,
            )
        else:
            parts = [
                f"{dirs_scanned} dirs",
            ]
            if dirs_skipped:
                parts.append(f"{dirs_skipped} dirs skipped")
            parts.append(f"{len(video_files)} new videos")
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
        """Remux *file_path* to an MKV next to it and delete the original.

        Returns the MKV path, or ``None`` on failure with the reason in
        ``self.last_remux_error``.  The original is only deleted once the
        MKV holds every audio and subtitle stream that was mapped; streams
        are never dropped to make a remux succeed.
        """
        self.last_remux_error = None
        if file_path.suffix.lower() == ".mkv":
            return file_path
        if in_disc_structure(file_path):
            # A piece of a Blu-ray / DVD folder rip, not a standalone video.
            self.last_remux_error = "part of a Blu-ray/DVD folder structure"
            logger.warning("Not remuxing %s: %s", file_path, self.last_remux_error)
            return None
        mkv_path = file_path.with_suffix(".mkv")
        if mkv_path.exists():
            # Never touch an MKV we didn't just create.
            self.last_remux_error = (
                f"{mkv_path.name} already exists – not overwriting it; if one "
                f"of the two copies is incomplete, remove that one")
            logger.warning("Not remuxing %s: %s", file_path.name,
                           self.last_remux_error)
            return None

        plan = self._plan_remux(file_path)
        if plan is None:
            self.last_remux_error = "could not read the source streams"
            logger.error("Not remuxing %s: %s", file_path.name,
                         self.last_remux_error)
            return None
        if plan["unsupported_subs"]:
            self.last_remux_error = (
                "subtitle codec(s) MKV can't store: "
                + ", ".join(sorted(set(plan["unsupported_subs"]))))
            logger.error("Not remuxing %s: %s", file_path.name,
                         self.last_remux_error)
            return None

        if self.config.dry_run:
            print(f"[DRY RUN] Would remux {file_path.name} → {mkv_path.name}")
            return mkv_path

        # ffmpeg writes to a name the scan never picks up; only a verified
        # result is renamed to .mkv.  An interrupted remux (docker stop,
        # crash, power cut) therefore never leaves a half-written MKV.
        partial = mkv_path.with_name(mkv_path.name + PARTIAL_SUFFIX)
        _unlink_quiet(partial)          # leftover of an interrupted attempt
        try:
            print(f"Remuxing {file_path.name} to MKV…")
            for strat in self._build_remux_strategies(file_path, partial, plan):
                try:
                    limited = limit_subprocess_resources(strat["args"])
                    subprocess.run(limited, check=True, capture_output=True,
                                   text=True, encoding="utf-8", errors="replace")
                except subprocess.CalledProcessError as exc:
                    logger.debug("Remux strategy '%s' failed for %s: %s",
                                 strat["name"], file_path.name,
                                 (exc.stderr or "").strip()[-500:])
                    _unlink_quiet(partial)
                    continue

                counts = self._stream_counts(partial)
                if (counts is None or not partial.exists()
                        or partial.stat().st_size <= 10_000):
                    _unlink_quiet(partial)
                    continue
                if counts != plan["expected"]:
                    logger.warning(
                        "Remux strategy '%s' for %s produced %d audio / %d "
                        "subtitle stream(s), expected %d / %d – discarding",
                        strat["name"], file_path.name,
                        counts["audio"], counts["subtitle"],
                        plan["expected"]["audio"], plan["expected"]["subtitle"],
                    )
                    _unlink_quiet(partial)
                    continue

                if mkv_path.exists():
                    self.last_remux_error = (f"{mkv_path.name} appeared during "
                                             f"the remux – not overwriting it")
                    logger.warning("Not remuxing %s: %s", file_path.name,
                                   self.last_remux_error)
                    return None
                os.replace(partial, mkv_path)
                if self.config.show_details:
                    logger.info("Remuxed with strategy '%s'", strat["name"])
                else:
                    print(f"Successfully remuxed to: {mkv_path.name}")
                self._remove_original(file_path, mkv_path)
                return mkv_path

            self.last_remux_error = "all remux strategies failed"
            logger.error("All remux strategies failed for %s", file_path)
            return None
        except Exception as exc:
            self.last_remux_error = str(exc)
            logger.error("Unexpected remux error: %s", exc)
            return None
        finally:
            # Also runs on SystemExit (SIGTERM) and KeyboardInterrupt.
            _unlink_quiet(partial)

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

    @staticmethod
    def _remux_probe_args(src: Path) -> list[str]:
        """Input probing options, shared by the stream probe and the remux
        itself so both see the same streams at the same indices."""
        if src.suffix.lower() in _TS_SUFFIXES:
            return ["-analyzeduration", "100M", "-probesize", "100M"]
        return []

    def _plan_remux(self, src: Path) -> Optional[dict]:
        """Work out which streams go into the MKV and how.

        Returns ``None`` if the source can't be probed.  ``expected`` holds
        the audio / subtitle stream counts the MKV must end up with;
        ``unsupported_subs`` lists subtitle codecs Matroska can't hold.
        """
        try:
            r = subprocess.run(
                [self.ffprobe, "-v", "quiet", *self._remux_probe_args(src),
                 "-print_format", "json", "-show_streams", str(src)],
                check=True, capture_output=True, text=True,
                encoding="utf-8", errors="replace",
            )
            streams = json.loads(r.stdout).get("streams", [])
        except Exception:
            return None

        maps: list[str] = []
        codecs: list[str] = []
        unsupported: list[str] = []
        n_audio = n_subs = 0
        for s in streams:
            ct = s.get("codec_type", "")
            cn = (s.get("codec_name") or "").lower()
            spec = f"0:{s.get('index', 0)}"
            if ct == "video":
                maps += ["-map", spec]
            elif ct == "audio":
                maps += ["-map", spec]
                if cn in _MKV_FLAC_AUDIO:
                    codecs += [f"-c:a:{n_audio}", "flac"]
                n_audio += 1
            elif ct == "subtitle":
                if cn in _MKV_SRT_SUBS:
                    maps += ["-map", spec]
                    codecs += [f"-c:s:{n_subs}", "srt"]
                    n_subs += 1
                elif cn in _MKV_COPY_SUBS:
                    maps += ["-map", spec]
                    n_subs += 1
                else:
                    unsupported.append(cn or "unknown")
        if not maps:
            return None
        return {
            "maps": maps,
            "codecs": codecs,
            "unsupported_subs": unsupported,
            "expected": {"audio": n_audio, "subtitle": n_subs},
        }

    def _stream_counts(self, path: Path) -> Optional[Dict[str, int]]:
        """Audio / subtitle stream counts of *path*, or None if unreadable."""
        try:
            r = subprocess.run(
                [self.ffprobe, "-v", "quiet", "-print_format", "json",
                 "-show_streams", str(path)],
                check=True, capture_output=True, text=True,
                encoding="utf-8", errors="replace",
            )
            streams = json.loads(r.stdout).get("streams", [])
        except Exception:
            return None
        return {
            "audio": sum(1 for s in streams if s.get("codec_type") == "audio"),
            "subtitle": sum(1 for s in streams
                            if s.get("codec_type") == "subtitle"),
        }

    def _build_remux_strategies(self, src: Path, dst: Path,
                                plan: dict) -> list[dict]:
        """Return the ffmpeg remux strategies to try in order.

        Every strategy carries all mapped streams; a result that lost any
        is rejected by the caller.
        """
        probe = self._remux_probe_args(src)
        streams = plan["maps"] + ["-c", "copy"] + plan["codecs"]

        strats: list[dict] = []
        if src.suffix.lower() in _TS_SUFFIXES:
            strats.append({
                "name": "m2ts_optimized",
                "args": [
                    self.ffmpeg, "-y", "-v", "warning",
                    "-fflags", "+genpts+discardcorrupt", *probe,
                    "-i", str(src), *streams,
                    "-avoid_negative_ts", "make_zero", "-map_metadata", "0",
                    "-f", "matroska", str(dst),
                ],
            })
        strats.append({
            "name": "selective_copy",
            "args": [
                self.ffmpeg, "-y", "-v", "warning", "-fflags", "+genpts",
                *probe, "-i", str(src), *streams,
                "-avoid_negative_ts", "make_zero", "-map_metadata", "0",
                "-f", "matroska", str(dst),
            ],
        })
        return strats

    # ── MKV info ─────────────────────────────────────────────────────────
    def get_mkv_info(self, file_path: Path) -> Dict:
        cache_key = str(file_path)
        cached = self._mkv_info_cache.get(cache_key)
        if cached is not None:
            return cached

        mkvmerge = find_executable("mkvmerge")
        if not mkvmerge or self._is_inplace_mp4(file_path):
            # For in-place MP4 work stick to ffprobe so stream indices and
            # codec names (e.g. mov_text) are the native ones the ffmpeg
            # extraction and the mp4 patcher both expect.
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
            self._speech_cache.pop(key, None)

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
            # IETF/BCP-47 overrides the legacy field (Matroska spec), and
            # the language index reads it the same way.
            lang = props.get("language_ietf") or props.get("language", "")
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
            duration=self.get_file_duration(file_path),
        )

    def _is_inplace_mp4(self, file_path: Path) -> bool:
        """True when *file_path* is an MP4/M4V that should be labeled in
        place (MP4 support on) rather than remuxed or skipped."""
        return bool(self.config.mp4_support) and mp4_mod.is_mp4(file_path)

    # ── Audio metadata update ────────────────────────────────────────────
    def update_mkv_language(self, file_path: Path, track_index: int,
                            language_code: str, dry_run: bool = False) -> bool:
        if dry_run:
            print(f"[DRY RUN] Would update track {track_index} → {language_code}")
            return True
        if self._is_inplace_mp4(file_path):
            lang3 = mp4_language_code(language_code)
            ok = mp4_mod.set_track_language(
                file_path, "audio", track_index, lang3,
                expected_count=len(self.find_all_audio_tracks(file_path)),
            )
            if ok and self.config.show_details:
                logger.info("Updated audio track %d → %s (MP4 in place)",
                            track_index, lang3)
            return ok
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
        """Detect and label embedded subtitle tracks.

        ``skipped`` (low confidence, no readable text, unsupported codec)
        and ``failed`` (extraction failed) outcomes are final for this
        version of the file: retrying gives the same result.  Only tracks
        in ``retry_subtitle_tracks`` (the write failed, or an unexpected
        error) keep the file queued for the next run.
        """
        results = {
            "subtitle_tracks_found": 0,
            "processed_subtitle_tracks": [],
            "failed_subtitle_tracks": [],
            "skipped_subtitle_tracks": [],
            "retry_subtitle_tracks": [],
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

        # MP4 can only take a language tag — no track name, no forced
        # flag — so forced/SDH analysis would be wasted effort there.
        is_mp4 = self._is_inplace_mp4(file_path)
        if is_mp4 and self.config.show_details and (
                self.config.analyze_forced_subtitles or self.config.detect_sdh_subtitles):
            logger.info("Skipping forced/SDH analysis for %s: MP4 only "
                        "supports a language tag", file_path.name)
        expected_sub_count = len(self.find_all_subtitle_tracks(file_path)) if is_mp4 else None
        set_forced = bool(self.config.analyze_forced_subtitles) and not is_mp4

        # Bitmap formats with no OCR path (VobSub, DVB) can't be read at
        # all: skip them without reading the file.
        readable = []
        for track in tracks:
            codec = track[1].get("codec_name")
            if sub_mod.is_unsupported_subtitle_codec(codec):
                results["skipped_subtitle_tracks"].append({
                    "track_index": track[0],
                    "detected_language": None,
                    "confidence": 0.0,
                    "reason": f"unsupported_codec ({codec})",
                })
            else:
                readable.append(track)
        if not readable:
            return results

        extracted = sub_mod.extract_subtitle_tracks(
            self.ffmpeg, file_path, [(t[0], t[2]) for t in readable],
            self.get_mkv_info, self.config.show_details,
        )
        try:
            for sub_idx, stream_info, stream_idx, cur_lang in readable:
                self._process_one_subtitle(
                    file_path, sub_idx, stream_info, stream_idx, cur_lang,
                    extracted.get(sub_idx), is_mp4, expected_sub_count,
                    set_forced, results,
                )
        finally:
            for path in extracted.values():
                _unlink_quiet(path)

        return results

    def _process_one_subtitle(self, file_path: Path, sub_idx: int,
                              stream_info: dict, stream_idx: int,
                              cur_lang: str, subtitle_path: Optional[Path],
                              is_mp4: bool, expected_sub_count: Optional[int],
                              set_forced: bool, results: Dict) -> None:
        """Detect, analyse and label one extracted subtitle track."""
        if not subtitle_path:
            logger.error("Could not extract subtitle track %d of %s",
                         sub_idx, file_path.name)
            results["failed_subtitle_tracks"].append(sub_idx)
            return
        try:
            lang_result = sub_mod.detect_subtitle_language(
                subtitle_path, file_path, sub_idx, stream_idx,
                self.ffmpeg, self.ffprobe, self.config.show_details,
            )
            if not lang_result:
                results["skipped_subtitle_tracks"].append({
                    "track_index": sub_idx,
                    "detected_language": None,
                    "confidence": 0.0,
                    "reason": "no_readable_text",
                })
                return

            code = lang_result["language_code"]
            conf = lang_result["confidence"]

            if conf < self.config.subtitle_confidence_threshold:
                results["skipped_subtitle_tracks"].append({
                    "track_index": sub_idx,
                    "detected_language": code,
                    "confidence": conf,
                    "reason": "confidence_below_threshold",
                })
                return

            is_forced = False
            if set_forced:
                is_forced = self._detect_forced(file_path, sub_idx, stream_idx, subtitle_path)

            is_sdh = False
            if (self.config.detect_sdh_subtitles and not is_mp4
                    and subtitle_path.suffix.lower() != ".sup"):
                is_sdh = sub_mod.detect_sdh_subtitles(subtitle_path)

            ok = sub_mod.update_subtitle_metadata(
                self.mkvpropedit, file_path, sub_idx, code,
                is_forced, is_sdh, self.config.dry_run, self.config.show_details,
                is_mp4=is_mp4, expected_count=expected_sub_count,
                set_forced=set_forced,
                current_name=(stream_info.get("tags") or {}).get("title"),
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
                results["retry_subtitle_tracks"].append(sub_idx)
        except Exception as exc:
            logger.error("Error processing subtitle track %d: %s", sub_idx, exc)
            results["failed_subtitle_tracks"].append(sub_idx)
            results["retry_subtitle_tracks"].append(sub_idx)

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
            is_text = sub_path.suffix.lower() not in (".sup", ".idx")
            content = None
            if is_text:
                # Read (and decode) once for both language and SDH detection.
                try:
                    content = ext_sub_mod.read_subtitle_text(sub_path)
                except OSError as exc:
                    logger.error("Could not read subtitle file %s: %s", sub_path, exc)
                    result["reason"] = "detection_failed"
                    return result

            lang_result = ext_sub_mod.detect_external_subtitle_language(
                sub_path, self.config.show_details, content=content,
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
            elif conf < self.config.subtitle_confidence_threshold:
                result["status"] = "skipped"
                result["reason"] = "confidence_below_threshold"
            if result["status"] == "skipped":
                # The same file gives the same answer next run: park it
                # until it changes instead of re-detecting it every run.
                if (self.config.use_tracking
                        and hasattr(self, "tracker")
                        and not self.config.dry_run):
                    self.tracker.mark_external_subtitle_skipped(
                        sub_path, result["reason"],
                    )
                return result

            # SDH detection for text-based subtitle files
            is_sdh = False
            if self.config.detect_sdh_subtitles and is_text:
                is_sdh = ext_sub_mod.detect_sdh_in_external_subtitle(
                    sub_path, content=content,
                )
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

    def _speech_intervals(self, file_path: Path) -> Optional[List[Tuple[float, float]]]:
        """Speech intervals (seconds) of the first audio track, from VAD.

        Memoized per file, so several ambiguous subtitle tracks share one
        decode.  Returns None if the audio can't be analysed.
        """
        key = str(file_path)
        if key in self._speech_cache:
            return self._speech_cache[key]

        intervals = None
        audio_tracks = self.find_all_audio_tracks(file_path)
        full_audio = None
        if audio_tracks:
            full_audio = audio_mod.extract_full_audio_track(
                self.ffmpeg, file_path, 0, audio_tracks[0][2],
                self.config.operation_timeout_seconds, self.config.show_details,
            )
        if full_audio:
            try:
                audio = audio_mod._decode_audio(full_audio)
                if audio is not None:
                    chunks = audio_mod._vad_speech(audio, self.config,
                                                   self.config.show_details)
                    if chunks is not None:
                        sr = audio_mod.SAMPLE_RATE
                        intervals = [(c["start"] / sr, c["end"] / sr)
                                     for c in chunks]
            finally:
                _unlink_quiet(full_audio)
        self._speech_cache[key] = intervals
        return intervals

    def _detect_forced(self, file_path, sub_idx, stream_idx, subtitle_path):
        """Forced detection dispatcher."""
        if subtitle_path.suffix.lower() == ".sup":
            return sub_mod.detect_forced_pgs_subtitles(
                self.ffprobe, file_path, sub_idx, self.config.show_details,
                sup_path=subtitle_path,
                duration=self.get_file_duration(file_path),
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

        # Low confidence → compare subtitle timing with where speech is.
        # Only the speech intervals are needed, which VAD gives directly
        # (a full Whisper transcription of the film would cost minutes).
        try:
            speech = self._speech_intervals(file_path)
        except Exception as exc:
            logger.debug("Speech analysis failed for %s: %s", file_path.name, exc)
            speech = None
        if speech is None:
            return stats["density"] < 5.5 or stats["coverage_percent"] < 37.5
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
        if self.config.show_details:
            logger.info("Forced check: subtitles cover %.0f%% of speech", pct)
        return pct < 50

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
            "inconclusive_tracks": [],
            "errors": [],
            "subtitle_results": None,
            "external_subtitle_results": None,
            "skipped_due_to_tracking": False,
        }

        abs_key = _cached_key or os.path.abspath(str(file_path))

        skip_audio = skip_subs = False
        had_entry = False
        if self.config.use_tracking and hasattr(self, "tracker"):
            bypass = (
                self.config.reprocess_all
                or self.config.reprocess_all_subtitles
                or bool(self.config.reprocess_language)
                or self.config.force_reprocess
            )
            entry = self.tracker.get_entry(file_path, key=abs_key)
            if entry and bypass:
                # Reprocess / force runs skip the batch check, so only
                # merge into the entry if the file is still the same one.
                had_entry = self.tracker.is_unchanged(abs_key)
            if not bypass:
                if entry:
                    # Unchanged (the batch check drops entries of changed
                    # files) but only partly done: finish the rest and
                    # merge into this entry rather than replacing it.
                    had_entry = True
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
        if (self.config.remux_to_mkv and file_path.suffix.lower() != ".mkv"
                and not self._is_inplace_mp4(file_path)):
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
            if mkv_path and mkv_path != file_path and self.config.dry_run:
                # Nothing was written: preview detection on the source.
                # The remux keeps audio / subtitle order, so track
                # numbers match the MKV it would produce.
                results["was_remuxed"] = True
                results["mkv_file"] = str(mkv_path)
                mkv_path = file_path
            elif mkv_path and mkv_path != file_path:
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
                reason = getattr(self, "last_remux_error", None)
                results["errors"].append(
                    f"Failed to remux: {reason}" if reason else "Failed to remux")
                return results
        results["mkv_file"] = results["mkv_file"] or str(mkv_path)

        audio_ok = subtitle_ok = False

        # ── Audio ────────────────────────────────────────────────────────
        if self.config.reprocess_language:
            # One-shot mode: only look at audio tracks currently tagged
            # with the requested language code, and leave subtitle tracks
            # untouched regardless of the "Process Subtitles" setting.
            skip_subs = True
        if not skip_audio:
            if self.config.reprocess_language:
                # Only the tracks currently tagged with the code, on both
                # the index-driven and the full-scan path.  If the index
                # was stale, the refresh at the end of this file fixes it.
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
                        if code == audio_mod.INCONCLUSIVE:
                            # Speech but no confident language: keep the
                            # current tag rather than guess.
                            results["inconclusive_tracks"].append({
                                "track_index": tidx,
                                "previous_language": cur_lang,
                            })
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

            # Skipped / unextractable tracks are final for this version of
            # the file; only write failures and unexpected errors retry.
            subtitle_ok = (
                not sub_res["retry_subtitle_tracks"]
                and not sub_res["subtitle_errors"]
            )
        elif skip_subs:
            subtitle_ok = True

        # ── Tracking ─────────────────────────────────────────────────────
        if (self.config.use_tracking and hasattr(self, "tracker")
                and not self.config.dry_run):
            # Re-detection runs merge into the existing entry so lifetime
            # label counts aren't inflated. A remuxed file has a new key
            # and is tracked like a fresh one.
            # A partly processed file (e.g. audio done, subtitles retried)
            # also merges, so the half that was skipped this run keeps its
            # flags, counts and track details.
            partial = had_entry and not results["was_remuxed"]
            merge_mode = (self.config.reprocess_language
                          or partial
                          or (self.config.reprocess_files
                              and not results["was_remuxed"]))
            if merge_mode:
                sub_res = results.get("subtitle_results")
                touched = partial or bool(
                    results["processed_tracks"]
                    or results["failed_tracks"]
                    or results["inconclusive_tracks"]
                    or (sub_res and (sub_res["processed_subtitle_tracks"]
                                     or sub_res["failed_subtitle_tracks"]
                                     or sub_res["skipped_subtitle_tracks"])))
                if touched:
                    self._update_tracker_relabel(mkv_path, abs_key,
                                                 audio_ok, results,
                                                 subtitle_ok=subtitle_ok)
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
                if results["inconclusive_tracks"]:
                    flags.append("inconclusive")
                subtitle_count = 0
                sub_res = results.get("subtitle_results")
                if sub_res and sub_res.get("processed_subtitle_tracks"):
                    subtitle_count = len(sub_res["processed_subtitle_tracks"])
                    flags.append("subtitle_labeled")
                if not flags:
                    flags.append("no_action_required")
                track_details = _audio_track_details(results) or None
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
                from uldas.language_index import (
                    _probe_track_langs, track_langs_from_info,
                )
                sub_res = results.get("subtitle_results") or {}
                wrote = bool(results["was_remuxed"]
                             or results["processed_tracks"]
                             or results["failed_tracks"]
                             or sub_res.get("processed_subtitle_tracks")
                             or sub_res.get("failed_subtitle_tracks")
                             or sub_res.get("retry_subtitle_tracks"))
                if wrote:
                    # Re-probe so the index reflects the tags just written.
                    self._invalidate_file_caches(mkv_path)
                    audio_codes, sub_codes = _probe_track_langs(
                        self.ffprobe, mkv_path, mkvmerge=self.mkvmerge,
                        mp4_inplace=bool(self.config.mp4_support),
                    )
                else:
                    # Nothing changed: the metadata read at the start of
                    # this file is still current.
                    audio_codes, sub_codes = track_langs_from_info(
                        self.get_mkv_info(mkv_path))
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
                                audio_ok: bool, results: Dict,
                                subtitle_ok: bool = False) -> None:
        """Merge a re-detection result into the existing entry."""
        sub_res = results.get("subtitle_results")
        self.tracker.merge_audio_relabel(
            mkv_path, _audio_track_details(results),
            audio_ok=audio_ok, key=abs_key,
            subtitle_details=(sub_res["processed_subtitle_tracks"]
                              if sub_res else None),
            subtitle_ok=subtitle_ok if sub_res else None,
        )

    # ── Process explicit file list (no directory walk) ───────────────────
    def process_files(self, file_paths: List[Path]) -> List[Dict]:
        """Process a pre-built list of video files, bypassing the
        directory walk entirely.
        Used by the reprocess-by-language fast path and the
        reprocess-specific-files action.
        """
        results: List[Dict] = []

        if self._should_cancel() or not file_paths:
            return results

        from uldas.language_index import filter_index_paths
        actionable, missing, ignored = filter_index_paths(file_paths, self.config)

        if missing or ignored:
            logger.info(
                "process_files filtering: %d missing, %d ignored (ignore_tags / "
                "unsupported container), %d to process",
                missing, ignored, len(actionable),
            )

        try:
            if actionable:
                results = self._process_video_files(actionable)
        finally:
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

            # Prune tracker entries for files that no longer exist.  Skipped
            # when the walk errored or saw nothing at all: an offline share
            # mounted as an empty folder must not wipe a library's history.
            seen = getattr(self, "_last_scan_seen_paths", None)
            complete = bool(getattr(self, "_last_scan_complete", False))
            prune_safe = complete and bool(seen)
            if not complete:
                logger.warning("Scan of %s was incomplete – not removing "
                               "tracking entries for missing files", directory)
            elif not seen:
                logger.info("No media files found in %s – not removing "
                            "tracking entries (is the share mounted?)", directory)
            if (prune_safe and self.config.use_tracking
                    and hasattr(self, "tracker")
                    and not self.config.dry_run):
                on_vid_remove = None
                on_sub_remove = None
                if hasattr(self, "language_index"):
                    on_vid_remove = self.language_index.remove_file
                    on_sub_remove = self.language_index.remove_ext_sub
                pruned = self.tracker.prune_missing_files(
                    directory,
                    seen_paths=seen,
                    on_remove_video=on_vid_remove,
                    on_remove_ext_sub=on_sub_remove,
                    extensions=getattr(self, "_last_scan_exts", None),
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
                res = self.process_file(fp, _cached_key=cached_key)
                results.append(res)
                # Notify external scan tools once the remuxed file is fully processed (language tags already re-applied above).
                if (res.get("was_remuxed") and not self.config.dry_run
                        and getattr(self.config, "webhook_enabled", False)):
                    try:
                        from uldas import webhook
                        webhook.send_remux_webhook(
                            self.config,
                            Path(res["mkv_file"]),
                            Path(res["original_file"]),
                        )
                    except Exception as exc:
                        logger.warning("Webhook dispatch failed for %s: %s",
                                       fp, exc)
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