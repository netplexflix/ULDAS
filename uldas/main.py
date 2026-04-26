#file: uldas/main.py

import argparse
import os
import sys
import time
import signal
import tempfile
import logging

from uldas.constants import VERSION
from uldas.config import Config
from uldas.logging_setup import setup_logging
from uldas.tools import find_executable, find_mkvtoolnix_installation
from uldas.tracking import ProcessingTracker
from uldas.updater import check_for_updates
from uldas.summary import print_detailed_summary
from uldas.scheduler_state import SchedulerState

logger = logging.getLogger(__name__)


def _setup_signal_handlers() -> None:
    """Install signal handlers so the process shuts down gracefully."""
    def _handler(signum, frame):
        sig_name = signal.Signals(signum).name if hasattr(signal, 'Signals') else str(signum)
        logger.warning("Received signal %s – shutting down gracefully", sig_name)
        print(f"\nReceived signal {sig_name} – shutting down...", flush=True)
        for handler in logging.getLogger().handlers:
            try:
                handler.flush()
            except Exception:
                pass
        sys.exit(128 + signum)

    for sig in (signal.SIGTERM, signal.SIGINT):
        try:
            signal.signal(sig, _handler)
        except (OSError, ValueError):
            pass


def _tune_third_party_logging(show_details: bool) -> None:
    """Silence noisy third-party loggers unless the user asked for details.

    huggingface_hub emits an 'unauthenticated requests' advisory when no
    HF_TOKEN is set. The Whisper models we use are public, so the warning
    is noise — hide it unless show_details is on.
    """
    import warnings
    if show_details:
        return
    try:
        from huggingface_hub.utils import logging as hf_logging
        hf_logging.set_verbosity_error()
    except Exception:
        logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
    warnings.filterwarnings(
        "ignore",
        message=r".*unauthenticated requests to the HF Hub.*",
    )


def _apply_temp_dir(config: Config) -> None:
    """Override the global tempfile directory if configured."""
    temp_dir = config.temp_dir
    if temp_dir:
        temp_dir = temp_dir.strip()
    if temp_dir:
        os.makedirs(temp_dir, exist_ok=True)
        tempfile.tempdir = temp_dir
        logger.info("Temporary directory set to: %s", temp_dir)
    else:
        logger.debug("Using default temporary directory: %s", tempfile.gettempdir())


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Detect and update language metadata for video file audio and subtitle tracks",
    )

    # ── Meta / utility commands ──────────────────────────────────────────
    p.add_argument("--version", action="version", version=f"%(prog)s {VERSION}")
    p.add_argument("--config", default="config/config.yml", help="Config file path")
    p.add_argument("--create-config", action="store_true", help="Create sample config")
    p.add_argument("--find-mkv", action="store_true", help="Locate MKVToolNix")
    p.add_argument("--clear-tracking", action="store_true",
                   help="Clear all tracking data and exit")
    p.add_argument("--skip-update-check", action="store_true",
                   help="Skip checking for updates on startup")

    # ── Logging ──────────────────────────────────────────────────────────
    p.add_argument("--verbose", "-v", action="store_true",
                   help="Enable verbose/debug output (sets show_details=True)")
    p.add_argument("--quiet", "-q", action="store_true",
                   help="Suppress most console output (sets show_details=False)")

    # ── Paths & general ──────────────────────────────────────────────────
    p.add_argument("--directory", nargs="+", metavar="DIR",
                   help="Override directory/directories to scan (config: path)")
    p.add_argument("--remux-to-mkv", action="store_true", default=None,
                   help="Remux non-MKV video files to MKV before processing (config: remux_to_mkv)")
    p.add_argument("--no-remux-to-mkv", action="store_true", default=None,
                   help="Disable remuxing non-MKV files to MKV")
    p.add_argument("--show-details", action="store_true", default=None,
                   help="Show detailed processing information (config: show_details)")
    p.add_argument("--no-show-details", action="store_true", default=None,
                   help="Hide detailed processing information")
    p.add_argument("--model", choices=["tiny", "base", "small", "medium", "large"],
                   help="Whisper model size (config: whisper_model)")
    p.add_argument("--dry-run", action="store_true", default=None,
                   help="Simulate changes without modifying any files (config: dry_run)")
    p.add_argument("--temp-dir", metavar="DIR",
                   help="Override temporary directory (config: temp_dir)")

    # ── VAD ──────────────────────────────────────────────────────────────
    p.add_argument("--no-vad", action="store_true", default=None,
                   help="Disable VAD (Voice Activity Detection) filter (config: vad_filter)")
    p.add_argument("--vad", action="store_true", default=None,
                   help="Enable VAD filter (default)")
    p.add_argument("--vad-min-speech-duration-ms", type=int, metavar="MS",
                   help="Minimum speech duration in ms for VAD (config: vad_min_speech_duration_ms)")
    p.add_argument("--vad-max-speech-duration-s", type=int, metavar="S",
                   help="Maximum speech duration in seconds for VAD (config: vad_max_speech_duration_s)")

    # ── Device / compute ─────────────────────────────────────────────────
    p.add_argument("--device", choices=["auto", "cpu", "cuda"],
                   help="Device for Whisper inference (config: device)")
    p.add_argument("--compute-type",
                   choices=["auto", "int8", "int8_float16", "int16", "float16", "float32"],
                   help="Compute type for Whisper inference (config: compute_type)")
    p.add_argument("--cpu-threads", type=int, metavar="N",
                   help="Number of CPU threads (0 = auto) (config: cpu_threads)")

    # ── Confidence / reprocessing ────────────────────────────────────────
    p.add_argument("--confidence-threshold", type=float, metavar="F",
                   help="Confidence threshold for audio language detection (0.0-1.0) (config: confidence_threshold)")
    p.add_argument("--reprocess-all", action="store_true", default=None,
                   help="Reprocess all audio tracks, even those already tagged (config: reprocess_all)")
    p.add_argument("--reprocess-language", metavar="CODE", default=None,
                   help="Re-detect only audio tracks currently tagged with "
                        "this ISO 639-3 code (e.g. 'zxx' for 'no speech', "
                        "'ja' for Japanese). Ignores tracking cache; skips "
                        "subtitle processing.")
    p.add_argument("--force-reprocess", action="store_true", default=None,
                   help="Force reprocessing, ignoring tracking cache (config: force_reprocess)")
    p.add_argument("--index-languages", action="store_true", default=None,
                   help="Walk every configured library, record the language "
                        "tag on every audio/subtitle track (internal + "
                        "external), and write counts to "
                        "config/language_index.json. No files are modified "
                        "and Whisper is not loaded.")

    # ── Tracking ─────────────────────────────────────────────────────────
    p.add_argument("--no-tracking", action="store_true", default=None,
                   help="Disable file processing tracking (config: use_tracking)")
    p.add_argument("--tracking", action="store_true", default=None,
                   help="Enable file processing tracking (default)")

    # ── Subtitle processing ──────────────────────────────────────────────
    p.add_argument("--process-subtitles", action="store_true", default=None,
                   help="Process embedded subtitle tracks (config: process_subtitles)")
    p.add_argument("--no-process-subtitles", action="store_true", default=None,
                   help="Disable processing of embedded subtitle tracks")
    p.add_argument("--process-external-subtitles", action="store_true", default=None,
                   help="Process external subtitle files (config: process_external_subtitles)")
    p.add_argument("--no-process-external-subtitles", action="store_true", default=None,
                   help="Disable processing of external subtitle files")
    p.add_argument("--analyze-forced", action="store_true", default=None,
                   help="Analyze and tag forced subtitles (config: analyze_forced_subtitles)")
    p.add_argument("--no-analyze-forced", action="store_true", default=None,
                   help="Disable forced subtitle analysis")
    p.add_argument("--detect-sdh", action="store_true", default=None,
                   help="Detect and tag SDH subtitles (default) (config: detect_sdh_subtitles)")
    p.add_argument("--no-sdh-detection", action="store_true", default=None,
                   help="Disable SDH subtitle detection")
    p.add_argument("--subtitle-confidence-threshold", type=float, metavar="F",
                   help="Confidence threshold for subtitle language detection (0.0-1.0) (config: subtitle_confidence_threshold)")
    p.add_argument("--reprocess-all-subtitles", action="store_true", default=None,
                   help="Reprocess all subtitle tracks, even those already tagged (config: reprocess_all_subtitles)")

    # ── Timeouts ─────────────────────────────────────────────────────────
    p.add_argument("--operation-timeout", type=int, metavar="S",
                   help="Timeout in seconds for long operations like full-track extraction (config: operation_timeout_seconds)")

    # ── Forced subtitle thresholds ───────────────────────────────────────
    p.add_argument("--forced-sub-low-coverage", type=float, metavar="F",
                   help="Low coverage threshold (%%) for forced subtitle detection (config: forced_subtitle_low_coverage_threshold)")
    p.add_argument("--forced-sub-high-coverage", type=float, metavar="F",
                   help="High coverage threshold (%%) for forced subtitle detection (config: forced_subtitle_high_coverage_threshold)")
    p.add_argument("--forced-sub-low-density", type=float, metavar="F",
                   help="Low density threshold for forced subtitle detection (config: forced_subtitle_low_density_threshold)")
    p.add_argument("--forced-sub-high-density", type=float, metavar="F",
                   help="High density threshold for forced subtitle detection (config: forced_subtitle_high_density_threshold)")
    p.add_argument("--forced-sub-min-count", type=int, metavar="N",
                   help="Minimum subtitle count threshold for forced detection (config: forced_subtitle_min_count_threshold)")
    p.add_argument("--forced-sub-max-count", type=int, metavar="N",
                   help="Maximum subtitle count threshold for forced detection (config: forced_subtitle_max_count_threshold)")

    return p


def _apply_cli_overrides(config: Config, args) -> None:
    """Apply CLI arguments on top of the (already loaded) config.

    Only values that were explicitly provided on the command line override
    the config-file / default values.  For boolean toggle pairs
    (e.g. ``--remux-to-mkv`` / ``--no-remux-to-mkv``) the "positive"
    flag wins if both are given.
    """

    # ── Paths & general ──────────────────────────────────────────────────
    if args.directory:
        config.path = args.directory

    if args.remux_to_mkv:
        config.remux_to_mkv = True
    elif args.no_remux_to_mkv:
        config.remux_to_mkv = False

    if args.show_details:
        config.show_details = True
    elif args.no_show_details:
        config.show_details = False

    if args.model:
        config.whisper_model = args.model

    if args.dry_run:
        config.dry_run = True

    if args.temp_dir:
        config.temp_dir = args.temp_dir

    # ── VAD ──────────────────────────────────────────────────────────────
    if args.no_vad:
        config.vad_filter = False
    elif args.vad:
        config.vad_filter = True

    if args.vad_min_speech_duration_ms is not None:
        config.vad_min_speech_duration_ms = args.vad_min_speech_duration_ms

    if args.vad_max_speech_duration_s is not None:
        config.vad_max_speech_duration_s = args.vad_max_speech_duration_s

    # ── Device / compute ─────────────────────────────────────────────────
    if args.device:
        config.device = args.device

    if args.compute_type:
        config.compute_type = args.compute_type

    if args.cpu_threads is not None:
        config.cpu_threads = args.cpu_threads

    # ── Confidence / reprocessing ────────────────────────────────────────
    if args.confidence_threshold is not None:
        config.confidence_threshold = args.confidence_threshold

    if args.reprocess_all:
        config.reprocess_all = True

    if args.reprocess_language:
        config.reprocess_language = args.reprocess_language.strip().lower()

    if args.force_reprocess:
        config.force_reprocess = True

    if args.index_languages:
        config.index_languages_only = True

    # ── Tracking ─────────────────────────────────────────────────────────
    if args.no_tracking:
        config.use_tracking = False
    elif args.tracking:
        config.use_tracking = True

    # ── Subtitle processing ──────────────────────────────────────────────
    if args.process_subtitles:
        config.process_subtitles = True
    elif args.no_process_subtitles:
        config.process_subtitles = False

    if args.process_external_subtitles:
        config.process_external_subtitles = True
    elif args.no_process_external_subtitles:
        config.process_external_subtitles = False

    if args.analyze_forced:
        config.analyze_forced_subtitles = True
    elif args.no_analyze_forced:
        config.analyze_forced_subtitles = False

    if args.detect_sdh:
        config.detect_sdh_subtitles = True
    elif args.no_sdh_detection:
        config.detect_sdh_subtitles = False

    if args.subtitle_confidence_threshold is not None:
        config.subtitle_confidence_threshold = args.subtitle_confidence_threshold

    if args.reprocess_all_subtitles:
        config.reprocess_all_subtitles = True

    # ── Timeouts ─────────────────────────────────────────────────────────
    if args.operation_timeout is not None:
        config.operation_timeout_seconds = args.operation_timeout

    # ── Forced subtitle thresholds ───────────────────────────────────────
    if args.forced_sub_low_coverage is not None:
        config.forced_subtitle_low_coverage_threshold = args.forced_sub_low_coverage

    if args.forced_sub_high_coverage is not None:
        config.forced_subtitle_high_coverage_threshold = args.forced_sub_high_coverage

    if args.forced_sub_low_density is not None:
        config.forced_subtitle_low_density_threshold = args.forced_sub_low_density

    if args.forced_sub_high_density is not None:
        config.forced_subtitle_high_density_threshold = args.forced_sub_high_density

    if args.forced_sub_min_count is not None:
        config.forced_subtitle_min_count_threshold = args.forced_sub_min_count

    if args.forced_sub_max_count is not None:
        config.forced_subtitle_max_count_threshold = args.forced_sub_max_count


def _build_run_summary(video_results: list, ext_sub_results: list,
                       runtime: float,
                       new_ext_subs: int = 0) -> dict:
    """Extract key stats from processing results for the web UI."""
    files_scanned = len(video_results)
    files_processed = 0
    files_failed = 0
    files_skipped = 0
    audio_tracks_labeled = 0
    subtitle_tracks_labeled = 0
    files_remuxed = 0

    for r in video_results:
        if r.get("skipped_due_to_tracking"):
            files_skipped += 1
            continue

        has_action = False
        if r.get("was_remuxed"):
            files_remuxed += 1
            has_action = True

        audio_tracks_labeled += len(r.get("processed_tracks", []))
        if r.get("processed_tracks"):
            has_action = True

        sr = r.get("subtitle_results")
        if sr:
            subtitle_tracks_labeled += len(sr.get("processed_subtitle_tracks", []))
            if sr.get("processed_subtitle_tracks"):
                has_action = True

        if r.get("failed_tracks") or r.get("errors"):
            files_failed += 1
            has_action = True

        if has_action:
            files_processed += 1

    ext_subs_processed = sum(
        1 for e in ext_sub_results if e.get("status") == "processed"
    )

    return {
        "runtime_seconds": round(runtime, 1),
        "files_scanned": files_scanned,
        "files_processed": files_processed,
        "files_skipped": files_skipped,
        "files_failed": files_failed,
        "files_remuxed": files_remuxed,
        "audio_tracks_labeled": audio_tracks_labeled,
        "subtitle_tracks_labeled": subtitle_tracks_labeled,
        "external_subs_processed": ext_subs_processed,
        "new_ext_subs": new_ext_subs,
    }


def _log_reprocess_language_discrepancy(language_index, code: str) -> None:
    """Warn when reprocess-by-language touched no tracks but the index
    still claims tracks of *code* exist. Lists up to 10 example paths so
    the discrepancy is visible in the run log without opening the UI.
    """
    snap = language_index.snapshot()
    per_file = snap.get("per_file") or {}
    matching = [path for path, info in per_file.items()
                if code in (info.get("audio") or [])]
    if not matching:
        return
    examples = sorted(matching)[:10]
    logger.warning(
        "Language index still lists %d file(s) with '%s' audio tracks "
        "after the reprocess-by-language run. If those tracks were "
        "expected to be re-detected, this points to ignore_tags "
        "filtering, mkvmerge/ffprobe disagreement, or a stale index. "
        "Examples:",
        len(matching), code,
    )
    for path in examples:
        logger.warning("  %s", path)
    if len(matching) > len(examples):
        logger.warning("  ... and %d more", len(matching) - len(examples))


def _path_under_configured(path: str, configured_paths: list) -> bool:
    """True iff *path* is under any of the configured library roots.

    Mirrors the prefix logic used by the language-index path helpers,
    so the fast path filters the same way ``prune_outside_paths`` does.
    """
    if not configured_paths:
        return False
    abs_path = os.path.abspath(path)
    cmp_path = abs_path if abs_path.endswith(os.sep) else abs_path + os.sep
    for cp in configured_paths:
        if not cp:
            continue
        root = os.path.abspath(cp)
        if not root.endswith(os.sep):
            root = root + os.sep
        if cmp_path.startswith(root):
            return True
    return False


def _try_reprocess_language_fast_path(detector, config: Config,
                                      all_video_results: list) -> bool:
    """Run reprocess-by-language using the language index (fast path).

    Returns ``True`` if the fast path ran (in which case the caller must
    skip the directory walk). Returns ``False`` to signal a fall-back
    to the existing full-library walk.
    """
    from pathlib import Path

    code = config.reprocess_language
    matched = detector.language_index.files_with_audio_language(code)
    if not matched:
        logger.warning(
            "Reprocess-by-language: language index has no '%s' entries "
            "(or index missing) — falling back to full library scan",
            code,
        )
        return False

    scoped = [p for p in matched
              if _path_under_configured(p, config.path)]
    out_of_scope = len(matched) - len(scoped)
    if not scoped:
        logger.warning(
            "Reprocess-by-language: %d index entr%s for '%s' lie outside "
            "configured paths — falling back to full library scan",
            len(matched), "y" if len(matched) == 1 else "ies", code,
        )
        return False

    existing = [Path(p) for p in scoped if os.path.exists(p)]
    missing = len(scoped) - len(existing)
    if not existing:
        logger.warning(
            "Reprocess-by-language: %d indexed file(s) for '%s' do not "
            "exist on disk — falling back to full library scan",
            len(scoped), code,
        )
        return False

    extras = []
    if missing:
        extras.append(f"{missing} indexed paths missing on disk, ignored")
    if out_of_scope:
        extras.append(f"{out_of_scope} outside configured paths, ignored")
    suffix = f" ({'; '.join(extras)})" if extras else ""
    logger.info(
        "Reprocess-by-language: %d file(s) matched in language index "
        "for '%s' — skipping full library scan%s",
        len(existing), code, suffix,
    )
    print(
        f"Reprocess-by-language: {len(existing)} file(s) matched in "
        f"language index for '{code}' — skipping full library scan",
        flush=True,
    )

    all_video_results.extend(detector.process_files(existing))
    return True


def _run_processing(config: Config, skip_update_check: bool = False,
                    state: "Optional[SchedulerState]" = None,
                    config_path: str = "config/config.yml") -> None:
    start = time.time()

    if not skip_update_check:
        check_for_updates()

    # Reload config from disk so web UI settings changes take effect
    config.load_from_file(config_path)
    config.reset_transient_options()
    if state is not None:
        for key, val in state.consume_run_options().items():
            if hasattr(config, key):
                setattr(config, key, val)
                logger.info("One-shot run option applied: %s=%s", key, val)

    _tune_third_party_logging(config.show_details)

    # ── Index-languages one-shot mode ────────────────────────────────────
    if config.index_languages_only:
        _apply_temp_dir(config)
        if not config.path:
            msg = "No directories configured"
            logger.error(msg)
            if state is not None:
                state.set_status("error", msg)
                return
            sys.exit(1)
        for d in config.path:
            if not os.path.isdir(d):
                msg = f"Directory not found: {d}"
                logger.error(msg)
                if state is not None:
                    state.set_status("error", msg)
                    return
                sys.exit(1)

        from uldas.language_index import build_language_index, INDEX_FILENAME

        out_path = os.path.join("config", INDEX_FILENAME)
        try:
            result = build_language_index(
                directories=list(config.path),
                output_path=out_path,
                include_non_mkv_video=bool(config.remux_to_mkv),
                ignore_tags=list(config.ignore_tags or []),
                cancel_check=(state.is_stopped if state is not None else None),
                show_details=config.show_details,
            )
        except Exception as exc:
            msg = f"Language indexing failed: {exc}"
            logger.error(msg, exc_info=True)
            if state is not None:
                state.set_status("error", msg)
                return
            sys.exit(1)

        runtime = time.time() - start
        print(f"\n{'=' * 60}")
        print("LANGUAGE INDEX COMPLETE")
        print(f"{'=' * 60}")
        print(f"  Videos indexed:            {result['video_files_indexed']}")
        print(f"  External subs indexed:     {result['external_sub_files_indexed']}")
        if result.get("files_skipped"):
            print(f"  Skipped (ignore_tags):     {result['files_skipped']}")
        print(f"  Unique audio languages:    {len(result['counts']['audio'])}")
        print(f"  Unique embedded sub langs: {len(result['counts']['embedded_subs'])}")
        print(f"  Unique external sub langs: {len(result['counts']['external_subs'])}")
        print(f"  Saved to:                  {out_path}")
        print(f"  Runtime:                   {runtime:.1f}s")

        if state is not None:
            state.set_last_run_summary({
                "runtime_seconds": round(runtime, 1),
                "index_run": True,
                "video_files_indexed": result["video_files_indexed"],
                "external_sub_files_indexed": result["external_sub_files_indexed"],
                "unique_audio_languages": len(result["counts"]["audio"]),
                "unique_embedded_sub_languages": len(result["counts"]["embedded_subs"]),
                "unique_external_sub_languages": len(result["counts"]["external_subs"]),
                "cancelled": result.get("cancelled", False),
            })
        return

    # Apply temp_dir setting
    _apply_temp_dir(config)

    # Validate directories
    if not config.path:
        msg = "No directories configured"
        logger.error(msg)
        if state is not None:
            state.set_status("error", msg)
            return
        sys.exit(1)

    for d in config.path:
        if not os.path.isdir(d):
            msg = f"Directory not found: {d}"
            logger.error(msg)
            if state is not None:
                state.set_status("error", msg)
                return
            sys.exit(1)

    # Check dependencies
    deps = {
        "ffmpeg": "FFmpeg (https://ffmpeg.org/download.html)",
        "ffprobe": "FFmpeg (https://ffmpeg.org/download.html)",
        "mkvpropedit": "MKVToolNix (https://mkvtoolnix.download/downloads.html)",
    }
    missing = [(n, s) for n, s in deps.items() if not find_executable(n)]
    if missing:
        msgs = [f"Missing: {n} – install from {s}" for n, s in missing]
        for m in msgs:
            logger.error(m)
        if state is not None:
            state.set_status("error", "; ".join(msgs))
            return
        sys.exit(1)

    # Import detector here (heavy import due to Whisper)
    from uldas.detector import MKVLanguageDetector

    try:
        if config.show_details:
            logger.info("Loading faster-whisper model: %s", config.whisper_model)
        else:
            print(f"Loading faster-whisper model: {config.whisper_model}")
        cancel_check = state.is_stopped if state is not None else None
        detector = MKVLanguageDetector(config, cancel_check=cancel_check)
    except RuntimeError as exc:
        msg = f"Failed to initialise detector: {exc}"
        logger.error(msg)
        if state is not None:
            state.set_status("error", msg)
            return
        sys.exit(1)

    if config.show_details:
        logger.info("Scanning directories: %s", config.path)
    else:
        print(f"Scanning directories: {config.path}")

    if config.dry_run:
        print("DRY RUN MODE – No files will be modified")

    all_video_results = []
    all_ext_sub_results = []
    total_ext_subs_found = 0
    total_new_ext_subs = 0
    try:
        fast_path_used = False
        if config.reprocess_language and hasattr(detector, "language_index"):
            fast_path_used = _try_reprocess_language_fast_path(
                detector, config, all_video_results,
            )

        if not fast_path_used:
            for d in config.path:
                if state is not None and state.is_stopped():
                    break
                video_results, ext_sub_results, dir_ext_subs_found, dir_new_ext_subs = detector.process_directory(d)
                all_video_results.extend(video_results)
                all_ext_sub_results.extend(ext_sub_results)
                total_ext_subs_found += dir_ext_subs_found
                total_new_ext_subs += dir_new_ext_subs

        if config.reprocess_language and hasattr(detector, "language_index"):
            try:
                _log_reprocess_language_discrepancy(
                    detector.language_index, config.reprocess_language,
                )
            except Exception:
                logger.debug("Reprocess discrepancy check failed",
                             exc_info=True)
    finally:
        # Final flush of batched tracker writes, even on exception.
        if config.use_tracking and hasattr(detector, "tracker"):
            detector.tracker.save_if_dirty()

    # Save failed files for the web UI
    if config.use_tracking and not config.dry_run:
        ProcessingTracker.save_failed_files_json(
            "config", all_video_results, all_ext_sub_results)

    runtime = time.time() - start
    print_detailed_summary(all_video_results, all_ext_sub_results,
                           config, runtime, detector,
                           total_ext_subs_found=total_ext_subs_found,
                           total_new_ext_subs=total_new_ext_subs)

    # Store run summary for the web UI status bar
    if state is not None:
        summary = _build_run_summary(all_video_results, all_ext_sub_results,
                                     runtime, total_new_ext_subs)
        summary["dry_run"] = bool(config.dry_run)
        state.set_last_run_summary(summary)


def main() -> None:
    _setup_signal_handlers()

    args = _build_parser().parse_args()

    # ── Quick-exit commands ──────────────────────────────────────────────
    if args.clear_tracking:
        tracker = ProcessingTracker("config")
        stats = tracker.get_stats()
        tracker.clear_all()
        print(f"Cleared tracking data for {stats['total_tracked']} entries")
        return

    if args.create_config:
        Config().create_sample_config(args.config)
        return

    if args.find_mkv:
        if sys.platform == "win32":
            found = find_mkvtoolnix_installation()
            if found:
                print(f"\nAdd to PATH: {os.path.dirname(found)}")
            else:
                print("MKVToolNix not found. Install from https://mkvtoolnix.download/")
        else:
            print("This option is only available on Windows")
        return

    # ── Logging ──────────────────────────────────────────────────────────
    log_file = setup_logging(
        config_dir="config",
        verbose=args.verbose,
        quiet=args.quiet,
    )
    logger.info("Log file: %s", log_file)

    # ── Config ───────────────────────────────────────────────────────────
    config = Config()
    if args.verbose:
        config.show_details = True
    elif args.quiet:
        config.show_details = False

    config.load_from_file(args.config)
    _apply_cli_overrides(config, args)

    # Adjust console log level after config is loaded
    if config.show_details:
        for h in logging.getLogger().handlers:
            if hasattr(h, "stream") and getattr(h.stream, "_original", None):
                h.setLevel(logging.DEBUG)
            elif isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler):
                h.setLevel(logging.DEBUG)

    # ── Scheduler state (shared between main thread and web UI) ────────
    sched_state = SchedulerState(config_dir="config")
    cli_one_shot: dict = {}
    for _key, _default in Config.TRANSIENT_RUN_OPTIONS.items():
        _current = getattr(config, _key)
        if _current != _default:
            cli_one_shot[_key] = _current
            setattr(config, _key, _default)
    if cli_one_shot:
        sched_state.stash_run_options(cli_one_shot)
        logger.info("CLI one-shot flags forwarded to first run: %s",
                    ", ".join(f"{k}={v}" for k, v in cli_one_shot.items()))

    # ── Web UI ───────────────────────────────────────────────────────────
    try:
        from uldas.webui import start_webui
        start_webui(config_path=args.config, scheduler_state=sched_state)
    except Exception as exc:
        logger.debug("Web UI not started: %s", exc)

    # ── Scheduling (Docker) ──────────────────────────────────────────────
    from uldas.scheduler import _load_initial_schedule, run_on_schedule

    import yaml as _yaml
    _cfg_has_schedule = False
    try:
        with open(args.config, "r", encoding="utf-8") as _fh:
            _raw = _yaml.safe_load(_fh) or {}
        _cfg_has_schedule = bool(_raw.get("schedule_type"))
    except Exception:
        pass

    _env_has_schedule = any(
        os.environ.get(k, "").strip()
        for k in ("CRON", "CRON_SCHEDULE", "SCHEDULE_HOURS")
    )

    if _cfg_has_schedule or _env_has_schedule:
        _load_initial_schedule(sched_state, args.config)
        run_on_schedule(
            lambda: _run_processing(config, skip_update_check=args.skip_update_check,
                                    state=sched_state, config_path=args.config),
            state=sched_state,
            run_on_startup=config.run_on_startup,
        )
        # run_on_schedule never returns
    else:
        sched_state.set_status("running")
        _run_processing(config, skip_update_check=args.skip_update_check,
                        state=sched_state, config_path=args.config)
        if sched_state.status != "error":
            sched_state.set_status("idle")