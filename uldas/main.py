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
    p.add_argument("--force-reprocess", action="store_true", default=None,
                   help="Force reprocessing, ignoring tracking cache (config: force_reprocess)")

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

    if args.force_reprocess:
        config.force_reprocess = True

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


def _run_processing(config: Config, skip_update_check: bool = False) -> None:
    start = time.time()

    if not skip_update_check:
        check_for_updates()

    # Apply temp_dir setting
    _apply_temp_dir(config)

    # Validate directories
    for d in config.path:
        if not os.path.isdir(d):
            logger.error("Directory not found: %s", d)
            sys.exit(1)

    # Check dependencies
    deps = {
        "ffmpeg": "FFmpeg (https://ffmpeg.org/download.html)",
        "ffprobe": "FFmpeg (https://ffmpeg.org/download.html)",
        "mkvpropedit": "MKVToolNix (https://mkvtoolnix.download/downloads.html)",
    }
    missing = [(n, s) for n, s in deps.items() if not find_executable(n)]
    if missing:
        for n, s in missing:
            logger.error("Missing: %s – install from %s", n, s)
        sys.exit(1)

    # Import detector here (heavy import due to Whisper)
    from uldas.detector import MKVLanguageDetector

    try:
        if config.show_details:
            logger.info("Loading faster-whisper model: %s", config.whisper_model)
        else:
            print(f"Loading faster-whisper model: {config.whisper_model}")
        detector = MKVLanguageDetector(config)
    except RuntimeError as exc:
        logger.error("Failed to initialise detector: %s", exc)
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
    for d in config.path:
        video_results, ext_sub_results, dir_ext_subs_found = detector.process_directory(d)
        all_video_results.extend(video_results)
        all_ext_sub_results.extend(ext_sub_results)
        total_ext_subs_found += dir_ext_subs_found

    runtime = time.time() - start
    print_detailed_summary(all_video_results, all_ext_sub_results,
                           config, runtime, detector,
                           total_ext_subs_found=total_ext_subs_found)


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

    # ── CRON scheduling (Docker) ─────────────────────────────────────────
    from uldas.scheduler import get_cron_schedule, run_on_schedule

    cron = get_cron_schedule()
    if cron:
        print(f"CRON scheduling enabled: {cron}")
        run_on_schedule(cron, lambda: _run_processing(config, skip_update_check=args.skip_update_check))
        # run_on_schedule never returns
    else:
        _run_processing(config, skip_update_check=args.skip_update_check)