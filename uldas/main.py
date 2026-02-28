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
    p.add_argument("--config", default="config/config.yml", help="Config file path")
    p.add_argument("--create-config", action="store_true", help="Create sample config")
    p.add_argument("--directory", help="Override directory from config")
    p.add_argument("--model", choices=["tiny", "base", "small", "medium", "large"],
                   help="Override Whisper model")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--verbose", "-v", action="store_true")
    p.add_argument("--quiet", "-q", action="store_true")
    p.add_argument("--find-mkv", action="store_true", help="Locate MKVToolNix")
    p.add_argument("--skip-update-check", action="store_true")
    p.add_argument("--no-vad", action="store_true")
    p.add_argument("--device", choices=["auto", "cpu", "cuda"])
    p.add_argument("--compute-type",
                   choices=["auto", "int8", "int8_float16", "int16", "float16", "float32"])
    p.add_argument("--reprocess-all", action="store_true")
    p.add_argument("--process-subtitles", action="store_true")
    p.add_argument("--process-external-subtitles", action="store_true")
    p.add_argument("--analyze-forced", action="store_true")
    p.add_argument("--no-sdh-detection", action="store_true")
    p.add_argument("--reprocess-all-subtitles", action="store_true")
    p.add_argument("--force-reprocess", action="store_true")
    p.add_argument("--clear-tracking", action="store_true")
    p.add_argument("--no-tracking", action="store_true")
    p.add_argument("--temp-dir", help="Override temporary directory")
    return p


def _apply_cli_overrides(config: Config, args) -> None:
    if args.directory:
        config.path = [args.directory]
    if args.model:
        config.whisper_model = args.model
    if args.dry_run:
        config.dry_run = True
    if args.no_vad:
        config.vad_filter = False
    if args.device:
        config.device = args.device
    if args.compute_type:
        config.compute_type = args.compute_type
    if args.reprocess_all:
        config.reprocess_all = True
    if args.process_subtitles:
        config.process_subtitles = True
    if args.process_external_subtitles:
        config.process_external_subtitles = True
    if args.analyze_forced:
        config.analyze_forced_subtitles = True
    if args.no_sdh_detection:
        config.detect_sdh_subtitles = False
    if args.reprocess_all_subtitles:
        config.reprocess_all_subtitles = True
    if args.force_reprocess:
        config.force_reprocess = True
    if args.no_tracking:
        config.use_tracking = False
    if args.temp_dir:
        config.temp_dir = args.temp_dir


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