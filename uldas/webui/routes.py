#file: uldas/webui/routes.py

import os
import yaml
from flask import render_template, jsonify, request

import uldas.webui as webui
from uldas.tracking import ProcessingTracker
from uldas.constants import VERSION


class _QuotedDumper(yaml.SafeDumper):
    """YAML dumper that always quotes string values."""
    pass


def _quoted_str(dumper, data):
    return dumper.represent_scalar('tag:yaml.org,2002:str', data, style="'")


_QuotedDumper.add_representer(str, _quoted_str)

# ── Config option metadata ────────────────────────────────────────────────
# Each option defines its key, type, default, label, description, and
# whether it is an advanced option.

CONFIG_OPTIONS = [
    # ── Basic options ─────────────────────────────────────────────────────
    {
        "key": "path",
        "type": "path_list",
        "default": ["."],
        "label": "Scan Directories",
        "description": "Directories to scan for video files. Add one or more paths to your media libraries.",
        "advanced": False,
    },
    {
        "key": "remux_to_mkv",
        "type": "bool",
        "default": False,
        "label": "Remux to MKV",
        "description": "Remux non-MKV video files (MP4, AVI, etc.) to MKV format before processing. The original file is replaced.",
        "advanced": False,
    },
    {
        "key": "show_details",
        "type": "bool",
        "default": True,
        "label": "Show Details",
        "description": "Show detailed processing information and verbose logging output during runs.",
        "advanced": False,
    },
    {
        "key": "dry_run",
        "type": "bool",
        "default": False,
        "label": "Dry Run",
        "description": "Simulate all changes without actually modifying any files. Useful for previewing what would happen.",
        "advanced": False,
    },
    {
        "key": "process_subtitles",
        "type": "bool",
        "default": False,
        "label": "Process Subtitles",
        "description": "Process embedded subtitle tracks for language detection and labeling.",
        "advanced": False,
    },
    {
        "key": "process_external_subtitles",
        "type": "bool",
        "default": False,
        "label": "Process External Subtitles",
        "description": "Process external subtitle files (.srt, .ass, etc.) for language detection and file renaming.",
        "advanced": False,
    },
    {
        "key": "analyze_forced_subtitles",
        "type": "bool",
        "default": False,
        "label": "Analyze Forced Subtitles",
        "description": "Analyze subtitle tracks to detect and flag forced subtitles (e.g. foreign language dialogue only).",
        "advanced": False,
    },
    {
        "key": "detect_sdh_subtitles",
        "type": "bool",
        "default": True,
        "label": "Detect SDH Subtitles",
        "description": "Detect and tag SDH (Subtitles for the Deaf and Hard of Hearing) subtitle tracks.",
        "advanced": False,
    },
    # ── Advanced options ──────────────────────────────────────────────────
    {
        "key": "vad_filter",
        "type": "bool",
        "default": True,
        "label": "VAD Filter",
        "description": "Enables Voice Activity Detection to filter out silence and background noise before language analysis.",
        "advanced": True,
    },
    {
        "key": "vad_min_speech_duration_ms",
        "type": "int",
        "default": 250,
        "label": "VAD Min Speech Duration (ms)",
        "description": "Minimum speech segment length (in milliseconds) to consider as valid speech.",
        "advanced": True,
    },
    {
        "key": "vad_max_speech_duration_s",
        "type": "int",
        "default": 30,
        "label": "VAD Max Speech Duration (s)",
        "description": "Maximum continuous speech segment length (in seconds) before splitting.",
        "advanced": True,
    },
    {
        "key": "whisper_model",
        "type": "select",
        "default": "base",
        "label": "Whisper Model",
        "description": "Whisper model size for audio language detection. Larger models are more accurate but slower and use more memory.",
        "options": ["tiny", "base", "small", "medium", "large"],
        "advanced": True,
    },
    {
        "key": "device",
        "type": "select",
        "default": "auto",
        "label": "Device",
        "description": "Hardware acceleration preference. Auto-detects CUDA GPU if available, falls back to CPU.",
        "options": ["auto", "cpu", "cuda"],
        "advanced": True,
    },
    {
        "key": "compute_type",
        "type": "select",
        "default": "auto",
        "label": "Compute Type",
        "description": "Precision/performance trade-off. Auto-selects optimal type based on device.",
        "options": ["auto", "int8", "float16", "float32"],
        "advanced": True,
    },
    {
        "key": "cpu_threads",
        "type": "int",
        "default": 0,
        "label": "CPU Threads",
        "description": "Number of CPU threads to use. 0 = automatic detection based on system cores.",
        "advanced": True,
    },
    {
        "key": "confidence_threshold",
        "type": "float",
        "default": 0.9,
        "label": "Audio Confidence Threshold",
        "description": "Minimum confidence level (0.0-1.0) required to accept audio language detection. If sample-based detection falls below this, the entire track is analyzed.",
        "advanced": True,
    },
    {
        "key": "subtitle_confidence_threshold",
        "type": "float",
        "default": 0.85,
        "label": "Subtitle Confidence Threshold",
        "description": "Minimum confidence level (0.0-1.0) for subtitle language detection. Tracks below this threshold are skipped.",
        "advanced": True,
    },
    {
        "key": "reprocess_all",
        "type": "bool",
        "default": False,
        "label": "Reprocess All Audio",
        "description": "Reprocess ALL audio tracks, even if they already have a language tag.",
        "advanced": True,
    },
    {
        "key": "reprocess_all_subtitles",
        "type": "bool",
        "default": False,
        "label": "Reprocess All Subtitles",
        "description": "Reprocess ALL subtitle tracks, even if they already have a language tag.",
        "advanced": True,
    },
    {
        "key": "operation_timeout_seconds",
        "type": "int",
        "default": 600,
        "label": "Operation Timeout (seconds)",
        "description": "Timeout in seconds for long operations like full-track audio extraction. Default is 600 (10 minutes).",
        "advanced": True,
    },
    {
        "key": "temp_dir",
        "type": "string",
        "default": "",
        "label": "Temporary Directory",
        "description": "Custom temporary directory for audio/subtitle extraction. Leave empty to use system default (/tmp).",
        "advanced": True,
    },
    {
        "key": "forced_subtitle_low_density_threshold",
        "type": "float",
        "default": 3.0,
        "label": "Forced Sub Low Density",
        "description": "Subtitle density below this value = likely forced subtitles.",
        "advanced": True,
    },
    {
        "key": "forced_subtitle_high_density_threshold",
        "type": "float",
        "default": 8.0,
        "label": "Forced Sub High Density",
        "description": "Subtitle density above this value = likely full subtitles.",
        "advanced": True,
    },
    {
        "key": "forced_subtitle_low_coverage_threshold",
        "type": "float",
        "default": 25.0,
        "label": "Forced Sub Low Coverage (%)",
        "description": "Subtitle coverage below this percentage = likely forced subtitles.",
        "advanced": True,
    },
    {
        "key": "forced_subtitle_high_coverage_threshold",
        "type": "float",
        "default": 50.0,
        "label": "Forced Sub High Coverage (%)",
        "description": "Subtitle coverage above this percentage = likely full subtitles.",
        "advanced": True,
    },
    {
        "key": "forced_subtitle_min_count_threshold",
        "type": "int",
        "default": 50,
        "label": "Forced Sub Min Count",
        "description": "Subtitle count below this value = likely forced subtitles.",
        "advanced": True,
    },
    {
        "key": "forced_subtitle_max_count_threshold",
        "type": "int",
        "default": 300,
        "label": "Forced Sub Max Count",
        "description": "Subtitle count above this value = likely full subtitles.",
        "advanced": True,
    },
]


def register_routes(app, scheduler_state=None):

    @app.route("/")
    def index():
        return render_template("index.html", version=VERSION)

    # ── Scheduler status & control API ────────────────────────────────────

    @app.route("/api/status")
    def get_status():
        if scheduler_state is None:
            return jsonify({"status": "unknown", "has_cron": False,
                            "error_message": "", "next_run_time": None,
                            "next_run_seconds": None, "last_run_time": None,
                            "started_at": None, "cron_expression": None,
                            "last_run_summary": None})
        return jsonify(scheduler_state.get_status_dict())

    @app.route("/api/scheduler/run-now", methods=["POST"])
    def scheduler_run_now():
        if scheduler_state is None:
            return jsonify({"status": "error",
                            "message": "No scheduler active (not in CRON mode)"}), 404
        if scheduler_state.status == "running":
            return jsonify({"status": "error",
                            "message": "A processing run is already in progress"}), 409
        scheduler_state.request_run()
        return jsonify({"status": "ok", "message": "Run triggered"})

    @app.route("/api/scheduler/stop", methods=["POST"])
    def scheduler_stop():
        if scheduler_state is None:
            return jsonify({"status": "error",
                            "message": "No scheduler active (not in CRON mode)"}), 404
        scheduler_state.request_stop()
        note = ""
        if scheduler_state.status == "running":
            note = "Current processing run will complete before the scheduler stops"
        return jsonify({"status": "ok", "message": "Scheduler stopped",
                        "note": note})

    @app.route("/api/scheduler/start", methods=["POST"])
    def scheduler_start():
        if scheduler_state is None:
            return jsonify({"status": "error",
                            "message": "No scheduler active (not in CRON mode)"}), 404
        scheduler_state.request_resume()
        return jsonify({"status": "ok", "message": "Scheduler resumed"})

    # ── Config API ────────────────────────────────────────────────────────

    @app.route("/api/config", methods=["GET"])
    def get_config():
        config_path = webui._config_path
        config = {}

        if os.path.exists(config_path):
            try:
                with open(config_path, "r", encoding="utf-8") as fh:
                    config = yaml.safe_load(fh) or {}
            except Exception:
                config = {}
        else:
            # Create config with defaults
            config_dir = os.path.dirname(config_path)
            if config_dir:
                os.makedirs(config_dir, exist_ok=True)
            defaults = {opt["key"]: opt["default"] for opt in CONFIG_OPTIONS}
            try:
                with open(config_path, "w", encoding="utf-8") as fh:
                    yaml.dump(defaults, fh, Dumper=_QuotedDumper,
                              default_flow_style=False, sort_keys=False)
            except Exception:
                pass
            config = defaults

        result = []
        for opt in CONFIG_OPTIONS:
            entry = dict(opt)
            entry["value"] = config.get(opt["key"], opt["default"])
            result.append(entry)

        return jsonify(result)

    @app.route("/api/config", methods=["POST"])
    def save_config():
        new_values = request.get_json()
        config_path = webui._config_path

        # Preserve any extra keys not managed by the UI
        existing = {}
        if os.path.exists(config_path):
            try:
                with open(config_path, "r", encoding="utf-8") as fh:
                    existing = yaml.safe_load(fh) or {}
            except Exception:
                existing = {}

        existing.update(new_values)

        config_dir = os.path.dirname(config_path)
        if config_dir:
            os.makedirs(config_dir, exist_ok=True)

        try:
            with open(config_path, "w", encoding="utf-8") as fh:
                yaml.dump(existing, fh, Dumper=_QuotedDumper,
                          default_flow_style=False, sort_keys=False)
            return jsonify({"status": "ok"})
        except Exception as exc:
            return jsonify({"status": "error", "message": str(exc)}), 500

    # ── Stats API ─────────────────────────────────────────────────────────

    @app.route("/api/stats")
    def get_stats():
        tracker = ProcessingTracker(webui._config_dir, read_only=True)

        start = request.args.get("start", type=float)
        end = request.args.get("end", type=float)

        stats = tracker.get_stats(start_ts=start, end_ts=end)
        stats["version"] = VERSION

        cron = os.environ.get("CRON_SCHEDULE", "").strip()
        stats["cron_schedule"] = cron if cron else None

        return jsonify(stats)

    @app.route("/api/stats/timeseries")
    def get_timeseries():
        tracker = ProcessingTracker(webui._config_dir, read_only=True)
        granularity = request.args.get("granularity", "day")
        limit = request.args.get("limit", 30, type=int)
        return jsonify(tracker.get_time_series(granularity, limit))

    # ── Processing log API ────────────────────────────────────────────────

    @app.route("/api/log")
    def get_log():
        tracker = ProcessingTracker(webui._config_dir, read_only=True)
        entries = tracker.get_log_entries()
        failed = tracker.load_failed_files()

        # Merge and sort by timestamp descending
        all_entries = entries + failed
        all_entries.sort(key=lambda x: x.get("timestamp", 0), reverse=True)

        return jsonify(all_entries)
