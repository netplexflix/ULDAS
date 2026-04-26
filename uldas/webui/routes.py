#file: uldas/webui/routes.py

import os
import threading
import yaml
from flask import render_template, jsonify, request

import uldas.webui as webui
from uldas.tracking import ProcessingTracker
from uldas.constants import VERSION
from uldas.updater import get_update_status


# ── Tracker cache ────────────────────────────────────────────────────────
# Re-parsing the JSON tracking files on every API request is wasteful,
# especially with the dashboard polling.  Cache a single read-only
# tracker and invalidate it when the underlying files change on disk.
_tracker_cache_lock = threading.Lock()
_tracker_cache: "dict[str, object]" = {
    "tracker": None,
    "config_dir": None,
    "mtimes": (),
}


def _file_mtimes(config_dir: str) -> tuple:
    """Snapshot of the mtimes that influence tracker state.

    Files that don't exist contribute ``0.0`` so their later creation
    still shows up as a change.
    """
    names = (
        "processed_files.json",
        "processed_external_subtitles.json",
        "failed_files.json",
    )
    out = []
    for name in names:
        try:
            out.append(os.path.getmtime(os.path.join(config_dir, name)))
        except OSError:
            out.append(0.0)
    return tuple(out)


def _get_cached_tracker(config_dir: str) -> ProcessingTracker:
    """Return a cached read-only ProcessingTracker for *config_dir*.

    Reuses the existing instance if the tracking files haven't been
    modified since the last load.  Thread-safe — Flask serves API
    requests from a thread pool.
    """
    with _tracker_cache_lock:
        current_mtimes = _file_mtimes(config_dir)
        if (_tracker_cache["tracker"] is not None
                and _tracker_cache["config_dir"] == config_dir
                and _tracker_cache["mtimes"] == current_mtimes):
            return _tracker_cache["tracker"]
        tracker = ProcessingTracker(config_dir, read_only=True)
        _tracker_cache["tracker"] = tracker
        _tracker_cache["config_dir"] = config_dir
        _tracker_cache["mtimes"] = current_mtimes
        return tracker


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
    # ── Scheduler ─────────────────────────────────────────────────────────
    {
        "key": "schedule_type",
        "type": "select",
        "default": "cron",
        "label": "Schedule Type",
        "description": "Choose between a simple hours interval or a CRON expression.",
        "options": ["hours", "cron"],
        "section": "scheduler",
    },
    {
        "key": "schedule_hours",
        "type": "int",
        "default": 24,
        "label": "Hours Interval",
        "description": "Run every X hours (used when Schedule Type is 'hours').",
        "section": "scheduler",
    },
    {
        "key": "schedule_cron",
        "type": "string",
        "default": "0 5 * * 5",
        "label": "CRON Expression",
        "description": "Standard 5-field CRON expression. See crontab.guru for help.",
        "section": "scheduler",
    },
    {
        "key": "run_on_startup",
        "type": "bool",
        "default": False,
        "label": "Run on Startup",
        "description": "Run immediately when the container starts. Disabled by default so you can review settings in the Web UI first; enable if you want ULDAS to start processing as soon as the container is up.",
        "section": "scheduler",
    },
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
        "key": "ignore_tags",
        "type": "string_list",
        "default": [],
        "label": "Ignore Tags",
        "description": "Skip any file whose name contains one of these substrings (case-insensitive). Useful for excluding trailers, samples, featurettes, etc. Examples: -trailer, sample",
        "placeholder": "-trailer",
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
        "default": "small",
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

    @app.route("/api/scheduler/index-languages", methods=["POST"])
    def scheduler_index_languages():
        if scheduler_state is None:
            return jsonify({"status": "error",
                            "message": "No scheduler active (not in CRON mode)"}), 404
        if scheduler_state.status == "running":
            return jsonify({"status": "error",
                            "message": "A processing run is already in progress"}), 409
        scheduler_state.request_run(options={"index_languages_only": True})
        return jsonify({"status": "ok",
                        "message": "Language indexing run triggered"})

    @app.route("/api/scheduler/reprocess-language", methods=["POST"])
    def scheduler_reprocess_language():
        if scheduler_state is None:
            return jsonify({"status": "error",
                            "message": "No scheduler active (not in CRON mode)"}), 404
        if scheduler_state.status == "running":
            return jsonify({"status": "error",
                            "message": "A processing run is already in progress"}), 409
        body = request.get_json(silent=True) or {}
        code = (body.get("language") or "").strip().lower()
        if not code:
            return jsonify({"status": "error",
                            "message": "Missing 'language' in request body"}), 400
        if not code.replace("-", "").isalnum() or len(code) > 10:
            return jsonify({"status": "error",
                            "message": f"Invalid language code: {code}"}), 400
        scheduler_state.request_run(options={"reprocess_language": code})
        return jsonify({"status": "ok",
                        "message": f"Reprocess run triggered for language '{code}'"})

    @app.route("/api/scheduler/stop", methods=["POST"])
    def scheduler_stop():
        if scheduler_state is None:
            return jsonify({"status": "error",
                            "message": "No scheduler active (not in CRON mode)"}), 404
        was_running = scheduler_state.status == "running"
        if was_running:
            scheduler_state.set_status("stopping", "Cancelling current run...")
        scheduler_state.request_stop()
        note = ""
        if was_running:
            note = "Stopping after the current file finishes processing..."
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
        new_values = request.get_json() or {}
        config_path = webui._config_path

        # ── Validate scheduler fields BEFORE writing config ───────────
        if "schedule_type" in new_values or "schedule_hours" in new_values or "schedule_cron" in new_values:
            sched_type = (new_values.get("schedule_type") or "cron").strip().lower()
            if sched_type not in ("hours", "cron"):
                return jsonify({"status": "error",
                                "message": f"Invalid schedule_type: {sched_type}"}), 400
            new_values["schedule_type"] = sched_type

            try:
                sched_hours = int(new_values.get("schedule_hours", 24))
            except (TypeError, ValueError):
                return jsonify({"status": "error",
                                "message": "schedule_hours must be an integer"}), 400
            if sched_hours < 1:
                sched_hours = 1
            new_values["schedule_hours"] = sched_hours

            sched_cron = (new_values.get("schedule_cron") or "").strip()
            new_values["schedule_cron"] = sched_cron

            if sched_type == "cron":
                try:
                    from croniter import croniter
                except ImportError:
                    return jsonify({"status": "error",
                                    "message": "croniter package is not installed"}), 500
                if not croniter.is_valid(sched_cron):
                    return jsonify({"status": "error",
                                    "message": f"Invalid CRON expression: {sched_cron}"}), 400

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
        except Exception as exc:
            return jsonify({"status": "error", "message": str(exc)}), 500

        # ── Signal scheduler of live schedule change ──────────────────
        if webui._scheduler_state is not None and (
            "schedule_type" in new_values
            or "schedule_hours" in new_values
            or "schedule_cron" in new_values
        ):
            ok, err = webui._scheduler_state.update_schedule(
                existing.get("schedule_type", "cron"),
                int(existing.get("schedule_hours", 24)),
                existing.get("schedule_cron", "") or "",
            )
            if not ok:
                return jsonify({"status": "error", "message": err}), 400

        return jsonify({"status": "ok"})

    # ── Stats API ─────────────────────────────────────────────────────────

    @app.route("/api/stats")
    def get_stats():
        tracker = _get_cached_tracker(webui._config_dir)

        start = request.args.get("start", type=float)
        end = request.args.get("end", type=float)

        stats = tracker.get_stats(start_ts=start, end_ts=end)
        stats["version"] = VERSION

        cron = os.environ.get("CRON_SCHEDULE", "").strip()
        stats["cron_schedule"] = cron if cron else None

        return jsonify(stats)

    @app.route("/api/stats/timeseries")
    def get_timeseries():
        tracker = _get_cached_tracker(webui._config_dir)
        granularity = request.args.get("granularity", "day")
        limit = request.args.get("limit", 30, type=int)
        return jsonify(tracker.get_time_series(granularity, limit))

    def _name_for(code: str) -> str:
        from uldas.constants import LANGUAGE_NAMES
        name = LANGUAGE_NAMES.get(code)
        if not name and "-" in code:
            name = LANGUAGE_NAMES.get(code.split("-", 1)[0])
        if name:
            # Drop the alternate-name tail ("Foo; Bar; Baz") for UI
            # cleanliness — the primary name is the canonical label.
            return name.split(";", 1)[0].strip()
        return "Unknown"

    @app.route("/api/stats/languages")
    def get_languages():
        from uldas.language_index import load_language_index

        index = load_language_index(webui._config_dir)
        if index and isinstance(index.get("counts"), dict):
            audio = index["counts"].get("audio") or {}
            items = [{"code": code, "name": _name_for(code)}
                     for code in audio.keys()]
            items.sort(key=lambda x: (x["name"].lower(), x["code"]))
            return jsonify({
                "source": "index",
                "indexed_at": index.get("indexed_at"),
                "languages": items,
            })

        tracker = _get_cached_tracker(webui._config_dir)
        raw = tracker.count_tracked_audio_languages()
        items = [{"code": r["code"], "name": _name_for(r["code"])} for r in raw]
        items.sort(key=lambda x: (x["name"].lower(), x["code"]))
        return jsonify({
            "source": "tracker",
            "indexed_at": None,
            "languages": items,
        })

    @app.route("/api/stats/language-index")
    def get_language_index():
        """Return the full language-index document for the dashboard's
        language-distribution chart. 404 when no index has been built.
        """
        from uldas.language_index import load_language_index
        index = load_language_index(webui._config_dir)
        if not index:
            return jsonify({"status": "missing"}), 404

        # Attach a {code: name} map for every code present so the chart
        # can render "<code> - <name>" labels matching the dropdown style.
        names: dict = {}
        counts = index.get("counts") if isinstance(index, dict) else None
        if isinstance(counts, dict):
            for bucket in counts.values():
                if isinstance(bucket, dict):
                    for code in bucket.keys():
                        if code and code not in names:
                            names[code] = _name_for(code)
        index["names"] = names
        return jsonify(index)

    @app.route("/api/stats/language-index/files")
    def get_language_index_files():
        from uldas.language_index import load_language_index

        code = (request.args.get("code") or "").strip().lower()
        kind = (request.args.get("kind") or "audio").strip().lower()
        try:
            limit = int(request.args.get("limit", "200"))
        except ValueError:
            limit = 200
        limit = max(1, min(limit, 5000))

        if not code:
            return jsonify({"status": "error",
                            "message": "Missing 'code' query parameter"}), 400
        if kind not in ("audio", "embedded_subs", "external_subs"):
            return jsonify({"status": "error",
                            "message": ("Invalid 'kind': must be audio, "
                                        "embedded_subs, or external_subs")}), 400

        index = load_language_index(webui._config_dir)
        if not index:
            return jsonify({"status": "missing"}), 404

        matches: list = []
        total = 0
        if kind == "external_subs":
            for path, stored_code in (index.get("per_ext_sub") or {}).items():
                if stored_code == code:
                    total += 1
                    if len(matches) < limit:
                        matches.append({"path": path, "code": stored_code})
        else:
            for path, info in (index.get("per_file") or {}).items():
                codes = list(info.get(kind) or [])
                if code in codes:
                    total += 1
                    if len(matches) < limit:
                        matches.append({"path": path, "tracks": codes})
        matches.sort(key=lambda m: m["path"].lower())
        return jsonify({
            "code": code,
            "kind": kind,
            "matches": matches,
            "total": total,
            "truncated": total > len(matches),
        })

    # ── Cache maintenance API ─────────────────────────────────────────────

    @app.route("/api/cache/prune-orphans", methods=["POST"])
    def prune_orphans():
        """Remove tracker entries whose path is no longer under any of the currently-configured library paths."""
        from uldas.tracking import ProcessingTracker

        # Read the live config.yml to discover the current library paths.
        config_path = webui._config_path
        configured_paths: list = []
        if os.path.exists(config_path):
            try:
                with open(config_path, "r", encoding="utf-8") as fh:
                    cfg = yaml.safe_load(fh) or {}
                raw = cfg.get("path") or []
                if isinstance(raw, str):
                    raw = [raw]
                configured_paths = [p for p in raw if isinstance(p, str) and p.strip()]
            except Exception as exc:
                return jsonify({"status": "error",
                                "message": f"Could not read config: {exc}"}), 500

        tracker = ProcessingTracker(webui._config_dir, read_only=False)
        removed = tracker.prune_entries_outside_paths(configured_paths)

        # Also prune the language index so the Dashboard chart + the
        # Reprocess-by-language dropdown stop counting the removed files.
        from uldas.language_index import LanguageIndex
        lang_index = LanguageIndex(config_dir=webui._config_dir, read_only=False)
        idx_removed = lang_index.prune_outside_paths(configured_paths)
        lang_index.save_if_dirty()

        # Invalidate the cached read-only tracker so the dashboard reflects
        # the cleanup on the next poll.
        with _tracker_cache_lock:
            _tracker_cache["tracker"] = None
            _tracker_cache["mtimes"] = ()

        total = (removed["videos"] + removed["ext_subs"] + removed["failed"]
                 + idx_removed["files"] + idx_removed["ext_subs"])
        return jsonify({
            "status": "ok",
            "removed": removed,
            "index_removed": idx_removed,
            "total": total,
            "configured_paths": configured_paths,
        })

    # ── Update check API ──────────────────────────────────────────────────

    @app.route("/api/update")
    def get_update():
        return jsonify(get_update_status())

    # ── Processing log API ────────────────────────────────────────────────

    @app.route("/api/log")
    def get_log():
        tracker = _get_cached_tracker(webui._config_dir)
        entries = tracker.get_log_entries()
        failed = tracker.load_failed_files()

        # Merge and sort by timestamp descending
        all_entries = entries + failed
        all_entries.sort(key=lambda x: x.get("timestamp", 0), reverse=True)

        return jsonify(all_entries)
