#file: uldas/config.py

import os
import logging
import yaml

logger = logging.getLogger(__name__)


class Config:

    def __init__(self):
        # ── Paths & general ──────────────────────────────────────────────
        self.path: list[str] = ["."]
        self.remux_to_mkv: bool = False
        self.show_details: bool = True
        self.whisper_model: str = "base"
        self.dry_run: bool = False
        self.temp_dir: str = ""

        # ── VAD ──────────────────────────────────────────────────────────
        self.vad_filter: bool = True
        self.vad_min_speech_duration_ms: int = 250
        self.vad_max_speech_duration_s: int = 30

        # ── Device / compute ─────────────────────────────────────────────
        self.device: str = "auto"
        self.compute_type: str = "auto"
        self.cpu_threads: int = 0

        # ── Confidence ───────────────────────────────────────────────────
        self.confidence_threshold: float = 0.9
        self.reprocess_all: bool = False
        self.use_tracking: bool = True
        self.force_reprocess: bool = False

        # ── Subtitle processing ──────────────────────────────────────────
        self.process_subtitles: bool = False
        self.process_external_subtitles: bool = False
        self.analyze_forced_subtitles: bool = False
        self.detect_sdh_subtitles: bool = True
        self.subtitle_confidence_threshold: float = 0.85
        self.reprocess_all_subtitles: bool = False

        # ── Timeouts ─────────────────────────────────────────────────────
        self.operation_timeout_seconds: int = 600

        # ── Forced subtitle thresholds ───────────────────────────────────
        self.forced_subtitle_low_coverage_threshold: float = 25.0
        self.forced_subtitle_high_coverage_threshold: float = 50.0
        self.forced_subtitle_low_density_threshold: float = 3.0
        self.forced_subtitle_high_density_threshold: float = 8.0
        self.forced_subtitle_min_count_threshold: int = 50
        self.forced_subtitle_max_count_threshold: int = 300

    # ── Load from YAML ───────────────────────────────────────────────────
    def load_from_file(self, config_path: str = "config/config.yml") -> None:
        config_dir = os.path.dirname(config_path)
        if config_dir and not os.path.exists(config_dir):
            os.makedirs(config_dir, exist_ok=True)

        if not os.path.exists(config_path):
            logger.info("Config file %s not found, using defaults", config_path)
            return

        try:
            with open(config_path, "r", encoding="utf-8") as fh:
                data = yaml.safe_load(fh)

            if not data:
                return

            # Map every known key automatically
            for key in vars(self):
                if key in data:
                    setattr(self, key, data[key])

            logger.info("Configuration loaded from %s", config_path)

        except Exception as exc:
            logger.error("Error loading config file %s: %s", config_path, exc)
            logger.info("Using default configuration")

    # ── Create sample config ─────────────────────────────────────────────
    def create_sample_config(self, config_path: str = "config/config.yml") -> None:
        sample = {
            "path": ["P:/Movies", "P:/TV"],
            "remux_to_mkv": True,
            "show_details": False,
            "whisper_model": "small",
            "dry_run": False,
            "temp_dir": "",
            "vad_filter": True,
            "vad_min_speech_duration_ms": 250,
            "vad_max_speech_duration_s": 30,
            "device": "auto",
            "compute_type": "auto",
            "cpu_threads": 0,
            "confidence_threshold": 0.9,
            "reprocess_all": False,
            "process_subtitles": True,
            "process_external_subtitles": False,
            "analyze_forced_subtitles": True,
            "detect_sdh_subtitles": True,
            "subtitle_confidence_threshold": 0.85,
            "reprocess_all_subtitles": False,
            "operation_timeout_seconds": 600,
            "forced_subtitle_low_density_threshold": 3.0,
            "forced_subtitle_high_density_threshold": 8.0,
            "forced_subtitle_low_coverage_threshold": 25.0,
            "forced_subtitle_high_coverage_threshold": 50.0,
            "forced_subtitle_min_count_threshold": 50,
            "forced_subtitle_max_count_threshold": 300,
        }

        try:
            config_dir = os.path.dirname(config_path)
            if config_dir:
                os.makedirs(config_dir, exist_ok=True)

            with open(config_path, "w", encoding="utf-8") as fh:
                yaml.dump(sample, fh, default_flow_style=False, sort_keys=False)

            print(f"Sample configuration file created: {config_path}")
            print("Edit this file to customize your settings")
        except Exception as exc:
            logger.error("Error creating config file: %s", exc)