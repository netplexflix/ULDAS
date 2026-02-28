#file: uldas/tracking.py

import json
import os
import time
import logging
from pathlib import Path
from typing import Dict, Optional

logger = logging.getLogger(__name__)


class ProcessingTracker:
    def __init__(self, config_dir: str = "config"):
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(exist_ok=True)
        self.tracker_file = self.config_dir / "processed_files.json"
        self.data: Dict = self._load()
        self._dirty = False

    # ── Persistence ──────────────────────────────────────────────────────
    def _load(self) -> Dict:
        if self.tracker_file.exists():
            try:
                with open(self.tracker_file, "r", encoding="utf-8") as fh:
                    return json.load(fh)
            except (json.JSONDecodeError, IOError) as exc:
                logger.warning("Could not load tracking file: %s. Starting fresh.", exc)
                return {}
        return {}

    def _save(self) -> None:
        try:
            with open(self.tracker_file, "w", encoding="utf-8") as fh:
                json.dump(self.data, fh, indent=2)
            self._dirty = False
        except IOError as exc:
            logger.error("Could not save tracking file: %s", exc)

    def save_if_dirty(self) -> None:
        """Write to disk only if in-memory data has changed."""
        if self._dirty:
            self._save()

    # ── Fast-skip key set for directory scanning ─────────────────────────
    def get_fully_processed_keys(self, process_subtitles: bool = False) -> set[str]:
        keys: set[str] = set()
        for key, entry in self.data.items():
            # Skip external subtitle entries
            if entry.get("type") == "external_subtitle":
                continue
            if not entry.get("audio_processed", False):
                continue
            if process_subtitles and not entry.get("subtitle_processed", False):
                continue
            keys.add(key)
        return keys

    # ── Bulk check (the fast path) ───────────────────────────────────────
    def check_files_batch(
        self,
        file_paths: list[Path],
        process_subtitles: bool = False,
    ) -> tuple[list[Path], list[Path], dict[str, str]]:
        """Partition *file_paths* into (actionable, skipped) lists.

        Returns ``(actionable, skipped, key_cache)`` where *key_cache*
        maps ``str(file_path) -> absolute_key``.
        """
        actionable: list[Path] = []
        skipped: list[Path] = []
        key_cache: dict[str, str] = {}

        tracked_keys: set[str] = set(self.data.keys())
        removals: list[str] = []

        for fp in file_paths:
            fp_str = str(fp)
            key = os.path.abspath(fp_str)
            key_cache[fp_str] = key

            if key not in tracked_keys:
                actionable.append(fp)
                continue

            entry = self.data[key]

            # Skip external subtitle entries in video file batch check
            if entry.get("type") == "external_subtitle":
                actionable.append(fp)
                continue

            # One stat() call — validates existence + size + mtime
            try:
                stat = fp.stat()
            except OSError:
                removals.append(key)
                actionable.append(fp)
                continue

            if (entry.get("size") != stat.st_size
                    or abs(entry.get("mtime", 0) - stat.st_mtime) > 1):
                removals.append(key)
                actionable.append(fp)
                continue

            # File is tracked and unchanged — check completeness
            skip_audio = entry.get("audio_processed", False)
            skip_subs = entry.get("subtitle_processed", False)

            if skip_audio and (skip_subs or not process_subtitles):
                skipped.append(fp)
            else:
                actionable.append(fp)

        if removals:
            for key in removals:
                self.data.pop(key, None)
            self._dirty = True
            self._save()

        return actionable, skipped, key_cache

    # ── Single-file query (kept for backward compat) ─────────────────────
    def is_processed(self, file_path: Path) -> bool:
        key = os.path.abspath(str(file_path))
        if key not in self.data:
            return False

        entry = self.data[key]

        try:
            stat = file_path.stat()
        except OSError:
            del self.data[key]
            self._dirty = True
            self._save()
            return False

        if (entry.get("size") != stat.st_size
                or abs(entry.get("mtime", 0) - stat.st_mtime) > 1):
            del self.data[key]
            self._dirty = True
            self._save()
            return False

        return True

    def get_entry(self, file_path: Path, key: str = None) -> dict:
        """Return the tracker entry without any filesystem I/O."""
        if key is None:
            key = os.path.abspath(str(file_path))
        return self.data.get(key, {})

    def mark_processed(
        self,
        file_path: Path,
        audio_success: bool = False,
        subtitle_success: bool = False,
        key: str = None,
    ) -> None:
        if not (audio_success or subtitle_success):
            return
        if key is None:
            key = os.path.abspath(str(file_path))
        try:
            stat = file_path.stat()
        except OSError:
            logger.warning("Cannot stat file for tracking: %s", file_path)
            return
        self.data[key] = {
            "size": stat.st_size,
            "mtime": stat.st_mtime,
            "audio_processed": audio_success,
            "subtitle_processed": subtitle_success,
            "processed_date": time.time(),
        }
        self._dirty = True
        self._save()

    # ── External subtitle tracking ───────────────────────────────────────
    def is_external_subtitle_processed(self, subtitle_path: Path) -> bool:
        key = os.path.abspath(str(subtitle_path))
        if key not in self.data:
            return False

        entry = self.data[key]
        if entry.get("type") != "external_subtitle":
            return False

        try:
            stat = subtitle_path.stat()
        except OSError:
            del self.data[key]
            self._dirty = True
            self._save()
            return False

        if (entry.get("size") != stat.st_size
                or abs(entry.get("mtime", 0) - stat.st_mtime) > 1):
            del self.data[key]
            self._dirty = True
            self._save()
            return False

        return True

    def mark_external_subtitle_processed(
        self,
        subtitle_path: Path,
        language_code: str,
        new_path: Path = None,
    ) -> None:
        track_path = new_path if new_path else subtitle_path
        key = os.path.abspath(str(track_path))
        try:
            stat = track_path.stat()
        except OSError:
            logger.warning("Cannot stat external subtitle for tracking: %s", track_path)
            return
        self.data[key] = {
            "type": "external_subtitle",
            "size": stat.st_size,
            "mtime": stat.st_mtime,
            "language_code": language_code,
            "original_path": str(subtitle_path),
            "processed_date": time.time(),
        }
        # If the file was renamed, also remove the old key if present
        if new_path and new_path != subtitle_path:
            old_key = os.path.abspath(str(subtitle_path))
            if old_key in self.data:
                del self.data[old_key]
        self._dirty = True
        self._save()

    def clear_entry(self, file_path: Path) -> None:
        key = os.path.abspath(str(file_path))
        if key in self.data:
            del self.data[key]
            self._dirty = True
            self._save()

    def clear_all(self) -> None:
        self.data = {}
        self._save()

    def get_stats(self) -> Dict:
        total = len(self.data)
        ext_subs = sum(
            1 for e in self.data.values()
            if e.get("type") == "external_subtitle"
        )
        video_entries = total - ext_subs
        audio_only = sum(
            1 for e in self.data.values()
            if e.get("type") != "external_subtitle"
            and e.get("audio_processed") and not e.get("subtitle_processed")
        )
        subtitle_only = sum(
            1 for e in self.data.values()
            if e.get("type") != "external_subtitle"
            and e.get("subtitle_processed") and not e.get("audio_processed")
        )
        both = sum(
            1 for e in self.data.values()
            if e.get("type") != "external_subtitle"
            and e.get("audio_processed") and e.get("subtitle_processed")
        )
        return {
            "total_tracked": total,
            "video_files_tracked": video_entries,
            "external_subtitles_tracked": ext_subs,
            "audio_only": audio_only,
            "subtitle_only": subtitle_only,
            "both": both,
        }