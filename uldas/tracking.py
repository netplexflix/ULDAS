#file: uldas/tracking.py

import json
import os
import tempfile
import time
import logging
from pathlib import Path
from typing import Dict, Optional

logger = logging.getLogger(__name__)


class ProcessingTracker:
    def __init__(self, config_dir: str = "config", read_only: bool = False):
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(exist_ok=True)
        self.tracker_file = self.config_dir / "processed_files.json"
        self._read_only = read_only
        self.data: Dict = self._load()
        self._dirty = False
        if not read_only:
            self._ensure_migrated()

    # ── Persistence ──────────────────────────────────────────────────────
    def _load(self) -> Dict:
        if self.tracker_file.exists():
            try:
                with open(self.tracker_file, "r", encoding="utf-8") as fh:
                    data = json.load(fh)
                if not isinstance(data, dict):
                    logger.warning("Tracking file is not a dict, ignoring.")
                    return {}
                return data
            except (json.JSONDecodeError, IOError) as exc:
                # If the file exists but can't be read (e.g. mid-write),
                # try the backup before giving up.
                backup = self.tracker_file.with_suffix(".json.bak")
                if backup.exists():
                    try:
                        with open(backup, "r", encoding="utf-8") as fh:
                            data = json.load(fh)
                        if isinstance(data, dict):
                            logger.warning(
                                "Primary tracking file corrupt (%s), "
                                "restored from backup (%d entries).",
                                exc, len(data),
                            )
                            return data
                    except (json.JSONDecodeError, IOError):
                        pass
                logger.warning("Could not load tracking file: %s. Starting fresh.", exc)
                return {}
        return {}

    def _save(self) -> None:
        if self._read_only:
            return
        try:
            # Atomic write: write to temp file, then rename.
            # This prevents truncation of the real file on crash / race.
            tmp_fd, tmp_path = tempfile.mkstemp(
                dir=str(self.config_dir), suffix=".tmp", prefix="pf_",
            )
            try:
                with os.fdopen(tmp_fd, "w", encoding="utf-8") as fh:
                    json.dump(self.data, fh, indent=2)
                    fh.flush()
                    os.fsync(fh.fileno())
            except BaseException:
                os.unlink(tmp_path)
                raise

            # Keep a backup of the previous version
            backup = self.tracker_file.with_suffix(".json.bak")
            if self.tracker_file.exists():
                try:
                    os.replace(str(self.tracker_file), str(backup))
                except OSError:
                    pass

            os.replace(tmp_path, str(self.tracker_file))
            self._dirty = False
        except IOError as exc:
            logger.error("Could not save tracking file: %s", exc)

    def _ensure_migrated(self) -> None:
        """Ensure all entries have proper flags and filenames."""
        migrated = False
        for key, entry in self.data.items():
            if "flags" not in entry:
                if entry.get("type") == "external_subtitle":
                    entry["flags"] = ["external_subtitle"]
                else:
                    entry["flags"] = ["no_action_required"]
                migrated = True
            elif (entry.get("type") == "external_subtitle"
                  and "external_subtitle" not in entry.get("flags", [])):
                entry["flags"] = ["external_subtitle"]
                migrated = True
            if "filename" not in entry:
                entry["filename"] = Path(key).name
                migrated = True
        if migrated:
            self._dirty = True
            self._save()

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
        flags: list = None,
        audio_tracks_labeled: int = 0,
        subtitle_tracks_labeled: int = 0,
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
            "filename": file_path.name,
            "flags": flags or ["no_action_required"],
            "audio_tracks_labeled": audio_tracks_labeled,
            "subtitle_tracks_labeled": subtitle_tracks_labeled,
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
            "filename": track_path.name,
            "flags": ["external_subtitle"],
        }
        # If the file was renamed, also remove the old key if present
        if new_path and new_path != subtitle_path:
            old_key = os.path.abspath(str(subtitle_path))
            if old_key in self.data:
                del self.data[old_key]
        self._dirty = True
        self._save()

    def prune_missing_files(self, directory: str = None) -> int:
        """Remove entries for files that no longer exist on disk.

        If *directory* is given, only entries under that directory are
        checked — this avoids accidentally pruning entries on drives
        that happen to be offline.
        """
        prefix = os.path.abspath(directory) if directory else None
        removals = [
            key for key in self.data
            if (not prefix or key.startswith(prefix))
            and not os.path.exists(key)
        ]
        if removals:
            for key in removals:
                del self.data[key]
            self._dirty = True
            self._save()
        return len(removals)

    def clear_entry(self, file_path: Path) -> None:
        key = os.path.abspath(str(file_path))
        if key in self.data:
            del self.data[key]
            self._dirty = True
            self._save()

    def clear_all(self) -> None:
        self.data = {}
        self._save()

    # ── External subtitle scan count (persisted between runs) ─────────
    def save_ext_sub_scan_count(self, count: int) -> None:
        """Save the total external subtitle count from a filesystem scan."""
        scan_file = self.config_dir / "ext_sub_scan_count.json"
        try:
            with open(scan_file, "w", encoding="utf-8") as fh:
                json.dump({"count": count}, fh)
        except IOError:
            pass

    def _load_ext_sub_scan_count(self) -> int:
        scan_file = self.config_dir / "ext_sub_scan_count.json"
        if scan_file.exists():
            try:
                with open(scan_file, "r", encoding="utf-8") as fh:
                    return json.load(fh).get("count", 0)
            except (json.JSONDecodeError, IOError):
                pass
        return 0

    def get_stats(self, start_ts: float = None, end_ts: float = None) -> Dict:
        is_filtered = start_ts is not None or end_ts is not None
        if is_filtered:
            entries = {k: e for k, e in self.data.items()
                       if (start_ts is None or e.get("processed_date", 0) >= start_ts)
                       and (end_ts is None or e.get("processed_date", 0) <= end_ts)}
        else:
            entries = self.data

        total = len(entries)
        ext_subs = sum(
            1 for e in entries.values()
            if e.get("type") == "external_subtitle"
        )
        video_entries = total - ext_subs
        audio_only = sum(
            1 for e in entries.values()
            if e.get("type") != "external_subtitle"
            and e.get("audio_processed") and not e.get("subtitle_processed")
        )
        subtitle_only = sum(
            1 for e in entries.values()
            if e.get("type") != "external_subtitle"
            and e.get("subtitle_processed") and not e.get("audio_processed")
        )
        both = sum(
            1 for e in entries.values()
            if e.get("type") != "external_subtitle"
            and e.get("audio_processed") and e.get("subtitle_processed")
        )

        # Flag-based counts for web UI dashboard
        remuxed = sum(1 for e in entries.values()
                      if "remuxed" in e.get("flags", []))
        # Sum actual track counts; fall back to 1 per file for old entries
        audio_labeled = sum(
            e.get("audio_tracks_labeled",
                   1 if "audio_labeled" in e.get("flags", []) else 0)
            for e in entries.values()
        )
        subtitle_labeled = sum(
            e.get("subtitle_tracks_labeled",
                   1 if "subtitle_labeled" in e.get("flags", []) else 0)
            for e in entries.values()
            if e.get("type") != "external_subtitle"
        )
        no_action = sum(1 for e in entries.values()
                        if "no_action_required" in e.get("flags", []))

        failed_count = 0
        failed_file = self.config_dir / "failed_files.json"
        if failed_file.exists():
            try:
                with open(failed_file, "r", encoding="utf-8") as fh:
                    failed_data = json.load(fh)
                    if is_filtered:
                        failed_count = sum(
                            1 for e in failed_data.values()
                            if (start_ts is None or e.get("processed_date", 0) >= start_ts)
                            and (end_ts is None or e.get("processed_date", 0) <= end_ts)
                        )
                    else:
                        failed_count = len(failed_data)
            except (json.JSONDecodeError, IOError):
                pass

        if is_filtered:
            total_display = total
            ext_subs_display = ext_subs
        else:
            # Use the higher of tracked DB count vs last filesystem scan count
            ext_sub_scan = self._load_ext_sub_scan_count()
            ext_subs_display = max(ext_subs, ext_sub_scan)
            total_display = video_entries + ext_subs_display

        return {
            "total_tracked": total_display,
            "video_files_tracked": video_entries,
            "external_subtitles_tracked": ext_subs_display,
            "audio_only": audio_only,
            "subtitle_only": subtitle_only,
            "both": both,
            "remuxed": remuxed,
            "audio_labeled": audio_labeled,
            "subtitle_labeled": subtitle_labeled,
            "external_subtitle_labeled": ext_subs,
            "no_action_required": no_action,
            "failed": failed_count,
        }

    def get_time_series(self, granularity: str = "day", limit: int = 30) -> Dict:
        """Get time series data grouped by period for charting.

        Always returns *limit* contiguous periods ending at the current
        date, with zeros for periods that have no data.
        """
        from datetime import datetime, timedelta

        categories = [
            "no_action_required", "remuxed", "audio_labeled",
            "subtitle_labeled", "external_subtitle", "failed",
        ]

        now = datetime.now()

        def period_key(ts):
            dt = datetime.fromtimestamp(ts)
            if granularity == "week":
                start = dt - timedelta(days=dt.weekday())
                return start.strftime("%Y-%m-%d")
            elif granularity == "month":
                return dt.strftime("%Y-%m")
            elif granularity == "year":
                return dt.strftime("%Y")
            return dt.strftime("%Y-%m-%d")

        # Build a complete set of labels for the requested range
        labels: list[str] = []
        if granularity == "week":
            monday = now - timedelta(days=now.weekday())
            for i in range(limit - 1, -1, -1):
                labels.append((monday - timedelta(weeks=i)).strftime("%Y-%m-%d"))
        elif granularity == "month":
            for i in range(limit - 1, -1, -1):
                m = now.month - i
                y = now.year
                while m <= 0:
                    m += 12
                    y -= 1
                labels.append(f"{y:04d}-{m:02d}")
        elif granularity == "year":
            for i in range(limit - 1, -1, -1):
                labels.append(str(now.year - i))
        else:  # day
            for i in range(limit - 1, -1, -1):
                labels.append((now - timedelta(days=i)).strftime("%Y-%m-%d"))

        # Count entries per period
        period_counts: Dict[str, Dict[str, int]] = {}

        for entry in self.data.values():
            ts = entry.get("processed_date", 0)
            if ts == 0:
                continue
            pk = period_key(ts)
            if pk not in period_counts:
                period_counts[pk] = {cat: 0 for cat in categories}
            for flag in entry.get("flags", []):
                if flag in period_counts[pk]:
                    period_counts[pk][flag] += 1

        # Include failed entries
        failed_file = self.config_dir / "failed_files.json"
        if failed_file.exists():
            try:
                with open(failed_file, "r", encoding="utf-8") as fh:
                    failed_data = json.load(fh)
                for entry in failed_data.values():
                    ts = entry.get("processed_date", 0)
                    if ts == 0:
                        continue
                    pk = period_key(ts)
                    if pk not in period_counts:
                        period_counts[pk] = {cat: 0 for cat in categories}
                    period_counts[pk]["failed"] += 1
            except (json.JSONDecodeError, IOError):
                pass

        # Build datasets aligned to labels (0-filled for missing periods)
        datasets = {cat: [] for cat in categories}
        for label in labels:
            counts = period_counts.get(label, {})
            for cat in categories:
                datasets[cat].append(counts.get(cat, 0))

        return {"labels": labels, "datasets": datasets}

    # ── Web UI: log entries ───────────────────────────────────────────────
    def get_log_entries(self) -> list:
        """Return all entries formatted for the web UI processing log."""
        entries = []
        migrated = False
        for key, entry in self.data.items():
            # Migration: add flags for entries created before the web UI
            if "flags" not in entry:
                if entry.get("type") == "external_subtitle":
                    entry["flags"] = ["external_subtitle"]
                else:
                    entry["flags"] = ["no_action_required"]
                migrated = True
            if "filename" not in entry:
                entry["filename"] = Path(key).name
                migrated = True

            entries.append({
                "filepath": key,
                "filename": entry.get("filename", Path(key).name),
                "timestamp": entry.get("processed_date", 0),
                "flags": entry.get("flags", ["no_action_required"]),
                "audio_processed": entry.get("audio_processed", False),
                "subtitle_processed": entry.get("subtitle_processed", False),
                "type": entry.get("type"),
                "language_code": entry.get("language_code"),
            })

        if migrated:
            self._dirty = True
            self._save()

        return sorted(entries, key=lambda x: x["timestamp"], reverse=True)

    def load_failed_files(self) -> list:
        """Load failed files from the separate failed_files.json."""
        failed_file = self.config_dir / "failed_files.json"
        if not failed_file.exists():
            return []
        try:
            with open(failed_file, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            entries = []
            for key, entry in data.items():
                entries.append({
                    "filepath": key,
                    "filename": entry.get("filename", Path(key).name),
                    "timestamp": entry.get("processed_date", 0),
                    "flags": ["failed"],
                    "errors": entry.get("errors", []),
                    "failed_tracks": entry.get("failed_tracks", []),
                })
            return entries
        except (json.JSONDecodeError, IOError):
            return []

    @staticmethod
    def save_failed_files_json(
        config_dir: str,
        video_results: list,
        ext_sub_results: list,
    ) -> None:
        """Save failed files from processing results. Overwrites each run."""
        failed: Dict = {}
        for r in video_results:
            if r.get("skipped_due_to_tracking"):
                continue
            has_failure = bool(r.get("failed_tracks")) or bool(r.get("errors"))
            if has_failure:
                filepath = r.get("original_file", "")
                failed[filepath] = {
                    "filename": Path(filepath).name,
                    "processed_date": time.time(),
                    "failed_tracks": r.get("failed_tracks", []),
                    "errors": r.get("errors", []),
                }
        for esr in ext_sub_results:
            if esr.get("status") == "failed":
                filepath = esr.get("original_file", "")
                failed[filepath] = {
                    "filename": Path(filepath).name,
                    "processed_date": time.time(),
                    "errors": [esr.get("reason", "Unknown error")],
                }

        failed_path = Path(config_dir) / "failed_files.json"
        try:
            Path(config_dir).mkdir(exist_ok=True)
            with open(failed_path, "w", encoding="utf-8") as fh:
                json.dump(failed, fh, indent=2)
        except IOError as exc:
            logger.error("Could not save failed files: %s", exc)