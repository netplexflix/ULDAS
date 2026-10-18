#file: uldas/tracking.py

import json
import os
import shutil
import tempfile
import time
import logging
from pathlib import Path
from typing import Callable, Dict, Optional

logger = logging.getLogger(__name__)


def _json_default(obj):
    """Coerce non-JSON-native values (numpy bool/float, etc.) to primitives.

    Logs a warning so we can identify the offending type/value next time
    a serialization mismatch happens.
    """
    # bool MUST be checked before int — bool is a subclass of int
    if isinstance(obj, bool):
        coerced = bool(obj)
    elif isinstance(obj, int):
        coerced = int(obj)
    elif isinstance(obj, float):
        coerced = float(obj)
    elif hasattr(obj, "item"):  # numpy scalar / 0-d array
        coerced = obj.item()
    else:
        coerced = str(obj)  # last-resort, never raise
    logger.warning(
        "JSON coercion: type=%s value=%r -> %r",
        type(obj).__name__, obj, coerced,
    )
    return coerced


def _refresh_backup(path: Path) -> None:
    """Make ``<file>.bak`` a copy of the current *path* without ever
    removing *path* itself.

    The file stays in place until the new version replaces it atomically,
    so a crash or SIGTERM mid-save never leaves the tracker missing.  A
    hard link costs nothing; filesystems without them get a copy.
    """
    if not path.exists():
        return
    backup = path.with_suffix(".json.bak")
    staging = path.with_suffix(".json.bak.tmp")
    try:
        try:
            if staging.exists():
                staging.unlink()
            os.link(path, staging)
        except OSError:
            shutil.copy2(path, staging)
        os.replace(staging, backup)
    except OSError as exc:
        logger.debug("Could not refresh backup of %s: %s", path, exc)


def _merge_track_details(old_details, new_details) -> list:
    """Replace *old_details* entries per ``track_index`` with *new_details*."""
    new_details = list(new_details or [])
    new_idxs = {t["track_index"] for t in new_details}
    merged = [t for t in (old_details or [])
              if t.get("track_index") not in new_idxs]
    merged.extend(new_details)
    merged.sort(key=lambda t: t.get("track_index", 0))
    return merged


def _count_newly_labeled(details: list) -> int:
    """Tracks given a real language that differs from their previous tag.

    A re-detection that confirms the existing tag (``en`` → ``en``) is not
    a new label, so it must not add to the lifetime totals.
    """
    from uldas.utils import normalize_language_code

    count = 0
    for t in details:
        code = normalize_language_code(t.get("detected_language") or "")
        if code in ("zxx", "und"):
            continue
        if code == normalize_language_code(t.get("previous_language") or ""):
            continue
        count += 1
    return count


class ProcessingTracker:
    def __init__(self, config_dir: str = "config", read_only: bool = False):
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(exist_ok=True)
        self.tracker_file = self.config_dir / "processed_files.json"
        self.ext_sub_file = self.config_dir / "processed_external_subtitles.json"
        self._read_only = read_only
        self.data: Dict = self._load()
        self.ext_sub_data: Dict = self._load_ext_sub()
        self._dirty = False
        self._ext_sub_dirty = False
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
                    json.dump(self.data, fh, indent=2, default=_json_default)
                    fh.flush()
                    os.fsync(fh.fileno())
            except BaseException:
                os.unlink(tmp_path)
                raise

            _refresh_backup(self.tracker_file)
            os.replace(tmp_path, str(self.tracker_file))
            self._dirty = False
        except (IOError, TypeError, ValueError) as exc:
            logger.error("Could not save tracking file: %s", exc, exc_info=True)

    # ── External subtitle persistence ────────────────────────────────────
    def _load_ext_sub(self) -> Dict:
        if self.ext_sub_file.exists():
            try:
                with open(self.ext_sub_file, "r", encoding="utf-8") as fh:
                    data = json.load(fh)
                if not isinstance(data, dict):
                    logger.warning("External subtitle tracking file is not a dict, ignoring.")
                    return {}
                return data
            except (json.JSONDecodeError, IOError) as exc:
                backup = self.ext_sub_file.with_suffix(".json.bak")
                if backup.exists():
                    try:
                        with open(backup, "r", encoding="utf-8") as fh:
                            data = json.load(fh)
                        if isinstance(data, dict):
                            logger.warning(
                                "Primary ext-sub tracking file corrupt (%s), "
                                "restored from backup (%d entries).",
                                exc, len(data),
                            )
                            return data
                    except (json.JSONDecodeError, IOError):
                        pass
                logger.warning("Could not load ext-sub tracking file: %s. Starting fresh.", exc)
                return {}
        return {}

    def _save_ext_sub(self) -> None:
        if self._read_only:
            return
        try:
            tmp_fd, tmp_path = tempfile.mkstemp(
                dir=str(self.config_dir), suffix=".tmp", prefix="es_",
            )
            try:
                with os.fdopen(tmp_fd, "w", encoding="utf-8") as fh:
                    json.dump(self.ext_sub_data, fh, indent=2, default=_json_default)
                    fh.flush()
                    os.fsync(fh.fileno())
            except BaseException:
                os.unlink(tmp_path)
                raise

            _refresh_backup(self.ext_sub_file)
            os.replace(tmp_path, str(self.ext_sub_file))
            self._ext_sub_dirty = False
        except (IOError, TypeError, ValueError) as exc:
            logger.error("Could not save ext-sub tracking file: %s", exc, exc_info=True)

    def save_ext_sub_if_dirty(self) -> None:
        """Write external subtitle data to disk only if changed."""
        if self._ext_sub_dirty:
            self._save_ext_sub()

    def _ensure_migrated(self) -> None:
        """Ensure all entries have proper flags and filenames.

        Also migrates external subtitle entries from the main
        ``processed_files.json`` into the separate
        ``processed_external_subtitles.json``.
        """
        migrated = False
        ext_sub_keys_to_move: list[str] = []

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

            # Collect external subtitle entries to migrate to new file
            if entry.get("type") == "external_subtitle":
                ext_sub_keys_to_move.append(key)

        # Move external subtitle entries to the dedicated JSON
        if ext_sub_keys_to_move:
            for key in ext_sub_keys_to_move:
                if key not in self.ext_sub_data:
                    self.ext_sub_data[key] = self.data[key]
                    self._ext_sub_dirty = True
                del self.data[key]
            migrated = True

        if migrated:
            self._dirty = True
            self._save()
        if self._ext_sub_dirty:
            self._save_ext_sub()

    def save_if_dirty(self) -> None:
        """Write to disk only if in-memory data has changed."""
        if self._dirty:
            self._save()
        if self._ext_sub_dirty:
            self._save_ext_sub()

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

        return actionable, skipped, key_cache

    def is_unchanged(self, key: str) -> bool:
        """True if the file at *key* still has its tracked size and mtime.

        A file replaced at the same path (an upgrade keeping the filename,
        an in-place re-encode) returns False so it gets processed again.
        """
        entry = self.data.get(key)
        if not entry:
            return False
        try:
            stat = os.stat(key)
        except OSError:
            return False
        return (entry.get("size") == stat.st_size
                and abs(entry.get("mtime", 0) - stat.st_mtime) <= 1)

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
        track_details: list = None,
        subtitle_details: list = None,
        original_format: str = None,
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
        entry = {
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
        if track_details:
            entry["track_details"] = track_details
        if subtitle_details:
            entry["subtitle_details"] = subtitle_details
        if original_format:
            entry["original_format"] = original_format
        self.data[key] = entry
        self._dirty = True

    def merge_audio_relabel(
        self,
        file_path: Path,
        new_details: list,
        audio_ok: bool = True,
        key: str = None,
        add_flags: tuple = (),
        subtitle_details: list = None,
        subtitle_ok: bool = None,
    ) -> None:
        """Merge re-detected or relabeled tracks into an existing entry.

        Used by reprocess-by-language, reprocess-specific-files and the
        change-language-tag action.  Track details are replaced per track
        index, the derived flags (silent_content / inconclusive /
        no_action_required) are recomputed, and lifetime label counts are
        preserved.  Subtitle details are merged the same way when
        *subtitle_details* is given.  The file is re-stat'ed, so an
        in-place tag edit doesn't look like a changed file on the next run.
        """
        existing = self.get_entry(file_path, key=key) or {}

        merged_details = _merge_track_details(existing.get("track_details"),
                                              new_details)
        new_details = list(new_details or [])

        prior_flags = list(existing.get("flags") or [])
        derived = ("silent_content", "inconclusive", "relabeled",
                   "no_action_required")
        flags = [f for f in prior_flags if f not in derived]
        for flag in add_flags:
            if flag not in flags:
                flags.append(flag)
        if any(t.get("detected_language") == "zxx" for t in merged_details):
            flags.append("silent_content")
        if any(t.get("inconclusive") for t in merged_details):
            flags.append("inconclusive")
        # A later re-detection replaces a relabeled track's detail, so the
        # flag only stays while some track still carries the relabel.
        if any(t.get("relabeled") for t in merged_details):
            flags.append("relabeled")

        # Preserve prior counts: relabels do not add to the lifetime total.
        audio_count = int(existing.get("audio_tracks_labeled", 0))
        if audio_count == 0 and "audio_labeled" not in prior_flags and new_details:
            audio_count = _count_newly_labeled(new_details)
            if audio_count and "audio_labeled" not in flags:
                flags.append("audio_labeled")

        subtitle_processed = bool(existing.get("subtitle_processed", False))
        subtitle_count = int(existing.get("subtitle_tracks_labeled", 0))
        merged_sub_details = existing.get("subtitle_details")
        if subtitle_details is not None:
            subtitle_details = list(subtitle_details)
            merged_sub_details = _merge_track_details(merged_sub_details,
                                                      subtitle_details)
            subtitle_processed = subtitle_processed or bool(subtitle_ok)
            if (subtitle_count == 0 and "subtitle_labeled" not in prior_flags
                    and subtitle_details):
                subtitle_count = _count_newly_labeled(subtitle_details)
                if subtitle_count and "subtitle_labeled" not in flags:
                    flags.append("subtitle_labeled")
        if not flags:
            flags.append("no_action_required")

        self.mark_processed(
            file_path,
            bool(existing.get("audio_processed", False)) or audio_ok,
            subtitle_processed,
            key=key,
            flags=flags,
            audio_tracks_labeled=audio_count,
            subtitle_tracks_labeled=subtitle_count,
            track_details=merged_details or None,
            subtitle_details=merged_sub_details or None,
            original_format=existing.get("original_format"),
        )

    # ── External subtitle tracking ───────────────────────────────────────
    def is_external_subtitle_tracked(self, subtitle_path: Path) -> bool:
        """Check if an external subtitle file is already tracked (any flag)."""
        key = os.path.abspath(str(subtitle_path))
        if key not in self.ext_sub_data:
            return False

        entry = self.ext_sub_data[key]
        try:
            stat = subtitle_path.stat()
        except OSError:
            del self.ext_sub_data[key]
            self._ext_sub_dirty = True
            return False

        if (entry.get("size") != stat.st_size
                or abs(entry.get("mtime", 0) - stat.st_mtime) > 1):
            del self.ext_sub_data[key]
            self._ext_sub_dirty = True
            return False

        return True

    def mark_external_subtitle_tracked(
        self,
        subtitle_path: Path,
        language_code: str = None,
    ) -> None:
        """Track an already-tagged external subtitle as no_action_required.

        Does NOT flush to disk — call :meth:`save_ext_sub_if_dirty`
        after the scan loop for batch efficiency.
        """
        key = os.path.abspath(str(subtitle_path))
        if key in self.ext_sub_data:
            return
        try:
            stat = subtitle_path.stat()
        except OSError:
            return
        self.ext_sub_data[key] = {
            "type": "external_subtitle",
            "size": stat.st_size,
            "mtime": stat.st_mtime,
            "language_code": language_code,
            "processed_date": time.time(),
            "filename": subtitle_path.name,
            "flags": ["no_action_required"],
        }
        self._ext_sub_dirty = True

    def mark_external_subtitle_skipped(self, subtitle_path: Path,
                                       reason: str) -> None:
        """Track an external subtitle whose language couldn't be determined.

        Detection is deterministic, so an unchanged file gives the same
        answer next run: it is skipped until its size or mtime changes (or
        subtitles are reprocessed).  Flagged 'inconclusive', not labeled.
        """
        key = os.path.abspath(str(subtitle_path))
        try:
            stat = subtitle_path.stat()
        except OSError:
            return
        self.ext_sub_data[key] = {
            "type": "external_subtitle",
            "size": stat.st_size,
            "mtime": stat.st_mtime,
            "language_code": None,
            "reason": reason,
            "processed_date": time.time(),
            "filename": subtitle_path.name,
            "flags": ["inconclusive"],
        }
        self._ext_sub_dirty = True

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
        self.ext_sub_data[key] = {
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
            if old_key in self.ext_sub_data:
                del self.ext_sub_data[old_key]
        self._ext_sub_dirty = True

    def prune_missing_files(
        self,
        directory: str = None,
        seen_paths: "Optional[set[str]]" = None,
        on_remove_video: "Optional[Callable[[str], None]]" = None,
        on_remove_ext_sub: "Optional[Callable[[str], None]]" = None,
        extensions: "Optional[set[str]]" = None,
    ) -> int:
        """Remove entries for files that no longer exist on disk.

        If *directory* is given, only entries under that directory are
        checked — this avoids accidentally pruning entries on drives
        that happen to be offline (or in a sibling library such as
        ``Movies 4K`` next to ``Movies``).

        If *seen_paths* is provided, it is taken as the authoritative
        set of absolute paths present on disk in *directory* (as
        collected during a recent ``os.walk``).  Tracker entries whose
        keys are absent from this set are pruned without any extra
        filesystem calls — much faster than ``os.path.exists`` per key.
        *extensions* (lower-case, with the dot) limits that to the file
        types the scan looked for: a file the current settings don't
        scan is absent from *seen_paths* without being missing.
        """
        prefix = None
        if directory:
            prefix = os.path.abspath(directory)
            if not prefix.endswith(os.sep):
                prefix += os.sep

        def _missing(key: str) -> bool:
            if prefix and not key.startswith(prefix):
                return False
            if seen_paths is not None:
                if (extensions is not None
                        and os.path.splitext(key)[1].lower() not in extensions):
                    return False
                return key not in seen_paths
            return not os.path.exists(key)

        removals = [key for key in self.data if _missing(key)]
        if removals:
            for key in removals:
                if on_remove_video is not None:
                    try:
                        on_remove_video(key)
                    except Exception:
                        logger.debug("on_remove_video failed for %s", key, exc_info=True)
                del self.data[key]
            self._dirty = True

        ext_removals = [key for key in self.ext_sub_data if _missing(key)]
        if ext_removals:
            for key in ext_removals:
                if on_remove_ext_sub is not None:
                    try:
                        on_remove_ext_sub(key)
                    except Exception:
                        logger.debug("on_remove_ext_sub failed for %s", key, exc_info=True)
                del self.ext_sub_data[key]
            self._ext_sub_dirty = True

        return len(removals) + len(ext_removals)

    def prune_entries_outside_paths(self, configured_paths: list) -> dict:
        """Remove tracker entries whose path is not under any of configured_paths."""
        normalized = []
        for p in configured_paths or []:
            if not p:
                continue
            ap = os.path.abspath(p)
            if not ap.endswith(os.sep):
                ap = ap + os.sep
            normalized.append(ap)

        def _outside(key: str) -> bool:
            if not normalized:
                # No libraries configured — treat every entry as outside.
                return True
            key_cmp = key if key.endswith(os.sep) else key + os.sep
            return not any(key_cmp.startswith(p) for p in normalized)

        vid_removals = [k for k in self.data if _outside(k)]
        for k in vid_removals:
            del self.data[k]
        if vid_removals:
            self._dirty = True

        ext_removals = [k for k in self.ext_sub_data if _outside(k)]
        for k in ext_removals:
            del self.ext_sub_data[k]
        if ext_removals:
            self._ext_sub_dirty = True

        self.save_if_dirty()

        # Also drop matching entries from failed_files.json so the
        # dashboard's "Failed" count reflects the cleanup.
        failed_removed = 0
        failed_file = self.config_dir / "failed_files.json"
        if failed_file.exists():
            try:
                with open(failed_file, "r", encoding="utf-8") as fh:
                    failed_data = json.load(fh)
                kept = {k: v for k, v in failed_data.items() if not _outside(k)}
                failed_removed = len(failed_data) - len(kept)
                if failed_removed:
                    tmp = failed_file.with_suffix(".json.tmp")
                    with open(tmp, "w", encoding="utf-8") as fh:
                        json.dump(kept, fh, indent=2, default=_json_default)
                    os.replace(tmp, failed_file)
            except (json.JSONDecodeError, IOError, OSError) as exc:
                logger.warning("Could not prune failed_files.json: %s", exc)

        return {
            "videos": len(vid_removals),
            "ext_subs": len(ext_removals),
            "failed": failed_removed,
        }

    def count_tracked_audio_languages(self) -> list:
        counts: dict = {}
        for entry in self.data.values():
            if entry.get("type") == "external_subtitle":
                continue
            details = entry.get("track_details") or []
            seen_in_file: set = set()
            for t in details:
                code = (t.get("detected_language") or "").strip().lower()
                if not code:
                    continue
                bucket = counts.setdefault(code, {"tracks": 0, "files": 0})
                bucket["tracks"] += 1
                seen_in_file.add(code)
            for code in seen_in_file:
                counts[code]["files"] += 1
        return [
            {"code": c, "tracks": v["tracks"], "files": v["files"]}
            for c, v in sorted(counts.items(), key=lambda x: (-x[1]["tracks"], x[0]))
        ]

    def clear_entry(self, file_path: Path) -> None:
        key = os.path.abspath(str(file_path))
        if key in self.data:
            del self.data[key]
            self._dirty = True
        if key in self.ext_sub_data:
            del self.ext_sub_data[key]
            self._ext_sub_dirty = True
        self.save_if_dirty()

    def clear_all(self) -> None:
        self.data = {}
        self._save()
        self.ext_sub_data = {}
        self._save_ext_sub()

    def get_stats(self, start_ts: float = None, end_ts: float = None) -> Dict:
        is_filtered = start_ts is not None or end_ts is not None
        if is_filtered:
            entries = {k: e for k, e in self.data.items()
                       if (start_ts is None or e.get("processed_date", 0) >= start_ts)
                       and (end_ts is None or e.get("processed_date", 0) <= end_ts)}
            ext_entries = {k: e for k, e in self.ext_sub_data.items()
                          if (start_ts is None or e.get("processed_date", 0) >= start_ts)
                          and (end_ts is None or e.get("processed_date", 0) <= end_ts)}
        else:
            entries = self.data
            ext_entries = self.ext_sub_data

        video_entries = len(entries)
        ext_subs_total = len(ext_entries)
        ext_subs_labeled = sum(
            1 for e in ext_entries.values()
            if "external_subtitle" in e.get("flags", [])
        )

        audio_only = sum(
            1 for e in entries.values()
            if e.get("audio_processed") and not e.get("subtitle_processed")
        )
        subtitle_only = sum(
            1 for e in entries.values()
            if e.get("subtitle_processed") and not e.get("audio_processed")
        )
        both = sum(
            1 for e in entries.values()
            if e.get("audio_processed") and e.get("subtitle_processed")
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

        total_display = video_entries + ext_subs_total

        return {
            "total_tracked": total_display,
            "video_files_tracked": video_entries,
            "external_subtitles_tracked": ext_subs_total,
            "audio_only": audio_only,
            "subtitle_only": subtitle_only,
            "both": both,
            "remuxed": remuxed,
            "audio_labeled": audio_labeled,
            "subtitle_labeled": subtitle_labeled,
            "external_subtitle_labeled": ext_subs_labeled,
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

        # Include external subtitle entries
        for entry in self.ext_sub_data.values():
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
            # Migration: a 'relabeled' flag outlived its track detail
            # before the flag was derived from the details.
            if ("relabeled" in entry["flags"] and not any(
                    t.get("relabeled") for t in entry.get("track_details") or [])):
                entry["flags"] = ([f for f in entry["flags"] if f != "relabeled"]
                                  or ["no_action_required"])
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
                "track_details": entry.get("track_details"),
                "subtitle_details": entry.get("subtitle_details"),
                "original_format": entry.get("original_format"),
                "audio_tracks_labeled": entry.get("audio_tracks_labeled", 0),
                "subtitle_tracks_labeled": entry.get("subtitle_tracks_labeled", 0),
            })

        if migrated:
            self._dirty = True
            self.save_if_dirty()

        # Include external subtitle entries from the dedicated JSON
        for key, entry in self.ext_sub_data.items():
            entries.append({
                "filepath": key,
                "filename": entry.get("filename", Path(key).name),
                "timestamp": entry.get("processed_date", 0),
                "flags": entry.get("flags", ["external_subtitle"]),
                "audio_processed": False,
                "subtitle_processed": False,
                "type": entry.get("type"),
                "language_code": entry.get("language_code"),
            })

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
        replace: bool = True,
    ) -> None:
        """Save failed files from processing results.

        With *replace* (a complete library run) the list is rebuilt from
        this run.  Otherwise (selected files, a cancelled run) only the
        entries of files processed in this run are updated.
        """
        failed: Dict = {}
        failed_path = Path(config_dir) / "failed_files.json"
        if not replace and failed_path.exists():
            try:
                with open(failed_path, "r", encoding="utf-8") as fh:
                    existing = json.load(fh)
                touched = {r.get("original_file") for r in video_results
                           if not r.get("skipped_due_to_tracking")}
                touched |= {e.get("original_file") for e in ext_sub_results}
                if isinstance(existing, dict):
                    failed = {k: v for k, v in existing.items()
                              if k not in touched}
            except (json.JSONDecodeError, IOError):
                failed = {}
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

        try:
            Path(config_dir).mkdir(exist_ok=True)
            with open(failed_path, "w", encoding="utf-8") as fh:
                json.dump(failed, fh, indent=2, default=_json_default)
        except (IOError, TypeError, ValueError) as exc:
            logger.error("Could not save failed files: %s", exc, exc_info=True)