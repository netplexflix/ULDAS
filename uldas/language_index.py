#file: uldas/language_index.py
import json
import logging
import os
import subprocess
import tempfile
import threading
import time
from pathlib import Path
from typing import Callable, Optional

from uldas.constants import (
    EXTERNAL_SUBTITLE_EXTENSIONS,
    VIDEO_EXTENSIONS,
)
from uldas import external_subtitles as ext_sub_mod
from uldas.tools import find_executable
from uldas.utils import normalize_language_code

logger = logging.getLogger(__name__)

INDEX_FILENAME = "language_index.json"


# ── Probe helper ─────────────────────────────────────────────────────────
def _probe_track_langs(ffprobe: str, file_path: Path,
                       mkvmerge: Optional[str] = None
                       ) -> "tuple[list[str], list[str]]":
    """Return ``(audio_codes, subtitle_codes)`` for *file_path*."""
    if mkvmerge:
        try:
            r = subprocess.run(
                [mkvmerge, "-J", str(file_path)],
                capture_output=True, text=True, check=True,
                encoding="utf-8", errors="replace",
            )
            data = json.loads(r.stdout) or {}
            audio: list[str] = []
            subs: list[str] = []
            for track in data.get("tracks") or []:
                ttype = (track.get("type") or "").lower()
                if ttype not in ("audio", "subtitles"):
                    continue
                props = track.get("properties") or {}
                # IETF/BCP-47 first (modern writers populate this),
                # falling back to the legacy ISO 639-2 field.
                lang = (props.get("language_ietf")
                        or props.get("language") or "")
                code = normalize_language_code(lang) if lang else "und"
                (audio if ttype == "audio" else subs).append(code)
            return audio, subs
        except (subprocess.CalledProcessError,
                json.JSONDecodeError, OSError):
            # Non-MKV containers, an mkvmerge build that can't read
            # this file, or any other failure — fall through to ffprobe.
            pass

    try:
        cmd = [
            ffprobe, "-v", "quiet", "-print_format", "json",
            "-show_streams", str(file_path),
        ]
        r = subprocess.run(
            cmd, capture_output=True, text=True, check=True,
            encoding="utf-8", errors="replace",
        )
        streams = json.loads(r.stdout).get("streams", []) or []
    except (subprocess.CalledProcessError, json.JSONDecodeError, OSError):
        return [], []

    audio = []
    subs = []
    for s in streams:
        codec_type = s.get("codec_type")
        if codec_type not in ("audio", "subtitle"):
            continue
        tags = s.get("tags") or {}
        lang = None
        for k, v in tags.items():
            if isinstance(k, str) and k.lower() in ("language", "lang"):
                lang = v
                break
        code = normalize_language_code(lang) if lang else "und"
        (audio if codec_type == "audio" else subs).append(code)
    return audio, subs


# ── Stateful index ────────────────────────────────────────────────────────
class LanguageIndex:
    """Persisted, per-file language index backed by a single JSON file."""

    def __init__(self, config_dir: str = "config", read_only: bool = False):
        self.config_dir = Path(config_dir)
        self.path = self.config_dir / INDEX_FILENAME
        self._read_only = read_only
        self._lock = threading.RLock()
        self._dirty = False
        self._data: dict = self._load()

    # ── Load / save ─────────────────────────────────────────────────────
    def _blank(self) -> dict:
        return {
            "indexed_at": None,
            "directories": [],
            "per_file": {},
            "per_ext_sub": {},
            "counts": {
                "audio": {},
                "embedded_subs": {},
                "external_subs": {},
            },
        }

    def _load(self) -> dict:
        if not self.path.exists():
            return self._blank()
        try:
            with open(self.path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            if not isinstance(data, dict):
                raise ValueError("language_index.json is not an object")
            # Defensive defaults for older / partial files
            data.setdefault("indexed_at", None)
            data.setdefault("directories", [])
            data.setdefault("per_file", {})
            data.setdefault("per_ext_sub", {})
            data.setdefault("counts", {})
            for k in ("audio", "embedded_subs", "external_subs"):
                data["counts"].setdefault(k, {})
            return data
        except (json.JSONDecodeError, ValueError, OSError) as exc:
            logger.warning("Could not load language index: %s — starting fresh", exc)
            return self._blank()

    def save_if_dirty(self) -> None:
        with self._lock:
            if self._read_only or not self._dirty:
                return
            self._recompute_counts_locked()
            _atomic_write_json(str(self.path), self._data)
            self._dirty = False

    # ── Aggregate counts (derived) ───────────────────────────────────────
    def _recompute_counts_locked(self) -> None:
        audio: dict = {}
        embedded: dict = {}
        external: dict = {}
        for info in self._data["per_file"].values():
            for code in info.get("audio", []) or []:
                audio.setdefault(code, {"tracks": 0, "files": 0})["tracks"] += 1
            for code in set(info.get("audio", []) or []):
                audio.setdefault(code, {"tracks": 0, "files": 0})["files"] += 1
            for code in info.get("embedded_subs", []) or []:
                embedded.setdefault(code, {"tracks": 0, "files": 0})["tracks"] += 1
            for code in set(info.get("embedded_subs", []) or []):
                embedded.setdefault(code, {"tracks": 0, "files": 0})["files"] += 1
        for code in self._data["per_ext_sub"].values():
            bucket = external.setdefault(code, {"tracks": 0, "files": 0})
            bucket["tracks"] += 1
            bucket["files"] += 1
        self._data["counts"] = {
            "audio": audio,
            "embedded_subs": embedded,
            "external_subs": external,
        }

    # ── Mutations ────────────────────────────────────────────────────────
    def update_file(self, file_path, audio_codes: list, sub_codes: list) -> None:
        """Record the current language tags for a video file."""
        if self._read_only:
            return
        key = os.path.abspath(str(file_path))
        with self._lock:
            prev = self._data["per_file"].get(key)
            entry = {
                "audio": list(audio_codes or []),
                "embedded_subs": list(sub_codes or []),
            }
            if prev == entry:
                return
            self._data["per_file"][key] = entry
            self._dirty = True

    def update_ext_sub(self, sub_path, language_code: Optional[str]) -> None:
        if self._read_only:
            return
        key = os.path.abspath(str(sub_path))
        code = normalize_language_code(language_code) if language_code else "und"
        with self._lock:
            prev = self._data["per_ext_sub"].get(key)
            if prev == code:
                return
            self._data["per_ext_sub"][key] = code
            self._dirty = True

    def rename_ext_sub(self, old_path, new_path, language_code: Optional[str]) -> None:
        """Move an external subtitle entry to a new path (e.g. after a rename that added a language tag to the filename)."""
        if self._read_only:
            return
        old_key = os.path.abspath(str(old_path))
        new_key = os.path.abspath(str(new_path))
        code = normalize_language_code(language_code) if language_code else "und"
        with self._lock:
            if old_key != new_key:
                self._data["per_ext_sub"].pop(old_key, None)
            self._data["per_ext_sub"][new_key] = code
            self._dirty = True

    def remove_file(self, file_path) -> None:
        if self._read_only:
            return
        key = os.path.abspath(str(file_path))
        with self._lock:
            if key in self._data["per_file"]:
                del self._data["per_file"][key]
                self._dirty = True

    def remove_ext_sub(self, sub_path) -> None:
        if self._read_only:
            return
        key = os.path.abspath(str(sub_path))
        with self._lock:
            if key in self._data["per_ext_sub"]:
                del self._data["per_ext_sub"][key]
                self._dirty = True

    def prune_outside_paths(self, configured_paths: list) -> dict:
        """Remove index entries whose path is not under any of
        *configured_paths*. Mirrors ``ProcessingTracker.prune_entries_outside_paths``.
        """
        if self._read_only:
            return {"files": 0, "ext_subs": 0}
        normalized = _normalize_path_prefixes(configured_paths)

        def outside(key: str) -> bool:
            return _key_outside(key, normalized)

        with self._lock:
            file_keys = [k for k in self._data["per_file"] if outside(k)]
            for k in file_keys:
                del self._data["per_file"][k]
            ext_keys = [k for k in self._data["per_ext_sub"] if outside(k)]
            for k in ext_keys:
                del self._data["per_ext_sub"][k]
            if file_keys or ext_keys:
                self._dirty = True
        return {"files": len(file_keys), "ext_subs": len(ext_keys)}

    def prune_ignored_tags(self, ignore_tags: list) -> dict:
        if self._read_only:
            return {"files": 0, "ext_subs": 0}
        tags_lower = [t.lower() for t in (ignore_tags or [])
                      if isinstance(t, str) and t]
        if not tags_lower:
            return {"files": 0, "ext_subs": 0}

        def matches(key: str) -> bool:
            name = os.path.basename(key)
            dot = name.rfind(".")
            stem = (name[:dot] if dot > 0 else name).lower()
            return any(tag in stem for tag in tags_lower)

        with self._lock:
            file_keys = [k for k in self._data["per_file"] if matches(k)]
            for k in file_keys:
                del self._data["per_file"][k]
            ext_keys = [k for k in self._data["per_ext_sub"] if matches(k)]
            for k in ext_keys:
                del self._data["per_ext_sub"][k]
            if file_keys or ext_keys:
                self._dirty = True
        return {"files": len(file_keys), "ext_subs": len(ext_keys)}

    # ── Metadata ─────────────────────────────────────────────────────────
    def note_indexed_now(self, directories: list) -> None:
        """Mark a full rebuild as having completed at wall-clock now."""
        if self._read_only:
            return
        with self._lock:
            self._data["indexed_at"] = time.time()
            self._data["directories"] = [
                os.path.abspath(d) for d in (directories or [])
            ]
            self._dirty = True

    def snapshot(self) -> dict:
        """Return a copy of the current index state with counts refreshed."""
        with self._lock:
            self._recompute_counts_locked()
            return json.loads(json.dumps(self._data))


# ── Module-level path helpers (shared with tracker prune) ────────────────
def _normalize_path_prefixes(configured_paths: list) -> list:
    out = []
    for p in configured_paths or []:
        if not p:
            continue
        ap = os.path.abspath(p)
        if not ap.endswith(os.sep):
            ap = ap + os.sep
        out.append(ap)
    return out


def _key_outside(key: str, normalized_prefixes: list) -> bool:
    if not normalized_prefixes:
        return True
    key_cmp = key if key.endswith(os.sep) else key + os.sep
    return not any(key_cmp.startswith(p) for p in normalized_prefixes)


# ── Atomic JSON write ────────────────────────────────────────────────────
def _atomic_write_json(path: str, data: dict) -> None:
    parent = os.path.dirname(path) or "."
    os.makedirs(parent, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=parent, prefix="lix_", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


# ── Public read helper (routes / UI) ─────────────────────────────────────
def load_language_index(config_dir: str = "config") -> Optional[dict]:
    """Return the parsed language index snapshot, or None if missing/empty."""
    path = os.path.join(config_dir, INDEX_FILENAME)
    if not os.path.exists(path):
        return None
    try:
        idx = LanguageIndex(config_dir=config_dir, read_only=True)
        snap = idx.snapshot()
        # If the file is an empty skeleton (nothing indexed yet), treat as missing.
        if (not snap.get("per_file") and not snap.get("per_ext_sub")
                and not snap.get("indexed_at")):
            return None
        # Add legacy-shaped convenience fields so the existing /api/stats
        # response can keep its shape.
        snap["video_files_indexed"] = len(snap.get("per_file") or {})
        snap["external_sub_files_indexed"] = len(snap.get("per_ext_sub") or {})
        return snap
    except Exception as exc:
        logger.warning("Could not load language index: %s", exc)
        return None


# ── Full rebuild (the "Index Languages" maintenance action) ──────────────
def build_language_index(
    directories: list,
    output_path: str,
    include_non_mkv_video: bool = False,
    ignore_tags: Optional[list] = None,
    cancel_check: Optional[Callable[[], bool]] = None,
    show_details: bool = False,
) -> dict:
    """Walk *directories* and rebuild the index from scratch."""
    ffprobe = find_executable("ffprobe")
    if not ffprobe:
        raise RuntimeError("ffprobe not found — cannot index languages")
    # Optional — mkvmerge is preferred for matroska files (it reads the
    # same fields the processing scan does), but not strictly required.
    mkvmerge = find_executable("mkvmerge")

    config_dir = os.path.dirname(output_path) or "config"
    idx = LanguageIndex(config_dir=config_dir)

    # Drop any existing entries for files under the directories we are
    # about to walk — they'll be re-populated from disk truth below.
    prefixes = _normalize_path_prefixes(directories)
    with idx._lock:
        removed_files = [k for k in list(idx._data["per_file"])
                         if not _key_outside(k, prefixes)]
        for k in removed_files:
            del idx._data["per_file"][k]
        removed_ext = [k for k in list(idx._data["per_ext_sub"])
                       if not _key_outside(k, prefixes)]
        for k in removed_ext:
            del idx._data["per_ext_sub"][k]
        if removed_files or removed_ext:
            idx._dirty = True

    video_exts: set = {".mkv"}
    if include_non_mkv_video:
        video_exts.update(VIDEO_EXTENSIONS)
    sub_exts: set = EXTERNAL_SUBTITLE_EXTENSIONS

    # Lowercase the ignore-tag list once so we can match against the
    # filename stem case-insensitively in the inner loop.
    ignore_lower: list = [
        t.lower() for t in (ignore_tags or []) if isinstance(t, str) and t
    ]

    videos_indexed = 0
    ext_subs_indexed = 0
    files_skipped = 0
    dirs_scanned = 0
    started = time.monotonic()
    last_report = started
    last_save = started
    cancelled = False

    for directory in directories or []:
        if cancel_check and cancel_check():
            cancelled = True
            break
        if not directory or not os.path.isdir(directory):
            logger.warning("Skipping missing/invalid directory: %s", directory)
            continue

        if show_details:
            logger.info("Indexing languages under: %s", directory)
        else:
            print(f"Indexing languages under: {directory}", flush=True)

        for dirpath, _dirnames, filenames in os.walk(directory, followlinks=False):
            if cancel_check and cancel_check():
                cancelled = True
                break
            dirs_scanned += 1
            for filename in filenames:
                dot = filename.rfind(".")
                if dot <= 0:
                    continue
                ext = filename[dot:].lower()
                if ext not in video_exts and ext not in sub_exts:
                    continue

                # Honour Config.ignore_tags — same case-insensitive
                # substring match the normal scan applies in _scan_tree.
                if ignore_lower:
                    stem_lower = filename[:dot].lower()
                    if any(tag in stem_lower for tag in ignore_lower):
                        files_skipped += 1
                        continue

                full = os.path.join(dirpath, filename)
                path = Path(full)

                if ext in video_exts:
                    audio_codes, sub_codes = _probe_track_langs(
                        ffprobe, path, mkvmerge=mkvmerge,
                    )
                    videos_indexed += 1
                    idx.update_file(path, audio_codes, sub_codes)
                else:  # ext in sub_exts
                    ext_subs_indexed += 1
                    lang = ext_sub_mod.get_language_tag(path)
                    idx.update_ext_sub(path, lang)

            now = time.monotonic()
            if now - last_report >= 5.0:
                last_report = now
                msg = (
                    f"Indexing… {dirs_scanned} dirs, "
                    f"{videos_indexed} videos, {ext_subs_indexed} ext subs"
                )
                if show_details:
                    logger.info(msg)
                else:
                    print(msg, flush=True)
            # Periodic flush so progress isn't lost on a crash / cancel
            if now - last_save >= 30.0:
                last_save = now
                idx.save_if_dirty()

    idx.note_indexed_now(directories)
    ignored_pruned = idx.prune_ignored_tags(ignore_tags)
    idx.save_if_dirty()

    snap = idx.snapshot()
    snap["duration_seconds"] = round(time.monotonic() - started, 1)
    snap["include_non_mkv_video"] = bool(include_non_mkv_video)
    snap["video_files_indexed"] = videos_indexed
    snap["external_sub_files_indexed"] = ext_subs_indexed
    snap["files_skipped"] = files_skipped
    snap["dirs_scanned"] = dirs_scanned
    snap["cancelled"] = cancelled
    snap["index_ignored_pruned"] = ignored_pruned
    return snap
