#file: uldas/relabel.py

"""Change-language-tag maintenance action: rewrite every audio track
tagged with one language to another, without running Whisper."""

import logging
import os
import subprocess
from pathlib import Path
from typing import Callable, Dict, List, Optional

from uldas import mp4 as mp4_mod
from uldas.language_index import (
    LanguageIndex,
    _key_outside,
    _normalize_path_prefixes,
    _probe_track_langs,
    filter_index_paths,
)
from uldas.tools import find_executable
from uldas.tracking import ProcessingTracker
from uldas.utils import convert_iso639_1_to_2, normalize_language_code

logger = logging.getLogger(__name__)


def _write_audio_languages(mkvpropedit: str, file_path: Path,
                           track_indices: List[int], code: str,
                           mp4_inplace: bool, audio_count: int) -> bool:
    """Set *code* on the given audio ordinals of *file_path*."""
    if mp4_inplace:
        lang3 = convert_iso639_1_to_2(code.split("-")[0].lower())
        return all(
            mp4_mod.set_track_language(file_path, "audio", idx, lang3,
                                       expected_count=audio_count)
            for idx in track_indices
        )
    cmd = [mkvpropedit, str(file_path)]
    for idx in track_indices:
        cmd += ["--edit", f"track:a{idx + 1}", "--set", f"language={code}"]
    try:
        subprocess.run(cmd, capture_output=True, text=True, check=True,
                       encoding="utf-8", errors="replace")
        return True
    except subprocess.CalledProcessError as exc:
        logger.error("mkvpropedit failed for %s: %s", file_path,
                     (exc.stdout or exc.stderr or str(exc)).strip())
        return False
    except OSError as exc:
        logger.error("mkvpropedit failed for %s: %s", file_path, exc)
        return False


def relabel_audio_language(
    config,
    from_code: str,
    to_code: str,
    cancel_check: Optional[Callable[[], bool]] = None,
) -> Dict:
    """Change every audio track tagged *from_code* to *to_code* across the
    configured libraries.

    Candidates come from the language index (kept up to date by every
    processing run); each file is re-probed before writing, so a stale
    index entry is refreshed instead of acted on.  Honors ``dry_run``.
    """
    src = normalize_language_code(from_code)
    dst = normalize_language_code(to_code)
    dry_run = bool(config.dry_run)

    stats = {
        "from": src, "to": dst, "dry_run": dry_run,
        "files_matched": 0, "files_changed": 0, "tracks_changed": 0,
        "files_failed": 0, "files_stale": 0, "files_unsupported": 0,
        "files_missing": 0, "files_ignored": 0, "cancelled": False,
    }

    ffprobe = find_executable("ffprobe")
    mkvmerge = find_executable("mkvmerge")
    mkvpropedit = find_executable("mkvpropedit")

    index = LanguageIndex(config_dir="config", read_only=dry_run)
    tracker = None
    if config.use_tracking:
        tracker = ProcessingTracker("config", read_only=dry_run)

    roots = _normalize_path_prefixes(list(config.path or []))
    in_scope = [p for p in index.files_with_audio_language(src)
                if not _key_outside(os.path.abspath(p), roots)]
    files, stats["files_missing"], stats["files_ignored"] = (
        filter_index_paths(in_scope, config)
    )
    stats["files_matched"] = len(files)

    prefix = "[DRY RUN] " if dry_run else ""
    logger.info("%sChange audio language tag %s → %s: %d file(s) in the "
                "language index", prefix, src, dst, len(files))
    if not files:
        logger.info("Nothing to relabel. Files processed by older ULDAS "
                    "versions only appear after running Index languages.")

    try:
        for n, fp in enumerate(files, 1):
            if cancel_check is not None and cancel_check():
                logger.info("Cancellation requested — stopping relabel run")
                stats["cancelled"] = True
                break

            mp4_inplace = bool(config.mp4_support) and mp4_mod.is_mp4(fp)
            if not mp4_inplace and fp.suffix.lower() != ".mkv":
                logger.warning("Skipping %s: only MKV (and MP4 with MP4 "
                               "Support) can be relabeled in place", fp.name)
                stats["files_unsupported"] += 1
                continue

            audio_codes, sub_codes = _probe_track_langs(
                ffprobe, fp, mkvmerge=mkvmerge, mp4_inplace=bool(config.mp4_support),
            )
            targets = [i for i, c in enumerate(audio_codes) if c == src]
            if not targets:
                # The index was out of date; record what is really there.
                index.update_file(fp, audio_codes, sub_codes)
                stats["files_stale"] += 1
                continue

            tracks_desc = ", ".join(f"track{i}" for i in targets)
            if dry_run:
                print(f"[DRY RUN] [{n}/{len(files)}] Would relabel {fp.name}: "
                      f"{tracks_desc} {src} -> {dst}")
                stats["files_changed"] += 1
                stats["tracks_changed"] += len(targets)
                continue

            print(f"[{n}/{len(files)}] Relabeling {fp.name}: {tracks_desc} "
                  f"{src} -> {dst}", flush=True)
            if not _write_audio_languages(mkvpropedit, fp, targets, dst,
                                          mp4_inplace, len(audio_codes)):
                stats["files_failed"] += 1
                continue
            stats["files_changed"] += 1
            stats["tracks_changed"] += len(targets)

            if tracker is not None and tracker.get_entry(fp):
                tracker.merge_audio_relabel(
                    fp,
                    [{"track_index": i, "detected_language": dst,
                      "previous_language": src, "relabeled": True}
                     for i in targets],
                    audio_ok=False,
                    add_flags=("relabeled",),
                )

            audio_codes, sub_codes = _probe_track_langs(
                ffprobe, fp, mkvmerge=mkvmerge, mp4_inplace=bool(config.mp4_support),
            )
            index.update_file(fp, audio_codes, sub_codes)

            if n % 25 == 0:
                index.save_if_dirty()
                if tracker is not None:
                    tracker.save_if_dirty()
    finally:
        index.save_if_dirty()
        if tracker is not None:
            tracker.save_if_dirty()

    return stats
