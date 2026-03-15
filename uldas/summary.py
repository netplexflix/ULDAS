#file: uldas/summary.py
from pathlib import Path
from typing import List, Dict, Optional

from uldas.config import Config
from uldas.utils import format_duration

# ANSI colours
YELLOW = "\033[93m"
RED = "\033[91m"
GREEN = "\033[92m"
CYAN = "\033[96m"
RESET = "\033[0m"


def print_detailed_summary(
    video_results: List[Dict],
    ext_sub_results: List[Dict],
    config: Config,
    runtime_seconds: float,
    detector=None,
    total_ext_subs_found: int = 0,
    total_new_ext_subs: int = 0,
) -> None:

    # ── Tracking Statistics (printed first, above everything else) ───────
    if config.use_tracking and detector and hasattr(detector, "tracker"):
        stats = detector.tracker.get_stats()
        ext_sub_count = max(stats["external_subtitles_tracked"],
                            total_ext_subs_found)
        video_count = stats["video_files_tracked"]
        total_count = video_count + ext_sub_count

        if total_count:
            print(f"\n{'=' * 60}")
            print("TRACKING STATISTICS")
            print(f"{'=' * 60}")
            print(f"  Video files tracked: {video_count}")
            if ext_sub_count:
                print(f"  External subtitles tracked: {ext_sub_count}")
            print(f"  Total entries tracked: {total_count}")

    # ── Audio Processing Summary ─────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print("AUDIO PROCESSING SUMMARY")
    print(f"{'=' * 60}")

    total = len(video_results)
    acted = success = failed = processed = skipped = 0
    silent_files: list[str] = []

    for r in video_results:
        if r.get("skipped_due_to_tracking"):
            skipped += 1
            continue

        name = Path(r["original_file"]).name
        actions: list[str] = []
        has_fail = False
        has_silent = False

        if r["was_remuxed"]:
            actions.append("remuxed")

        for t in sorted(r["processed_tracks"], key=lambda x: x["track_index"]):
            idx, lang, prev = t["track_index"], t["detected_language"], t.get("previous_language", "und")
            if lang == "zxx":
                actions.append(f"{YELLOW}track{idx}: {prev} -> {lang} (no speech){RESET}")
                has_silent = True
            elif prev == lang:
                actions.append(f"track{idx}: {prev} -> {lang}")
            else:
                actions.append(f"{CYAN}track{idx}: {prev} -> {lang}{RESET}")

        for idx in sorted(r.get("failed_tracks", [])):
            actions.append(f"{RED}track{idx}: failed{RESET}")
            has_fail = True

        if actions:
            print(f"{name}: {', '.join(actions)}")
            acted += 1
            processed += 1
            if has_fail:
                failed += 1
            else:
                success += 1
            if has_silent:
                silent_files.append(name)
        elif r["errors"]:
            print(f"{name}: {RED}error{RESET} – {r['errors'][0]}")
            acted += 1
            failed += 1

    if skipped:
        print(f"\n{GREEN}Skipped {skipped} already-processed file(s){RESET}")
    if acted:
        print(f"\nShowing {acted} files that required action (out of {total} total)")
        parts = []
        if success:
            parts.append(f"Successfully processed {success} files")
        if failed:
            parts.append(f"{RED}{failed} files failed!{RESET}")
        if parts:
            print(". ".join(parts))
    else:
        print("No tracks required any action")

    if silent_files:
        print(f"\n{YELLOW}⚠️  WARNING: Silent content detected in {len(silent_files)} file(s){RESET}")
        for f in silent_files[:5]:
            print(f"{YELLOW}   - {f}{RESET}")
        if len(silent_files) > 5:
            print(f"{YELLOW}   ... and {len(silent_files) - 5} more{RESET}")

    if detector and detector.deletion_failures:
        print(f"\n{YELLOW}⚠️  WARNING: {len(detector.deletion_failures)} original file(s) could not be deleted{RESET}")
        for df in detector.deletion_failures:
            print(f"{YELLOW}   - {Path(df['original_file']).name}{RESET}")

    # ── Subtitle summary (embedded) ──────────────────────────────────────
    if config.process_subtitles:
        print(f"\n{'=' * 60}")
        print("SUBTITLE PROCESSING SUMMARY (EMBEDDED)")
        print(f"{'=' * 60}")

        t_found = t_proc = t_fail = t_skip = t_forced = t_sdh = 0
        for r in video_results:
            sr = r.get("subtitle_results")
            if not sr:
                continue
            t_found += sr["subtitle_tracks_found"]
            t_proc += len(sr["processed_subtitle_tracks"])
            t_fail += len(sr["failed_subtitle_tracks"])
            t_skip += len(sr.get("skipped_subtitle_tracks", []))

            for st in sr["processed_subtitle_tracks"]:
                if st.get("is_forced"):
                    t_forced += 1
                if st.get("is_sdh"):
                    t_sdh += 1

            if sr["processed_subtitle_tracks"] or sr["failed_subtitle_tracks"] or sr.get("skipped_subtitle_tracks"):
                name = Path(r["original_file"]).name
                print(f"\n{name}:")
                for st in sr["processed_subtitle_tracks"]:
                    flags = []
                    if st.get("is_forced"):
                        flags.append(f"{YELLOW}Forced{RESET}")
                    if st.get("is_sdh"):
                        flags.append(f"{CYAN}SDH{RESET}")
                    flag_str = f" [{', '.join(flags)}]" if flags else ""
                    prev = st["previous_language"]
                    lang = st["detected_language"]
                    conf = st["confidence"]
                    colour = CYAN if prev != lang else ""
                    reset = RESET if colour else ""
                    print(f"  {colour}subtitle track{st['track_index']}: {prev} -> {lang}{reset} (conf: {conf:.2f}){flag_str}")
                for st in sr.get("skipped_subtitle_tracks", []):
                    print(f"  {YELLOW}subtitle track{st['track_index']}: skipped{RESET} "
                          f"(detected: {st['detected_language']}, conf: {st['confidence']:.2f})")
                for idx in sr["failed_subtitle_tracks"]:
                    print(f"  {RED}subtitle track{idx}: failed{RESET}")

        if t_found:
            print(f"\nSubtitle tracks found: {t_found}")
            print(f"Successfully processed: {t_proc}")
            if t_skip:
                print(f"{YELLOW}Skipped (low confidence): {t_skip}{RESET}")
            if t_fail:
                print(f"{RED}Failed: {t_fail}{RESET}")
            if t_forced:
                print(f"Forced subtitles detected: {t_forced}")
            if t_sdh:
                print(f"SDH subtitles detected: {t_sdh}")
        else:
            print("\nNo tracks required any action")

    # ── External subtitle summary ────────────────────────────────────────
    if config.process_external_subtitles:
        print(f"\n{'=' * 60}")
        print("EXTERNAL SUBTITLE PROCESSING SUMMARY")
        print(f"{'=' * 60}")

        ext_proc = ext_fail = ext_skip = 0
        ext_total = len(ext_sub_results)

        for esr in ext_sub_results:
            status = esr.get("status", "failed")
            if status == "processed":
                ext_proc += 1
                orig_name = Path(esr["original_file"]).name
                new_name = Path(esr["new_file"]).name if esr.get("new_file") else "?"
                lang = esr.get("detected_language", "?")
                conf = esr.get("confidence", 0.0)
                sdh_tag = f" [{CYAN}SDH{RESET}]" if esr.get("is_sdh") else ""
                print(f"  {CYAN}{orig_name} → {new_name}{RESET} (lang: {lang}, conf: {conf:.2f}){sdh_tag}")
            elif status == "skipped":
                ext_skip += 1
                sub_name = Path(esr["original_file"]).name
                reason = esr.get("reason", "unknown")
                lang = esr.get("detected_language", "?")
                conf = esr.get("confidence", 0.0)
                print(f"  {YELLOW}{sub_name}: skipped{RESET} "
                      f"(detected: {lang}, conf: {conf:.2f}, reason: {reason})")
            else:
                ext_fail += 1
                sub_name = Path(esr["original_file"]).name
                reason = esr.get("reason", "unknown")
                print(f"  {RED}{sub_name}: failed ({reason}){RESET}")

        new_count = total_new_ext_subs if total_new_ext_subs else ext_total
        if new_count:
            print(f"\nNew external subtitle files found: {new_count}")
            print(f"Successfully processed: {ext_proc}")
            if ext_skip:
                print(f"{YELLOW}Skipped: {ext_skip}{RESET}")
            if ext_fail:
                print(f"{RED}Failed: {ext_fail}{RESET}")
        else:
            print("\nNo external subtitle files required any action")

    print(f"\nTotal runtime: {format_duration(runtime_seconds)}")
    if config.dry_run:
        print("(Dry run – no files were actually modified)")
    print()