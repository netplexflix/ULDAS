#file: uldas/logging_setup.py

import logging
import os
import sys
import io
from datetime import datetime
from pathlib import Path

MAX_LOG_FILES = 20
LOG_DIR_NAME = "logs"


class _SafeFileHandler(logging.FileHandler):
    def emit(self, record):
        try:
            msg = self.format(record)
            stream = self.stream
            # Strip ANSI escape codes for the log file
            import re
            clean_msg = re.sub(r'\033\[[0-9;]*m', '', msg)
            stream.write(clean_msg + self.terminator)
            self.flush()
        except Exception:
            self.handleError(record)


class _SafeStreamHandler(logging.StreamHandler):
    def emit(self, record):
        try:
            msg = self.format(record)
            stream = self.stream
            try:
                stream.write(msg + self.terminator)
            except UnicodeEncodeError:
                # Fallback: encode with replace and decode back
                safe = msg.encode(stream.encoding or 'utf-8', errors='replace').decode(
                    stream.encoding or 'utf-8', errors='replace'
                )
                stream.write(safe + self.terminator)
            self.flush()
        except Exception:
            self.handleError(record)


def _prune_old_logs(log_dir: Path, max_files: int = MAX_LOG_FILES) -> None:
    """Remove oldest log files if count exceeds *max_files*."""
    log_files = sorted(log_dir.glob("uldas_*.log"), key=lambda p: p.stat().st_mtime)
    while len(log_files) > max_files:
        oldest = log_files.pop(0)
        try:
            oldest.unlink()
        except OSError:
            pass


def setup_logging(
    config_dir: str = "config",
    verbose: bool = False,
    quiet: bool = False,
) -> Path:
    log_dir = Path(config_dir) / LOG_DIR_NAME
    log_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"uldas_{timestamp}.log"

    # Prune old logs (keep MAX_LOG_FILES - 1 to make room for the new one)
    _prune_old_logs(log_dir, MAX_LOG_FILES - 1)

    # ── Root logger ──────────────────────────────────────────────────────
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)

    # Remove any pre-existing handlers (e.g. from basicConfig)
    root.handlers.clear()

    fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    # ── Console handler ──────────────────────────────────────────────────
    console_level = logging.WARNING  # default
    if verbose:
        console_level = logging.DEBUG
    elif quiet:
        console_level = logging.ERROR

    ch = _SafeStreamHandler(sys.stdout)
    ch.setLevel(console_level)
    ch.setFormatter(fmt)
    root.addHandler(ch)

    # ── File handler ─────────────────────────────────────────────────────
    fh = _SafeFileHandler(str(log_file), encoding="utf-8", errors="replace")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    root.addHandler(fh)

    # Also capture print() calls by teeing stdout
    sys.stdout = _TeeStream(sys.stdout, log_file)
    sys.stderr = _TeeStream(sys.stderr, log_file)

    return log_file


class _TeeStream:
    def __init__(self, original_stream, log_path: Path):
        self._original = original_stream
        self._log_path = log_path
        # Inherit encoding from original stream
        self.encoding = getattr(original_stream, 'encoding', 'utf-8') or 'utf-8'
        self.errors = 'replace'

    # Forward attribute lookups to the original stream
    def __getattr__(self, name):
        return getattr(self._original, name)

    def write(self, text):
        if not text:
            return
        # Write to original stream
        try:
            self._original.write(text)
        except UnicodeEncodeError:
            safe = text.encode(self.encoding, errors='replace').decode(
                self.encoding, errors='replace'
            )
            self._original.write(safe)

        # Append to log file (strip ANSI codes)
        try:
            import re
            clean = re.sub(r'\033\[[0-9;]*m', '', text)
            with open(self._log_path, "a", encoding="utf-8", errors="replace") as f:
                f.write(clean)
        except Exception:
            pass  # never crash on logging

    def flush(self):
        try:
            self._original.flush()
        except Exception:
            pass

    def isatty(self):
        return getattr(self._original, 'isatty', lambda: False)()