#file: uldas/scheduler_state.py

import json
import os
import threading
import time
import logging
from datetime import datetime
from typing import Optional

logger = logging.getLogger(__name__)


class SchedulerState:
    """Thread-safe shared state between the scheduler (main thread) and web UI (daemon thread).

    Uses ``threading.Event`` objects for cross-thread signaling and maintains
    a ``config/status.json`` file as a secondary artifact.
    """

    def __init__(self, config_dir: str = "config") -> None:
        self._lock = threading.Lock()
        self._config_dir = config_dir

        # Current state
        self._status: str = "idle"          # running | stopping | idle | stopped | error
        self._error_message: str = ""
        self._next_run_time: Optional[datetime] = None
        self._last_run_time: Optional[datetime] = None
        self._cron_expression: Optional[str] = None
        self._has_cron: bool = False
        self._has_schedule: bool = False
        self._schedule_type: str = "cron"   # 'hours' | 'cron'
        self._schedule_hours: int = 24
        self._started_at: Optional[datetime] = None
        self._last_run_summary: Optional[dict] = None

        # One-shot options for the next run (e.g. reprocess_zxx from the
        # Web UI's Maintenance Actions). Cleared on consumption.
        self._run_options: dict = {}

        # Cross-thread signaling
        self._wake_event = threading.Event()
        self._run_requested = threading.Event()
        self._stop_requested = threading.Event()
        self._schedule_changed = threading.Event()

    # ── Getters ───────────────────────────────────────────────────────────

    def get_status_dict(self) -> dict:
        """Return a snapshot of the current state for the API."""
        with self._lock:
            now = datetime.now()
            next_run_iso = None
            next_run_seconds = None
            if self._next_run_time:
                next_run_iso = self._next_run_time.strftime("%Y-%m-%dT%H:%M:%S")
                delta = (self._next_run_time - now).total_seconds()
                next_run_seconds = max(0, int(delta))

            last_run_iso = None
            if self._last_run_time:
                last_run_iso = self._last_run_time.strftime("%Y-%m-%dT%H:%M:%S")

            started_at_iso = None
            if self._started_at:
                started_at_iso = self._started_at.strftime("%Y-%m-%dT%H:%M:%S")

            return {
                "status": self._status,
                "next_run_time": next_run_iso,
                "next_run_seconds": next_run_seconds,
                "last_run_time": last_run_iso,
                "started_at": started_at_iso,
                "cron_expression": self._cron_expression,
                "has_cron": self._has_cron,
                "has_schedule": self._has_schedule,
                "schedule_type": self._schedule_type,
                "schedule_hours": self._schedule_hours,
                "error_message": self._error_message,
                "last_run_summary": self._last_run_summary,
                # Server-local "now" so the UI can compare "today" against
                # the container's TZ instead of the browser's UTC clock.
                "server_now": now.strftime("%Y-%m-%dT%H:%M:%S"),
            }

    @property
    def status(self) -> str:
        with self._lock:
            return self._status

    # ── Setters ───────────────────────────────────────────────────────────

    def set_status(self, status: str, message: str = "") -> None:
        with self._lock:
            self._status = status
            self._error_message = message
            if status == "running":
                self._started_at = datetime.now()
            elif status != "running":
                self._started_at = None
        self._save_status()

    def set_next_run(self, dt: Optional[datetime]) -> None:
        with self._lock:
            self._next_run_time = dt

    def set_last_run(self, dt: Optional[datetime]) -> None:
        with self._lock:
            self._last_run_time = dt

    def set_cron(self, expression: str) -> None:
        with self._lock:
            self._cron_expression = expression
            self._has_cron = True
            self._has_schedule = True
            self._schedule_type = "cron"

    # ── Schedule management ──────────────────────────────────────────────

    def get_schedule(self) -> tuple:
        """Return a thread-safe snapshot: (schedule_type, schedule_hours, cron_expression)."""
        with self._lock:
            return (self._schedule_type, self._schedule_hours, self._cron_expression or "")

    def update_schedule(self, schedule_type: str, hours: int, cron_expr: str) -> tuple:
        """Update the active schedule and signal the scheduler loop.

        Returns (ok: bool, error: str).
        """
        schedule_type = (schedule_type or "").strip().lower()
        if schedule_type not in ("hours", "cron"):
            return (False, f"Invalid schedule_type: {schedule_type}")

        if schedule_type == "hours":
            try:
                hours = int(hours)
            except (TypeError, ValueError):
                return (False, "schedule_hours must be an integer")
            if hours < 1:
                return (False, "schedule_hours must be >= 1")
            cron_expr = ""
        else:
            cron_expr = (cron_expr or "").strip()
            try:
                from croniter import croniter
            except ImportError:
                return (False, "croniter package is not installed")
            if not croniter.is_valid(cron_expr):
                return (False, f"Invalid cron expression: {cron_expr}")

        with self._lock:
            self._schedule_type = schedule_type
            if schedule_type == "hours":
                self._schedule_hours = hours
            else:
                self._cron_expression = cron_expr
            self._has_cron = (schedule_type == "cron")
            self._has_schedule = True

        self._schedule_changed.set()
        self._wake_event.set()
        return (True, "")

    def is_schedule_changed(self) -> bool:
        return self._schedule_changed.is_set()

    def clear_schedule_changed(self) -> None:
        self._schedule_changed.clear()

    def set_last_run_summary(self, summary: dict) -> None:
        with self._lock:
            self._last_run_summary = summary

    # ── Event helpers ─────────────────────────────────────────────────────

    def is_stopped(self) -> bool:
        return self._stop_requested.is_set()

    def is_run_requested(self) -> bool:
        return self._run_requested.is_set()

    def clear_run_request(self) -> None:
        self._run_requested.clear()

    def wake(self) -> None:
        """Wake the scheduler from its interruptible sleep."""
        self._wake_event.set()

    # ── Commands (called by web UI) ───────────────────────────────────────

    def request_run(self, options: Optional[dict] = None) -> None:
        """Signal the scheduler to execute a run immediately.

        *options* is an optional dict of one-shot Config overrides applied
        to the next run only (e.g. ``{"reprocess_zxx": True}``).
        """
        with self._lock:
            self._run_options = dict(options) if options else {}
        self._run_requested.set()
        self._wake_event.set()

    def consume_run_options(self) -> dict:
        """Atomically read and clear the one-shot options for this run."""
        with self._lock:
            opts = self._run_options
            self._run_options = {}
        return opts

    def stash_run_options(self, options: Optional[dict]) -> None:
        """Stash one-shot run options without signalling a new run.

        Used at startup to forward CLI-supplied transient flags
        (``--reprocess-language``, ``--index-languages``, …) through
        the same ``consume_run_options`` channel the Web UI uses, so
        they fire on the first ``_run_processing`` call and are gone
        on subsequent iterations. Unlike :meth:`request_run`, this
        does not set the wake / run-requested events, so the scheduler
        loop won't kick off an extra unsolicited run.
        """
        with self._lock:
            self._run_options = dict(options) if options else {}

    def request_stop(self) -> None:
        """Signal the scheduler to pause after the current run completes."""
        self._stop_requested.set()
        self._wake_event.set()

    def request_resume(self) -> None:
        """Signal the scheduler to resume CRON scheduling."""
        self._stop_requested.clear()
        self._wake_event.set()

    # ── Persistence ───────────────────────────────────────────────────────

    def _save_status(self) -> None:
        """Write current status to config/status.json (atomic)."""
        from uldas.tracking import _json_default

        status_path = os.path.join(self._config_dir, "status.json")
        tmp_path = status_path + ".tmp"
        try:
            os.makedirs(self._config_dir, exist_ok=True)
            data = self.get_status_dict()
            data["updated_at"] = time.time()
            with open(tmp_path, "w", encoding="utf-8") as fh:
                json.dump(data, fh, indent=2, default=_json_default)
            os.replace(tmp_path, status_path)
        except Exception:
            logger.debug("Failed to write status.json", exc_info=True)
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
