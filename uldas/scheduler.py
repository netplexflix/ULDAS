#file: uldas/scheduler.py

import os
import time
import logging
from datetime import datetime, timedelta
from typing import Optional, Callable

logger = logging.getLogger(__name__)


def _parse_cron_field(field: str, min_val: int, max_val: int) -> list[int]:
    """Parse a single CRON field into a sorted list of matching integers."""
    values: set[int] = set()

    for part in field.split(","):
        part = part.strip()

        # Handle */N (step)
        if part.startswith("*/"):
            step = int(part[2:])
            values.update(range(min_val, max_val + 1, step))
        elif part == "*":
            values.update(range(min_val, max_val + 1))
        elif "-" in part:
            # Range: e.g. 1-5
            if "/" in part:
                range_part, step_part = part.split("/")
                lo, hi = map(int, range_part.split("-"))
                step = int(step_part)
                values.update(range(lo, hi + 1, step))
            else:
                lo, hi = map(int, part.split("-"))
                values.update(range(lo, hi + 1))
        else:
            values.add(int(part))

    return sorted(v for v in values if min_val <= v <= max_val)


def next_cron_time(cron_expr: str, after: Optional[datetime] = None) -> datetime:
    """
    Return the next datetime matching *cron_expr* (5-field: min hour dom month dow).
    Raises ``ValueError`` on invalid expressions.
    """
    fields = cron_expr.strip().split()
    if len(fields) != 5:
        raise ValueError(f"CRON expression must have 5 fields, got {len(fields)}: '{cron_expr}'")

    minutes = _parse_cron_field(fields[0], 0, 59)
    hours = _parse_cron_field(fields[1], 0, 23)
    doms = _parse_cron_field(fields[2], 1, 31)
    months = _parse_cron_field(fields[3], 1, 12)
    dows = _parse_cron_field(fields[4], 0, 6)  # 0=Sunday

    if after is None:
        after = datetime.now()

    # Start searching from the next minute
    candidate = after.replace(second=0, microsecond=0) + timedelta(minutes=1)

    # Safety: don't search more than 2 years ahead
    max_search = after + timedelta(days=366 * 2)

    while candidate < max_search:
        if (candidate.month in months
                and candidate.day in doms
                and candidate.weekday() in _convert_dow(dows)
                and candidate.hour in hours
                and candidate.minute in minutes):
            return candidate
        candidate += timedelta(minutes=1)

    raise ValueError(f"No matching time found for CRON expression: '{cron_expr}'")


def _convert_dow(cron_dows: list[int]) -> set[int]:
    """
    Convert CRON day-of-week (0=Sunday) to Python weekday (0=Monday).
    """
    mapping = {0: 6, 1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5}  # cron → python
    return {mapping[d] for d in cron_dows}


def run_on_schedule(cron_expr: str, run_fn: Callable[[], None]) -> None:
    """
    Execute *run_fn* immediately, then re-execute at every CRON match.
    Blocks forever (designed for Docker entrypoint use).
    """
    logger.info("CRON schedule: %s", cron_expr)

    # Validate expression early
    try:
        nxt = next_cron_time(cron_expr)
        logger.info("Next scheduled run: %s", nxt.strftime("%Y-%m-%d %H:%M"))
    except ValueError as exc:
        logger.error("Invalid CRON expression: %s", exc)
        raise

    # Run immediately on start
    logger.info("Running initial execution...")
    print("=" * 60)
    print("ULDAS – Initial run on container start")
    print("=" * 60)
    run_fn()

    # Schedule loop
    while True:
        now = datetime.now()
        nxt = next_cron_time(cron_expr, after=now)
        wait_seconds = (nxt - now).total_seconds()

        print(f"\n{'=' * 60}")
        print(f"Next scheduled run: {nxt.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Waiting {format_wait(wait_seconds)}...")
        print(f"{'=' * 60}\n")

        logger.info("Sleeping until %s (%.0f seconds)",
                     nxt.strftime("%Y-%m-%d %H:%M"), wait_seconds)

        time.sleep(max(0, wait_seconds))

        print("=" * 60)
        print(f"ULDAS – Scheduled run at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 60)
        run_fn()


def format_wait(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    if h > 0:
        return f"{h}h {m}m"
    return f"{m}m"


def get_cron_schedule() -> Optional[str]:
    """Read CRON_SCHEDULE from environment. Returns ``None`` if not set."""
    val = os.environ.get("CRON_SCHEDULE", "").strip()
    return val if val else None