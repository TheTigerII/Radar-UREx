"""Shared default filenames for logs produced by one capture run."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Optional


DEFAULT_LOG_DIRECTORY = Path(__file__).resolve().parent.parent / "log"


def new_run_log_id(
    now: Optional[datetime] = None,
    *,
    log_dir: Optional[Path] = None,
) -> str:
    """Return a second-resolution identifier unused by existing run logs."""

    base_id = (now or datetime.now().astimezone()).strftime("%Y%m%d_%H%M%S")
    directory = log_dir or DEFAULT_LOG_DIRECTORY
    run_id = base_id
    collision_number = 2
    while any(
        (
            directory / f"{prefix}_{run_id}{suffix}"
        ).exists()
        for prefix, suffix in (
            ("livedatacapture", ".log"),
            ("performance", ".jsonl"),
            ("live_inference", ".jsonl"),
        )
    ):
        run_id = f"{base_id}_{collision_number}"
        collision_number += 1
    return run_id


def default_run_log_path(
    prefix: str,
    suffix: str,
    *,
    run_id: str,
    log_dir: Optional[Path] = None,
) -> Path:
    """Build a default log path using the capture run's shared identifier."""

    directory = log_dir or DEFAULT_LOG_DIRECTORY
    return directory / f"{prefix}_{run_id}{suffix}"


def default_terminal_log_path(
    *,
    run_id: str,
    log_dir: Optional[Path] = None,
) -> Path:
    return default_run_log_path(
        "livedatacapture",
        ".log",
        run_id=run_id,
        log_dir=log_dir,
    )
