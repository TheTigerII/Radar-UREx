"""Stop Radar-UREx hardware and capture processes without killing the GUI."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import signal
import socket
import sys
from typing import Iterable

import psutil


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RADAR_SCRIPT_NAMES = {"run.py", "startup.py", "livedatacapture.py"}
RADAR_APPLICATION_DIRECTORIES = {
    "main",
    "experimental_radar",
    "aesa_radar",
    "conventional_radar",
}


def is_radar_process_command(
    command: Iterable[str],
    *,
    project_root: Path = PROJECT_ROOT,
) -> bool:
    """Return whether a command runs one of this repository's radar workers."""

    resolved_root = project_root.resolve()
    for argument in command:
        text = str(argument)
        if not text.endswith(tuple(RADAR_SCRIPT_NAMES)):
            continue
        try:
            path = Path(text).expanduser().resolve()
            path.relative_to(resolved_root)
        except (OSError, ValueError):
            continue
        relative = path.relative_to(resolved_root)
        if path.name not in RADAR_SCRIPT_NAMES:
            continue
        if "gui.py" in relative.parts:
            continue
        if relative.parts and relative.parts[0] in RADAR_APPLICATION_DIRECTORIES:
            return True
    return False


def discover_radar_processes(*, excluded_pids: set[int] | None = None) -> list[psutil.Process]:
    """Find launchers/workers plus their child display and DSP processes."""

    excluded = set(excluded_pids or ()) | {os.getpid()}
    direct: dict[int, psutil.Process] = {}
    for process in psutil.process_iter(("pid", "cmdline")):
        try:
            if process.pid in excluded:
                continue
            if is_radar_process_command(process.info.get("cmdline") or ()):
                direct[process.pid] = process
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue

    found = dict(direct)
    for process in tuple(direct.values()):
        try:
            for child in process.children(recursive=True):
                if child.pid not in excluded:
                    found[child.pid] = child
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return sorted(found.values(), key=lambda process: process.pid)


def _top_level_processes(processes: list[psutil.Process]) -> list[psutil.Process]:
    process_ids = {process.pid for process in processes}
    roots: list[psutil.Process] = []
    for process in processes:
        try:
            if process.ppid() not in process_ids:
                roots.append(process)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return roots


def _describe(process: psutil.Process) -> str:
    try:
        command = process.cmdline()
        script = next(
            (Path(part).name for part in command if part.endswith(".py")),
            process.name(),
        )
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        script = "exited"
    return f"pid={process.pid} ({script})"


def stop_all_radar_processes() -> tuple[int, int]:
    """Gracefully stop all repository radar sessions, then force survivors."""

    processes = discover_radar_processes()
    if not processes:
        print("No running Radar-UREx capture or startup processes were found.")
        return 0, 0

    print(f"Stopping {len(processes)} Radar-UREx process(es)...", flush=True)
    roots = _top_level_processes(processes)
    for process in roots:
        try:
            print(f"Sending interrupt to {_describe(process)}", flush=True)
            process.send_signal(signal.SIGINT)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue

    _, alive = psutil.wait_procs(processes, timeout=8.0)
    for process in alive:
        try:
            process.terminate()
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    _, alive = psutil.wait_procs(alive, timeout=3.0)
    for process in alive:
        try:
            process.kill()
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    _, alive = psutil.wait_procs(alive, timeout=2.0)

    remaining = discover_radar_processes()
    if remaining:
        print(
            "WARNING: radar process(es) still running: "
            + ", ".join(_describe(process) for process in remaining),
            file=sys.stderr,
        )
        return len(processes), len(remaining)
    print("All Radar-UREx processes stopped; UART and UDP ports are released.")
    return len(processes), 0


def _port_is_available(host: str, port: int) -> bool:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as probe:
            probe.bind((host, port))
    except OSError:
        return False
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--list",
        action="store_true",
        help="List matching processes without stopping them.",
    )
    args = parser.parse_args()
    if args.list:
        processes = discover_radar_processes()
        for process in processes:
            print(_describe(process))
        return 1 if processes else 0

    _stopped, remaining = stop_all_radar_processes()
    data_port_free = _port_is_available("192.168.33.30", 4098)
    config_port_free = _port_is_available("0.0.0.0", 4096)
    print(
        "Receiver readiness: "
        f"UDP 4098={'free' if data_port_free else 'busy'}, "
        f"UDP 4096={'free' if config_port_free else 'busy'}"
    )
    return 1 if remaining or not data_port_free or not config_port_free else 0


if __name__ == "__main__":
    raise SystemExit(main())
