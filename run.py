"""Integrated Mini4 PMM capture and hardware launcher."""

from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import NamedTuple, Optional


ROOT = Path(__file__).resolve().parent
RAW_DATA_DIR = ROOT / "rawdatacapture"
DEFAULT_CONFIG_PATH = RAW_DATA_DIR / "profile-mini4-20m.cfg"
DEFAULT_SETUP_PATH = RAW_DATA_DIR / "setup.json"
DEFAULT_CAPTURE_DIR = RAW_DATA_DIR / "captures"
DEFAULT_HOST_IP = "192.168.33.30"
DEFAULT_DATA_PORT = 4098
DEFAULT_SOCKET_RECV_BUFFER_BYTES = 4 * 1024 * 1024
DEFAULT_PACKET_QUEUE_SIZE = 8192
DEFAULT_PROCESSING_QUEUE_SIZE = 32
DEFAULT_RADAR_BAUD = 115200
DEFAULT_DURATION_MINUTES = 3.0
DEFAULT_MAX_RANGE_M = 20.0
DISPLAY_CHOICES = (
    "none",
    "range",
    "range-doppler",
    "point-cloud",
    "combined",
)
CAPTURE_READY_PREFIX = "Listening for live radar stream "
CAPTURE_STARTUP_TIMEOUT_SECONDS = 30.0
DEFAULT_RADAR_UART_DESCRIPTION = (
    "CP2105 Dual USB to UART Bridge Controller - Enhanced COM Port"
)
WINDOWS_DEFAULT_RADAR_PORT = "COM4"
LINUX_DEFAULT_RADAR_PORT = "/dev/ttyUSB0"


class SerialPortInfo(NamedTuple):
    device: str
    description: str = ""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Start Mini4 PMM capture and configure IWR6843/DCA1000."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--setup", type=Path, default=DEFAULT_SETUP_PATH)
    parser.add_argument("--host-ip", default=DEFAULT_HOST_IP)
    parser.add_argument("--data-port", type=int, default=DEFAULT_DATA_PORT)
    parser.add_argument(
        "--socket-recv-buffer",
        type=int,
        default=DEFAULT_SOCKET_RECV_BUFFER_BYTES,
    )
    parser.add_argument(
        "--packet-queue-size",
        type=int,
        default=DEFAULT_PACKET_QUEUE_SIZE,
    )
    parser.add_argument(
        "--processing-queue-size",
        type=int,
        default=DEFAULT_PROCESSING_QUEUE_SIZE,
    )
    parser.add_argument("--radar-port")
    parser.add_argument("--radar-baud", type=int, default=DEFAULT_RADAR_BAUD)
    parser.add_argument("--radar-command-timeout", type=float, default=10.0)
    parser.add_argument("--dca-timeout", type=float, default=3.0)
    parser.add_argument("--dca-retries", type=int, default=5)
    parser.add_argument("--display", choices=DISPLAY_CHOICES)
    parser.add_argument("--display-update-every", type=int, default=1)
    parser.add_argument("--max-range-m", type=float, default=DEFAULT_MAX_RANGE_M)
    parser.add_argument("--duration-minutes", type=float)
    parser.add_argument("--capture-dir", type=Path, default=DEFAULT_CAPTURE_DIR)
    parser.add_argument("--processed-output", type=Path)
    parser.add_argument("--raw-output", type=Path)
    parser.add_argument(
        "--pmm-background-calibration-seconds",
        type=float,
        default=30.0,
    )
    parser.add_argument("--pmm-max-target-speed-m-s", type=float, default=4.0)
    parser.add_argument("--pmm-folding-size-min", type=int, default=2)
    parser.add_argument("--pmm-folding-size-max", type=int, default=20)
    parser.add_argument("--pmm-detection-threshold", type=float, default=30_000.0)
    parser.add_argument("--pmm-history-seconds", type=float, default=3.6)
    parser.add_argument("--pmm-provisional-frames", type=int, default=5)
    parser.add_argument("--pmm-confirmation-window-frames", type=int, default=10)
    parser.add_argument("--pmm-confirmation-hits", type=int, default=7)
    parser.add_argument("--pmm-coast-frames", type=int, default=10)
    return parser.parse_args()


def choose_display(display: Optional[str]) -> str:
    if display is not None:
        return display
    print("Display type:")
    for index, name in enumerate(DISPLAY_CHOICES, start=1):
        print(f"  {index}. {name}")
    while True:
        value = input("Select display type [5]: ").strip()
        if not value:
            return "combined"
        if value in DISPLAY_CHOICES:
            return value
        if value.isdigit() and 1 <= int(value) <= len(DISPLAY_CHOICES):
            return DISPLAY_CHOICES[int(value) - 1]
        print("Choose a listed number or display name.")


def choose_duration_minutes(value: Optional[float]) -> float:
    if value is None:
        text = input(
            f"Run duration in minutes [{DEFAULT_DURATION_MINUTES:g}]: "
        ).strip()
        value = DEFAULT_DURATION_MINUTES if not text else float(text)
    if not float("-inf") < float(value) < float("inf") or value < 0.0:
        raise ValueError("duration must be finite and non-negative")
    return float(value)


def list_serial_ports() -> list[SerialPortInfo]:
    try:
        from serial.tools import list_ports
    except ImportError:
        return []
    return [
        SerialPortInfo(port.device, port.description or "")
        for port in list_ports.comports()
    ]


def resolve_radar_port(explicit_port: Optional[str]) -> Optional[str]:
    if explicit_port:
        return explicit_port
    ports = list_serial_ports()
    preferred_ports = [
        port
        for port in ports
        if "cp2105" in port.description.casefold()
        and "enhanced" in port.description.casefold()
        and "com port" in port.description.casefold()
    ]
    if len(preferred_ports) == 1:
        selected = preferred_ports[0]
        print(
            "Using default radar command UART: "
            f"{selected.device} {selected.description}"
        )
        return selected.device
    if len(ports) == 1:
        return ports[0].device
    if ports:
        print("Radar command UART:")
        for index, port in enumerate(ports, start=1):
            print(f"  {index}. {port.device} {port.description}".rstrip())
        while True:
            value = input("Select radar UART: ").strip()
            if value.isdigit() and 1 <= int(value) <= len(ports):
                return ports[int(value) - 1].device
            if value:
                return value
    default = (
        WINDOWS_DEFAULT_RADAR_PORT
        if os.name == "nt"
        else LINUX_DEFAULT_RADAR_PORT
    )
    value = input(f"Radar command UART [{default}]: ").strip()
    return value or default


def default_processed_output(capture_dir: Path) -> Path:
    timestamp = datetime.now().astimezone().strftime("%Y_%m_%dT%H_%M_%S")
    return capture_dir / f"pmm_capture_{timestamp}.jsonl"


def build_capture_command(
    args: argparse.Namespace,
    display: str,
    processed_output: Path,
    raw_output: Optional[Path] = None,
) -> list[str]:
    command = [
        sys.executable,
        "-u",
        str(RAW_DATA_DIR / "livedatacapture.py"),
        "--config",
        str(args.config),
        "--setup",
        str(args.setup),
        "--host-ip",
        str(args.host_ip),
        "--data-port",
        str(args.data_port),
        "--socket-recv-buffer",
        str(max(args.socket_recv_buffer, 0)),
        "--packet-queue-size",
        str(max(args.packet_queue_size, 1)),
        "--processing-queue-size",
        str(max(args.processing_queue_size, 1)),
        "--display",
        display,
        "--display-update-every",
        str(max(args.display_update_every, 1)),
        "--max-range-m",
        str(min(max(args.max_range_m, 0.3), 20.0)),
        "--processed-output",
        str(processed_output),
        "--pmm-background-calibration-seconds",
        str(args.pmm_background_calibration_seconds),
        "--pmm-max-target-speed-m-s",
        str(args.pmm_max_target_speed_m_s),
        "--pmm-folding-size-min",
        str(args.pmm_folding_size_min),
        "--pmm-folding-size-max",
        str(args.pmm_folding_size_max),
        "--pmm-detection-threshold",
        str(args.pmm_detection_threshold),
        "--pmm-history-seconds",
        str(args.pmm_history_seconds),
        "--pmm-provisional-frames",
        str(args.pmm_provisional_frames),
        "--pmm-confirmation-window-frames",
        str(args.pmm_confirmation_window_frames),
        "--pmm-confirmation-hits",
        str(args.pmm_confirmation_hits),
        "--pmm-coast-frames",
        str(args.pmm_coast_frames),
    ]
    if raw_output is not None:
        command.extend(("--raw-output", str(raw_output)))
    return command


def build_startup_command(
    args: argparse.Namespace,
    radar_port: str,
) -> list[str]:
    return [
        sys.executable,
        str(RAW_DATA_DIR / "startup.py"),
        "--config",
        str(args.config),
        "--sdk-profile",
        str(args.config),
        "--setup",
        str(args.setup),
        "--host-ip",
        str(args.host_ip),
        "--data-port",
        str(args.data_port),
        "--radar-backend",
        "direct-serial",
        "--dca-backend",
        "direct-udp",
        "--skip-socket-preflight",
        "--radar-port",
        radar_port,
        "--radar-baud",
        str(args.radar_baud),
        "--radar-command-timeout",
        str(args.radar_command_timeout),
        "--dca-timeout",
        str(args.dca_timeout),
        "--dca-retries",
        str(args.dca_retries),
    ]


def _subprocess_kwargs() -> dict:
    if os.name == "nt":
        return {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP}
    return {"start_new_session": True}


def start_process(
    label: str,
    command: list[str],
    *,
    capture_output: bool = False,
) -> subprocess.Popen:
    print(f"Starting {label}: {' '.join(command)}")
    return subprocess.Popen(
        command,
        cwd=ROOT,
        stdout=subprocess.PIPE if capture_output else None,
        stderr=subprocess.STDOUT if capture_output else None,
        text=capture_output,
        bufsize=1,
        **_subprocess_kwargs(),
    )


def relay_capture_output(
    process: subprocess.Popen,
    capture_ready: threading.Event,
) -> None:
    if process.stdout is None:
        return
    for raw_line in process.stdout:
        line = raw_line.rstrip()
        print(line, flush=True)
        if line.startswith(CAPTURE_READY_PREFIX):
            capture_ready.set()


def wait_for_capture_ready(
    process: subprocess.Popen,
    ready_event: threading.Event,
    timeout_seconds: float = CAPTURE_STARTUP_TIMEOUT_SECONDS,
) -> bool:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        if ready_event.wait(timeout=0.1):
            return True
        if process.poll() is not None:
            return False
    return ready_event.is_set()


def terminate_process(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return
    if os.name == "nt":
        process.terminate()
    else:
        os.killpg(process.pid, signal.SIGINT)


def kill_process(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return
    if os.name == "nt":
        process.kill()
    else:
        os.killpg(process.pid, signal.SIGKILL)


def stop_process(process: Optional[subprocess.Popen], timeout: float = 10.0) -> None:
    if process is None or process.poll() is not None:
        return
    terminate_process(process)
    try:
        process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        kill_process(process)
        process.wait(timeout=2.0)


def main() -> int:
    args = parse_args()
    display = choose_display(args.display)
    try:
        duration_minutes = choose_duration_minutes(args.duration_minutes)
    except ValueError as exc:
        print(f"Invalid duration: {exc}", file=sys.stderr)
        return 2
    radar_port = resolve_radar_port(args.radar_port)
    if not radar_port:
        print("No radar command UART was supplied.", file=sys.stderr)
        return 2

    processed_output = args.processed_output or default_processed_output(
        args.capture_dir
    )
    if not processed_output.is_absolute():
        processed_output = ROOT / processed_output
    processed_output.parent.mkdir(parents=True, exist_ok=True)
    raw_output = args.raw_output
    if raw_output is not None and not raw_output.is_absolute():
        raw_output = ROOT / raw_output
    if raw_output is not None:
        raw_output.parent.mkdir(parents=True, exist_ok=True)

    print(f"Display mode: {display}")
    print(f"Processed output: {processed_output}")
    print(f"Raw output: {raw_output or 'disabled'}")
    print(
        "PMM tracking: "
        f"calibration={args.pmm_background_calibration_seconds:g}s, "
        f"threshold={args.pmm_detection_threshold:g}"
    )

    capture: Optional[subprocess.Popen] = None
    startup: Optional[subprocess.Popen] = None
    relay_thread: Optional[threading.Thread] = None
    try:
        capture = start_process(
            "live capture",
            build_capture_command(args, display, processed_output, raw_output),
            capture_output=True,
        )
        ready = threading.Event()
        relay_thread = threading.Thread(
            target=relay_capture_output,
            args=(capture, ready),
            daemon=True,
        )
        relay_thread.start()
        if not wait_for_capture_ready(capture, ready):
            print("Live capture failed to become ready.", file=sys.stderr)
            return 1
        startup = start_process(
            "radar/DCA1000 startup",
            build_startup_command(args, radar_port),
        )
        start_time = time.monotonic()
        while True:
            if capture.poll() is not None:
                return capture.returncode or 1
            if startup.poll() is not None:
                return startup.returncode or 1
            if (
                duration_minutes > 0.0
                and time.monotonic() - start_time >= duration_minutes * 60.0
            ):
                break
            time.sleep(0.2)
    except KeyboardInterrupt:
        print("Stopping...")
    finally:
        stop_process(startup)
        stop_process(capture)
        if relay_thread is not None:
            relay_thread.join(timeout=2.0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
