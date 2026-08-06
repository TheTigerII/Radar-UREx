"""Integrated Mini4 PMM capture and hardware launcher."""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import json
import math
import os
import queue
import signal
import subprocess
import sys
import tempfile
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import NamedTuple, Optional

from rawdatacapture.pmm import MINI4_DEFAULT_DETECTION_THRESHOLD
from rawdatacapture import calibrate as radar_calibration


ROOT = Path(__file__).resolve().parent
RAW_DATA_DIR = ROOT / "rawdatacapture"
DEFAULT_CONFIG_PATH = RAW_DATA_DIR / "profile-mini4-20m.cfg"
DEFAULT_CALIBRATION_PROFILE_PATH = ROOT / "profiles" / "profile_calibration.cfg"
DEFAULT_SETUP_PATH = RAW_DATA_DIR / "setup.json"
DEFAULT_CAPTURE_DIR = RAW_DATA_DIR / "captures"
DEFAULT_DATASET_DIR = ROOT / "dataset"
DEFAULT_MODEL_WEIGHTS_DIR = ROOT / "model_weights"
DEFAULT_HOST_IP = "192.168.33.30"
DEFAULT_DATA_PORT = 4098
DEFAULT_SOCKET_RECV_BUFFER_BYTES = 4 * 1024 * 1024
DEFAULT_PACKET_QUEUE_SIZE = 8192
DEFAULT_PROCESSING_QUEUE_SIZE = 32
DEFAULT_RADAR_BAUD = 115200
DEFAULT_DURATION_MINUTES = 5.0
DEFAULT_MAX_RANGE_M = 20.0
DISPLAY_CHOICES = (
    "none",
    "range",
    "range-doppler",
    "point-cloud",
    "combined",
    radar_calibration.CALIBRATION_DISPLAY_MODE,
    radar_calibration.AZIMUTH_CALIBRATION_DISPLAY_MODE,
    radar_calibration.ELEVATION_CALIBRATION_DISPLAY_MODE,
)
DISPLAY_LABELS = {
    radar_calibration.CALIBRATION_DISPLAY_MODE: "range calibration",
    radar_calibration.AZIMUTH_CALIBRATION_DISPLAY_MODE: "azimuth calibration",
    radar_calibration.ELEVATION_CALIBRATION_DISPLAY_MODE: "elevation calibration",
}
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
    parser.add_argument(
        "--calibration-profile",
        type=Path,
        default=DEFAULT_CALIBRATION_PROFILE_PATH,
    )
    parser.add_argument("--calibration-distance-m", type=float)
    parser.add_argument(
        "--calibration-search-window-m",
        type=float,
        default=radar_calibration.DEFAULT_SEARCH_WINDOW_M,
    )
    parser.add_argument(
        "--calibration-warmup-frames",
        type=int,
        default=radar_calibration.DEFAULT_WARMUP_FRAMES,
    )
    parser.add_argument(
        "--calibration-frames",
        type=int,
        default=radar_calibration.DEFAULT_ACCEPTED_FRAMES,
    )
    parser.add_argument(
        "--calibration-timeout-seconds",
        type=float,
        default=radar_calibration.DEFAULT_TIMEOUT_SECONDS,
    )
    parser.add_argument("--calibration-angle-deg", type=float)
    parser.add_argument("--calibration-output", type=Path)
    parser.add_argument("--display-update-every", type=int, default=1)
    parser.add_argument("--max-range-m", type=float, default=DEFAULT_MAX_RANGE_M)
    parser.add_argument("--duration-minutes", type=float)
    parser.add_argument("--capture-dir", type=Path, default=DEFAULT_CAPTURE_DIR)
    parser.add_argument("--processed-output", type=Path)
    parser.add_argument("--raw-output", type=Path)
    parser.add_argument(
        "--dataset-destination",
        choices=("dataset", "uav", "other"),
        help="default processed-output directory; combined display prompts if omitted",
    )
    parser.add_argument(
        "--classification",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="enable or disable real-time UAV classification",
    )
    parser.add_argument(
        "--model-weights-dir",
        type=Path,
        default=DEFAULT_MODEL_WEIGHTS_DIR,
    )
    parser.add_argument(
        "--pmm-background-calibration-seconds",
        type=float,
        default=30.0,
    )
    parser.add_argument("--pmm-max-target-speed-m-s", type=float, default=4.0)
    parser.add_argument("--pmm-folding-size-min", type=int, default=2)
    parser.add_argument("--pmm-folding-size-max", type=int, default=20)
    parser.add_argument(
        "--pmm-detection-threshold",
        type=float,
        default=MINI4_DEFAULT_DETECTION_THRESHOLD,
    )
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
        print(f"  {index}. {DISPLAY_LABELS.get(name, name)}")
    while True:
        value = input("Select display type [5]: ").strip()
        if not value:
            return "combined"
        if value in DISPLAY_CHOICES:
            return value
        normalized_value = value.casefold().replace("-", " ").replace("_", " ")
        for name, label in DISPLAY_LABELS.items():
            if normalized_value == label:
                return name
        if value.isdigit() and 1 <= int(value) <= len(DISPLAY_CHOICES):
            return DISPLAY_CHOICES[int(value) - 1]
        print("Choose a listed number or display name.")


def choose_realtime_classification(value: Optional[bool]) -> bool:
    if value is not None:
        return bool(value)
    while True:
        try:
            response = input(
                "Enable real-time UAV classification? [y/N]: "
            ).strip().lower()
        except (EOFError, KeyboardInterrupt):
            return False
        if not response or response in {"n", "no", "off", "0"}:
            return False
        if response in {"y", "yes", "on", "1"}:
            return True
        print("Choose yes or no.")


def choose_dataset_destination(value: Optional[str]) -> Path:
    choices = {
        "1": DEFAULT_DATASET_DIR / "uav",
        "uav": DEFAULT_DATASET_DIR / "uav",
        "dataset/uav": DEFAULT_DATASET_DIR / "uav",
        "2": DEFAULT_DATASET_DIR / "other",
        "other": DEFAULT_DATASET_DIR / "other",
        "dataset/other": DEFAULT_DATASET_DIR / "other",
        "3": DEFAULT_DATASET_DIR,
        "dataset": DEFAULT_DATASET_DIR,
    }
    if value is not None:
        return choices[value]
    print("Save processed data to:")
    print("  1. dataset/uav")
    print("  2. dataset/other")
    print("  3. dataset")
    while True:
        try:
            response = input("Select dataset destination [3]: ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            return DEFAULT_DATASET_DIR
        if not response:
            return DEFAULT_DATASET_DIR
        if response in choices:
            return choices[response]
        print("Choose 1, 2, or 3.")


def choose_duration_minutes(value: Optional[float]) -> float:
    if value is None:
        text = input(
            f"Run duration in minutes [{DEFAULT_DURATION_MINUTES:g}]: "
        ).strip()
        value = DEFAULT_DURATION_MINUTES if not text else float(text)
    if not float("-inf") < float(value) < float("inf") or value < 0.0:
        raise ValueError("duration must be finite and non-negative")
    return float(value)


def choose_calibration_distance_m(value: Optional[float]) -> float:
    if value is None:
        text = input(
            "Laser-measured reflector distance in meters "
            f"[{radar_calibration.DEFAULT_TARGET_DISTANCE_M:g}]: "
        ).strip()
        value = radar_calibration.DEFAULT_TARGET_DISTANCE_M if not text else float(text)
    if not math.isfinite(float(value)) or float(value) <= 0.0:
        raise ValueError("calibration distance must be finite and positive")
    return float(value)


def choose_calibration_angle_deg(
    value: Optional[float],
    calibration_type: str,
) -> float:
    if value is None:
        text = input(
            f"Known tripod {calibration_type} angle in degrees "
            f"[{radar_calibration.DEFAULT_REFERENCE_ANGLE_DEG:g}]: "
        ).strip()
        value = radar_calibration.DEFAULT_REFERENCE_ANGLE_DEG if not text else float(text)
    if not math.isfinite(float(value)) or abs(float(value)) > 60.0:
        raise ValueError("calibration angle must be within -60 to +60 degrees")
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
    processed_output: Optional[Path],
    raw_output: Optional[Path] = None,
    config_path: Optional[Path] = None,
    host_compensation_profile: Optional[Path] = None,
    model_weights_dir: Optional[Path] = None,
) -> list[str]:
    command = [
        sys.executable,
        "-u",
        str(RAW_DATA_DIR / "livedatacapture.py"),
        "--config",
        str(config_path or args.config),
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
    if processed_output is not None and display not in radar_calibration.CALIBRATION_DISPLAY_MODES:
        command.extend(("--processed-output", str(processed_output)))
    if display in radar_calibration.CALIBRATION_DISPLAY_MODES:
        distance_m = getattr(args, "calibration_distance_m", None)
        angle_deg = getattr(args, "calibration_angle_deg", None)
        command.extend(
            (
                "--calibration-distance-m",
                str(
                    radar_calibration.DEFAULT_TARGET_DISTANCE_M
                    if distance_m is None
                    else distance_m
                ),
                "--calibration-search-window-m",
                str(args.calibration_search_window_m),
                "--calibration-warmup-frames",
                str(max(args.calibration_warmup_frames, 0)),
                "--calibration-frames",
                str(max(args.calibration_frames, 1)),
                "--calibration-timeout-seconds",
                str(args.calibration_timeout_seconds),
                "--calibration-angle-deg",
                str(
                    radar_calibration.DEFAULT_REFERENCE_ANGLE_DEG
                    if angle_deg is None
                    else angle_deg
                ),
            )
        )
    if display in {
        radar_calibration.AZIMUTH_CALIBRATION_DISPLAY_MODE,
        radar_calibration.ELEVATION_CALIBRATION_DISPLAY_MODE,
    }:
        if host_compensation_profile is None:
            raise ValueError("Angular calibration requires host compensation profile")
        command.extend(
            ("--host-compensation-profile", str(host_compensation_profile))
        )
    if raw_output is not None:
        command.extend(("--raw-output", str(raw_output)))
    if model_weights_dir is not None:
        command.extend(("--model-weights-dir", str(model_weights_dir)))
    return command


def build_startup_command(
    args: argparse.Namespace,
    radar_port: str,
    config_path: Optional[Path] = None,
) -> list[str]:
    effective_config = config_path or args.config
    return [
        sys.executable,
        str(RAW_DATA_DIR / "startup.py"),
        "--config",
        str(effective_config),
        "--sdk-profile",
        str(effective_config),
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
    calibration_queue: Optional[queue.SimpleQueue] = None,
) -> None:
    if process.stdout is None:
        return
    for raw_line in process.stdout:
        line = raw_line.rstrip()
        if line.startswith(CAPTURE_READY_PREFIX):
            capture_ready.set()
        marker_index = line.find(radar_calibration.CALIBRATION_RESULT_PREFIX)
        if marker_index >= 0:
            try:
                payload = json.loads(
                    line[
                        marker_index
                        + len(radar_calibration.CALIBRATION_RESULT_PREFIX) :
                    ]
                )
            except ValueError:
                print(line, flush=True)
                continue
            if calibration_queue is not None and isinstance(payload, dict):
                calibration_queue.put(payload)
            continue
        print(line, flush=True)


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


def _default_calibration_output(
    capture_dir: Path,
    calibration_type: str,
) -> Path:
    timestamp = datetime.now().astimezone().strftime("%Y_%m_%dT%H_%M_%S")
    prefix = "calibration" if calibration_type == "range" else f"{calibration_type}_calibration"
    return capture_dir / f"{prefix}_{timestamp}.json"


def _calibration_result_from_payload(
    payload: object,
) -> radar_calibration.CalibrationResult | radar_calibration.AngularCalibrationResult:
    if not isinstance(payload, dict):
        raise ValueError("calibration result is not a JSON object")
    if payload.get("calibration_type", "range") == "range":
        return radar_calibration.CalibrationResult.from_dict(payload)
    return radar_calibration.AngularCalibrationResult.from_dict(payload)


def _wait_for_calibration_payload(
    result_queue: queue.SimpleQueue,
    relay_thread: threading.Thread,
    timeout_seconds: float = 5.0,
) -> Optional[dict]:
    """Allow the stdout relay to deliver a result printed during shutdown."""

    deadline = time.monotonic() + max(timeout_seconds, 0.0)
    while True:
        try:
            payload = result_queue.get_nowait()
        except queue.Empty:
            payload = None
        if isinstance(payload, dict):
            return payload
        remaining = deadline - time.monotonic()
        if remaining <= 0.0:
            return None
        relay_thread.join(timeout=min(0.1, remaining))
        if not relay_thread.is_alive():
            try:
                payload = result_queue.get_nowait()
            except queue.Empty:
                return None
            return payload if isinstance(payload, dict) else None


def run_calibration_mode(
    args: argparse.Namespace,
    display: str,
    radar_port: str,
) -> int:
    calibration_type = {
        radar_calibration.CALIBRATION_DISPLAY_MODE: "range",
        radar_calibration.AZIMUTH_CALIBRATION_DISPLAY_MODE: "azimuth",
        radar_calibration.ELEVATION_CALIBRATION_DISPLAY_MODE: "elevation",
    }[display]
    try:
        distance_m = choose_calibration_distance_m(args.calibration_distance_m)
        reference_angle_deg = (
            choose_calibration_angle_deg(
                args.calibration_angle_deg, calibration_type
            )
            if calibration_type in {"azimuth", "elevation"}
            else radar_calibration.DEFAULT_REFERENCE_ANGLE_DEG
        )
        settings = radar_calibration.CalibrationSettings(
            target_distance_m=distance_m,
            search_window_m=args.calibration_search_window_m,
            warmup_frames=args.calibration_warmup_frames,
            accepted_frames=args.calibration_frames,
            timeout_seconds=args.calibration_timeout_seconds,
            calibration_type=calibration_type,
            reference_angle_deg=reference_angle_deg,
        )
    except ValueError as exc:
        print(f"Invalid calibration settings: {exc}", file=sys.stderr)
        return 2
    args.calibration_distance_m = settings.target_distance_m
    args.calibration_angle_deg = settings.reference_angle_deg

    source_profile = args.calibration_profile
    if not source_profile.is_absolute():
        source_profile = ROOT / source_profile
    operational_profile = args.config
    if not operational_profile.is_absolute():
        operational_profile = ROOT / operational_profile
    report_path = args.calibration_output or _default_calibration_output(
        args.capture_dir, calibration_type
    )
    if not report_path.is_absolute():
        report_path = ROOT / report_path

    try:
        from rawdatacapture.livedatacapture import RadarCaptureConfig

        source_config = RadarCaptureConfig.from_file(source_profile)
    except (OSError, ValueError) as exc:
        print(f"Calibration profile is invalid: {exc}", file=sys.stderr)
        return 2

    capture: Optional[subprocess.Popen] = None
    startup: Optional[subprocess.Popen] = None
    relay_thread: Optional[threading.Thread] = None
    result: Optional[
        radar_calibration.CalibrationResult
        | radar_calibration.AngularCalibrationResult
    ] = None
    ready = threading.Event()
    result_queue: queue.SimpleQueue = queue.SimpleQueue()
    temporary_profiles: Optional[tempfile.TemporaryDirectory] = None
    try:
        temporary_profiles = tempfile.TemporaryDirectory(
            prefix="radar_calibration_"
        )
        with nullcontext(temporary_profiles.name) as temporary_directory:
            runtime_profile = Path(temporary_directory) / "calibration_runtime.cfg"
            radar_calibration.create_runtime_profile(
                source_profile,
                runtime_profile,
                source_config,
                settings,
            )
            print(f"Display mode: {display}")
            print(f"Laser distance: {settings.target_distance_m:g} m")
            if calibration_type != "range":
                print(
                    f"Known {calibration_type} angle: "
                    f"{settings.reference_angle_deg:+g} degrees"
                )
            capture = start_process(
                "calibration capture",
                build_capture_command(
                    args,
                    display,
                    None,
                    None,
                    runtime_profile,
                    (
                        operational_profile
                        if calibration_type in {"azimuth", "elevation"}
                        else None
                    ),
                ),
                capture_output=True,
            )
            relay_thread = threading.Thread(
                target=relay_capture_output,
                args=(capture, ready, result_queue),
                daemon=True,
            )
            relay_thread.start()
            if not wait_for_capture_ready(capture, ready):
                print("Calibration capture failed to become ready.", file=sys.stderr)
                return 1
            startup = start_process(
                "calibration startup",
                build_startup_command(args, radar_port, runtime_profile),
            )
            deadline = time.monotonic() + settings.timeout_seconds
            while time.monotonic() < deadline:
                try:
                    payload = result_queue.get_nowait()
                except queue.Empty:
                    payload = None
                if payload is not None:
                    result = _calibration_result_from_payload(payload)
                    break
                if startup.poll() is not None:
                    print(
                        "Calibration startup exited early with code "
                        f"{startup.returncode}.",
                        file=sys.stderr,
                    )
                    return startup.returncode or 1
                if capture.poll() is not None:
                    # The result is printed immediately before the capture exits.
                    # Give the relay enough time to consume the remaining pipe
                    # content before treating an automatic exit as a failure.
                    payload = _wait_for_calibration_payload(
                        result_queue,
                        relay_thread,
                    )
                    if payload is not None:
                        result = _calibration_result_from_payload(payload)
                        break
                    print(
                        "Calibration capture exited early with code "
                        f"{capture.returncode}.",
                        file=sys.stderr,
                    )
                    return capture.returncode or 1
                time.sleep(0.1)
            if result is None:
                # A result emitted at the timeout boundary can still be waiting
                # in the capture process's stdout pipe.
                payload = _wait_for_calibration_payload(
                    result_queue,
                    relay_thread,
                    timeout_seconds=1.0,
                )
                if payload is not None:
                    result = _calibration_result_from_payload(payload)
            if result is None:
                print(
                    "Calibration timed out without a stable result; profile unchanged.",
                    file=sys.stderr,
                )
                return 1
    except KeyboardInterrupt:
        print("Calibration interrupted; profile unchanged.")
        return 130
    except (KeyError, OSError, TypeError, ValueError) as exc:
        print(f"Calibration failed: {exc}", file=sys.stderr)
        return 1
    finally:
        stop_process(startup)
        stop_process(capture)
        if relay_thread is not None:
            relay_thread.join(timeout=2.0)
        if temporary_profiles is not None:
            temporary_profiles.cleanup()

    assert result is not None
    print("Calibration completed successfully.")
    try:
        radar_calibration.write_calibration_report(report_path, result)
    except OSError as exc:
        print(f"Could not write calibration report: {exc}", file=sys.stderr)
        return 1
    print(f"Calibration report: {report_path}")
    if isinstance(result, radar_calibration.CalibrationResult):
        print(result.command)
    else:
        print(
            f"{result.calibration_type}: expected "
            f"{result.reference_angle_deg:+.3f}, measured "
            f"{result.measured_angle_deg:+.3f}, bias "
            f"{result.angle_bias_deg:+.3f} degrees"
        )
    try:
        apply_result = input(
            f"Apply this calibration to {operational_profile}? [y/N]: "
        ).strip().lower()
    except (EOFError, KeyboardInterrupt):
        apply_result = ""
    if apply_result not in {"y", "yes"}:
        print("Calibration not applied; operational profile unchanged.")
        return 0
    try:
        backup = (
            radar_calibration.apply_calibration_to_profile(
                operational_profile, result
            )
            if isinstance(result, radar_calibration.CalibrationResult)
            else radar_calibration.apply_angular_calibration_to_profile(
                operational_profile, result
            )
        )
    except (OSError, ValueError) as exc:
        print(f"Could not apply calibration: {exc}", file=sys.stderr)
        return 1
    print(f"Calibration applied. Backup: {backup}")
    return 0


def main() -> int:
    args = parse_args()
    display = choose_display(args.display)
    duration_minutes = 0.0
    realtime_classification = False
    default_output_directory = args.capture_dir
    if display not in radar_calibration.CALIBRATION_DISPLAY_MODES:
        realtime_classification = choose_realtime_classification(
            args.classification
        )
        if args.processed_output is None and (
            display == "combined" or args.dataset_destination is not None
        ):
            default_output_directory = choose_dataset_destination(
                args.dataset_destination
            )
        try:
            duration_minutes = choose_duration_minutes(args.duration_minutes)
        except ValueError as exc:
            print(f"Invalid duration: {exc}", file=sys.stderr)
            return 2
    radar_port = resolve_radar_port(args.radar_port)
    if not radar_port:
        print("No radar command UART was supplied.", file=sys.stderr)
        return 2
    if display in radar_calibration.CALIBRATION_DISPLAY_MODES:
        return run_calibration_mode(args, display, radar_port)

    model_weights_dir: Optional[Path] = None
    if realtime_classification:
        model_weights_dir = args.model_weights_dir
        if not model_weights_dir.is_absolute():
            model_weights_dir = ROOT / model_weights_dir
        expected_model = model_weights_dir / "model_state.pt"
        if not expected_model.is_file():
            print(
                "Real-time classification model is missing: "
                f"expected {expected_model}",
                file=sys.stderr,
            )
            return 2

    processed_output = args.processed_output or default_processed_output(
        default_output_directory
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
        "Real-time classification: "
        + (
            f"enabled ({model_weights_dir})"
            if model_weights_dir is not None
            else "disabled"
        )
    )
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
            build_capture_command(
                args,
                display,
                processed_output,
                raw_output,
                model_weights_dir=model_weights_dir,
            ),
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
