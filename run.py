import argparse
import json
import math
import os
import queue
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
DEFAULT_CONFIG_PATH = RAW_DATA_DIR / "profile.cfg"
DEFAULT_SETUP_PATH = RAW_DATA_DIR / "setup.json"
DEFAULT_CAPTURE_DIR = RAW_DATA_DIR / "captures"
DEFAULT_CLASSIFICATION_ARTIFACT_DIR = ROOT / "Radar-UREx-output" / "artifacts"
DEFAULT_HOST_IP = "192.168.33.30"
DEFAULT_DATA_PORT = 4098
DEFAULT_SOCKET_RECV_BUFFER_BYTES = 4 * 1024 * 1024
DEFAULT_PACKET_QUEUE_SIZE = 8192
DEFAULT_PROCESSING_QUEUE_SIZE = 32
DEFAULT_RADAR_BAUD = 115200
DEFAULT_RADAR_COMMAND_TIMEOUT = 10.0
DEFAULT_DCA_TIMEOUT = 3.0
DEFAULT_DCA_RETRIES = 5
DEFAULT_DURATION_MINUTES = 5.0
DEFAULT_MICRO_DOPPLER_RANGE_M = 2.15
DEFAULT_ROTOR_BLADES = 2
DEFAULT_ROTOR_RPM_MAX = 10_700.0
DEFAULT_MAX_RANGE_M = 10.0
DEFAULT_CLUSTER_EPS_M = 0.4
DEFAULT_CLUSTER_MIN_SAMPLES = 2
DEFAULT_CLUTTER_MAP_UPDATE_RATE = 0.02
DEFAULT_CLUTTER_MAP_WARMUP_FRAMES = 30
DEFAULT_CLUTTER_MAP_MIN_SNR_DB = 3.0
DEFAULT_STATIC_DETECTION = True
DEFAULT_STATIC_WARMUP_FRAMES = 30
DEFAULT_STATIC_REFERENCE_FRAMES = 150
DEFAULT_STATIC_MIN_CHANGE_DB = 3.0
DEFAULT_STATIC_BACKGROUND_UPDATE_RATE = 0.0
DEFAULT_STATIC_CLUSTER_MIN_SAMPLES = 1
DISPLAY_CHOICES = (
    "none",
    "range",
    "range-doppler",
    "point-cloud",
    "micro-doppler",
    "point-cloud-micro-doppler",
)
WINDOWS_DEFAULT_RADAR_PORT = "COM4"
LINUX_DEFAULT_RADAR_PORT = "/dev/ttyUSB0"
CLASSIFICATION_RESULT_PREFIX = "CLASSIFICATION_RESULT "
CAPTURE_READY_PREFIX = "Listening for live radar stream "
CAPTURE_STARTUP_TIMEOUT_SECONDS = 30.0
CAPTURE_GPU_STARTUP_TIMEOUT_SECONDS = 300.0
JETSON_MODEL_PATH = Path("/proc/device-tree/model")


class SerialPortInfo(NamedTuple):
    device: str
    description: str = ""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Start live DCA1000 capture, then start radar/DCA1000 hardware. "
            "Stop automatically after the selected duration, or press Ctrl+C."
        )
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--setup", type=Path, default=DEFAULT_SETUP_PATH)
    parser.add_argument("--host-ip", default=DEFAULT_HOST_IP)
    parser.add_argument("--data-port", type=int, default=DEFAULT_DATA_PORT)
    parser.add_argument(
        "--socket-recv-buffer",
        type=int,
        default=DEFAULT_SOCKET_RECV_BUFFER_BYTES,
        help="Requested UDP socket receive buffer in bytes; use 0 for OS default.",
    )
    parser.add_argument(
        "--packet-queue-size",
        type=int,
        default=DEFAULT_PACKET_QUEUE_SIZE,
        help="Maximum UDP datagrams buffered before frame assembly.",
    )
    parser.add_argument(
        "--processing-queue-size",
        type=int,
        default=DEFAULT_PROCESSING_QUEUE_SIZE,
        help="Maximum complete frames waiting for DSP processing.",
    )
    parser.add_argument("--radar-port")
    parser.add_argument("--radar-baud", type=int, default=DEFAULT_RADAR_BAUD)
    parser.add_argument(
        "--radar-command-timeout",
        type=float,
        default=DEFAULT_RADAR_COMMAND_TIMEOUT,
    )
    parser.add_argument("--dca-timeout", type=float, default=DEFAULT_DCA_TIMEOUT)
    parser.add_argument("--dca-retries", type=int, default=DEFAULT_DCA_RETRIES)
    parser.add_argument("--display", choices=DISPLAY_CHOICES)
    parser.add_argument(
        "--micro-doppler-range-m",
        type=float,
        help=(
            "Fixed range gate for dedicated micro-doppler mode. "
            f"When omitted, prompt with a default of "
            f"{DEFAULT_MICRO_DOPPLER_RANGE_M:g} m."
        ),
    )
    parser.add_argument(
        "--micro-doppler-range-half-width-bins",
        type=int,
        default=1,
        help="Range bins on each side of the dedicated rotor gate.",
    )
    parser.add_argument(
        "--rotor-blades",
        type=int,
        default=DEFAULT_ROTOR_BLADES,
    )
    parser.add_argument("--rotor-count", type=int, default=1)
    parser.add_argument("--rotor-radius-m", type=float)
    parser.add_argument("--rotor-rpm-min", type=float, default=500.0)
    parser.add_argument(
        "--rotor-rpm-max",
        type=float,
        default=DEFAULT_ROTOR_RPM_MAX,
    )
    parser.add_argument(
        "--max-range-m",
        type=float,
        default=DEFAULT_MAX_RANGE_M,
        help=(
            "Maximum range shown by the live display. "
            "Defaults to 10 m; use 0 for the full computed range."
        ),
    )
    parser.add_argument(
        "--cluster-eps-m",
        type=float,
        default=DEFAULT_CLUSTER_EPS_M,
        help=(
            "DBSCAN XYZ neighborhood radius for point-cloud clustering. "
            "Defaults to 0.4 m; use 0 to disable clustering."
        ),
    )
    parser.add_argument(
        "--cluster-min-samples",
        type=int,
        default=DEFAULT_CLUSTER_MIN_SAMPLES,
        help="Minimum points required for a DBSCAN cluster. Defaults to 2.",
    )
    parser.add_argument(
        "--clutter-map-update-rate",
        type=float,
        default=DEFAULT_CLUTTER_MAP_UPDATE_RATE,
        help=(
            "Clutter-map startup-learning EMA rate; the learned map is fixed "
            "after warm-up. Use 0 to disable the map."
        ),
    )
    parser.add_argument(
        "--clutter-map-warmup-frames",
        type=int,
        default=DEFAULT_CLUTTER_MAP_WARMUP_FRAMES,
        help="Frames used to learn the initial clutter map. Defaults to 30.",
    )
    parser.add_argument(
        "--clutter-map-min-snr-db",
        type=float,
        default=DEFAULT_CLUTTER_MAP_MIN_SNR_DB,
        help="Minimum target-to-background power ratio. Defaults to 3 dB.",
    )
    parser.add_argument(
        "--static-detection",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_STATIC_DETECTION,
        help=(
            "Detect motion-qualified stationary changes against a startup "
            "reference. "
            "Enabled by default."
        ),
    )
    parser.add_argument(
        "--static-warmup-frames",
        type=int,
        default=DEFAULT_STATIC_WARMUP_FRAMES,
        help="Processed updates discarded before static calibration. Defaults to 30.",
    )
    parser.add_argument(
        "--static-reference-frames",
        type=int,
        default=DEFAULT_STATIC_REFERENCE_FRAMES,
        help="Processed updates used for static-scene calibration. Defaults to 150.",
    )
    parser.add_argument(
        "--static-min-change-db",
        type=float,
        default=DEFAULT_STATIC_MIN_CHANGE_DB,
        help="Minimum static target-to-reference change. Defaults to 3 dB.",
    )
    parser.add_argument(
        "--static-background-update-rate",
        type=float,
        default=DEFAULT_STATIC_BACKGROUND_UPDATE_RATE,
        help=(
            "Adaptive static-background update rate; use 0 to keep the "
            "startup reference fixed. Defaults to 0."
        ),
    )
    parser.add_argument(
        "--static-cluster-min-samples",
        type=int,
        default=DEFAULT_STATIC_CLUSTER_MIN_SAMPLES,
        help=(
            "Minimum same-frame points in a static handoff cluster. "
            "Defaults to 1; temporal tracking still requires 3 consecutive "
            "associated updates."
        ),
    )
    parser.add_argument(
        "--duration-minutes",
        type=float,
        help=(
            "Run duration in minutes. Use 0 for no time limit. "
            "When omitted, prompt with a default of 5 minutes."
        ),
    )
    parser.add_argument(
        "--display-update-every",
        type=int,
        help=(
            "Update the live display every N valid frames. "
            "Defaults to 1 for all display modes."
        ),
    )
    parser.add_argument("--capture-dir", type=Path, default=DEFAULT_CAPTURE_DIR)
    parser.add_argument(
        "--processed-output",
        type=Path,
        help=(
            "Processed point-cloud and micro-Doppler JSONL output. Defaults "
            "to a timestamped file in --capture-dir."
        ),
    )
    parser.add_argument(
        "--classification",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Run the trained drone/not-drone CNN. Enabled by default; use "
            "--no-classification to run without PyTorch."
        ),
    )
    parser.add_argument(
        "--classification-artifacts",
        type=Path,
        default=DEFAULT_CLASSIFICATION_ARTIFACT_DIR,
        help="Directory containing the trained CNN deployment artifacts.",
    )
    parser.add_argument(
        "--classification-device",
        choices=("auto", "cuda", "cpu"),
        default="auto",
        help=(
            "Inference device. On Jetson, auto requires TensorRT CUDA; "
            "use cpu only as an explicit override."
        ),
    )
    parser.add_argument(
        "--raw-output",
        type=Path,
        help="Optional raw ADC output. Raw recording is disabled by default.",
    )
    return parser.parse_args()


def choose_display(display_arg: Optional[str]) -> str:
    if display_arg:
        return display_arg

    print("Display type:")
    print("  1. none")
    print("  2. range")
    print("  3. range-doppler")
    print("  4. point-cloud")
    print("  5. point-cloud + micro-doppler")
    print("  6. dedicated rotor micro-doppler")
    while True:
        choice = input("Select display type [5]: ").strip()
        if not choice:
            return "point-cloud-micro-doppler"
        if choice in {"1", "none"}:
            return "none"
        if choice in {"2", "range"}:
            return "range"
        if choice in {"3", "range-doppler", "range_doppler"}:
            return "range-doppler"
        if choice in {"4", "point-cloud", "point_cloud"}:
            return "point-cloud"
        if choice in {
            "5",
            "point-cloud-micro-doppler",
            "point_cloud_micro_doppler",
        }:
            return "point-cloud-micro-doppler"
        if choice in {"6", "micro-doppler", "micro_doppler"}:
            return "micro-doppler"
        print(
            "Choose 1, 2, 3, 4, 5, 6, none, range, range-doppler, "
            "point-cloud, micro-doppler, or point-cloud-micro-doppler."
        )


def choose_duration_minutes(duration_arg: Optional[float]) -> float:
    if duration_arg is not None:
        if not math.isfinite(duration_arg) or duration_arg < 0:
            raise ValueError("Run duration must be a finite, non-negative number.")
        return duration_arg

    while True:
        choice = input(
            f"Run duration in minutes (0 for unlimited) "
            f"[{DEFAULT_DURATION_MINUTES:g}]: "
        ).strip()
        if not choice:
            return DEFAULT_DURATION_MINUTES
        try:
            duration_minutes = float(choice)
        except ValueError:
            print("Enter a non-negative number of minutes, or 0 for unlimited.")
            continue
        if math.isfinite(duration_minutes) and duration_minutes >= 0:
            return duration_minutes
        print("Enter a non-negative number of minutes, or 0 for unlimited.")


def choose_micro_doppler_range_m(range_arg: Optional[float]) -> float:
    if range_arg is not None:
        if not math.isfinite(range_arg) or range_arg <= 0.0:
            raise ValueError(
                "Micro-Doppler range must be a finite positive number."
            )
        return range_arg

    while True:
        choice = input(
            "Rotor target range in meters "
            f"[{DEFAULT_MICRO_DOPPLER_RANGE_M:g}]: "
        ).strip()
        if not choice:
            return DEFAULT_MICRO_DOPPLER_RANGE_M
        try:
            target_range_m = float(choice)
        except ValueError:
            print("Enter a finite positive range in meters.")
            continue
        if math.isfinite(target_range_m) and target_range_m > 0.0:
            return target_range_m
        print("Enter a finite positive range in meters.")


def resolve_radar_port(radar_port_arg: Optional[str]) -> str:
    if radar_port_arg:
        return radar_port_arg

    env_port = os.environ.get("RADAR_PORT")
    if env_port:
        return env_port

    detected_ports = detect_serial_ports()
    default_port = default_radar_port()
    if default_port and any(port.device == default_port for port in detected_ports):
        return default_port
    if len(detected_ports) == 1:
        return detected_ports[0].device
    if detected_ports:
        return choose_radar_port(detected_ports)

    return default_port or input("Radar command UART port: ").strip()


def choose_radar_port(ports: list[SerialPortInfo]) -> str:
    print("Radar command UART ports:")
    for index, port in enumerate(ports, start=1):
        description = f" - {port.description}" if port.description else ""
        print(f"  {index}. {port.device}{description}")

    while True:
        choice = input("Select radar command UART: ").strip()
        if choice.isdigit():
            selected_index = int(choice)
            if 1 <= selected_index <= len(ports):
                return ports[selected_index - 1].device
        for port in ports:
            if choice == port.device:
                return port.device
        print("Choose a listed number or serial device path.")


def detect_serial_ports() -> list[SerialPortInfo]:
    try:
        from serial.tools import list_ports
    except ImportError:
        return fallback_serial_ports()

    ports = [
        SerialPortInfo(port.device, port.description or "")
        for port in list_ports.comports()
    ]
    return sorted(ports, key=serial_port_sort_key)


def fallback_serial_ports() -> list[SerialPortInfo]:
    if os.name == "nt":
        return [SerialPortInfo(WINDOWS_DEFAULT_RADAR_PORT)]

    preferred_ports = [
        LINUX_DEFAULT_RADAR_PORT,
        "/dev/ttyUSB1",
        "/dev/ttyACM1",
        "/dev/ttyACM0",
    ]
    return [SerialPortInfo(port) for port in preferred_ports if Path(port).exists()]


def serial_port_sort_key(port: SerialPortInfo) -> tuple[int, str]:
    preferred_order = {
        WINDOWS_DEFAULT_RADAR_PORT: 0,
        LINUX_DEFAULT_RADAR_PORT: 0,
        "/dev/ttyUSB1": 1,
        "/dev/ttyACM1": 2,
        "/dev/ttyACM0": 3,
    }
    return (preferred_order.get(port.device, 99), port.device)


def default_radar_port() -> Optional[str]:
    if os.name == "nt":
        return WINDOWS_DEFAULT_RADAR_PORT
    return LINUX_DEFAULT_RADAR_PORT


def default_processed_output(capture_dir: Path) -> Path:
    timestamp = datetime.now().astimezone().strftime("%Y_%m_%dT%H_%M_%S")
    return capture_dir / f"processed_capture_{timestamp}.jsonl"


def start_process(
    label: str,
    command: list[str],
    *,
    capture_output: bool = False,
) -> subprocess.Popen:
    print(f"Starting {label}:")
    print(" ".join(str(part) for part in command))
    output_options = (
        {
            "stdout": subprocess.PIPE,
            "stderr": subprocess.STDOUT,
            "text": True,
            "bufsize": 1,
        }
        if capture_output
        else {}
    )
    return subprocess.Popen(
        command,
        env=subprocess_environment(),
        **output_options,
        **subprocess_startup_options(),
    )


def relay_capture_output(
    process: subprocess.Popen,
    classification_queue: queue.SimpleQueue,
    capture_ready: Optional[threading.Event] = None,
) -> None:
    if process.stdout is None:
        return
    for raw_line in process.stdout:
        line = raw_line.rstrip("\r\n")
        if capture_ready is not None and line.startswith(CAPTURE_READY_PREFIX):
            capture_ready.set()
        marker_index = line.find(CLASSIFICATION_RESULT_PREFIX)
        if marker_index < 0:
            print(line, flush=True)
            continue
        payload = line[marker_index + len(CLASSIFICATION_RESULT_PREFIX) :]
        try:
            result = json.loads(payload)
        except (TypeError, ValueError):
            print(line, flush=True)
            continue
        if isinstance(result, dict):
            classification_queue.put(result)


def wait_for_capture_ready(
    process: subprocess.Popen,
    capture_ready: threading.Event,
    timeout_seconds: float = CAPTURE_STARTUP_TIMEOUT_SECONDS,
) -> bool:
    deadline = time.monotonic() + max(timeout_seconds, 0.0)
    while not capture_ready.is_set():
        if process.poll() is not None:
            return False
        remaining_seconds = deadline - time.monotonic()
        if remaining_seconds <= 0.0:
            return False
        capture_ready.wait(min(0.1, remaining_seconds))
    return process.poll() is None


def capture_startup_timeout_seconds(args: argparse.Namespace) -> float:
    """Allow first-run TensorRT build and parity validation to finish."""
    if not bool(getattr(args, "classification", False)):
        return CAPTURE_STARTUP_TIMEOUT_SECONDS
    requested_device = str(
        getattr(args, "classification_device", "auto")
    ).strip().lower()
    if requested_device == "cpu":
        return CAPTURE_STARTUP_TIMEOUT_SECONDS
    if requested_device == "cuda":
        return CAPTURE_GPU_STARTUP_TIMEOUT_SECONDS
    try:
        model = JETSON_MODEL_PATH.read_text(
            encoding="utf-8",
            errors="ignore",
        )
    except OSError:
        return CAPTURE_STARTUP_TIMEOUT_SECONDS
    return (
        CAPTURE_GPU_STARTUP_TIMEOUT_SECONDS
        if "nvidia" in model.lower()
        else CAPTURE_STARTUP_TIMEOUT_SECONDS
    )


def report_pending_classifications(
    classification_queue: queue.SimpleQueue,
    latest: Optional[dict] = None,
) -> Optional[dict]:
    while True:
        try:
            latest = classification_queue.get_nowait()
        except queue.Empty:
            break
        label = latest.get("label", "unknown")
        if label in {"drone", "not_drone"}:
            probability = latest.get("p_drone")
            probability_text = (
                f", p_drone={float(probability):.3f}"
                if probability is not None
                else ""
            )
            print(
                f"Classification: {label.upper()}{probability_text}",
                flush=True,
            )
        else:
            print(
                "Classification: UNKNOWN "
                f"({latest.get('reason') or latest.get('status', 'waiting')}, "
                f"valid_steps={int(latest.get('valid_steps', 0))}/48)",
                flush=True,
            )
    return latest


def subprocess_startup_options() -> dict:
    if os.name == "nt":
        return {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP}
    return {"start_new_session": True}


def subprocess_environment() -> dict[str, str]:
    environment = os.environ.copy()
    if os.name == "nt":
        return environment

    x11_socket = Path("/tmp/.X11-unix/X0")
    if not environment.get("DISPLAY") and x11_socket.exists():
        environment["DISPLAY"] = ":0"

    if not environment.get("XAUTHORITY") and hasattr(os, "getuid"):
        gdm_xauthority = Path(
            f"/run/user/{os.getuid()}/gdm/Xauthority"
        )
        if gdm_xauthority.is_file():
            environment["XAUTHORITY"] = str(gdm_xauthority)
    return environment


def stop_process(label: str, process: Optional[subprocess.Popen]) -> None:
    if process is None or process.poll() is not None:
        return

    print(f"Stopping {label}...")
    try:
        send_interrupt(process)
    except OSError:
        return
    try:
        process.wait(timeout=8.0)
        print(f"{label} stopped.")
        return
    except subprocess.TimeoutExpired:
        print(f"{label} did not stop after Ctrl+C; terminating.")

    terminate_process(process)
    try:
        process.wait(timeout=3.0)
        return
    except subprocess.TimeoutExpired:
        print(f"{label} did not terminate; killing.")
        kill_process(process)
        process.wait(timeout=3.0)


def send_interrupt(process: subprocess.Popen) -> None:
    if os.name == "nt":
        process.send_signal(signal.CTRL_BREAK_EVENT)
    else:
        os.killpg(process.pid, signal.SIGINT)


def terminate_process(process: subprocess.Popen) -> None:
    if os.name == "nt":
        process.terminate()
    else:
        os.killpg(process.pid, signal.SIGTERM)


def kill_process(process: subprocess.Popen) -> None:
    if os.name == "nt":
        process.kill()
    else:
        os.killpg(process.pid, signal.SIGKILL)


def build_capture_command(
    args: argparse.Namespace,
    display: str,
    processed_output: Path,
    raw_output: Optional[Path] = None,
) -> list[str]:
    display_update_every = args.display_update_every
    if display_update_every is None:
        display_update_every = 1

    command = [
        sys.executable,
        "-u",
        str(RAW_DATA_DIR / "livedatacapture.py"),
        "--config",
        str(args.config),
        "--setup",
        str(args.setup),
        "--host-ip",
        args.host_ip,
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
        str(max(display_update_every, 1)),
        "--max-range-m",
        str(max(args.max_range_m, 0.0)),
        "--cluster-eps-m",
        str(max(args.cluster_eps_m, 0.0)),
        "--cluster-min-samples",
        str(max(args.cluster_min_samples, 1)),
        "--clutter-map-update-rate",
        str(max(args.clutter_map_update_rate, 0.0)),
        "--clutter-map-warmup-frames",
        str(max(args.clutter_map_warmup_frames, 1)),
        "--clutter-map-min-snr-db",
        str(max(args.clutter_map_min_snr_db, 0.0)),
        (
            "--static-detection"
            if args.static_detection and display != "micro-doppler"
            else "--no-static-detection"
        ),
        "--static-warmup-frames",
        str(max(args.static_warmup_frames, 0)),
        "--static-reference-frames",
        str(max(args.static_reference_frames, 1)),
        "--static-min-change-db",
        str(max(args.static_min_change_db, 0.0)),
        "--static-background-update-rate",
        str(min(max(args.static_background_update_rate, 0.0), 1.0)),
        "--static-cluster-min-samples",
        str(max(args.static_cluster_min_samples, 1)),
        "--processed-output",
        str(processed_output),
    ]
    if getattr(args, "classification", True):
        classification_artifacts = getattr(
            args,
            "classification_artifacts",
            DEFAULT_CLASSIFICATION_ARTIFACT_DIR,
        )
        command.extend(
            (
                "--classification",
                "--classification-artifacts",
                str(classification_artifacts),
                "--classification-device",
                str(getattr(args, "classification_device", "auto")),
            )
        )
    else:
        command.append("--no-classification")
    if display == "micro-doppler":
        target_range_m = getattr(args, "micro_doppler_range_m", None)
        if target_range_m is None:
            raise ValueError(
                "--micro-doppler-range-m is required for dedicated "
                "micro-doppler mode"
            )
        command.extend(
            (
                "--micro-doppler-range-m",
                str(target_range_m),
                "--micro-doppler-range-half-width-bins",
                str(
                    max(
                        getattr(
                            args,
                            "micro_doppler_range_half_width_bins",
                            1,
                        ),
                        0,
                    )
                ),
                "--rotor-blades",
                str(
                    max(
                        getattr(
                            args,
                            "rotor_blades",
                            DEFAULT_ROTOR_BLADES,
                        ),
                        1,
                    )
                ),
                "--rotor-count",
                str(max(getattr(args, "rotor_count", 1), 1)),
                "--rotor-rpm-min",
                str(max(getattr(args, "rotor_rpm_min", 500.0), 1.0)),
                "--rotor-rpm-max",
                str(
                    max(
                        getattr(
                            args,
                            "rotor_rpm_max",
                            DEFAULT_ROTOR_RPM_MAX,
                        ),
                        1.0,
                    )
                ),
            )
        )
        rotor_radius_m = getattr(args, "rotor_radius_m", None)
        if rotor_radius_m is not None:
            command.extend(
                ("--rotor-radius-m", str(max(rotor_radius_m, 0.0)))
            )
    if raw_output is not None:
        command.extend(("--raw-output", str(raw_output)))
    return command


def build_startup_command(args: argparse.Namespace, radar_port: str) -> list[str]:
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
        args.host_ip,
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


def main() -> int:
    args = parse_args()
    display = choose_display(args.display)
    if display == "micro-doppler":
        try:
            args.micro_doppler_range_m = choose_micro_doppler_range_m(
                args.micro_doppler_range_m
            )
        except ValueError as exc:
            print(f"Invalid micro-Doppler range: {exc}", file=sys.stderr)
            return 2
    try:
        duration_minutes = choose_duration_minutes(args.duration_minutes)
    except ValueError as exc:
        print(f"Invalid duration: {exc}", file=sys.stderr)
        return 2
    radar_port = resolve_radar_port(args.radar_port)
    if not radar_port:
        print("No radar command UART port was provided.", file=sys.stderr)
        return 2

    processed_output = args.processed_output or default_processed_output(
        args.capture_dir
    )
    processed_output = (
        processed_output if processed_output.is_absolute() else ROOT / processed_output
    )
    processed_output.parent.mkdir(parents=True, exist_ok=True)
    raw_output = args.raw_output
    if raw_output is not None:
        raw_output = raw_output if raw_output.is_absolute() else ROOT / raw_output
        raw_output.parent.mkdir(parents=True, exist_ok=True)
    if not args.classification_artifacts.is_absolute():
        args.classification_artifacts = ROOT / args.classification_artifacts

    print(f"Display mode: {display}")
    duration_text = (
        "unlimited" if duration_minutes == 0 else f"{duration_minutes:g} minute(s)"
    )
    print(f"Run duration: {duration_text}")
    print(f"Radar command UART: {radar_port}")
    print(f"Processed output: {processed_output}")
    print(f"Raw output: {raw_output if raw_output is not None else 'disabled'}")
    print(
        "CNN classification: "
        + (
            f"enabled ({args.classification_artifacts}, "
            f"device={args.classification_device})"
            if args.classification
            else "disabled"
        )
    )

    capture_process: Optional[subprocess.Popen] = None
    startup_process: Optional[subprocess.Popen] = None
    capture_output_thread: Optional[threading.Thread] = None
    classification_queue: queue.SimpleQueue = queue.SimpleQueue()
    capture_ready = threading.Event()
    latest_classification: Optional[dict] = None
    try:
        capture_process = start_process(
            "live capture",
            build_capture_command(args, display, processed_output, raw_output),
            capture_output=True,
        )
        capture_output_thread = threading.Thread(
            target=relay_capture_output,
            args=(capture_process, classification_queue, capture_ready),
            name="LiveCaptureOutputRelay",
            daemon=True,
        )
        capture_output_thread.start()
        capture_startup_timeout = capture_startup_timeout_seconds(args)
        if not wait_for_capture_ready(
            capture_process,
            capture_ready,
            timeout_seconds=capture_startup_timeout,
        ):
            capture_returncode = capture_process.poll()
            if capture_returncode is None:
                print(
                    "live capture did not become ready within "
                    f"{capture_startup_timeout:g} seconds",
                    file=sys.stderr,
                )
                return 1
            capture_output_thread.join(timeout=2.0)
            print(
                f"live capture exited before becoming ready with code "
                f"{capture_returncode}",
                file=sys.stderr,
            )
            return capture_returncode or 1

        startup_process = start_process(
            "startup",
            build_startup_command(args, radar_port),
        )
        deadline = (
            None
            if duration_minutes == 0
            else time.monotonic() + (duration_minutes * 60.0)
        )

        while True:
            latest_classification = report_pending_classifications(
                classification_queue,
                latest_classification,
            )
            if startup_process.poll() is not None:
                print(f"startup exited with code {startup_process.returncode}")
                return startup_process.returncode or 0
            if capture_process.poll() is not None:
                print(f"live capture exited with code {capture_process.returncode}")
                return capture_process.returncode or 0
            if deadline is not None:
                remaining_seconds = deadline - time.monotonic()
                if remaining_seconds <= 0:
                    print(
                        f"Run duration of {duration_minutes:g} minute(s) reached. "
                        "Stopping radar and capture."
                    )
                    return 0
                time.sleep(min(0.5, remaining_seconds))
            else:
                time.sleep(0.5)
    except KeyboardInterrupt:
        print("\nCtrl+C received. Stopping radar and capture.")
        return 0
    finally:
        stop_process("startup", startup_process)
        stop_process("live capture", capture_process)
        if capture_output_thread is not None:
            capture_output_thread.join(timeout=2.0)
        report_pending_classifications(
            classification_queue,
            latest_classification,
        )
        print(f"Processed data file: {processed_output}")
        if raw_output is not None:
            print(f"Raw data file: {raw_output}")
            print(f"Raw metadata file: {raw_output}.json")


if __name__ == "__main__":
    raise SystemExit(main())
