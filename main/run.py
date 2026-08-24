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

if __package__ in {None, ""}:
    repository_root = str(Path(__file__).resolve().parent.parent)
    if repository_root not in sys.path:
        sys.path.insert(0, repository_root)
    from main import calibrate as radar_calibration
else:
    from . import calibrate as radar_calibration


ROOT = Path(__file__).resolve().parent.parent
MAIN_DIR = ROOT / "main"
PROFILES_DIR = ROOT / "profiles"
DEFAULT_CONFIG_PATH = PROFILES_DIR / "profile.cfg"
DEFAULT_CALIBRATION_PROFILE_PATH = ROOT / "profiles" / "profile_calibration.cfg"
DEFAULT_SETUP_PATH = PROFILES_DIR / "setup.json"
DEFAULT_CALIBRATION_OUTPUT_DIR = ROOT / "calibrationoutput"
DEFAULT_DATASET_DIR = ROOT / "dataset"
UAV_DATASET_DIR = DEFAULT_DATASET_DIR / "uav"
OTHERS_DATASET_DIR = DEFAULT_DATASET_DIR / "others"
DEFAULT_CLASSIFICATION_ARTIFACT_DIR = ROOT / "model_weights"
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
    radar_calibration.CALIBRATION_DISPLAY_MODE,
    radar_calibration.AZIMUTH_CALIBRATION_DISPLAY_MODE,
    radar_calibration.ELEVATION_CALIBRATION_DISPLAY_MODE,
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
        "--calibration-profile",
        type=Path,
        default=DEFAULT_CALIBRATION_PROFILE_PATH,
        help="Source profile used only to create the temporary calibration profile.",
    )
    parser.add_argument(
        "--calibration-distance-m",
        type=float,
        help="Laser-measured corner-reflector distance in meters.",
    )
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
    parser.add_argument(
        "--calibration-output",
        type=Path,
        help="JSON calibration report path (timestamped in --capture-dir by default).",
    )
    parser.add_argument(
        "--calibration-angle-deg",
        type=float,
        help="Known tripod angle for azimuth/elevation calibration.",
    )
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
            "Defaults to 1; a motion-qualified handoff confirms on the first "
            "associated static return."
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
    parser.add_argument(
        "--capture-dir",
        type=Path,
        default=DEFAULT_CALIBRATION_OUTPUT_DIR,
    )
    parser.add_argument(
        "--processed-output",
        type=Path,
        help=(
            "Processed point-cloud and micro-Doppler JSONL output. Combined "
            "mode prompts for a dataset folder; other modes default to a "
            "timestamped file in --capture-dir."
        ),
    )
    parser.add_argument(
        "--classification",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Run the trained drone/not-drone CNN. Combined display mode "
            "prompts when omitted; use --classification or "
            "--no-classification to skip that prompt."
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
    print("  7. range + channel calibration")
    print("  8. azimuth calibration")
    print("  9. elevation calibration")
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
        if choice in {"7", "calibration"}:
            return radar_calibration.CALIBRATION_DISPLAY_MODE
        if choice in {"8", "azimuth-calibration", "azimuth_calibration"}:
            return radar_calibration.AZIMUTH_CALIBRATION_DISPLAY_MODE
        if choice in {"9", "elevation-calibration", "elevation_calibration"}:
            return radar_calibration.ELEVATION_CALIBRATION_DISPLAY_MODE
        print(
            "Choose 1 through 9, none, range, range-doppler, point-cloud, "
            "micro-doppler, point-cloud-micro-doppler, calibration, "
            "azimuth-calibration, or elevation-calibration."
        )


def choose_live_classification(classification_arg: Optional[bool]) -> bool:
    if classification_arg is not None:
        return bool(classification_arg)

    while True:
        choice = input("Use live CNN classification? [y/N]: ").strip().lower()
        if choice in {"", "n", "no"}:
            return False
        if choice in {"y", "yes"}:
            return True
        print("Enter y for yes or n for no.")


def choose_dataset_output_directory() -> Path:
    print("Processed-data location:")
    print("  1. dataset/uav")
    print("  2. dataset/others")
    print("  3. dataset")
    while True:
        choice = input("Select data location [3]: ").strip().lower()
        if choice in {"", "3", "dataset", "/dataset"}:
            return DEFAULT_DATASET_DIR
        if choice in {"1", "uav", "dataset/uav", "/dataset/uav"}:
            return UAV_DATASET_DIR
        if choice in {
            "2",
            "others",
            "dataset/others",
            "/dataset/others",
        }:
            return OTHERS_DATASET_DIR
        print("Choose 1 for UAV, 2 for others, or 3 for the dataset root.")


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


def choose_calibration_distance_m(distance_arg: Optional[float]) -> float:
    if distance_arg is not None:
        if not math.isfinite(distance_arg) or distance_arg <= 0.0:
            raise ValueError("Laser distance must be a finite positive number.")
        return distance_arg

    while True:
        choice = input(
            "Laser-measured reflector distance in meters "
            f"[{radar_calibration.DEFAULT_TARGET_DISTANCE_M:g}]: "
        ).strip()
        if not choice:
            return radar_calibration.DEFAULT_TARGET_DISTANCE_M
        try:
            distance_m = float(choice)
        except ValueError:
            print("Enter a finite positive distance in meters.")
            continue
        if math.isfinite(distance_m) and distance_m > 0.0:
            return distance_m
        print("Enter a finite positive distance in meters.")


def choose_calibration_angle_deg(
    angle_arg: Optional[float],
    calibration_type: str,
) -> float:
    label = "Azimuth" if calibration_type == "azimuth" else "Elevation"
    if angle_arg is not None:
        if not math.isfinite(angle_arg) or abs(angle_arg) > 60.0:
            raise ValueError("Calibration angle must be within -60 to +60 degrees.")
        return angle_arg
    while True:
        choice = input(
            f"Known tripod {label.lower()} angle in degrees "
            f"[{radar_calibration.DEFAULT_REFERENCE_ANGLE_DEG:g}]: "
        ).strip()
        if not choice:
            return radar_calibration.DEFAULT_REFERENCE_ANGLE_DEG
        try:
            angle_deg = float(choice)
        except ValueError:
            print("Enter an angle from -60 to +60 degrees.")
            continue
        if math.isfinite(angle_deg) and abs(angle_deg) <= 60.0:
            return angle_deg
        print("Enter an angle from -60 to +60 degrees.")


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
    calibration_queue: Optional[queue.SimpleQueue] = None,
) -> None:
    if process.stdout is None:
        return
    for raw_line in process.stdout:
        line = raw_line.rstrip("\r\n")
        if capture_ready is not None and line.startswith(CAPTURE_READY_PREFIX):
            capture_ready.set()
        calibration_index = line.find(radar_calibration.CALIBRATION_RESULT_PREFIX)
        if calibration_index >= 0:
            payload = line[
                calibration_index + len(radar_calibration.CALIBRATION_RESULT_PREFIX) :
            ]
            try:
                result = json.loads(payload)
            except (TypeError, ValueError):
                print(line, flush=True)
                continue
            if isinstance(result, dict) and calibration_queue is not None:
                calibration_queue.put(result)
            continue
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
    """Drain live classification messages without printing them.

    Classification is rendered in the PyQt display by the capture process.
    The channel remains for compatibility with existing capture subprocesses.
    """
    while True:
        try:
            latest = classification_queue.get_nowait()
        except queue.Empty:
            break
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
    processed_output: Optional[Path],
    raw_output: Optional[Path] = None,
    config_path: Optional[Path] = None,
    host_compensation_profile: Optional[Path] = None,
) -> list[str]:
    display_update_every = args.display_update_every
    if display_update_every is None:
        display_update_every = 1

    command = [
        sys.executable,
        "-u",
        str(MAIN_DIR / "livedatacapture.py"),
        "--config",
        str(config_path or args.config),
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
            if args.static_detection
            and display not in (
                {"micro-doppler"} | radar_calibration.CALIBRATION_DISPLAY_MODES
            )
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
    ]
    if processed_output is not None and display not in radar_calibration.CALIBRATION_DISPLAY_MODES:
        command.extend(("--processed-output", str(processed_output)))
    if (
        display not in radar_calibration.CALIBRATION_DISPLAY_MODES
        and getattr(args, "classification", True)
    ):
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
    if display in radar_calibration.CALIBRATION_DISPLAY_MODES:
        calibration_distance_m = getattr(args, "calibration_distance_m", None)
        if calibration_distance_m is None:
            calibration_distance_m = radar_calibration.DEFAULT_TARGET_DISTANCE_M
        calibration_angle_deg = getattr(args, "calibration_angle_deg", None)
        if calibration_angle_deg is None:
            calibration_angle_deg = radar_calibration.DEFAULT_REFERENCE_ANGLE_DEG
        command.extend(
            (
                "--calibration-distance-m",
                str(calibration_distance_m),
                "--calibration-search-window-m",
                str(getattr(args, "calibration_search_window_m", radar_calibration.DEFAULT_SEARCH_WINDOW_M)),
                "--calibration-warmup-frames",
                str(max(getattr(args, "calibration_warmup_frames", radar_calibration.DEFAULT_WARMUP_FRAMES), 0)),
                "--calibration-frames",
                str(max(getattr(args, "calibration_frames", radar_calibration.DEFAULT_ACCEPTED_FRAMES), 1)),
                "--calibration-timeout-seconds",
                str(getattr(args, "calibration_timeout_seconds", radar_calibration.DEFAULT_TIMEOUT_SECONDS)),
                "--calibration-angle-deg",
                str(calibration_angle_deg),
            )
        )
    if display in {
        radar_calibration.AZIMUTH_CALIBRATION_DISPLAY_MODE,
        radar_calibration.ELEVATION_CALIBRATION_DISPLAY_MODE,
    }:
        if host_compensation_profile is None:
            raise ValueError(
                "Angular calibration requires a host compensation profile"
            )
        command.extend(
            ("--host-compensation-profile", str(host_compensation_profile))
        )
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


def build_startup_command(
    args: argparse.Namespace,
    radar_port: str,
    config_path: Optional[Path] = None,
) -> list[str]:
    effective_config = config_path or args.config
    return [
        sys.executable,
        str(MAIN_DIR / "startup.py"),
        "--config",
        str(effective_config),
        "--sdk-profile",
        str(effective_config),
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


def default_calibration_output(
    capture_dir: Path,
    calibration_type: str = "range",
) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    prefix = "calibration" if calibration_type == "range" else f"{calibration_type}_calibration"
    return capture_dir / f"{prefix}_{timestamp}.json"


def run_calibration_mode(
    args: argparse.Namespace,
    radar_port: str,
    display: str = radar_calibration.CALIBRATION_DISPLAY_MODE,
) -> int:
    """Run calibration to completion, stop hardware, then optionally apply it."""

    calibration_type = {
        radar_calibration.CALIBRATION_DISPLAY_MODE: "range",
        radar_calibration.AZIMUTH_CALIBRATION_DISPLAY_MODE: "azimuth",
        radar_calibration.ELEVATION_CALIBRATION_DISPLAY_MODE: "elevation",
    }.get(display)
    if calibration_type is None:
        print(f"Unsupported calibration display mode: {display}", file=sys.stderr)
        return 2
    try:
        reference_angle_deg = (
            choose_calibration_angle_deg(
                getattr(args, "calibration_angle_deg", None), calibration_type
            )
            if calibration_type in {"azimuth", "elevation"}
            else radar_calibration.DEFAULT_REFERENCE_ANGLE_DEG
        )
        settings = radar_calibration.CalibrationSettings(
            target_distance_m=choose_calibration_distance_m(
                args.calibration_distance_m
            ),
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
    args.classification = False
    args.static_detection = False

    source_profile = args.calibration_profile
    if not source_profile.is_absolute():
        source_profile = ROOT / source_profile
    operational_profile = args.config
    if not operational_profile.is_absolute():
        operational_profile = ROOT / operational_profile
    report_path = args.calibration_output or default_calibration_output(
        args.capture_dir, calibration_type
    )
    if not report_path.is_absolute():
        report_path = ROOT / report_path

    try:
        from main.livedatacapture import RadarCaptureConfig

        source_config = RadarCaptureConfig.from_file(source_profile)
    except (OSError, ValueError) as exc:
        print(f"Calibration profile is invalid: {exc}", file=sys.stderr)
        return 2

    capture_process: Optional[subprocess.Popen] = None
    startup_process: Optional[subprocess.Popen] = None
    capture_output_thread: Optional[threading.Thread] = None
    classification_queue: queue.SimpleQueue = queue.SimpleQueue()
    calibration_queue: queue.SimpleQueue = queue.SimpleQueue()
    capture_ready = threading.Event()
    result: Optional[
        radar_calibration.CalibrationResult
        | radar_calibration.AngularCalibrationResult
    ] = None
    temporary_profile_directory: Optional[tempfile.TemporaryDirectory] = None

    try:
        temporary_profile_directory = tempfile.TemporaryDirectory(
            prefix="radar_calibration_"
        )
        with nullcontext(temporary_profile_directory.name) as temporary_dir:
            runtime_profile = Path(temporary_dir) / "profile_calibration_runtime.cfg"
            try:
                radar_calibration.create_runtime_profile(
                    source_profile,
                    runtime_profile,
                    source_config,
                    settings,
                )
            except (OSError, ValueError) as exc:
                print(f"Could not create calibration runtime profile: {exc}", file=sys.stderr)
                return 2

            print(f"Display mode: {display}")
            print(f"Laser distance: {settings.target_distance_m:g} m")
            print(f"Search window: ±{settings.search_window_m:g} m")
            if calibration_type in {"azimuth", "elevation"}:
                print(
                    f"Known {calibration_type} angle: "
                    f"{settings.reference_angle_deg:+g} degrees"
                )
            print(
                "Calibration frames: "
                f"ignore {settings.warmup_frames}, accept {settings.accepted_frames}"
            )
            print(f"Calibration timeout: {settings.timeout_seconds:g} seconds")
            print(f"Calibration source profile: {source_profile}")
            print(f"Radar command UART: {radar_port}")

            capture_process = start_process(
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
            capture_output_thread = threading.Thread(
                target=relay_capture_output,
                args=(
                    capture_process,
                    classification_queue,
                    capture_ready,
                    calibration_queue,
                ),
                name="CalibrationCaptureOutputRelay",
                daemon=True,
            )
            capture_output_thread.start()
            if not wait_for_capture_ready(
                capture_process,
                capture_ready,
                timeout_seconds=CAPTURE_STARTUP_TIMEOUT_SECONDS,
            ):
                print("Calibration capture did not become ready.", file=sys.stderr)
                return capture_process.poll() or 1

            startup_process = start_process(
                "calibration startup",
                build_startup_command(args, radar_port, runtime_profile),
            )
            deadline = time.monotonic() + settings.timeout_seconds
            while time.monotonic() < deadline:
                try:
                    payload = calibration_queue.get_nowait()
                except queue.Empty:
                    payload = None
                if payload is not None:
                    try:
                        if payload.get("calibration_type", "range") == "range":
                            result = radar_calibration.CalibrationResult.from_dict(payload)
                        else:
                            result = radar_calibration.AngularCalibrationResult.from_dict(
                                payload
                            )
                    except (KeyError, TypeError, ValueError) as exc:
                        print(f"Invalid calibration result: {exc}", file=sys.stderr)
                        return 1
                    break
                if startup_process.poll() is not None:
                    print(
                        f"Calibration startup exited with code {startup_process.returncode}",
                        file=sys.stderr,
                    )
                    return startup_process.returncode or 1
                if capture_process.poll() is not None:
                    capture_output_thread.join(timeout=1.0)
                    try:
                        payload = calibration_queue.get_nowait()
                    except queue.Empty:
                        payload = None
                    if payload is not None:
                        if payload.get("calibration_type", "range") == "range":
                            result = radar_calibration.CalibrationResult.from_dict(payload)
                        else:
                            result = radar_calibration.AngularCalibrationResult.from_dict(
                                payload
                            )
                        break
                    print(
                        f"Calibration capture exited with code {capture_process.returncode}",
                        file=sys.stderr,
                    )
                    return capture_process.returncode or 1
                time.sleep(0.1)
            if result is None:
                print(
                    "Calibration timed out before a stable result; operational profile unchanged.",
                    file=sys.stderr,
                )
                return 1
    except KeyboardInterrupt:
        print("\nCalibration interrupted; operational profile unchanged.")
        return 130
    finally:
        # Hardware is always stopped before a result is previewed or applied.
        stop_process("calibration startup", startup_process)
        stop_process("calibration capture", capture_process)
        if capture_output_thread is not None:
            capture_output_thread.join(timeout=2.0)
        if temporary_profile_directory is not None:
            temporary_profile_directory.cleanup()

    if result is None:
        return 1
    try:
        radar_calibration.write_calibration_report(report_path, result)
    except OSError as exc:
        print(f"Could not write calibration report: {exc}", file=sys.stderr)
        return 1

    print(f"Calibration report: {report_path}")
    if isinstance(result, radar_calibration.CalibrationResult):
        print("Calibration command preview:")
        print(result.command)
    else:
        print("Host angle calibration preview:")
        print(
            f"{result.calibration_type}: reference={result.reference_angle_deg:+.3f} deg, "
            f"measured={result.measured_angle_deg:+.3f} deg, "
            f"bias={result.angle_bias_deg:+.3f} deg"
        )
    try:
        confirmation = input(
            f"Apply this command to {operational_profile}? [y/N]: "
        ).strip().lower()
    except (EOFError, KeyboardInterrupt):
        confirmation = ""
    if confirmation not in {"y", "yes"}:
        print("Calibration was not applied; operational profile unchanged.")
        return 0
    try:
        if isinstance(result, radar_calibration.CalibrationResult):
            backup = radar_calibration.apply_calibration_to_profile(
                operational_profile,
                result,
            )
        else:
            backup = radar_calibration.apply_angular_calibration_to_profile(
                operational_profile,
                result,
            )
    except (OSError, ValueError) as exc:
        print(f"Could not apply calibration: {exc}", file=sys.stderr)
        return 1
    print(f"Calibration applied to: {operational_profile}")
    print(f"Profile backup: {backup}")
    return 0


def main() -> int:
    args = parse_args()
    display = choose_display(args.display)
    if display == "point-cloud-micro-doppler":
        args.classification = choose_live_classification(args.classification)
        if args.processed_output is None:
            args.processed_output = default_processed_output(
                choose_dataset_output_directory()
            )
    elif args.classification is None:
        # Preserve the historical default outside the newly prompted combined
        # data-collection workflow.
        args.classification = True
    if display == "micro-doppler":
        try:
            args.micro_doppler_range_m = choose_micro_doppler_range_m(
                args.micro_doppler_range_m
            )
        except ValueError as exc:
            print(f"Invalid micro-Doppler range: {exc}", file=sys.stderr)
            return 2
    duration_minutes = 0.0
    if display not in radar_calibration.CALIBRATION_DISPLAY_MODES:
        try:
            duration_minutes = choose_duration_minutes(args.duration_minutes)
        except ValueError as exc:
            print(f"Invalid duration: {exc}", file=sys.stderr)
            return 2
    radar_port = resolve_radar_port(args.radar_port)
    if not radar_port:
        print("No radar command UART port was provided.", file=sys.stderr)
        return 2

    if display in radar_calibration.CALIBRATION_DISPLAY_MODES:
        return run_calibration_mode(args, radar_port, display)

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
