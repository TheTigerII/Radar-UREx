import argparse
import os
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional


ROOT = Path(__file__).resolve().parent
RAW_DATA_DIR = ROOT / "rawdatacapture"
DEFAULT_CONFIG_PATH = RAW_DATA_DIR / "profile.cfg"
DEFAULT_SETUP_PATH = RAW_DATA_DIR / "setup.json"
DEFAULT_CAPTURE_DIR = RAW_DATA_DIR / "captures"
DEFAULT_HOST_IP = "192.168.33.30"
DEFAULT_DATA_PORT = 4098
DEFAULT_RADAR_BAUD = 115200
DEFAULT_RADAR_COMMAND_TIMEOUT = 10.0
DEFAULT_DCA_TIMEOUT = 3.0
DEFAULT_DCA_RETRIES = 5
DISPLAY_CHOICES = ("none", "range", "range-doppler")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Start live DCA1000 capture, then start radar/DCA1000 hardware. "
            "Press Ctrl+C to stop startup first, then capture."
        )
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--setup", type=Path, default=DEFAULT_SETUP_PATH)
    parser.add_argument("--host-ip", default=DEFAULT_HOST_IP)
    parser.add_argument("--data-port", type=int, default=DEFAULT_DATA_PORT)
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
    parser.add_argument("--capture-dir", type=Path, default=DEFAULT_CAPTURE_DIR)
    parser.add_argument("--raw-output", type=Path)
    return parser.parse_args()


def choose_display(display_arg: Optional[str]) -> str:
    if display_arg:
        return display_arg

    print("Display type:")
    print("  1. none")
    print("  2. range")
    print("  3. range-doppler")
    while True:
        choice = input("Select display type [2]: ").strip()
        if not choice:
            return "range"
        if choice in {"1", "none"}:
            return "none"
        if choice in {"2", "range"}:
            return "range"
        if choice in {"3", "range-doppler", "range_doppler"}:
            return "range-doppler"
        print("Choose 1, 2, 3, none, range, or range-doppler.")


def resolve_radar_port(radar_port_arg: Optional[str]) -> str:
    if radar_port_arg:
        return radar_port_arg

    env_port = os.environ.get("RADAR_PORT")
    if env_port:
        return env_port

    detected_port = detect_radar_port()
    if detected_port:
        return detected_port

    return input("Radar command UART port: ").strip()


def detect_radar_port() -> Optional[str]:
    if os.name == "nt":
        try:
            from serial.tools import list_ports
        except ImportError:
            return "COM4"

        ports = [port.device for port in list_ports.comports()]
        if "COM4" in ports:
            return "COM4"
        return ports[0] if ports else "COM4"

    preferred_ports = (
        "/dev/ttyUSB1",
        "/dev/ttyUSB0",
        "/dev/ttyACM1",
        "/dev/ttyACM0",
    )
    for port in preferred_ports:
        if Path(port).exists():
            return port
    return None


def default_raw_output(capture_dir: Path) -> Path:
    timestamp = datetime.now().astimezone().strftime("%Y_%m_%dT%H_%M_%S")
    return capture_dir / f"raw_capture_{timestamp}.bin"


def start_process(label: str, command: list[str]) -> subprocess.Popen:
    print(f"Starting {label}:")
    print(" ".join(str(part) for part in command))
    return subprocess.Popen(command, **subprocess_startup_options())


def subprocess_startup_options() -> dict:
    if os.name == "nt":
        return {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP}
    return {"start_new_session": True}


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


def build_capture_command(args: argparse.Namespace, display: str, raw_output: Path) -> list[str]:
    return [
        sys.executable,
        str(RAW_DATA_DIR / "livedatacapture.py"),
        "--config",
        str(args.config),
        "--setup",
        str(args.setup),
        "--host-ip",
        args.host_ip,
        "--data-port",
        str(args.data_port),
        "--display",
        display,
        "--raw-output",
        str(raw_output),
    ]


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
    radar_port = resolve_radar_port(args.radar_port)
    if not radar_port:
        print("No radar command UART port was provided.", file=sys.stderr)
        return 2

    raw_output = args.raw_output or default_raw_output(args.capture_dir)
    raw_output = raw_output if raw_output.is_absolute() else ROOT / raw_output
    raw_output.parent.mkdir(parents=True, exist_ok=True)

    print(f"Display mode: {display}")
    print(f"Radar command UART: {radar_port}")
    print(f"Raw output: {raw_output}")

    capture_process: Optional[subprocess.Popen] = None
    startup_process: Optional[subprocess.Popen] = None
    try:
        capture_process = start_process(
            "live capture",
            build_capture_command(args, display, raw_output),
        )
        time.sleep(1.0)
        if capture_process.poll() is not None:
            print(
                f"live capture exited early with code {capture_process.returncode}",
                file=sys.stderr,
            )
            return capture_process.returncode or 1

        startup_process = start_process(
            "startup",
            build_startup_command(args, radar_port),
        )

        while True:
            if startup_process.poll() is not None:
                print(f"startup exited with code {startup_process.returncode}")
                return startup_process.returncode or 0
            if capture_process.poll() is not None:
                print(f"live capture exited with code {capture_process.returncode}")
                return capture_process.returncode or 0
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("\nCtrl+C received. Stopping radar and capture.")
        return 0
    finally:
        stop_process("startup", startup_process)
        stop_process("live capture", capture_process)
        print(f"Raw data file: {raw_output}")
        print(f"Raw metadata file: {raw_output}.json")


if __name__ == "__main__":
    raise SystemExit(main())
