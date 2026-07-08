import argparse
import json
import socket
import sys
import time
from dataclasses import dataclass
from enum import Enum, IntEnum
from pathlib import Path
from typing import Any, Callable, Iterable, Optional, Protocol

try:
    from .livedatacapture import (
        DEFAULT_CONFIG_PATH,
        DEFAULT_SETUP_PATH,
        UDP_IP,
        UDP_PORT,
        CaptureSetupConfig,
        RadarCaptureConfig,
    )
except ImportError:
    from livedatacapture import (
        DEFAULT_CONFIG_PATH,
        DEFAULT_SETUP_PATH,
        UDP_IP,
        UDP_PORT,
        CaptureSetupConfig,
        RadarCaptureConfig,
    )


DCA1000_IP = "192.168.33.180"
DCA1000_CONFIG_PORT = 4096
DCA1000_COMMAND_HEADER = 0xA55A
DCA1000_COMMAND_FOOTER = 0xEEAA
DCA1000_MAX_RESPONSE_BYTES = 2048
DCA1000_MAX_BYTES_PER_PACKET = 1470
DCA1000_FPGA_CONFIG_DEFAULT_TIMER = 30
DCA1000_FPGA_CLK_CONVERSION_FACTOR = 1000
DCA1000_FPGA_CLK_PERIOD_NS = 8
DEFAULT_SDK_PROFILE_PATH = Path(__file__).with_name("profile.cfg")
EmitFunc = Callable[[str], None]


class StartupError(RuntimeError):
    """Raised for expected startup failures that should not print a traceback."""


class StartupState(str, Enum):
    IDLE = "idle"
    CONFIGS_LOADED = "configs_loaded"
    PREFLIGHT_PASSED = "preflight_passed"
    DCA1000_READY = "dca1000_ready"
    RADAR_READY = "radar_ready"
    RECEIVER_READY = "receiver_ready"
    DCA1000_ARMED = "dca1000_armed"
    RADAR_STREAMING = "radar_streaming"
    STOPPING = "stopping"
    STOPPED = "stopped"


class DCA1000Command(IntEnum):
    RESET_FPGA = 0x01
    RESET_AR_DEVICE = 0x02
    CONFIG_FPGA_GEN = 0x03
    RECORD_START = 0x05
    RECORD_STOP = 0x06
    SYSTEM_CONNECT = 0x09
    CONFIG_RECORD = 0x0B
    CONFIG_PACKET_DATA = 0x0B
    READ_FPGA_VERSION = 0x0E


class DCA1000Status(IntEnum):
    SUCCESS = 0


@dataclass(frozen=True)
class RuntimeOptions:
    config_path: Path
    setup_path: Path
    host_ip: str = UDP_IP
    data_port: int = UDP_PORT
    dca_ip: str = DCA1000_IP
    dca_config_port: int = DCA1000_CONFIG_PORT
    dca_backend: str = "dry-run"
    radar_backend: str = "dry-run"
    capture_backend: str = "dry-run"
    dca_timeout_s: float = 1.0
    dca_retries: int = 2
    sdk_profile_path: Path = DEFAULT_SDK_PROFILE_PATH
    radar_port: Optional[str] = None
    radar_baud: Optional[int] = None
    radar_command_timeout_s: float = 2.0
    radar_command_delay_s: float = 0.03
    radar_line_ending: str = "crlf"
    load_firmware: bool = False
    skip_socket_preflight: bool = False
    readiness_delay_s: float = 0.25


@dataclass(frozen=True)
class RadarDeviceSetup:
    device: Optional[str]
    operating_frequency_ghz: Optional[float]
    control_port: Optional[str]
    baud_rate: Optional[int]
    radar_ss_firmware: Optional[Path]
    master_ss_firmware: Optional[Path]


@dataclass(frozen=True)
class DCA1000Setup:
    capture_hardware: Optional[str]
    data_logging_mode: Optional[str]
    data_transfer_mode: Optional[str]
    data_capture_mode: Optional[str]
    packet_sequence_enable: bool
    packet_delay_us: Optional[int]


@dataclass(frozen=True)
class StartupConfig:
    radar_config: RadarCaptureConfig
    setup_config: CaptureSetupConfig
    radar_device: RadarDeviceSetup
    dca1000: DCA1000Setup
    mmwave_json: dict[str, Any]
    setup_json: dict[str, Any]


class DCA1000Control(Protocol):
    def configure(self, config: StartupConfig) -> None:
        ...

    def arm(self, config: StartupConfig) -> None:
        ...

    def stop(self) -> None:
        ...


class RadarControl(Protocol):
    def configure(self, config: StartupConfig) -> None:
        ...

    def start_sensor(self) -> None:
        ...

    def stop_sensor(self) -> None:
        ...


class CapturePipeline(Protocol):
    def start_receiver(self, config: StartupConfig) -> None:
        ...

    def close(self) -> None:
        ...


class ConfigLoader:
    def __init__(self, options: RuntimeOptions) -> None:
        self.options = options

    def load(self) -> StartupConfig:
        config_path = _resolve_existing_path(self.options.config_path)
        setup_path = _resolve_existing_path(self.options.setup_path)

        radar_config = RadarCaptureConfig.from_file(config_path)
        setup_config = CaptureSetupConfig.from_file(setup_path)
        mmwave_json = _read_json_object(config_path) if config_path.suffix.lower() == ".json" else {}
        setup_json = _read_json_object(setup_path)

        return StartupConfig(
            radar_config=radar_config,
            setup_config=setup_config,
            radar_device=_parse_radar_device_setup(setup_json),
            dca1000=_parse_dca1000_setup(setup_json, setup_config),
            mmwave_json=mmwave_json,
            setup_json=setup_json,
        )


class PreflightValidator:
    def __init__(self, options: RuntimeOptions) -> None:
        self.options = options

    def validate(self, config: StartupConfig) -> None:
        errors: list[str] = []

        self._validate_radar_dimensions(config, errors)
        self._validate_dca1000_settings(config, errors)
        self._validate_radar_control_settings(config, errors)
        self._validate_sdk_profile(errors)
        self._validate_firmware_paths(config, errors)
        self._validate_socket_bind(errors)

        if errors:
            formatted_errors = "\n".join(f"- {error}" for error in errors)
            raise StartupError(f"Startup preflight failed:\n{formatted_errors}")

    def _validate_radar_dimensions(
        self, config: StartupConfig, errors: list[str]
    ) -> None:
        radar = config.radar_config
        if radar.num_adc_samples <= 0:
            errors.append("num_adc_samples must be positive")
        if radar.num_rx_channels <= 0:
            errors.append("num_rx_channels must be positive")
        if radar.num_chirps_per_frame <= 0:
            errors.append("num_chirps_per_frame must be positive")
        if radar.bytes_per_frame <= 0:
            errors.append("bytes_per_frame must be positive")
        if radar.lvds_lanes != 2:
            errors.append(
                "unsupported LVDS lane mode: "
                f"{radar.lvds_lanes}; current reshape supports 2 lanes"
            )

    def _validate_dca1000_settings(
        self, config: StartupConfig, errors: list[str]
    ) -> None:
        dca = config.dca1000
        if dca.capture_hardware and dca.capture_hardware.upper() != "DCA1000":
            errors.append(f"unsupported capture hardware: {dca.capture_hardware}")
        if dca.packet_delay_us is not None and dca.packet_delay_us < 0:
            errors.append("packetDelay_us must be non-negative")
        if not (1 <= self.options.dca_config_port <= 65535):
            errors.append(f"invalid DCA1000 config port: {self.options.dca_config_port}")
        if not (1 <= self.options.data_port <= 65535):
            errors.append(f"invalid DCA1000 data port: {self.options.data_port}")

    def _validate_radar_control_settings(
        self, config: StartupConfig, errors: list[str]
    ) -> None:
        radar_device = config.radar_device
        control_port = self.options.radar_port or radar_device.control_port
        baud_rate = self.options.radar_baud or radar_device.baud_rate
        if not control_port:
            errors.append("missing radar control port in setup.json")
        if baud_rate is None or baud_rate <= 0:
            errors.append("missing or invalid radar RS232 baud rate in setup.json")

    def _validate_sdk_profile(self, errors: list[str]) -> None:
        if self.options.radar_backend != "direct-serial":
            return

        try:
            profile_path = _resolve_existing_path(self.options.sdk_profile_path)
        except StartupError as exc:
            errors.append(str(exc))
            return

        commands = list(_iter_sdk_cli_profile_commands(profile_path))
        if not commands:
            errors.append(f"SDK CLI profile has no commands: {profile_path}")
        if not any(command.split(maxsplit=1)[0] == "sensorStart" for command in commands):
            errors.append(f"SDK CLI profile is missing sensorStart: {profile_path}")

    def _validate_firmware_paths(
        self, config: StartupConfig, errors: list[str]
    ) -> None:
        if not self.options.load_firmware:
            return

        radar_device = config.radar_device
        for label, firmware_path in (
            ("radar SS firmware", radar_device.radar_ss_firmware),
            ("master SS firmware", radar_device.master_ss_firmware),
        ):
            if firmware_path is None:
                errors.append(f"missing {label} path in setup.json")
            elif not firmware_path.exists():
                errors.append(f"{label} not found: {firmware_path}")

    def _validate_socket_bind(self, errors: list[str]) -> None:
        if self.options.skip_socket_preflight:
            return

        try:
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
                sock.bind((self.options.host_ip, self.options.data_port))
        except OSError as exc:
            errors.append(
                "could not bind DCA1000 data socket during preflight: "
                f"{self.options.host_ip}:{self.options.data_port}: {exc}"
            )


class HealthMonitor:
    def __init__(self, emit: EmitFunc = print) -> None:
        self.emit = emit

    def transition(self, previous: StartupState, current: StartupState) -> None:
        self.emit(f"startup state: {previous.value} -> {current.value}")

    def status(self, message: str) -> None:
        self.emit(message)


class DCA1000UdpClient:
    def __init__(
        self,
        *,
        local_port: int,
        ip: str,
        port: int,
        timeout_s: float,
        retries: int,
        emit: EmitFunc = print,
    ) -> None:
        self.address = (ip, port)
        self.timeout_s = timeout_s
        self.retries = max(retries, 0)
        self.emit = emit
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.settimeout(timeout_s)
        self.sock.bind(("0.0.0.0", local_port))

    def send_command(
        self,
        command: DCA1000Command,
        payload: bytes = b"",
    ) -> bytes:
        packet = _build_dca1000_command_packet(command, payload)
        last_error: Optional[Exception] = None

        for attempt in range(self.retries + 1):
            try:
                self.sock.sendto(packet, self.address)
                response, _addr = self.sock.recvfrom(DCA1000_MAX_RESPONSE_BYTES)
                _validate_dca1000_response(response, command)
                return response
            except (OSError, TimeoutError, StartupError) as exc:
                last_error = exc
                if attempt < self.retries:
                    self.emit(
                        "DCA1000 command retry: "
                        f"{command.name}, attempt={attempt + 2}/{self.retries + 1}, "
                        f"reason={exc}"
                    )

        raise StartupError(
            f"DCA1000 command failed after {self.retries + 1} attempt(s): "
            f"{command.name}: {last_error}"
        )

    def close(self) -> None:
        self.sock.close()


class DirectUdpDCA1000Control:
    def __init__(self, options: RuntimeOptions, emit: EmitFunc = print) -> None:
        self.options = options
        self.emit = emit
        self.client: Optional[DCA1000UdpClient] = None
        self.armed = False

    def configure(self, config: StartupConfig) -> None:
        self.client = DCA1000UdpClient(
            local_port=self.options.dca_config_port,
            ip=self.options.dca_ip,
            port=self.options.dca_config_port,
            timeout_s=self.options.dca_timeout_s,
            retries=self.options.dca_retries,
            emit=self.emit,
        )

        self.emit(
            "DCA1000 UDP configure: "
            f"{self.options.dca_ip}:{self.options.dca_config_port}"
        )
        self._send(DCA1000Command.SYSTEM_CONNECT)
        self._send(DCA1000Command.RESET_FPGA)
        self._send(
            DCA1000Command.CONFIG_FPGA_GEN,
            _build_dca1000_fpga_payload(config),
        )
        self._send(
            DCA1000Command.CONFIG_RECORD,
            _build_dca1000_packet_payload(config),
        )

    def arm(self, config: StartupConfig) -> None:
        self._send(DCA1000Command.RECORD_START)
        self.armed = True
        self.emit("DCA1000 UDP record start acknowledged")

    def stop(self) -> None:
        try:
            if self.client is not None and self.armed:
                self._send(DCA1000Command.RECORD_STOP)
                self.emit("DCA1000 UDP record stop acknowledged")
        finally:
            self.armed = False
            if self.client is not None:
                self.client.close()
                self.client = None

    def _send(
        self,
        command: DCA1000Command,
        payload: bytes = b"",
    ) -> bytes:
        if self.client is None:
            raise StartupError("DCA1000 UDP client is not open")

        self.emit(
            "DCA1000 UDP command: "
            f"{command.name}, payload_len={len(payload)}"
        )
        return self.client.send_command(command, payload)


class DryRunDCA1000Control:
    def __init__(self, options: RuntimeOptions, emit: EmitFunc = print) -> None:
        self.options = options
        self.emit = emit
        self.armed = False

    def configure(self, config: StartupConfig) -> None:
        self.emit(
            "dry-run DCA1000 configure: "
            f"ip={self.options.dca_ip}:{self.options.dca_config_port}, "
            f"packet_delay_us={config.dca1000.packet_delay_us}, "
            f"packet_sequence_enable={config.dca1000.packet_sequence_enable}"
        )

    def arm(self, config: StartupConfig) -> None:
        self.armed = True
        self.emit(
            "dry-run DCA1000 arm: "
            f"mode={config.dca1000.data_capture_mode}, "
            f"transfer={config.dca1000.data_transfer_mode}"
        )

    def stop(self) -> None:
        if self.armed:
            self.emit("dry-run DCA1000 stop")
        self.armed = False


class DryRunRadarControl:
    def __init__(self, options: RuntimeOptions, emit: EmitFunc = print) -> None:
        self.options = options
        self.emit = emit
        self.started = False

    def configure(self, config: StartupConfig) -> None:
        radar_device = config.radar_device
        radar = config.radar_config
        self.emit(
            "dry-run radar configure: "
            f"device={radar_device.device}, "
            f"control={radar_device.control_port}@{radar_device.baud_rate}, "
            f"adc_samples={radar.num_adc_samples}, "
            f"rx={radar.num_rx_channels}, "
            f"chirps_per_frame={radar.num_chirps_per_frame}, "
            f"bytes_per_frame={radar.bytes_per_frame}"
        )
        if self.options.load_firmware:
            self.emit(
                "dry-run radar firmware load: "
                f"radar_ss={radar_device.radar_ss_firmware}, "
                f"master_ss={radar_device.master_ss_firmware}"
            )

    def start_sensor(self) -> None:
        self.started = True
        self.emit("dry-run radar sensor start")

    def stop_sensor(self) -> None:
        if self.started:
            self.emit("dry-run radar sensor stop")
        self.started = False


class SdkCliRadarControl:
    def __init__(self, options: RuntimeOptions, emit: EmitFunc = print) -> None:
        self.options = options
        self.emit = emit
        self.serial = None
        self.started = False
        self.start_command = "sensorStart"

    def configure(self, config: StartupConfig) -> None:
        try:
            import serial
        except ImportError as exc:
            raise StartupError(
                "pyserial is required for --radar-backend direct-serial"
            ) from exc

        profile_path = _resolve_existing_path(self.options.sdk_profile_path)
        commands = list(_iter_sdk_cli_profile_commands(profile_path))
        configure_commands: list[str] = []
        for command in commands:
            name = command.split(maxsplit=1)[0]
            if name == "sensorStart":
                self.start_command = command
                continue
            configure_commands.append(command)

        port = _radar_control_port(config, self.options)
        baud = _radar_baud_rate(config, self.options)
        self.emit(
            "SDK CLI radar open: "
            f"profile={profile_path}, port={port}, baud={baud}"
        )
        try:
            self.serial = serial.Serial(
                port=port,
                baudrate=baud,
                timeout=self.options.radar_command_timeout_s,
                write_timeout=self.options.radar_command_timeout_s,
            )
            time.sleep(0.2)
            self._drain_input()

            for command in configure_commands:
                self._send_command(command)
        except Exception as exc:
            if self.serial is not None:
                self.serial.close()
                self.serial = None
            raise StartupError(f"SDK CLI radar configuration failed: {exc}") from exc

        self.emit(
            "SDK CLI radar configured: "
            f"commands_sent={len(configure_commands)}, "
            "sensorStart deferred"
        )

    def start_sensor(self) -> None:
        self._send_command(self.start_command)
        self.started = True
        self.emit("SDK CLI radar sensor start acknowledged")

    def stop_sensor(self) -> None:
        if self.serial is None:
            return

        try:
            if self.started:
                self._send_command("sensorStop", allow_error=True)
                self.emit("SDK CLI radar sensor stop sent")
        finally:
            self.started = False
            self.serial.close()
            self.serial = None

    def _send_command(self, command: str, *, allow_error: bool = False) -> str:
        if self.serial is None:
            raise StartupError("SDK CLI serial port is not open")

        self.emit(f"SDK CLI command: {command}")
        line_ending = _serial_line_ending(self.options)
        self.serial.write(f"{command}{line_ending}".encode("ascii"))
        self.serial.flush()
        response = self._read_command_response()
        normalized_response = response.lower()
        if "error" in normalized_response and not allow_error:
            raise StartupError(
                f"SDK CLI command failed: {command}\n{response.strip()}"
            )
        if self.options.radar_command_delay_s > 0:
            time.sleep(self.options.radar_command_delay_s)
        return response

    def _read_command_response(self) -> str:
        if self.serial is None:
            return ""

        deadline = time.monotonic() + self.options.radar_command_timeout_s
        chunks: list[bytes] = []
        while time.monotonic() < deadline:
            waiting = getattr(self.serial, "in_waiting", 0)
            line = self.serial.readline() if waiting else self.serial.readline()
            if line:
                chunks.append(line)
                text = b"".join(chunks).decode("utf-8", errors="replace")
                if (
                    "Done" in text
                    or "Error" in text
                    or "mmwDemo:/>" in text
                    or "Ignored" in text
                    or "Skipped" in text
                ):
                    return text
            else:
                time.sleep(0.01)

        response = b"".join(chunks).decode("utf-8", errors="replace")
        raise StartupError(
            "SDK CLI command response timed out. "
            f"Partial response: {response.strip()}"
        )

    def _drain_input(self) -> None:
        if self.serial is None:
            return
        time.sleep(0.05)
        waiting = getattr(self.serial, "in_waiting", 0)
        if waiting:
            self.serial.read(waiting)


class DryRunCapturePipeline:
    def __init__(self, options: RuntimeOptions, emit: EmitFunc = print) -> None:
        self.options = options
        self.emit = emit
        self.running = False

    def start_receiver(self, config: StartupConfig) -> None:
        self.running = True
        self.emit(
            "dry-run capture receiver start: "
            f"{self.options.host_ip}:{self.options.data_port}, "
            f"bytes_per_frame={config.radar_config.bytes_per_frame}"
        )

    def close(self) -> None:
        if self.running:
            self.emit("dry-run capture receiver close")
        self.running = False


class UnsupportedDCA1000Control:
    def __init__(self, backend: str) -> None:
        self.backend = backend

    def configure(self, config: StartupConfig) -> None:
        raise StartupError(
            f"DCA1000 backend '{self.backend}' is not implemented yet. "
            "Use --dca-backend dry-run for preflight/state-machine validation."
        )

    def arm(self, config: StartupConfig) -> None:
        raise StartupError(f"DCA1000 backend '{self.backend}' cannot arm capture")

    def stop(self) -> None:
        return


class UnsupportedRadarControl:
    def __init__(self, backend: str) -> None:
        self.backend = backend

    def configure(self, config: StartupConfig) -> None:
        raise StartupError(
            f"radar backend '{self.backend}' is not implemented yet. "
            "Use --radar-backend dry-run for preflight/state-machine validation."
        )

    def start_sensor(self) -> None:
        raise StartupError(f"radar backend '{self.backend}' cannot start sensor")

    def stop_sensor(self) -> None:
        return


class StartupOrchestrator:
    def __init__(
        self,
        options: RuntimeOptions,
        *,
        emit: EmitFunc = print,
        dca1000_control: Optional[DCA1000Control] = None,
        radar_control: Optional[RadarControl] = None,
        capture_pipeline: Optional[CapturePipeline] = None,
    ) -> None:
        self.options = options
        self.emit = emit
        self.monitor = HealthMonitor(emit)
        self.config_loader = ConfigLoader(options)
        self.preflight_validator = PreflightValidator(options)
        self.dca1000_control = dca1000_control or _make_dca1000_control(options, emit)
        self.radar_control = radar_control or _make_radar_control(options, emit)
        self.capture_pipeline = capture_pipeline or _make_capture_pipeline(
            options,
            emit,
        )
        self.state = StartupState.IDLE
        self.config: Optional[StartupConfig] = None

    def load_configs(self) -> StartupConfig:
        self.config = self.config_loader.load()
        self._transition(StartupState.CONFIGS_LOADED)
        self.monitor.status(f"Loaded radar config: {self.config.radar_config}")
        self.monitor.status(f"Loaded capture setup: {self.config.setup_config}")
        return self.config

    def run_preflight(self) -> StartupConfig:
        config = self.config or self.load_configs()
        self.preflight_validator.validate(config)
        self._transition(StartupState.PREFLIGHT_PASSED)
        return config

    def configure_dca1000(self) -> StartupConfig:
        config = self.config or self.run_preflight()
        self.dca1000_control.configure(config)
        self._transition(StartupState.DCA1000_READY)
        return config

    def configure_radar(self) -> StartupConfig:
        config = self.config or self.configure_dca1000()
        self.radar_control.configure(config)
        self._transition(StartupState.RADAR_READY)
        return config

    def start_receiver(self) -> StartupConfig:
        config = self.config or self.configure_radar()
        self.capture_pipeline.start_receiver(config)
        self._transition(StartupState.RECEIVER_READY)
        return config

    def arm_capture(self) -> StartupConfig:
        config = self.config or self.start_receiver()
        self.dca1000_control.arm(config)
        self._transition(StartupState.DCA1000_ARMED)
        if self.options.readiness_delay_s > 0:
            time.sleep(self.options.readiness_delay_s)
        return config

    def start_radar(self) -> StartupConfig:
        config = self.config or self.arm_capture()
        self.radar_control.start_sensor()
        self._transition(StartupState.RADAR_STREAMING)
        return config

    def startup(self) -> StartupConfig:
        self.load_configs()
        self.run_preflight()
        self.configure_dca1000()
        self.configure_radar()
        self.start_receiver()
        self.arm_capture()
        return self.start_radar()

    def stop(self) -> None:
        if self.state in {StartupState.IDLE, StartupState.STOPPED}:
            return

        previous = self.state
        self.state = StartupState.STOPPING
        self.monitor.transition(previous, self.state)

        stop_errors: list[str] = []
        for label, stop_func in (
            ("radar sensor", self.radar_control.stop_sensor),
            ("DCA1000", self.dca1000_control.stop),
            ("capture pipeline", self.capture_pipeline.close),
        ):
            try:
                stop_func()
            except Exception as exc:
                stop_errors.append(f"{label}: {exc!r}")

        self._transition(StartupState.STOPPED)
        if stop_errors:
            formatted_errors = "\n".join(f"- {error}" for error in stop_errors)
            raise StartupError(f"Startup cleanup had errors:\n{formatted_errors}")

    def _transition(self, next_state: StartupState) -> None:
        previous = self.state
        self.state = next_state
        self.monitor.transition(previous, next_state)


def _make_dca1000_control(
    options: RuntimeOptions,
    emit: EmitFunc,
) -> DCA1000Control:
    if options.dca_backend == "dry-run":
        return DryRunDCA1000Control(options, emit)
    if options.dca_backend == "direct-udp":
        return DirectUdpDCA1000Control(options, emit)
    return UnsupportedDCA1000Control(options.dca_backend)


def _make_radar_control(options: RuntimeOptions, emit: EmitFunc) -> RadarControl:
    if options.radar_backend == "dry-run":
        return DryRunRadarControl(options, emit)
    if options.radar_backend == "direct-serial":
        return SdkCliRadarControl(options, emit)
    return UnsupportedRadarControl(options.radar_backend)


def _make_capture_pipeline(
    options: RuntimeOptions,
    emit: EmitFunc,
) -> CapturePipeline:
    if options.capture_backend == "dry-run":
        return DryRunCapturePipeline(options, emit)
    raise StartupError(
        f"capture backend '{options.capture_backend}' is not implemented yet. "
        "Use --capture-backend dry-run for startup validation."
    )


def _parse_radar_device_setup(setup_json: dict[str, Any]) -> RadarDeviceSetup:
    device_config = _as_mapping(setup_json.get("mmWaveDeviceConfig")) or {}
    return RadarDeviceSetup(
        device=_optional_string(setup_json, "mmWaveDevice"),
        operating_frequency_ghz=_optional_float(setup_json, "operatingFreq"),
        control_port=_optional_string(device_config, "RS232COMPort"),
        baud_rate=_optional_int(device_config, "RS232BaudRate"),
        radar_ss_firmware=_optional_path(device_config, "radarSSFirmware"),
        master_ss_firmware=_optional_path(device_config, "masterSSFirmware"),
    )


def _parse_dca1000_setup(
    setup_json: dict[str, Any],
    setup_config: CaptureSetupConfig,
) -> DCA1000Setup:
    dca_config = _as_mapping(setup_json.get("DCA1000Config")) or {}
    return DCA1000Setup(
        capture_hardware=setup_config.capture_hardware,
        data_logging_mode=_optional_string(dca_config, "dataLoggingMode"),
        data_transfer_mode=_optional_string(dca_config, "dataTransferMode"),
        data_capture_mode=_optional_string(dca_config, "dataCaptureMode"),
        packet_sequence_enable=setup_config.packet_sequence_enable,
        packet_delay_us=setup_config.packet_delay_us,
    )


def _iter_sdk_cli_profile_commands(profile_path: Path) -> Iterable[str]:
    for raw_line in profile_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.split("%", 1)[0].split("#", 1)[0].strip()
        if line:
            yield line


def _radar_control_port(config: StartupConfig, options: RuntimeOptions) -> str:
    port = options.radar_port or config.radar_device.control_port
    if not port:
        raise StartupError("missing radar control port")
    return port


def _radar_baud_rate(config: StartupConfig, options: RuntimeOptions) -> int:
    baud = options.radar_baud or config.radar_device.baud_rate
    if baud is None or baud <= 0:
        raise StartupError("missing or invalid radar baud rate")
    return baud


def _serial_line_ending(options: RuntimeOptions) -> str:
    if options.radar_line_ending == "crlf":
        return "\r\n"
    if options.radar_line_ending == "cr":
        return "\r"
    return "\n"


def _build_dca1000_command_packet(
    command: DCA1000Command,
    payload: bytes = b"",
) -> bytes:
    return b"".join(
        (
            DCA1000_COMMAND_HEADER.to_bytes(2, byteorder="little"),
            int(command).to_bytes(2, byteorder="little"),
            len(payload).to_bytes(2, byteorder="little"),
            payload,
            DCA1000_COMMAND_FOOTER.to_bytes(2, byteorder="little"),
        )
    )


def _validate_dca1000_response(
    response: bytes,
    command: DCA1000Command,
) -> None:
    if len(response) < 8:
        raise StartupError(
            f"DCA1000 response too short for {command.name}: {response.hex()}"
        )

    header = int.from_bytes(response[0:2], byteorder="little")
    response_command = int.from_bytes(response[2:4], byteorder="little")
    status = int.from_bytes(response[4:6], byteorder="little")
    footer = int.from_bytes(response[-2:], byteorder="little")

    if header != DCA1000_COMMAND_HEADER:
        raise StartupError(
            f"DCA1000 response has bad header for {command.name}: 0x{header:04X}"
        )
    if footer != DCA1000_COMMAND_FOOTER:
        raise StartupError(
            f"DCA1000 response has bad footer for {command.name}: 0x{footer:04X}"
        )
    if response_command != int(command):
        raise StartupError(
            "DCA1000 response command mismatch: "
            f"sent={command.name}/0x{int(command):04X}, "
            f"got=0x{response_command:04X}"
        )
    if status != DCA1000Status.SUCCESS:
        raise StartupError(
            f"DCA1000 command {command.name} returned status 0x{status:04X}"
        )


def _build_dca1000_fpga_payload(config: StartupConfig) -> bytes:
    override = _dca1000_payload_override(config, DCA1000Command.CONFIG_FPGA_GEN)
    if override is not None:
        return override

    dca = config.dca1000
    dca_json = _as_mapping(config.setup_json.get("DCA1000Config")) or {}

    data_logging_mode = _enum_value(
        _optional_value(dca_json, "dataLoggingMode") or dca.data_logging_mode,
        {"raw": 1, "multi": 2},
        default=1,
    )
    lvds_mode = _optional_int(dca_json, "lvdsMode")
    if lvds_mode is None:
        lvds_mode = 2 if config.radar_config.lvds_lanes == 2 else 1

    data_transfer_mode = _enum_value(
        _optional_value(dca_json, "dataTransferMode") or dca.data_transfer_mode,
        {"lvdscapture": 1, "playback": 2},
        default=1,
    )
    data_capture_mode = _enum_value(
        _optional_value(dca_json, "dataCaptureMode") or dca.data_capture_mode,
        {"ethernetstream": 2, "sdcard": 1},
        default=2,
    )
    data_format_mode = _optional_int(dca_json, "dataFormatMode")
    if data_format_mode is None:
        data_format_mode = _adc_data_format_mode(config)

    timer = _optional_int(dca_json, "timer", "fpgaConfigTimer")
    if timer is None:
        timer = DCA1000_FPGA_CONFIG_DEFAULT_TIMER

    return bytes(
        (
            data_logging_mode,
            lvds_mode,
            data_transfer_mode,
            data_capture_mode,
            data_format_mode,
            timer,
        )
    )


def _build_dca1000_packet_payload(config: StartupConfig) -> bytes:
    override = _dca1000_payload_override(config, DCA1000Command.CONFIG_RECORD)
    if override is not None:
        return override

    dca = config.dca1000
    dca_json = _as_mapping(config.setup_json.get("DCA1000Config")) or {}
    packet_delay_us = dca.packet_delay_us or 0
    max_packet_size = _optional_int(dca_json, "maxPacketSize", "packetDataSize")
    if max_packet_size is None:
        max_packet_size = DCA1000_MAX_BYTES_PER_PACKET
    delay_cycles = (
        int(packet_delay_us)
        * DCA1000_FPGA_CLK_CONVERSION_FACTOR
        // DCA1000_FPGA_CLK_PERIOD_NS
    )

    return b"".join(
        (
            int(max_packet_size).to_bytes(2, byteorder="little"),
            int(delay_cycles).to_bytes(2, byteorder="little"),
            (0).to_bytes(2, byteorder="little"),
        )
    )


def _dca1000_payload_override(
    config: StartupConfig,
    command: DCA1000Command,
) -> Optional[bytes]:
    direct_udp_config = _as_mapping(config.setup_json.get("directUdpDCA1000")) or {}
    payloads = _as_mapping(direct_udp_config.get("payloads")) or {}
    value = (
        payloads.get(command.name)
        or payloads.get(command.name.lower())
        or (
            payloads.get("CONFIG_PACKET_DATA")
            if command is DCA1000Command.CONFIG_RECORD
            else None
        )
        or payloads.get(f"0x{int(command):04X}")
        or payloads.get(str(int(command)))
    )
    if value is None:
        return None
    if not isinstance(value, str):
        raise StartupError(
            f"directUdpDCA1000 payload override for {command.name} must be hex text"
        )
    return bytes.fromhex(value.replace(" ", "").replace("_", ""))


def _adc_data_format_mode(config: StartupConfig) -> int:
    devices = config.mmwave_json.get("mmWaveDevices")
    if isinstance(devices, list) and devices:
        device = _as_mapping(devices[0]) or {}
        rf_config = _as_mapping(device.get("rfConfig")) or {}
        adc_out_config = _as_mapping(rf_config.get("rlAdcOutCfg_t")) or {}
        fmt = _as_mapping(adc_out_config.get("fmt")) or {}
        adc_bits = _optional_int(fmt, "b2AdcBits")
        if adc_bits is not None:
            # mmWaveLink encodes 0=12-bit, 1=14-bit, 2=16-bit. DCA1000 uses
            # 1=12-bit, 2=14-bit, 3=16-bit for CONFIG_FPGA.
            return adc_bits + 1

    return 3


def _resolve_existing_path(path: Path) -> Path:
    if path.exists():
        return path

    if not path.is_absolute():
        script_relative_path = Path(__file__).resolve().parent / path.name
        if script_relative_path.exists():
            return script_relative_path

    raise StartupError(f"required startup file not found: {path}")


def _read_json_object(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise StartupError(f"expected JSON object in {path}")
    return data


def _as_mapping(value: Any) -> Optional[dict[str, Any]]:
    return value if isinstance(value, dict) else None


def _optional_value(data: dict[str, Any], *names: str) -> Any:
    normalized_names = {_normalize_key(name) for name in names}
    for key, value in data.items():
        if _normalize_key(key) in normalized_names:
            return value
    return None


def _optional_string(data: dict[str, Any], *names: str) -> Optional[str]:
    value = _optional_value(data, *names)
    return None if value is None else str(value)


def _optional_int(data: dict[str, Any], *names: str) -> Optional[int]:
    value = _optional_value(data, *names)
    if value is None:
        return None
    if isinstance(value, int):
        return value
    return int(str(value).strip(), 0)


def _optional_float(data: dict[str, Any], *names: str) -> Optional[float]:
    value = _optional_value(data, *names)
    return None if value is None else float(value)


def _optional_path(data: dict[str, Any], *names: str) -> Optional[Path]:
    value = _optional_string(data, *names)
    return None if value is None or not value else Path(value)


def _enum_value(value: Any, mapping: dict[str, int], *, default: int) -> int:
    if value is None:
        return default
    if isinstance(value, int):
        return value

    text = str(value).strip()
    if not text:
        return default
    try:
        return int(text, 0)
    except ValueError:
        normalized = _normalize_key(text)
        if normalized not in mapping:
            raise StartupError(f"unsupported DCA1000 enum value: {value}")
        return mapping[normalized]


def _normalize_key(value: str) -> str:
    return "".join(character for character in value.lower() if character.isalnum())


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validate and orchestrate radar/DCA1000 startup without mmWave Studio "
            "as the runtime controller."
        )
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--setup", type=Path, default=DEFAULT_SETUP_PATH)
    parser.add_argument(
        "--sdk-profile",
        type=Path,
        default=DEFAULT_SDK_PROFILE_PATH,
        help="SDK CLI .cfg command profile for --radar-backend direct-serial.",
    )
    parser.add_argument("--host-ip", default=UDP_IP)
    parser.add_argument("--data-port", type=int, default=UDP_PORT)
    parser.add_argument(
        "--radar-port",
        help="Override radar command UART port from setup.json, e.g. COM10.",
    )
    parser.add_argument(
        "--radar-baud",
        type=int,
        help="Override radar command UART baud rate from setup.json.",
    )
    parser.add_argument(
        "--radar-command-timeout",
        type=float,
        default=2.0,
        help="Seconds to wait for each SDK CLI command response.",
    )
    parser.add_argument(
        "--radar-command-delay",
        type=float,
        default=0.03,
        help="Seconds to wait between SDK CLI commands.",
    )
    parser.add_argument(
        "--radar-line-ending",
        choices=("crlf", "lf", "cr"),
        default="crlf",
        help="Line ending to use when sending SDK CLI commands.",
    )
    parser.add_argument("--dca-ip", default=DCA1000_IP)
    parser.add_argument("--dca-config-port", type=int, default=DCA1000_CONFIG_PORT)
    parser.add_argument(
        "--dca-backend",
        choices=("dry-run", "direct-udp", "cli"),
        default="dry-run",
    )
    parser.add_argument(
        "--radar-backend",
        choices=("dry-run", "direct-serial", "cli"),
        default="dry-run",
    )
    parser.add_argument(
        "--capture-backend",
        choices=("dry-run",),
        default="dry-run",
    )
    parser.add_argument(
        "--dca-timeout",
        type=float,
        default=1.0,
        help="Seconds to wait for each DCA1000 UDP command response.",
    )
    parser.add_argument(
        "--dca-retries",
        type=int,
        default=2,
        help="Number of retries for each DCA1000 UDP command.",
    )
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="Load configs and run preflight checks without configuring hardware.",
    )
    parser.add_argument(
        "--load-firmware",
        action="store_true",
        help="Require MSS/BSS firmware paths and include firmware loading in startup.",
    )
    parser.add_argument(
        "--skip-socket-preflight",
        action="store_true",
        help="Skip the host-ip/data-port bind check.",
    )
    parser.add_argument(
        "--readiness-delay",
        type=float,
        default=0.25,
        help="Seconds to wait after arming DCA1000 before starting radar.",
    )
    return parser.parse_args(argv)


def options_from_args(args: argparse.Namespace) -> RuntimeOptions:
    config_path = args.config
    if args.radar_backend == "direct-serial" and config_path == DEFAULT_CONFIG_PATH:
        config_path = args.sdk_profile

    return RuntimeOptions(
        config_path=config_path,
        setup_path=args.setup,
        host_ip=args.host_ip,
        data_port=args.data_port,
        dca_ip=args.dca_ip,
        dca_config_port=args.dca_config_port,
        dca_backend=args.dca_backend,
        radar_backend=args.radar_backend,
        capture_backend=args.capture_backend,
        dca_timeout_s=max(args.dca_timeout, 0.1),
        dca_retries=max(args.dca_retries, 0),
        sdk_profile_path=args.sdk_profile,
        radar_port=args.radar_port,
        radar_baud=args.radar_baud,
        radar_command_timeout_s=max(args.radar_command_timeout, 0.1),
        radar_command_delay_s=max(args.radar_command_delay, 0.0),
        radar_line_ending=args.radar_line_ending,
        load_firmware=args.load_firmware,
        skip_socket_preflight=args.skip_socket_preflight,
        readiness_delay_s=max(args.readiness_delay, 0.0),
    )


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    orchestrator = StartupOrchestrator(options_from_args(args))

    try:
        if args.preflight_only:
            orchestrator.run_preflight()
            print("Startup preflight passed.")
            return 0

        orchestrator.startup()
        print("Startup sequence reached radar_streaming.")
        print("Radar is running. Press Ctrl+C to stop.")
        _wait_until_interrupted()
        return 0
    except KeyboardInterrupt:
        print("\nCtrl+C received. Stopping radar startup session.")
        return 0
    except StartupError as exc:
        print(f"Startup failed: {exc}", file=sys.stderr)
        return 2
    finally:
        try:
            orchestrator.stop()
        except StartupError as exc:
            print(f"Startup cleanup failed: {exc}", file=sys.stderr)


def _wait_until_interrupted() -> None:
    while True:
        time.sleep(1.0)


if __name__ == "__main__":
    raise SystemExit(main())
