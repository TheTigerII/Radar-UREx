"""Settings and command construction for the Radar-UREx desktop launcher.

This module deliberately has no Qt dependency so command generation and
validation can be tested in headless environments.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
import sys

from main import calibrate
from main import run


@dataclass
class GuiSettings:
    """All user-editable settings exposed by the desktop application."""

    display: str = "combined"
    radar_port: str = ""
    duration_minutes: float = run.DEFAULT_DURATION_MINUTES
    dataset_destination: str = "dataset"
    config: Path = run.DEFAULT_CONFIG_PATH
    setup: Path = run.DEFAULT_SETUP_PATH
    host_ip: str = run.DEFAULT_HOST_IP
    data_port: int = run.DEFAULT_DATA_PORT
    radar_baud: int = run.DEFAULT_RADAR_BAUD
    radar_command_timeout: float = 10.0
    dca_timeout: float = 3.0
    dca_retries: int = 5
    socket_recv_buffer: int = run.DEFAULT_SOCKET_RECV_BUFFER_BYTES
    packet_queue_size: int = run.DEFAULT_PACKET_QUEUE_SIZE
    processing_queue_size: int = run.DEFAULT_PROCESSING_QUEUE_SIZE
    display_update_every: int = 1
    max_range_m: float = run.DEFAULT_MAX_RANGE_M
    capture_dir: Path = run.DEFAULT_CAPTURE_DIR
    processed_output: Path | None = None
    raw_output: Path | None = None
    classification: bool = False
    model_weights_dir: Path = run.DEFAULT_MODEL_WEIGHTS_DIR
    inference_logging: bool = False
    inference_log: Path | None = None
    evaluation_label: str = "unlabeled"
    performance_logging: bool = True
    performance_log: Path | None = None
    resource_sample_interval_seconds: float = 1.0
    pmm_background_calibration_seconds: float = 30.0
    pmm_max_target_speed_m_s: float = 4.0
    pmm_folding_size_min: int = 2
    pmm_folding_size_max: int = 32
    pmm_detection_threshold: float | None = None
    pmm_adaptive_threshold_sigma: float = run.MINI4_DEFAULT_ADAPTIVE_THRESHOLD_SIGMA
    pmm_adaptive_threshold_minimum: float = (
        run.MINI4_DEFAULT_ADAPTIVE_THRESHOLD_MINIMUM
    )
    pmm_history_seconds: float = 3.6
    pmm_provisional_frames: int = 5
    pmm_confirmation_window_frames: int = 10
    pmm_confirmation_hits: int = 7
    pmm_coast_frames: int = 10
    calibration_profile: Path = run.DEFAULT_CALIBRATION_PROFILE_PATH
    calibration_distance_m: float = calibrate.DEFAULT_TARGET_DISTANCE_M
    calibration_search_window_m: float = calibrate.DEFAULT_SEARCH_WINDOW_M
    calibration_warmup_frames: int = calibrate.DEFAULT_WARMUP_FRAMES
    calibration_frames: int = calibrate.DEFAULT_ACCEPTED_FRAMES
    calibration_timeout_seconds: float = calibrate.DEFAULT_TIMEOUT_SECONDS
    calibration_angle_deg: float = calibrate.DEFAULT_REFERENCE_ANGLE_DEG
    calibration_output: Path | None = None

    @property
    def is_calibration(self) -> bool:
        return self.display in calibrate.CALIBRATION_DISPLAY_MODES

    def validation_errors(self) -> list[str]:
        errors: list[str] = []
        if self.display not in run.DISPLAY_CHOICES:
            errors.append("Choose a valid display or calibration mode.")
        if not self.radar_port.strip():
            errors.append("Choose or enter the radar command UART.")
        if not self.config.is_file():
            errors.append(f"Radar profile does not exist: {self.config}")
        if not self.setup.is_file():
            errors.append(f"DCA1000 setup file does not exist: {self.setup}")
        if not self.host_ip.strip():
            errors.append("Host IP cannot be empty.")
        if not 1 <= self.data_port <= 65535:
            errors.append("Data port must be between 1 and 65535.")
        if self.duration_minutes < 0 or not math.isfinite(self.duration_minutes):
            errors.append("Duration must be finite and non-negative.")
        if self.classification and not self.is_calibration:
            required = (
                self.model_weights_dir / "model.onnx",
                self.model_weights_dir / "manifest.json",
            )
            missing = [str(path) for path in required if not path.is_file()]
            if missing:
                errors.append("Missing classification file(s): " + ", ".join(missing))
        if self.pmm_folding_size_min > self.pmm_folding_size_max:
            errors.append("Minimum folding size cannot exceed maximum folding size.")
        if self.pmm_confirmation_hits > self.pmm_confirmation_window_frames:
            errors.append("Confirmation hits cannot exceed the confirmation window.")
        if self.is_calibration:
            if not self.calibration_profile.is_file():
                errors.append(
                    f"Calibration profile does not exist: {self.calibration_profile}"
                )
            if self.calibration_distance_m <= 0:
                errors.append("Calibration distance must be positive.")
            if abs(self.calibration_angle_deg) > 60:
                errors.append("Calibration angle must be between -60 and +60 degrees.")
        return errors

    def command(self, python_executable: str | None = None) -> list[str]:
        """Return a prompt-free ``main/run.py`` command for these settings."""
        command = [
            python_executable or sys.executable,
            "-u",
            str(run.MAIN_DIR / "run.py"),
            "--display",
            self.display,
            "--radar-port",
            self.radar_port.strip(),
            "--config",
            str(self.config),
            "--setup",
            str(self.setup),
            "--host-ip",
            self.host_ip.strip(),
            "--data-port",
            str(self.data_port),
            "--radar-baud",
            str(self.radar_baud),
            "--radar-command-timeout",
            str(self.radar_command_timeout),
            "--dca-timeout",
            str(self.dca_timeout),
            "--dca-retries",
            str(self.dca_retries),
            "--socket-recv-buffer",
            str(self.socket_recv_buffer),
            "--packet-queue-size",
            str(self.packet_queue_size),
            "--processing-queue-size",
            str(self.processing_queue_size),
            "--display-update-every",
            str(self.display_update_every),
            "--max-range-m",
            str(self.max_range_m),
            "--capture-dir",
            str(self.capture_dir),
            "--pmm-background-calibration-seconds",
            str(self.pmm_background_calibration_seconds),
            "--pmm-max-target-speed-m-s",
            str(self.pmm_max_target_speed_m_s),
            "--pmm-folding-size-min",
            str(self.pmm_folding_size_min),
            "--pmm-folding-size-max",
            str(self.pmm_folding_size_max),
            "--pmm-adaptive-threshold-sigma",
            str(self.pmm_adaptive_threshold_sigma),
            "--pmm-adaptive-threshold-minimum",
            str(self.pmm_adaptive_threshold_minimum),
            "--pmm-history-seconds",
            str(self.pmm_history_seconds),
            "--pmm-provisional-frames",
            str(self.pmm_provisional_frames),
            "--pmm-confirmation-window-frames",
            str(self.pmm_confirmation_window_frames),
            "--pmm-confirmation-hits",
            str(self.pmm_confirmation_hits),
            "--pmm-coast-frames",
            str(self.pmm_coast_frames),
        ]
        if self.pmm_detection_threshold is not None:
            command.extend(
                ("--pmm-detection-threshold", str(self.pmm_detection_threshold))
            )

        if self.is_calibration:
            command.extend(
                (
                    "--calibration-profile",
                    str(self.calibration_profile),
                    "--calibration-distance-m",
                    str(self.calibration_distance_m),
                    "--calibration-search-window-m",
                    str(self.calibration_search_window_m),
                    "--calibration-warmup-frames",
                    str(self.calibration_warmup_frames),
                    "--calibration-frames",
                    str(self.calibration_frames),
                    "--calibration-timeout-seconds",
                    str(self.calibration_timeout_seconds),
                    "--calibration-angle-deg",
                    str(self.calibration_angle_deg),
                )
            )
            if self.calibration_output is not None:
                command.extend(("--calibration-output", str(self.calibration_output)))
            return command

        command.extend(
            (
                "--duration-minutes",
                str(self.duration_minutes),
                "--dataset-destination",
                self.dataset_destination,
                "--model-weights-dir",
                str(self.model_weights_dir),
                "--resource-sample-interval-seconds",
                str(self.resource_sample_interval_seconds),
                "--classification" if self.classification else "--no-classification",
                (
                    "--performance-logging"
                    if self.performance_logging
                    else "--no-performance-logging"
                ),
            )
        )
        if self.processed_output is not None:
            command.extend(("--processed-output", str(self.processed_output)))
        if self.raw_output is not None:
            command.extend(("--raw-output", str(self.raw_output)))
        if self.performance_log is not None and self.performance_logging:
            command.extend(("--performance-log", str(self.performance_log)))
        if self.classification:
            command.append(
                "--inference-logging"
                if self.inference_logging
                else "--no-inference-logging"
            )
            if self.inference_logging:
                command.extend(("--evaluation-label", self.evaluation_label))
                if self.inference_log is not None:
                    command.extend(("--inference-log", str(self.inference_log)))
        return command
