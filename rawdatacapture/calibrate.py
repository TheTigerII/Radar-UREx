"""Raw-ADC range-bias and RX-channel calibration support.

The calibration path intentionally stays independent of the normal processed
radar output.  It consumes range-FFT cubes, estimates a corner-reflector peak,
and produces the compensation command understood by the radar profiles.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from datetime import datetime
import json
import math
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any, Iterable, Optional

import numpy as np


CALIBRATION_DISPLAY_MODE = "calibration"
CALIBRATION_RESULT_PREFIX = "CALIBRATION_RESULT "
DEFAULT_TARGET_DISTANCE_M = 1.0
DEFAULT_SEARCH_WINDOW_M = 0.20
DEFAULT_WARMUP_FRAMES = 16
DEFAULT_ACCEPTED_FRAMES = 64
DEFAULT_TIMEOUT_SECONDS = 90.0

_PROFILE_OVERRIDES = {
    "guiMonitor": "guiMonitor -1 0 0 0 0 0 0",
    "lvdsStreamCfg": "lvdsStreamCfg -1 0 1 0",
}


@dataclass(frozen=True)
class CalibrationSettings:
    target_distance_m: float = DEFAULT_TARGET_DISTANCE_M
    search_window_m: float = DEFAULT_SEARCH_WINDOW_M
    warmup_frames: int = DEFAULT_WARMUP_FRAMES
    accepted_frames: int = DEFAULT_ACCEPTED_FRAMES
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS
    min_peak_prominence_db: float = 10.0
    max_range_std_m: float = 0.01
    max_phase_std_deg: float = 5.0
    max_magnitude_cv: float = 0.05

    def __post_init__(self) -> None:
        if not math.isfinite(self.target_distance_m) or self.target_distance_m <= 0:
            raise ValueError("Laser distance must be a positive finite value")
        if not math.isfinite(self.search_window_m) or self.search_window_m <= 0:
            raise ValueError("Search window must be a positive finite value")
        if self.warmup_frames < 0:
            raise ValueError("Warm-up frame count cannot be negative")
        if self.accepted_frames < 1:
            raise ValueError("Accepted frame count must be at least one")
        if not math.isfinite(self.timeout_seconds) or self.timeout_seconds <= 0:
            raise ValueError("Calibration timeout must be positive")


@dataclass(frozen=True)
class CalibrationResult:
    target_distance_m: float
    search_window_m: float
    measured_range_m: float
    range_bias_m: float
    coefficients: tuple[complex, ...]
    accepted_frames: int
    range_std_m: float
    max_phase_std_deg: float
    max_magnitude_cv: float
    tx_order: tuple[int, ...]

    def __post_init__(self) -> None:
        if len(self.coefficients) != 12:
            raise ValueError("Calibration requires exactly 12 TX/RX coefficients")

    @property
    def command(self) -> str:
        values = [f"{self.range_bias_m:.7g}"]
        for coefficient in self.coefficients:
            values.extend((f"{coefficient.real:.7g}", f"{coefficient.imag:.7g}"))
        return "compRangeBiasAndRxChanPhase " + " ".join(values)

    def to_dict(self) -> dict[str, Any]:
        return {
            "target_distance_m": self.target_distance_m,
            "search_window_m": self.search_window_m,
            "measured_range_m": self.measured_range_m,
            "range_bias_m": self.range_bias_m,
            "accepted_frames": self.accepted_frames,
            "range_std_m": self.range_std_m,
            "max_phase_std_deg": self.max_phase_std_deg,
            "max_magnitude_cv": self.max_magnitude_cv,
            "tx_order": list(self.tx_order),
            "coefficients": [
                {"real": coefficient.real, "imag": coefficient.imag}
                for coefficient in self.coefficients
            ],
            "command": self.command,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CalibrationResult":
        coefficients = tuple(
            complex(float(item["real"]), float(item["imag"]))
            for item in data["coefficients"]
        )
        return cls(
            target_distance_m=float(data["target_distance_m"]),
            search_window_m=float(data["search_window_m"]),
            measured_range_m=float(data["measured_range_m"]),
            range_bias_m=float(data["range_bias_m"]),
            coefficients=coefficients,
            accepted_frames=int(data["accepted_frames"]),
            range_std_m=float(data["range_std_m"]),
            max_phase_std_deg=float(data["max_phase_std_deg"]),
            max_magnitude_cv=float(data["max_magnitude_cv"]),
            tx_order=tuple(int(value) for value in data["tx_order"]),
        )


@dataclass(frozen=True)
class CalibrationPayload:
    range_axis_m: np.ndarray
    range_profile_db: np.ndarray
    search_min_m: float
    search_max_m: float
    peak_range_m: Optional[float]
    range_bias_m: Optional[float]
    accepted_frames: int
    required_frames: int
    coefficient_magnitudes: np.ndarray
    coefficient_phases_deg: np.ndarray
    range_std_m: Optional[float]
    max_phase_std_deg: Optional[float]
    max_magnitude_cv: Optional[float]
    status: str


def _active_profile_lines(text: str) -> Iterable[tuple[int, str, list[str]]]:
    for line_number, raw_line in enumerate(text.splitlines(), 1):
        stripped = raw_line.strip()
        if not stripped or stripped.startswith(("%", "#")):
            continue
        tokens = stripped.split()
        yield line_number, raw_line, tokens


def _command_occurrences(text: str, command: str) -> list[tuple[int, str, list[str]]]:
    return [entry for entry in _active_profile_lines(text) if entry[2][0] == command]


def validate_calibration_profile_text(
    text: str,
    radar_config: Any,
    settings: CalibrationSettings,
    *,
    require_raw_lvds: bool = False,
) -> None:
    """Validate requirements specific to the raw-ADC calibration workflow."""

    for command in ("guiMonitor", "measureRangeBiasAndRxChanPhase", "lvdsStreamCfg"):
        count = len(_command_occurrences(text, command))
        if count != 1:
            raise ValueError(
                f"Calibration profile must contain exactly one {command} command; found {count}"
            )

    rx_channels = int(getattr(radar_config, "rx_channels"))
    chirps_per_loop = int(getattr(radar_config, "chirps_per_loop"))
    tx_masks = tuple(int(value) for value in getattr(radar_config, "chirp_tx_masks"))
    if rx_channels != 4:
        raise ValueError(f"Calibration requires four RX channels; profile has {rx_channels}")
    if chirps_per_loop != 3 or len(tx_masks) != 3:
        raise ValueError("Calibration requires exactly three TDM chirps per loop")
    if sorted(tx_masks) != [1, 2, 4] or any(mask & (mask - 1) for mask in tx_masks):
        raise ValueError(f"Calibration requires one chirp for each physical TX; found {tx_masks}")

    adc_samples = int(getattr(radar_config, "adc_samples"))
    if adc_samples < 4:
        raise ValueError("Calibration profile has too few ADC samples")
    configured_range_axis = radar_config.range_axis_m()
    if configured_range_axis is None:
        raise ValueError(
            "Calibration profile must define sample rate and frequency slope"
        )
    range_axis = np.asarray(configured_range_axis, dtype=np.float64)
    if range_axis.ndim != 1 or range_axis.size < 4:
        raise ValueError("Calibration profile has an incompatible range axis")
    search_min = settings.target_distance_m - settings.search_window_m
    search_max = settings.target_distance_m + settings.search_window_m
    if search_min < range_axis[1] or search_max > range_axis[-2]:
        raise ValueError(
            "Laser distance/search window falls outside the usable three-bin range interval"
        )

    lvds_tokens = _command_occurrences(text, "lvdsStreamCfg")[0][2]
    if len(lvds_tokens) < 5:
        raise ValueError("Malformed lvdsStreamCfg command")
    measure_tokens = _command_occurrences(
        text, "measureRangeBiasAndRxChanPhase"
    )[0][2]
    if len(measure_tokens) != 4:
        raise ValueError("Malformed measureRangeBiasAndRxChanPhase command")
    try:
        measure_distance = float(measure_tokens[2])
        measure_window = float(measure_tokens[3])
    except ValueError as exc:
        raise ValueError("Malformed measureRangeBiasAndRxChanPhase values") from exc
    if require_raw_lvds:
        if lvds_tokens != ["lvdsStreamCfg", "-1", "0", "1", "0"]:
            raise ValueError(
                "Calibration runtime profile has raw LVDS streaming disabled "
                "or an incompatible LVDS mode"
            )
        gui_tokens = _command_occurrences(text, "guiMonitor")[0][2]
        if gui_tokens != ["guiMonitor", "-1", "0", "0", "0", "0", "0", "0"]:
            raise ValueError("Calibration runtime profile must disable guiMonitor output")
        if measure_tokens[1] != "0":
            raise ValueError(
                "Calibration runtime profile must disable firmware range calibration"
            )
        if not math.isclose(measure_distance, settings.target_distance_m) or not math.isclose(
            measure_window, settings.search_window_m
        ):
            raise ValueError(
                "Calibration runtime profile distance/window does not match requested settings"
            )


def create_runtime_profile(
    source_path: Path | str,
    destination_path: Path | str,
    radar_config: Any,
    settings: CalibrationSettings,
) -> Path:
    """Create the temporary capture profile while leaving the source untouched."""

    source = Path(source_path)
    destination = Path(destination_path)
    text = source.read_text(encoding="utf-8")
    validate_calibration_profile_text(text, radar_config, settings)

    replacements = dict(_PROFILE_OVERRIDES)
    replacements["measureRangeBiasAndRxChanPhase"] = (
        "measureRangeBiasAndRxChanPhase 0 "
        f"{settings.target_distance_m:.7g} {settings.search_window_m:.7g}"
    )
    output_lines: list[str] = []
    replaced = {key: 0 for key in replacements}
    for raw_line in text.splitlines(keepends=True):
        body = raw_line.rstrip("\r\n")
        ending = raw_line[len(body) :]
        stripped = body.strip()
        command = stripped.split(maxsplit=1)[0] if stripped and not stripped.startswith(("%", "#")) else ""
        if command in replacements:
            indent = body[: len(body) - len(body.lstrip())]
            output_lines.append(indent + replacements[command] + ending)
            replaced[command] += 1
        else:
            output_lines.append(raw_line)
    if any(count != 1 for count in replaced.values()):
        raise ValueError(f"Ambiguous runtime profile commands: {replaced}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes("".join(output_lines).encode("utf-8"))
    return destination


def write_calibration_report(path: Path | str, result: CalibrationResult) -> Path:
    report_path = Path(path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(result.to_dict(), indent=2) + "\n", encoding="utf-8")
    return report_path


def apply_calibration_to_profile(
    profile_path: Path | str,
    result: CalibrationResult,
) -> Path:
    """Back up and atomically update the profile's single compensation line."""

    profile = Path(profile_path)
    text = profile.read_text(encoding="utf-8")
    occurrences = _command_occurrences(text, "compRangeBiasAndRxChanPhase")
    if len(occurrences) != 1:
        raise ValueError(
            "Operational profile must contain exactly one compRangeBiasAndRxChanPhase command; "
            f"found {len(occurrences)}"
        )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup = profile.with_name(f"{profile.name}.{timestamp}.bak")
    suffix = 1
    while backup.exists():
        backup = profile.with_name(f"{profile.name}.{timestamp}_{suffix}.bak")
        suffix += 1
    shutil.copy2(profile, backup)

    output_lines: list[str] = []
    replaced = 0
    for raw_line in text.splitlines(keepends=True):
        body = raw_line.rstrip("\r\n")
        ending = raw_line[len(body) :]
        stripped = body.strip()
        command = stripped.split(maxsplit=1)[0] if stripped and not stripped.startswith(("%", "#")) else ""
        if command == "compRangeBiasAndRxChanPhase":
            indent = body[: len(body) - len(body.lstrip())]
            output_lines.append(indent + result.command + ending)
            replaced += 1
        else:
            output_lines.append(raw_line)
    if replaced != 1:
        raise ValueError("Compensation line changed while applying calibration")

    temporary_handle, temporary_name = tempfile.mkstemp(
        prefix=f".{profile.name}.", suffix=".tmp", dir=profile.parent
    )
    try:
        with os.fdopen(temporary_handle, "w", encoding="utf-8", newline="") as stream:
            stream.write("".join(output_lines))
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, profile)
    except Exception:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise
    return backup


class CalibrationAccumulator:
    """Accumulate valid frames and estimate range/channel corrections."""

    def __init__(self, radar_config: Any, settings: CalibrationSettings) -> None:
        self.config = radar_config
        self.settings = settings
        configured_range_axis = radar_config.range_axis_m()
        if configured_range_axis is None:
            raise ValueError("Calibration requires a configured range axis")
        self.range_axis_m = np.asarray(configured_range_axis, dtype=np.float64)
        self.tx_masks = tuple(int(value) for value in radar_config.chirp_tx_masks)
        self._physical_tx_indices = tuple(mask.bit_length() - 1 for mask in self.tx_masks)
        self._warmup_seen = 0
        self._accepted_profiles: deque[np.ndarray] = deque(maxlen=settings.accepted_frames)
        self._accepted_biases: deque[float] = deque(maxlen=settings.accepted_frames)
        self.result: Optional[CalibrationResult] = None
        self.last_payload: Optional[CalibrationPayload] = None

    def _empty_coefficients(self) -> tuple[np.ndarray, np.ndarray]:
        return np.zeros(12, dtype=np.float64), np.zeros(12, dtype=np.float64)

    def _payload(
        self,
        profile_db: np.ndarray,
        status: str,
        peak_range_m: Optional[float] = None,
        range_bias_m: Optional[float] = None,
        channels: Optional[np.ndarray] = None,
    ) -> CalibrationPayload:
        if channels is None:
            magnitudes, phases = self._empty_coefficients()
        else:
            reference_amplitude = float(np.min(np.abs(channels)))
            provisional = reference_amplitude * np.conj(channels) / np.maximum(
                np.abs(channels) ** 2, np.finfo(float).tiny
            )
            magnitudes = np.abs(provisional)
            phases = np.rad2deg(np.angle(provisional))
        range_std_m: Optional[float] = None
        max_phase_std_deg: Optional[float] = None
        max_magnitude_cv: Optional[float] = None
        if len(self._accepted_profiles) >= 2:
            stack = np.asarray(self._accepted_profiles)
            range_std_m = float(np.std(np.asarray(self._accepted_biases)))
            max_phase_std_deg = float(
                np.max(np.std(np.unwrap(np.angle(stack), axis=0), axis=0))
                * 180.0
                / np.pi
            )
            channel_magnitudes = np.abs(stack)
            max_magnitude_cv = float(
                np.max(
                    np.std(channel_magnitudes, axis=0)
                    / np.maximum(
                        np.mean(channel_magnitudes, axis=0),
                        np.finfo(float).tiny,
                    )
                )
            )
        payload = CalibrationPayload(
            range_axis_m=self.range_axis_m.copy(),
            range_profile_db=np.asarray(profile_db, dtype=np.float64),
            search_min_m=self.settings.target_distance_m - self.settings.search_window_m,
            search_max_m=self.settings.target_distance_m + self.settings.search_window_m,
            peak_range_m=peak_range_m,
            range_bias_m=range_bias_m,
            accepted_frames=len(self._accepted_profiles),
            required_frames=self.settings.accepted_frames,
            coefficient_magnitudes=np.asarray(magnitudes, dtype=np.float64),
            coefficient_phases_deg=np.asarray(phases, dtype=np.float64),
            range_std_m=range_std_m,
            max_phase_std_deg=max_phase_std_deg,
            max_magnitude_cv=max_magnitude_cv,
            status=status,
        )
        self.last_payload = payload
        return payload

    def update(self, range_fft: np.ndarray) -> CalibrationPayload:
        cube = np.asarray(range_fft)
        expected_chirps = int(self.config.loops_per_frame) * int(self.config.chirps_per_loop)
        expected_shape = (expected_chirps, int(self.config.rx_channels), int(self.config.adc_samples))
        if cube.shape != expected_shape:
            raise ValueError(f"Unexpected range FFT shape {cube.shape}; expected {expected_shape}")

        loops = cube.reshape(
            int(self.config.loops_per_frame),
            int(self.config.chirps_per_loop),
            int(self.config.rx_channels),
            int(self.config.adc_samples),
        )
        if loops.shape[0] > 1:
            window = np.hanning(loops.shape[0]).astype(np.float64)
            if not np.any(window):
                window = np.ones(loops.shape[0], dtype=np.float64)
        else:
            window = np.ones(1, dtype=np.float64)
        zero_doppler = np.tensordot(window / window.sum(), loops, axes=(0, 0))

        physical = np.empty((3, 4, cube.shape[-1]), dtype=np.complex128)
        for chirp_index, physical_tx_index in enumerate(self._physical_tx_indices):
            physical[physical_tx_index] = zero_doppler[chirp_index]
        tx_rx_profiles = physical.reshape(12, cube.shape[-1])
        combined = np.sqrt(np.sum(np.abs(tx_rx_profiles) ** 2, axis=0))
        scale = max(float(np.max(combined)), np.finfo(np.float64).tiny)
        profile_db = 20.0 * np.log10(np.maximum(combined / scale, np.finfo(np.float64).tiny))

        if self._warmup_seen < self.settings.warmup_frames:
            self._warmup_seen += 1
            return self._payload(
                profile_db,
                f"Warm-up frame {self._warmup_seen}/{self.settings.warmup_frames}",
            )

        minimum = self.settings.target_distance_m - self.settings.search_window_m
        maximum = self.settings.target_distance_m + self.settings.search_window_m
        candidates = np.flatnonzero((self.range_axis_m >= minimum) & (self.range_axis_m <= maximum))
        if candidates.size < 3:
            return self._payload(profile_db, "Search window contains fewer than three bins")
        local_index = int(np.argmax(combined[candidates]))
        peak_index = int(candidates[local_index])
        if peak_index <= candidates[0] or peak_index >= candidates[-1] or peak_index <= 0 or peak_index >= combined.size - 1:
            return self._payload(profile_db, "Peak is on the search-window boundary")

        background_mask = np.ones(candidates.size, dtype=bool)
        background_mask[max(0, local_index - 1) : min(candidates.size, local_index + 2)] = False
        background_values = combined[candidates][background_mask]
        background = float(np.median(background_values)) if background_values.size else 0.0
        prominence_db = 20.0 * math.log10(
            max(float(combined[peak_index]), np.finfo(float).tiny)
            / max(background, np.finfo(float).tiny)
        )
        if prominence_db < self.settings.min_peak_prominence_db:
            return self._payload(
                profile_db,
                f"Target peak prominence {prominence_db:.1f} dB is too low",
            )

        left, centre, right = (float(combined[index]) for index in (peak_index - 1, peak_index, peak_index + 1))
        denominator = left - 2.0 * centre + right
        offset = 0.0 if abs(denominator) < np.finfo(float).eps else 0.5 * (left - right) / denominator
        offset = float(np.clip(offset, -0.5, 0.5))
        range_step = float(self.range_axis_m[1] - self.range_axis_m[0])
        measured_range = float(self.range_axis_m[peak_index] + offset * range_step)
        range_bias = measured_range - self.settings.target_distance_m

        channels = np.asarray(tx_rx_profiles[:, peak_index], dtype=np.complex128)
        if not np.all(np.isfinite(channels)) or float(np.min(np.abs(channels))) <= np.finfo(float).tiny:
            return self._payload(profile_db, "One or more virtual channels have no valid target signal")
        aligned = channels * np.exp(-1j * np.angle(channels[0]))
        self._accepted_profiles.append(aligned)
        self._accepted_biases.append(range_bias)

        stack = np.asarray(self._accepted_profiles)
        mean_channels = np.mean(stack, axis=0)
        if len(stack) < self.settings.accepted_frames:
            return self._payload(
                profile_db,
                f"Accepted frame {len(stack)}/{self.settings.accepted_frames}",
                measured_range,
                range_bias,
                mean_channels,
            )

        range_std = float(np.std(np.asarray(self._accepted_biases)))
        phases = np.unwrap(np.angle(stack), axis=0)
        max_phase_std = float(np.max(np.std(phases, axis=0)) * 180.0 / np.pi)
        magnitudes = np.abs(stack)
        mean_magnitudes = np.maximum(np.mean(magnitudes, axis=0), np.finfo(float).tiny)
        max_magnitude_cv = float(np.max(np.std(magnitudes, axis=0) / mean_magnitudes))
        stable = (
            range_std <= self.settings.max_range_std_m
            and max_phase_std <= self.settings.max_phase_std_deg
            and max_magnitude_cv <= self.settings.max_magnitude_cv
        )
        if not stable:
            status = (
                "Waiting for stability: "
                f"range σ={range_std:.4f} m, phase σ={max_phase_std:.1f}°, "
                f"magnitude CV={max_magnitude_cv:.3f}"
            )
            return self._payload(profile_db, status, measured_range, range_bias, mean_channels)

        reference_amplitude = float(np.min(np.abs(mean_channels)))
        coefficients = reference_amplitude * np.conj(mean_channels) / np.maximum(
            np.abs(mean_channels) ** 2, np.finfo(float).tiny
        )
        mean_bias = float(np.mean(np.asarray(self._accepted_biases)))
        self.result = CalibrationResult(
            target_distance_m=self.settings.target_distance_m,
            search_window_m=self.settings.search_window_m,
            measured_range_m=self.settings.target_distance_m + mean_bias,
            range_bias_m=mean_bias,
            coefficients=tuple(complex(value) for value in coefficients),
            accepted_frames=len(stack),
            range_std_m=range_std,
            max_phase_std_deg=max_phase_std,
            max_magnitude_cv=max_magnitude_cv,
            tx_order=self.tx_masks,
        )
        return self._payload(
            profile_db,
            "Stable calibration result ready",
            self.result.measured_range_m,
            self.result.range_bias_m,
            mean_channels,
        )


def run_calibration_display(
    display_queue: Any,
    stop_event: Any,
    startup_status_queue: Any = None,
) -> None:
    """Run the calibration-specific PyQtGraph display in a child process."""

    try:
        from pyqtgraph.Qt import QtCore, QtWidgets
        import pyqtgraph as pg
    except Exception as exc:  # pragma: no cover - depends on optional GUI runtime
        if startup_status_queue is not None:
            startup_status_queue.put(
                {"state": "error", "message": f"Calibration display unavailable: {exc}"}
            )
        return

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    window = QtWidgets.QWidget()
    window.setWindowTitle("Radar calibration")
    layout = QtWidgets.QVBoxLayout(window)
    status_label = QtWidgets.QLabel("Waiting for calibration frames")
    layout.addWidget(status_label)
    profile_plot = pg.PlotWidget(title="Zero-Doppler range profile")
    profile_plot.setLabel("bottom", "Range", units="m")
    profile_plot.setLabel("left", "Relative magnitude", units="dB")
    profile_curve = profile_plot.plot(pen="y")
    search_low = pg.InfiniteLine(angle=90, pen=pg.mkPen("c", style=QtCore.Qt.PenStyle.DashLine))
    search_high = pg.InfiniteLine(angle=90, pen=pg.mkPen("c", style=QtCore.Qt.PenStyle.DashLine))
    peak_line = pg.InfiniteLine(angle=90, movable=False, pen=pg.mkPen("r", width=2))
    for line in (search_low, search_high, peak_line):
        profile_plot.addItem(line)
    layout.addWidget(profile_plot)
    channel_plot = pg.PlotWidget(title="TX-major/RX-minor channel correction")
    channel_plot.setLabel("bottom", "Physical channel (TX-major / RX-minor)")
    channel_plot.setLabel("left", "Magnitude / phase")
    magnitude_curve = channel_plot.plot(pen="g", symbol="o", name="Magnitude")
    phase_curve = channel_plot.plot(pen="m", symbol="t", name="Phase (deg)")
    layout.addWidget(channel_plot)
    window.resize(1000, 800)
    window.show()

    if startup_status_queue is not None:
        startup_status_queue.put(
            {"state": "ready", "message": "Live calibration display ready."}
        )

    def poll() -> None:
        latest = None
        while True:
            try:
                latest = display_queue.get_nowait()
            except Exception:
                break
        if isinstance(latest, CalibrationPayload):
            profile_curve.setData(latest.range_axis_m, latest.range_profile_db)
            search_low.setValue(latest.search_min_m)
            search_high.setValue(latest.search_max_m)
            peak_line.setVisible(latest.peak_range_m is not None)
            if latest.peak_range_m is not None:
                peak_line.setValue(latest.peak_range_m)
            channels = np.arange(1, 13)
            magnitude_curve.setData(channels, latest.coefficient_magnitudes)
            phase_curve.setData(channels, latest.coefficient_phases_deg)
            bias_text = "" if latest.range_bias_m is None else f", bias {latest.range_bias_m:+.4f} m"
            stability_text = (
                ""
                if latest.range_std_m is None
                else (
                    f", σrange {latest.range_std_m:.4f} m, "
                    f"σphase {latest.max_phase_std_deg:.1f}°, "
                    f"magnitude CV {latest.max_magnitude_cv:.3f}"
                )
            )
            status_label.setText(
                f"{latest.status} — {latest.accepted_frames}/{latest.required_frames}"
                f"{bias_text}{stability_text}"
            )
        if stop_event.is_set():
            window.close()

    timer = QtCore.QTimer()
    timer.timeout.connect(poll)
    timer.start(50)
    app.exec()
