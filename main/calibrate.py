"""Raw-ADC range, RX-channel, azimuth, and elevation calibration support.

The calibration path intentionally stays independent of the normal processed
radar output.  It consumes range-FFT cubes, estimates a corner-reflector peak,
and produces firmware compensation or host-side angular corrections.
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

from .pmm import PmmConfig, capon_pmm_angle_scores


CALIBRATION_DISPLAY_MODE = "calibration"
AZIMUTH_CALIBRATION_DISPLAY_MODE = "azimuth-calibration"
ELEVATION_CALIBRATION_DISPLAY_MODE = "elevation-calibration"
CALIBRATION_DISPLAY_MODES = frozenset(
    {
        CALIBRATION_DISPLAY_MODE,
        AZIMUTH_CALIBRATION_DISPLAY_MODE,
        ELEVATION_CALIBRATION_DISPLAY_MODE,
    }
)
CALIBRATION_RESULT_PREFIX = "CALIBRATION_RESULT "
DEFAULT_TARGET_DISTANCE_M = 1.0
DEFAULT_SEARCH_WINDOW_M = 0.20
DEFAULT_WARMUP_FRAMES = 16
DEFAULT_ACCEPTED_FRAMES = 64
DEFAULT_TIMEOUT_SECONDS = 90.0
DEFAULT_REFERENCE_ANGLE_DEG = 0.0
DEFAULT_MAX_ANGLE_STD_DEG = 1.0
HOST_ANGLE_CALIBRATION_MARKER = "% hostAngleCalibration"

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
    calibration_type: str = "range"
    reference_angle_deg: float = DEFAULT_REFERENCE_ANGLE_DEG
    max_angle_std_deg: float = DEFAULT_MAX_ANGLE_STD_DEG

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
        if self.calibration_type not in {"range", "azimuth", "elevation"}:
            raise ValueError("Calibration type must be range, azimuth, or elevation")
        if (
            not math.isfinite(self.reference_angle_deg)
            or abs(self.reference_angle_deg) > 60.0
        ):
            raise ValueError("Reference angle must be within -60 to +60 degrees")
        if not math.isfinite(self.max_angle_std_deg) or self.max_angle_std_deg <= 0:
            raise ValueError("Maximum angle standard deviation must be positive")


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
        values = [f"{self.range_bias_m:.7f}"]
        for coefficient in self.coefficients:
            values.extend((f"{coefficient.real:.5f}", f"{coefficient.imag:.5f}"))
        return "compRangeBiasAndRxChanPhase " + " ".join(values)

    def to_dict(self) -> dict[str, Any]:
        return {
            "calibration_type": "range",
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
class AngularCalibrationResult:
    calibration_type: str
    target_distance_m: float
    search_window_m: float
    reference_angle_deg: float
    measured_angle_deg: float
    angle_bias_deg: float
    accepted_frames: int
    angle_std_deg: float
    measured_range_m: float
    tx_order: tuple[int, ...]

    def __post_init__(self) -> None:
        if self.calibration_type not in {"azimuth", "elevation"}:
            raise ValueError("Angular result type must be azimuth or elevation")

    def to_dict(self) -> dict[str, Any]:
        return {
            "calibration_type": self.calibration_type,
            "target_distance_m": self.target_distance_m,
            "search_window_m": self.search_window_m,
            "reference_angle_deg": self.reference_angle_deg,
            "measured_angle_deg": self.measured_angle_deg,
            "angle_bias_deg": self.angle_bias_deg,
            "accepted_frames": self.accepted_frames,
            "angle_std_deg": self.angle_std_deg,
            "measured_range_m": self.measured_range_m,
            "tx_order": list(self.tx_order),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AngularCalibrationResult":
        return cls(
            calibration_type=str(data["calibration_type"]),
            target_distance_m=float(data["target_distance_m"]),
            search_window_m=float(data["search_window_m"]),
            reference_angle_deg=float(data["reference_angle_deg"]),
            measured_angle_deg=float(data["measured_angle_deg"]),
            angle_bias_deg=float(data["angle_bias_deg"]),
            accepted_frames=int(data["accepted_frames"]),
            angle_std_deg=float(data["angle_std_deg"]),
            measured_range_m=float(data["measured_range_m"]),
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


@dataclass(frozen=True)
class AngularCalibrationPayload:
    calibration_type: str
    range_axis_m: np.ndarray
    range_profile_db: np.ndarray
    search_min_m: float
    search_max_m: float
    angle_axis_deg: np.ndarray
    angle_profile_db: np.ndarray
    reference_angle_deg: float
    measured_angle_deg: Optional[float]
    angle_bias_deg: Optional[float]
    angle_std_deg: Optional[float]
    accepted_frames: int
    required_frames: int
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


def write_calibration_report(
    path: Path | str,
    result: CalibrationResult | AngularCalibrationResult,
) -> Path:
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


def _parse_host_angle_calibration(text: str) -> tuple[float, float, int]:
    azimuth_bias_deg = 0.0
    elevation_bias_deg = 0.0
    count = 0
    for raw_line in text.splitlines():
        stripped = raw_line.strip()
        if not stripped.startswith(HOST_ANGLE_CALIBRATION_MARKER):
            continue
        count += 1
        tokens = stripped.split()
        if len(tokens) != 6 or tokens[2] != "azimuthBiasDeg" or tokens[4] != "elevationBiasDeg":
            raise ValueError("Malformed hostAngleCalibration profile marker")
        try:
            azimuth_bias_deg = float(tokens[3])
            elevation_bias_deg = float(tokens[5])
        except ValueError as exc:
            raise ValueError("Malformed host angle calibration values") from exc
        if not math.isfinite(azimuth_bias_deg) or not math.isfinite(elevation_bias_deg):
            raise ValueError("Host angle calibration values must be finite")
    return azimuth_bias_deg, elevation_bias_deg, count


def apply_angular_calibration_to_profile(
    profile_path: Path | str,
    result: AngularCalibrationResult,
) -> Path:
    """Store a host-only angular offset in an SDK-safe profile comment."""

    profile = Path(profile_path)
    text = profile.read_text(encoding="utf-8")
    azimuth_bias_deg, elevation_bias_deg, count = _parse_host_angle_calibration(text)
    if count > 1:
        raise ValueError(
            "Operational profile contains duplicate hostAngleCalibration markers"
        )
    if result.calibration_type == "azimuth":
        azimuth_bias_deg = result.angle_bias_deg
    else:
        elevation_bias_deg = result.angle_bias_deg
    marker = (
        f"{HOST_ANGLE_CALIBRATION_MARKER} "
        f"azimuthBiasDeg {azimuth_bias_deg:.7g} "
        f"elevationBiasDeg {elevation_bias_deg:.7g}"
    )

    output_lines: list[str] = []
    replaced = False
    inserted = False
    for raw_line in text.splitlines(keepends=True):
        body = raw_line.rstrip("\r\n")
        ending = raw_line[len(body) :]
        if body.strip().startswith(HOST_ANGLE_CALIBRATION_MARKER):
            output_lines.append(marker + ending)
            replaced = True
            continue
        if not replaced and not inserted and body.strip().startswith("sensorStart"):
            output_lines.append(marker + (ending or "\n"))
            inserted = True
        output_lines.append(raw_line)
    if not replaced and not inserted:
        if output_lines and not output_lines[-1].endswith(("\n", "\r")):
            output_lines[-1] += "\n"
        output_lines.append(marker + "\n")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup = profile.with_name(f"{profile.name}.{timestamp}.bak")
    suffix = 1
    while backup.exists():
        backup = profile.with_name(f"{profile.name}.{timestamp}_{suffix}.bak")
        suffix += 1
    shutil.copy2(profile, backup)
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
        return CalibrationPayload(
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


class AngularCalibrationAccumulator:
    """Estimate an azimuth or elevation offset from a known tripod angle."""

    def __init__(
        self,
        radar_config: Any,
        settings: CalibrationSettings,
        pmm_config: Optional[PmmConfig] = None,
    ) -> None:
        if settings.calibration_type not in {"azimuth", "elevation"}:
            raise ValueError("Angular accumulator requires azimuth or elevation mode")
        self.config = radar_config
        self.settings = settings
        self.pmm_config = pmm_config or PmmConfig()
        configured_range_axis = radar_config.range_axis_m()
        if configured_range_axis is None:
            raise ValueError("Angular calibration requires a configured range axis")
        # range_axis_m() is already the physical axis corrected by the
        # operational profile's range_bias_m. Use it for both reflector-bin
        # selection and the range reported in the calibration result.
        self.range_axis_m = np.asarray(configured_range_axis, dtype=np.float64)
        self.tx_masks = tuple(int(value) for value in radar_config.chirp_tx_masks)
        self._warmup_seen = 0
        self._accepted_angles: deque[float] = deque(maxlen=settings.accepted_frames)
        self._accepted_ranges: deque[float] = deque(maxlen=settings.accepted_frames)
        self.result: Optional[AngularCalibrationResult] = None
        self.angle_axis_deg = np.arange(
            -self.pmm_config.angle_limit_deg,
            self.pmm_config.angle_limit_deg
            + self.pmm_config.angle_step_deg * 0.5,
            self.pmm_config.angle_step_deg,
            dtype=np.float64,
        )

    def _payload(
        self,
        range_profile_db: np.ndarray,
        angle_profile_db: Optional[np.ndarray],
        status: str,
        measured_angle_deg: Optional[float] = None,
    ) -> AngularCalibrationPayload:
        angle_std = (
            float(np.std(np.asarray(self._accepted_angles)))
            if len(self._accepted_angles) >= 2
            else None
        )
        return AngularCalibrationPayload(
            calibration_type=self.settings.calibration_type,
            range_axis_m=self.range_axis_m.copy(),
            range_profile_db=np.asarray(range_profile_db, dtype=np.float64),
            search_min_m=self.settings.target_distance_m - self.settings.search_window_m,
            search_max_m=self.settings.target_distance_m + self.settings.search_window_m,
            angle_axis_deg=self.angle_axis_deg.copy(),
            angle_profile_db=(
                np.asarray(angle_profile_db, dtype=np.float64)
                if angle_profile_db is not None
                else np.full(self.angle_axis_deg.size, -120.0, dtype=np.float64)
            ),
            reference_angle_deg=self.settings.reference_angle_deg,
            measured_angle_deg=measured_angle_deg,
            angle_bias_deg=(
                measured_angle_deg - self.settings.reference_angle_deg
                if measured_angle_deg is not None
                else None
            ),
            angle_std_deg=angle_std,
            accepted_frames=len(self._accepted_angles),
            required_frames=self.settings.accepted_frames,
            status=status,
        )

    def update(self, range_fft: np.ndarray) -> AngularCalibrationPayload:
        cube = np.asarray(range_fft)
        expected_chirps = int(self.config.loops_per_frame) * int(
            self.config.chirps_per_loop
        )
        expected_shape = (
            expected_chirps,
            int(self.config.rx_channels),
            int(self.config.adc_samples),
        )
        if cube.shape != expected_shape:
            raise ValueError(
                f"Unexpected range FFT shape {cube.shape}; expected {expected_shape}"
            )
        loops = cube.reshape(
            int(self.config.loops_per_frame),
            int(self.config.chirps_per_loop),
            int(self.config.rx_channels),
            int(self.config.adc_samples),
        )
        window = (
            np.hanning(loops.shape[0]).astype(np.float64)
            if loops.shape[0] > 1
            else np.ones(1, dtype=np.float64)
        )
        if not np.any(window):
            window = np.ones(loops.shape[0], dtype=np.float64)
        zero_doppler = np.tensordot(window / window.sum(), loops, axes=(0, 0))
        combined = np.sqrt(np.sum(np.abs(zero_doppler) ** 2, axis=(0, 1)))
        scale = max(float(np.max(combined)), np.finfo(float).tiny)
        range_profile_db = 20.0 * np.log10(
            np.maximum(combined / scale, np.finfo(float).tiny)
        )
        if self._warmup_seen < self.settings.warmup_frames:
            self._warmup_seen += 1
            return self._payload(
                range_profile_db,
                None,
                f"Warm-up frame {self._warmup_seen}/{self.settings.warmup_frames}",
            )

        minimum = self.settings.target_distance_m - self.settings.search_window_m
        maximum = self.settings.target_distance_m + self.settings.search_window_m
        candidates = np.flatnonzero(
            (self.range_axis_m >= minimum) & (self.range_axis_m <= maximum)
        )
        if candidates.size < 3:
            return self._payload(
                range_profile_db, None, "Search window contains fewer than three bins"
            )
        local_index = int(np.argmax(combined[candidates]))
        peak_index = int(candidates[local_index])
        if (
            peak_index <= candidates[0]
            or peak_index >= candidates[-1]
            or peak_index <= 0
            or peak_index >= combined.size - 1
        ):
            return self._payload(
                range_profile_db, None, "Peak is on the search-window boundary"
            )
        background_mask = np.ones(candidates.size, dtype=bool)
        background_mask[
            max(0, local_index - 1) : min(candidates.size, local_index + 2)
        ] = False
        background_values = combined[candidates][background_mask]
        background = (
            float(np.median(background_values)) if background_values.size else 0.0
        )
        prominence_db = 20.0 * math.log10(
            max(float(combined[peak_index]), np.finfo(float).tiny)
            / max(background, np.finfo(float).tiny)
        )
        if prominence_db < self.settings.min_peak_prominence_db:
            return self._payload(
                range_profile_db,
                None,
                f"Target peak prominence {prominence_db:.1f} dB is too low",
            )

        left, centre, right = (
            float(combined[index])
            for index in (peak_index - 1, peak_index, peak_index + 1)
        )
        denominator = left - 2.0 * centre + right
        range_offset = (
            0.0
            if abs(denominator) < np.finfo(float).eps
            else 0.5 * (left - right) / denominator
        )
        range_offset = float(np.clip(range_offset, -0.5, 0.5))
        range_step = float(self.range_axis_m[1] - self.range_axis_m[0])
        measured_range = float(
            self.range_axis_m[peak_index] + range_offset * range_step
        )

        # Reuse the runtime compensated ODS geometry, Capon beamformer,
        # Doppler FFT, PMM folding, and angle grid. Calibration deliberately
        # averages stable reflector estimates rather than using tracking's
        # target-free background and temporal path state.
        angles, angle_scores = capon_pmm_angle_scores(
            cube,
            peak_index,
            self.config,
            angle_limit_deg=self.pmm_config.angle_limit_deg,
            angle_step_deg=self.pmm_config.angle_step_deg,
            folding_size_min=self.pmm_config.folding_size_min,
            folding_size_max=self.pmm_config.folding_size_max,
        )
        angle_profile = (
            np.max(angle_scores, axis=0)
            if self.settings.calibration_type == "azimuth"
            else np.max(angle_scores, axis=1)
        )
        angle_peak_index = int(np.argmax(angle_profile))
        if angle_peak_index <= 0 or angle_peak_index >= angles.size - 1:
            return self._payload(
                range_profile_db, None, "Angular peak is on the search-grid boundary"
            )
        measured_angle = float(angles[angle_peak_index])
        self._accepted_angles.append(measured_angle)
        self._accepted_ranges.append(measured_range)
        angle_scale = max(float(np.max(angle_profile)), np.finfo(float).tiny)
        angle_profile_db = 20.0 * np.log10(
            np.maximum(angle_profile / angle_scale, np.finfo(float).tiny)
        )
        if len(self._accepted_angles) < self.settings.accepted_frames:
            return self._payload(
                range_profile_db,
                angle_profile_db,
                f"Accepted frame {len(self._accepted_angles)}/{self.settings.accepted_frames}",
                measured_angle,
            )
        angle_std = float(np.std(np.asarray(self._accepted_angles)))
        if angle_std > self.settings.max_angle_std_deg:
            return self._payload(
                range_profile_db,
                angle_profile_db,
                f"Waiting for stability: angle sigma={angle_std:.2f} deg",
                measured_angle,
            )
        mean_angle = float(np.mean(np.asarray(self._accepted_angles)))
        self.result = AngularCalibrationResult(
            calibration_type=self.settings.calibration_type,
            target_distance_m=self.settings.target_distance_m,
            search_window_m=self.settings.search_window_m,
            reference_angle_deg=self.settings.reference_angle_deg,
            measured_angle_deg=mean_angle,
            angle_bias_deg=mean_angle - self.settings.reference_angle_deg,
            accepted_frames=len(self._accepted_angles),
            angle_std_deg=angle_std,
            measured_range_m=float(np.mean(np.asarray(self._accepted_ranges))),
            tx_order=self.tx_masks,
        )
        return self._payload(
            range_profile_db,
            angle_profile_db,
            "Stable angular calibration result ready",
            mean_angle,
        )


def create_calibration_accumulator(
    radar_config: Any,
    settings: CalibrationSettings,
    pmm_config: Optional[PmmConfig] = None,
) -> CalibrationAccumulator | AngularCalibrationAccumulator:
    if settings.calibration_type == "range":
        return CalibrationAccumulator(radar_config, settings)
    return AngularCalibrationAccumulator(radar_config, settings, pmm_config)


def run_calibration_display(
    display_queue: Any,
    stop_event: Any,
    startup_status_queue: Any = None,
    mode: str = CALIBRATION_DISPLAY_MODE,
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
    angular_mode = mode in {
        AZIMUTH_CALIBRATION_DISPLAY_MODE,
        ELEVATION_CALIBRATION_DISPLAY_MODE,
    }
    channel_plot = pg.PlotWidget(
        title=(
            f"{mode.removesuffix('-calibration').title()} angle spectrum"
            if angular_mode
            else "TX-major/RX-minor channel correction"
        )
    )
    channel_plot.setLabel(
        "bottom",
        "Angle (degrees)"
        if angular_mode
        else "Physical channel (TX-major / RX-minor)",
    )
    channel_plot.setLabel("left", "Magnitude / phase")
    magnitude_curve = channel_plot.plot(pen="g", symbol="o", name="Magnitude")
    phase_curve = channel_plot.plot(pen="m", symbol="t", name="Phase (deg)")
    angle_reference_line = pg.InfiniteLine(
        angle=90, movable=False, pen=pg.mkPen("g", width=2)
    )
    angle_measured_line = pg.InfiniteLine(
        angle=90, movable=False, pen=pg.mkPen("r", width=2)
    )
    channel_plot.addItem(angle_reference_line)
    channel_plot.addItem(angle_measured_line)
    angle_reference_line.setVisible(angular_mode)
    angle_measured_line.setVisible(False)
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
        elif isinstance(latest, AngularCalibrationPayload):
            profile_curve.setData(latest.range_axis_m, latest.range_profile_db)
            search_low.setValue(latest.search_min_m)
            search_high.setValue(latest.search_max_m)
            peak_line.setVisible(False)
            magnitude_curve.setData(latest.angle_axis_deg, latest.angle_profile_db)
            phase_curve.setData([], [])
            angle_reference_line.setValue(latest.reference_angle_deg)
            angle_measured_line.setVisible(latest.measured_angle_deg is not None)
            if latest.measured_angle_deg is not None:
                angle_measured_line.setValue(latest.measured_angle_deg)
            measured_text = (
                ""
                if latest.measured_angle_deg is None
                else (
                    f", measured {latest.measured_angle_deg:+.2f} deg, "
                    f"bias {latest.angle_bias_deg:+.2f} deg"
                )
            )
            stability_text = (
                ""
                if latest.angle_std_deg is None
                else f", angle sigma {latest.angle_std_deg:.2f} deg"
            )
            status_label.setText(
                f"{latest.status} - {latest.accepted_frames}/{latest.required_frames}"
                f"{measured_text}{stability_text}"
            )
        if stop_event.is_set():
            window.close()

    timer = QtCore.QTimer()
    timer.timeout.connect(poll)
    timer.start(50)
    app.exec()
