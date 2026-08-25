from __future__ import annotations

import hashlib
import json
from collections import deque
from dataclasses import asdict, dataclass
from typing import Any, Literal, Optional, Protocol

import numpy as np
from scipy import fft as scipy_fft


SPEED_OF_LIGHT_M_PER_S = 299_792_458.0
MINI4_FEATURE_VERSION = "mini4-pmm-tracking-v8"
MINI4_NUM_ADC_SAMPLES = 256
MINI4_NUM_RX_CHANNELS = 4
MINI4_NUM_LOOPS = 32
MINI4_NUM_TX = 3
MINI4_NUM_CHIRPS = MINI4_NUM_LOOPS * MINI4_NUM_TX
MINI4_SAMPLE_RATE_KSPS = 6250.0
MINI4_SLOPE_MHZ_PER_US = 46.875
MINI4_START_FREQUENCY_GHZ = 60.25
MINI4_IDLE_TIME_US = 850.0
MINI4_ADC_START_TIME_US = 6.0
MINI4_RAMP_END_TIME_US = 50.0
MINI4_FRAME_PERIODICITY_MS = 100.0
MINI4_DOPPLER_FFT_SIZE = 64
MINI4_MIN_RANGE_M = 0.3
MINI4_MAX_RANGE_M = 20.0
MINI4_DEFAULT_ADAPTIVE_THRESHOLD_SIGMA = 6.0
MINI4_DEFAULT_ADAPTIVE_THRESHOLD_MINIMUM = 700.0

TrackState = Literal[
    "calibrating",
    "searching",
    "tentative",
    "confirmed",
    "coasting",
    "lost",
]


class PmmRadarConfig(Protocol):
    num_adc_samples: int
    num_rx_channels: int
    num_chirps_per_frame: int
    num_loops: Optional[int]
    num_chirps_per_loop: Optional[int]
    tx_channel_masks: Optional[tuple[int, ...]]
    sample_rate_ksps: Optional[float]
    frequency_slope_mhz_per_us: Optional[float]
    start_frequency_ghz: Optional[float]
    idle_time_us: Optional[float]
    adc_start_time_us: Optional[float]
    ramp_end_time_us: Optional[float]
    frame_periodicity_ms: Optional[float]
    rx_channel_compensation: Optional[tuple[complex, ...]]
    azimuth_bias_deg: float
    elevation_bias_deg: float


@dataclass(frozen=True)
class PmmConfig:
    background_calibration_seconds: float = 30.0
    maximum_target_speed_m_s: float = 4.0
    folding_size_min: int = 2
    folding_size_max: int = 32
    detection_threshold: Optional[float] = None
    adaptive_threshold_sigma: float = MINI4_DEFAULT_ADAPTIVE_THRESHOLD_SIGMA
    adaptive_threshold_minimum: float = MINI4_DEFAULT_ADAPTIVE_THRESHOLD_MINIMUM
    history_seconds: float = 3.6
    provisional_frames: int = 5
    confirmation_window_frames: int = 10
    confirmation_hits: int = 7
    coast_frames: int = 10
    particle_count: int = 5_000
    min_range_m: float = MINI4_MIN_RANGE_M
    max_range_m: float = MINI4_MAX_RANGE_M
    angle_limit_deg: float = 60.0
    angle_step_deg: float = 2.0
    random_seed: int = 6843

    def __post_init__(self) -> None:
        if (
            not np.isfinite(self.background_calibration_seconds)
            or self.background_calibration_seconds <= 0.0
        ):
            raise ValueError(
                "PMM background calibration duration must be finite and positive"
            )
        if (
            not np.isfinite(self.maximum_target_speed_m_s)
            or self.maximum_target_speed_m_s <= 0.0
        ):
            raise ValueError("PMM maximum target speed must be finite and positive")
        if not (
            2
            <= self.folding_size_min
            <= self.folding_size_max
            <= MINI4_DOPPLER_FFT_SIZE // 2
        ):
            raise ValueError(
                "PMM folding sizes must be increasing, start at 2 or later, "
                "and retain at least two folding rows"
            )
        if (
            self.detection_threshold is not None
            and (
                not np.isfinite(self.detection_threshold)
                or self.detection_threshold < 0.0
            )
        ):
            raise ValueError("PMM detection threshold must be finite and non-negative")
        if (
            not np.isfinite(self.adaptive_threshold_sigma)
            or self.adaptive_threshold_sigma <= 0.0
        ):
            raise ValueError("PMM adaptive threshold sigma must be finite and positive")
        if (
            not np.isfinite(self.adaptive_threshold_minimum)
            or self.adaptive_threshold_minimum < 0.0
        ):
            raise ValueError(
                "PMM adaptive threshold minimum must be finite and non-negative"
            )
        if not np.isfinite(self.history_seconds) or self.history_seconds <= 0.0:
            raise ValueError("PMM history duration must be finite and positive")
        if self.provisional_frames < 1:
            raise ValueError("PMM provisional frame count must be positive")
        if self.confirmation_window_frames < self.provisional_frames:
            raise ValueError("PMM confirmation window is shorter than provisional history")
        if not 1 <= self.confirmation_hits <= self.confirmation_window_frames:
            raise ValueError("PMM confirmation hits are outside the confirmation window")
        if self.coast_frames < 0:
            raise ValueError("PMM coast frame count cannot be negative")
        if self.particle_count < 1:
            raise ValueError("PMM particle count must be positive")
        if (
            not np.isfinite(self.min_range_m)
            or not np.isfinite(self.max_range_m)
            or not 0.0 <= self.min_range_m < self.max_range_m
        ):
            raise ValueError("PMM range bounds are invalid")
        if (
            not np.isfinite(self.angle_limit_deg)
            or not 0.0 < self.angle_limit_deg <= 90.0
        ):
            raise ValueError("PMM angle limit must be in (0, 90]")
        if not np.isfinite(self.angle_step_deg) or self.angle_step_deg <= 0.0:
            raise ValueError("PMM angle step must be finite and positive")


@dataclass(frozen=True)
class PmmTrackResult:
    state: TrackState
    label: str
    calibration_frames_seen: int
    calibration_frames_required: int
    history_frames: int
    range_bin: Optional[int]
    range_m: Optional[float]
    radial_velocity_m_s: Optional[float]
    azimuth_deg: Optional[float]
    elevation_deg: Optional[float]
    raw_pmm_score: Optional[float]
    pmm_score: Optional[float]
    folding_size: Optional[int]
    background_projection_gain: Optional[float]
    azimuth_background_projection_gain: Optional[float]
    elevation_background_projection_gain: Optional[float]
    threshold: float
    age_frames: int
    hits: int
    misses: int
    predicted: bool
    dp_transition_bins: int
    dp_path_score: Optional[float]
    particle_count: int
    processing_ms: Optional[dict[str, float]] = None

    @property
    def has_track(self) -> bool:
        return (
            self.state in {"tentative", "confirmed", "coasting"}
            and self.range_m is not None
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def mini4_profile_fingerprint(config: PmmRadarConfig) -> str:
    values = {
        "feature_version": MINI4_FEATURE_VERSION,
        "num_adc_samples": int(config.num_adc_samples),
        "num_rx_channels": int(config.num_rx_channels),
        "num_chirps_per_frame": int(config.num_chirps_per_frame),
        "num_loops": int(config.num_loops or 0),
        "num_chirps_per_loop": int(config.num_chirps_per_loop or 0),
        "tx_channel_masks": list(config.tx_channel_masks or ()),
        "sample_rate_ksps": float(config.sample_rate_ksps or 0.0),
        "frequency_slope_mhz_per_us": float(
            config.frequency_slope_mhz_per_us or 0.0
        ),
        "start_frequency_ghz": float(config.start_frequency_ghz or 0.0),
        "idle_time_us": float(config.idle_time_us or 0.0),
        "adc_start_time_us": float(config.adc_start_time_us or 0.0),
        "ramp_end_time_us": float(config.ramp_end_time_us or 0.0),
        "frame_periodicity_ms": float(config.frame_periodicity_ms or 0.0),
        "doppler_fft_size": MINI4_DOPPLER_FFT_SIZE,
    }
    serialized = json.dumps(values, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(serialized).hexdigest()


def validate_mini4_profile(config: PmmRadarConfig) -> None:
    exact_dimensions = (
        int(config.num_adc_samples),
        int(config.num_rx_channels),
        int(config.num_chirps_per_frame),
        int(config.num_loops or 0),
        int(config.num_chirps_per_loop or 0),
    )
    expected_dimensions = (
        MINI4_NUM_ADC_SAMPLES,
        MINI4_NUM_RX_CHANNELS,
        MINI4_NUM_CHIRPS,
        MINI4_NUM_LOOPS,
        MINI4_NUM_TX,
    )
    if exact_dimensions != expected_dimensions:
        raise ValueError(
            "Mini4-20m profile dimensions mismatch: "
            f"expected={expected_dimensions}, observed={exact_dimensions}"
        )
    if tuple(config.tx_channel_masks or ()) != (1, 4, 2):
        raise ValueError(
            "Mini4-20m profile requires the ODS TX schedule (1, 4, 2)"
        )
    if int(getattr(config, "lvds_lanes", 2)) != 2:
        raise ValueError("Mini4-20m profile requires two LVDS lanes")
    if not bool(getattr(config, "iq_swap", True)):
        raise ValueError("Mini4-20m profile requires IQ swap enabled")
    if bool(getattr(config, "channel_interleave", False)):
        raise ValueError("Mini4-20m profile requires non-interleaved RX channels")
    range_bias_m = float(getattr(config, "range_bias_m", 0.0))
    if not np.isfinite(range_bias_m):
        raise ValueError("Mini4-20m profile range bias must be finite")
    for field in ("azimuth_bias_deg", "elevation_bias_deg"):
        value = float(getattr(config, field, 0.0))
        if not np.isfinite(value):
            raise ValueError(f"Mini4-20m profile {field} must be finite")
    compensation = getattr(config, "rx_channel_compensation", None)
    if compensation is not None:
        coefficients = np.asarray(compensation, dtype=np.complex64)
        if coefficients.shape != (12,) or not np.isfinite(coefficients).all():
            raise ValueError(
                "Mini4-20m profile RX compensation must contain 12 finite values"
            )
    expected_scalars = {
        "sample_rate_ksps": MINI4_SAMPLE_RATE_KSPS,
        "frequency_slope_mhz_per_us": MINI4_SLOPE_MHZ_PER_US,
        "start_frequency_ghz": MINI4_START_FREQUENCY_GHZ,
        "idle_time_us": MINI4_IDLE_TIME_US,
        "adc_start_time_us": MINI4_ADC_START_TIME_US,
        "ramp_end_time_us": MINI4_RAMP_END_TIME_US,
        "frame_periodicity_ms": MINI4_FRAME_PERIODICITY_MS,
    }
    for field, expected in expected_scalars.items():
        observed = getattr(config, field)
        if observed is None or not np.isclose(float(observed), expected, atol=1e-3):
            raise ValueError(
                f"Mini4-20m profile {field} mismatch: "
                f"expected={expected}, observed={observed}"
            )


def spectrum_folding(
    spectra: np.ndarray,
    folding_size_min: int = 2,
    folding_size_max: int = 32,
) -> tuple[np.ndarray, np.ndarray]:
    """Return mmHawkeye folding scores and winning sizes along the last axis."""
    values = np.asarray(spectra, dtype=np.float32)
    if values.ndim < 1 or values.shape[-1] < folding_size_min:
        raise ValueError("PMM spectra are too short for spectrum folding")
    if not np.isfinite(values).all():
        raise ValueError("PMM spectra contain non-finite values")
    if not 2 <= folding_size_min <= folding_size_max:
        raise ValueError("Invalid PMM folding-size interval")

    flattened = values.reshape(-1, values.shape[-1])
    best_scores = np.full(flattened.shape[0], -np.inf, dtype=np.float32)
    best_sizes = np.full(flattened.shape[0], folding_size_min, dtype=np.int16)

    for folding_size in range(folding_size_min, folding_size_max + 1):
        if folding_size > flattened.shape[1]:
            break
        row_count = flattened.shape[1] // folding_size
        usable_bins = row_count * folding_size
        folded = flattened[:, :usable_bins].reshape(
            flattened.shape[0],
            row_count,
            folding_size,
        )
        score = np.max(folded.mean(axis=1), axis=1)
        improved = score > best_scores
        best_scores[improved] = score[improved]
        best_sizes[improved] = folding_size

    output_shape = values.shape[:-1]
    return best_scores.reshape(output_shape), best_sizes.reshape(output_shape)


def spectral_subtraction(
    measured: np.ndarray,
    background: np.ndarray,
) -> tuple[np.ndarray, float]:
    measured_values = np.asarray(measured, dtype=np.float32)
    background_values = np.asarray(background, dtype=np.float32)
    if measured_values.shape != background_values.shape:
        raise ValueError("PMM background shape does not match the measurement")
    denominator = float(np.dot(background_values, background_values))
    gain = (
        float(np.dot(background_values, measured_values)) / denominator
        if denominator > np.finfo(np.float32).eps
        else 0.0
    )
    corrected = (measured_values - gain * background_values).astype(np.float32)
    return corrected, gain


def constrained_maximum_path(
    scores: np.ndarray,
    maximum_step_bins: int | np.ndarray | list[int] | tuple[int, ...],
) -> tuple[np.ndarray, float]:
    """Find the maximum cumulative path with bounded adjacent-bin movement."""
    values = np.asarray(scores, dtype=np.float32)
    if values.ndim != 2 or values.shape[0] == 0 or values.shape[1] == 0:
        raise ValueError("DP scores must have shape [time, bins]")
    if not np.isfinite(values).all():
        raise ValueError("DP scores contain non-finite values")

    time_count, bin_count = values.shape
    if np.isscalar(maximum_step_bins):
        transitions = np.full(
            max(time_count - 1, 0),
            int(maximum_step_bins),
            dtype=np.int32,
        )
    else:
        transitions = np.asarray(maximum_step_bins, dtype=np.int32)
        if transitions.shape != (max(time_count - 1, 0),):
            raise ValueError("DP transition history does not match score history")
    if np.any(transitions < 0):
        raise ValueError("DP maximum step cannot be negative")
    cumulative = np.empty_like(values)
    parents = np.zeros((time_count, bin_count), dtype=np.int32)
    cumulative[0] = values[0]
    parents[0] = np.arange(bin_count, dtype=np.int32)

    for time_index in range(1, time_count):
        previous = cumulative[time_index - 1]
        maximum_step = int(transitions[time_index - 1])
        if maximum_step == 0:
            parent_indices = np.arange(bin_count, dtype=np.int32)
        else:
            padded = np.pad(
                previous,
                (maximum_step, maximum_step),
                constant_values=-np.inf,
            )
            windows = np.lib.stride_tricks.sliding_window_view(
                padded,
                2 * maximum_step + 1,
            )
            offsets = np.argmax(windows, axis=1).astype(np.int32)
            parent_indices = (
                np.arange(bin_count, dtype=np.int32)
                + offsets
                - maximum_step
            )
        parents[time_index] = parent_indices
        cumulative[time_index] = (
            previous[parent_indices] + values[time_index]
        )

    path = np.empty(time_count, dtype=np.int32)
    path[-1] = int(np.argmax(cumulative[-1]))
    for time_index in range(time_count - 1, 0, -1):
        path[time_index - 1] = parents[time_index, path[time_index]]
    return path, float(cumulative[-1, path[-1]])


class ParticleFilter1D:
    """Constant-velocity one-dimensional particle filter."""

    def __init__(
        self,
        particle_count: int,
        value_bounds: tuple[float, float],
        velocity_bounds: tuple[float, float],
        *,
        process_value_std: float,
        process_velocity_std: float,
        observation_std: float,
        seed: int,
    ) -> None:
        self.particle_count = max(int(particle_count), 1)
        self.value_bounds = tuple(float(value) for value in value_bounds)
        self.velocity_bounds = tuple(float(value) for value in velocity_bounds)
        self.process_value_std = max(float(process_value_std), 1e-6)
        self.process_velocity_std = max(float(process_velocity_std), 1e-6)
        self.observation_std = max(float(observation_std), 1e-6)
        self.rng = np.random.default_rng(seed)
        self.particles = np.empty((self.particle_count, 2), dtype=np.float64)
        self.weights = np.full(
            self.particle_count,
            1.0 / self.particle_count,
            dtype=np.float64,
        )
        self.initialized = False

    def initialize(self, observation: float) -> None:
        low, high = self.value_bounds
        self.particles[:, 0] = self.rng.uniform(low, high, self.particle_count)
        self.particles[:, 1] = self.rng.uniform(
            self.velocity_bounds[0],
            self.velocity_bounds[1],
            self.particle_count,
        )
        self.weights.fill(1.0 / self.particle_count)
        self.initialized = True

    def predict(self, delta_time_s: float) -> None:
        if not self.initialized:
            return
        delta_time_s = max(float(delta_time_s), 0.0)
        self.particles[:, 0] += (
            self.particles[:, 1] * delta_time_s
            + self.rng.normal(
                0.0,
                self.process_value_std,
                self.particle_count,
            )
        )
        self.particles[:, 1] += self.rng.normal(
            0.0,
            self.process_velocity_std,
            self.particle_count,
        )
        self.particles[:, 0] = np.clip(
            self.particles[:, 0],
            *self.value_bounds,
        )
        self.particles[:, 1] = np.clip(
            self.particles[:, 1],
            *self.velocity_bounds,
        )

    def update(self, observation: float) -> None:
        if not self.initialized:
            self.initialize(observation)
        residual = (self.particles[:, 0] - observation) / self.observation_std
        likelihood = np.exp(-0.5 * residual * residual)
        self.weights *= likelihood + np.finfo(np.float64).tiny
        total = float(self.weights.sum())
        if not np.isfinite(total) or total <= 0.0:
            self.weights.fill(1.0 / self.particle_count)
        else:
            self.weights /= total
        indices = self.rng.choice(
            self.particle_count,
            size=self.particle_count,
            replace=True,
            p=self.weights,
        )
        self.particles = self.particles[indices]
        self.weights.fill(1.0 / self.particle_count)

    @property
    def estimate(self) -> tuple[float, float]:
        if not self.initialized:
            raise RuntimeError("Particle filter has not been initialized")
        return (
            float(np.average(self.particles[:, 0], weights=self.weights)),
            float(np.average(self.particles[:, 1], weights=self.weights)),
        )

    def reset(self) -> None:
        self.initialized = False
        self.weights.fill(1.0 / self.particle_count)


def _ods_virtual_coordinates(
    config: PmmRadarConfig,
) -> tuple[np.ndarray, np.ndarray]:
    # IWR6843ISK-ODS virtual-array layout (coordinates in half wavelengths).
    positions = {
        (1, 1): (0.0, 3.0),
        (1, 2): (0.0, 2.0),
        (1, 3): (1.0, 2.0),
        (1, 4): (1.0, 3.0),
        (2, 1): (2.0, 3.0),
        (2, 2): (2.0, 2.0),
        (2, 3): (3.0, 2.0),
        (2, 4): (3.0, 3.0),
        (3, 1): (2.0, 1.0),
        (3, 2): (2.0, 0.0),
        (3, 3): (3.0, 0.0),
        (3, 4): (3.0, 1.0),
    }
    rx_phase_correction = (1.0, -1.0, -1.0, 1.0)
    configured = getattr(config, "rx_channel_compensation", None)
    measured = None
    if configured is not None and len(configured) == 12:
        candidate = np.asarray(configured, dtype=np.complex64)
        if np.all(np.isfinite(candidate)) and not np.allclose(candidate, 1.0 + 0.0j):
            measured = candidate
    coordinates: list[tuple[float, float]] = []
    phase_corrections: list[complex] = []
    for mask in tuple(config.tx_channel_masks or ()):
        enabled = [bit + 1 for bit in range(3) if mask & (1 << bit)]
        if len(enabled) != 1:
            raise ValueError("ODS PMM angle estimation requires one TX per chirp")
        tx_number = enabled[0]
        for rx_number in range(1, MINI4_NUM_RX_CHANNELS + 1):
            coordinates.append(positions[(tx_number, rx_number)])
            phase_corrections.append(
                complex(measured[(tx_number - 1) * 4 + rx_number - 1])
                if measured is not None
                else complex(rx_phase_correction[rx_number - 1])
            )
    return (
        np.asarray(coordinates, dtype=np.float64),
        np.asarray(phase_corrections, dtype=np.complex64),
    )


def capon_pmm_angle_scores(
    range_fft: np.ndarray,
    target_range_bin: int,
    config: PmmRadarConfig,
    *,
    angle_limit_deg: float,
    angle_step_deg: float,
    folding_size_min: int,
    folding_size_max: int,
    doppler_fft_size: int = MINI4_DOPPLER_FFT_SIZE,
) -> tuple[np.ndarray, np.ndarray]:
    """Return angle grid and 2-D PMM folding scores at one target range."""
    loops = int(config.num_loops or 0)
    tx_count = int(config.num_chirps_per_loop or 0)
    if loops < 2 or tx_count != MINI4_NUM_TX:
        raise ValueError("Capon PMM requires the Mini4 three-TX loop layout")
    if tuple(range_fft.shape[:2]) != (
        int(config.num_chirps_per_frame),
        int(config.num_rx_channels),
    ):
        raise ValueError("Range FFT shape is incompatible with PMM Capon input")
    if not 0 <= target_range_bin < range_fft.shape[-1]:
        raise ValueError("PMM target range bin is outside the range FFT")

    slow_time = np.asarray(
        range_fft[..., target_range_bin],
        dtype=np.complex64,
    ).reshape(loops, tx_count * int(config.num_rx_channels))
    coordinates, phase_corrections = _ods_virtual_coordinates(config)
    slow_time *= phase_corrections[np.newaxis, :]
    covariance = (slow_time.T @ slow_time.conj()) / max(loops, 1)
    diagonal_loading = max(
        float(np.trace(covariance).real) / covariance.shape[0] * 1e-3,
        1e-6,
    )
    inverse = np.linalg.pinv(
        covariance
        + np.eye(covariance.shape[0], dtype=np.complex64) * diagonal_loading,
        hermitian=True,
    )

    angles = np.arange(
        -angle_limit_deg,
        angle_limit_deg + angle_step_deg * 0.5,
        angle_step_deg,
        dtype=np.float32,
    )
    azimuth, elevation = np.meshgrid(angles, angles, indexing="xy")
    azimuth_rad = np.deg2rad(azimuth.ravel())
    elevation_rad = np.deg2rad(elevation.ravel())
    direction_x = np.sin(azimuth_rad) * np.cos(elevation_rad)
    # The ODS vertical antenna-coordinate direction is opposite the physical
    # display Z direction. Report positive elevation as upward.
    direction_z = -np.sin(elevation_rad)
    steering = np.exp(
        1j
        * np.pi
        * (
            coordinates[:, 0:1] * direction_x[np.newaxis, :]
            + coordinates[:, 1:2] * direction_z[np.newaxis, :]
        )
    )
    inverse_steering = inverse @ steering
    denominator = np.sum(steering.conj() * inverse_steering, axis=0)
    safe_denominator = np.where(
        np.abs(denominator) > np.finfo(np.float32).eps,
        denominator,
        np.finfo(np.float32).eps,
    )
    weights = inverse_steering / safe_denominator[np.newaxis, :]
    beamformed = slow_time @ weights.conj()
    window = np.hanning(loops).astype(np.float32)
    doppler = scipy_fft.fftshift(
        scipy_fft.fft(
            beamformed * window[:, np.newaxis],
            n=doppler_fft_size,
            axis=0,
        ),
        axes=0,
    )
    folded, _sizes = spectrum_folding(
        np.abs(doppler).T,
        folding_size_min,
        folding_size_max,
    )
    return angles, folded.reshape(angles.size, angles.size)


class PmmTracker:
    def __init__(self, radar_config: PmmRadarConfig, config: PmmConfig) -> None:
        validate_mini4_profile(radar_config)
        self.radar_config = radar_config
        self.config = config
        self.frame_period_s = float(
            radar_config.frame_periodicity_ms or MINI4_FRAME_PERIODICITY_MS
        ) * 1e-3
        self.calibration_frames_required = max(
            int(round(config.background_calibration_seconds / self.frame_period_s)),
            1,
        )
        self.history_frames_limit = max(
            int(round(config.history_seconds / self.frame_period_s)),
            config.provisional_frames,
        )
        self.calibration_sum: Optional[np.ndarray] = None
        self.calibration_score_samples: list[np.ndarray] = []
        self.background: Optional[np.ndarray] = None
        self.adaptive_thresholds: Optional[np.ndarray] = None
        self.angle_az_calibration_sum: Optional[np.ndarray] = None
        self.angle_el_calibration_sum: Optional[np.ndarray] = None
        self.angle_az_background: Optional[np.ndarray] = None
        self.angle_el_background: Optional[np.ndarray] = None
        self.angle_calibration_frames_seen = 0
        self.calibration_frames_seen = 0
        self.score_history: deque[np.ndarray] = deque(
            maxlen=self.history_frames_limit
        )
        self.folding_size_history: deque[np.ndarray] = deque(
            maxlen=self.history_frames_limit
        )
        self.transition_history: deque[int] = deque(
            maxlen=max(self.history_frames_limit - 1, 1)
        )
        self.doppler_history: deque[np.ndarray] = deque(
            maxlen=self.history_frames_limit
        )
        self.angle_az_history: deque[np.ndarray] = deque(
            maxlen=self.history_frames_limit
        )
        self.angle_el_history: deque[np.ndarray] = deque(
            maxlen=self.history_frames_limit
        )
        self.angle_transition_history: deque[int] = deque(
            maxlen=max(self.history_frames_limit - 1, 1)
        )
        self.angle_elapsed_s = 0.0
        self.evidence_history: deque[bool] = deque(
            maxlen=config.confirmation_window_frames
        )
        self.state: TrackState = "calibrating"
        self.age_frames = 0
        self.hits = 0
        self.misses = 0
        self.last_timestamp_s: Optional[float] = None
        self.range_indices: Optional[np.ndarray] = None
        self.range_axis: Optional[np.ndarray] = None
        self.dp_transition_bins = 0
        self.current_delta_time_s = self.frame_period_s
        range_resolution = (
            SPEED_OF_LIGHT_M_PER_S
            * MINI4_SAMPLE_RATE_KSPS
            * 1e3
            / (
                2.0
                * MINI4_SLOPE_MHZ_PER_US
                * 1e12
                * MINI4_NUM_ADC_SAMPLES
            )
        )
        self.range_filter = ParticleFilter1D(
            config.particle_count,
            (config.min_range_m, config.max_range_m),
            (
                -config.maximum_target_speed_m_s,
                config.maximum_target_speed_m_s,
            ),
            process_value_std=0.5 * range_resolution,
            process_velocity_std=0.25,
            observation_std=range_resolution,
            seed=config.random_seed,
        )
        self.azimuth_filter = ParticleFilter1D(
            config.particle_count,
            (-config.angle_limit_deg, config.angle_limit_deg),
            (-90.0, 90.0),
            process_value_std=0.5 * config.angle_step_deg,
            process_velocity_std=2.0,
            observation_std=config.angle_step_deg,
            seed=config.random_seed + 1,
        )
        self.elevation_filter = ParticleFilter1D(
            config.particle_count,
            (-config.angle_limit_deg, config.angle_limit_deg),
            (-90.0, 90.0),
            process_value_std=0.5 * config.angle_step_deg,
            process_velocity_std=2.0,
            observation_std=config.angle_step_deg,
            seed=config.random_seed + 2,
        )
        self.latest_result = self._empty_result("calibrating")

    @property
    def metadata(self) -> dict[str, Any]:
        feature_values = {
            "feature_version": MINI4_FEATURE_VERSION,
            "config": asdict(self.config),
            "doppler_fft_size": MINI4_DOPPLER_FFT_SIZE,
            "tracking_clutter_suppression": (
                "paper R-PMM background projection subtraction after folding"
            ),
            "detection_threshold": (
                "fixed override"
                if self.config.detection_threshold is not None
                else "frozen per-range median plus scaled MAD from initial calibration"
            ),
            "angle_clutter_suppression": (
                "paper A-PMM background projection subtraction after folding; "
                "target-free background sampled at each frame's strongest "
                "range-PMM bin"
            ),
            "identification_dc_removal": (
                "paper center-bin baseline subtraction before body-peak alignment"
            ),
        }
        feature_fingerprint = hashlib.sha256(
            json.dumps(
                feature_values,
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        ).hexdigest()
        return {
            "enabled": True,
            "label": "PMM target",
            "feature_version": MINI4_FEATURE_VERSION,
            "profile_fingerprint": mini4_profile_fingerprint(self.radar_config),
            "feature_fingerprint": feature_fingerprint,
            "doppler_fft_size": MINI4_DOPPLER_FFT_SIZE,
            "folding_score": "maximum column-averaged linear magnitude",
            "config": asdict(self.config),
        }

    @property
    def spectrogram_db(self) -> np.ndarray:
        if not self.doppler_history:
            return np.empty((MINI4_DOPPLER_FFT_SIZE, 0), dtype=np.float32)
        spectra = np.stack(tuple(self.doppler_history), axis=1)
        return (20.0 * np.log10(spectra + 1e-6)).astype(np.float32)

    def reset_tracking(self, *, lost: bool = False) -> None:
        self.score_history.clear()
        self.folding_size_history.clear()
        self.transition_history.clear()
        self.doppler_history.clear()
        self.angle_az_history.clear()
        self.angle_el_history.clear()
        self.angle_transition_history.clear()
        self.angle_elapsed_s = 0.0
        self.evidence_history.clear()
        self.range_filter.reset()
        self.azimuth_filter.reset()
        self.elevation_filter.reset()
        self.age_frames = 0
        self.hits = 0
        self.misses = 0
        self.state = "lost" if lost else (
            "searching" if self.background is not None else "calibrating"
        )

    def _prepare_range_axis(self, range_axis_m: np.ndarray) -> None:
        range_axis = np.asarray(range_axis_m, dtype=np.float64)
        if (
            range_axis.ndim != 1
            or range_axis.size != MINI4_NUM_ADC_SAMPLES
            or not np.isfinite(range_axis).all()
        ):
            raise ValueError("Mini4 PMM range axis is invalid")
        indices = np.flatnonzero(
            (range_axis >= self.config.min_range_m)
            & (range_axis <= self.config.max_range_m)
        )
        if indices.size < 2:
            raise ValueError("Mini4 PMM range gate contains too few bins")
        if self.range_axis is not None and not np.array_equal(range_axis, self.range_axis):
            self.reset_calibration()
        self.range_axis = range_axis
        self.range_indices = indices
        spacing = float(np.median(np.diff(range_axis[indices])))
        self.dp_transition_bins = max(
            int(
                np.ceil(
                    self.config.maximum_target_speed_m_s
                    * self.frame_period_s
                    / spacing
                )
            ),
            1,
        )

    def reset_calibration(self) -> None:
        self.calibration_sum = None
        self.calibration_score_samples.clear()
        self.background = None
        self.adaptive_thresholds = None
        self.angle_az_calibration_sum = None
        self.angle_el_calibration_sum = None
        self.angle_az_background = None
        self.angle_el_background = None
        self.angle_calibration_frames_seen = 0
        self.calibration_frames_seen = 0
        self.reset_tracking()
        self.state = "calibrating"

    def update(
        self,
        doppler_cube: np.ndarray,
        range_fft: np.ndarray,
        range_axis_m: np.ndarray,
        *,
        timestamp_s: Optional[float] = None,
    ) -> PmmTrackResult:
        self._prepare_range_axis(range_axis_m)
        expected_doppler_shape = (
            MINI4_DOPPLER_FFT_SIZE,
            MINI4_NUM_TX,
            MINI4_NUM_RX_CHANNELS,
            MINI4_NUM_ADC_SAMPLES,
        )
        if tuple(doppler_cube.shape) != expected_doppler_shape:
            self.reset_calibration()
            raise ValueError(
                "Mini4 PMM Doppler cube mismatch: "
                f"expected={expected_doppler_shape}, observed={doppler_cube.shape}"
            )
        if timestamp_s is not None:
            timestamp_s = float(timestamp_s)
            observed_interval_s = (
                timestamp_s - self.last_timestamp_s
                if self.last_timestamp_s is not None
                else self.frame_period_s
            )
            if (
                self.last_timestamp_s is not None
                and (
                    timestamp_s <= self.last_timestamp_s
                    or timestamp_s - self.last_timestamp_s
                    > 1.5 * self.frame_period_s
                )
            ):
                self.reset_tracking(lost=True)
                observed_interval_s = self.frame_period_s
            self.current_delta_time_s = max(observed_interval_s, 0.0)
            self.last_timestamp_s = timestamp_s
        else:
            self.current_delta_time_s = self.frame_period_s
        assert self.range_axis is not None
        spacing = float(np.median(np.diff(self.range_axis[self.range_indices])))
        self.dp_transition_bins = max(
            int(
                np.ceil(
                    self.config.maximum_target_speed_m_s
                    * self.current_delta_time_s
                    / spacing
                )
            ),
            1,
        )

        assert self.range_indices is not None
        magnitude = np.abs(doppler_cube).mean(axis=(1, 2)).T
        raw_scores, raw_sizes = spectrum_folding(
            magnitude,
            self.config.folding_size_min,
            self.config.folding_size_max,
        )
        gated_scores = raw_scores[self.range_indices]

        if self.background is None:
            if self.calibration_sum is None:
                self.calibration_sum = np.zeros_like(gated_scores, dtype=np.float64)
            self.calibration_sum += gated_scores
            self.calibration_score_samples.append(gated_scores.copy())
            calibration_local_bin = int(np.argmax(gated_scores))
            calibration_range_bin = int(
                self.range_indices[calibration_local_bin]
            )
            _, angle_scores = capon_pmm_angle_scores(
                range_fft,
                calibration_range_bin,
                self.radar_config,
                angle_limit_deg=self.config.angle_limit_deg,
                angle_step_deg=self.config.angle_step_deg,
                folding_size_min=self.config.folding_size_min,
                folding_size_max=self.config.folding_size_max,
            )
            angle_az_scores = angle_scores.max(axis=0)
            angle_el_scores = angle_scores.max(axis=1)
            if self.angle_az_calibration_sum is None:
                self.angle_az_calibration_sum = np.zeros_like(
                    angle_az_scores,
                    dtype=np.float64,
                )
                self.angle_el_calibration_sum = np.zeros_like(
                    angle_el_scores,
                    dtype=np.float64,
                )
            assert self.angle_el_calibration_sum is not None
            self.angle_az_calibration_sum += angle_az_scores
            self.angle_el_calibration_sum += angle_el_scores
            self.angle_calibration_frames_seen += 1
            self.calibration_frames_seen += 1
            if self.calibration_frames_seen >= self.calibration_frames_required:
                self.background = (
                    self.calibration_sum / self.calibration_frames_seen
                ).astype(np.float32)
                if self.config.detection_threshold is None:
                    calibration_residuals = np.stack(
                        [
                            spectral_subtraction(sample, self.background)[0]
                            for sample in self.calibration_score_samples
                        ],
                        axis=0,
                    )
                    residual_median = np.median(
                        calibration_residuals,
                        axis=0,
                    )
                    residual_mad = np.median(
                        np.abs(calibration_residuals - residual_median),
                        axis=0,
                    )
                    robust_sigma = 1.4826 * residual_mad
                    self.adaptive_thresholds = np.maximum(
                        residual_median
                        + self.config.adaptive_threshold_sigma * robust_sigma,
                        self.config.adaptive_threshold_minimum,
                    ).astype(np.float32)
                else:
                    self.adaptive_thresholds = np.full_like(
                        self.background,
                        self.config.detection_threshold,
                        dtype=np.float32,
                    )
                self.calibration_score_samples.clear()
                assert self.angle_az_calibration_sum is not None
                assert self.angle_el_calibration_sum is not None
                self.angle_az_background = (
                    self.angle_az_calibration_sum
                    / self.angle_calibration_frames_seen
                ).astype(np.float32)
                self.angle_el_background = (
                    self.angle_el_calibration_sum
                    / self.angle_calibration_frames_seen
                ).astype(np.float32)
                self.state = "searching"
            self.latest_result = self._empty_result(self.state)
            return self.latest_result

        corrected, background_gain = spectral_subtraction(
            gated_scores,
            self.background,
        )
        if self.adaptive_thresholds is None:
            raise RuntimeError("PMM adaptive thresholds are not calibrated")
        if self.config.detection_threshold is None:
            path_scores = corrected / self.adaptive_thresholds
        else:
            # A constant threshold does not change the relative path ranking.
            path_scores = corrected
        if self.score_history:
            self.transition_history.append(self.dp_transition_bins)
        self.score_history.append(path_scores)
        self.folding_size_history.append(raw_sizes[self.range_indices])
        if len(self.score_history) < self.config.provisional_frames:
            self.state = "searching"
            self.latest_result = self._empty_result("searching")
            return self.latest_result

        path, path_score = constrained_maximum_path(
            np.stack(tuple(self.score_history), axis=0),
            tuple(self.transition_history),
        )
        local_bin = int(path[-1])
        range_bin = int(self.range_indices[local_bin])
        measured_range_m = float(self.range_axis[range_bin])
        score = float(corrected[local_bin])
        threshold = float(self.adaptive_thresholds[local_bin])
        folding_size = int(raw_sizes[range_bin])
        detected = score >= threshold
        if detected and self.range_filter.initialized:
            previous_range_m = self.range_filter.estimate[0]
            ownership_gate_m = max(
                2.0
                * self.config.maximum_target_speed_m_s
                * self.current_delta_time_s,
                0.75,
            )
            if abs(measured_range_m - previous_range_m) > ownership_gate_m:
                self.reset_tracking(lost=True)
                self.latest_result = self._empty_result("lost")
                return self.latest_result
        self.evidence_history.append(detected)
        self.age_frames += 1

        self.range_filter.predict(self.current_delta_time_s)
        if detected:
            self.range_filter.update(measured_range_m)
            self.hits += 1
            self.misses = 0
            if self.state in {"searching", "lost"}:
                self.state = "tentative"
            elif self.state == "coasting":
                self.state = "confirmed"
            if (
                len(self.evidence_history)
                == self.config.confirmation_window_frames
                and sum(self.evidence_history) >= self.config.confirmation_hits
            ):
                self.state = "confirmed"
        else:
            self.misses += 1
            if self.state == "confirmed":
                self.state = "coasting"
            if (
                self.state == "coasting"
                and self.misses > self.config.coast_frames
            ):
                self.reset_tracking(lost=True)
                self.latest_result = self._empty_result("lost")
                return self.latest_result
            if (
                self.state == "tentative"
                and len(self.evidence_history)
                == self.config.confirmation_window_frames
                and sum(self.evidence_history) < self.config.confirmation_hits
            ):
                self.state = "searching"

        if self.range_filter.initialized:
            filtered_range_m, radial_velocity_m_s = self.range_filter.estimate
        else:
            filtered_range_m = measured_range_m
            radial_velocity_m_s = None

        # The paper's tracking and identification paths both start from the
        # non-demeaned Doppler spectrum. Identification applies its separate,
        # body-peak-aware center-bin subtraction to this rolling history.
        target_spectrum = magnitude[range_bin]
        self.doppler_history.append(target_spectrum.astype(np.float32))
        self.angle_elapsed_s += self.current_delta_time_s
        azimuth_deg: Optional[float] = None
        elevation_deg: Optional[float] = None
        azimuth_background_gain: Optional[float] = None
        elevation_background_gain: Optional[float] = None
        if detected:
            angles, angle_scores = capon_pmm_angle_scores(
                range_fft,
                range_bin,
                self.radar_config,
                angle_limit_deg=self.config.angle_limit_deg,
                angle_step_deg=self.config.angle_step_deg,
                folding_size_min=self.config.folding_size_min,
                folding_size_max=self.config.folding_size_max,
            )
            if (
                self.angle_az_background is None
                or self.angle_el_background is None
            ):
                raise RuntimeError("PMM angle background is not calibrated")
            corrected_azimuth, azimuth_background_gain = spectral_subtraction(
                angle_scores.max(axis=0),
                self.angle_az_background,
            )
            (
                corrected_elevation,
                elevation_background_gain,
            ) = spectral_subtraction(
                angle_scores.max(axis=1),
                self.angle_el_background,
            )
            self.angle_az_history.append(corrected_azimuth)
            self.angle_el_history.append(corrected_elevation)
            maximum_angle_step = max(
                int(
                    np.ceil(
                        np.degrees(
                            np.arctan2(
                                self.config.maximum_target_speed_m_s
                                * self.angle_elapsed_s,
                                max(filtered_range_m, self.config.min_range_m),
                            )
                        )
                        / self.config.angle_step_deg
                    )
                ),
                1,
            )
            if len(self.angle_az_history) > 1:
                self.angle_transition_history.append(maximum_angle_step)
            az_path, _ = constrained_maximum_path(
                np.stack(tuple(self.angle_az_history), axis=0),
                tuple(self.angle_transition_history),
            )
            el_path, _ = constrained_maximum_path(
                np.stack(tuple(self.angle_el_history), axis=0),
                tuple(self.angle_transition_history),
            )
            self.angle_elapsed_s = 0.0
            measured_azimuth = (
                float(angles[int(az_path[-1])])
                - float(getattr(self.radar_config, "azimuth_bias_deg", 0.0))
            )
            measured_elevation = (
                float(angles[int(el_path[-1])])
                - float(getattr(self.radar_config, "elevation_bias_deg", 0.0))
            )
            self.azimuth_filter.predict(self.current_delta_time_s)
            self.elevation_filter.predict(self.current_delta_time_s)
            self.azimuth_filter.update(measured_azimuth)
            self.elevation_filter.update(measured_elevation)
        else:
            self.azimuth_filter.predict(self.current_delta_time_s)
            self.elevation_filter.predict(self.current_delta_time_s)
        if self.azimuth_filter.initialized:
            azimuth_deg = self.azimuth_filter.estimate[0]
        if self.elevation_filter.initialized:
            elevation_deg = self.elevation_filter.estimate[0]

        self.latest_result = PmmTrackResult(
            state=self.state,
            label="PMM target",
            calibration_frames_seen=self.calibration_frames_seen,
            calibration_frames_required=self.calibration_frames_required,
            history_frames=len(self.score_history),
            range_bin=range_bin,
            range_m=filtered_range_m,
            radial_velocity_m_s=radial_velocity_m_s,
            azimuth_deg=azimuth_deg,
            elevation_deg=elevation_deg,
            raw_pmm_score=float(raw_scores[range_bin]),
            pmm_score=score,
            folding_size=folding_size,
            background_projection_gain=background_gain,
            azimuth_background_projection_gain=azimuth_background_gain,
            elevation_background_projection_gain=elevation_background_gain,
            threshold=threshold,
            age_frames=self.age_frames,
            hits=self.hits,
            misses=self.misses,
            predicted=not detected,
            dp_transition_bins=self.dp_transition_bins,
            dp_path_score=path_score,
            particle_count=self.config.particle_count,
        )
        return self.latest_result

    def _empty_result(self, state: TrackState) -> PmmTrackResult:
        return PmmTrackResult(
            state=state,
            label="PMM target",
            calibration_frames_seen=self.calibration_frames_seen,
            calibration_frames_required=self.calibration_frames_required,
            history_frames=len(self.score_history),
            range_bin=None,
            range_m=None,
            radial_velocity_m_s=None,
            azimuth_deg=None,
            elevation_deg=None,
            raw_pmm_score=None,
            pmm_score=None,
            folding_size=None,
            background_projection_gain=None,
            azimuth_background_projection_gain=None,
            elevation_background_projection_gain=None,
            threshold=(
                float(self.config.detection_threshold)
                if self.config.detection_threshold is not None
                else 0.0
            ),
            age_frames=self.age_frames,
            hits=self.hits,
            misses=self.misses,
            predicted=False,
            dp_transition_bins=self.dp_transition_bins,
            dp_path_score=None,
            particle_count=self.config.particle_count,
        )
