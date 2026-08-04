from collections import deque
from dataclasses import dataclass
from functools import lru_cache
from typing import Optional, Protocol

import numpy as np
from scipy import fft as scipy_fft
from scipy import ndimage as scipy_ndimage
from scipy import signal as scipy_signal

try:
    from .openradar_backend import (
        os_cfar_2d as openradar_os_cfar_2d,
        doppler_fft as openradar_doppler_fft,
        range_fft as openradar_range_fft,
        validate_openradar_backend,
    )
except ImportError:
    from openradar_backend import (
        os_cfar_2d as openradar_os_cfar_2d,
        doppler_fft as openradar_doppler_fft,
        range_fft as openradar_range_fft,
        validate_openradar_backend,
    )


SPEED_OF_LIGHT_M_PER_S = 299_792_458.0
DEFAULT_STATIC_ANGLE_FFT_SIZE = 32
DEFAULT_STATIC_DOPPLER_HALF_WIDTH_BINS = 1
DEFAULT_STATIC_MAX_POINTS = 256
DEFAULT_STATIC_WARMUP_FRAMES = 30
DEFAULT_STATIC_REFERENCE_FRAMES = 150
DEFAULT_STATIC_NOISE_SIGMA_MULTIPLIER = 4.0
DEFAULT_STATIC_SMOOTHING_RATE = 0.35
DEFAULT_STATIC_BACKGROUND_UPDATE_RATE = 0.0
DEFAULT_STATIC_REFERENCE_FLOOR_PERCENTILE = 20.0
DEFAULT_STATIC_MINIMUM_NOISE_DB = 1.0
DEFAULT_ROTOR_NOISE_GATE_MIN_DB = 3.0
DEFAULT_ROTOR_NOISE_GATE_MAX_DB = 15.0
DEFAULT_ROTOR_NOISE_SIGMA_MULTIPLIER = 3.0
DEFAULT_ROTOR_NOISE_SUPPORT_SHAPE = (3, 3)
DEFAULT_ROTOR_NOISE_MIN_SUPPORT = 3
ROBUST_GAUSSIAN_MAD_SCALE = 1.4826


class AdaptiveClutterMap:
    """Normalize range-Doppler power with a startup background map.

    The map is learned during an initial warm-up period. After warm-up, cells
    are frozen by default so persistent targets and environmental changes are
    not absorbed into the background. Optional post-warm-up adaptation protects
    cells around current detections. A shape change resets the map so profiles
    with different FFT dimensions cannot be mixed.
    """

    def __init__(
        self,
        *,
        update_rate: float = 0.02,
        warmup_frames: int = 30,
        minimum_snr_db: float = 3.0,
        range_protection_cells: int = 2,
        doppler_protection_cells: int = 1,
        adapt_after_warmup: bool = False,
    ) -> None:
        if not 0.0 < update_rate <= 1.0:
            raise ValueError("Clutter-map update rate must be in (0, 1]")
        if warmup_frames < 1:
            raise ValueError("Clutter-map warm-up must contain at least one frame")
        if not np.isfinite(minimum_snr_db) or minimum_snr_db < 0.0:
            raise ValueError("Clutter-map minimum SNR must be finite and non-negative")
        if range_protection_cells < 0 or doppler_protection_cells < 0:
            raise ValueError("Clutter-map protection sizes cannot be negative")

        self.update_rate = float(update_rate)
        self.warmup_frames = int(warmup_frames)
        self.minimum_snr_db = float(minimum_snr_db)
        self.range_protection_cells = int(range_protection_cells)
        self.doppler_protection_cells = int(doppler_protection_cells)
        self.adapt_after_warmup = bool(adapt_after_warmup)
        self._background_power: Optional[np.ndarray] = None
        self._frames_seen = 0

    @property
    def is_ready(self) -> bool:
        return self._frames_seen >= self.warmup_frames

    @property
    def frames_seen(self) -> int:
        return self._frames_seen

    def reset(self) -> None:
        self._background_power = None
        self._frames_seen = 0

    @property
    def minimum_snr_linear(self) -> float:
        return float(10.0 ** (self.minimum_snr_db / 10.0))

    def normalize(self, power_map: np.ndarray) -> np.ndarray:
        """Return target-to-background power ratio without modifying the map."""
        power = self._validated_power(power_map)
        self._ensure_shape(power.shape)
        if not self.is_ready:
            return np.zeros_like(power)

        assert self._background_power is not None
        positive_background = self._background_power[self._background_power > 0.0]
        background_floor = (
            max(float(np.median(positive_background)) * 1e-6, np.finfo(float).tiny)
            if positive_background.size
            else np.finfo(float).tiny
        )
        return power / np.maximum(self._background_power, background_floor)

    def update(
        self,
        power_map: np.ndarray,
        protected_detections: Optional[np.ndarray] = None,
    ) -> None:
        """Learn the map during warm-up and optionally adapt it afterward."""
        power = self._validated_power(power_map)
        self._ensure_shape(power.shape)
        assert self._background_power is not None

        if self.is_ready and not self.adapt_after_warmup:
            return

        if self._frames_seen == 0:
            self._background_power[...] = power
            self._frames_seen = 1
            return

        update_mask = np.ones(power.shape, dtype=bool)
        if self.is_ready and protected_detections is not None:
            protected = np.asarray(protected_detections, dtype=bool)
            if protected.shape != power.shape:
                raise ValueError(
                    "Protected detections must match the clutter-map shape"
                )
            update_mask &= ~self._expanded_protection_mask(protected)

        alpha = self.update_rate
        background = self._background_power
        background[update_mask] = (
            (1.0 - alpha) * background[update_mask]
            + alpha * power[update_mask]
        )
        self._frames_seen += 1

    @staticmethod
    def _validated_power(power_map: np.ndarray) -> np.ndarray:
        power = np.asarray(power_map, dtype=np.float64)
        if power.ndim != 2 or power.size == 0:
            raise ValueError("Clutter-map power must be a non-empty 2D array")
        return np.maximum(power, 0.0)

    def _ensure_shape(self, shape: tuple[int, ...]) -> None:
        if self._background_power is None or self._background_power.shape != shape:
            self._background_power = np.zeros(shape, dtype=np.float64)
            self._frames_seen = 0

    def _expanded_protection_mask(self, detections: np.ndarray) -> np.ndarray:
        doppler_size = 2 * self.doppler_protection_cells + 1
        range_size = 2 * self.range_protection_cells + 1
        structure = np.ones((doppler_size, range_size), dtype=bool)
        return scipy_ndimage.binary_dilation(detections, structure=structure)


class StaticSceneMap:
    """Learn a noise-aware adaptive reference for static-scene changes.

    Calibration records both the median power and normal per-cell variation.
    Detection uses log power with a per-range noise floor, removes common gain
    drift, and smooths changes over time. Unprotected cells adapt continuously;
    cells around a motion-qualified or validated target can be protected.
    """

    def __init__(
        self,
        *,
        warmup_frames: int = DEFAULT_STATIC_WARMUP_FRAMES,
        reference_frames: int = DEFAULT_STATIC_REFERENCE_FRAMES,
        minimum_change_db: float = 3.0,
        noise_sigma_multiplier: float = DEFAULT_STATIC_NOISE_SIGMA_MULTIPLIER,
        smoothing_rate: float = DEFAULT_STATIC_SMOOTHING_RATE,
        background_update_rate: float = DEFAULT_STATIC_BACKGROUND_UPDATE_RATE,
        reference_floor_percentile: float = (
            DEFAULT_STATIC_REFERENCE_FLOOR_PERCENTILE
        ),
        minimum_noise_db: float = DEFAULT_STATIC_MINIMUM_NOISE_DB,
    ) -> None:
        if warmup_frames < 0:
            raise ValueError("Static warm-up frames cannot be negative")
        if reference_frames < 1:
            raise ValueError("Static reference must contain at least one frame")
        if not np.isfinite(minimum_change_db) or minimum_change_db < 0.0:
            raise ValueError(
                "Static minimum change must be finite and non-negative"
            )
        if (
            not np.isfinite(noise_sigma_multiplier)
            or noise_sigma_multiplier < 0.0
        ):
            raise ValueError(
                "Static noise sigma multiplier must be finite and non-negative"
            )
        if not np.isfinite(smoothing_rate) or not 0.0 < smoothing_rate <= 1.0:
            raise ValueError("Static smoothing rate must be in (0, 1]")
        if (
            not np.isfinite(background_update_rate)
            or not 0.0 <= background_update_rate <= 1.0
        ):
            raise ValueError("Static background update rate must be in [0, 1]")
        if (
            not np.isfinite(reference_floor_percentile)
            or not 0.0 <= reference_floor_percentile <= 100.0
        ):
            raise ValueError(
                "Static reference floor percentile must be between 0 and 100"
            )
        if not np.isfinite(minimum_noise_db) or minimum_noise_db < 0.0:
            raise ValueError(
                "Static minimum noise must be finite and non-negative"
            )

        self.warmup_frames = int(warmup_frames)
        self.reference_frames = int(reference_frames)
        self.minimum_change_db = float(minimum_change_db)
        self.noise_sigma_multiplier = float(noise_sigma_multiplier)
        self.smoothing_rate = float(smoothing_rate)
        self.background_update_rate = float(background_update_rate)
        self.reference_floor_percentile = float(
            reference_floor_percentile
        )
        self.minimum_noise_db = float(minimum_noise_db)
        self._warmup_frames_seen = 0
        self._calibration_frames: list[np.ndarray] = []
        self._reference_power: Optional[np.ndarray] = None
        self._reference_log_power: Optional[np.ndarray] = None
        self._reference_floor: Optional[np.ndarray] = None
        self._noise_db: Optional[np.ndarray] = None
        self._noise_mad_db: Optional[np.ndarray] = None
        self._detection_threshold_db: Optional[np.ndarray] = None
        self._smoothed_change_db: Optional[np.ndarray] = None
        self._pending_log_power: Optional[np.ndarray] = None
        self._pending_change_db: Optional[np.ndarray] = None
        self._shape: Optional[tuple[int, ...]] = None

    @property
    def is_ready(self) -> bool:
        return self._reference_power is not None

    @property
    def frames_seen(self) -> int:
        return (
            self.reference_frames
            if self.is_ready
            else len(self._calibration_frames)
        )

    @property
    def warmup_frames_seen(self) -> int:
        return self._warmup_frames_seen

    @property
    def is_warming_up(self) -> bool:
        return self._warmup_frames_seen < self.warmup_frames

    @property
    def reference_shape(self) -> Optional[tuple[int, ...]]:
        return self._shape if self.is_ready else None

    def reset(self) -> None:
        self._warmup_frames_seen = 0
        self._calibration_frames.clear()
        self._reference_power = None
        self._reference_log_power = None
        self._reference_floor = None
        self._noise_db = None
        self._noise_mad_db = None
        self._detection_threshold_db = None
        self._smoothed_change_db = None
        self._pending_log_power = None
        self._pending_change_db = None
        self._shape = None

    def observe(self, power_cube: np.ndarray) -> Optional[np.ndarray]:
        """Calibrate or return smoothed change in dB against the scene map."""
        power = self._validated_power(power_cube)
        if self._shape != power.shape:
            self.reset()
            self._shape = power.shape

        if not self.is_ready:
            if self._warmup_frames_seen < self.warmup_frames:
                self._warmup_frames_seen += 1
                return None
            self._calibration_frames.append(
                power.astype(np.float32, copy=True)
            )
            if len(self._calibration_frames) >= self.reference_frames:
                self._finish_calibration()
            return None

        assert self._reference_power is not None
        assert self._reference_log_power is not None
        assert self._reference_floor is not None
        current_log_power = 10.0 * np.log10(
            np.maximum(power, self._reference_floor)
        )
        instantaneous_change_db = (
            current_log_power - self._reference_log_power
        )
        instantaneous_change_db -= np.median(
            instantaneous_change_db,
            axis=(-2, -1),
            keepdims=True,
        )

        if self._smoothed_change_db is None:
            self._smoothed_change_db = np.zeros_like(
                instantaneous_change_db,
                dtype=np.float32,
            )
        alpha = self.smoothing_rate
        self._smoothed_change_db *= 1.0 - alpha
        self._smoothed_change_db += alpha * instantaneous_change_db
        self._pending_log_power = current_log_power.astype(
            np.float32,
            copy=True,
        )
        self._pending_change_db = instantaneous_change_db.astype(
            np.float32,
            copy=True,
        )
        return self._smoothed_change_db.astype(np.float32, copy=True)

    def detection_mask(self, change_db: np.ndarray) -> np.ndarray:
        """Return changes that exceed both absolute and learned noise limits."""
        changes = np.asarray(change_db, dtype=np.float32)
        if (
            self._detection_threshold_db is None
            or changes.shape != self._detection_threshold_db.shape
        ):
            return np.zeros(changes.shape, dtype=bool)
        return changes >= self._detection_threshold_db

    def adapt(self, protected_cells: Optional[np.ndarray] = None) -> None:
        """Adapt unprotected background and noise cells after one observation."""
        if (
            self.background_update_rate <= 0.0
            or self._pending_log_power is None
            or self._pending_change_db is None
            or self._reference_log_power is None
            or self._noise_mad_db is None
        ):
            self._pending_log_power = None
            self._pending_change_db = None
            return

        update_mask = np.ones(self._pending_log_power.shape, dtype=bool)
        if protected_cells is not None:
            protected = np.asarray(protected_cells, dtype=bool)
            if protected.shape != update_mask.shape:
                raise ValueError(
                    "Static protected-cell mask must match the reference shape"
                )
            update_mask &= ~protected

        alpha = self.background_update_rate
        reference = self._reference_log_power
        reference[update_mask] = (
            (1.0 - alpha) * reference[update_mask]
            + alpha * self._pending_log_power[update_mask]
        )
        absolute_residual = np.abs(self._pending_change_db)
        noise_mad = self._noise_mad_db
        noise_mad[update_mask] = (
            (1.0 - alpha) * noise_mad[update_mask]
            + alpha * absolute_residual[update_mask]
        )
        assert self._noise_db is not None
        assert self._detection_threshold_db is not None
        self._noise_db[update_mask] = np.maximum(
            1.4826 * noise_mad[update_mask],
            self.minimum_noise_db,
        )
        self._detection_threshold_db[update_mask] = np.maximum(
            self.minimum_change_db,
            self.noise_sigma_multiplier * self._noise_db[update_mask],
        )
        self._pending_log_power = None
        self._pending_change_db = None

    def _finish_calibration(self) -> None:
        calibration = np.stack(self._calibration_frames, axis=0).astype(
            np.float32,
            copy=False,
        )
        reference_power = np.median(calibration, axis=0).astype(
            np.float32,
            copy=False,
        )
        positive_reference = reference_power[reference_power > 0.0]
        absolute_floor = (
            float(np.median(positive_reference)) * 1e-3
            if positive_reference.size
            else np.finfo(np.float32).tiny
        )
        range_floor = np.percentile(
            reference_power,
            self.reference_floor_percentile,
            axis=(-2, -1),
            keepdims=True,
        )
        reference_floor = np.maximum(
            range_floor,
            max(absolute_floor, np.finfo(np.float32).tiny),
        ).astype(np.float32, copy=False)
        calibration_log_power = 10.0 * np.log10(
            np.maximum(calibration, reference_floor[np.newaxis, ...])
        )
        reference_log_power = np.median(
            calibration_log_power,
            axis=0,
        )
        residual_db = calibration_log_power - reference_log_power
        residual_db -= np.median(
            residual_db,
            axis=(-2, -1),
            keepdims=True,
        )
        median_absolute_deviation = np.median(
            np.abs(residual_db),
            axis=0,
        )

        self._reference_power = reference_power
        self._reference_log_power = reference_log_power.astype(
            np.float32,
            copy=False,
        )
        self._reference_floor = reference_floor
        self._noise_mad_db = median_absolute_deviation.astype(
            np.float32,
            copy=False,
        )
        self._noise_db = np.maximum(
            1.4826 * median_absolute_deviation,
            self.minimum_noise_db,
        ).astype(np.float32, copy=False)
        self._detection_threshold_db = np.maximum(
            self.minimum_change_db,
            self.noise_sigma_multiplier * self._noise_db,
        ).astype(np.float32, copy=False)
        self._smoothed_change_db = np.zeros_like(
            reference_power,
            dtype=np.float32,
        )
        self._calibration_frames.clear()

    @staticmethod
    def _validated_power(power_cube: np.ndarray) -> np.ndarray:
        power = np.asarray(power_cube, dtype=np.float32)
        if power.ndim != 3 or power.size == 0:
            raise ValueError(
                "Static scene power must be a non-empty "
                "[range, elevation, azimuth] cube"
            )
        return np.maximum(power, 0.0)


class RadarDspConfig(Protocol):
    num_adc_samples: int
    num_rx_channels: int
    num_chirps_per_frame: int
    bytes_per_frame: int
    iq_swap: bool
    channel_interleave: bool
    lvds_lanes: int
    num_loops: Optional[int]
    num_chirps_per_loop: Optional[int]
    tx_channel_masks: Optional[tuple[int, ...]]
    sample_rate_ksps: Optional[float]
    frequency_slope_mhz_per_us: Optional[float]
    start_frequency_ghz: Optional[float]
    idle_time_us: Optional[float]
    ramp_end_time_us: Optional[float]
    frame_periodicity_ms: Optional[float]


def range_resolution_m(config: RadarDspConfig) -> Optional[float]:
    if not config.sample_rate_ksps or not config.frequency_slope_mhz_per_us:
        return None

    sample_rate_hz = config.sample_rate_ksps * 1_000.0
    slope_hz_per_s = config.frequency_slope_mhz_per_us * 1e12
    return SPEED_OF_LIGHT_M_PER_S * sample_rate_hz / (
        2.0 * slope_hz_per_s * config.num_adc_samples
    )


def range_axis_m(config: RadarDspConfig) -> Optional[np.ndarray]:
    resolution = range_resolution_m(config)
    if resolution is None:
        return None
    return np.arange(config.num_adc_samples, dtype=np.float32) * resolution


def compute_range_fft(radar_cube: np.ndarray) -> np.ndarray:
    return openradar_range_fft(radar_cube)


def compute_range_profile(range_fft: np.ndarray) -> np.ndarray:
    return np.abs(range_fft).mean(axis=(0, 1))


def compute_range_doppler_heatmap(
    range_fft: np.ndarray,
    config: RadarDspConfig,
) -> np.ndarray:
    doppler_fft = compute_range_doppler_fft(range_fft, config)
    magnitude = np.abs(doppler_fft).mean(axis=(1, 2))
    return 20.0 * np.log10(magnitude + 1e-6)


def compute_range_doppler_fft(
    range_fft: np.ndarray,
    config: RadarDspConfig,
) -> np.ndarray:
    loops = config.num_loops or config.num_chirps_per_frame
    chirps_per_loop = config.num_chirps_per_loop or 1
    if loops * chirps_per_loop != range_fft.shape[0]:
        chirps_per_loop = 1

    return openradar_doppler_fft(
        range_fft,
        num_tx_antennas=chirps_per_loop,
    )


def compute_point_cloud(
    range_fft: np.ndarray,
    range_axis: Optional[np.ndarray],
    config: RadarDspConfig,
    *,
    doppler_cube: Optional[np.ndarray] = None,
    clutter_map: Optional[AdaptiveClutterMap] = None,
    max_points: Optional[int] = None,
    false_alarm_rate: float = 1e-3,
    range_guard_cells: int = 2,
    doppler_guard_cells: int = 1,
    range_training_cells: int = 4,
    doppler_training_cells: int = 2,
    min_range_m: float = 0.25,
    max_range_m: float = 10.0,
    azimuth_fov_deg: float = 60.0,
    elevation_fov_deg: float = 60.0,
) -> np.ndarray:
    if max_points is not None and max_points <= 0:
        return np.empty((0, 4), dtype=np.float32)

    if doppler_cube is None:
        doppler_cube = compute_range_doppler_fft(range_fft, config)
    raw_detection_power = np.mean(np.abs(doppler_cube) ** 2, axis=(1, 2))
    if raw_detection_power.size == 0:
        return np.empty((0, 4), dtype=np.float32)
    detection_power = (
        clutter_map.normalize(raw_detection_power)
        if clutter_map is not None
        else raw_detection_power
    )

    detections = os_cfar_2d(
        detection_power,
        false_alarm_rate=false_alarm_rate,
        range_guard_cells=range_guard_cells,
        doppler_guard_cells=doppler_guard_cells,
        range_training_cells=range_training_cells,
        doppler_training_cells=doppler_training_cells,
    )
    if clutter_map is not None:
        detections &= detection_power >= clutter_map.minimum_snr_linear
    detections &= doppler_peak_mask(detection_power)
    detections = _apply_min_range_gate(detections, range_axis, min_range_m)
    if clutter_map is not None:
        clutter_map.update(raw_detection_power, detections)

    candidate_indices = np.argwhere(detections)
    if candidate_indices.size == 0:
        return np.empty((0, 4), dtype=np.float32)

    doppler_bins = candidate_indices[:, 0]
    range_bins = candidate_indices[:, 1]
    candidate_scores = detection_power[doppler_bins, range_bins]
    candidate_powers = raw_detection_power[doppler_bins, range_bins]

    # Unlimited displays do not need strongest-first ordering. Retain it only
    # when a finite point cap makes candidate priority observable.
    if max_points is not None:
        order = np.argsort(candidate_scores)[::-1]
        doppler_bins = doppler_bins[order]
        range_bins = range_bins[order]
        candidate_powers = candidate_powers[order]

    if range_axis is not None and range_axis.size == detection_power.shape[1]:
        target_ranges_m = np.asarray(range_axis[range_bins], dtype=np.float64)
    else:
        target_ranges_m = range_bins.astype(np.float64)

    if max_range_m > 0.0:
        in_range = target_ranges_m <= max_range_m
        doppler_bins = doppler_bins[in_range]
        range_bins = range_bins[in_range]
        candidate_powers = candidate_powers[in_range]
        target_ranges_m = target_ranges_m[in_range]
    if target_ranges_m.size == 0:
        return np.empty((0, 4), dtype=np.float32)

    virtual_samples = doppler_cube[doppler_bins, :, :, range_bins]
    xyz_m, angle_valid = estimate_xyz_from_virtual_arrays(
        virtual_samples,
        target_ranges_m,
        config,
    )
    valid = angle_valid & _points_are_within_fov(
        xyz_m,
        azimuth_fov_deg=azimuth_fov_deg,
        elevation_fov_deg=elevation_fov_deg,
    )
    xyz_m = xyz_m[valid]
    candidate_powers = candidate_powers[valid]
    if max_points is not None:
        xyz_m = xyz_m[:max_points]
        candidate_powers = candidate_powers[:max_points]
    if xyz_m.size == 0:
        return np.empty((0, 4), dtype=np.float32)

    magnitude_db = 10.0 * np.log10(candidate_powers + 1e-12)
    return np.column_stack((xyz_m, magnitude_db)).astype(np.float32, copy=False)


def compute_static_angle_power(
    doppler_cube: np.ndarray,
    config: RadarDspConfig,
    *,
    doppler_half_width_bins: int = DEFAULT_STATIC_DOPPLER_HALF_WIDTH_BINS,
    angle_fft_size: int = DEFAULT_STATIC_ANGLE_FFT_SIZE,
) -> np.ndarray:
    """Return near-zero-Doppler power as [range, elevation, azimuth]."""
    if doppler_cube.ndim != 4:
        raise ValueError("Doppler cube must have shape [doppler, tx, rx, range]")
    if doppler_cube.size == 0:
        return np.empty((0, 0, 0), dtype=np.float32)
    if angle_fft_size < 2:
        raise ValueError("Static angle FFT size must be at least two")

    doppler_bins, tx_count, rx_count, range_bins = doppler_cube.shape
    half_width = max(int(doppler_half_width_bins), 0)
    zero_bin = doppler_bins // 2
    start = max(zero_bin - half_width, 0)
    end = min(zero_bin + half_width + 1, doppler_bins)

    selected = np.moveaxis(
        doppler_cube[start:end],
        -1,
        1,
    ).reshape((-1, tx_count, rx_count))
    virtual_grids = build_virtual_antenna_grids(selected, config)
    # scipy.fft retains complex64 input precision and avoids NumPy's promotion
    # to complex128. Shift only the summed float32 power cube, not the three
    # larger complex Doppler-neighbor responses.
    angle_response = scipy_fft.fft2(
        virtual_grids,
        s=(angle_fft_size, angle_fft_size),
        axes=(-2, -1),
        workers=4,
    )
    angle_power = (
        angle_response.real * angle_response.real
        + angle_response.imag * angle_response.imag
    )
    angle_power = angle_power.reshape(
        (end - start, range_bins, angle_fft_size, angle_fft_size)
    ).sum(axis=0)
    angle_power = scipy_fft.fftshift(
        angle_power,
        axes=(-2, -1),
    )
    return angle_power.astype(np.float32, copy=False)


def compute_static_point_cloud(
    doppler_cube: np.ndarray,
    range_axis: Optional[np.ndarray],
    config: RadarDspConfig,
    scene_map: StaticSceneMap,
    *,
    max_points: int = DEFAULT_STATIC_MAX_POINTS,
    min_range_m: float = 0.25,
    max_range_m: float = 10.0,
    azimuth_fov_deg: float = 60.0,
    elevation_fov_deg: float = 60.0,
    doppler_half_width_bins: int = DEFAULT_STATIC_DOPPLER_HALF_WIDTH_BINS,
    angle_fft_size: int = DEFAULT_STATIC_ANGLE_FFT_SIZE,
) -> np.ndarray:
    """Return changed static reflectors as XYZ, magnitude dB, and change dB."""
    if max_points <= 0:
        return np.empty((0, 5), dtype=np.float32)

    power_cube = compute_static_angle_power(
        doppler_cube,
        config,
        doppler_half_width_bins=doppler_half_width_bins,
        angle_fft_size=angle_fft_size,
    )
    if power_cube.size == 0:
        return np.empty((0, 5), dtype=np.float32)
    change_db = scene_map.observe(power_cube)
    if change_db is None:
        return np.empty((0, 5), dtype=np.float32)

    range_bins = power_cube.shape[0]
    physical_range_axis = (
        np.asarray(range_axis, dtype=np.float64)
        if range_axis is not None and range_axis.size == range_bins
        else np.arange(range_bins, dtype=np.float64)
    )
    valid_ranges = physical_range_axis >= max(float(min_range_m), 0.0)
    if range_axis is None or range_axis.size != range_bins:
        valid_ranges[0] = False
    if max_range_m > 0.0:
        valid_ranges &= physical_range_axis <= max_range_m

    azimuth_u, elevation_u, valid_angles = _static_angle_geometry(
        angle_fft_size,
        float(azimuth_fov_deg),
        float(elevation_fov_deg),
    )

    detections = scene_map.detection_mask(change_db)
    detections &= valid_ranges[:, np.newaxis, np.newaxis]
    detections &= valid_angles[np.newaxis, :, :]
    candidate_indices = np.argwhere(detections)
    if candidate_indices.size == 0:
        return np.empty((0, 5), dtype=np.float32)
    candidate_indices = _local_maximum_candidate_indices(
        change_db,
        candidate_indices,
    )
    if candidate_indices.size == 0:
        return np.empty((0, 5), dtype=np.float32)

    candidate_changes = change_db[tuple(candidate_indices.T)]
    order = np.argsort(candidate_changes)[::-1][:max_points]
    candidate_indices = candidate_indices[order]
    candidate_changes = candidate_changes[order]

    selected_ranges = physical_range_axis[candidate_indices[:, 0]]
    selected_elevation_u = elevation_u[candidate_indices[:, 1]]
    selected_azimuth_u = azimuth_u[candidate_indices[:, 2]]
    radial_scale = np.sqrt(
        np.maximum(
            0.0,
            1.0 - selected_azimuth_u**2 - selected_elevation_u**2,
        )
    )
    xyz_m = np.column_stack(
        (
            selected_ranges * selected_azimuth_u,
            selected_ranges * radial_scale,
            -selected_ranges * selected_elevation_u,
        )
    )
    selected_power = power_cube[tuple(candidate_indices.T)]
    magnitude_db = 10.0 * np.log10(selected_power + 1e-12)
    return np.column_stack(
        (xyz_m, magnitude_db, candidate_changes)
    ).astype(np.float32, copy=False)


def static_target_protection_mask(
    shape: tuple[int, int, int],
    target_positions_m: np.ndarray,
    range_axis: Optional[np.ndarray],
    *,
    neighborhood_cells: int = 2,
) -> np.ndarray:
    """Return protected range-angle cells around validated XYZ targets."""
    protected = np.zeros(shape, dtype=bool)
    positions = np.asarray(target_positions_m, dtype=np.float64)
    if positions.size == 0:
        return protected
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError("Static protection targets must have shape [target, 3]")

    range_count, elevation_count, azimuth_count = shape
    physical_range_axis = (
        np.asarray(range_axis, dtype=np.float64)
        if range_axis is not None and range_axis.size == range_count
        else np.arange(range_count, dtype=np.float64)
    )
    radius = max(int(neighborhood_cells), 0)
    for position in positions:
        target_range = float(np.linalg.norm(position))
        if target_range <= 0.0:
            continue
        range_bin = int(np.argmin(np.abs(physical_range_axis - target_range)))
        azimuth_u = float(np.clip(position[0] / target_range, -1.0, 1.0))
        elevation_u = float(np.clip(-position[2] / target_range, -1.0, 1.0))
        azimuth_bin = int(
            np.clip(
                np.rint((azimuth_u * 0.5 + 0.5) * azimuth_count),
                0,
                azimuth_count - 1,
            )
        )
        elevation_bin = int(
            np.clip(
                np.rint((elevation_u * 0.5 + 0.5) * elevation_count),
                0,
                elevation_count - 1,
            )
        )
        protected[
            max(range_bin - radius, 0) : min(
                range_bin + radius + 1,
                range_count,
            ),
            max(elevation_bin - radius, 0) : min(
                elevation_bin + radius + 1,
                elevation_count,
            ),
            max(azimuth_bin - radius, 0) : min(
                azimuth_bin + radius + 1,
                azimuth_count,
            ),
        ] = True
    return protected


def _local_maxima_3d(values: np.ndarray) -> np.ndarray:
    """Return non-wrapping 3x3x3 local maxima for a three-dimensional cube."""
    if values.ndim != 3 or values.size == 0:
        return np.zeros_like(values, dtype=bool)

    padded = np.pad(
        values,
        ((1, 1), (1, 1), (1, 1)),
        mode="constant",
        constant_values=-np.inf,
    )
    local_maximum = np.full_like(values, -np.inf)
    for range_offset in range(3):
        for elevation_offset in range(3):
            for azimuth_offset in range(3):
                local_maximum = np.maximum(
                    local_maximum,
                    padded[
                        range_offset : range_offset + values.shape[0],
                        elevation_offset : elevation_offset + values.shape[1],
                        azimuth_offset : azimuth_offset + values.shape[2],
                    ],
                )
    return values >= local_maximum


def _local_maximum_candidate_indices(
    values: np.ndarray,
    candidate_indices: np.ndarray,
) -> np.ndarray:
    """Filter thresholded candidates without scanning an empty full cube."""
    if candidate_indices.shape[0] > 128:
        maxima = _local_maxima_3d(values)
        return candidate_indices[maxima[tuple(candidate_indices.T)]]

    keep = np.empty(candidate_indices.shape[0], dtype=bool)
    range_count, elevation_count, azimuth_count = values.shape
    for index, (range_bin, elevation_bin, azimuth_bin) in enumerate(
        candidate_indices
    ):
        neighborhood = values[
            max(int(range_bin) - 1, 0) : min(int(range_bin) + 2, range_count),
            max(int(elevation_bin) - 1, 0) : min(
                int(elevation_bin) + 2,
                elevation_count,
            ),
            max(int(azimuth_bin) - 1, 0) : min(
                int(azimuth_bin) + 2,
                azimuth_count,
            ),
        ]
        keep[index] = values[
            range_bin,
            elevation_bin,
            azimuth_bin,
        ] >= np.max(neighborhood)
    return candidate_indices[keep]


@lru_cache(maxsize=32)
def _static_angle_geometry(
    angle_fft_size: int,
    azimuth_fov_deg: float,
    elevation_fov_deg: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    bins = np.arange(angle_fft_size)
    azimuth_u = _spatial_bins_to_direction_cosines(bins, angle_fft_size)
    elevation_u = _spatial_bins_to_direction_cosines(bins, angle_fft_size)
    elevation_grid, azimuth_grid = np.meshgrid(
        elevation_u,
        azimuth_u,
        indexing="ij",
    )
    valid_angles = azimuth_grid**2 + elevation_grid**2 <= 1.0
    valid_angles &= (
        np.abs(azimuth_grid)
        <= np.sin(np.deg2rad(np.clip(azimuth_fov_deg, 0.0, 90.0))) + 1e-9
    )
    valid_angles &= (
        np.abs(elevation_grid)
        <= np.sin(np.deg2rad(np.clip(elevation_fov_deg, 0.0, 90.0))) + 1e-9
    )
    azimuth_u.setflags(write=False)
    elevation_u.setflags(write=False)
    valid_angles.setflags(write=False)
    return azimuth_u, elevation_u, valid_angles


def compute_micro_doppler_spectrum(
    doppler_cube: np.ndarray,
    range_axis: Optional[np.ndarray],
    *,
    target_range_m: Optional[float] = None,
    range_half_width_bins: int = 2,
    min_range_m: float = 0.25,
    max_range_m: float = 10.0,
) -> tuple[np.ndarray, Optional[float]]:
    """Return one range-gated Doppler spectrum and its selected range.

    The input layout is [doppler, tx, rx, range]. When no target range is
    supplied, the range gate follows the strongest non-zero-Doppler return in
    the requested range interval. Power is combined before converting to dB.
    """
    if doppler_cube.ndim != 4:
        raise ValueError("Doppler cube must have shape [doppler, tx, rx, range]")

    doppler_bins, _tx_count, _rx_count, range_bins = doppler_cube.shape
    if doppler_bins == 0 or range_bins == 0:
        return np.empty((0,), dtype=np.float32), None

    power_map = np.sum(np.abs(doppler_cube) ** 2, axis=(1, 2))
    physical_range_axis = (
        np.asarray(range_axis, dtype=np.float64)
        if range_axis is not None and range_axis.size == range_bins
        else None
    )
    valid_ranges = np.ones(range_bins, dtype=bool)
    if physical_range_axis is not None:
        valid_ranges &= physical_range_axis >= max(float(min_range_m), 0.0)
        if max_range_m > 0.0:
            valid_ranges &= physical_range_axis <= max_range_m
    else:
        valid_ranges[0] = False

    if not np.any(valid_ranges):
        return np.empty((0,), dtype=np.float32), None

    if target_range_m is not None and physical_range_axis is not None:
        candidate_bins = np.flatnonzero(valid_ranges)
        center_bin = int(
            candidate_bins[
                np.argmin(np.abs(physical_range_axis[candidate_bins] - target_range_m))
            ]
        )
    else:
        dynamic_power = power_map.copy()
        zero_doppler_bin = doppler_bins // 2
        dynamic_power[
            max(zero_doppler_bin - 1, 0) : min(zero_doppler_bin + 2, doppler_bins),
            :,
        ] = 0.0
        range_scores = np.max(dynamic_power, axis=0)
        if not np.any(range_scores[valid_ranges] > 0.0):
            range_scores = np.max(power_map, axis=0)
        range_scores[~valid_ranges] = -np.inf
        center_bin = int(np.argmax(range_scores))

    half_width = max(int(range_half_width_bins), 0)
    gate_start = max(center_bin - half_width, 0)
    gate_end = min(center_bin + half_width + 1, range_bins)
    spectrum_power = np.sum(
        np.abs(doppler_cube[..., gate_start:gate_end]) ** 2,
        axis=(1, 2, 3),
    )
    spectrum_db = 10.0 * np.log10(spectrum_power + 1e-12)
    selected_range_m = (
        float(physical_range_axis[center_bin])
        if physical_range_axis is not None
        else None
    )
    return spectrum_db.astype(np.float32, copy=False), selected_range_m


def compute_per_tx_micro_doppler_spectrogram(
    range_fft: np.ndarray,
    range_axis: Optional[np.ndarray],
    config: RadarDspConfig,
    *,
    target_range_m: float,
    range_half_width_bins: int = 2,
    window_loops: int = 64,
    hop_loops: int = 32,
    fft_size: int = 128,
) -> np.ndarray:
    """Return range-gated STFT spectra without coherently merging TDM TX slots.

    Slow-time FFTs are calculated along the loop axis independently for every
    TX slot, RX channel, and gated range bin. Their powers are summed only
    after the FFT, so no inter-TX phase or amplitude calibration is required.
    Output layout is [centered Doppler bin, short-time window].
    """
    if range_fft.ndim != 3 or range_fft.size == 0:
        return np.empty((0, 0), dtype=np.float32)
    if window_loops <= 0 or hop_loops <= 0 or fft_size < window_loops:
        raise ValueError(
            "Micro-Doppler window/hop must be positive and FFT size must "
            "cover the window"
        )

    chirps_per_loop = config.num_chirps_per_loop or 1
    if chirps_per_loop <= 0 or range_fft.shape[0] % chirps_per_loop:
        return np.empty((0, 0), dtype=np.float32)
    loop_count = range_fft.shape[0] // chirps_per_loop
    if loop_count < window_loops:
        return np.empty((0, 0), dtype=np.float32)

    range_bins = range_fft.shape[-1]
    physical_range_axis = (
        np.asarray(range_axis, dtype=np.float64)
        if range_axis is not None and range_axis.size == range_bins
        else None
    )
    if physical_range_axis is None or not np.isfinite(target_range_m):
        return np.empty((0, 0), dtype=np.float32)

    center_bin = int(np.argmin(np.abs(physical_range_axis - target_range_m)))
    half_width = max(int(range_half_width_bins), 0)
    gate_start = max(center_bin - half_width, 0)
    gate_end = min(center_bin + half_width + 1, range_bins)
    loop_cube = range_fft.reshape(
        loop_count,
        chirps_per_loop,
        range_fft.shape[1],
        range_bins,
    )
    gated_loop_cube = loop_cube[..., gate_start:gate_end]

    hann = _hann_window_float32(window_loops)
    spectra = []
    for start in range(0, loop_count - window_loops + 1, hop_loops):
        windowed = (
            gated_loop_cube[start : start + window_loops]
            * hann[:, np.newaxis, np.newaxis, np.newaxis]
        )
        transformed = scipy_fft.fftshift(
            scipy_fft.fft(
                windowed,
                n=fft_size,
                axis=0,
                workers=4,
            ),
            axes=0,
        )
        # Combine TX slots only after their independent slow-time FFTs.
        power = np.sum(
            transformed.real * transformed.real
            + transformed.imag * transformed.imag,
            axis=(1, 2, 3),
        )
        spectra.append(10.0 * np.log10(power + 1e-12))

    if not spectra:
        return np.empty((fft_size, 0), dtype=np.float32)
    return np.stack(spectra, axis=1).astype(np.float32, copy=False)


@dataclass(frozen=True)
class RotorEstimate:
    blade_passage_hz: float
    rpm: float
    confidence: float
    harmonic_rank: int = 1
    velocity_alias_risk: bool = False


@dataclass(frozen=True)
class MicroDopplerResult:
    raw_spectrogram_db: np.ndarray
    enhanced_spectrogram_db: np.ndarray
    window_times_s: np.ndarray
    velocity_axis_m_s: np.ndarray
    flash_scores_db: np.ndarray
    noise_floor_db: np.ndarray
    selected_range_m: float
    nominal_hop_s: float
    unambiguous_velocity_m_s: Optional[float]
    rotor_estimates: tuple[RotorEstimate, ...] = ()
    velocity_alias_risk: bool = False
    alias_warning: Optional[str] = None
    noise_gate_db: Optional[np.ndarray] = None


def radar_slow_time_interval_s(config: RadarDspConfig) -> Optional[float]:
    if (
        config.idle_time_us is None
        or config.ramp_end_time_us is None
        or config.idle_time_us < 0.0
        or config.ramp_end_time_us <= 0.0
    ):
        return None
    chirps_per_loop = config.num_chirps_per_loop or 1
    return (
        float(config.idle_time_us + config.ramp_end_time_us)
        * 1e-6
        * chirps_per_loop
    )


def micro_doppler_velocity_axis_m_s(
    config: RadarDspConfig,
    fft_size: int,
) -> np.ndarray:
    interval_s = radar_slow_time_interval_s(config)
    if (
        interval_s is None
        or not config.start_frequency_ghz
        or config.start_frequency_ghz <= 0.0
        or fft_size <= 0
    ):
        return np.empty((0,), dtype=np.float32)
    wavelength_m = SPEED_OF_LIGHT_M_PER_S / (
        float(config.start_frequency_ghz) * 1e9
    )
    frequencies_hz = scipy_fft.fftshift(
        scipy_fft.fftfreq(fft_size, d=interval_s)
    )
    return (frequencies_hz * wavelength_m * 0.5).astype(
        np.float32,
        copy=False,
    )


def rotor_velocity_alias_diagnostic(
    config: RadarDspConfig,
    *,
    rotor_radius_m: Optional[float],
    rotor_rpm_max: Optional[float],
    warning_fraction: float = 0.8,
) -> tuple[Optional[float], bool, Optional[str]]:
    velocity_axis = micro_doppler_velocity_axis_m_s(config, 2)
    if velocity_axis.size == 0:
        return None, False, "Doppler velocity is unavailable from this profile."

    unambiguous_velocity_m_s = float(abs(velocity_axis[0]))
    if (
        rotor_radius_m is None
        or rotor_rpm_max is None
        or rotor_radius_m <= 0.0
        or rotor_rpm_max <= 0.0
    ):
        return unambiguous_velocity_m_s, False, None

    tip_speed_m_s = (
        2.0 * np.pi * float(rotor_radius_m) * float(rotor_rpm_max) / 60.0
    )
    threshold_m_s = warning_fraction * unambiguous_velocity_m_s
    alias_risk = bool(tip_speed_m_s > threshold_m_s)
    warning = None
    if alias_risk:
        warning = (
            f"Expected tip speed {tip_speed_m_s:.1f} m/s exceeds "
            f"{warning_fraction:.0%} of the ±{unambiguous_velocity_m_s:.1f} "
            "m/s unambiguous velocity; spectral velocity may alias."
        )
    return unambiguous_velocity_m_s, alias_risk, warning


def compute_rotor_micro_doppler_frame(
    range_fft: np.ndarray,
    range_axis: Optional[np.ndarray],
    config: RadarDspConfig,
    *,
    target_range_m: float,
    frame_time_s: float,
    range_half_width_bins: int = 1,
    window_loops: int = 16,
    hop_loops: int = 2,
    fft_size: int = 128,
    dc_notch_bins: int = 2,
    rotor_radius_m: Optional[float] = None,
    rotor_rpm_max: Optional[float] = None,
) -> MicroDopplerResult:
    """Return raw and clutter-rejected spectra for one radar frame."""
    empty = np.empty((0, 0), dtype=np.float32)
    empty_axis = np.empty((0,), dtype=np.float32)
    interval_s = radar_slow_time_interval_s(config)
    if interval_s is None:
        raise ValueError(
            "Rotor micro-Doppler requires start frequency, idle time, and "
            "ramp-end timing in the radar configuration"
        )
    if range_fft.ndim != 3 or range_fft.size == 0:
        return MicroDopplerResult(
            empty,
            empty,
            empty_axis,
            empty_axis,
            empty_axis,
            empty_axis,
            float(target_range_m),
            hop_loops * interval_s,
            None,
        )
    if window_loops <= 0 or hop_loops <= 0 or fft_size < window_loops:
        raise ValueError(
            "Rotor micro-Doppler window/hop must be positive and FFT size "
            "must cover the window"
        )

    chirps_per_loop = config.num_chirps_per_loop or 1
    if range_fft.shape[0] % chirps_per_loop:
        raise ValueError("Frame chirp count is not divisible by chirps per loop")
    loop_count = range_fft.shape[0] // chirps_per_loop
    if loop_count < window_loops:
        return MicroDopplerResult(
            empty,
            empty,
            empty_axis,
            micro_doppler_velocity_axis_m_s(config, fft_size),
            empty_axis,
            empty_axis,
            float(target_range_m),
            hop_loops * interval_s,
            None,
        )

    physical_range_axis = (
        np.asarray(range_axis, dtype=np.float64)
        if range_axis is not None and range_axis.size == range_fft.shape[-1]
        else None
    )
    if physical_range_axis is None or not np.isfinite(target_range_m):
        raise ValueError(
            "Rotor micro-Doppler requires a finite target range and physical "
            "range axis"
        )
    center_bin = int(np.argmin(np.abs(physical_range_axis - target_range_m)))
    half_width = max(int(range_half_width_bins), 0)
    gate_start = max(center_bin - half_width, 0)
    gate_end = min(center_bin + half_width + 1, range_fft.shape[-1])
    selected_range_m = float(physical_range_axis[center_bin])

    loop_cube = range_fft.reshape(
        loop_count,
        chirps_per_loop,
        range_fft.shape[1],
        range_fft.shape[2],
    )
    gated = loop_cube[..., gate_start:gate_end]
    window = _hann_window_float32(window_loops)
    window_sum = float(np.sum(window))
    window_starts = np.arange(
        0,
        loop_count - window_loops + 1,
        hop_loops,
        dtype=np.intp,
    )
    samples = np.stack(
        [gated[start : start + window_loops] for start in window_starts],
        axis=0,
    )
    weights = window[np.newaxis, :, np.newaxis, np.newaxis, np.newaxis]
    weighted_mean = np.sum(samples * weights, axis=1, keepdims=True) / max(
        window_sum,
        1e-12,
    )
    raw_fft = scipy_fft.fftshift(
        scipy_fft.fft(
            samples * weights,
            n=fft_size,
            axis=1,
            workers=2,
        ),
        axes=1,
    )
    window_spectrum = _centered_hann_fft_complex64(
        window_loops,
        fft_size,
    )
    # FFT((x - mean) * window) is exactly FFT(x * window) minus the
    # weighted mean times FFT(window). Reusing raw_fft avoids a second
    # batched FFT for every highly overlapped STFT window.
    cancelled_fft = raw_fft - (
        weighted_mean
        * window_spectrum[
            np.newaxis,
            :,
            np.newaxis,
            np.newaxis,
            np.newaxis,
        ]
    )
    raw_power = np.sum(
        raw_fft.real * raw_fft.real + raw_fft.imag * raw_fft.imag,
        axis=(2, 3, 4),
    )
    cancelled_power = np.sum(
        cancelled_fft.real * cancelled_fft.real
        + cancelled_fft.imag * cancelled_fft.imag,
        axis=(2, 3, 4),
    )
    raw_spectrogram_db = (
        10.0 * np.log10(raw_power + 1e-12)
    ).T.astype(
        np.float32,
        copy=False,
    )
    cancelled_spectrogram_db = (
        10.0 * np.log10(cancelled_power + 1e-12)
    ).T.astype(
        np.float32,
        copy=False,
    )
    window_times_s = (
        float(frame_time_s)
        + (window_starts + (window_loops - 1) * 0.5) * interval_s
    )
    fft_center = fft_size // 2
    notch = max(int(dc_notch_bins), 0)
    off_center = np.ones(fft_size, dtype=bool)
    off_center[
        max(fft_center - notch, 0) : min(fft_center + notch + 1, fft_size)
    ] = False
    if not np.any(off_center):
        raise ValueError("DC notch excludes every Doppler bin")
    noise_floor_db, noise_gate_db = _rotor_noise_floor_and_gate(
        cancelled_spectrogram_db,
        off_center,
    )
    relative_spectrogram_db = np.maximum(
        cancelled_spectrogram_db - noise_floor_db[np.newaxis, :],
        0.0,
    ).astype(np.float32, copy=False)
    enhanced_spectrogram_db = _apply_rotor_noise_support_filter(
        relative_spectrogram_db,
        noise_gate_db,
        off_center,
    )
    excess_linear = np.maximum(
        10.0 ** (
            np.clip(enhanced_spectrogram_db[off_center], 0.0, 30.0) / 10.0
        )
        - 1.0,
        0.0,
    )
    flash_scores_db = (
        10.0 * np.log10(1.0 + np.mean(excess_linear, axis=0))
    ).astype(np.float32, copy=False)
    velocity_axis = micro_doppler_velocity_axis_m_s(config, fft_size)
    unambiguous_velocity_m_s, alias_risk, alias_warning = (
        rotor_velocity_alias_diagnostic(
            config,
            rotor_radius_m=rotor_radius_m,
            rotor_rpm_max=rotor_rpm_max,
        )
    )
    return MicroDopplerResult(
        raw_spectrogram_db=raw_spectrogram_db,
        enhanced_spectrogram_db=enhanced_spectrogram_db,
        window_times_s=np.asarray(window_times_s, dtype=np.float64),
        velocity_axis_m_s=velocity_axis,
        flash_scores_db=flash_scores_db,
        noise_floor_db=noise_floor_db,
        selected_range_m=selected_range_m,
        nominal_hop_s=hop_loops * interval_s,
        unambiguous_velocity_m_s=unambiguous_velocity_m_s,
        noise_gate_db=noise_gate_db,
        velocity_alias_risk=alias_risk,
        alias_warning=alias_warning,
    )


def _rotor_noise_floor_and_gate(
    cancelled_spectrogram_db: np.ndarray,
    off_center_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Estimate per-window floor and lower-tail MAD without blade bias."""
    spectrogram = np.asarray(cancelled_spectrogram_db, dtype=np.float32)
    off_center = np.asarray(off_center_mask, dtype=bool)
    if spectrogram.ndim != 2:
        raise ValueError("Rotor cancelled spectrogram must be two-dimensional")
    if off_center.shape != (spectrogram.shape[0],):
        raise ValueError("Rotor off-centre mask must contain one value per bin")

    noise_samples_db = spectrogram[off_center]
    noise_floor_db = np.median(
        noise_samples_db,
        axis=0,
    ).astype(np.float32, copy=False)
    lower_deviations_db = np.where(
        noise_samples_db <= noise_floor_db[np.newaxis, :],
        noise_floor_db[np.newaxis, :] - noise_samples_db,
        np.nan,
    )
    noise_mad_db = np.nanmedian(
        lower_deviations_db,
        axis=0,
    )
    robust_noise_sigma_db = (
        ROBUST_GAUSSIAN_MAD_SCALE * noise_mad_db
    ).astype(np.float32, copy=False)
    noise_gate_db = np.maximum(
        DEFAULT_ROTOR_NOISE_GATE_MIN_DB,
        DEFAULT_ROTOR_NOISE_SIGMA_MULTIPLIER * robust_noise_sigma_db,
    ).astype(np.float32, copy=False)
    # Deep deterministic FFT/cancellation nulls can make the lower-tail MAD
    # enormous even when visible rotor ridges are present. Keep the adaptive
    # gate useful without allowing those nulls to blank the whole display.
    noise_gate_db = np.minimum(
        noise_gate_db,
        DEFAULT_ROTOR_NOISE_GATE_MAX_DB,
    ).astype(np.float32, copy=False)
    return noise_floor_db, noise_gate_db


def _apply_rotor_noise_support_filter(
    relative_spectrogram_db: np.ndarray,
    noise_gate_db: np.ndarray,
    off_center_mask: np.ndarray,
) -> np.ndarray:
    """Blank sub-gate and isolated rotor STFT cells without attenuating peaks."""
    relative = np.asarray(relative_spectrogram_db, dtype=np.float32)
    gates = np.asarray(noise_gate_db, dtype=np.float32)
    off_center = np.asarray(off_center_mask, dtype=bool)
    if relative.ndim != 2:
        raise ValueError("Rotor relative spectrogram must be two-dimensional")
    if gates.shape != (relative.shape[1],):
        raise ValueError("Rotor noise gate must contain one value per window")
    if off_center.shape != (relative.shape[0],):
        raise ValueError("Rotor off-centre mask must contain one value per bin")

    candidates = (
        (relative >= gates[np.newaxis, :])
        & off_center[:, np.newaxis]
    )
    support = scipy_ndimage.convolve(
        candidates.astype(np.uint8),
        np.ones(DEFAULT_ROTOR_NOISE_SUPPORT_SHAPE, dtype=np.uint8),
        mode="constant",
        cval=0,
    )
    retained = candidates & (support >= DEFAULT_ROTOR_NOISE_MIN_SUPPORT)
    return np.where(retained, relative, 0.0).astype(np.float32, copy=False)


def estimate_rotor_rpm(
    window_times_s: np.ndarray,
    flash_scores_db: np.ndarray,
    *,
    blade_count: int,
    rotor_count: int = 1,
    rpm_min: float = 500.0,
    rpm_max: float = 10_700.0,
    velocity_alias_risk: bool = False,
    minimum_duration_s: float = 0.5,
) -> tuple[RotorEstimate, ...]:
    """Estimate blade-passage rates from irregularly timed flash scores."""
    times = np.asarray(window_times_s, dtype=np.float64)
    scores = np.asarray(flash_scores_db, dtype=np.float64)
    finite = np.isfinite(times) & np.isfinite(scores)
    times = times[finite]
    scores = scores[finite]
    if (
        blade_count <= 0
        or rotor_count <= 0
        or rpm_min <= 0.0
        or rpm_max <= rpm_min
        or times.size < 8
        or float(np.ptp(times)) < minimum_duration_s
    ):
        return ()

    order = np.argsort(times)
    times = times[order]
    scores = scores[order]
    if times.size > 1024:
        sample_indices = np.linspace(
            0,
            times.size - 1,
            1024,
            dtype=np.intp,
        )
        times = times[sample_indices]
        scores = scores[sample_indices]
    times = times - times[0]
    scores = scores - np.mean(scores)
    if not np.any(np.abs(scores) > 1e-9):
        return ()

    minimum_hz = float(rpm_min) * blade_count / 60.0
    maximum_hz = float(rpm_max) * blade_count / 60.0
    frequencies_hz = np.linspace(
        minimum_hz,
        maximum_hz,
        2048,
        dtype=np.float64,
    )
    angular_frequencies = 2.0 * np.pi * frequencies_hz
    periodogram = scipy_signal.lombscargle(
        times,
        scores,
        angular_frequencies,
        precenter=False,
        normalize=True,
    )
    peak_indices, _properties = scipy_signal.find_peaks(periodogram)
    boundary_peaks: list[int] = []
    if periodogram.size == 1:
        boundary_peaks.append(0)
    else:
        if periodogram[0] >= periodogram[1]:
            boundary_peaks.append(0)
        if periodogram[-1] >= periodogram[-2]:
            boundary_peaks.append(periodogram.size - 1)
    candidate_indices = np.unique(
        np.concatenate(
            (
                peak_indices,
                np.asarray(boundary_peaks, dtype=np.intp),
            )
        )
    )
    if candidate_indices.size == 0:
        candidate_indices = np.asarray(
            (int(np.argmax(periodogram)),),
            dtype=np.intp,
        )
    ranked_indices = candidate_indices[
        np.argsort(periodogram[candidate_indices])[::-1]
    ]
    baseline = float(np.percentile(periodogram, 95.0))
    estimates: list[RotorEstimate] = []
    selected_frequencies: list[float] = []
    minimum_separation_hz = max(1.0 / np.ptp(times), minimum_hz * 0.02)

    for peak_index in ranked_indices:
        frequency_hz = float(frequencies_hz[peak_index])
        if any(
            abs(frequency_hz - selected) < minimum_separation_hz
            for selected in selected_frequencies
        ):
            continue
        peak_power = float(periodogram[peak_index])
        confidence = float(
            np.clip(
                (peak_power - baseline) / max(peak_power, 1e-12),
                0.0,
                1.0,
            )
        )
        harmonic_rank = 1
        for lower_index in candidate_indices:
            lower_frequency_hz = float(frequencies_hz[lower_index])
            if lower_frequency_hz >= frequency_hz:
                continue
            ratio = frequency_hz / lower_frequency_hz
            nearest_harmonic = int(round(ratio))
            if (
                2 <= nearest_harmonic <= 6
                and abs(ratio - nearest_harmonic) <= 0.03
                and periodogram[lower_index] >= 0.35 * peak_power
            ):
                harmonic_rank = nearest_harmonic
                break
        estimates.append(
            RotorEstimate(
                blade_passage_hz=frequency_hz,
                rpm=60.0 * frequency_hz / blade_count,
                confidence=confidence,
                harmonic_rank=harmonic_rank,
                velocity_alias_risk=bool(velocity_alias_risk),
            )
        )
        selected_frequencies.append(frequency_hz)
        if len(estimates) >= rotor_count:
            break
    return tuple(estimates)


@lru_cache(maxsize=16)
def _hann_window_float32(length: int) -> np.ndarray:
    window = np.hanning(length).astype(np.float32)
    window.setflags(write=False)
    return window


@lru_cache(maxsize=16)
def _centered_hann_fft_complex64(
    length: int,
    fft_size: int,
) -> np.ndarray:
    spectrum = scipy_fft.fftshift(
        scipy_fft.fft(
            _hann_window_float32(length),
            n=fft_size,
        )
    ).astype(np.complex64, copy=False)
    spectrum.setflags(write=False)
    return spectrum


def cluster_point_cloud(
    points: np.ndarray,
    *,
    eps_m: float = 0.4,
    min_samples: int = 2,
) -> np.ndarray:
    """Return DBSCAN cluster centers as [x, y, z, point_count]."""
    centers, _labels = cluster_point_cloud_with_labels(
        points,
        eps_m=eps_m,
        min_samples=min_samples,
    )
    return centers


def cluster_point_cloud_with_labels(
    points: np.ndarray,
    *,
    eps_m: float = 0.4,
    min_samples: int = 2,
) -> tuple[np.ndarray, np.ndarray]:
    """Return DBSCAN centers and one membership label per input point."""
    if points.ndim != 2 or (points.size and points.shape[1] < 3):
        raise ValueError("Point cloud must have shape [point, at least 3 values]")
    if points.size == 0 or eps_m <= 0.0:
        return (
            np.empty((0, 4), dtype=np.float32),
            np.full(points.shape[0], -1, dtype=np.intp),
        )
    if min_samples < 1:
        raise ValueError("DBSCAN min_samples must be at least 1")

    if points.shape[0] <= 128:
        labels = _dbscan_labels_small_cloud(
            points[:, :3],
            eps_m=float(eps_m),
            min_samples=int(min_samples),
        )
    else:
        from sklearn.cluster import DBSCAN

        labels = DBSCAN(
            eps=float(eps_m),
            min_samples=int(min_samples),
        ).fit_predict(points[:, :3])
    centers = []
    for label in sorted(set(int(label) for label in labels if label >= 0)):
        members = points[labels == label, :3]
        center = members.mean(axis=0)
        centers.append((center[0], center[1], center[2], members.shape[0]))

    if not centers:
        return np.empty((0, 4), dtype=np.float32), labels
    return np.asarray(centers, dtype=np.float32), labels


def _dbscan_labels_small_cloud(
    xyz_m: np.ndarray,
    *,
    eps_m: float,
    min_samples: int,
) -> np.ndarray:
    """Deterministic DBSCAN without sklearn's per-frame validation overhead."""
    coordinates = np.asarray(xyz_m, dtype=np.float64)
    point_count = coordinates.shape[0]
    differences = (
        coordinates[:, np.newaxis, :] - coordinates[np.newaxis, :, :]
    )
    neighbors = np.sum(differences * differences, axis=2) <= eps_m * eps_m
    labels = np.full(point_count, -1, dtype=np.intp)
    visited = np.zeros(point_count, dtype=bool)
    cluster_label = 0

    for point_index in range(point_count):
        if visited[point_index]:
            continue
        visited[point_index] = True
        point_neighbors = np.flatnonzero(neighbors[point_index])
        if point_neighbors.size < min_samples:
            continue

        labels[point_index] = cluster_label
        seeds = deque(
            int(index) for index in point_neighbors if index != point_index
        )
        queued = np.zeros(point_count, dtype=bool)
        queued[point_neighbors] = True
        while seeds:
            neighbor_index = seeds.popleft()
            if not visited[neighbor_index]:
                visited[neighbor_index] = True
                neighbor_neighbors = np.flatnonzero(neighbors[neighbor_index])
                if neighbor_neighbors.size >= min_samples:
                    for new_index in neighbor_neighbors:
                        if not queued[new_index]:
                            queued[new_index] = True
                            seeds.append(int(new_index))
            if labels[neighbor_index] < 0:
                labels[neighbor_index] = cluster_label
        cluster_label += 1
    return labels


def _point_is_within_fov(
    x_m: float,
    y_m: float,
    z_m: float,
    *,
    azimuth_fov_deg: float,
    elevation_fov_deg: float,
) -> bool:
    """Gate XYZ coordinates using the array's azimuth/elevation direction cosines."""
    point = np.asarray(((x_m, y_m, z_m),), dtype=np.float64)
    return bool(
        _points_are_within_fov(
            point,
            azimuth_fov_deg=azimuth_fov_deg,
            elevation_fov_deg=elevation_fov_deg,
        )[0]
    )


def _points_are_within_fov(
    xyz_m: np.ndarray,
    *,
    azimuth_fov_deg: float,
    elevation_fov_deg: float,
) -> np.ndarray:
    """Vectorized FOV gate for an [point, xyz] array."""
    ranges_m = np.linalg.norm(xyz_m, axis=1)
    nonzero = ranges_m > 0.0
    safe_ranges_m = np.where(nonzero, ranges_m, 1.0)
    azimuth_limit = np.sin(np.deg2rad(np.clip(azimuth_fov_deg, 0.0, 90.0)))
    elevation_limit = np.sin(np.deg2rad(np.clip(elevation_fov_deg, 0.0, 90.0)))
    tolerance = 1e-9
    within = (
        (np.abs(xyz_m[:, 0] / safe_ranges_m) <= azimuth_limit + tolerance)
        & (np.abs(xyz_m[:, 2] / safe_ranges_m) <= elevation_limit + tolerance)
    )
    return ~nonzero | within


def os_cfar_2d(
    power_map: np.ndarray,
    *,
    false_alarm_rate: float,
    range_guard_cells: int,
    doppler_guard_cells: int,
    range_training_cells: int,
    doppler_training_cells: int,
) -> np.ndarray:
    """Run OpenRadar OS-CFAR on a [doppler, range] power map."""
    return openradar_os_cfar_2d(
        power_map,
        false_alarm_rate=false_alarm_rate,
        range_guard_cells=range_guard_cells,
        doppler_guard_cells=doppler_guard_cells,
        range_training_cells=range_training_cells,
        doppler_training_cells=doppler_training_cells,
    )


def doppler_peak_mask(power_map: np.ndarray) -> np.ndarray:
    """Keep cells that are not weaker than either cyclic Doppler neighbor."""
    if power_map.ndim != 2 or power_map.size == 0:
        return np.zeros_like(power_map, dtype=bool)

    previous_doppler = np.roll(power_map, shift=1, axis=0)
    next_doppler = np.roll(power_map, shift=-1, axis=0)
    return (power_map >= previous_doppler) & (power_map >= next_doppler)


def _apply_min_range_gate(
    detections: np.ndarray,
    range_axis: Optional[np.ndarray],
    min_range_m: float,
) -> np.ndarray:
    if min_range_m <= 0:
        return detections
    gated = detections.copy()
    if range_axis is not None and range_axis.size == detections.shape[1]:
        gated[:, range_axis < min_range_m] = False
    else:
        gated[:, 0] = False
    return gated


def estimate_xyz_from_virtual_array(
    virtual_samples: np.ndarray,
    target_range_m: float,
    config: RadarDspConfig,
    *,
    angle_fft_size: int = 32,
) -> Optional[tuple[float, float, float]]:
    """Estimate x/y/z from one range-Doppler cell using a simple 2D angle FFT.

    The returned coordinate system is x=left/right, y=forward range, z=elevation.
    This is an uncalibrated planar-array estimate intended for live visualization.
    """
    xyz_m, valid = estimate_xyz_from_virtual_arrays(
        virtual_samples[np.newaxis, ...],
        np.asarray((target_range_m,)),
        config,
        angle_fft_size=angle_fft_size,
    )
    if not valid[0]:
        return None
    return tuple(float(value) for value in xyz_m[0])


def estimate_xyz_from_virtual_arrays(
    virtual_samples: np.ndarray,
    target_ranges_m: np.ndarray,
    config: RadarDspConfig,
    *,
    angle_fft_size: int = 32,
) -> tuple[np.ndarray, np.ndarray]:
    """Estimate XYZ for many range-Doppler cells with one batched 2D FFT."""
    virtual_arrays = build_virtual_antenna_grids(virtual_samples, config)
    target_ranges_m = np.asarray(target_ranges_m, dtype=np.float64)
    if target_ranges_m.shape != (virtual_arrays.shape[0],):
        raise ValueError("Target ranges must contain one value per virtual array")

    angle_response = np.fft.fftshift(
        np.fft.fft2(
            virtual_arrays,
            s=(angle_fft_size, angle_fft_size),
            axes=(-2, -1),
        ),
        axes=(-2, -1),
    )
    magnitude = np.abs(angle_response)
    has_signal = np.any(magnitude, axis=(-2, -1))
    peak_indices = np.argmax(magnitude.reshape(magnitude.shape[0], -1), axis=1)
    elevation_bins, azimuth_bins = np.divmod(peak_indices, angle_fft_size)
    azimuth_u = _spatial_bins_to_direction_cosines(azimuth_bins, angle_fft_size)
    elevation_u = _spatial_bins_to_direction_cosines(elevation_bins, angle_fft_size)
    azimuth_u = np.where(has_signal, azimuth_u, 0.0)
    elevation_u = np.where(has_signal, elevation_u, 0.0)

    direction_norm_sq = azimuth_u**2 + elevation_u**2
    valid = direction_norm_sq <= 1.0
    radial_scale = np.sqrt(np.maximum(0.0, 1.0 - direction_norm_sq))
    ranges_m = np.maximum(target_ranges_m, 0.0)
    xyz_m = np.column_stack(
        (
            ranges_m * azimuth_u,
            ranges_m * radial_scale,
            -ranges_m * elevation_u,
        )
    )
    return xyz_m, valid


def build_virtual_antenna_grid(
    virtual_samples: np.ndarray,
    config: RadarDspConfig,
) -> np.ndarray:
    return build_virtual_antenna_grids(virtual_samples[np.newaxis, ...], config)[0]


def build_virtual_antenna_grids(
    virtual_samples: np.ndarray,
    config: RadarDspConfig,
) -> np.ndarray:
    """Map [point, tx chirp, rx] samples into planar virtual-array grids."""
    if virtual_samples.ndim != 3:
        raise ValueError("Virtual samples must have shape [point, tx chirp, rx]")

    point_count, chirps_per_loop, rx_count = virtual_samples.shape
    tx_indices = _tx_indices_for_chirps(config, chirps_per_loop)
    ods_grids = _build_iwr6843isk_ods_virtual_antenna_grids(
        virtual_samples,
        tx_indices,
        rx_count,
    )
    if ods_grids is not None:
        return ods_grids

    elevation_size = max(max(tx_indices, default=0) + 1, chirps_per_loop)
    grids = np.zeros(
        (point_count, elevation_size, rx_count),
        dtype=np.complex64,
    )

    for chirp_index, tx_index in enumerate(tx_indices[:chirps_per_loop]):
        grids[:, tx_index, :rx_count] = virtual_samples[:, chirp_index, :rx_count]
    return grids


def _tx_indices_for_chirps(
    config: RadarDspConfig,
    chirps_per_loop: int,
) -> list[int]:
    masks = config.tx_channel_masks
    if masks:
        indices = []
        for mask in masks[:chirps_per_loop]:
            enabled = [bit for bit in range(8) if mask & (1 << bit)]
            indices.append(enabled[0] if enabled else len(indices))
        if len(indices) == chirps_per_loop:
            return indices
    return list(range(chirps_per_loop))


def _spatial_bins_to_direction_cosines(
    bin_indices: np.ndarray,
    fft_size: int,
) -> np.ndarray:
    direction_cosines = 2.0 * (
        (bin_indices - (fft_size // 2)) / float(fft_size)
    )
    return np.clip(direction_cosines, -1.0, 1.0)


def _build_iwr6843isk_ods_virtual_antenna_grids(
    virtual_samples: np.ndarray,
    tx_indices: list[int],
    rx_count: int,
) -> Optional[np.ndarray]:
    if rx_count != 4:
        return None

    tx_numbers = [tx_index + 1 for tx_index in tx_indices]
    if not set(tx_numbers).issubset({1, 2, 3}):
        return None

    # IWR6843ISK-ODS page-40 layout. Rows are bottom-to-top elevation; columns
    # are left-to-right azimuth. RX2/RX3 require 180 degree phase inversion.
    positions = {
        (1, 1): (3, 0),
        (1, 2): (2, 0),
        (1, 3): (2, 1),
        (1, 4): (3, 1),
        (2, 1): (3, 2),
        (2, 2): (2, 2),
        (2, 3): (2, 3),
        (2, 4): (3, 3),
        (3, 1): (1, 2),
        (3, 2): (0, 2),
        (3, 3): (0, 3),
        (3, 4): (1, 3),
    }
    rx_phase = {
        1: 1.0,
        2: -1.0,
        3: -1.0,
        4: 1.0,
    }

    grids = np.zeros((virtual_samples.shape[0], 4, 4), dtype=np.complex64)
    for chirp_index, tx_number in enumerate(tx_numbers[: virtual_samples.shape[1]]):
        for rx_number in range(1, rx_count + 1):
            row, col = positions[(tx_number, rx_number)]
            grids[:, row, col] = (
                rx_phase[rx_number]
                * virtual_samples[:, chirp_index, rx_number - 1]
            )
    return grids


def frame_bytes_to_radar_cube(
    frame_bytes: bytes,
    config: RadarDspConfig,
) -> np.ndarray:
    """Convert one complete DCA1000 frame into [chirp, rx, sample] complex data."""
    expected_int16_count = config.bytes_per_frame // np.dtype("<i2").itemsize
    adc_samples = np.frombuffer(frame_bytes, dtype="<i2", count=expected_int16_count)

    if adc_samples.size != expected_int16_count:
        raise ValueError(
            f"Frame has {adc_samples.size} int16 values; expected {expected_int16_count}"
        )
    if adc_samples.size % 4 != 0:
        raise ValueError("Complex 2-lane LVDS data must contain int16 groups of 4")
    if config.lvds_lanes != 2:
        raise NotImplementedError(
            f"LVDS reshape currently supports 2 lanes, got {config.lvds_lanes}"
        )

    grouped = adc_samples.reshape(-1, 4)
    first = grouped[:, 0:2].reshape(-1).astype(np.float32)
    second = grouped[:, 2:4].reshape(-1).astype(np.float32)

    if config.iq_swap:
        complex_samples = second + 1j * first
    else:
        complex_samples = first + 1j * second

    expected_complex_count = (
        config.num_chirps_per_frame
        * config.num_rx_channels
        * config.num_adc_samples
    )
    if complex_samples.size != expected_complex_count:
        raise ValueError(
            "Frame produced "
            f"{complex_samples.size} complex samples; expected {expected_complex_count}"
        )

    if config.channel_interleave:
        return complex_samples.reshape(
            config.num_chirps_per_frame,
            config.num_adc_samples,
            config.num_rx_channels,
        ).transpose(0, 2, 1)

    return complex_samples.reshape(
        config.num_chirps_per_frame,
        config.num_rx_channels,
        config.num_adc_samples,
    )
