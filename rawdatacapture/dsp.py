from typing import Optional, Protocol

import numpy as np

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
    max_points: Optional[int] = None,
    false_alarm_rate: float = 1e-2,
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

    doppler_fft = compute_range_doppler_fft(range_fft, config)
    detection_power = np.mean(np.abs(doppler_fft) ** 2, axis=(1, 2))
    if detection_power.size == 0:
        return np.empty((0, 4), dtype=np.float32)

    detections = os_cfar_2d(
        detection_power,
        false_alarm_rate=false_alarm_rate,
        range_guard_cells=range_guard_cells,
        doppler_guard_cells=doppler_guard_cells,
        range_training_cells=range_training_cells,
        doppler_training_cells=doppler_training_cells,
    )
    detections &= doppler_peak_mask(detection_power)
    detections = _apply_min_range_gate(detections, range_axis, min_range_m)

    candidate_indices = np.argwhere(detections)
    if candidate_indices.size == 0:
        return np.empty((0, 4), dtype=np.float32)

    doppler_bins = candidate_indices[:, 0]
    range_bins = candidate_indices[:, 1]
    candidate_powers = detection_power[doppler_bins, range_bins]

    # Unlimited displays do not need strongest-first ordering. Retain it only
    # when a finite point cap makes candidate priority observable.
    if max_points is not None:
        order = np.argsort(candidate_powers)[::-1]
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

    virtual_samples = doppler_fft[doppler_bins, :, :, range_bins]
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


def cluster_point_cloud(
    points: np.ndarray,
    *,
    eps_m: float = 0.5,
    min_samples: int = 2,
) -> np.ndarray:
    """Return DBSCAN cluster centers as [x, y, z, point_count]."""
    if points.ndim != 2 or (points.size and points.shape[1] < 3):
        raise ValueError("Point cloud must have shape [point, at least 3 values]")
    if points.size == 0 or eps_m <= 0.0:
        return np.empty((0, 4), dtype=np.float32)
    if min_samples < 1:
        raise ValueError("DBSCAN min_samples must be at least 1")

    from sklearn.cluster import DBSCAN

    labels = DBSCAN(eps=float(eps_m), min_samples=int(min_samples)).fit_predict(
        points[:, :3]
    )
    centers = []
    for label in sorted(set(int(label) for label in labels if label >= 0)):
        members = points[labels == label, :3]
        center = members.mean(axis=0)
        centers.append((center[0], center[1], center[2], members.shape[0]))

    if not centers:
        return np.empty((0, 4), dtype=np.float32)
    return np.asarray(centers, dtype=np.float32)


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
