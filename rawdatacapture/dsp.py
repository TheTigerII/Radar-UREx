from typing import Optional, Protocol

import numpy as np


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
    window = np.hanning(radar_cube.shape[2]).astype(np.float32)
    return np.fft.fft(radar_cube * window, axis=2)


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
        loops = range_fft.shape[0]
        chirps_per_loop = 1

    tdm_cube = range_fft.reshape(
        loops,
        chirps_per_loop,
        config.num_rx_channels,
        config.num_adc_samples,
    )
    return np.fft.fftshift(np.fft.fft(tdm_cube, axis=0), axes=0)


def compute_point_cloud(
    range_fft: np.ndarray,
    range_axis: Optional[np.ndarray],
    config: RadarDspConfig,
    *,
    max_points: int = 50,
    false_alarm_rate: float = 1e-3,
    range_guard_cells: int = 2,
    doppler_guard_cells: int = 1,
    range_training_cells: int = 8,
    doppler_training_cells: int = 2,
    min_range_m: float = 0.15,
) -> np.ndarray:
    doppler_fft = compute_range_doppler_fft(range_fft, config)
    detection_power = np.mean(np.abs(doppler_fft) ** 2, axis=(1, 2))
    if detection_power.size == 0:
        return np.empty((0, 4), dtype=np.float32)

    detections = ca_cfar_2d(
        detection_power,
        false_alarm_rate=false_alarm_rate,
        range_guard_cells=range_guard_cells,
        doppler_guard_cells=doppler_guard_cells,
        range_training_cells=range_training_cells,
        doppler_training_cells=doppler_training_cells,
    )
    detections &= local_peak_mask(detection_power)
    detections = _apply_min_range_gate(detections, range_axis, min_range_m)

    candidate_indices = np.argwhere(detections)
    if candidate_indices.size == 0:
        return np.empty((0, 4), dtype=np.float32)

    candidate_powers = detection_power[candidate_indices[:, 0], candidate_indices[:, 1]]
    order = np.argsort(candidate_powers)[::-1][:max_points]
    selected_indices = candidate_indices[order]
    selected_magnitudes_db = 10.0 * np.log10(candidate_powers[order] + 1e-12)

    points = []
    for (doppler_bin, range_bin), magnitude_db in zip(
        selected_indices,
        selected_magnitudes_db,
    ):
        if range_axis is not None and range_axis.size == detection_power.shape[1]:
            target_range_m = float(range_axis[range_bin])
        else:
            target_range_m = float(range_bin)

        x_m, y_m, z_m = estimate_xyz_from_virtual_array(
            doppler_fft[int(doppler_bin), :, :, int(range_bin)],
            target_range_m,
            config,
        )
        points.append((x_m, y_m, z_m, float(magnitude_db)))

    if not points:
        return np.empty((0, 4), dtype=np.float32)
    return np.asarray(points, dtype=np.float32)


def ca_cfar_2d(
    power_map: np.ndarray,
    *,
    false_alarm_rate: float,
    range_guard_cells: int,
    doppler_guard_cells: int,
    range_training_cells: int,
    doppler_training_cells: int,
) -> np.ndarray:
    """Run 2D cell-averaging CFAR on a [doppler, range] power map."""
    if power_map.ndim != 2 or power_map.size == 0:
        return np.zeros_like(power_map, dtype=bool)

    doppler_bins, range_bins = power_map.shape
    detections = np.zeros_like(power_map, dtype=bool)
    range_margin = range_training_cells + range_guard_cells
    doppler_window = doppler_training_cells + doppler_guard_cells
    training_cells = (
        (2 * doppler_window + 1) * (2 * range_margin + 1)
        - (2 * doppler_guard_cells + 1) * (2 * range_guard_cells + 1)
    )
    if training_cells <= 0 or range_bins <= (2 * range_margin):
        return detections

    pfa = min(max(false_alarm_rate, 1e-9), 0.5)
    threshold_scale = training_cells * (pfa ** (-1.0 / training_cells) - 1.0)

    doppler_extended = np.concatenate(
        (
            power_map[-doppler_window:, :],
            power_map,
            power_map[:doppler_window, :],
        ),
        axis=0,
    )
    total_windows = np.lib.stride_tricks.sliding_window_view(
        doppler_extended,
        (2 * doppler_window + 1, 2 * range_margin + 1),
    )
    total_sum = total_windows.sum(axis=(-2, -1), dtype=np.float64)

    guard_source = doppler_extended[
        doppler_window - doppler_guard_cells : doppler_window
        - doppler_guard_cells
        + doppler_bins
        + (2 * doppler_guard_cells),
        :,
    ]
    guard_windows = np.lib.stride_tricks.sliding_window_view(
        guard_source,
        (2 * doppler_guard_cells + 1, 2 * range_guard_cells + 1),
    )
    guard_start = range_margin - range_guard_cells
    guard_sum = guard_windows[
        :,
        guard_start : guard_start + total_sum.shape[1],
    ].sum(axis=(-2, -1), dtype=np.float64)

    training_sum = total_sum - guard_sum
    noise_estimate = training_sum / training_cells
    test_cells = power_map[:, range_margin : range_bins - range_margin]
    detections[:, range_margin : range_bins - range_margin] = (
        test_cells > threshold_scale * noise_estimate
    )

    return detections


def local_peak_mask(power_map: np.ndarray) -> np.ndarray:
    if power_map.ndim != 2 or power_map.size == 0:
        return np.zeros_like(power_map, dtype=bool)

    doppler_bins, range_bins = power_map.shape
    peaks = np.ones_like(power_map, dtype=bool)
    for doppler_offset in (-1, 0, 1):
        for range_offset in (-1, 0, 1):
            if doppler_offset == 0 and range_offset == 0:
                continue
            shifted = np.roll(power_map, shift=doppler_offset, axis=0)
            if range_offset < 0:
                neighbor = np.empty_like(power_map)
                neighbor[:, :-1] = shifted[:, 1:]
                neighbor[:, -1] = -np.inf
            elif range_offset > 0:
                neighbor = np.empty_like(power_map)
                neighbor[:, 1:] = shifted[:, :-1]
                neighbor[:, 0] = -np.inf
            else:
                neighbor = shifted
            peaks &= power_map >= neighbor
    return peaks


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
) -> tuple[float, float, float]:
    """Estimate x/y/z from one range-Doppler cell using a simple 2D angle FFT.

    The returned coordinate system is x=left/right, y=forward range, z=elevation.
    This is an uncalibrated planar-array estimate intended for live visualization.
    """
    virtual_array = build_virtual_antenna_grid(virtual_samples, config)
    angle_response = np.fft.fftshift(
        np.fft.fft2(virtual_array, s=(angle_fft_size, angle_fft_size)),
    )
    magnitude = np.abs(angle_response)
    if not np.any(magnitude):
        return 0.0, max(target_range_m, 0.0), 0.0

    elevation_bin, azimuth_bin = np.unravel_index(
        int(np.argmax(magnitude)),
        magnitude.shape,
    )
    azimuth_u = _spatial_bin_to_direction_cosine(azimuth_bin, angle_fft_size)
    elevation_u = _spatial_bin_to_direction_cosine(elevation_bin, angle_fft_size)

    radial_scale = max(0.0, 1.0 - azimuth_u**2 - elevation_u**2) ** 0.5
    x_m = target_range_m * azimuth_u
    z_m = target_range_m * elevation_u
    y_m = target_range_m * radial_scale
    return float(x_m), float(y_m), float(z_m)


def build_virtual_antenna_grid(
    virtual_samples: np.ndarray,
    config: RadarDspConfig,
) -> np.ndarray:
    chirps_per_loop = virtual_samples.shape[0]
    rx_count = virtual_samples.shape[1]
    tx_indices = _tx_indices_for_chirps(config, chirps_per_loop)
    elevation_size = max(max(tx_indices, default=0) + 1, chirps_per_loop)
    grid = np.zeros((elevation_size, rx_count), dtype=np.complex64)

    for chirp_index, tx_index in enumerate(tx_indices[:chirps_per_loop]):
        grid[tx_index, :rx_count] = virtual_samples[chirp_index, :rx_count]
    return grid


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


def _spatial_bin_to_direction_cosine(bin_index: int, fft_size: int) -> float:
    # For half-wavelength antenna spacing, direction cosine is 2 * FFT frequency.
    direction_cosine = 2.0 * ((bin_index - (fft_size // 2)) / float(fft_size))
    return float(np.clip(direction_cosine, -1.0, 1.0))


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
