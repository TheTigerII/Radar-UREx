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
    doppler_fft = np.fft.fftshift(np.fft.fft(tdm_cube, axis=0), axes=0)
    magnitude = np.abs(doppler_fft).mean(axis=(1, 2))
    return 20.0 * np.log10(magnitude + 1e-6)


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
