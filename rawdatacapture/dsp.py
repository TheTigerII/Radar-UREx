"""Minimal DSP primitives used by the Mini4 Phase 1 PMM pipeline."""

from __future__ import annotations

from functools import lru_cache
from typing import Optional, Protocol

import numpy as np
from scipy import fft as scipy_fft


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


@lru_cache(maxsize=8)
def _hann_window(length: int) -> np.ndarray:
    return np.hanning(length).astype(np.float32)


def frame_bytes_to_radar_cube(
    frame_bytes: bytes,
    config: RadarDspConfig,
) -> np.ndarray:
    """Decode DCA1000 two-lane complex int16 data as [chirp, rx, sample].

    The non-interleaved layout follows section 24.8 of TI's mmWave Studio
    guide: all samples for one receiver are contiguous within each chirp.
    """
    expected_int16_count = config.bytes_per_frame // np.dtype("<i2").itemsize
    adc_samples = np.frombuffer(
        frame_bytes,
        dtype="<i2",
        count=expected_int16_count,
    )
    if adc_samples.size != expected_int16_count:
        raise ValueError(
            f"Frame has {adc_samples.size} int16 values; "
            f"expected {expected_int16_count}"
        )
    if adc_samples.size % 4:
        raise ValueError("Complex two-lane LVDS data requires int16 groups of four")
    if config.lvds_lanes != 2:
        raise NotImplementedError(
            f"Only two-lane DCA1000 LVDS is supported, got {config.lvds_lanes}"
        )

    grouped = adc_samples.reshape(-1, 4)
    first = grouped[:, :2].reshape(-1).astype(np.float32)
    second = grouped[:, 2:].reshape(-1).astype(np.float32)
    complex_samples = (
        second + 1j * first
        if config.iq_swap
        else first + 1j * second
    ).astype(np.complex64, copy=False)

    expected_complex_count = (
        config.num_chirps_per_frame
        * config.num_rx_channels
        * config.num_adc_samples
    )
    if complex_samples.size != expected_complex_count:
        raise ValueError(
            f"Frame contains {complex_samples.size} complex samples; "
            f"expected {expected_complex_count}"
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


def compute_range_fft(radar_cube: np.ndarray) -> np.ndarray:
    if radar_cube.ndim != 3:
        raise ValueError("Radar cube must have shape [chirp, rx, sample]")
    window = _hann_window(radar_cube.shape[-1])
    return scipy_fft.fft(
        np.asarray(radar_cube, dtype=np.complex64) * window,
        axis=-1,
        workers=4,
    ).astype(np.complex64, copy=False)


def compute_range_doppler_fft(
    range_fft: np.ndarray,
    config: RadarDspConfig,
    *,
    fft_size: int = 64,
) -> np.ndarray:
    """Return DC-clutter-suppressed TDM data as [doppler, tx, rx, range]."""
    loops = int(config.num_loops or 0)
    tx_count = int(config.num_chirps_per_loop or 0)
    if loops <= 0 or tx_count <= 0:
        raise ValueError("Doppler FFT requires loop and TDM transmitter counts")
    if range_fft.shape[0] != loops * tx_count:
        raise ValueError("Range FFT chirp count does not match the TDM profile")
    if fft_size < loops:
        raise ValueError("Doppler FFT size cannot be smaller than loop count")
    explicit = np.asarray(range_fft, dtype=np.complex64).reshape(
        loops,
        tx_count,
        config.num_rx_channels,
        config.num_adc_samples,
    )
    dc_removed = explicit - explicit.mean(axis=0, keepdims=True)
    windowed = dc_removed * _hann_window(loops)[:, None, None, None]
    transformed = scipy_fft.fft(
        windowed,
        n=fft_size,
        axis=0,
        workers=4,
    )
    return scipy_fft.fftshift(transformed, axes=0).astype(
        np.complex64,
        copy=False,
    )


def compute_range_profile(range_fft: np.ndarray) -> np.ndarray:
    return np.mean(np.abs(range_fft), axis=(0, 1)).astype(np.float32)


def compute_range_doppler_heatmap(
    range_fft: np.ndarray,
    config: RadarDspConfig,
) -> np.ndarray:
    cube = compute_range_doppler_fft(range_fft, config, fft_size=64)
    return np.mean(np.abs(cube), axis=(1, 2)).astype(np.float32)


def range_resolution_m(config: RadarDspConfig) -> Optional[float]:
    if (
        config.sample_rate_ksps is None
        or config.frequency_slope_mhz_per_us is None
        or config.frequency_slope_mhz_per_us <= 0.0
    ):
        return None
    return (
        SPEED_OF_LIGHT_M_PER_S
        * config.sample_rate_ksps
        * 1e3
        / (
            2.0
            * config.frequency_slope_mhz_per_us
            * 1e12
            * config.num_adc_samples
        )
    )


def range_axis_m(config: RadarDspConfig) -> Optional[np.ndarray]:
    resolution = range_resolution_m(config)
    if resolution is None:
        return None
    return np.arange(config.num_adc_samples, dtype=np.float32) * resolution
