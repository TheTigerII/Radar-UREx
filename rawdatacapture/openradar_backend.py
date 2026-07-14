"""Adapters between this project's cube layout and OpenRadar's DSP APIs."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np


OPENRADAR_INSTALL_HINT = (
    "OpenRadar DSP is required. Create the project virtual environment as "
    "described in 'rawdatacapture/User guide.md', install requirements.txt, "
    "and run the application with that environment's Python."
)


@lru_cache(maxsize=1)
def _openradar_dsp() -> Any:
    try:
        import mmwave.dsp as openradar_dsp
    except (ImportError, ModuleNotFoundError) as exc:
        raise RuntimeError(f"{OPENRADAR_INSTALL_HINT} Import error: {exc}") from exc
    return openradar_dsp


def validate_openradar_backend() -> str:
    """Import OpenRadar early and return a useful backend description."""
    openradar_dsp = _openradar_dsp()
    module_path = Path(openradar_dsp.__file__).resolve()
    return f"OpenRadar ({module_path})"


def range_fft(adc_cube: np.ndarray) -> np.ndarray:
    """Run OpenRadar's Hann-windowed range FFT."""
    openradar_dsp = _openradar_dsp()
    return openradar_dsp.range_processing(
        adc_cube,
        window_type_1d=openradar_dsp.Window.HANNING,
    )


def doppler_fft(
    range_cube: np.ndarray,
    *,
    num_tx_antennas: int,
) -> np.ndarray:
    """Return OpenRadar Doppler output as [doppler, tx, rx, range]."""
    if num_tx_antennas <= 0:
        raise ValueError("num_tx_antennas must be positive")
    if range_cube.shape[0] % num_tx_antennas:
        raise ValueError(
            "Chirp count must be divisible by the number of TDM transmitters: "
            f"chirps={range_cube.shape[0]}, tx={num_tx_antennas}"
        )

    openradar_dsp = _openradar_dsp()
    # OpenRadar also calculates an unused log2 magnitude matrix internally;
    # zero FFT cells can legitimately produce log2(0).
    with np.errstate(divide="ignore", invalid="ignore"):
        _det_matrix, aoa_input = openradar_dsp.doppler_processing(
            range_cube,
            num_tx_antennas=num_tx_antennas,
            clutter_removal_enabled=False,
            interleaved=True,
            window_type_2d=openradar_dsp.Window.HANNING,
            accumulate=False,
        )

    # OpenRadar returns [range, virtual_rx, doppler], with virtual receivers
    # grouped by chirp/TX order. Restore this project's explicit TX/RX axes.
    num_range_bins, num_virtual_antennas, num_doppler_bins = aoa_input.shape
    num_rx_antennas = range_cube.shape[1]
    expected_virtual_antennas = num_tx_antennas * num_rx_antennas
    if num_virtual_antennas != expected_virtual_antennas:
        raise ValueError(
            "Unexpected OpenRadar virtual antenna count: "
            f"got {num_virtual_antennas}, expected {expected_virtual_antennas}"
        )

    explicit_cube = aoa_input.transpose(2, 1, 0).reshape(
        num_doppler_bins,
        num_tx_antennas,
        num_rx_antennas,
        num_range_bins,
    )
    return np.fft.fftshift(explicit_cube, axes=0)


def ca_cfar_2d(
    power_map: np.ndarray,
    *,
    false_alarm_rate: float,
    range_guard_cells: int,
    doppler_guard_cells: int,
    range_training_cells: int,
    doppler_training_cells: int,
) -> np.ndarray:
    """Apply OpenRadar CA-CFAR independently in range and Doppler."""
    if power_map.ndim != 2 or power_map.size == 0:
        return np.zeros_like(power_map, dtype=bool)

    doppler_bins, range_bins = power_map.shape
    range_margin = range_guard_cells + range_training_cells
    doppler_margin = doppler_guard_cells + doppler_training_cells
    if range_bins <= 2 * range_margin or doppler_bins <= 2 * doppler_margin:
        return np.zeros_like(power_map, dtype=bool)

    openradar_dsp = _openradar_dsp()
    log_power_db = 10.0 * np.log10(
        np.maximum(power_map.astype(np.float64), np.finfo(np.float64).tiny)
    )
    pfa = min(max(false_alarm_rate, 1e-9), 0.5)

    def threshold_offset_db(training_cells_per_side: int) -> float:
        total_training_cells = 2 * training_cells_per_side
        scale = total_training_cells * (
            pfa ** (-1.0 / total_training_cells) - 1.0
        )
        return float(10.0 * np.log10(scale))

    def range_threshold(row: np.ndarray) -> np.ndarray:
        threshold, _noise = openradar_dsp.ca_(
            row,
            guard_len=range_guard_cells,
            noise_len=range_training_cells,
            mode="constant",
            l_bound=threshold_offset_db(range_training_cells),
        )
        return threshold

    def doppler_threshold(column: np.ndarray) -> np.ndarray:
        threshold, _noise = openradar_dsp.ca_(
            column,
            guard_len=doppler_guard_cells,
            noise_len=doppler_training_cells,
            mode="wrap",
            l_bound=threshold_offset_db(doppler_training_cells),
        )
        return threshold

    range_thresholds = np.apply_along_axis(range_threshold, 1, log_power_db)
    doppler_thresholds = np.apply_along_axis(doppler_threshold, 0, log_power_db)
    detections = (log_power_db > range_thresholds) & (
        log_power_db > doppler_thresholds
    )

    # OpenRadar's constant-padding range mode supplies thresholds at the edge.
    # Keep those bins disabled because they do not have a complete noise window.
    detections[:, :range_margin] = False
    detections[:, range_bins - range_margin :] = False
    return detections
