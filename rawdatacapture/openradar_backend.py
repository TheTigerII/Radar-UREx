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


def os_cfar_2d(
    power_map: np.ndarray,
    *,
    false_alarm_rate: float,
    range_guard_cells: int,
    doppler_guard_cells: int,
    range_training_cells: int,
    doppler_training_cells: int,
) -> np.ndarray:
    """Apply vectorized OS-CFAR independently in range and Doppler."""
    if power_map.ndim != 2 or power_map.size == 0:
        return np.zeros_like(power_map, dtype=bool)

    doppler_bins, range_bins = power_map.shape
    range_margin = range_guard_cells + range_training_cells
    doppler_margin = doppler_guard_cells + doppler_training_cells
    if range_bins <= 2 * range_margin or doppler_bins <= 2 * doppler_margin:
        return np.zeros_like(power_map, dtype=bool)

    power = np.maximum(power_map.astype(np.float64), 0.0)
    pfa = min(max(false_alarm_rate, 1e-9), 0.5)

    def os_parameters(training_cells_per_side: int) -> tuple[int, float]:
        total_training_cells = 2 * training_cells_per_side
        # Use the conventional 75th-percentile ordered statistic. OpenRadar's
        # rank is zero-based, while the Pfa calculation uses a one-based rank.
        rank = max(1, int(np.ceil(0.75 * total_training_cells)))
        return rank - 1, _os_scale(total_training_cells, rank, pfa)

    range_rank, range_scale = os_parameters(range_training_cells)
    doppler_rank, doppler_scale = os_parameters(doppler_training_cells)
    range_thresholds = _os_thresholds_along_axis(
        power,
        axis=1,
        guard_cells=range_guard_cells,
        training_cells=range_training_cells,
        rank_index=range_rank,
        scale=range_scale,
    )
    doppler_thresholds = _os_thresholds_along_axis(
        power,
        axis=0,
        guard_cells=doppler_guard_cells,
        training_cells=doppler_training_cells,
        rank_index=doppler_rank,
        scale=doppler_scale,
    )
    detections = (power > range_thresholds) & (power > doppler_thresholds)

    # Doppler is cyclic. Range uses the same wrapped calculation internally,
    # then disables cells where a complete non-cyclic window is unavailable.
    detections[:, :range_margin] = False
    detections[:, range_bins - range_margin :] = False
    return detections


def _os_thresholds_along_axis(
    power: np.ndarray,
    *,
    axis: int,
    guard_cells: int,
    training_cells: int,
    rank_index: int,
    scale: float,
) -> np.ndarray:
    """Calculate cyclic OS-CFAR thresholds for every cell without Python loops."""
    offsets = np.concatenate(
        (
            np.arange(-guard_cells - training_cells, -guard_cells),
            np.arange(guard_cells + 1, guard_cells + training_cells + 1),
        )
    )
    axis_size = power.shape[axis]
    window_indices = (
        np.arange(axis_size, dtype=np.intp)[:, np.newaxis] + offsets
    ) % axis_size

    oriented_power = np.moveaxis(power, axis, -1)
    training_windows = np.take(oriented_power, window_indices, axis=-1)
    ordered_noise = np.partition(
        training_windows,
        rank_index,
        axis=-1,
    )[..., rank_index]
    return np.moveaxis(ordered_noise, -1, axis) * scale


def _os_scale(total_training_cells: int, rank: int, pfa: float) -> float:
    """Return the exponential-noise OS-CFAR multiplier for a requested Pfa."""
    if not 1 <= rank <= total_training_cells:
        raise ValueError("OS-CFAR rank must be within the training window")

    target_log_pfa = float(np.log(pfa))

    def log_pfa(scale: float) -> float:
        indices = np.arange(rank, dtype=np.float64)
        denominators = total_training_cells - indices
        return float(-np.log1p(scale / denominators).sum())

    lower = 0.0
    upper = 1.0
    while log_pfa(upper) > target_log_pfa:
        upper *= 2.0

    for _ in range(64):
        midpoint = (lower + upper) / 2.0
        if log_pfa(midpoint) > target_log_pfa:
            lower = midpoint
        else:
            upper = midpoint
    return (lower + upper) / 2.0
