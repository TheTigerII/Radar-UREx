from pathlib import Path

import numpy as np

from main.livedatacapture import RadarCaptureConfig
from main.pmm import (
    LONG_RANGE_MAX_RANGE_M,
    PmmConfig,
    PmmTracker,
    _ods_virtual_coordinates,
    fixed_receive_beam_magnitude,
    validate_mini4_profile,
)


ROOT = Path(__file__).resolve().parents[1]


def load_profile(name: str) -> RadarCaptureConfig:
    return RadarCaptureConfig.from_file(ROOT / "profiles" / name)


def test_all_three_profile_mode_combinations_validate() -> None:
    baseline = load_profile("profile-mini4-20m.cfg")
    long_range = load_profile("profile-mini4-long-range.cfg")
    fixed_beam = load_profile("profile-mini4-fixed-beam.cfg")
    validate_mini4_profile(baseline, "baseline")
    validate_mini4_profile(long_range, "long-range-mimo")
    validate_mini4_profile(fixed_beam, "fixed-beam")
    assert 93.0 < float(long_range.range_axis_m()[-1]) < 94.0
    assert np.allclose(long_range.range_axis_m(), fixed_beam.range_axis_m())


def test_long_range_tracker_uses_requested_gate() -> None:
    config = load_profile("profile-mini4-long-range.cfg")
    tracker = PmmTracker(
        config,
        PmmConfig(operation_mode="long-range-mimo", max_range_m=LONG_RANGE_MAX_RANGE_M),
    )
    tracker._prepare_range_axis(config.range_axis_m())
    assert tracker.range_indices is not None
    assert float(config.range_axis_m()[tracker.range_indices[-1]]) >= 89.0


def test_fixed_beam_prefers_its_selected_direction() -> None:
    config = load_profile("profile-mini4-fixed-beam.cfg")
    coordinates, phase_corrections = _ods_virtual_coordinates(config)
    azimuth_deg = 20.0
    azimuth_rad = np.deg2rad(azimuth_deg)
    steering = np.exp(1j * np.pi * coordinates[:, 0] * np.sin(azimuth_rad))
    doppler_index = 42
    centered_bin = doppler_index - 32
    tx_offsets = np.arange(3, dtype=np.float32) / 3.0
    tdm_phase = np.exp(1j * 2.0 * np.pi * centered_bin / 64.0 * tx_offsets)
    channels = (
        steering.reshape(3, 4)
        * tdm_phase[:, np.newaxis]
        / phase_corrections.reshape(3, 4)
    )
    cube = np.zeros((64, 3, 4, 256), dtype=np.complex64)
    cube[doppler_index, :, :, 40] = channels
    on_axis = fixed_receive_beam_magnitude(
        cube, config, azimuth_deg=azimuth_deg, elevation_deg=0.0
    )[40, doppler_index]
    off_axis = fixed_receive_beam_magnitude(
        cube, config, azimuth_deg=-40.0, elevation_deg=0.0
    )[40, doppler_index]
    assert on_axis > 2.0 * off_axis
