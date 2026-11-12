import json
from pathlib import Path

import numpy as np
import pytest

from main import calibrate
from main.inference import RealtimeUavClassifier
from main.livedatacapture import RadarCaptureConfig
from main.pmm import (
    LITERATURE_TBD_MINIMUM_FRAMES,
    LITERATURE_TBD_NORMALIZED_THRESHOLD,
    LITERATURE_TBD_OPERATION_MODE,
    LONG_RANGE_MAX_RANGE_M,
    MODE3_BEAMSPACE_INTEGRATED_THRESHOLD,
    MODE3_BEAMSPACE_MINIMUM_FRAMES,
    PmmConfig,
    PmmTracker,
    _ods_virtual_coordinates,
    _ods_rx_coordinates,
    fixed_receive_beam_magnitude,
    mode3_beam_space_frame,
    mode3_track_before_detect,
    normalized_track_before_detect,
    phased_stt_receive_magnitude,
    sector_receive_beam_magnitude,
    single_tx_receive_scan,
    validate_mini4_profile,
)
from main.phased_array import (
    create_phased_stt_runtime_profile,
    encode_tx1_phase_word,
    tx1_steering_phase_deg,
)


ROOT = Path(__file__).resolve().parents[1]


def load_profile(name: str) -> RadarCaptureConfig:
    return RadarCaptureConfig.from_file(ROOT / "profiles" / name)


@pytest.mark.parametrize(
    ("profile_name", "operation_mode"),
    (
        ("profile-mini4-fixed-beam.cfg", "fixed-beam"),
        ("profile-mini4-phased-stt.cfg", "phased-array-stt"),
        ("profile-mini4-literature-tbd.cfg", LITERATURE_TBD_OPERATION_MODE),
    ),
)
def test_classifier_allows_logged_cross_profile_inference_for_test_modes(
    profile_name: str,
    operation_mode: str,
) -> None:
    config = load_profile(profile_name)
    tracker = PmmTracker(config, PmmConfig(operation_mode=operation_mode))
    manifest = json.loads(
        (ROOT / "model_weights" / "manifest.json").read_text(encoding="utf-8")
    )
    mismatches = RealtimeUavClassifier._validate_manifest(
        manifest,
        tracker.metadata,
        allow_cross_profile=True,
    )
    assert set(mismatches) == {
        "feature_version",
        "feature_fingerprint",
        "profile_fingerprint",
    }
    with pytest.raises(ValueError, match="dataset contract mismatch"):
        RealtimeUavClassifier._validate_manifest(manifest, tracker.metadata)


def synthetic_mimo_frame(
    config: RadarCaptureConfig,
    *,
    range_bin: int,
    azimuth_deg: float,
    elevation_deg: float,
    amplitude: float = 1.0,
) -> np.ndarray:
    """Return range-FFT chirps for one calibrated far-field MIMO target."""

    coordinates, phase_corrections = _ods_virtual_coordinates(config)
    azimuth_rad = np.deg2rad(azimuth_deg)
    elevation_rad = np.deg2rad(elevation_deg)
    direction_x = np.sin(azimuth_rad) * np.cos(elevation_rad)
    direction_z = -np.sin(elevation_rad)
    steering = np.exp(
        1j
        * np.pi
        * (coordinates[:, 0] * direction_x + coordinates[:, 1] * direction_z)
    )
    channels = (steering / phase_corrections).reshape(3, 4) * amplitude
    frame = np.zeros((96, 4, 256), dtype=np.complex64)
    for loop_index in range(32):
        frame[loop_index * 3 : loop_index * 3 + 3, :, range_bin] = channels
    return frame


def synthetic_mode3_doppler_cube(
    config: RadarCaptureConfig,
    *,
    range_bin: int,
    doppler_index: int,
    azimuth_deg: float,
    elevation_deg: float,
    amplitude: float,
) -> np.ndarray:
    """Return a calibrated target after Mode 3 TDM Doppler compensation."""

    coordinates, phase_corrections = _ods_virtual_coordinates(config)
    azimuth_rad = np.deg2rad(azimuth_deg)
    elevation_rad = np.deg2rad(elevation_deg)
    steering = np.exp(
        1j
        * np.pi
        * (
            coordinates[:, 0] * np.sin(azimuth_rad) * np.cos(elevation_rad)
            + coordinates[:, 1] * -np.sin(elevation_rad)
        )
    )
    centered_doppler_bin = doppler_index - 32
    tx_offsets = np.arange(3, dtype=np.float32) / 3.0
    compensation = np.exp(
        -1j * 2.0 * np.pi * centered_doppler_bin / 64.0 * tx_offsets
    )
    physical = (
        steering.reshape(3, 4)
        / phase_corrections.reshape(3, 4)
        / compensation[:, np.newaxis]
    )
    cube = np.zeros((64, 3, 4, 256), dtype=np.complex64)
    cube[doppler_index, :, :, range_bin] = physical * amplitude
    return cube


def test_mode3_beam_space_tbd_accepts_persistent_angle_range_path() -> None:
    evidence = np.zeros((MODE3_BEAMSPACE_MINIMUM_FRAMES, 3, 5, 12), dtype=np.float32)
    for frame_index in range(evidence.shape[0]):
        evidence[frame_index, 1, 2, 4 + frame_index // 3] = (
            MODE3_BEAMSPACE_INTEGRATED_THRESHOLD + 0.5
        )
    cell, integrated, detected = mode3_track_before_detect(evidence, 1)
    assert detected
    assert cell == (1, 2, 5)
    assert integrated >= MODE3_BEAMSPACE_INTEGRATED_THRESHOLD


def test_mode3_beam_space_tbd_rejects_stale_spike() -> None:
    evidence = np.zeros((MODE3_BEAMSPACE_MINIMUM_FRAMES, 3, 5, 12), dtype=np.float32)
    evidence[0, 1, 2, 4] = 30.0
    _cell, _integrated, detected = mode3_track_before_detect(evidence, 1)
    assert not detected


def test_all_three_profile_mode_combinations_validate() -> None:
    baseline = load_profile("profile-mini4-20m.cfg")
    long_range = load_profile("profile-mini4-long-range.cfg")
    fixed_beam = load_profile("profile-mini4-fixed-beam.cfg")
    validate_mini4_profile(baseline, "baseline")
    validate_mini4_profile(long_range, "long-range-mimo")
    validate_mini4_profile(fixed_beam, "fixed-beam")
    assert 93.0 < float(long_range.range_axis_m()[-1]) < 94.0
    assert np.allclose(
        np.diff(long_range.range_axis_m()),
        np.diff(fixed_beam.range_axis_m()),
        atol=1e-5,
    )


def test_literature_tbd_profile_is_long_range_three_tx_mimo() -> None:
    config = load_profile("profile-mini4-literature-tbd.cfg")
    validate_mini4_profile(config, LITERATURE_TBD_OPERATION_MODE)
    assert config.tx_channel_masks == (1, 4, 2)
    assert config.num_chirps_per_frame == 96
    assert 93.0 < float(config.range_axis_m()[-1]) < 94.0


def test_track_before_detect_accepts_persistent_subthreshold_path() -> None:
    scores = np.zeros((12, 10), dtype=np.float32)
    scores[:, 4] = LITERATURE_TBD_NORMALIZED_THRESHOLD + 0.05
    path, integrated_score, detected = normalized_track_before_detect(
        scores,
        maximum_step_bins=1,
    )
    assert detected
    assert np.all(path == 4)
    assert integrated_score < 1.0


def test_track_before_detect_rejects_old_spike_without_current_evidence() -> None:
    scores = np.zeros((LITERATURE_TBD_MINIMUM_FRAMES, 10), dtype=np.float32)
    scores[0, 4] = 8.0
    path, _integrated_score, detected = normalized_track_before_detect(
        scores,
        maximum_step_bins=1,
    )
    assert path[-1] in range(10)
    assert not detected


def test_literature_tbd_tracker_integrates_persistent_subthreshold_target() -> None:
    """Mode 5 must acquire weak persistent evidence without a one-frame hit."""

    config = load_profile("profile-mini4-literature-tbd.cfg")
    tracker = PmmTracker(
        config,
        PmmConfig(
            operation_mode=LITERATURE_TBD_OPERATION_MODE,
            max_range_m=LONG_RANGE_MAX_RANGE_M,
            background_calibration_seconds=0.1,
            detection_threshold=1.0,
            provisional_frames=1,
            confirmation_window_frames=1,
            confirmation_hits=1,
        ),
    )
    range_axis = config.range_axis_m()
    empty_range_fft = np.zeros((96, 4, 256), dtype=np.complex64)
    empty_cube = np.zeros((64, 3, 4, 256), dtype=np.complex64)
    assert tracker.update(empty_cube, empty_range_fft, range_axis).state == "searching"

    target_range_m = 20.0
    range_bin = int(np.argmin(np.abs(range_axis - target_range_m)))
    target_range_fft = synthetic_mimo_frame(
        config,
        range_bin=range_bin,
        azimuth_deg=12.0,
        elevation_deg=5.0,
        amplitude=0.04,
    )
    target_cube = np.fft.fftshift(
        np.fft.fft(
            target_range_fft.reshape(32, 3, 4, 256),
            n=64,
            axis=0,
        ),
        axes=0,
    ).astype(np.complex64)

    result = None
    for _ in range(LITERATURE_TBD_MINIMUM_FRAMES - 1):
        result = tracker.update(target_cube, target_range_fft, range_axis)
        assert not result.has_track
    result = tracker.update(target_cube, target_range_fft, range_axis)

    assert result.has_track
    assert result.pmm_score is not None
    assert result.pmm_score < result.threshold
    assert result.tbd_integrated_score is not None
    assert result.tbd_integrated_score >= LITERATURE_TBD_NORMALIZED_THRESHOLD
    assert result.tbd_window_frames == LITERATURE_TBD_MINIMUM_FRAMES
    assert result.range_m == pytest.approx(target_range_m, abs=0.15)
    assert result.azimuth_deg == pytest.approx(12.0, abs=3.0)
    assert result.elevation_deg == pytest.approx(5.0, abs=3.0)


def test_literature_tbd_metadata_records_integration_and_angles() -> None:
    config = load_profile("profile-mini4-literature-tbd.cfg")
    tracker = PmmTracker(
        config,
        PmmConfig(
            operation_mode=LITERATURE_TBD_OPERATION_MODE,
            max_range_m=LONG_RANGE_MAX_RANGE_M,
        ),
    )
    metadata = tracker.metadata
    assert metadata["operation_mode"] == LITERATURE_TBD_OPERATION_MODE
    assert metadata["config"]["max_range_m"] == LONG_RANGE_MAX_RANGE_M
    assert metadata["feature_version"]


def test_single_tx_scan_profile_validates_as_non_mimo() -> None:
    config = load_profile("profile-mini4-simo-scan.cfg")
    validate_mini4_profile(config, "single-tx-scan")
    assert config.num_chirps_per_loop == 1
    assert config.tx_channel_masks == (1,)
    assert config.num_chirps_per_frame == 32


def test_phased_stt_profile_uses_exactly_two_simultaneous_transmitters() -> None:
    config = load_profile("profile-mini4-phased-stt.cfg")
    validate_mini4_profile(config, "phased-array-stt")
    assert config.num_chirps_per_loop == 1
    assert config.tx_channel_masks == (3,)
    assert config.num_chirps_per_frame == 32
    assert config.tx_phase_shifter == 0


def test_phased_stt_runtime_profile_quantizes_and_records_tx_phase(tmp_path) -> None:
    source = ROOT / "profiles" / "profile-mini4-phased-stt.cfg"
    result = create_phased_stt_runtime_profile(
        source,
        tmp_path / "runtime.cfg",
        azimuth_deg=10.0,
        elevation_deg=0.0,
    )
    config = RadarCaptureConfig.from_file(result.path)
    validate_mini4_profile(config, "phased-array-stt")
    assert config.tx_channel_masks == (3,)
    assert config.tx_phase_shifter == result.packed_phase_word
    assert result.tx1_phase_code in range(64)
    assert "hostPhasedStt" in result.path.read_text(encoding="utf-8")


def test_phased_stt_phase_word_has_expected_boresight_and_scan_behavior() -> None:
    assert tx1_steering_phase_deg(0.0, 0.0) == 0.0
    assert np.isclose(tx1_steering_phase_deg(30.0, 0.0), -180.0)
    code, word, programmed = encode_tx1_phase_word(-180.0)
    assert code == 32
    assert word == 32 << 8
    assert programmed == 180.0


def test_phased_stt_receive_beam_has_four_channel_coherent_gain() -> None:
    config = load_profile("profile-mini4-phased-stt.cfg")
    coordinates, phase_corrections = _ods_rx_coordinates(config)
    azimuth_deg = 10.0
    elevation_deg = 0.0
    direction_x = np.sin(np.deg2rad(azimuth_deg))
    steering = np.exp(1j * np.pi * coordinates[:, 0] * direction_x)
    cube = np.zeros((64, 1, 4, 256), dtype=np.complex64)
    cube[32, 0, :, 40] = steering / phase_corrections
    coherent = phased_stt_receive_magnitude(
        cube,
        config,
        azimuth_deg=azimuth_deg,
        elevation_deg=elevation_deg,
    )[40, 32]
    noncoherent = np.abs(
        cube[32, 0, :, 40] * phase_corrections
    ).mean()
    assert np.isclose(coherent / noncoherent, 2.0, rtol=1e-5)


def test_phased_stt_tracker_uses_selected_stt_direction() -> None:
    config = load_profile("profile-mini4-phased-stt.cfg")
    tracker = PmmTracker(
        config,
        PmmConfig(
            operation_mode="phased-array-stt",
            beam_azimuth_deg=10.0,
            beam_elevation_deg=5.0,
            background_calibration_seconds=0.1,
            detection_threshold=1.0,
            provisional_frames=1,
            confirmation_window_frames=1,
            confirmation_hits=1,
        ),
    )
    range_axis = config.range_axis_m()
    range_fft = np.zeros((32, 4, 256), dtype=np.complex64)
    empty_cube = np.zeros((64, 1, 4, 256), dtype=np.complex64)
    assert tracker.update(empty_cube, range_fft, range_axis).state == "searching"
    coordinates, phase_corrections = _ods_rx_coordinates(config)
    azimuth_rad = np.deg2rad(10.0)
    elevation_rad = np.deg2rad(5.0)
    steering = np.exp(
        1j
        * np.pi
        * (
            coordinates[:, 0] * np.sin(azimuth_rad) * np.cos(elevation_rad)
            + coordinates[:, 1] * -np.sin(elevation_rad)
        )
    )
    signal_cube = empty_cube.copy()
    signal_cube[32, 0, :, 40] = steering / phase_corrections * 1e4
    result = tracker.update(signal_cube, range_fft, range_axis)
    assert result.has_track
    assert result.azimuth_deg == 10.0
    assert result.elevation_deg == 5.0


def test_single_tx_receive_scan_finds_synthetic_direction() -> None:
    config = load_profile("profile-mini4-simo-scan.cfg")
    coordinates, phase_corrections = _ods_virtual_coordinates(config)
    coordinates = coordinates[:4] - coordinates[:4].mean(axis=0, keepdims=True)
    target_azimuth_deg = 30.0
    target_elevation_deg = 20.0
    azimuth_rad = np.deg2rad(target_azimuth_deg)
    elevation_rad = np.deg2rad(target_elevation_deg)
    steering = np.exp(
        1j
        * np.pi
        * (
            coordinates[:, 0]
            * np.sin(azimuth_rad)
            * np.cos(elevation_rad)
            + coordinates[:, 1] * -np.sin(elevation_rad)
        )
    )
    cube = np.zeros((64, 1, 4, 256), dtype=np.complex64)
    cube[32, 0, :, 40] = steering / phase_corrections[:4]
    magnitude, beam_bank, azimuths, elevations = single_tx_receive_scan(
        cube,
        config,
        angle_step_deg=10.0,
    )
    winning = int(np.argmax(beam_bank[32, :, 40]))
    assert azimuths[winning] == target_azimuth_deg
    assert elevations[winning] == target_elevation_deg
    assert magnitude.shape == (256, 64)


def test_single_tx_tracker_reports_range_bearing_and_elevation() -> None:
    config = load_profile("profile-mini4-simo-scan.cfg")
    tracker = PmmTracker(
        config,
        PmmConfig(
            operation_mode="single-tx-scan",
            background_calibration_seconds=0.1,
            detection_threshold=1.0,
            provisional_frames=1,
            confirmation_window_frames=1,
            confirmation_hits=1,
        ),
    )
    range_axis = config.range_axis_m()
    range_fft = np.zeros((32, 4, 256), dtype=np.complex64)
    empty_cube = np.zeros((64, 1, 4, 256), dtype=np.complex64)
    assert tracker.update(empty_cube, range_fft, range_axis).state == "searching"

    coordinates, phase_corrections = _ods_virtual_coordinates(config)
    coordinates = coordinates[:4] - coordinates[:4].mean(axis=0, keepdims=True)
    azimuth_deg = 20.0
    elevation_deg = 10.0
    azimuth_rad = np.deg2rad(azimuth_deg)
    elevation_rad = np.deg2rad(elevation_deg)
    steering = np.exp(
        1j
        * np.pi
        * (
            coordinates[:, 0]
            * np.sin(azimuth_rad)
            * np.cos(elevation_rad)
            + coordinates[:, 1] * -np.sin(elevation_rad)
        )
    )
    signal_cube = empty_cube.copy()
    signal_cube[32, 0, :, 40] = steering / phase_corrections[:4] * 1e4
    result = tracker.update(signal_cube, range_fft, range_axis)
    assert result.has_track
    assert result.range_m is not None
    assert result.azimuth_deg is not None
    assert result.elevation_deg is not None
    assert abs(result.azimuth_deg - azimuth_deg) < 5.0
    assert abs(result.elevation_deg - elevation_deg) < 5.0


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


def test_sector_beam_recovers_target_away_from_manual_centre() -> None:
    config = load_profile("profile-mini4-fixed-beam.cfg")
    coordinates, phase_corrections = _ods_virtual_coordinates(config)
    target_azimuth_deg = 20.0
    steering = np.exp(
        1j
        * np.pi
        * coordinates[:, 0]
        * np.sin(np.deg2rad(target_azimuth_deg))
    )
    cube = np.zeros((64, 3, 4, 256), dtype=np.complex64)
    cube[32, :, :, 40] = (steering / phase_corrections).reshape(3, 4)
    single_centre = fixed_receive_beam_magnitude(
        cube, config, azimuth_deg=0.0, elevation_deg=0.0
    )[40, 32]
    sector = sector_receive_beam_magnitude(
        cube,
        config,
        center_azimuth_deg=0.0,
        center_elevation_deg=0.0,
    )[40, 32]
    assert sector > 2.0 * single_centre


def test_mode3_beam_bank_covers_full_baseline_field_of_view() -> None:
    config = load_profile("profile-mini4-fixed-beam.cfg")
    coordinates, phase_corrections = _ods_virtual_coordinates(config)
    azimuth_deg = 50.0
    elevation_deg = 30.0
    azimuth_rad = np.deg2rad(azimuth_deg)
    elevation_rad = np.deg2rad(elevation_deg)
    steering = np.exp(
        1j
        * np.pi
        * (
            coordinates[:, 0] * np.sin(azimuth_rad) * np.cos(elevation_rad)
            + coordinates[:, 1] * -np.sin(elevation_rad)
        )
    )
    cube = np.zeros((64, 3, 4, 256), dtype=np.complex64)
    cube[32, :, :, 40] = (steering / phase_corrections).reshape(3, 4)
    sector = sector_receive_beam_magnitude(
        cube,
        config,
        center_azimuth_deg=0.0,
        center_elevation_deg=0.0,
    )[40, 32]
    on_target = fixed_receive_beam_magnitude(
        cube,
        config,
        azimuth_deg=azimuth_deg,
        elevation_deg=elevation_deg,
    )[40, 32]
    assert sector >= 0.99 * on_target


def test_mode3_tracker_reports_beam_space_direction_and_classifier_shape() -> None:
    config = load_profile("profile-mini4-fixed-beam.cfg")
    tracker = PmmTracker(
        config,
        PmmConfig(
            operation_mode="fixed-beam",
            beam_azimuth_deg=0.0,
            beam_elevation_deg=0.0,
            background_calibration_seconds=0.1,
            detection_threshold=1.0,
            provisional_frames=1,
            confirmation_window_frames=1,
            confirmation_hits=1,
        ),
    )
    range_axis = config.range_axis_m()
    range_fft = np.zeros((96, 4, 256), dtype=np.complex64)
    empty_cube = np.zeros((64, 3, 4, 256), dtype=np.complex64)
    calibration = tracker.update(empty_cube, range_fft, range_axis)
    assert calibration.state == "searching"
    assert tracker.mode3_background is not None
    assert tracker.mode3_noise_scale is not None
    assert tracker.angle_az_background is None
    assert tracker.angle_el_background is None

    target_azimuth_deg = 12.0
    target_elevation_deg = -4.0
    signal_cube = synthetic_mode3_doppler_cube(
        config,
        range_bin=40,
        doppler_index=36,
        azimuth_deg=target_azimuth_deg,
        elevation_deg=target_elevation_deg,
        amplitude=1e4,
    )
    target_range_fft = synthetic_mimo_frame(
        config,
        range_bin=40,
        azimuth_deg=target_azimuth_deg,
        elevation_deg=target_elevation_deg,
        amplitude=1e4,
    )
    original_range_fft = target_range_fft.copy()
    original_signal_cube = signal_cube.copy()
    result = None
    # Six frames establish the path; continue until the classifier's rolling
    # 36-frame segment is full.
    for _ in range(42):
        result = tracker.update(signal_cube, target_range_fft, range_axis)
    assert result is not None
    assert result.has_track
    assert result.azimuth_deg is not None
    assert result.elevation_deg is not None
    assert abs(result.azimuth_deg - target_azimuth_deg) <= 3.0
    assert abs(result.elevation_deg - target_elevation_deg) <= 3.0
    assert result.azimuth_deg != tracker.config.beam_azimuth_deg
    assert result.elevation_deg != tracker.config.beam_elevation_deg
    assert result.label == "Beam-space target"
    assert result.tbd_integrated_score is not None
    assert tracker.spectrogram_db.shape == (64, 36)
    assert np.array_equal(target_range_fft, original_range_fft)
    assert np.array_equal(signal_cube, original_signal_cube)


def test_mode3_metadata_preserves_existing_calibration_and_classifier_contract() -> None:
    config = load_profile("profile-mini4-fixed-beam.cfg")
    tracker = PmmTracker(config, PmmConfig(operation_mode="fixed-beam"))
    metadata = tracker.metadata
    receive = metadata["config"]["operation_mode"]
    assert receive == "fixed-beam"

    # The detailed contract contributes to the fingerprint and documents the
    # deliberate compatibility boundary used by the cross-profile classifier.
    feature_values = metadata["feature_contract"]
    assert feature_values["receive_beamforming"]["transmit_schedule"] == (
        "existing calibrated 3-TX TDM"
    )
    assert not feature_values["receive_beamforming"][
        "new_hardware_calibration_required"
    ]
    assert not feature_values["track_before_detect"]["per_frame_cfar_or_pmm"]
    assert feature_values["classification_feature"]["shape"] == [36, 64]
    assert not feature_values["classification_feature"]["dataset_contract_changed"]


def test_fixed_beam_has_expected_coherent_gain_when_calibrated() -> None:
    config = load_profile("profile-mini4-fixed-beam.cfg")
    coordinates, phase_corrections = _ods_virtual_coordinates(config)
    cube = np.zeros((64, 3, 4, 256), dtype=np.complex64)
    cube[32, :, :, 40] = (np.ones(12) / phase_corrections).reshape(3, 4)
    coherent = fixed_receive_beam_magnitude(
        cube, config, azimuth_deg=0.0, elevation_deg=0.0
    )[40, 32]
    noncoherent = np.abs(cube[32, :, :, 40] * phase_corrections.reshape(3, 4)).mean()
    assert np.isclose(coherent / noncoherent, np.sqrt(12.0), rtol=1e-5)


def test_fixed_beam_profile_is_valid_same_waveform_calibration_source(tmp_path) -> None:
    source = ROOT / "profiles" / "profile-mini4-fixed-beam.cfg"
    source_config = RadarCaptureConfig.from_file(source)
    settings = calibrate.CalibrationSettings(
        target_distance_m=1.5,
        search_window_m=0.75,
    )
    runtime = tmp_path / "fixed_beam_calibration.cfg"
    calibrate.create_runtime_profile(source, runtime, source_config, settings)
    runtime_config = RadarCaptureConfig.from_file(runtime)
    calibrate.validate_calibration_profile_text(
        runtime.read_text(encoding="utf-8"),
        runtime_config,
        settings,
        require_raw_lvds=True,
    )
    assert runtime_config.start_frequency_ghz == source_config.start_frequency_ghz
    assert runtime_config.frequency_slope_mhz_per_us == source_config.frequency_slope_mhz_per_us
    candidates = np.flatnonzero(
        (runtime_config.range_axis_m() >= 0.75)
        & (runtime_config.range_axis_m() <= 2.25)
    )
    assert candidates.size >= 3


def test_fixed_beam_same_waveform_calibration_recovers_channel_alignment() -> None:
    config = load_profile("profile-mini4-fixed-beam.cfg")
    settings = calibrate.CalibrationSettings(
        target_distance_m=1.5,
        search_window_m=0.75,
        warmup_frames=0,
        accepted_frames=3,
    )
    accumulator = calibrate.CalibrationAccumulator(config, settings)
    physical_channels = np.linspace(0.7, 1.0, 12) * np.exp(
        1j * np.linspace(0.0, 1.8, 12)
    )
    range_shape = np.zeros(256, dtype=np.float64)
    range_shape[3:6] = (0.15, 1.0, 0.20)
    frame = np.zeros((96, 4, 256), dtype=np.complex128)
    tx_masks = tuple(config.tx_channel_masks or ())
    for loop_index in range(32):
        for chirp_index, mask in enumerate(tx_masks):
            physical_tx = int(mask).bit_length() - 1
            channels = physical_channels[physical_tx * 4 : (physical_tx + 1) * 4]
            frame[loop_index * 3 + chirp_index] = (
                channels[:, np.newaxis] * range_shape[np.newaxis, :]
            )
    for _ in range(settings.accepted_frames):
        accumulator.update(frame)
    assert accumulator.result is not None
    corrected = np.asarray(accumulator.result.coefficients) * physical_channels
    assert np.allclose(corrected, corrected[0], rtol=1e-6, atol=1e-6)
    assert accumulator.result.max_phase_std_deg < 1e-6


def test_fixed_beam_same_waveform_calibration_recovers_channel_alignment() -> None:
    config = load_profile("profile-mini4-fixed-beam.cfg")
    settings = calibrate.CalibrationSettings(
        target_distance_m=1.5,
        search_window_m=0.75,
        warmup_frames=0,
        accepted_frames=3,
    )
    accumulator = calibrate.CalibrationAccumulator(config, settings)
    physical_channels = np.linspace(0.7, 1.0, 12) * np.exp(
        1j * np.linspace(0.0, 1.8, 12)
    )
    range_shape = np.zeros(256, dtype=np.float64)
    range_shape[3:6] = (0.15, 1.0, 0.20)
    frame = np.zeros((96, 4, 256), dtype=np.complex128)
    tx_masks = tuple(config.tx_channel_masks or ())
    for loop_index in range(32):
        for chirp_index, mask in enumerate(tx_masks):
            physical_tx = int(mask).bit_length() - 1
            channels = physical_channels[physical_tx * 4 : (physical_tx + 1) * 4]
            frame[loop_index * 3 + chirp_index] = (
                channels[:, np.newaxis] * range_shape[np.newaxis, :]
            )
    for _ in range(settings.accepted_frames):
        accumulator.update(frame)
    assert accumulator.result is not None
    corrected = np.asarray(accumulator.result.coefficients) * physical_channels
    assert np.allclose(corrected, corrected[0], rtol=1e-6, atol=1e-6)
    assert accumulator.result.max_phase_std_deg < 1e-6
