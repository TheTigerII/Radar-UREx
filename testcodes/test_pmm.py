import unittest
import json
import queue
import tempfile
import threading
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from main.pmm import (
    MINI4_DOPPLER_FFT_SIZE,
    MINI4_ADC_START_TIME_US,
    MINI4_FRAME_PERIODICITY_MS,
    MINI4_IDLE_TIME_US,
    MINI4_NUM_ADC_SAMPLES,
    MINI4_NUM_CHIRPS,
    MINI4_NUM_LOOPS,
    MINI4_NUM_RX_CHANNELS,
    MINI4_NUM_TX,
    MINI4_RAMP_END_TIME_US,
    MINI4_SAMPLE_RATE_KSPS,
    MINI4_SLOPE_MHZ_PER_US,
    MINI4_START_FREQUENCY_GHZ,
    ParticleFilter1D,
    PmmConfig,
    PmmTracker,
    capon_pmm_angle_scores,
    constrained_maximum_path,
    spectral_subtraction,
    spectrum_folding,
    validate_mini4_profile,
)

REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
PROFILE_PATH = REPOSITORY_ROOT / "profiles" / "profile-mini4-20m.cfg"


def _mini4_config() -> SimpleNamespace:
    return SimpleNamespace(
        num_adc_samples=MINI4_NUM_ADC_SAMPLES,
        num_rx_channels=MINI4_NUM_RX_CHANNELS,
        num_chirps_per_frame=MINI4_NUM_CHIRPS,
        num_loops=MINI4_NUM_LOOPS,
        num_chirps_per_loop=MINI4_NUM_TX,
        tx_channel_masks=(1, 4, 2),
        sample_rate_ksps=MINI4_SAMPLE_RATE_KSPS,
        frequency_slope_mhz_per_us=MINI4_SLOPE_MHZ_PER_US,
        start_frequency_ghz=MINI4_START_FREQUENCY_GHZ,
        idle_time_us=MINI4_IDLE_TIME_US,
        adc_start_time_us=MINI4_ADC_START_TIME_US,
        ramp_end_time_us=MINI4_RAMP_END_TIME_US,
        frame_periodicity_ms=MINI4_FRAME_PERIODICITY_MS,
    )


class SpectrumFoldingTests(unittest.TestCase):
    def test_recovers_five_bin_period(self) -> None:
        spectra = np.ones((3, 64), dtype=np.float32)
        spectra[:, ::5] = 100.0

        scores, sizes = spectrum_folding(spectra, 2, 20)

        np.testing.assert_array_equal(sizes, np.full(3, 5))
        self.assertTrue(np.all(scores > 90.0))

    def test_noise_only_score_remains_at_noise_level(self) -> None:
        spectra = np.ones((8, 64), dtype=np.float32)

        scores, sizes = spectrum_folding(spectra)

        np.testing.assert_allclose(scores, 1.0)
        np.testing.assert_array_equal(sizes, np.full(8, 2))

    def test_default_range_recovers_31_bin_period(self) -> None:
        spectrum = np.zeros((1, 64), dtype=np.float32)
        spectrum[0, (1, 32, 63)] = 100.0

        scores, sizes = spectrum_folding(spectrum)

        self.assertEqual(int(sizes[0]), 31)
        self.assertEqual(float(scores[0]), 100.0)

    def test_tracker_rejects_folding_sizes_with_fewer_than_two_rows(self) -> None:
        with self.assertRaisesRegex(ValueError, "two folding rows"):
            PmmConfig(folding_size_max=33)


class BackgroundSubtractionTests(unittest.TestCase):
    def test_removes_scaled_static_background_and_keeps_target(self) -> None:
        background = np.asarray((1.0, 2.0, 1.0), dtype=np.float32)
        measured = 3.0 * background
        measured[1] += 20.0

        corrected, gain = spectral_subtraction(measured, background)

        self.assertGreater(gain, 3.0)
        self.assertLess(corrected[0], 0.0)
        self.assertGreater(corrected[1], 5.0)


class DynamicProgrammingTests(unittest.TestCase):
    def test_backtracks_continuous_maximum_path(self) -> None:
        scores = np.zeros((5, 8), dtype=np.float32)
        expected = np.asarray((1, 2, 3, 4, 5))
        scores[np.arange(5), expected] = 10.0
        scores[np.arange(5), [7, 0, 7, 0, 7]] = 11.0

        path, total = constrained_maximum_path(scores, maximum_step_bins=1)

        np.testing.assert_array_equal(path, expected)
        self.assertEqual(total, 50.0)

    def test_uses_per_frame_transition_limits(self) -> None:
        scores = np.zeros((3, 6), dtype=np.float32)
        scores[0, 0] = 10.0
        scores[1, 2] = 10.0
        scores[2, 3] = 10.0

        path, total = constrained_maximum_path(scores, (2, 1))

        np.testing.assert_array_equal(path, np.asarray((0, 2, 3)))
        self.assertEqual(total, 30.0)


class ParticleFilterTests(unittest.TestCase):
    def test_converges_and_coasts_with_constant_velocity(self) -> None:
        particle_filter = ParticleFilter1D(
            2_000,
            (0.0, 20.0),
            (-4.0, 4.0),
            process_value_std=0.01,
            process_velocity_std=0.02,
            observation_std=0.05,
            seed=7,
        )

        for observation in np.linspace(5.0, 5.9, 10):
            particle_filter.predict(0.1)
            particle_filter.update(float(observation))

        position, velocity = particle_filter.estimate
        self.assertAlmostEqual(position, 5.9, delta=0.15)
        self.assertGreater(velocity, 0.2)


class Mini4ProfileTests(unittest.TestCase):
    def test_accepts_exact_profile_and_rejects_dimension_change(self) -> None:
        config = _mini4_config()
        validate_mini4_profile(config)

        config.num_adc_samples = 128
        with self.assertRaisesRegex(ValueError, "dimensions mismatch"):
            validate_mini4_profile(config)

    def test_repository_profile_has_required_frame_size_and_range_spacing(
        self,
    ) -> None:
        from main.livedatacapture import RadarCaptureConfig

        config = RadarCaptureConfig.from_file(PROFILE_PATH)
        validate_mini4_profile(config)
        range_axis = config.range_axis_m()
        assert range_axis is not None
        self.assertEqual(config.bytes_per_frame, 393_216)
        self.assertAlmostEqual(
            float(range_axis[1] - range_axis[0]),
            0.078,
            delta=0.001,
        )
        self.assertGreater(
            float(range_axis[-1]) + config.range_bias_m,
            19.8,
        )


class CaponAngleTests(unittest.TestCase):
    def test_static_angle_signal_is_retained_until_apmm_subtraction(self) -> None:
        config = _mini4_config()
        range_fft = np.zeros(
            (MINI4_NUM_CHIRPS, MINI4_NUM_RX_CHANNELS, MINI4_NUM_ADC_SAMPLES),
            dtype=np.complex64,
        )
        range_fft[..., 40] = 1.0

        _, scores = capon_pmm_angle_scores(
            range_fft,
            40,
            config,
            angle_limit_deg=60.0,
            angle_step_deg=2.0,
            folding_size_min=2,
            folding_size_max=32,
        )

        self.assertTrue(np.isfinite(scores).all())
        self.assertGreater(float(scores.max()), 0.0)

    def test_recovers_synthetic_ods_direction(self) -> None:
        config = _mini4_config()
        target_bin = 40
        azimuth_deg = 20.0
        elevation_deg = 10.0
        positions = {
            (1, 1): (0.0, 3.0), (1, 2): (0.0, 2.0),
            (1, 3): (1.0, 2.0), (1, 4): (1.0, 3.0),
            (2, 1): (2.0, 3.0), (2, 2): (2.0, 2.0),
            (2, 3): (3.0, 2.0), (2, 4): (3.0, 3.0),
            (3, 1): (2.0, 1.0), (3, 2): (2.0, 0.0),
            (3, 3): (3.0, 0.0), (3, 4): (3.0, 1.0),
        }
        rx_phase = np.asarray((1.0, -1.0, -1.0, 1.0))
        coordinates = []
        phase = []
        for mask in config.tx_channel_masks:
            tx_number = [bit + 1 for bit in range(3) if mask & (1 << bit)][0]
            coordinates.extend(
                positions[(tx_number, rx_number)]
                for rx_number in range(1, 5)
            )
            phase.extend(rx_phase)
        coordinates = np.asarray(coordinates)
        azimuth_rad = np.deg2rad(azimuth_deg)
        elevation_rad = np.deg2rad(elevation_deg)
        direction_x = np.sin(azimuth_rad) * np.cos(elevation_rad)
        direction_z = -np.sin(elevation_rad)
        steering = np.exp(
            1j
            * np.pi
            * (
                coordinates[:, 0] * direction_x
                + coordinates[:, 1] * direction_z
            )
        )
        slow_tone = np.exp(
            2j * np.pi * 5 * np.arange(MINI4_NUM_LOOPS) / MINI4_NUM_LOOPS
        )
        samples = slow_tone[:, None] * steering[None, :] * np.asarray(phase)[None, :]
        range_fft = np.zeros(
            (MINI4_NUM_CHIRPS, MINI4_NUM_RX_CHANNELS, MINI4_NUM_ADC_SAMPLES),
            dtype=np.complex64,
        )
        range_fft[..., target_bin] = samples.reshape(
            MINI4_NUM_LOOPS,
            MINI4_NUM_TX,
            MINI4_NUM_RX_CHANNELS,
        ).reshape(MINI4_NUM_CHIRPS, MINI4_NUM_RX_CHANNELS)

        angles, scores = capon_pmm_angle_scores(
            range_fft,
            target_bin,
            config,
            angle_limit_deg=60.0,
            angle_step_deg=2.0,
            folding_size_min=2,
            folding_size_max=20,
        )

        elevation_index, azimuth_index = np.unravel_index(
            np.argmax(scores),
            scores.shape,
        )
        self.assertAlmostEqual(float(angles[azimuth_index]), azimuth_deg, delta=4.0)
        self.assertAlmostEqual(
            float(angles[elevation_index]),
            elevation_deg,
            delta=4.0,
        )


class PmmTrackerStateTests(unittest.TestCase):
    def test_3d_display_suppresses_tentative_tracks(self) -> None:
        from main.livedatacapture import _target_track

        tracker = PmmTracker(
            _mini4_config(),
            PmmConfig(background_calibration_seconds=0.1),
        )
        tentative = replace(
            tracker.latest_result,
            state="tentative",
            range_m=5.0,
            azimuth_deg=0.0,
            elevation_deg=0.0,
        )

        self.assertIsNone(_target_track(tentative))
        self.assertIsNotNone(_target_track(replace(tentative, state="confirmed")))
        self.assertIsNotNone(_target_track(replace(tentative, state="coasting")))

    def test_adaptive_thresholds_are_per_range_and_frozen_after_calibration(
        self,
    ) -> None:
        tracker = PmmTracker(
            _mini4_config(),
            PmmConfig(
                background_calibration_seconds=0.3,
                adaptive_threshold_sigma=6.0,
                provisional_frames=1,
                confirmation_window_frames=1,
                confirmation_hits=1,
                particle_count=100,
            ),
        )
        range_axis = np.arange(MINI4_NUM_ADC_SAMPLES) * 0.07807095
        doppler_cube = np.ones((64, 3, 4, 256), dtype=np.complex64)
        range_fft = np.ones((96, 4, 256), dtype=np.complex64)
        sizes = np.full(256, 2, dtype=np.int16)

        score_frames = []
        for small_noise, large_noise in (
            (-10.0, -1_000.0),
            (0.0, 0.0),
            (10.0, 1_000.0),
        ):
            scores = np.full(256, 2_000.0, dtype=np.float32)
            scores[40] += small_noise
            scores[50] += large_noise
            score_frames.append((scores, sizes))
        target_scores = np.full(256, 2_000.0, dtype=np.float32)
        target_scores[50] = 10_000.0
        score_frames.append((target_scores, sizes))

        angles = np.arange(-60.0, 62.0, 2.0, dtype=np.float32)
        angle_scores = np.ones((61, 61), dtype=np.float32)
        with patch(
            "main.pmm.spectrum_folding",
            side_effect=score_frames,
        ), patch(
            "main.pmm.capon_pmm_angle_scores",
            return_value=(angles, angle_scores),
        ):
            for _ in range(3):
                tracker.update(doppler_cube, range_fft, range_axis)
            assert tracker.adaptive_thresholds is not None
            frozen = tracker.adaptive_thresholds.copy()
            assert tracker.range_indices is not None
            local_40 = int(np.flatnonzero(tracker.range_indices == 40)[0])
            local_50 = int(np.flatnonzero(tracker.range_indices == 50)[0])
            self.assertTrue(np.all(frozen >= 700.0))
            self.assertEqual(float(frozen[local_40]), 700.0)
            self.assertGreater(frozen[local_50], frozen[local_40])

            result = tracker.update(doppler_cube, range_fft, range_axis)

        np.testing.assert_array_equal(tracker.adaptive_thresholds, frozen)
        self.assertEqual(result.range_bin, 50)
        self.assertAlmostEqual(result.threshold, float(frozen[local_50]))

    def test_angle_background_is_subtracted_before_angle_paths(self) -> None:
        config = _mini4_config()
        tracker = PmmTracker(
            config,
            PmmConfig(
                background_calibration_seconds=0.1,
                detection_threshold=10.0,
                provisional_frames=1,
                confirmation_window_frames=1,
                confirmation_hits=1,
                particle_count=100,
            ),
        )
        range_axis = np.arange(MINI4_NUM_ADC_SAMPLES) * 0.07807095
        background = np.ones((64, 3, 4, 256), dtype=np.complex64)
        target = background.copy()
        target[::5, ..., 50] = 1_000.0
        range_fft = np.ones((96, 4, 256), dtype=np.complex64)
        angles = np.arange(-60.0, 62.0, 2.0, dtype=np.float32)
        background_angles = np.ones((61, 61), dtype=np.float32)
        measured_angles = 3.0 * background_angles
        measured_angles[40, 20] += 20.0

        with patch(
            "main.pmm.capon_pmm_angle_scores",
            side_effect=(
                (angles, background_angles),
                (angles, measured_angles),
            ),
        ):
            tracker.update(background, range_fft, range_axis)
            result = tracker.update(target, range_fft, range_axis)

        np.testing.assert_allclose(tracker.angle_az_background, 1.0)
        np.testing.assert_allclose(tracker.angle_el_background, 1.0)
        self.assertEqual(int(np.argmax(tracker.angle_az_history[-1])), 20)
        self.assertEqual(int(np.argmax(tracker.angle_el_history[-1])), 40)
        self.assertGreater(result.azimuth_background_projection_gain or 0.0, 3.0)
        self.assertGreater(result.elevation_background_projection_gain or 0.0, 3.0)

    def test_searching_retains_history_without_claiming_noise_ownership(
        self,
    ) -> None:
        config = _mini4_config()
        tracker = PmmTracker(
            config,
            PmmConfig(
                background_calibration_seconds=0.1,
                detection_threshold=1_000_000.0,
                provisional_frames=2,
                confirmation_window_frames=2,
                confirmation_hits=2,
                coast_frames=2,
                particle_count=100,
            ),
        )
        range_axis = np.arange(MINI4_NUM_ADC_SAMPLES) * 0.07807095
        background = np.ones((64, 3, 4, 256), dtype=np.complex64)
        range_fft = np.ones((96, 4, 256), dtype=np.complex64)
        tracker.update(background, range_fft, range_axis)

        result = None
        for _ in range(20):
            result = tracker.update(background, range_fft, range_axis)

        assert result is not None
        self.assertEqual(result.state, "searching")
        self.assertEqual(result.history_frames, 20)
        self.assertEqual(tracker.spectrogram_db.shape, (64, 19))
        np.testing.assert_array_equal(
            np.argmax(tracker.spectrogram_db, axis=0),
            np.zeros(19, dtype=np.int64),
        )
        self.assertFalse(tracker.range_filter.initialized)
        self.assertIsNone(result.radial_velocity_m_s)

    def test_calibrates_then_confirms_and_coasts(self) -> None:
        config = _mini4_config()
        tracker = PmmTracker(
            config,
            PmmConfig(
                background_calibration_seconds=0.1,
                detection_threshold=10.0,
                provisional_frames=2,
                confirmation_window_frames=2,
                confirmation_hits=2,
                coast_frames=2,
                particle_count=200,
            ),
        )
        spacing = (
            299_792_458.0
            * MINI4_SAMPLE_RATE_KSPS
            * 1e3
            / (
                2.0
                * MINI4_SLOPE_MHZ_PER_US
                * 1e12
                * MINI4_NUM_ADC_SAMPLES
            )
        )
        range_axis = np.arange(MINI4_NUM_ADC_SAMPLES) * spacing
        background = np.ones(
            (
                MINI4_DOPPLER_FFT_SIZE,
                MINI4_NUM_TX,
                MINI4_NUM_RX_CHANNELS,
                MINI4_NUM_ADC_SAMPLES,
            ),
            dtype=np.complex64,
        )
        range_fft = np.ones(
            (MINI4_NUM_CHIRPS, MINI4_NUM_RX_CHANNELS, MINI4_NUM_ADC_SAMPLES),
            dtype=np.complex64,
        )
        self.assertEqual(
            tracker.update(background, range_fft, range_axis).state,
            "searching",
        )

        target = background.copy()
        target[::5, ..., 50] = 1_000.0
        angles = np.arange(-60.0, 62.0, 2.0, dtype=np.float32)
        angle_scores = np.zeros((angles.size, angles.size), dtype=np.float32)
        angle_scores[30, 30] = 100.0
        with patch(
            "main.pmm.capon_pmm_angle_scores",
            return_value=(angles, angle_scores),
        ):
            first = tracker.update(target, range_fft, range_axis)
            second = tracker.update(target, range_fft, range_axis)
            third = tracker.update(target, range_fft, range_axis)

        self.assertEqual(first.state, "searching")
        self.assertEqual(second.state, "tentative")
        self.assertEqual(third.state, "confirmed")
        self.assertEqual(third.label, "PMM target")
        self.assertIsNotNone(third.range_m)

        miss = background.copy()
        with patch(
            "main.pmm.capon_pmm_angle_scores",
            return_value=(angles, angle_scores),
        ):
            coast_one = tracker.update(miss, range_fft, range_axis)
            coast_two = tracker.update(miss, range_fft, range_axis)
            lost = tracker.update(miss, range_fft, range_axis)
            reacquire_one = tracker.update(target, range_fft, range_axis)
            reacquire_two = tracker.update(target, range_fft, range_axis)
            reacquired = tracker.update(target, range_fft, range_axis)

        self.assertEqual(coast_one.state, "coasting")
        self.assertEqual(coast_two.state, "coasting")
        self.assertEqual(lost.state, "lost")
        self.assertEqual(reacquire_one.state, "searching")
        self.assertEqual(reacquire_two.state, "tentative")
        self.assertEqual(reacquired.state, "confirmed")

    def test_timestamp_discontinuity_resets_track_ownership(self) -> None:
        config = _mini4_config()
        tracker = PmmTracker(
            config,
            PmmConfig(
                background_calibration_seconds=0.1,
                detection_threshold=10.0,
                provisional_frames=2,
                confirmation_window_frames=2,
                confirmation_hits=2,
                particle_count=100,
            ),
        )
        range_axis = np.arange(MINI4_NUM_ADC_SAMPLES) * 0.07807095
        background = np.ones(
            (64, 3, 4, 256),
            dtype=np.complex64,
        )
        range_fft = np.ones((96, 4, 256), dtype=np.complex64)
        tracker.update(
            background,
            range_fft,
            range_axis,
            timestamp_s=1.0,
        )
        target = background.copy()
        target[::5, ..., 50] = 1_000.0
        angles = np.arange(-60.0, 62.0, 2.0, dtype=np.float32)
        angle_scores = np.zeros((61, 61), dtype=np.float32)
        with patch(
            "main.pmm.capon_pmm_angle_scores",
            return_value=(angles, angle_scores),
        ):
            tracker.update(target, range_fft, range_axis, timestamp_s=1.1)
            tracker.update(target, range_fft, range_axis, timestamp_s=1.2)
            tracker.update(target, range_fft, range_axis, timestamp_s=1.3)
            reset = tracker.update(
                target,
                range_fft,
                range_axis,
                timestamp_s=2.0,
            )

        self.assertEqual(reset.state, "searching")
        self.assertEqual(reset.history_frames, 1)

    def test_fft_dimension_mismatch_resets_background_calibration(self) -> None:
        config = _mini4_config()
        tracker = PmmTracker(
            config,
            PmmConfig(background_calibration_seconds=0.1),
        )
        range_axis = np.arange(MINI4_NUM_ADC_SAMPLES) * 0.07807095
        range_fft = np.ones((96, 4, 256), dtype=np.complex64)
        valid = np.ones((64, 3, 4, 256), dtype=np.complex64)
        tracker.update(valid, range_fft, range_axis)
        self.assertIsNotNone(tracker.background)
        self.assertIsNotNone(tracker.angle_az_background)
        self.assertIsNotNone(tracker.angle_el_background)

        with self.assertRaisesRegex(ValueError, "Doppler cube mismatch"):
            tracker.update(valid[:-1], range_fft, range_axis)

        self.assertIsNone(tracker.background)
        self.assertIsNone(tracker.angle_az_background)
        self.assertIsNone(tracker.angle_el_background)
        self.assertEqual(tracker.state, "calibrating")


class PmmJsonlTests(unittest.TestCase):
    def test_schema_contains_only_pmm_tracking_fields(self) -> None:
        from main.livedatacapture import (
            ProcessedOutputWriter,
            RadarCaptureConfig,
        )

        config = RadarCaptureConfig.from_file(PROFILE_PATH)
        tracker = PmmTracker(
            config,
            PmmConfig(background_calibration_seconds=0.1),
        )
        result = tracker.latest_result
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "pmm.jsonl"
            writer = ProcessedOutputWriter(
                path,
                config,
                pmm_metadata=tracker.metadata,
            )
            writer.write_update(
                frame_index=1,
                pmm_result=result,
                doppler_time_db=np.empty((64, 0), dtype=np.float32),
                diagnostics={},
            )
            writer.close()
            metadata, update = (
                json.loads(line)
                for line in path.read_text(encoding="utf-8").splitlines()
            )

        self.assertEqual(metadata["format"], "mini4-pmm-jsonl")
        self.assertIn("profile_fingerprint", metadata["pmm_tracking"])
        self.assertIn("feature_fingerprint", metadata["pmm_tracking"])
        self.assertEqual(update["pmm_tracking"]["label"], "PMM target")

    def test_schema_includes_optional_classification_metadata_and_result(self) -> None:
        from main.livedatacapture import (
            ProcessedOutputWriter,
            RadarCaptureConfig,
        )

        config = RadarCaptureConfig.from_file(PROFILE_PATH)
        tracker = PmmTracker(
            config,
            PmmConfig(background_calibration_seconds=0.1),
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "classified.jsonl"
            writer = ProcessedOutputWriter(
                path,
                config,
                pmm_metadata=tracker.metadata,
                classification_metadata={"enabled": True},
            )
            writer.write_update(
                frame_index=1,
                pmm_result=tracker.latest_result,
                doppler_time_db=np.empty((64, 0), dtype=np.float32),
                diagnostics={},
                classification={
                    "status": "warming_up",
                    "label": "unknown",
                },
            )
            writer.close()
            metadata, update = (
                json.loads(line)
                for line in path.read_text(encoding="utf-8").splitlines()
            )

        self.assertTrue(metadata["classification"]["enabled"])
        self.assertEqual(update["classification"]["status"], "warming_up")


class RawReplayTests(unittest.TestCase):
    def test_complete_raw_frames_replay_in_order(self) -> None:
        from main.livedatacapture import RadarCaptureConfig
        from main.replay_pmm import replay_raw_frames

        config = RadarCaptureConfig.from_file(PROFILE_PATH)
        with tempfile.TemporaryDirectory() as directory:
            raw_path = Path(directory) / "clutter-only.bin"
            raw_path.write_bytes(bytes(config.bytes_per_frame * 2))
            records = list(
                replay_raw_frames(
                    raw_path,
                    config,
                    PmmConfig(background_calibration_seconds=0.1),
                )
            )

        self.assertEqual([record["frame_index"] for record in records], [0, 1])
        self.assertTrue(
            all(
                record["pmm_tracking"]["state"] == "searching"
                for record in records
            )
        )
        self.assertTrue(
            all(
                record["pmm_tracking"]["label"] == "PMM target"
                for record in records
            )
        )

    def test_incomplete_raw_frame_is_rejected(self) -> None:
        from main.livedatacapture import RadarCaptureConfig
        from main.replay_pmm import replay_raw_frames

        config = RadarCaptureConfig.from_file(PROFILE_PATH)
        with tempfile.TemporaryDirectory() as directory:
            raw_path = Path(directory) / "packet-gap.bin"
            raw_path.write_bytes(bytes(config.bytes_per_frame - 1))
            with self.assertRaisesRegex(ValueError, "incomplete frame"):
                list(
                    replay_raw_frames(
                        raw_path,
                        config,
                        PmmConfig(background_calibration_seconds=0.1),
                    )
                )


class PmmWorkerIntegrationTests(unittest.TestCase):
    def test_worker_processes_mini4_frame(self) -> None:
        from main.livedatacapture import (
            CapturedFrame,
            CombinedDisplayPayload,
            RadarCaptureConfig,
            _run_frame_processor,
        )

        config = RadarCaptureConfig.from_file(PROFILE_PATH)
        frame_queue = queue.Queue()
        frame_queue.put(
            CapturedFrame(
                data=bytes(config.bytes_per_frame),
                gap_bytes=0,
                first_byte_at_s=1.0,
            )
        )
        frame_queue.put(None)
        log_queue = queue.Queue()
        payload_queue = queue.Queue(maxsize=1)
        status_queue = queue.Queue(maxsize=1)
        counter = type(
            "Counter",
            (),
            {"value": 0, "get_lock": lambda self: threading.Lock()},
        )()

        with patch("main.livedatacapture.signal.signal"):
            _run_frame_processor(
                config=config,
                pmm_config=PmmConfig(
                    background_calibration_seconds=0.1
                ),
                frame_queue=frame_queue,
                log_queue=log_queue,
                processed_frames_counter=counter,
                display_payload_queue=payload_queue,
                display_skipped_counter=None,
                raw_output=None,
                raw_metadata=None,
                processed_output=None,
                display_mode="combined",
                display_update_every=1,
                startup_status_queue=status_queue,
            )

        self.assertEqual(counter.value, 1)
        self.assertEqual(status_queue.get_nowait()["state"], "ready")
        payload = payload_queue.get_nowait()
        self.assertIsInstance(payload, CombinedDisplayPayload)
        self.assertIn("PMM target", payload.point_cloud.tracking_status)
        status_rows = payload.point_cloud.tracking_status.splitlines()
        self.assertEqual(len(status_rows), 2)
        self.assertIn("calibration", status_rows[0])
        self.assertTrue(status_rows[1].startswith("score "))


if __name__ == "__main__":
    unittest.main()
