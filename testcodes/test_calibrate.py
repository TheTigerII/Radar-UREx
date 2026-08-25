import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import numpy as np

from main import calibrate
from main.livedatacapture import RadarCaptureConfig, _with_host_compensation
from main.pmm import PmmConfig


ROOT = Path(__file__).resolve().parent.parent
CALIBRATION_PROFILE = ROOT / "profiles" / "profile_calibration.cfg"
OPERATIONAL_PROFILE = ROOT / "profiles" / "profile-mini4-20m.cfg"


class CalibrationProfileTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = RadarCaptureConfig.from_file(CALIBRATION_PROFILE)
        self.settings = calibrate.CalibrationSettings()

    def test_source_profile_dimensions_and_timing(self) -> None:
        self.assertEqual(self.config.num_adc_samples, 256)
        self.assertEqual(self.config.num_rx_channels, 4)
        self.assertEqual(self.config.num_chirps_per_loop, 3)
        self.assertEqual(self.config.num_loops, 32)
        self.assertEqual(self.config.num_chirps_per_frame, 96)
        self.assertEqual(self.config.bytes_per_frame, 393_216)
        self.assertEqual(self.config.tx_channel_masks, (1, 4, 2))
        self.assertAlmostEqual(self.config.range_resolution_m, 0.0440913, places=6)
        self.assertAlmostEqual(self.config.frame_periodicity_ms, 500.0)

    def test_runtime_profile_changes_only_required_commands(self) -> None:
        source_lines = CALIBRATION_PROFILE.read_text(encoding="utf-8").splitlines()
        with tempfile.TemporaryDirectory() as directory:
            runtime = Path(directory) / "runtime.cfg"
            calibrate.create_runtime_profile(
                CALIBRATION_PROFILE, runtime, self.config, self.settings
            )
            runtime_lines = runtime.read_text(encoding="utf-8").splitlines()
            changed = [
                (before, after)
                for before, after in zip(source_lines, runtime_lines)
                if before != after
            ]
            self.assertEqual(
                {after for _, after in changed},
                {
                    "guiMonitor -1 0 0 0 0 0 0",
                    "measureRangeBiasAndRxChanPhase 0 1 0.2",
                    "lvdsStreamCfg -1 0 1 0",
                },
            )
            self.assertEqual(len(changed), 3)
            runtime_config = RadarCaptureConfig.from_file(runtime)
            calibrate.validate_calibration_profile_text(
                runtime.read_text(encoding="utf-8"),
                runtime_config,
                self.settings,
                require_raw_lvds=True,
            )

    def test_angular_runtime_is_firmware_neutral_with_host_compensation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            runtime = Path(directory) / "runtime.cfg"
            settings = calibrate.CalibrationSettings(calibration_type="azimuth")
            calibrate.create_runtime_profile(
                CALIBRATION_PROFILE, runtime, self.config, settings
            )
            source_command = next(
                line
                for line in CALIBRATION_PROFILE.read_text(encoding="utf-8").splitlines()
                if line.startswith("compRangeBiasAndRxChanPhase")
            )
            runtime_command = next(
                line
                for line in runtime.read_text(encoding="utf-8").splitlines()
                if line.startswith("compRangeBiasAndRxChanPhase")
            )
            self.assertEqual(runtime_command, source_command)
            operational = RadarCaptureConfig.from_file(OPERATIONAL_PROFILE)
            host_config = _with_host_compensation(
                RadarCaptureConfig.from_file(runtime), operational
            )
            self.assertEqual(
                host_config.rx_channel_compensation,
                operational.rx_channel_compensation,
            )

    def test_rejects_duplicate_command_and_disabled_runtime_lvds(self) -> None:
        text = CALIBRATION_PROFILE.read_text(encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "exactly one guiMonitor"):
            calibrate.validate_calibration_profile_text(
                text + "\nguiMonitor -1 0 0 0 0 0 0\n",
                self.config,
                self.settings,
            )
        with self.assertRaisesRegex(ValueError, "raw LVDS streaming disabled"):
            calibrate.validate_calibration_profile_text(
                text,
                self.config,
                self.settings,
                require_raw_lvds=True,
            )


class CalibrationAlgorithmTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = RadarCaptureConfig.from_file(CALIBRATION_PROFILE)

    def _synthetic_range_fft(
        self, range_bias_m: float
    ) -> tuple[np.ndarray, np.ndarray]:
        target_bin = (1.0 + range_bias_m) / self.config.range_resolution_m
        bins = np.arange(self.config.num_adc_samples)
        shape = np.exp(-0.5 * ((bins - target_bin) / 0.55) ** 2)
        gains = np.asarray(
            [
                (1.0 + 0.07 * index)
                * np.exp(1j * (-0.8 + 0.13 * index))
                for index in range(12)
            ]
        )
        cube = np.zeros(
            (self.config.num_chirps_per_frame, 4, self.config.num_adc_samples),
            dtype=np.complex128,
        )
        for loop in range(self.config.num_loops):
            for slot, physical_tx in enumerate((0, 2, 1)):
                chirp = loop * 3 + slot
                for rx in range(4):
                    cube[chirp, rx] = gains[physical_tx * 4 + rx] * shape
        return cube, gains

    def test_recovers_range_bias_equalization_and_physical_tx_order(self) -> None:
        settings = calibrate.CalibrationSettings(
            target_distance_m=1.0,
            search_window_m=0.2,
            warmup_frames=0,
            accepted_frames=4,
            min_peak_prominence_db=3.0,
            max_range_std_m=1e-6,
            max_phase_std_deg=1e-6,
            max_magnitude_cv=1e-6,
        )
        accumulator = calibrate.CalibrationAccumulator(self.config, settings)
        cube, gains = self._synthetic_range_fft(0.06)
        for _ in range(settings.accepted_frames):
            accumulator.update(cube)
        result = accumulator.result
        self.assertIsNotNone(result)
        assert result is not None
        self.assertAlmostEqual(result.range_bias_m, 0.06, delta=0.01)
        self.assertEqual(result.tx_order, (1, 4, 2))
        equalized = np.asarray(result.coefficients) * gains
        self.assertTrue(np.allclose(equalized, equalized[0], rtol=1e-6, atol=1e-6))

    def _synthetic_angular_range_fft(
        self,
        azimuth_deg: float,
        elevation_deg: float,
        raw_target_range_m: float = 1.05,
    ) -> np.ndarray:
        positions = {
            (1, 1): (3, 0), (1, 2): (2, 0), (1, 3): (2, 1), (1, 4): (3, 1),
            (2, 1): (3, 2), (2, 2): (2, 2), (2, 3): (2, 3), (2, 4): (3, 3),
            (3, 1): (1, 2), (3, 2): (0, 2), (3, 3): (0, 3), (3, 4): (1, 3),
        }
        target_bin = raw_target_range_m / self.config.range_resolution_m
        bins = np.arange(self.config.num_adc_samples)
        shape = np.exp(-0.5 * ((bins - target_bin) / 0.55) ** 2)
        azimuth_u = np.sin(np.deg2rad(azimuth_deg))
        elevation_u = -np.sin(np.deg2rad(elevation_deg))
        cube = np.zeros(
            (self.config.num_chirps_per_frame, 4, self.config.num_adc_samples),
            dtype=np.complex128,
        )
        for loop in range(self.config.num_loops):
            for slot, tx_number in enumerate((1, 3, 2)):
                for rx_number in range(1, 5):
                    row, column = positions[(tx_number, rx_number)]
                    signal = np.exp(
                        1j * np.pi * (elevation_u * row + azimuth_u * column)
                    )
                    polarity = (1.0, -1.0, -1.0, 1.0)[rx_number - 1]
                    cube[loop * 3 + slot, rx_number - 1] = signal / polarity * shape
        return cube

    def test_recovers_azimuth_and_elevation_offsets(self) -> None:
        cases = (
            ("azimuth", 10.0, 12.0, 0.0, 2.0),
            ("elevation", -10.0, 0.0, -14.0, -4.0),
        )
        for calibration_type, reference, azimuth, elevation, expected_bias in cases:
            with self.subTest(calibration_type=calibration_type):
                settings = calibrate.CalibrationSettings(
                    calibration_type=calibration_type,
                    reference_angle_deg=reference,
                    warmup_frames=0,
                    accepted_frames=4,
                    min_peak_prominence_db=3.0,
                    max_angle_std_deg=0.1,
                )
                accumulator = calibrate.AngularCalibrationAccumulator(
                    self.config, settings
                )
                cube = self._synthetic_angular_range_fft(azimuth, elevation)
                for _ in range(settings.accepted_frames):
                    accumulator.update(cube)
                self.assertIsNotNone(accumulator.result)
                assert accumulator.result is not None
                self.assertAlmostEqual(
                    accumulator.result.angle_bias_deg, expected_bias, delta=0.1
                )

    def test_angular_calibration_uses_bias_corrected_range_axis(self) -> None:
        config = replace(self.config, range_bias_m=0.10)
        settings = calibrate.CalibrationSettings(
            calibration_type="azimuth",
            reference_angle_deg=12.0,
            target_distance_m=1.0,
            search_window_m=0.08,
            warmup_frames=0,
            accepted_frames=2,
            min_peak_prominence_db=3.0,
            max_angle_std_deg=0.1,
        )
        accumulator = calibrate.AngularCalibrationAccumulator(config, settings)
        cube = self._synthetic_angular_range_fft(
            12.0,
            0.0,
            raw_target_range_m=1.10,
        )

        for _ in range(settings.accepted_frames):
            accumulator.update(cube)

        self.assertIsNotNone(accumulator.result)
        assert accumulator.result is not None
        self.assertAlmostEqual(accumulator.result.measured_range_m, 1.0, delta=0.02)

    def test_angular_calibration_forwards_runtime_capon_settings(self) -> None:
        pmm_config = PmmConfig(
            angle_limit_deg=40.0,
            angle_step_deg=4.0,
            folding_size_min=3,
            folding_size_max=12,
        )
        settings = calibrate.CalibrationSettings(
            calibration_type="azimuth",
            reference_angle_deg=12.0,
            warmup_frames=0,
            accepted_frames=1,
            min_peak_prominence_db=3.0,
        )
        accumulator = calibrate.AngularCalibrationAccumulator(
            self.config,
            settings,
            pmm_config,
        )
        cube = self._synthetic_angular_range_fft(12.0, 0.0)
        runtime_estimator = calibrate.capon_pmm_angle_scores

        with patch(
            "main.calibrate.capon_pmm_angle_scores",
            wraps=runtime_estimator,
        ) as estimator:
            accumulator.update(cube)

        estimator.assert_called_once()
        self.assertEqual(estimator.call_args.kwargs["angle_limit_deg"], 40.0)
        self.assertEqual(estimator.call_args.kwargs["angle_step_deg"], 4.0)
        self.assertEqual(estimator.call_args.kwargs["folding_size_min"], 3)
        self.assertEqual(estimator.call_args.kwargs["folding_size_max"], 12)


class CalibrationResultSerializationTests(unittest.TestCase):
    def test_command_uses_ti_fixed_decimal_format_without_scientific_notation(
        self,
    ) -> None:
        coefficient_values = (
            0.8824776,
            -4.209382e-18,
            -0.6997676,
            0.312786,
            -0.8351456,
            0.1228002,
            0.8103891,
            0.04118465,
            0.6537936,
            -0.5620742,
            -0.3811256,
            0.8054636,
            -0.5437871,
            0.7257938,
            0.7176921,
            -0.6019038,
            0.3390693,
            -0.9407614,
            0.008614688,
            0.8587477,
            -0.1351867,
            0.9454902,
            0.4026296,
            -0.7913112,
        )
        result = calibrate.CalibrationResult(
            target_distance_m=1.0,
            search_window_m=0.2,
            measured_range_m=1.08009389,
            range_bias_m=0.08009389,
            coefficients=tuple(
                complex(coefficient_values[index], coefficient_values[index + 1])
                for index in range(0, len(coefficient_values), 2)
            ),
            accepted_frames=64,
            range_std_m=0.001,
            max_phase_std_deg=1.0,
            max_magnitude_cv=0.01,
            tx_order=(1, 4, 2),
        )

        self.assertEqual(
            result.command,
            "compRangeBiasAndRxChanPhase 0.0800939 "
            "0.88248 -0.00000 -0.69977 0.31279 -0.83515 0.12280 "
            "0.81039 0.04118 0.65379 -0.56207 -0.38113 0.80546 "
            "-0.54379 0.72579 0.71769 -0.60190 0.33907 -0.94076 "
            "0.00861 0.85875 -0.13519 0.94549 0.40263 -0.79131",
        )
        self.assertTrue(
            all("e" not in value.lower() for value in result.command.split()[1:])
        )


class CalibrationApplyTests(unittest.TestCase):
    def test_range_and_angular_results_are_parsed_by_normal_processing(self) -> None:
        range_result = calibrate.CalibrationResult(
            target_distance_m=1.0,
            search_window_m=0.2,
            measured_range_m=1.04,
            range_bias_m=0.04,
            coefficients=tuple(1 + 0.01j * index for index in range(12)),
            accepted_frames=64,
            range_std_m=0.001,
            max_phase_std_deg=1.0,
            max_magnitude_cv=0.01,
            tx_order=(1, 4, 2),
        )
        angle_result = calibrate.AngularCalibrationResult(
            calibration_type="azimuth",
            target_distance_m=1.0,
            search_window_m=0.2,
            reference_angle_deg=10.0,
            measured_angle_deg=12.0,
            angle_bias_deg=2.0,
            accepted_frames=64,
            angle_std_deg=0.2,
            measured_range_m=1.0,
            tx_order=(1, 4, 2),
        )
        with tempfile.TemporaryDirectory() as directory:
            profile = Path(directory) / "profile.cfg"
            profile.write_bytes(OPERATIONAL_PROFILE.read_bytes())
            original = RadarCaptureConfig.from_file(profile)
            first_backup = calibrate.apply_calibration_to_profile(
                profile, range_result
            )
            second_backup = calibrate.apply_angular_calibration_to_profile(
                profile, angle_result
            )
            self.assertTrue(first_backup.exists())
            self.assertTrue(second_backup.exists())
            parsed = RadarCaptureConfig.from_file(profile)
            self.assertAlmostEqual(parsed.range_axis_m()[0], -0.04, places=6)
            self.assertTrue(
                np.allclose(parsed.rx_channel_compensation, range_result.coefficients)
            )
            self.assertAlmostEqual(parsed.azimuth_bias_deg, 2.0)
            self.assertAlmostEqual(
                parsed.elevation_bias_deg,
                original.elevation_bias_deg,
            )


if __name__ == "__main__":
    unittest.main()
