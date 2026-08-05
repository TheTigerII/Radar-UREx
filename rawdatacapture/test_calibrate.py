import tempfile
import unittest
from pathlib import Path

import numpy as np

from rawdatacapture import calibrate
from rawdatacapture.dsp import (
    _apply_host_angle_calibration,
    build_virtual_antenna_grids,
)
from rawdatacapture.livedatacapture import RadarCaptureConfig


ROOT = Path(__file__).resolve().parent.parent
CALIBRATION_PROFILE = ROOT / "profiles" / "profile_calibration.cfg"


class CalibrationProfileTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = RadarCaptureConfig.from_file(CALIBRATION_PROFILE)
        self.settings = calibrate.CalibrationSettings()

    def test_discovered_profile_dimensions_and_timing(self) -> None:
        self.assertEqual(self.config.num_adc_samples, 256)
        self.assertEqual(self.config.num_rx_channels, 4)
        self.assertEqual(self.config.num_chirps_per_loop, 3)
        self.assertEqual(self.config.num_loops, 32)
        self.assertEqual(self.config.num_chirps_per_frame, 96)
        self.assertEqual(self.config.bytes_per_frame, 393_216)
        self.assertEqual(self.config.tx_channel_masks, (1, 4, 2))
        self.assertAlmostEqual(self.config.range_resolution_m, 0.0440913, places=6)
        self.assertAlmostEqual(self.config.frame_periodicity_ms, 500.0)
        self.assertAlmostEqual(self.config.frame_duty_cycle, 0.005952)

    def test_runtime_profile_changes_only_required_commands(self) -> None:
        source_lines = CALIBRATION_PROFILE.read_text(encoding="utf-8").splitlines()
        with tempfile.TemporaryDirectory() as directory:
            runtime = Path(directory) / "runtime.cfg"
            calibrate.create_runtime_profile(
                CALIBRATION_PROFILE, runtime, self.config, self.settings
            )
            runtime_lines = runtime.read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(source_lines), len(runtime_lines))
            changed = [
                (before, after)
                for before, after in zip(source_lines, runtime_lines)
                if before != after
            ]
            self.assertEqual(len(changed), 3)
            self.assertEqual(
                {after for _, after in changed},
                {
                    "guiMonitor -1 0 0 0 0 0 0",
                    "measureRangeBiasAndRxChanPhase 0 1 0.2",
                    "lvdsStreamCfg -1 0 1 0",
                },
            )
            runtime_config = RadarCaptureConfig.from_file(runtime)
            calibrate.validate_calibration_profile_text(
                runtime.read_text(encoding="utf-8"),
                runtime_config,
                self.settings,
                require_raw_lvds=True,
            )

    def test_angular_runtime_imports_operational_channel_compensation(self) -> None:
        operational = ROOT / "rawdatacapture" / "profile.cfg"
        operational_text = operational.read_text(encoding="utf-8")
        operational_command = next(
            line.strip()
            for line in operational_text.splitlines()
            if line.strip().startswith("compRangeBiasAndRxChanPhase")
        )
        with tempfile.TemporaryDirectory() as directory:
            runtime = Path(directory) / "runtime.cfg"
            calibrate.create_runtime_profile(
                CALIBRATION_PROFILE,
                runtime,
                self.config,
                calibrate.CalibrationSettings(calibration_type="azimuth"),
                operational,
            )
            runtime_command = next(
                line.strip()
                for line in runtime.read_text(encoding="utf-8").splitlines()
                if line.strip().startswith("compRangeBiasAndRxChanPhase")
            )
            self.assertEqual(runtime_command, operational_command)

    def test_duplicate_required_command_is_rejected(self) -> None:
        text = CALIBRATION_PROFILE.read_text(encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "exactly one guiMonitor"):
            calibrate.validate_calibration_profile_text(
                text + "\nguiMonitor -1 0 0 0 0 0 0\n",
                self.config,
                self.settings,
            )

    def test_disabled_runtime_lvds_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "raw LVDS streaming disabled"):
            calibrate.validate_calibration_profile_text(
                CALIBRATION_PROFILE.read_text(encoding="utf-8"),
                self.config,
                self.settings,
                require_raw_lvds=True,
            )

    def test_missing_antennas_and_invalid_target_window_are_rejected(self) -> None:
        invalid = RadarCaptureConfig.from_dimensions(
            num_adc_samples=256,
            num_rx_channels=3,
            num_chirps_per_frame=96,
            num_loops=32,
            num_chirps_per_loop=3,
            tx_channel_masks=(1, 4, 2),
            sample_rate_ksps=12500,
            frequency_slope_mhz_per_us=166,
        )
        with self.assertRaisesRegex(ValueError, "four RX"):
            calibrate.validate_calibration_profile_text(
                CALIBRATION_PROFILE.read_text(encoding="utf-8"),
                invalid,
                self.settings,
            )
        with self.assertRaisesRegex(ValueError, "outside"):
            calibrate.validate_calibration_profile_text(
                CALIBRATION_PROFILE.read_text(encoding="utf-8"),
                self.config,
                calibrate.CalibrationSettings(
                    target_distance_m=0.01, search_window_m=0.2
                ),
            )


class CalibrationAlgorithmTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = RadarCaptureConfig.from_file(CALIBRATION_PROFILE)

    def _synthetic_range_fft(self, range_bias_m: float) -> tuple[np.ndarray, np.ndarray]:
        axis = self.config.range_axis_m()
        target_bin = (1.0 + range_bias_m) / self.config.range_resolution_m
        bins = np.arange(self.config.num_adc_samples)
        shape = np.exp(-0.5 * ((bins - target_bin) / 0.55) ** 2)
        gains = np.asarray(
            [
                (1.0 + 0.07 * index)
                * np.exp(1j * (-0.8 + 0.13 * index))
                for index in range(12)
            ],
            dtype=np.complex128,
        )
        cube = np.zeros(
            (
                self.config.num_chirps_per_frame,
                self.config.num_rx_channels,
                self.config.num_adc_samples,
            ),
            dtype=np.complex128,
        )
        # Profile chirps are TX1, TX3, TX2; gains are physical TX-major/RX-minor.
        slot_to_physical_tx = (0, 2, 1)
        for loop in range(self.config.num_loops):
            for slot, physical_tx in enumerate(slot_to_physical_tx):
                chirp = loop * 3 + slot
                for rx in range(4):
                    cube[chirp, rx] = gains[physical_tx * 4 + rx] * shape
        return cube, gains

    def test_recovers_bias_equalization_and_physical_tx_order(self) -> None:
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

    def test_normal_dsp_uses_measured_coefficients_by_physical_tx(self) -> None:
        coefficients = tuple(complex(index + 1, 0) for index in range(12))
        config = RadarCaptureConfig.from_dimensions(
            num_adc_samples=4,
            num_rx_channels=4,
            num_chirps_per_frame=3,
            num_loops=1,
            num_chirps_per_loop=3,
            tx_channel_masks=(1, 4, 2),
            rx_channel_compensation=coefficients,
        )
        samples = np.ones((1, 3, 4), dtype=np.complex64)
        grid = build_virtual_antenna_grids(samples, config)[0]
        # Physical TX2 is chirp slot 3 and uses coefficient indices 4..7.
        self.assertEqual(grid[3, 2], coefficients[4])
        self.assertEqual(grid[2, 2], coefficients[5])
        self.assertEqual(grid[2, 3], coefficients[6])
        self.assertEqual(grid[3, 3], coefficients[7])

    def _synthetic_angular_range_fft(
        self,
        azimuth_deg: float,
        elevation_deg: float,
    ) -> np.ndarray:
        positions = {
            (1, 1): (3, 0), (1, 2): (2, 0), (1, 3): (2, 1), (1, 4): (3, 1),
            (2, 1): (3, 2), (2, 2): (2, 2), (2, 3): (2, 3), (2, 4): (3, 3),
            (3, 1): (1, 2), (3, 2): (0, 2), (3, 3): (0, 3), (3, 4): (1, 3),
        }
        rx_polarity = {1: 1.0, 2: -1.0, 3: -1.0, 4: 1.0}
        bins = np.arange(self.config.num_adc_samples)
        target_bin = 1.05 / self.config.range_resolution_m
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
                    grid_sample = np.exp(
                        1j * np.pi * (elevation_u * row + azimuth_u * column)
                    )
                    cube[loop * 3 + slot, rx_number - 1] = (
                        grid_sample / rx_polarity[rx_number] * shape
                    )
        return cube

    def test_azimuth_calibration_recovers_known_offset(self) -> None:
        settings = calibrate.CalibrationSettings(
            calibration_type="azimuth",
            reference_angle_deg=10.0,
            warmup_frames=0,
            accepted_frames=4,
            min_peak_prominence_db=3.0,
            max_angle_std_deg=0.1,
        )
        accumulator = calibrate.AngularCalibrationAccumulator(self.config, settings)
        cube = self._synthetic_angular_range_fft(12.0, 0.0)
        for _ in range(4):
            accumulator.update(cube)
        self.assertIsNotNone(accumulator.result)
        assert accumulator.result is not None
        self.assertAlmostEqual(accumulator.result.measured_angle_deg, 12.0, delta=0.1)
        self.assertAlmostEqual(accumulator.result.angle_bias_deg, 2.0, delta=0.1)

    def test_elevation_calibration_uses_positive_up_angle_convention(self) -> None:
        settings = calibrate.CalibrationSettings(
            calibration_type="elevation",
            reference_angle_deg=-10.0,
            warmup_frames=0,
            accepted_frames=4,
            min_peak_prominence_db=3.0,
            max_angle_std_deg=0.1,
        )
        accumulator = calibrate.AngularCalibrationAccumulator(self.config, settings)
        cube = self._synthetic_angular_range_fft(0.0, -15.0)
        for _ in range(4):
            accumulator.update(cube)
        self.assertIsNotNone(accumulator.result)
        assert accumulator.result is not None
        self.assertAlmostEqual(accumulator.result.measured_angle_deg, -15.0, delta=0.1)
        self.assertAlmostEqual(accumulator.result.angle_bias_deg, -5.0, delta=0.1)


class CalibrationApplyTests(unittest.TestCase):
    def test_apply_creates_backup_and_replaces_one_line(self) -> None:
        result = calibrate.CalibrationResult(
            target_distance_m=1.0,
            search_window_m=0.2,
            measured_range_m=1.05,
            range_bias_m=0.05,
            coefficients=tuple(1 + 0j for _ in range(12)),
            accepted_frames=64,
            range_std_m=0.001,
            max_phase_std_deg=1.0,
            max_magnitude_cv=0.01,
            tx_order=(1, 4, 2),
        )
        with tempfile.TemporaryDirectory() as directory:
            profile = Path(directory) / "profile.cfg"
            original = "sensorStop\ncompRangeBiasAndRxChanPhase 0 " + "1 0 " * 12 + "\nsensorStart\n"
            profile.write_text(original, encoding="utf-8")
            backup = calibrate.apply_calibration_to_profile(profile, result)
            self.assertTrue(backup.exists())
            self.assertEqual(backup.read_text(encoding="utf-8"), original)
            updated = profile.read_text(encoding="utf-8")
            self.assertEqual(updated.count("compRangeBiasAndRxChanPhase"), 1)
            self.assertIn(result.command, updated)

    def test_applied_line_drives_normal_range_and_angle_processing(self) -> None:
        coefficients = tuple(
            (0.8 + 0.02 * index) * np.exp(1j * 0.05 * index)
            for index in range(12)
        )
        result = calibrate.CalibrationResult(
            target_distance_m=1.0,
            search_window_m=0.2,
            measured_range_m=1.04,
            range_bias_m=0.04,
            coefficients=coefficients,
            accepted_frames=64,
            range_std_m=0.001,
            max_phase_std_deg=1.0,
            max_magnitude_cv=0.01,
            tx_order=(1, 4, 2),
        )
        operational = ROOT / "rawdatacapture" / "profile.cfg"
        with tempfile.TemporaryDirectory() as directory:
            profile = Path(directory) / "profile.cfg"
            profile.write_bytes(operational.read_bytes())
            calibrate.apply_calibration_to_profile(profile, result)
            parsed = RadarCaptureConfig.from_file(profile)
            self.assertAlmostEqual(parsed.range_axis_m()[0], -0.04, places=6)
            self.assertTrue(
                np.allclose(parsed.rx_channel_compensation, coefficients, atol=1e-6)
            )

    def test_angular_apply_preserves_other_axis_and_is_parsed_by_normal_dsp(self) -> None:
        operational = ROOT / "rawdatacapture" / "profile.cfg"
        with tempfile.TemporaryDirectory() as directory:
            profile = Path(directory) / "profile.cfg"
            profile.write_bytes(operational.read_bytes())
            azimuth = calibrate.AngularCalibrationResult(
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
            elevation = calibrate.AngularCalibrationResult(
                calibration_type="elevation",
                target_distance_m=1.0,
                search_window_m=0.2,
                reference_angle_deg=-10.0,
                measured_angle_deg=-13.0,
                angle_bias_deg=-3.0,
                accepted_frames=64,
                angle_std_deg=0.2,
                measured_range_m=1.0,
                tx_order=(1, 4, 2),
            )
            first_backup = calibrate.apply_angular_calibration_to_profile(
                profile, azimuth
            )
            second_backup = calibrate.apply_angular_calibration_to_profile(
                profile, elevation
            )
            self.assertTrue(first_backup.exists())
            self.assertTrue(second_backup.exists())
            text = profile.read_text(encoding="utf-8")
            self.assertEqual(text.count(calibrate.HOST_ANGLE_CALIBRATION_MARKER), 1)
            parsed = RadarCaptureConfig.from_file(profile)
            self.assertAlmostEqual(parsed.azimuth_bias_deg, 2.0)
            self.assertAlmostEqual(parsed.elevation_bias_deg, -3.0)
            corrected_azimuth_u, corrected_elevation_u = (
                _apply_host_angle_calibration(
                    np.asarray([np.sin(np.deg2rad(12.0))]),
                    np.asarray([-np.sin(np.deg2rad(-13.0))]),
                    parsed,
                )
            )
            self.assertAlmostEqual(
                np.rad2deg(np.arcsin(corrected_azimuth_u[0])), 10.0, places=5
            )
            self.assertAlmostEqual(
                -np.rad2deg(np.arcsin(corrected_elevation_u[0])), -10.0, places=5
            )


if __name__ == "__main__":
    unittest.main()
