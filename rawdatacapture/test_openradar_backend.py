import importlib.util
import unittest

import numpy as np

from rawdatacapture.openradar_backend import (
    _os_scale,
    _os_thresholds_along_axis,
    doppler_fft,
    os_cfar_2d,
    range_fft,
)


OPENRADAR_AVAILABLE = importlib.util.find_spec("mmwave") is not None


class OsCfarParameterTests(unittest.TestCase):
    def test_scale_matches_requested_false_alarm_rate(self) -> None:
        training_cells = 8
        rank = 6
        expected_pfa = 1e-3

        scale = _os_scale(training_cells, rank, expected_pfa)
        actual_pfa = np.prod(
            [
                (training_cells - index) / (training_cells - index + scale)
                for index in range(rank)
            ]
        )

        self.assertAlmostEqual(actual_pfa, expected_pfa, places=12)

    def test_vectorized_windows_exclude_cut_and_guard_cells(self) -> None:
        power = np.arange(12, dtype=np.float64)[np.newaxis, :]

        thresholds = _os_thresholds_along_axis(
            power,
            axis=1,
            guard_cells=1,
            training_cells=2,
            rank_index=2,
            scale=1.0,
        )

        # CUT 5 trains on [2, 3, 7, 8], whose zero-based rank 2 is 7.
        self.assertEqual(thresholds[0, 5], 7.0)
        # CUT 0 wraps and trains on [9, 10, 2, 3], whose rank 2 is 9.
        self.assertEqual(thresholds[0, 0], 9.0)

    def test_vectorized_thresholds_support_doppler_axis(self) -> None:
        power = np.arange(12, dtype=np.float64)[:, np.newaxis]

        thresholds = _os_thresholds_along_axis(
            power,
            axis=0,
            guard_cells=1,
            training_cells=2,
            rank_index=2,
            scale=1.0,
        )

        self.assertEqual(thresholds[5, 0], 7.0)


@unittest.skipUnless(OPENRADAR_AVAILABLE, "OpenRadar is not installed")
class OpenRadarBackendTests(unittest.TestCase):
    def test_range_and_doppler_cube_layouts(self) -> None:
        rng = np.random.default_rng(7)
        adc_cube = (
            rng.normal(size=(6, 4, 8))
            + 1j * rng.normal(size=(6, 4, 8))
        ).astype(np.complex64)

        range_cube = range_fft(adc_cube)
        doppler_cube = doppler_fft(range_cube, num_tx_antennas=3)

        self.assertEqual(range_cube.shape, (6, 4, 8))
        self.assertEqual(doppler_cube.shape, (2, 3, 4, 8))

    def test_os_cfar_detects_an_isolated_strong_cell(self) -> None:
        power_map = np.ones((16, 32), dtype=np.float64)
        power_map[8, 16] = 1e8

        detections = os_cfar_2d(
            power_map,
            false_alarm_rate=1e-3,
            range_guard_cells=2,
            doppler_guard_cells=1,
            range_training_cells=4,
            doppler_training_cells=2,
        )

        self.assertTrue(detections[8, 16])


if __name__ == "__main__":
    unittest.main()
