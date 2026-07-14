import importlib.util
import unittest

import numpy as np

from rawdatacapture.openradar_backend import ca_cfar_2d, doppler_fft, range_fft


OPENRADAR_AVAILABLE = importlib.util.find_spec("mmwave") is not None


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

    def test_cfar_detects_an_isolated_strong_cell(self) -> None:
        power_map = np.ones((16, 32), dtype=np.float64)
        power_map[8, 16] = 1e8

        detections = ca_cfar_2d(
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
