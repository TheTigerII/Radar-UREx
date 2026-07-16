import importlib.util
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from rawdatacapture.dsp import (
    build_virtual_antenna_grid,
    build_virtual_antenna_grids,
    cluster_point_cloud,
    compute_micro_doppler_spectrum,
    compute_point_cloud,
    doppler_peak_mask,
    estimate_xyz_from_virtual_array,
    estimate_xyz_from_virtual_arrays,
)
from rawdatacapture.openradar_backend import (
    _os_scale,
    _os_thresholds_along_axis,
    doppler_fft,
    os_cfar_2d,
    range_fft,
)


OPENRADAR_AVAILABLE = importlib.util.find_spec("mmwave") is not None


class BatchedAngleFftTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = SimpleNamespace(tx_channel_masks=(1, 4, 2))

    def test_batched_virtual_grids_match_single_cell_mapping(self) -> None:
        rng = np.random.default_rng(11)
        samples = (
            rng.normal(size=(7, 3, 4))
            + 1j * rng.normal(size=(7, 3, 4))
        ).astype(np.complex64)

        batched = build_virtual_antenna_grids(samples, self.config)
        singles = np.stack(
            [build_virtual_antenna_grid(cell, self.config) for cell in samples]
        )

        np.testing.assert_array_equal(batched, singles)

    def test_batched_angle_fft_matches_single_cell_results(self) -> None:
        rng = np.random.default_rng(17)
        samples = (
            rng.normal(size=(9, 3, 4))
            + 1j * rng.normal(size=(9, 3, 4))
        ).astype(np.complex64)
        ranges_m = np.linspace(0.25, 2.25, samples.shape[0])

        batched_xyz, batched_valid = estimate_xyz_from_virtual_arrays(
            samples,
            ranges_m,
            self.config,
        )
        single_results = [
            estimate_xyz_from_virtual_array(cell, range_m, self.config)
            for cell, range_m in zip(samples, ranges_m)
        ]
        single_valid = np.asarray([result is not None for result in single_results])

        np.testing.assert_array_equal(batched_valid, single_valid)
        for index in np.flatnonzero(single_valid):
            np.testing.assert_allclose(batched_xyz[index], single_results[index])

    def test_zero_virtual_arrays_point_forward(self) -> None:
        samples = np.zeros((3, 3, 4), dtype=np.complex64)
        ranges_m = np.asarray((0.25, 1.0, 2.0))

        xyz_m, valid = estimate_xyz_from_virtual_arrays(
            samples,
            ranges_m,
            self.config,
        )

        np.testing.assert_array_equal(valid, np.ones(3, dtype=bool))
        np.testing.assert_allclose(
            xyz_m,
            np.column_stack((np.zeros(3), ranges_m, np.zeros(3))),
        )


class PointCloudCandidateOrderingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = SimpleNamespace(tx_channel_masks=(1, 4, 2))
        self.doppler_cube = np.ones((2, 3, 4, 3), dtype=np.complex64)
        self.doppler_cube[0, :, :, 0] *= 1.0
        self.doppler_cube[0, :, :, 1] *= 3.0
        self.doppler_cube[1, :, :, 2] *= 2.0
        self.detections = np.zeros((2, 3), dtype=bool)
        self.detections[0, 0] = True
        self.detections[0, 1] = True
        self.detections[1, 2] = True

    @staticmethod
    def _forward_xyz(samples, ranges_m, config):
        del samples, config
        return (
            np.column_stack(
                (np.zeros(ranges_m.size), ranges_m, np.zeros(ranges_m.size))
            ),
            np.ones(ranges_m.size, dtype=bool),
        )

    def test_unlimited_points_bypass_power_sorting(self) -> None:
        with (
            patch(
                "rawdatacapture.dsp.compute_range_doppler_fft",
                return_value=self.doppler_cube,
            ),
            patch(
                "rawdatacapture.dsp.os_cfar_2d",
                return_value=self.detections.copy(),
            ),
            patch(
                "rawdatacapture.dsp.doppler_peak_mask",
                return_value=np.ones_like(self.detections),
            ),
            patch(
                "rawdatacapture.dsp.estimate_xyz_from_virtual_arrays",
                side_effect=self._forward_xyz,
            ),
            patch("rawdatacapture.dsp.np.argsort") as argsort,
        ):
            points = compute_point_cloud(
                np.empty((0,)),
                np.asarray((0.25, 0.5, 0.75)),
                self.config,
                max_points=None,
            )

        argsort.assert_not_called()
        np.testing.assert_allclose(points[:, 1], (0.25, 0.5, 0.75))

    def test_finite_point_cap_keeps_strongest_first(self) -> None:
        with (
            patch(
                "rawdatacapture.dsp.compute_range_doppler_fft",
                return_value=self.doppler_cube,
            ),
            patch(
                "rawdatacapture.dsp.os_cfar_2d",
                return_value=self.detections.copy(),
            ),
            patch(
                "rawdatacapture.dsp.doppler_peak_mask",
                return_value=np.ones_like(self.detections),
            ),
            patch(
                "rawdatacapture.dsp.estimate_xyz_from_virtual_arrays",
                side_effect=self._forward_xyz,
            ),
        ):
            points = compute_point_cloud(
                np.empty((0,)),
                np.asarray((0.25, 0.5, 0.75)),
                self.config,
                max_points=2,
            )

        np.testing.assert_allclose(points[:, 1], (0.5, 0.75))


class DopplerPeakMaskTests(unittest.TestCase):
    def test_preserves_adjacent_range_peaks(self) -> None:
        power = np.ones((5, 4), dtype=np.float64)
        power[2, 1] = 10.0
        power[2, 2] = 9.0

        peaks = doppler_peak_mask(power)

        self.assertTrue(peaks[2, 1])
        self.assertTrue(peaks[2, 2])

    def test_rejects_weaker_cyclic_doppler_neighbor(self) -> None:
        power = np.ones((5, 1), dtype=np.float64)
        power[0, 0] = 2.0
        power[-1, 0] = 3.0

        peaks = doppler_peak_mask(power)

        self.assertFalse(peaks[0, 0])
        self.assertTrue(peaks[-1, 0])


class MicroDopplerSpectrumTests(unittest.TestCase):
    def test_auto_gate_uses_strongest_nonzero_doppler_range(self) -> None:
        doppler_cube = np.zeros((8, 1, 1, 6), dtype=np.complex64)
        doppler_cube[0, 0, 0, 4] = 3.0
        doppler_cube[4, 0, 0, 2] = 100.0
        range_axis = np.arange(6, dtype=np.float32)

        spectrum_db, selected_range_m = compute_micro_doppler_spectrum(
            doppler_cube,
            range_axis,
            range_half_width_bins=0,
            min_range_m=1.0,
            max_range_m=5.0,
        )

        self.assertEqual(selected_range_m, 4.0)
        self.assertEqual(int(np.argmax(spectrum_db)), 0)

    def test_explicit_target_range_combines_gate_power_before_log(self) -> None:
        doppler_cube = np.zeros((4, 2, 1, 5), dtype=np.complex64)
        doppler_cube[1, :, :, 1:4] = 1.0
        range_axis = np.arange(5, dtype=np.float32) * 0.5

        spectrum_db, selected_range_m = compute_micro_doppler_spectrum(
            doppler_cube,
            range_axis,
            target_range_m=1.0,
            range_half_width_bins=1,
            min_range_m=0.25,
            max_range_m=2.0,
        )

        self.assertEqual(selected_range_m, 1.0)
        self.assertAlmostEqual(
            float(spectrum_db[1]),
            10.0 * np.log10(6.0),
            places=6,
        )


class PointCloudClusteringTests(unittest.TestCase):
    def test_dbscan_returns_cluster_centers_and_ignores_noise(self) -> None:
        points = np.asarray(
            (
                (0.0, 1.0, 0.0, 50.0),
                (0.1, 1.1, 0.0, 55.0),
                (3.0, 3.0, 0.0, 60.0),
            ),
            dtype=np.float32,
        )

        clusters = cluster_point_cloud(points, eps_m=0.25, min_samples=2)

        self.assertEqual(clusters.shape, (1, 4))
        np.testing.assert_allclose(clusters[0, :3], (0.05, 1.05, 0.0))
        self.assertEqual(clusters[0, 3], 2.0)

    def test_zero_radius_disables_clustering(self) -> None:
        points = np.ones((2, 4), dtype=np.float32)

        clusters = cluster_point_cloud(points, eps_m=0.0)

        self.assertEqual(clusters.shape, (0, 4))


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
