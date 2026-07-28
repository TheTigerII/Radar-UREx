import importlib.util
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

from rawdatacapture.dsp import (
    AdaptiveClutterMap,
    DEFAULT_ROTOR_NOISE_GATE_MAX_DB,
    DEFAULT_ROTOR_NOISE_GATE_MIN_DB,
    StaticSceneMap,
    _apply_rotor_noise_support_filter,
    _rotor_noise_floor_and_gate,
    _local_maxima_3d,
    _local_maximum_candidate_indices,
    build_virtual_antenna_grid,
    build_virtual_antenna_grids,
    cluster_point_cloud,
    cluster_point_cloud_with_labels,
    compute_micro_doppler_spectrum,
    compute_per_tx_micro_doppler_spectrogram,
    compute_rotor_micro_doppler_frame,
    compute_point_cloud,
    compute_static_angle_power,
    compute_static_point_cloud,
    doppler_peak_mask,
    estimate_xyz_from_virtual_array,
    estimate_xyz_from_virtual_arrays,
    estimate_rotor_rpm,
    micro_doppler_velocity_axis_m_s,
    rotor_velocity_alias_diagnostic,
    static_target_protection_mask,
)
from rawdatacapture.openradar_backend import (
    _os_scale,
    _os_thresholds_along_axis,
    doppler_fft,
    os_cfar_2d,
    range_fft,
)


OPENRADAR_AVAILABLE = importlib.util.find_spec("mmwave") is not None
SKLEARN_AVAILABLE = importlib.util.find_spec("sklearn") is not None


class AdaptiveClutterMapTests(unittest.TestCase):
    def test_normalizes_background_after_warmup_and_preserves_new_target(self) -> None:
        clutter = AdaptiveClutterMap(update_rate=0.5, warmup_frames=2)
        background = np.full((5, 8), 10.0)

        np.testing.assert_array_equal(clutter.normalize(background), 0.0)
        clutter.update(background)
        np.testing.assert_array_equal(clutter.normalize(background), 0.0)
        clutter.update(background)

        target_frame = background.copy()
        target_frame[2, 4] = 30.0
        normalized = clutter.normalize(target_frame)

        self.assertTrue(clutter.is_ready)
        self.assertEqual(normalized[2, 4], 3.0)
        np.testing.assert_array_equal(normalized[normalized != 3.0], 1.0)

    def test_detection_protection_prevents_target_absorption(self) -> None:
        clutter = AdaptiveClutterMap(update_rate=1.0, warmup_frames=1)
        background = np.ones((5, 8))
        clutter.normalize(background)
        clutter.update(background)

        target_frame = background.copy()
        target_frame[2, 4] = 20.0
        detections = np.zeros_like(target_frame, dtype=bool)
        detections[2, 4] = True
        clutter.update(target_frame, detections)

        self.assertEqual(clutter.normalize(target_frame)[2, 4], 20.0)

    def test_shape_change_resets_warmup(self) -> None:
        clutter = AdaptiveClutterMap(warmup_frames=1)
        first = np.ones((2, 3))
        clutter.normalize(first)
        clutter.update(first)
        self.assertTrue(clutter.is_ready)

        np.testing.assert_array_equal(clutter.normalize(np.ones((3, 2))), 0.0)
        self.assertFalse(clutter.is_ready)


class StaticSceneMapTests(unittest.TestCase):
    def test_frozen_reference_preserves_new_static_change(self) -> None:
        scene = StaticSceneMap(
            warmup_frames=0,
            reference_frames=2,
            minimum_change_db=6.0,
        )
        background = np.full((3, 4, 5), 10.0, dtype=np.float32)

        self.assertIsNone(scene.observe(background))
        self.assertIsNone(scene.observe(background))
        self.assertTrue(scene.is_ready)

        unchanged = scene.observe(background)
        assert unchanged is not None
        np.testing.assert_allclose(unchanged, 0.0)

        changed = background.copy()
        changed[1, 2, 3] = 100.0
        first_change_db = scene.observe(changed)
        second_change_db = scene.observe(changed)
        change_db = scene.observe(changed)
        assert first_change_db is not None
        assert second_change_db is not None
        assert change_db is not None
        self.assertLess(float(first_change_db[1, 2, 3]), 6.0)
        self.assertLess(float(second_change_db[1, 2, 3]), 6.0)
        self.assertGreater(float(change_db[1, 2, 3]), 6.0)
        self.assertTrue(scene.detection_mask(change_db)[1, 2, 3])

        repeated_change_db = scene.observe(changed)
        assert repeated_change_db is not None
        self.assertGreater(
            float(repeated_change_db[1, 2, 3]),
            float(change_db[1, 2, 3]),
        )

    def test_common_gain_drift_and_weak_reference_cells_are_suppressed(
        self,
    ) -> None:
        scene = StaticSceneMap(
            warmup_frames=0,
            reference_frames=3,
            minimum_change_db=6.0,
        )
        background = np.full((2, 8, 8), 100.0, dtype=np.float32)
        background[1, 3, 4] = 1e-12
        for _ in range(3):
            self.assertIsNone(scene.observe(background))

        gain_drift = background * 4.0
        gain_drift[1, 3, 4] = 1e-6
        for _ in range(5):
            change_db = scene.observe(gain_drift)

        assert change_db is not None
        self.assertFalse(np.any(scene.detection_mask(change_db)))

    def test_calibration_variability_raises_per_cell_detection_limit(
        self,
    ) -> None:
        scene = StaticSceneMap(
            warmup_frames=0,
            reference_frames=5,
            minimum_change_db=6.0,
            smoothing_rate=1.0,
        )
        background = np.full((2, 8, 8), 100.0, dtype=np.float32)
        noisy_levels = (100.0, 200.0, 400.0, 800.0, 1600.0)
        for level in noisy_levels:
            sample = background.copy()
            sample[1, 3, 4] = level
            self.assertIsNone(scene.observe(sample))

        changed = background.copy()
        changed[0, 2, 3] = 1000.0
        changed[1, 3, 4] = 4000.0
        change_db = scene.observe(changed)

        assert change_db is not None
        detections = scene.detection_mask(change_db)
        self.assertTrue(detections[0, 2, 3])
        self.assertFalse(detections[1, 3, 4])

    def test_target_present_during_calibration_is_part_of_reference(self) -> None:
        scene = StaticSceneMap(warmup_frames=0, reference_frames=3)
        occupied_scene = np.ones((2, 3, 4), dtype=np.float32)
        occupied_scene[1, 1, 2] = 20.0

        for _ in range(3):
            self.assertIsNone(scene.observe(occupied_scene))

        change_db = scene.observe(occupied_scene)

        assert change_db is not None
        np.testing.assert_allclose(change_db, 0.0)

    def test_warmup_frames_are_discarded_before_calibration(self) -> None:
        scene = StaticSceneMap(warmup_frames=2, reference_frames=3)
        background = np.ones((2, 3, 4), dtype=np.float32)

        for _ in range(2):
            self.assertIsNone(scene.observe(background))
        self.assertFalse(scene.is_warming_up)
        self.assertEqual(scene.frames_seen, 0)
        for _ in range(3):
            self.assertIsNone(scene.observe(background))

        self.assertTrue(scene.is_ready)
        self.assertEqual(scene.warmup_frames_seen, 2)

    def test_adaptive_background_absorbs_unprotected_drift(self) -> None:
        scene = StaticSceneMap(
            warmup_frames=0,
            reference_frames=3,
            minimum_change_db=6.0,
            smoothing_rate=1.0,
            background_update_rate=0.1,
        )
        background = np.full((2, 8, 8), 100.0, dtype=np.float32)
        for _ in range(3):
            scene.observe(background)
        drifted = background.copy()
        drifted[1, 3, 4] = 1000.0

        for _ in range(12):
            change_db = scene.observe(drifted)
            assert change_db is not None
            scene.adapt()

        self.assertFalse(scene.detection_mask(change_db)[1, 3, 4])

    def test_protected_background_change_remains_detectable(self) -> None:
        scene = StaticSceneMap(
            warmup_frames=0,
            reference_frames=3,
            minimum_change_db=6.0,
            smoothing_rate=1.0,
            background_update_rate=0.1,
        )
        background = np.full((2, 8, 8), 100.0, dtype=np.float32)
        for _ in range(3):
            scene.observe(background)
        changed = background.copy()
        changed[1, 3, 4] = 1000.0
        protected = np.zeros(changed.shape, dtype=bool)
        protected[1, 3, 4] = True

        for _ in range(12):
            change_db = scene.observe(changed)
            assert change_db is not None
            scene.adapt(protected)

        self.assertTrue(scene.detection_mask(change_db)[1, 3, 4])

    def test_released_target_is_eventually_absorbed(self) -> None:
        scene = StaticSceneMap(
            warmup_frames=0,
            reference_frames=3,
            minimum_change_db=6.0,
            smoothing_rate=1.0,
            background_update_rate=0.1,
        )
        background = np.full((2, 8, 8), 100.0, dtype=np.float32)
        for _ in range(3):
            scene.observe(background)
        changed = background.copy()
        changed[1, 3, 4] = 1000.0
        protected = np.zeros(changed.shape, dtype=bool)
        protected[1, 3, 4] = True

        for _ in range(5):
            change_db = scene.observe(changed)
            assert change_db is not None
            scene.adapt(protected)
        self.assertTrue(scene.detection_mask(change_db)[1, 3, 4])

        for _ in range(15):
            change_db = scene.observe(changed)
            assert change_db is not None
            scene.adapt()
        self.assertFalse(scene.detection_mask(change_db)[1, 3, 4])

    def test_shape_change_restarts_static_warmup_and_calibration(self) -> None:
        scene = StaticSceneMap(warmup_frames=1, reference_frames=2)
        first_shape = np.ones((2, 3, 4), dtype=np.float32)
        for _ in range(3):
            scene.observe(first_shape)
        self.assertTrue(scene.is_ready)

        self.assertIsNone(scene.observe(np.ones((3, 3, 4), dtype=np.float32)))

        self.assertFalse(scene.is_ready)
        self.assertEqual(scene.warmup_frames_seen, 1)
        self.assertEqual(scene.frames_seen, 0)

    def test_static_target_protection_mask_uses_range_and_angle_cells(self) -> None:
        range_axis = np.arange(10, dtype=np.float32)

        protected = static_target_protection_mask(
            (10, 32, 32),
            np.asarray(((0.0, 4.0, 0.0),), dtype=np.float32),
            range_axis,
            neighborhood_cells=2,
        )

        self.assertTrue(protected[4, 16, 16])
        self.assertEqual(np.count_nonzero(protected), 125)

    def test_static_angle_power_integrates_centered_doppler_neighbors(self) -> None:
        config = SimpleNamespace(tx_channel_masks=(1,))
        doppler_cube = np.zeros((8, 1, 2, 5), dtype=np.complex64)
        doppler_cube[3:6] = 1.0

        power = compute_static_angle_power(
            doppler_cube,
            config,
            doppler_half_width_bins=1,
            angle_fft_size=8,
        )

        self.assertEqual(power.shape, (5, 8, 8))
        self.assertGreater(float(np.max(power)), 0.0)

    def test_static_angle_power_matches_full_precision_32_point_fft(self) -> None:
        config = SimpleNamespace(tx_channel_masks=(1, 4, 2))
        rng = np.random.default_rng(12)
        doppler_cube = (
            rng.standard_normal((8, 3, 4, 5))
            + 1j * rng.standard_normal((8, 3, 4, 5))
        ).astype(np.complex64)

        actual = compute_static_angle_power(doppler_cube, config)

        selected = np.moveaxis(doppler_cube[3:6], -1, 1).reshape(
            (-1, 3, 4)
        )
        grids = build_virtual_antenna_grids(selected, config)
        response = np.fft.fftshift(
            np.fft.fft2(grids, s=(32, 32), axes=(-2, -1)),
            axes=(-2, -1),
        )
        expected = (np.abs(response) ** 2).reshape((3, 5, 32, 32)).sum(
            axis=0
        )

        self.assertEqual(actual.shape, (5, 32, 32))
        np.testing.assert_allclose(actual, expected, rtol=2e-6, atol=2e-5)

    def test_candidate_local_maxima_matches_full_cube_filter(self) -> None:
        rng = np.random.default_rng(4)
        values = rng.standard_normal((7, 8, 9)).astype(np.float32)
        all_indices = np.argwhere(np.ones(values.shape, dtype=bool))
        expected_mask = _local_maxima_3d(values)

        for candidates in (all_indices[::17], all_indices):
            actual = _local_maximum_candidate_indices(values, candidates)
            expected = candidates[expected_mask[tuple(candidates.T)]]
            np.testing.assert_array_equal(actual, expected)

    def test_static_points_separate_changes_at_same_range_by_angle(self) -> None:
        config = SimpleNamespace(tx_channel_masks=(1,))
        scene = StaticSceneMap(
            warmup_frames=0,
            reference_frames=1,
            minimum_change_db=6.0,
            smoothing_rate=1.0,
        )
        doppler_cube = np.zeros((8, 1, 1, 5), dtype=np.complex64)
        range_axis = np.arange(5, dtype=np.float32)
        background = np.ones((5, 8, 8), dtype=np.float32)
        changed = background.copy()
        changed[2, 4, 2] = 10.0
        changed[2, 4, 6] = 20.0

        with patch(
            "rawdatacapture.dsp.compute_static_angle_power",
            side_effect=(background, changed),
        ):
            calibration_points = compute_static_point_cloud(
                doppler_cube,
                range_axis,
                config,
                scene,
                angle_fft_size=8,
                azimuth_fov_deg=90.0,
                elevation_fov_deg=90.0,
            )
            points = compute_static_point_cloud(
                doppler_cube,
                range_axis,
                config,
                scene,
                angle_fft_size=8,
                azimuth_fov_deg=90.0,
                elevation_fov_deg=90.0,
            )

        self.assertEqual(calibration_points.shape, (0, 5))
        self.assertEqual(points.shape, (2, 5))
        np.testing.assert_allclose(points[:, 1], np.sqrt(3.0), atol=1e-6)
        np.testing.assert_allclose(np.sort(points[:, 0]), (-1.0, 1.0))
        np.testing.assert_allclose(points[:, 2], 0.0)

    def test_point_detection_uses_normalized_power_and_updates_map(self) -> None:
        doppler_cube = np.full((5, 1, 1, 8), 2.0, dtype=np.complex64)
        normalized_power = np.zeros((5, 8), dtype=np.float64)
        detections = np.zeros((5, 8), dtype=bool)
        clutter = Mock(spec=AdaptiveClutterMap)
        clutter.normalize.return_value = normalized_power
        clutter.minimum_snr_linear = 4.0

        with patch(
            "rawdatacapture.dsp.os_cfar_2d",
            return_value=detections,
        ) as cfar:
            points = compute_point_cloud(
                np.empty((0,)),
                np.arange(8, dtype=np.float32),
                SimpleNamespace(),
                doppler_cube=doppler_cube,
                clutter_map=clutter,
                min_range_m=0.0,
            )

        self.assertEqual(points.shape, (0, 4))
        np.testing.assert_array_equal(clutter.normalize.call_args.args[0], 4.0)
        np.testing.assert_array_equal(cfar.call_args.args[0], normalized_power)
        self.assertEqual(cfar.call_args.kwargs["false_alarm_rate"], 1e-3)
        np.testing.assert_array_equal(clutter.update.call_args.args[0], 4.0)
        np.testing.assert_array_equal(clutter.update.call_args.args[1], detections)

    def test_minimum_snr_gate_rejects_normalized_background(self) -> None:
        doppler_cube = np.full((5, 1, 1, 8), 2.0, dtype=np.complex64)
        normalized_power = np.ones((5, 8), dtype=np.float64)
        normalized_power[2, 4] = 5.0
        clutter = Mock(spec=AdaptiveClutterMap)
        clutter.normalize.return_value = normalized_power
        clutter.minimum_snr_linear = 4.0

        with (
            patch(
                "rawdatacapture.dsp.os_cfar_2d",
                return_value=np.ones((5, 8), dtype=bool),
            ),
            patch(
                "rawdatacapture.dsp.estimate_xyz_from_virtual_arrays",
                return_value=(
                    np.asarray(((0.0, 4.0, 0.0),)),
                    np.asarray((True,)),
                ),
            ),
        ):
            points = compute_point_cloud(
                np.empty((0,)),
                np.arange(8, dtype=np.float32),
                SimpleNamespace(tx_channel_masks=None),
                doppler_cube=doppler_cube,
                clutter_map=clutter,
                min_range_m=0.0,
            )

        self.assertEqual(points.shape, (1, 4))
        self.assertAlmostEqual(float(points[0, 3]), 10.0 * np.log10(4.0), places=5)
        protected = clutter.update.call_args.args[1]
        self.assertEqual(np.count_nonzero(protected), 1)
        self.assertTrue(protected[2, 4])


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

    def test_explicit_static_target_keeps_centered_zero_doppler_bin(self) -> None:
        doppler_cube = np.zeros((8, 1, 1, 5), dtype=np.complex64)
        doppler_cube[4, 0, 0, 2] = 10.0
        doppler_cube[1, 0, 0, 4] = 20.0
        range_axis = np.arange(5, dtype=np.float32)

        spectrum_db, selected_range_m = compute_micro_doppler_spectrum(
            doppler_cube,
            range_axis,
            target_range_m=2.0,
            range_half_width_bins=0,
            min_range_m=1.0,
            max_range_m=4.0,
        )

        self.assertEqual(selected_range_m, 2.0)
        self.assertEqual(int(np.argmax(spectrum_db)), 4)

    def test_per_tx_stft_uses_64_loop_window_and_32_loop_hop(self) -> None:
        loop_count = 128
        chirps_per_loop = 3
        fft_size = 128
        tone_bin = 8
        loop_phase = np.exp(
            2j * np.pi * tone_bin * np.arange(loop_count) / fft_size
        )
        slot_gains = np.asarray(
            (1.0, 0.45 * np.exp(0.8j), 1.6 * np.exp(-1.1j))
        )
        range_cube = np.zeros(
            (loop_count * chirps_per_loop, 1, 3),
            dtype=np.complex64,
        )
        range_cube[:, 0, 1] = (
            loop_phase[:, np.newaxis] * slot_gains[np.newaxis, :]
        ).reshape(-1)

        spectrogram = compute_per_tx_micro_doppler_spectrogram(
            range_cube,
            np.asarray((0.0, 1.0, 2.0)),
            SimpleNamespace(num_chirps_per_loop=chirps_per_loop),
            target_range_m=1.0,
            range_half_width_bins=0,
            window_loops=64,
            hop_loops=32,
            fft_size=fft_size,
        )

        self.assertEqual(spectrogram.shape, (128, 3))
        np.testing.assert_array_equal(
            np.argmax(spectrogram, axis=0),
            np.full(3, (fft_size // 2) + tone_bin),
        )


class RotorMicroDopplerTests(unittest.TestCase):
    @staticmethod
    def config() -> SimpleNamespace:
        return SimpleNamespace(
            num_chirps_per_loop=1,
            idle_time_us=7.0,
            ramp_end_time_us=32.0,
            start_frequency_ghz=60.0,
        )

    def test_single_tx_profile_has_expected_velocity_span(self) -> None:
        velocity = micro_doppler_velocity_axis_m_s(self.config(), 128)

        self.assertEqual(velocity.shape, (128,))
        self.assertAlmostEqual(float(velocity[0]), -32.03, places=1)
        self.assertAlmostEqual(float(velocity[-1]), 31.53, places=1)

    def test_three_tx_profile_processes_each_tx_at_its_physical_slow_time_rate(
        self,
    ) -> None:
        config = SimpleNamespace(
            num_chirps_per_loop=3,
            idle_time_us=7.0,
            ramp_end_time_us=32.0,
            start_frequency_ghz=60.0,
        )
        range_fft_cube = np.ones((384, 4, 8), dtype=np.complex64)

        result = compute_rotor_micro_doppler_frame(
            range_fft_cube,
            np.arange(8, dtype=np.float32),
            config,
            target_range_m=2.0,
            frame_time_s=0.0,
            range_half_width_bins=0,
        )

        self.assertEqual(result.raw_spectrogram_db.shape, (128, 57))
        self.assertAlmostEqual(
            float(result.unambiguous_velocity_m_s),
            10.68,
            places=1,
        )
        self.assertAlmostEqual(
            float(result.window_times_s[0]),
            7.5 * 117e-6,
            places=9,
        )

    def test_weighted_mean_rejects_static_body_and_preserves_tone(self) -> None:
        loop_count = 128
        tone_bin = 12
        loops = np.arange(loop_count)
        rng = np.random.default_rng(22)
        receiver_noise = 0.1 * (
            rng.normal(size=loop_count)
            + 1j * rng.normal(size=loop_count)
        )
        target = (
            100.0
            + 10.0 * np.exp(2j * np.pi * tone_bin * loops / 128)
            + receiver_noise
        )
        range_fft_cube = np.zeros((loop_count, 1, 5), dtype=np.complex64)
        range_fft_cube[:, 0, 2] = target
        range_fft_cube[:, 0, 4] = 10_000.0

        result = compute_rotor_micro_doppler_frame(
            range_fft_cube,
            np.arange(5, dtype=np.float32),
            self.config(),
            target_range_m=2.0,
            frame_time_s=1.0,
            range_half_width_bins=0,
        )

        self.assertEqual(result.raw_spectrogram_db.shape, (128, 57))
        self.assertLessEqual(
            abs(
                int(np.argmax(result.enhanced_spectrogram_db[:, 0]))
                - (64 + tone_bin)
            ),
            1,
        )
        self.assertEqual(float(result.enhanced_spectrogram_db[64, 0]), 0.0)
        self.assertGreater(
            float(result.enhanced_spectrogram_db[64 + tone_bin, 0]),
            30.0,
        )
        self.assertAlmostEqual(
            float(result.window_times_s[0]),
            1.0 + 7.5 * 39e-6,
            places=9,
        )
        self.assertIsNotNone(result.noise_gate_db)
        assert result.noise_gate_db is not None
        self.assertEqual(result.noise_gate_db.shape, (57,))
        self.assertTrue(
            np.all(result.noise_gate_db >= DEFAULT_ROTOR_NOISE_GATE_MIN_DB)
        )
        self.assertTrue(
            np.all(result.noise_gate_db <= DEFAULT_ROTOR_NOISE_GATE_MAX_DB)
        )

    def test_deep_cancellation_nulls_cannot_blank_visible_ridges(self) -> None:
        spectrum_db = np.full((128, 7), -120.0, dtype=np.float32)
        off_center = np.ones(128, dtype=bool)
        off_center[62:67] = False
        spectrum_db[off_center, :] = np.linspace(
            -120.0,
            0.0,
            int(np.count_nonzero(off_center)),
            dtype=np.float32,
        )[:, np.newaxis]

        _, noise_gate_db = _rotor_noise_floor_and_gate(
            spectrum_db,
            off_center,
        )

        np.testing.assert_allclose(
            noise_gate_db,
            DEFAULT_ROTOR_NOISE_GATE_MAX_DB,
        )

    def test_adaptive_gate_blanks_complex_noise(self) -> None:
        rng = np.random.default_rng(91)
        range_fft_cube = (
            rng.normal(size=(128, 4, 5))
            + 1j * rng.normal(size=(128, 4, 5))
        ).astype(np.complex64)

        result = compute_rotor_micro_doppler_frame(
            range_fft_cube,
            np.arange(5, dtype=np.float32),
            self.config(),
            target_range_m=2.0,
            frame_time_s=0.0,
            range_half_width_bins=1,
        )

        occupancy = np.count_nonzero(
            result.enhanced_spectrogram_db
        ) / result.enhanced_spectrogram_db.size
        self.assertLess(occupancy, 0.01)

    def test_lower_tail_noise_gate_ignores_positive_blade_ridge(self) -> None:
        off_center = np.ones(128, dtype=bool)
        off_center[62:67] = False
        baseline = np.zeros((128, 4), dtype=np.float32)
        baseline[off_center] = np.linspace(
            -2.0,
            2.0,
            int(np.count_nonzero(off_center)),
            dtype=np.float32,
        )[:, np.newaxis]
        contaminated = baseline.copy()
        contaminated[np.flatnonzero(off_center)[-15:]] += 20.0

        baseline_floor, baseline_gate = _rotor_noise_floor_and_gate(
            baseline,
            off_center,
        )
        contaminated_floor, contaminated_gate = _rotor_noise_floor_and_gate(
            contaminated,
            off_center,
        )

        np.testing.assert_allclose(contaminated_floor, baseline_floor)
        np.testing.assert_allclose(contaminated_gate, baseline_gate)

    def test_support_filter_preserves_ridge_peak_and_blanks_isolated_cell(
        self,
    ) -> None:
        relative = np.zeros((128, 5), dtype=np.float32)
        relative[74:79, 1:4] = 8.0
        relative[76, 2] = 12.0
        relative[100, 0] = 15.0
        gates = np.full(5, 3.0, dtype=np.float32)
        off_center = np.ones(128, dtype=bool)
        off_center[62:67] = False

        filtered = _apply_rotor_noise_support_filter(
            relative,
            gates,
            off_center,
        )

        self.assertEqual(float(filtered[76, 2]), 12.0)
        self.assertEqual(float(filtered[100, 0]), 0.0)
        self.assertEqual(float(np.max(filtered[74:79, 1:4])), 12.0)

    def test_gap_aware_rpm_estimate_is_within_five_percent(self) -> None:
        times = []
        for frame_index in range(100):
            times.extend(
                frame_index * 0.010 + np.arange(13, dtype=float) * 0.000312
            )
        times = np.asarray(times)
        blade_passage_hz = 200.0
        scores = 4.0 + 3.0 * np.cos(
            2.0 * np.pi * blade_passage_hz * times
        )

        estimates = estimate_rotor_rpm(
            times,
            scores,
            blade_count=2,
            rpm_min=4_000.0,
            rpm_max=8_000.0,
        )

        self.assertTrue(estimates)
        self.assertLess(abs(estimates[0].rpm - 6_000.0) / 6_000.0, 0.05)
        self.assertGreater(estimates[0].confidence, 0.8)

    def test_gap_aware_rpm_estimate_includes_upper_search_boundary(self) -> None:
        times = np.concatenate(
            [
                frame_index * 0.03333
                + (7.5 + 2.0 * np.arange(57)) * 117e-6
                for frame_index in range(61)
            ]
        )
        rpm = 10_700.0
        blade_passage_hz = rpm * 2.0 / 60.0
        scores = 4.0 + 3.0 * np.cos(
            2.0 * np.pi * blade_passage_hz * times
        )

        estimates = estimate_rotor_rpm(
            times,
            scores,
            blade_count=2,
            rpm_min=500.0,
            rpm_max=rpm,
        )

        self.assertTrue(estimates)
        self.assertLess(abs(estimates[0].rpm - rpm) / rpm, 0.01)

    def test_tip_speed_alias_warning_uses_eighty_percent_margin(self) -> None:
        unambiguous, alias_risk, warning = rotor_velocity_alias_diagnostic(
            self.config(),
            rotor_radius_m=0.05,
            rotor_rpm_max=10_000.0,
        )

        self.assertAlmostEqual(float(unambiguous), 32.03, places=1)
        self.assertTrue(alias_risk)
        self.assertIn("Expected tip speed", warning)


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

    @unittest.skipUnless(SKLEARN_AVAILABLE, "scikit-learn is not installed")
    def test_small_cloud_dbscan_matches_sklearn(self) -> None:
        from sklearn.cluster import DBSCAN

        rng = np.random.default_rng(91)
        points = rng.normal(size=(100, 4)).astype(np.float32)
        expected_labels = DBSCAN(eps=0.55, min_samples=3).fit_predict(
            points[:, :3]
        )
        expected = []
        for label in sorted(set(expected_labels) - {-1}):
            members = points[expected_labels == label, :3]
            expected.append((*members.mean(axis=0), members.shape[0]))

        actual = cluster_point_cloud(points, eps_m=0.55, min_samples=3)

        np.testing.assert_allclose(
            actual,
            np.asarray(expected, dtype=np.float32).reshape((-1, 4)),
        )

    def test_cluster_labels_identify_exact_returned_members(self) -> None:
        points = np.asarray(
            (
                (0.0, 0.0, 0.0, 10.0),
                (0.1, 0.0, 0.0, 20.0),
                (2.0, 0.0, 0.0, 30.0),
            ),
            dtype=np.float32,
        )

        centers, labels = cluster_point_cloud_with_labels(
            points,
            eps_m=0.2,
            min_samples=2,
        )

        self.assertEqual(centers.shape, (1, 4))
        np.testing.assert_array_equal(labels, (0, 0, -1))


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

    def test_optimized_ffts_match_openradar_hann_processing(self) -> None:
        import mmwave.dsp as openradar_dsp

        rng = np.random.default_rng(73)
        adc_cube = (
            rng.normal(size=(12, 4, 16))
            + 1j * rng.normal(size=(12, 4, 16))
        ).astype(np.complex64)
        expected_range = openradar_dsp.range_processing(
            adc_cube,
            window_type_1d=openradar_dsp.Window.HANNING,
        )
        actual_range = range_fft(adc_cube)
        with np.errstate(divide="ignore", invalid="ignore"):
            _unused, expected_aoa = openradar_dsp.doppler_processing(
                expected_range,
                num_tx_antennas=3,
                clutter_removal_enabled=False,
                interleaved=True,
                window_type_2d=openradar_dsp.Window.HANNING,
                accumulate=False,
            )
        expected_doppler = np.fft.fftshift(
            expected_aoa.transpose(2, 1, 0).reshape((4, 3, 4, 16)),
            axes=0,
        )
        actual_doppler = doppler_fft(actual_range, num_tx_antennas=3)

        self.assertEqual(actual_range.dtype, np.complex64)
        self.assertEqual(actual_doppler.dtype, np.complex64)
        np.testing.assert_allclose(actual_range, expected_range, rtol=2e-6)
        np.testing.assert_allclose(actual_doppler, expected_doppler, rtol=3e-6)

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
