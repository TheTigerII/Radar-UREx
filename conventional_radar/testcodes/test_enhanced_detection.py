"""Tests for the opt-in Mode 4 detection enhancements."""

from __future__ import annotations

import argparse
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from main import dsp, dsp_kernels
from main import enhanced_detection as ed
from main import run
from main.livedatacapture import COMBINED_DISPLAY_MODE, DisplayPayloadSink

DOPPLER_BINS = 128
TX_SLOTS = 3
RX_COUNT = 4
RANGE_BINS = 64

# IWR6843ISK-ODS virtual-array layout used by dsp._build_iwr6843isk_ods_...
ODS_POSITIONS = {
    (1, 1): (3, 0), (1, 2): (2, 0), (1, 3): (2, 1), (1, 4): (3, 1),
    (2, 1): (3, 2), (2, 2): (2, 2), (2, 3): (2, 3), (2, 4): (3, 3),
    (3, 1): (1, 2), (3, 2): (0, 2), (3, 3): (0, 3), (3, 4): (1, 3),
}
ODS_RX_POLARITY = {1: 1.0, 2: -1.0, 3: -1.0, 4: 1.0}
# profile.cfg chirp order: TX mask 1, 4, 2 -> TX1, TX3, TX2.
SLOT_TX_NUMBER = (1, 3, 2)


def make_config() -> SimpleNamespace:
    return SimpleNamespace(
        num_loops=DOPPLER_BINS,
        num_chirps_per_frame=DOPPLER_BINS * TX_SLOTS,
        num_chirps_per_loop=TX_SLOTS,
        tx_channel_masks=(1, 4, 2),
        rx_channel_compensation=None,
        num_adc_samples=RANGE_BINS,
        azimuth_bias_deg=0.0,
        elevation_bias_deg=0.0,
    )


def synthetic_range_fft(
    rng: np.random.Generator,
    *,
    amplitude: float = 0.0,
    range_bin: int = 40,
    doppler_bins: float = 9.0,
    azimuth_u: float = 0.0,
    elevation_u: float = 0.0,
    noise: float = 1.0,
) -> np.ndarray:
    """Return a ``[chirp, rx, range]`` cube with one TDM-MIMO point target."""
    shape = (DOPPLER_BINS * TX_SLOTS, RX_COUNT, RANGE_BINS)
    cube = noise * (
        rng.standard_normal(shape) + 1j * rng.standard_normal(shape)
    ) / np.sqrt(2.0)
    cube = cube.astype(np.complex64)
    if amplitude > 0.0:
        chirps = np.arange(shape[0])
        slow_time = np.exp(
            2j * np.pi * (doppler_bins / (DOPPLER_BINS * TX_SLOTS)) * chirps
        )
        for slot in range(TX_SLOTS):
            for rx in range(RX_COUNT):
                row, col = ODS_POSITIONS[(SLOT_TX_NUMBER[slot], rx + 1)]
                spatial = np.exp(1j * np.pi * (col * azimuth_u + row * elevation_u))
                cube[slot::TX_SLOTS, rx, range_bin] += (
                    amplitude * slow_time[slot::TX_SLOTS] * spatial
                    / ODS_RX_POLARITY[rx + 1]
                )
    return cube


class OptionsTests(unittest.TestCase):
    def test_default_options_keep_mode_1(self) -> None:
        options = ed.EnhancedDetectionOptions()
        self.assertFalse(options.enabled)
        self.assertEqual(options.threshold_multiplier, 1.0)

    def test_mode_4_enables_every_stage_with_measured_multipliers(self) -> None:
        self.assertTrue(ed.MODE4_OPTIONS.enabled)
        self.assertEqual(ed.MODE4_OPTIONS.detection_power, "coherent-beam")
        self.assertEqual(ed.MODE4_OPTIONS.clutter_reference, "mean")
        self.assertEqual(ed.MODE4_OPTIONS.tbd_window_frames, 8)
        for value in ed.CALIBRATED_THRESHOLD_MULTIPLIER.values():
            self.assertIsNotNone(value)
            self.assertTrue(0.05 < value < 1.0)

    def test_invalid_options_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            ed.EnhancedDetectionOptions(detection_power="magic")
        with self.assertRaises(ValueError):
            ed.EnhancedDetectionOptions(tbd_window_frames=1)


class ThresholdTests(unittest.TestCase):
    def test_thresholds_reproduce_the_mode_1_cfar_decisions(self) -> None:
        rng = np.random.default_rng(0)
        power = rng.exponential(size=(DOPPLER_BINS, RANGE_BINS))
        power[rng.integers(0, DOPPLER_BINS, 40), rng.integers(0, RANGE_BINS, 40)] *= 60
        arguments = dict(
            false_alarm_rate=1e-3,
            range_guard_cells=2,
            doppler_guard_cells=1,
            range_training_cells=4,
            doppler_training_cells=2,
        )
        expected = dsp_kernels.os_cfar_2d(power, **arguments)
        threshold, valid = ed.os_cfar_thresholds(power, **arguments)
        self.assertGreater(int(expected.sum()), 0)
        np.testing.assert_array_equal(valid & (power > threshold), expected)


class ClutterReferenceTests(unittest.TestCase):
    def test_mean_reference_is_the_arithmetic_mean_and_then_freezes(self) -> None:
        rng = np.random.default_rng(1)
        frames = rng.exponential(size=(6, 8, 8))
        clutter = ed.create_clutter_map(
            "mean", update_rate=0.02, warmup_frames=5, minimum_snr_db=3.0
        )
        self.assertIsInstance(clutter, ed.MeanClutterMap)
        for frame in frames[:5]:
            self.assertFalse(clutter.is_ready)
            clutter.update(frame)
        self.assertTrue(clutter.is_ready)
        clutter.update(frames[5])
        np.testing.assert_allclose(
            clutter.normalize(frames[0]),
            frames[0] / frames[:5].mean(axis=0),
            rtol=1e-9,
        )

    def test_ema_reference_is_the_unmodified_mode_1_class(self) -> None:
        clutter = ed.create_clutter_map(
            "ema", update_rate=0.02, warmup_frames=30, minimum_snr_db=3.0
        )
        self.assertIs(type(clutter), dsp.AdaptiveClutterMap)


class CoherentBeamTests(unittest.TestCase):
    def test_moving_off_boresight_target_gains_the_full_array_factor(self) -> None:
        config = make_config()
        rng = np.random.default_rng(2)
        # On-grid direction (bin spacing 0.25) and a high Doppler bin, where
        # the uncompensated TDM phase would badly decohere the array.
        range_fft = synthetic_range_fft(
            rng,
            amplitude=1.0,
            doppler_bins=40.0,
            azimuth_u=0.5,
            elevation_u=-0.25,
            noise=0.0,
        )
        cube = dsp.compute_range_doppler_fft(range_fft, config)
        noncoherent = np.mean(np.abs(cube) ** 2, axis=(1, 2))
        peak = np.unravel_index(np.argmax(noncoherent), noncoherent.shape)
        self.assertEqual(peak, (DOPPLER_BINS // 2 + 40, 40))

        compensated = ed.tdm_doppler_compensation(cube)
        coherent = ed.coherent_beam_power(compensated, config)
        uncompensated = ed.coherent_beam_power(cube, config)
        gain = coherent[peak] / noncoherent[peak]
        self.assertAlmostEqual(float(gain), 12.0, delta=0.2)
        self.assertLess(float(uncompensated[peak] / noncoherent[peak]), 9.0)

    def test_noise_mean_matches_the_noncoherent_statistic_scale(self) -> None:
        config = make_config()
        cube = dsp.compute_range_doppler_fft(
            synthetic_range_fft(np.random.default_rng(3)), config
        )
        noncoherent = np.mean(np.abs(cube) ** 2, axis=(1, 2))
        coherent = ed.coherent_beam_power(ed.tdm_doppler_compensation(cube), config)
        # Each beam has the per-channel noise power; the maximum over the beam
        # bank is a few times larger but of the same order.
        ratio = float(np.median(coherent) / np.median(noncoherent))
        self.assertTrue(1.5 < ratio < 6.0, ratio)

    def test_beam_bank_stays_inside_the_field_of_view(self) -> None:
        weights, azimuth_u, elevation_u = ed.coherent_beam_steering(8, 60.0)
        self.assertEqual(weights.shape[0], 16)
        self.assertEqual(weights.shape[1], azimuth_u.size)
        self.assertTrue(np.all(np.abs(azimuth_u) <= np.sin(np.deg2rad(60.0))))
        self.assertTrue(np.all(azimuth_u**2 + elevation_u**2 <= 1.0))


class TrackBeforeDetectTests(unittest.TestCase):
    @staticmethod
    def _noise(rng: np.random.Generator) -> np.ndarray:
        return rng.exponential(0.06, size=(DOPPLER_BINS, RANGE_BINS))

    def test_noise_and_single_flash_do_not_declare(self) -> None:
        rng = np.random.default_rng(4)
        tracker = ed.TrackBeforeDetect()
        for index in range(40):
            margins = self._noise(rng)
            if index == 20:
                margins[70, 30] = 5.0  # clipped single-frame clutter flash
            result = tracker.update(margins)
            self.assertFalse(result.detected, index)

    def test_persistent_subthreshold_path_is_declared_at_its_cell(self) -> None:
        rng = np.random.default_rng(5)
        tracker = ed.TrackBeforeDetect()
        declared = []
        for index in range(16):
            margins = self._noise(rng)
            doppler_bin, range_bin = 70 + (index % 2), 30 + index // 4
            margins[doppler_bin, range_bin] = 0.7  # never crosses 1.0
            result = tracker.update(margins)
            if result.detected:
                declared.append(index)
                self.assertEqual((result.doppler_bin, result.range_bin),
                                 (doppler_bin, range_bin))
                self.assertGreaterEqual(result.integrated_score, 0.5)
        self.assertEqual(declared, list(range(tracker.window_frames - 1, 16)))

    def test_unreachable_jumps_are_not_integrated(self) -> None:
        tracker = ed.TrackBeforeDetect()
        for index in range(16):
            margins = np.zeros((DOPPLER_BINS, RANGE_BINS))
            margins[(17 * index) % DOPPLER_BINS, (11 * index) % RANGE_BINS] = 0.9
            self.assertFalse(tracker.update(margins).detected)


class EnhancedPointCloudTests(unittest.TestCase):
    def _run(self, options: ed.EnhancedDetectionOptions, amplitude: float):
        config = make_config()
        rng = np.random.default_rng(6)
        clutter = ed.create_clutter_map(
            options.clutter_reference,
            update_rate=0.02,
            warmup_frames=30,
            minimum_snr_db=3.0,
        )
        tracker = (
            ed.TrackBeforeDetect(options.tbd_window_frames)
            if options.tbd_window_frames
            else None
        )
        range_axis = np.arange(RANGE_BINS, dtype=np.float32) * 0.195

        def step(frame: np.ndarray) -> np.ndarray:
            if options.enabled:
                points, _ = ed.compute_enhanced_point_cloud(
                    frame, range_axis, config, options,
                    clutter_map=clutter, track_before_detect=tracker,
                    max_range_m=12.0,
                )
                return points
            return dsp.compute_point_cloud(
                frame, range_axis, config, clutter_map=clutter, max_range_m=12.0
            )

        noise_points = 0
        for _ in range(30):
            step(synthetic_range_fft(rng))
        for _ in range(12):
            noise_points += len(step(synthetic_range_fft(rng)))
        hits = 0
        target_range = range_axis[40]
        for _ in range(12):
            points = step(
                synthetic_range_fft(
                    rng, amplitude=amplitude, azimuth_u=0.3, elevation_u=0.1
                )
            )
            ranges = np.linalg.norm(points[:, :3], axis=1) if len(points) else []
            hits += bool(np.any(np.abs(np.asarray(ranges) - target_range) < 0.3))
        return hits, noise_points

    def test_mode_4_detects_a_target_that_mode_1_misses(self) -> None:
        mode1_hits, mode1_noise = self._run(ed.EnhancedDetectionOptions(), 0.25)
        mode4_hits, mode4_noise = self._run(ed.MODE4_OPTIONS, 0.25)
        self.assertEqual(mode1_hits, 0)
        self.assertGreaterEqual(mode4_hits, 11)
        self.assertEqual(mode1_noise, 0)
        self.assertLessEqual(mode4_noise, 1)

    def test_enhanced_path_with_mode_1_settings_matches_mode_1(self) -> None:
        config = make_config()
        range_axis = np.arange(RANGE_BINS, dtype=np.float32) * 0.195
        frame = synthetic_range_fft(
            np.random.default_rng(7), amplitude=3.0, azimuth_u=0.25
        )
        expected = dsp.compute_point_cloud(frame, range_axis, config, max_range_m=12.0)
        # A non-default but behaviour-neutral object exercises the new path
        # with the Mode 1 statistic, reference and threshold.
        options = ed.EnhancedDetectionOptions()
        actual, diagnostics = ed.compute_enhanced_point_cloud(
            frame, range_axis, config, options, max_range_m=12.0
        )
        self.assertGreater(len(expected), 0)
        np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-5)
        self.assertEqual(diagnostics.cfar_candidates, len(expected))
        self.assertFalse(diagnostics.tbd_injected)


class IntegrationTests(unittest.TestCase):
    def test_mode_1_capture_command_has_no_enhanced_flags(self) -> None:
        self.assertEqual(run.enhanced_detection_arguments(argparse.Namespace()), [])

    def test_mode_4_flags_are_forwarded(self) -> None:
        args = argparse.Namespace(
            detection_power="coherent-beam",
            clutter_reference="mean",
            cfar_threshold="calibrated",
            tbd_window_frames=8,
        )
        self.assertEqual(
            run.enhanced_detection_arguments(args),
            [
                "--detection-power", "coherent-beam",
                "--clutter-reference", "mean",
                "--cfar-threshold", "calibrated",
                "--tbd-window-frames", "8",
            ],
        )

    def test_sink_uses_mode_1_objects_unless_options_are_enabled(self) -> None:
        plain = DisplayPayloadSink(COMBINED_DISPLAY_MODE, 1, None, SimpleNamespace())
        self.assertIsNone(plain.enhanced_options)
        self.assertIsNone(plain.track_before_detect)
        self.assertIs(type(plain.clutter_map), dsp.AdaptiveClutterMap)

        neutral = DisplayPayloadSink(
            COMBINED_DISPLAY_MODE, 1, None, SimpleNamespace(),
            enhanced_detection_options=ed.EnhancedDetectionOptions(),
        )
        self.assertIsNone(neutral.enhanced_options)

        enhanced = DisplayPayloadSink(
            COMBINED_DISPLAY_MODE, 1, None, SimpleNamespace(),
            enhanced_detection_options=ed.MODE4_OPTIONS,
        )
        self.assertIs(enhanced.enhanced_options, ed.MODE4_OPTIONS)
        self.assertIsInstance(enhanced.clutter_map, ed.MeanClutterMap)
        self.assertEqual(enhanced.track_before_detect.window_frames, 8)

    def test_real_profile_loads_and_mode_4_runs_on_its_dimensions(self) -> None:
        from main.livedatacapture import RadarCaptureConfig

        profile = Path(__file__).resolve().parent.parent / "profiles" / "profile.cfg"
        config = RadarCaptureConfig.from_file(profile)
        self.assertEqual(len(config.rx_channel_compensation), 12)
        self.assertEqual(
            (config.num_loops, config.num_chirps_per_loop, config.num_adc_samples),
            (DOPPLER_BINS, TX_SLOTS, RANGE_BINS),
        )
        rng = np.random.default_rng(8)
        loops = config.num_loops
        chirps = loops * config.num_chirps_per_loop
        shape = (chirps, 4, config.num_adc_samples)
        frame = (
            rng.standard_normal(shape) + 1j * rng.standard_normal(shape)
        ).astype(np.complex64)
        points, diagnostics = ed.compute_enhanced_point_cloud(
            frame, dsp.range_axis_m(config), config, ed.MODE4_OPTIONS
        )
        self.assertEqual(points.shape[1], 4)
        self.assertFalse(diagnostics.tbd_injected)


if __name__ == "__main__":
    unittest.main()
