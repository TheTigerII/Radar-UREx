import math
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

from rawdatacapture.dsp import _point_is_within_fov

from rawdatacapture.livedatacapture import (
    CaptureStats,
    CapturedFrame,
    DEFAULT_CLUSTER_EPS_M,
    DEFAULT_CLUSTER_MIN_SAMPLES,
    DEFAULT_MAX_RANGE_M,
    DEFAULT_POINT_CLOUD_FOV_DEG,
    DCA1000PacketHeader,
    FrameBuffer,
    RadarCaptureConfig,
    _capture_error_counts,
    _draw_point_cloud,
    _draw_range_doppler,
    _draw_range_profile,
    _point_cloud_range_limit_m,
    _set_point_cloud_axes,
    process_complete_frame,
)


class FrameBufferTests(unittest.TestCase):
    def test_packet_gap_within_one_frame_is_marked_invalid(self) -> None:
        stats = CaptureStats()
        buffer = FrameBuffer(bytes_per_frame=16, stats=stats)

        frames = buffer.add_payload(DCA1000PacketHeader(1, 0), b"1234", 1.0)
        frames += buffer.add_payload(DCA1000PacketHeader(2, 8), b"abcdefgh", 2.0)

        self.assertEqual(len(frames), 1)
        self.assertEqual(frames[0].gap_bytes, 4)
        self.assertEqual(stats.stream_resyncs, 0)

    def test_gap_larger_than_one_frame_resynchronizes_without_padding(self) -> None:
        stats = CaptureStats()
        buffer = FrameBuffer(bytes_per_frame=16, stats=stats)

        buffer.add_payload(DCA1000PacketHeader(1, 0), b"partial", 1.0)
        frames = buffer.add_payload(
            DCA1000PacketHeader(2, 1_000_000_000),
            b"0123456789abcdef",
            2.0,
        )

        self.assertEqual(len(frames), 1)
        self.assertEqual(frames[0].data, b"0123456789abcdef")
        self.assertTrue(frames[0].is_valid)
        self.assertEqual(stats.stream_resyncs, 1)
        self.assertEqual(stats.byte_gap_bytes, 0)
        self.assertEqual(len(buffer.buffer), 0)


class FrameDiagnosticsTests(unittest.TestCase):
    def test_valid_frame_does_not_emit_routine_diagnostics(self) -> None:
        frame = CapturedFrame(data=b"valid", gap_bytes=0, first_byte_at_s=1.0)
        range_axis_m = np.asarray((0.0, 0.25), dtype=np.float32)
        config = SimpleNamespace(
            bytes_per_frame=len(frame.data),
            range_axis_m=Mock(return_value=range_axis_m),
        )
        display = Mock()
        raw_writer = Mock()
        emit_func = Mock()
        radar_cube = np.ones((1, 1, 2), dtype=np.complex64)
        range_fft = np.ones((1, 1, 2), dtype=np.complex64)

        with (
            patch(
                "rawdatacapture.livedatacapture.frame_bytes_to_radar_cube",
                return_value=radar_cube,
            ),
            patch(
                "rawdatacapture.livedatacapture.compute_range_fft",
                return_value=range_fft,
            ),
            patch(
                "rawdatacapture.livedatacapture.compute_range_profile"
            ) as range_profile,
        ):
            process_complete_frame(
                frame,
                config,
                display,
                raw_writer,
                emit_func,
            )

        emit_func.assert_not_called()
        range_profile.assert_not_called()
        display.update.assert_called_once_with(range_fft, range_axis_m)

    def test_error_snapshot_ignores_success_counters(self) -> None:
        initial = CaptureStats(packets_received=100, frames_emitted=5)
        later_success = CaptureStats(packets_received=200, frames_emitted=10)

        self.assertEqual(
            _capture_error_counts(initial),
            _capture_error_counts(later_success),
        )

        later_success.lost_packets = 1
        self.assertNotEqual(
            _capture_error_counts(initial),
            _capture_error_counts(later_success),
        )


class PointCloudBoundsTests(unittest.TestCase):
    def test_display_defaults_are_ten_meters_and_sixty_degrees(self) -> None:
        self.assertEqual(DEFAULT_MAX_RANGE_M, 10.0)
        self.assertEqual(DEFAULT_POINT_CLOUD_FOV_DEG, 60.0)
        self.assertEqual(DEFAULT_CLUSTER_EPS_M, 0.5)
        self.assertEqual(DEFAULT_CLUSTER_MIN_SAMPLES, 2)

    def test_point_cloud_axes_match_ten_meter_sixty_degree_fov(self) -> None:
        axis = Mock()

        _set_point_cloud_axes(axis, 10.0, 60.0)

        cross_range_m = 10.0 * math.sin(math.radians(60.0))
        axis.set_xlim.assert_called_once_with(-cross_range_m, cross_range_m)
        axis.set_ylim.assert_called_once_with(0.0, 10.0)
        axis.set_zlim.assert_called_once_with(-cross_range_m, cross_range_m)
        axis.set_box_aspect.assert_called_once_with(
            (2.0 * cross_range_m, 10.0, 2.0 * cross_range_m)
        )

    def test_direction_cosine_gate_rejects_points_beyond_sixty_degrees(self) -> None:
        range_m = 10.0
        inside_angle_deg = 60.0
        outside_angle_deg = 61.0

        self.assertTrue(
            _point_is_within_fov(
                range_m * math.sin(math.radians(inside_angle_deg)),
                range_m * math.cos(math.radians(inside_angle_deg)),
                0.0,
                azimuth_fov_deg=60.0,
                elevation_fov_deg=60.0,
            )
        )
        self.assertFalse(
            _point_is_within_fov(
                range_m * math.sin(math.radians(outside_angle_deg)),
                range_m * math.cos(math.radians(outside_angle_deg)),
                0.0,
                azimuth_fov_deg=60.0,
                elevation_fov_deg=60.0,
            )
        )

    def test_range_limit_includes_every_range_bin(self) -> None:
        config = RadarCaptureConfig.from_dimensions(
            num_adc_samples=128,
            num_rx_channels=4,
            num_chirps_per_frame=192,
            sample_rate_ksps=5000.0,
            frequency_slope_mhz_per_us=70.0,
        )

        limit_m = _point_cloud_range_limit_m(config)
        range_axis_m = config.range_axis_m()

        self.assertIsNotNone(limit_m)
        self.assertIsNotNone(range_axis_m)
        assert limit_m is not None
        assert range_axis_m is not None
        self.assertGreater(limit_m, float(range_axis_m[-1]))

    def test_draw_updates_points_and_cluster_centers(self) -> None:
        axis = Mock()
        scatter = Mock()
        cluster_scatter = Mock()
        points = np.asarray(((1.0, 2.0, 3.0, 50.0),), dtype=np.float32)
        clusters = np.asarray(((1.0, 2.0, 3.0, 2.0),), dtype=np.float32)

        _draw_point_cloud(
            axis,
            scatter,
            cluster_scatter,
            points,
            clusters,
            10.0,
            60.0,
        )

        for actual, expected in zip(scatter._offsets3d, points[:, :3].T):
            np.testing.assert_array_equal(actual, expected)
        for actual, expected in zip(cluster_scatter._offsets3d, clusters[:, :3].T):
            np.testing.assert_array_equal(actual, expected)
        cluster_scatter.set_sizes.assert_called_once()


class RangeDisplayBoundsTests(unittest.TestCase):
    def test_range_profile_uses_ten_meter_default_limit(self) -> None:
        axis = Mock()
        line = Mock()
        range_axis_m = np.linspace(0.0, 12.0, 5, dtype=np.float32)

        _draw_range_profile(
            axis,
            line,
            range_axis_m,
            np.ones(5, dtype=np.float32),
            DEFAULT_MAX_RANGE_M,
        )

        axis.set_xlim.assert_called_once_with(0.0, 10.0)

    def test_range_doppler_uses_ten_meter_default_limit(self) -> None:
        axis = Mock()
        image = Mock()
        range_axis_m = np.linspace(0.0, 12.0, 5, dtype=np.float32)

        _draw_range_doppler(
            axis,
            image,
            range_axis_m,
            np.ones((4, 5), dtype=np.float32),
            DEFAULT_MAX_RANGE_M,
        )

        axis.set_xlim.assert_called_once_with(0.0, 10.0)

if __name__ == "__main__":
    unittest.main()
