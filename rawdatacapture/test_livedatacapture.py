import math
import unittest
from unittest.mock import Mock

import numpy as np

from rawdatacapture.dsp import _point_is_within_fov

from rawdatacapture.livedatacapture import (
    CaptureStats,
    DEFAULT_MAX_RANGE_M,
    DEFAULT_POINT_CLOUD_FOV_DEG,
    DCA1000PacketHeader,
    FrameBuffer,
    RadarCaptureConfig,
    _draw_range_doppler,
    _draw_range_profile,
    _point_cloud_range_limit_m,
    _set_point_cloud_axes,
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


class PointCloudBoundsTests(unittest.TestCase):
    def test_display_defaults_are_ten_meters_and_sixty_degrees(self) -> None:
        self.assertEqual(DEFAULT_MAX_RANGE_M, 10.0)
        self.assertEqual(DEFAULT_POINT_CLOUD_FOV_DEG, 60.0)

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
