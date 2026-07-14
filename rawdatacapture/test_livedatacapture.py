import unittest

from rawdatacapture.livedatacapture import (
    CaptureStats,
    DCA1000PacketHeader,
    FrameBuffer,
    RadarCaptureConfig,
    _point_cloud_range_limit_m,
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

if __name__ == "__main__":
    unittest.main()
