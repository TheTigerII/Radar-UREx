import unittest

from rawdatacapture.livedatacapture import (
    CaptureStats,
    DCA1000PacketHeader,
    FrameBuffer,
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


if __name__ == "__main__":
    unittest.main()
