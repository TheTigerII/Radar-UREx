import queue
import socket
import unittest
from itertools import chain, repeat
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

from rawdatacapture.livedatacapture import (
    CapturedFrame,
    CaptureStartupError,
    CaptureStats,
    DCA1000PacketHeader,
    FrameBuffer,
    LiveDisplay,
    RadarCaptureConfig,
    SequenceTracker,
    UdpPacketReceiver,
    _format_capture_summary,
    _put_latest_queue_payload,
)
from rawdatacapture.dsp import (
    compute_range_doppler_fft,
    frame_bytes_to_radar_cube,
)


class DcaHeaderTests(unittest.TestCase):
    def test_parses_sequence_and_48_bit_byte_count(self) -> None:
        packet = (
            (17).to_bytes(4, "little")
            + (0x010203040506).to_bytes(6, "little")
            + b"payload"
        )

        header = DCA1000PacketHeader.parse(packet)

        self.assertEqual(header.sequence_number, 17)
        self.assertEqual(header.byte_count, 0x010203040506)

    def test_rejects_short_header(self) -> None:
        with self.assertRaises(ValueError):
            DCA1000PacketHeader.parse(bytes(9))


class SequenceTrackerTests(unittest.TestCase):
    def test_counts_loss_duplicates_and_out_of_order_packets(self) -> None:
        stats = CaptureStats()
        tracker = SequenceTracker(stats)

        tracker.observe(10)
        tracker.observe(12)
        tracker.observe(12)
        tracker.observe(11)

        self.assertEqual(stats.lost_packets, 1)
        self.assertEqual(stats.duplicate_packets, 1)
        self.assertEqual(stats.out_of_order_packets, 1)


class FrameBufferTests(unittest.TestCase):
    def test_complete_frame_is_emitted_without_copying_input_contract(self) -> None:
        stats = CaptureStats()
        buffer = FrameBuffer(8, stats)

        frames = buffer.add_payload(
            DCA1000PacketHeader(1, 100),
            memoryview(b"abcdefgh"),
            1.25,
        )

        self.assertEqual(frames, [CapturedFrame(b"abcdefgh", 0, 1.25)])
        self.assertEqual(stats.frames_emitted, 1)

    def test_packet_gap_marks_affected_frame_invalid(self) -> None:
        stats = CaptureStats()
        buffer = FrameBuffer(8, stats)
        self.assertEqual(
            buffer.add_payload(
                DCA1000PacketHeader(1, 0),
                b"abcd",
                1.0,
            ),
            [],
        )

        frames = buffer.add_payload(
            DCA1000PacketHeader(2, 6),
            b"ef",
            1.1,
        )

        self.assertEqual(len(frames), 1)
        self.assertEqual(frames[0].gap_bytes, 2)
        self.assertFalse(frames[0].is_valid)
        self.assertEqual(stats.invalid_frames, 1)
        self.assertEqual(stats.byte_gap_bytes, 2)

    def test_large_gap_resynchronizes_without_allocating_gap(self) -> None:
        stats = CaptureStats()
        buffer = FrameBuffer(8, stats)
        buffer.add_payload(DCA1000PacketHeader(1, 0), b"ab", 1.0)

        frames = buffer.add_payload(
            DCA1000PacketHeader(2, 100),
            b"abcdefgh",
            2.0,
        )

        self.assertEqual(frames[0].data, b"abcdefgh")
        self.assertEqual(frames[0].gap_bytes, 0)
        self.assertEqual(stats.stream_resyncs, 1)

    def test_overlap_is_trimmed(self) -> None:
        stats = CaptureStats()
        buffer = FrameBuffer(8, stats)
        buffer.add_payload(DCA1000PacketHeader(1, 0), b"abcd", 1.0)

        frames = buffer.add_payload(
            DCA1000PacketHeader(2, 2),
            b"cdefgh",
            1.1,
        )

        self.assertEqual(frames[0].data, b"abcdefgh")
        self.assertEqual(stats.byte_overlap_bytes, 2)


class QueueTests(unittest.TestCase):
    def test_latest_payload_replaces_stale_payload(self) -> None:
        payload_queue = queue.Queue(maxsize=1)
        payload_queue.put("old")

        skipped = _put_latest_queue_payload(payload_queue, "new")

        self.assertEqual(skipped, 1)
        self.assertEqual(payload_queue.get_nowait(), "new")


class LiveDisplayStartupTests(unittest.TestCase):
    def _context(self, startup_result):
        payload_queue = Mock()
        status_queue = Mock()
        if isinstance(startup_result, BaseException):
            status_queue.get.side_effect = startup_result
        else:
            status_queue.get.return_value = startup_result
        process = Mock()
        process.is_alive.return_value = True
        context = Mock()
        context.Queue.side_effect = (payload_queue, status_queue)
        context.Process.return_value = process
        return context, process

    def test_timeout_terminates_display_process(self) -> None:
        context, process = self._context(queue.Empty())

        with self.assertRaisesRegex(
            CaptureStartupError,
            "Display did not become ready",
        ):
            LiveDisplay("range", 20.0, 60.0, context)

        process.terminate.assert_called_once_with()

    def test_interrupt_terminates_display_process(self) -> None:
        context, process = self._context(KeyboardInterrupt())

        with self.assertRaises(KeyboardInterrupt):
            LiveDisplay("range", 20.0, 60.0, context)

        process.terminate.assert_called_once_with()

    def test_failed_startup_terminates_display_process(self) -> None:
        context, process = self._context(
            {"state": "failed", "message": "Display failed: import error"}
        )

        with self.assertRaisesRegex(CaptureStartupError, "import error"):
            LiveDisplay("range", 20.0, 60.0, context)

        process.terminate.assert_called_once_with()


class AdcLayoutTests(unittest.TestCase):
    @staticmethod
    def _frame_bytes() -> bytes:
        values = np.asarray(
            (
                (1, 2, 10, 20),
                (3, 4, 30, 40),
            ),
            dtype="<i2",
        )
        return values.tobytes()

    def test_non_interleaved_profile_keeps_each_rx_contiguous(self) -> None:
        config = RadarCaptureConfig.from_dimensions(
            num_adc_samples=2,
            num_rx_channels=2,
            num_chirps_per_frame=1,
            channel_interleave=False,
        )

        cube = frame_bytes_to_radar_cube(self._frame_bytes(), config)

        np.testing.assert_array_equal(
            cube,
            np.asarray(
                (((1 + 10j, 2 + 20j), (3 + 30j, 4 + 40j)),),
                dtype=np.complex64,
            ),
        )

    def test_interleaved_profile_groups_samples_before_receivers(self) -> None:
        config = RadarCaptureConfig.from_dimensions(
            num_adc_samples=2,
            num_rx_channels=2,
            num_chirps_per_frame=1,
            channel_interleave=True,
        )

        cube = frame_bytes_to_radar_cube(self._frame_bytes(), config)

        np.testing.assert_array_equal(
            cube,
            np.asarray(
                (((1 + 10j, 3 + 30j), (2 + 20j, 4 + 40j)),),
                dtype=np.complex64,
            ),
        )


class DopplerProcessingTests(unittest.TestCase):
    def test_slow_time_mean_subtraction_removes_static_clutter(self) -> None:
        loops = 8
        transmitters = 2
        config = SimpleNamespace(
            num_loops=loops,
            num_chirps_per_loop=transmitters,
            num_rx_channels=1,
            num_adc_samples=3,
        )
        static_range_fft = np.full(
            (loops * transmitters, 1, 3),
            100.0 + 25.0j,
            dtype=np.complex64,
        )

        doppler_cube = compute_range_doppler_fft(
            static_range_fft,
            config,
            fft_size=8,
        )

        np.testing.assert_allclose(doppler_cube, 0.0, atol=1e-5)


class UdpPacketReceiverTests(unittest.TestCase):
    def test_receiver_preserves_datagram_order(self) -> None:
        fake_socket = Mock()
        fake_socket.recv.side_effect = chain(
            (b"first", b"second"),
            repeat(socket.timeout()),
        )
        packet_queue = queue.Queue(maxsize=2)
        stats = CaptureStats()
        receiver = UdpPacketReceiver(
            fake_socket,
            packet_queue,
            65535,
            stats,
        )

        with patch("time.perf_counter", side_effect=(1.0, 2.0)):
            receiver.start()
            while packet_queue.qsize() < 2:
                pass
            receiver.stop()
            receiver.join(timeout=1.0)

        self.assertEqual(packet_queue.get_nowait(), (b"first", 1.0))
        self.assertEqual(packet_queue.get_nowait(), (b"second", 2.0))

    def test_receiver_counts_bounded_queue_drops(self) -> None:
        fake_socket = Mock()
        fake_socket.recv.side_effect = chain(
            (b"drop",),
            repeat(socket.timeout()),
        )
        packet_queue = queue.Queue(maxsize=1)
        packet_queue.put((b"existing", 0.0))
        stats = CaptureStats()
        receiver = UdpPacketReceiver(
            fake_socket,
            packet_queue,
            65535,
            stats,
        )

        receiver.start()
        while stats.receiver_queue_drops == 0:
            pass
        receiver.stop()
        receiver.join(timeout=1.0)

        self.assertEqual(stats.receiver_queue_drops, 1)


class DiagnosticsTests(unittest.TestCase):
    def test_summary_contains_loss_and_queue_counters(self) -> None:
        stats = CaptureStats(
            packets_received=10,
            frames_emitted=2,
            invalid_frames=1,
            lost_packets=3,
            receiver_queue_drops=4,
            processing_frames_dropped=5,
        )

        summary = _format_capture_summary(stats)

        self.assertIn("lost_packets=3", summary)
        self.assertIn("receiver_queue_drops=4", summary)
        self.assertIn("processing_drops=5", summary)


if __name__ == "__main__":
    unittest.main()
