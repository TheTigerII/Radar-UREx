import queue
import socket
import threading
import unittest
from itertools import chain, repeat
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

from main.livedatacapture import (
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
    _run_frame_processor,
)
from main.dsp import (
    compute_range_doppler_fft,
    frame_bytes_to_radar_cube,
)
from main.pmm import PmmConfig


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

    def test_frame_with_extra_bytes_is_rejected(self) -> None:
        config = RadarCaptureConfig.from_dimensions(
            num_adc_samples=2,
            num_rx_channels=2,
            num_chirps_per_frame=1,
        )

        with self.assertRaisesRegex(ValueError, "bytes; expected"):
            frame_bytes_to_radar_cube(self._frame_bytes() + b"extra", config)


class FrameProcessorFailureTests(unittest.TestCase):
    def test_processing_failure_is_raised_after_ready_status(self) -> None:
        config = RadarCaptureConfig.from_file(
            Path(__file__).resolve().parent.parent
            / "profiles"
            / "profile-mini4-20m.cfg"
        )
        frame_queue = queue.Queue()
        frame_queue.put(CapturedFrame(b"", 0, 1.0))
        log_queue = queue.Queue()
        status_queue = queue.Queue(maxsize=1)
        counter = SimpleNamespace(value=0, get_lock=lambda: threading.Lock())

        with (
            patch("main.livedatacapture.signal.signal"),
            self.assertRaisesRegex(ValueError, "bytes; expected"),
        ):
            _run_frame_processor(
                config=config,
                pmm_config=PmmConfig(background_calibration_seconds=0.1),
                frame_queue=frame_queue,
                log_queue=log_queue,
                processed_frames_counter=counter,
                display_payload_queue=None,
                display_skipped_counter=None,
                raw_output=None,
                raw_metadata=None,
                processed_output=None,
                display_mode="none",
                display_update_every=1,
                startup_status_queue=status_queue,
            )

        self.assertEqual(status_queue.get_nowait()["state"], "ready")
        messages = []
        while not log_queue.empty():
            messages.append(log_queue.get_nowait())
        self.assertTrue(
            any("Frame processor failed" in message for message in messages),
            messages,
        )


class DopplerProcessingTests(unittest.TestCase):
    def test_static_signal_is_retained_at_zero_doppler_for_paper_pipeline(self) -> None:
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

        center_bin = doppler_cube.shape[0] // 2
        self.assertGreater(float(np.abs(doppler_cube[center_bin]).min()), 0.0)
        np.testing.assert_array_equal(
            np.argmax(np.abs(doppler_cube), axis=0),
            np.full(doppler_cube.shape[1:], center_bin),
        )


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
