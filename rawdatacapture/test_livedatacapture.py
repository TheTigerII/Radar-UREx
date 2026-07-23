import json
import math
import queue
import socket
import tempfile
import threading
import unittest
from collections import deque
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

from rawdatacapture.dsp import _point_is_within_fov

from rawdatacapture.livedatacapture import (
    CaptureStats,
    CapturedFrame,
    COMBINED_DISPLAY_MODE,
    DEFAULT_CLUSTER_EPS_M,
    DEFAULT_CLUSTER_MIN_SAMPLES,
    DEFAULT_MAX_RANGE_M,
    DEFAULT_POINT_CLOUD_FOV_DEG,
    MAGNITUDE_COLORMAP,
    MICRO_DOPPLER_HISTORY_UPDATES,
    MICRO_DOPPLER_HOP_LOOPS,
    MICRO_DOPPLER_RANGE_HALF_WIDTH_BINS,
    MICRO_DOPPLER_WINDOW_LOOPS,
    POINT_CLOUD_MAGNITUDE_DB_MAX,
    POINT_CLOUD_MAGNITUDE_DB_MIN,
    DCA1000PacketHeader,
    DisplayPayloadSink,
    FrameBuffer,
    ProcessedOutputWriter,
    RadarCaptureConfig,
    SingleTargetTracker,
    TargetTrack,
    UdpPacketReceiver,
    _capture_error_counts,
    _draw_point_cloud,
    _draw_micro_doppler,
    _point_cloud_track_candidates,
    _draw_range_doppler,
    _draw_range_profile,
    _point_cloud_range_limit_m,
    _format_capture_summary,
    _put_latest_queue_payload,
    _record_event_rate,
    _report_new_error_stats,
    _request_processor_stop,
    _set_rate_indicator,
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

    def test_memoryview_payload_is_assembled_without_a_slice_copy(self) -> None:
        stats = CaptureStats()
        buffer = FrameBuffer(bytes_per_frame=8, stats=stats)
        packet = memoryview(b"abcdefgh")

        frames = buffer.add_payload(DCA1000PacketHeader(1, 0), packet, 1.0)

        self.assertEqual(len(frames), 1)
        self.assertEqual(frames[0].data, b"abcdefgh")


class _FakeDatagramSocket:
    def __init__(self, packets: list[bytes]) -> None:
        self.packets = deque(packets)
        self.exhausted = threading.Event()

    def recv(self, _buffer_size: int) -> bytes:
        try:
            return self.packets.popleft()
        except IndexError:
            self.exhausted.set()
            raise socket.timeout


class UdpPacketReceiverTests(unittest.TestCase):
    def test_receiver_preserves_datagram_order(self) -> None:
        sock = _FakeDatagramSocket([b"one", b"two", b"three"])
        packet_queue = queue.Queue(maxsize=3)
        stats = CaptureStats()
        receiver = UdpPacketReceiver(sock, packet_queue, 65535, stats)

        receiver.start()
        self.assertTrue(sock.exhausted.wait(timeout=1.0))
        receiver.stop()
        receiver.join(timeout=1.0)

        self.assertEqual(
            [packet_queue.get_nowait()[0] for _ in range(3)],
            [b"one", b"two", b"three"],
        )
        self.assertEqual(stats.receiver_queue_drops, 0)

    def test_receiver_counts_bounded_queue_drops(self) -> None:
        sock = _FakeDatagramSocket([b"dropped"])
        packet_queue = queue.Queue(maxsize=1)
        packet_queue.put_nowait((b"existing", 0.0))
        stats = CaptureStats()
        receiver = UdpPacketReceiver(sock, packet_queue, 65535, stats)

        receiver.start()
        self.assertTrue(sock.exhausted.wait(timeout=1.0))
        receiver.stop()
        receiver.join(timeout=1.0)

        self.assertEqual(stats.receiver_queue_drops, 1)
        self.assertEqual(packet_queue.get_nowait()[0], b"existing")


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

    def test_errors_are_reported_immediately_on_each_counter_change(self) -> None:
        stats = CaptureStats()
        reported = _capture_error_counts(stats)
        emit_func = Mock()

        stats.lost_packets = 2
        reported = _report_new_error_stats(stats, reported, emit_func)
        _report_new_error_stats(stats, reported, emit_func)
        stats.receiver_queue_drops = 1
        _report_new_error_stats(stats, reported, emit_func)

        self.assertEqual(emit_func.call_count, 2)
        self.assertIn("lost_packets=2", emit_func.call_args_list[0].args[0])
        self.assertIn(
            "receiver_queue_drops=1",
            emit_func.call_args_list[1].args[0],
        )

    def test_capture_summary_separates_valid_and_queued_frames(self) -> None:
        stats = CaptureStats(
            frames_emitted=100,
            invalid_frames=3,
            processing_frames_dropped=2,
        )

        summary = _format_capture_summary(stats)

        self.assertIn("valid_frames=97", summary)
        self.assertIn("queued_frames=95", summary)

    def test_graceful_processor_stop_does_not_discard_a_queued_frame(self) -> None:
        frame_queue = queue.Queue(maxsize=1)
        original_frame = object()
        consumed = []
        frame_queue.put_nowait(original_frame)

        consumer = threading.Thread(
            target=lambda: consumed.append(frame_queue.get(timeout=1.0))
        )
        consumer.start()

        self.assertTrue(_request_processor_stop(frame_queue, timeout_seconds=1.0))
        consumer.join(timeout=1.0)

        self.assertEqual(consumed, [original_frame])
        self.assertIsNone(frame_queue.get_nowait())

    def test_latest_payload_replacement_is_counted(self) -> None:
        payload_queue = queue.Queue(maxsize=1)
        payload_queue.put_nowait("old")

        skipped = _put_latest_queue_payload(payload_queue, "new")

        self.assertEqual(skipped, 1)
        self.assertEqual(payload_queue.get_nowait(), "new")


class ProcessedOutputWriterTests(unittest.TestCase):
    def test_writes_metadata_point_cloud_and_micro_doppler_jsonl(self) -> None:
        config = RadarCaptureConfig.from_dimensions(
            num_adc_samples=64,
            num_rx_channels=4,
            num_chirps_per_frame=384,
            num_loops=128,
            num_chirps_per_loop=3,
            sample_rate_ksps=5000.0,
            frequency_slope_mhz_per_us=60.0,
        )
        track = TargetTrack(
            position_m=(0.0, 2.0, 0.0),
            velocity_m_per_update=(0.1, 0.0, 0.0),
            age_updates=3,
            hits=3,
            missed_updates=0,
            confirmed=True,
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "processed.jsonl"
            writer = ProcessedOutputWriter(output_path, config, 1)
            writer.write_update(
                frame_index=7,
                points=np.asarray(((1.0, 2.0, 3.0, 50.0),), dtype=np.float32),
                clusters=np.asarray(((1.0, 2.0, 3.0, 1.0),), dtype=np.float32),
                target_track=track,
                micro_doppler_db=np.asarray((10.0, 20.0), dtype=np.float32),
                selected_range_m=2.0,
            )
            writer.close()

            records = [json.loads(line) for line in output_path.read_text().splitlines()]

        self.assertEqual(records[0]["record_type"], "metadata")
        self.assertEqual(records[0]["radar_config"]["num_loops"], 128)
        self.assertEqual(
            records[0]["micro_doppler_processing"]["window_loops"],
            64,
        )
        self.assertEqual(records[0]["micro_doppler_processing"]["hop_loops"], 32)
        self.assertEqual(records[1]["record_type"], "update")
        self.assertEqual(records[1]["processed_frame_index"], 7)
        self.assertEqual(records[1]["points"], [[1.0, 2.0, 3.0, 50.0]])
        self.assertEqual(records[1]["micro_doppler_db"], [10.0, 20.0])
        self.assertEqual(records[1]["micro_doppler_windows_db"], [[10.0, 20.0]])
        self.assertTrue(records[1]["target_track"]["confirmed"])


class PointCloudBoundsTests(unittest.TestCase):
    def test_display_defaults_are_ten_meters_and_sixty_degrees(self) -> None:
        self.assertEqual(DEFAULT_MAX_RANGE_M, 10.0)
        self.assertEqual(DEFAULT_POINT_CLOUD_FOV_DEG, 60.0)
        self.assertEqual(DEFAULT_CLUSTER_EPS_M, 0.4)
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
            update_rate_hz=29.8,
        )

        for actual, expected in zip(scatter._offsets3d, points[:, :3].T):
            np.testing.assert_array_equal(actual, expected)
        for actual, expected in zip(cluster_scatter._offsets3d, clusters[:, :3].T):
            np.testing.assert_array_equal(actual, expected)
        cluster_scatter.set_sizes.assert_called_once()
        axis.set_title.assert_called_once_with(
            "Live 3D Point Cloud (±60° FOV, 10 m, 29.8 Hz)"
        )

    def test_draw_updates_tracked_target_marker(self) -> None:
        axis = Mock()
        scatter = Mock()
        cluster_scatter = Mock()
        target_scatter = Mock()
        track = TargetTrack(
            position_m=(1.0, 2.0, 3.0),
            velocity_m_per_update=(0.0, 0.0, 0.0),
            age_updates=4,
            hits=3,
            missed_updates=0,
            confirmed=True,
        )

        _draw_point_cloud(
            axis,
            scatter,
            cluster_scatter,
            np.empty((0, 4), dtype=np.float32),
            np.empty((0, 4), dtype=np.float32),
            10.0,
            60.0,
            target_scatter,
            track,
        )

        np.testing.assert_array_equal(target_scatter._offsets3d[0], (1.0,))
        np.testing.assert_array_equal(target_scatter._offsets3d[1], (2.0,))
        np.testing.assert_array_equal(target_scatter._offsets3d[2], (3.0,))
        target_scatter.set_alpha.assert_called_once_with(1.0)

    def test_blitted_point_cloud_draw_does_not_mutate_static_axes(self) -> None:
        axis = Mock()

        _draw_point_cloud(
            axis,
            Mock(),
            Mock(),
            np.empty((0, 4), dtype=np.float32),
            np.empty((0, 4), dtype=np.float32),
            10.0,
            60.0,
            update_static_artists=False,
        )

        axis.set_xlim.assert_not_called()
        axis.set_ylim.assert_not_called()
        axis.set_zlim.assert_not_called()
        axis.set_title.assert_not_called()


class SingleTargetTrackerTests(unittest.TestCase):
    @staticmethod
    def _candidates(*rows: tuple[float, float, float, float]) -> np.ndarray:
        return np.asarray(rows, dtype=np.float32).reshape((-1, 4))

    def test_acquires_strongest_then_follows_nearest_candidate(self) -> None:
        tracker = SingleTargetTracker(
            association_distance_m=0.5,
            confirmation_hits=2,
            position_gain=1.0,
            velocity_gain=0.0,
        )

        first = tracker.update(
            self._candidates((0.0, 2.0, 0.0, 80.0), (3.0, 3.0, 0.0, 60.0))
        )
        second = tracker.update(
            self._candidates((0.1, 2.0, 0.0, 50.0), (3.0, 3.0, 0.0, 120.0))
        )

        self.assertIsNotNone(first)
        self.assertIsNotNone(second)
        assert first is not None
        assert second is not None
        self.assertFalse(first.confirmed)
        self.assertTrue(second.confirmed)
        np.testing.assert_allclose(second.position_m, (0.1, 2.0, 0.0))

    def test_coasts_through_miss_and_reassociates(self) -> None:
        tracker = SingleTargetTracker(
            association_distance_m=0.5,
            max_missed_updates=2,
            confirmation_hits=1,
            position_gain=1.0,
            velocity_gain=1.0,
        )
        tracker.update(self._candidates((0.0, 2.0, 0.0, 80.0)))
        moving = tracker.update(self._candidates((0.1, 2.0, 0.0, 80.0)))
        predicted = tracker.update(self._candidates())
        recovered = tracker.update(self._candidates((0.3, 2.0, 0.0, 70.0)))

        assert moving is not None
        assert predicted is not None
        assert recovered is not None
        self.assertTrue(predicted.is_predicted)
        np.testing.assert_allclose(predicted.position_m, (0.2, 2.0, 0.0))
        self.assertFalse(recovered.is_predicted)
        np.testing.assert_allclose(recovered.position_m, (0.3, 2.0, 0.0))

    def test_drops_track_after_missed_update_limit(self) -> None:
        tracker = SingleTargetTracker(
            max_missed_updates=1,
            confirmation_hits=1,
        )
        tracker.update(self._candidates((0.0, 2.0, 0.0, 80.0)))

        self.assertIsNotNone(tracker.update(self._candidates()))
        self.assertIsNone(tracker.update(self._candidates()))

    def test_cluster_candidates_use_assigned_point_magnitude(self) -> None:
        points = self._candidates(
            (0.0, 2.0, 0.0, 50.0),
            (0.1, 2.0, 0.0, 70.0),
            (3.0, 4.0, 0.0, 90.0),
        )
        clusters = np.asarray(
            ((0.05, 2.0, 0.0, 2.0), (3.0, 4.0, 0.0, 1.0)),
            dtype=np.float32,
        )

        candidates = _point_cloud_track_candidates(points, clusters)

        np.testing.assert_allclose(candidates[:, :3], clusters[:, :3])
        np.testing.assert_allclose(candidates[:, 3], (70.0, 90.0))


class TrackedDisplayPayloadTests(unittest.TestCase):
    def test_combined_display_uses_tracked_range_for_micro_doppler(self) -> None:
        payload_queue = queue.Queue(maxsize=1)
        config = SimpleNamespace()
        sink = DisplayPayloadSink(
            COMBINED_DISPLAY_MODE,
            1,
            payload_queue,
            config,
        )
        points = np.asarray(((0.0, 2.0, 0.0, 80.0),), dtype=np.float32)
        clusters = np.asarray(((0.0, 2.0, 0.0, 1.0),), dtype=np.float32)
        doppler_cube = np.zeros((4, 1, 1, 4), dtype=np.complex64)

        with (
            patch(
                "rawdatacapture.livedatacapture.compute_range_doppler_fft",
                return_value=doppler_cube,
            ),
            patch(
                "rawdatacapture.livedatacapture.compute_point_cloud",
                return_value=points,
            ),
            patch(
                "rawdatacapture.livedatacapture.cluster_point_cloud",
                return_value=clusters,
            ),
            patch(
                "rawdatacapture.livedatacapture.compute_micro_doppler_spectrum",
                return_value=(np.ones(4, dtype=np.float32), 2.0),
            ) as micro_doppler,
            patch(
                "rawdatacapture.livedatacapture."
                "compute_per_tx_micro_doppler_spectrogram",
                return_value=np.arange(12, dtype=np.float32).reshape(4, 3),
            ) as short_time_micro_doppler,
        ):
            sink.update(np.empty((0,), dtype=np.complex64), np.arange(4))

        micro_doppler.assert_called_once()
        self.assertAlmostEqual(
            micro_doppler.call_args.kwargs["target_range_m"],
            2.0,
        )
        short_time_micro_doppler.assert_called_once()
        np.testing.assert_array_equal(
            sink.latest_micro_doppler_db,
            np.asarray((2.0, 5.0, 8.0, 11.0)),
        )
        self.assertEqual(len(sink.micro_doppler_history), 3)
        payload = payload_queue.get_nowait()
        self.assertEqual(len(payload), 5)
        self.assertIsNotNone(payload[2])

    def test_processed_writer_runs_without_a_display_queue(self) -> None:
        writer = Mock(spec=ProcessedOutputWriter)
        writer.enabled = True
        sink = DisplayPayloadSink(
            "none",
            1,
            None,
            SimpleNamespace(),
            processed_writer=writer,
        )
        points = np.asarray(((0.0, 2.0, 0.0, 80.0),), dtype=np.float32)
        clusters = np.asarray(((0.0, 2.0, 0.0, 1.0),), dtype=np.float32)
        spectrum = np.arange(4, dtype=np.float32)

        with patch.object(
            sink,
            "_compute_combined_payload",
            return_value=(points, clusters, None, spectrum[:, None], 2.0),
        ):
            sink.latest_micro_doppler_db = spectrum
            sink.update(np.empty((0,), dtype=np.complex64), np.arange(4))

        writer.write_update.assert_called_once()
        np.testing.assert_array_equal(
            writer.write_update.call_args.kwargs["points"],
            points,
        )
        np.testing.assert_array_equal(
            writer.write_update.call_args.kwargs["micro_doppler_db"],
            spectrum,
        )


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
            29.8,
        )

        axis.set_xlim.assert_called_once_with(0.0, 10.0)
        axis.set_title.assert_called_once_with("Live Range Profile — 29.8 Hz")

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
            29.8,
        )

        axis.set_xlim.assert_called_once_with(0.0, 10.0)
        axis.set_title.assert_called_once_with(
            "Live Range-Doppler Heatmap — 29.8 Hz"
        )


class MicroDopplerDisplayTests(unittest.TestCase):
    def test_magnitude_colormap_runs_from_dark_blue_to_red(self) -> None:
        self.assertEqual(MAGNITUDE_COLORMAP, "turbo")

    def test_live_history_keeps_150_stft_windows(self) -> None:
        self.assertEqual(MICRO_DOPPLER_HISTORY_UPDATES, 150)

    def test_stft_uses_64_loop_window_and_32_loop_hop(self) -> None:
        self.assertEqual(MICRO_DOPPLER_WINDOW_LOOPS, 64)
        self.assertEqual(MICRO_DOPPLER_HOP_LOOPS, 32)

    def test_live_range_gate_uses_five_bins(self) -> None:
        self.assertEqual(2 * MICRO_DOPPLER_RANGE_HALF_WIDTH_BINS + 1, 5)

    def test_draw_sets_centered_doppler_and_history_axes(self) -> None:
        axis = Mock()
        image = Mock()
        spectrogram = np.arange(12, dtype=np.float32).reshape(4, 3)

        _draw_micro_doppler(axis, image, spectrogram, 2.5, 29.8)

        image.set_data.assert_called_once_with(spectrogram)
        image.set_extent.assert_called_once_with((-2, 0, -2, 1))
        image.set_clim.assert_called_once_with(
            POINT_CLOUD_MAGNITUDE_DB_MIN,
            POINT_CLOUD_MAGNITUDE_DB_MAX,
        )
        axis.set_xlim.assert_called_once_with(-2, 0)
        axis.set_ylim.assert_called_once_with(-2, 1)
        axis.set_title.assert_called_once_with(
            "Live Micro-Doppler Spectrogram — gate 2.50 m"
            " — 29.8 Hz"
        )

    def test_blitted_draw_does_not_mutate_static_axes(self) -> None:
        axis = Mock()
        image = Mock()
        spectrogram = np.arange(12, dtype=np.float32).reshape(4, 3)

        _draw_micro_doppler(
            axis,
            image,
            spectrogram,
            2.5,
            update_axes=False,
            update_title=False,
        )

        axis.set_xlim.assert_not_called()
        axis.set_ylim.assert_not_called()
        axis.set_title.assert_not_called()

    def test_event_rate_counts_units_after_initial_timestamp(self) -> None:
        events = deque()

        self.assertIsNone(_record_event_rate(events, 10.0, 7.0))
        self.assertEqual(_record_event_rate(events, 10.5, 7.0), 14.0)
        self.assertEqual(_record_event_rate(events, 11.0, 7.0), 14.0)

    def test_rate_indicator_formats_display_rate(self) -> None:
        artist = Mock()

        _set_rate_indicator(
            artist,
            29.8,
            range_gate_m=2.5,
        )

        artist.set_text.assert_called_once_with(
            "Gate: 2.50 m\nRefresh rate: 29.8 Hz"
        )

    def test_rate_indicator_shows_measurement_pending(self) -> None:
        artist = Mock()

        _set_rate_indicator(artist, None)

        artist.set_text.assert_called_once_with("Refresh rate: measuring...")

if __name__ == "__main__":
    unittest.main()
