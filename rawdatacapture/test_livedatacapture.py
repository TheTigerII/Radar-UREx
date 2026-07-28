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

from inference import InferenceResult
from rawdatacapture.dsp import (
    MicroDopplerResult,
    RotorEstimate,
    _point_is_within_fov,
)

from rawdatacapture.livedatacapture import (
    CaptureStats,
    CapturedFrame,
    CombinedDisplayPayload,
    COMBINED_DISPLAY_MODE,
    COMBINED_POINT_CLOUD_UPDATE_EVERY,
    DEFAULT_CLUSTER_EPS_M,
    DEFAULT_CLUSTER_MIN_SAMPLES,
    DEFAULT_MAX_RANGE_M,
    DEFAULT_POINT_CLOUD_FOV_DEG,
    DEFAULT_ROTOR_BLADES,
    DEFAULT_ROTOR_RPM_MAX,
    MAGNITUDE_COLORMAP,
    MICRO_DOPPLER_HISTORY_UPDATES,
    MICRO_DOPPLER_HOP_LOOPS,
    MICRO_DOPPLER_RANGE_HALF_WIDTH_BINS,
    MICRO_DOPPLER_WINDOW_LOOPS,
    ROTOR_DISPLAY_MODE,
    ROTOR_DISPLAY_HISTORY_SECONDS,
    ROTOR_DISPLAY_MAX_RATE_HZ,
    ROTOR_DISPLAY_TIME_BINS,
    ROTOR_MICRO_DOPPLER_HOP_LOOPS,
    ROTOR_MICRO_DOPPLER_HISTORY_SECONDS,
    ROTOR_MICRO_DOPPLER_RANGE_HALF_WIDTH_BINS,
    ROTOR_MICRO_DOPPLER_WINDOW_LOOPS,
    POINT_CLOUD_MAGNITUDE_DB_MAX,
    POINT_CLOUD_MAGNITUDE_DB_MIN,
    MotionHandoffQualifier,
    DCA1000PacketHeader,
    DisplayPayloadSink,
    FrameBuffer,
    ProcessedOutputWriter,
    PointCloudDisplayPayload,
    RadarCaptureConfig,
    RotorDisplayPayload,
    SingleTargetTracker,
    StaticReferenceStatus,
    TargetTrack,
    UdpPacketReceiver,
    _capture_error_counts,
    _combined_point_cloud_update_due,
    _colorize_rotor_spectrogram,
    _concatenated_active_window_times,
    _draw_point_cloud,
    _draw_micro_doppler,
    _draw_rotor_micro_doppler,
    _display_dependency_error,
    _fill_rotor_display_time_gaps,
    _gap_aware_series,
    _point_cloud_track_candidates,
    _draw_range_doppler,
    _draw_range_profile,
    _point_cloud_range_limit_m,
    _format_capture_summary,
    _put_latest_queue_payload,
    _record_event_rate,
    _report_display_startup,
    _rotor_qt_application_arguments,
    _prepare_rotor_display_frame,
    _report_new_error_stats,
    _request_processor_stop,
    _rasterize_rotor_spectrogram,
    _run_display_process,
    _run_frame_processor,
    _set_rate_indicator,
    _set_point_cloud_axes,
    _turbo_lookup_table,
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
        display.update.assert_called_once_with(
            range_fft,
            range_axis_m,
            captured_at_s=1.0,
        )

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


class RotorDisplayPayloadSinkTests(unittest.TestCase):
    def test_dedicated_mode_accepts_proven_three_tx_profile_and_bypasses_point_cloud(
        self,
    ) -> None:
        config = RadarCaptureConfig.from_file(
            Path(__file__).with_name("profile.cfg")
        )
        payload_queue = queue.Queue(maxsize=1)
        sink = DisplayPayloadSink(
            ROTOR_DISPLAY_MODE,
            1,
            payload_queue,
            config,
            clutter_map_update_rate=0.0,
            static_detection=False,
            micro_doppler_range_m=2.15,
        )
        frame_result = MicroDopplerResult(
            raw_spectrogram_db=np.ones((128, 1), dtype=np.float32),
            enhanced_spectrogram_db=np.ones((128, 1), dtype=np.float32),
            window_times_s=np.asarray((0.001,)),
            velocity_axis_m_s=np.linspace(-32.0, 31.5, 128, dtype=np.float32),
            flash_scores_db=np.asarray((3.0,), dtype=np.float32),
            noise_floor_db=np.asarray((60.0,), dtype=np.float32),
            selected_range_m=2.15,
            nominal_hop_s=0.000312,
            unambiguous_velocity_m_s=32.0,
        )

        with (
            patch(
                "rawdatacapture.livedatacapture."
                "compute_rotor_micro_doppler_frame",
                return_value=frame_result,
            ) as rotor_dsp,
            patch.object(sink, "_compute_point_cloud_payload") as point_cloud,
        ):
            sink.update(
                np.ones((384, 4, 64), dtype=np.complex64),
                config.range_axis_m(),
                captured_at_s=10.0,
            )

        rotor_dsp.assert_called_once()
        point_cloud.assert_not_called()
        self.assertIsInstance(payload_queue.get_nowait(), RotorDisplayPayload)

    def test_rotor_frame_worker_initializes_with_three_tx_profile(self) -> None:
        config = RadarCaptureConfig.from_file(
            Path(__file__).with_name("profile.cfg")
        )
        frame_queue = queue.Queue()
        frame_queue.put(None)
        log_queue = queue.Queue()

        with patch("rawdatacapture.livedatacapture.signal.signal"):
            _run_frame_processor(
                config=config,
                frame_queue=frame_queue,
                log_queue=log_queue,
                processed_frames_counter=Mock(),
                display_payload_queue=None,
                display_skipped_counter=None,
                raw_output=None,
                raw_metadata=None,
                processed_output=None,
                display_mode=ROTOR_DISPLAY_MODE,
                display_update_every=1,
                max_range_m=10.0,
                point_cloud_fov_deg=60.0,
                cluster_eps_m=0.4,
                cluster_min_samples=2,
                clutter_map_update_rate=0.02,
                clutter_map_warmup_frames=30,
                clutter_map_min_snr_db=6.0,
                static_detection=False,
                static_warmup_frames=30,
                static_reference_frames=90,
                static_min_change_db=6.0,
                static_background_update_rate=0.01,
                static_cluster_min_samples=1,
                micro_doppler_range_m=2.15,
                micro_doppler_range_half_width_bins=1,
                rotor_blades=2,
                rotor_count=1,
                rotor_radius_m=None,
                rotor_rpm_min=500.0,
                rotor_rpm_max=20_000.0,
            )

        messages = []
        while not log_queue.empty():
            messages.append(log_queue.get_nowait())
        self.assertTrue(
            any(
                "Dedicated rotor micro-Doppler enabled" in message
                and "chirps_per_loop=3" in message
                for message in messages
            )
        )
        self.assertFalse(
            any(
                "Frame processor failed during startup" in message
                for message in messages
            )
        )

    def test_rotor_frame_worker_processes_three_tx_frame(self) -> None:
        config = RadarCaptureConfig.from_file(
            Path(__file__).with_name("profile.cfg")
        )
        frame_queue = queue.Queue()
        frame_queue.put(
            CapturedFrame(
                data=bytes(config.bytes_per_frame),
                gap_bytes=0,
                first_byte_at_s=1.0,
            )
        )
        frame_queue.put(None)
        log_queue = queue.Queue()
        payload_queue = queue.Queue(maxsize=1)
        counter = SimpleNamespace(
            value=0,
            get_lock=lambda: threading.Lock(),
        )

        with patch("rawdatacapture.livedatacapture.signal.signal"):
            _run_frame_processor(
                config=config,
                frame_queue=frame_queue,
                log_queue=log_queue,
                processed_frames_counter=counter,
                display_payload_queue=payload_queue,
                display_skipped_counter=None,
                raw_output=None,
                raw_metadata=None,
                processed_output=None,
                display_mode=ROTOR_DISPLAY_MODE,
                display_update_every=1,
                max_range_m=10.0,
                point_cloud_fov_deg=60.0,
                cluster_eps_m=0.4,
                cluster_min_samples=2,
                clutter_map_update_rate=0.02,
                clutter_map_warmup_frames=30,
                clutter_map_min_snr_db=6.0,
                static_detection=False,
                static_warmup_frames=30,
                static_reference_frames=90,
                static_min_change_db=6.0,
                static_background_update_rate=0.01,
                static_cluster_min_samples=1,
                micro_doppler_range_m=2.15,
                micro_doppler_range_half_width_bins=1,
                rotor_blades=2,
                rotor_count=1,
                rotor_radius_m=None,
                rotor_rpm_min=500.0,
                rotor_rpm_max=20_000.0,
            )

        self.assertEqual(counter.value, 1)
        payload = payload_queue.get_nowait()
        self.assertIsInstance(payload, RotorDisplayPayload)
        self.assertEqual(payload.result.raw_spectrogram_db.shape, (0, 0))
        self.assertEqual(payload.result.noise_floor_db.shape, (0,))
        self.assertGreater(payload.result.enhanced_spectrogram_db.size, 0)
        messages = []
        while not log_queue.empty():
            messages.append(log_queue.get_nowait())
        self.assertFalse(
            any("Frame processor stopped after error" in message for message in messages)
        )


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
            writer = ProcessedOutputWriter(
                output_path,
                config,
                1,
                classification_metadata={
                    "enabled": True,
                    "model": "drone_bird_cnn",
                },
            )
            writer.write_update(
                frame_index=7,
                points=np.asarray(((1.0, 2.0, 3.0, 50.0),), dtype=np.float32),
                clusters=np.asarray(((1.0, 2.0, 3.0, 1.0),), dtype=np.float32),
                static_points=np.asarray(
                    ((-1.0, 2.0, 0.0, 55.0, 8.0),),
                    dtype=np.float32,
                ),
                static_clusters=np.asarray(
                    ((-1.0, 2.0, 0.0, 1.0),),
                    dtype=np.float32,
                ),
                static_reference=StaticReferenceStatus(True, True, 30, 30),
                target_track=track,
                target_source="static",
                static_candidate_count=7,
                static_validation="validated",
                micro_doppler_db=np.asarray((10.0, 20.0), dtype=np.float32),
                selected_range_m=2.0,
                classification=InferenceResult(
                    label="drone",
                    p_drone=0.99,
                    threshold=0.98,
                    status="ready",
                    reason=None,
                    valid_steps=48,
                ),
            )
            writer.close()

            records = [json.loads(line) for line in output_path.read_text().splitlines()]

        self.assertEqual(records[0]["record_type"], "metadata")
        self.assertEqual(records[0]["version"], 5)
        self.assertTrue(records[0]["classification"]["enabled"])
        self.assertEqual(records[0]["static_detection"]["warmup_frames"], 30)
        self.assertEqual(records[0]["static_detection"]["reference_frames"], 90)
        self.assertEqual(records[0]["radar_config"]["num_loops"], 128)
        self.assertEqual(
            records[0]["micro_doppler_processing"]["window_loops"],
            64,
        )
        self.assertEqual(records[0]["micro_doppler_processing"]["hop_loops"], 32)
        self.assertIn(
            "per-cell calibration variability",
            records[0]["static_detection"]["noise_policy"],
        )
        self.assertEqual(records[1]["record_type"], "update")
        self.assertEqual(records[1]["processed_frame_index"], 7)
        self.assertEqual(records[1]["points"], [[1.0, 2.0, 3.0, 50.0]])
        self.assertEqual(
            records[1]["static_points"],
            [[-1.0, 2.0, 0.0, 55.0, 8.0]],
        )
        self.assertTrue(records[1]["static_reference"]["ready"])
        self.assertEqual(records[1]["target_source"], "static")
        self.assertEqual(records[1]["static_candidate_count"], 7)
        self.assertEqual(records[1]["static_validation"], "validated")
        self.assertEqual(records[1]["micro_doppler_db"], [10.0, 20.0])
        self.assertEqual(records[1]["micro_doppler_windows_db"], [[10.0, 20.0]])
        self.assertTrue(records[1]["target_track"]["confirmed"])
        self.assertEqual(records[1]["classification"]["label"], "drone")
        self.assertEqual(records[1]["classification"]["valid_steps"], 48)

    def test_writes_structured_rotor_micro_doppler_result(self) -> None:
        config = RadarCaptureConfig.from_file(
            Path(__file__).with_name("profile.cfg")
        )
        result = MicroDopplerResult(
            raw_spectrogram_db=np.ones((4, 2), dtype=np.float32) * 1.2345,
            enhanced_spectrogram_db=np.ones((4, 2), dtype=np.float32) * 7.126,
            window_times_s=np.asarray((0.001, 0.002)),
            velocity_axis_m_s=np.asarray((-2.0, -1.0, 0.0, 1.0)),
            flash_scores_db=np.asarray((3.0, 4.0), dtype=np.float32),
            noise_floor_db=np.asarray((60.0, 61.0), dtype=np.float32),
            selected_range_m=2.15,
            nominal_hop_s=0.000312,
            unambiguous_velocity_m_s=32.0,
            noise_gate_db=np.asarray((3.0, 3.5), dtype=np.float32),
            rotor_estimates=(RotorEstimate(200.0, 6000.0, 0.9),),
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "rotor.jsonl"
            writer = ProcessedOutputWriter(
                output_path,
                config,
                1,
                rotor_processing={"mode": "test"},
            )
            writer.write_update(
                frame_index=1,
                points=np.empty((0, 4), dtype=np.float32),
                clusters=np.empty((0, 4), dtype=np.float32),
                target_track=None,
                micro_doppler_db=result.raw_spectrogram_db[:, -1],
                micro_doppler_windows_db=result.raw_spectrogram_db,
                selected_range_m=result.selected_range_m,
                rotor_micro_doppler=result,
            )
            writer.close()
            records = [
                json.loads(line)
                for line in output_path.read_text().splitlines()
            ]

        self.assertEqual(
            records[0]["radar_config"]["slow_time_rate_hz"],
            config.slow_time_rate_hz,
        )
        rotor = records[1]["rotor_micro_doppler"]
        self.assertEqual(rotor["selected_range_m"], 2.15)
        self.assertEqual(rotor["rotor_estimates"][0]["rpm"], 6000.0)
        self.assertEqual(len(rotor["enhanced_spectrogram_db"]), 2)
        self.assertEqual(rotor["raw_spectrogram_db"][0][0], 1.23)
        self.assertEqual(rotor["enhanced_spectrogram_db"][0][0], 7.13)
        self.assertEqual(rotor["noise_gate_db"], [3.0, 3.5])


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

    def test_draw_updates_static_points_and_calibration_indicator(self) -> None:
        static_scatter = Mock()
        static_cluster_scatter = Mock()
        reference_text = Mock()
        static_points = np.asarray(
            ((1.0, 2.0, 3.0, 55.0, 8.0),),
            dtype=np.float32,
        )
        static_clusters = np.asarray(
            ((1.0, 2.0, 3.0, 1.0),),
            dtype=np.float32,
        )

        _draw_point_cloud(
            Mock(),
            Mock(),
            Mock(),
            np.empty((0, 4), dtype=np.float32),
            np.empty((0, 4), dtype=np.float32),
            10.0,
            60.0,
            static_scatter=static_scatter,
            static_cluster_scatter=static_cluster_scatter,
            static_points=static_points,
            static_clusters=static_clusters,
            static_reference_text=reference_text,
            static_reference=StaticReferenceStatus(True, False, 12, 30),
        )

        for actual, expected in zip(
            static_scatter._offsets3d,
            static_points[:, :3].T,
        ):
            np.testing.assert_array_equal(actual, expected)
        reference_text.set_text.assert_called_once_with(
            "Calibrating static reference 12/30"
        )

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

    def test_nearest_policy_acquires_nearest_candidate(self) -> None:
        tracker = SingleTargetTracker(
            acquisition_policy="nearest",
            confirmation_hits=1,
        )

        track = tracker.update(
            self._candidates((0.0, 2.0, 0.0, 50.0), (0.0, 4.0, 0.0, 120.0))
        )

        assert track is not None
        np.testing.assert_allclose(track.position_m, (0.0, 2.0, 0.0))

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
            max_missed_updates=2,
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


class MotionHandoffQualifierTests(unittest.TestCase):
    @staticmethod
    def _track(x_m: float) -> TargetTrack:
        return TargetTrack(
            position_m=(x_m, 2.0, 0.0),
            velocity_m_per_update=(0.0, 0.0, 0.0),
            age_updates=3,
            hits=3,
            missed_updates=0,
            confirmed=True,
        )

    def test_stationary_dynamic_track_does_not_open_handoff(self) -> None:
        qualifier = MotionHandoffQualifier(
            history_updates=5,
            minimum_displacement_m=0.3,
            handoff_window_updates=4,
        )

        for x_m in (0.0, 0.05, 0.1, 0.08):
            handoff = qualifier.update(self._track(x_m))

        self.assertIsNone(handoff)

    def test_qualified_motion_opens_and_expires_handoff_window(self) -> None:
        qualifier = MotionHandoffQualifier(
            history_updates=5,
            minimum_displacement_m=0.3,
            handoff_window_updates=4,
        )
        qualifier.update(self._track(0.0))

        handoff = qualifier.update(self._track(0.4))

        assert handoff is not None
        np.testing.assert_allclose(handoff, (0.4, 2.0, 0.0))
        for _ in range(4):
            handoff = qualifier.update(None)
        self.assertIsNone(handoff)

    def test_motion_must_be_within_preceding_processed_updates(self) -> None:
        qualifier = MotionHandoffQualifier(
            history_updates=5,
            minimum_displacement_m=0.3,
            handoff_window_updates=4,
        )
        qualifier.update(self._track(0.0))
        for _ in range(5):
            qualifier.update(None)

        self.assertIsNone(qualifier.update(self._track(0.4)))

    def test_motion_protection_releases_after_configured_misses(self) -> None:
        qualifier = MotionHandoffQualifier(
            history_updates=5,
            minimum_displacement_m=0.3,
            handoff_window_updates=60,
            protection_missed_updates=30,
        )
        qualifier.update(self._track(0.0))
        qualifier.update(self._track(0.4))

        for _ in range(29):
            qualifier.update(None)
        self.assertIsNotNone(qualifier.protection_position_m)

        qualifier.update(None)
        self.assertIsNone(qualifier.protection_position_m)


class TrackedDisplayPayloadTests(unittest.TestCase):
    def test_combined_display_uses_tracked_range_for_micro_doppler(self) -> None:
        payload_queue = queue.Queue(maxsize=1)
        config = SimpleNamespace()
        sink = DisplayPayloadSink(
            COMBINED_DISPLAY_MODE,
            1,
            payload_queue,
            config,
            static_detection=False,
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
            for _ in range(3):
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
        self.assertIsInstance(payload, CombinedDisplayPayload)
        self.assertIsNotNone(payload.point_cloud.target_track)
        self.assertEqual(payload.point_cloud.target_source, "dynamic")

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
        point_cloud = PointCloudDisplayPayload(
            points=points,
            clusters=clusters,
            static_points=np.empty((0, 5), dtype=np.float32),
            static_clusters=np.empty((0, 4), dtype=np.float32),
            target_track=None,
            target_source=None,
            static_reference=StaticReferenceStatus(False, False, 0, 0),
        )
        combined = CombinedDisplayPayload(
            point_cloud=point_cloud,
            spectrogram_db=spectrum[:, None],
            selected_range_m=2.0,
        )

        with patch.object(
            sink,
            "_compute_combined_payload",
            return_value=combined,
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

    def test_unqualified_static_clusters_cannot_override_dynamic_track(self) -> None:
        sink = DisplayPayloadSink(
            COMBINED_DISPLAY_MODE,
            1,
            None,
            SimpleNamespace(),
        )
        sink.dynamic_target_tracker = SingleTargetTracker(confirmation_hits=1)
        sink.static_target_tracker = SingleTargetTracker(
            confirmation_hits=1,
            acquisition_policy="nearest",
        )
        dynamic_points = np.asarray(
            ((0.0, 1.0, 0.0, 100.0),),
            dtype=np.float32,
        )
        dynamic_clusters = np.asarray(
            ((0.0, 1.0, 0.0, 1.0),),
            dtype=np.float32,
        )
        static_points = np.asarray(
            (
                (0.0, 2.0, 0.0, 50.0, 8.0),
                (0.0, 4.0, 0.0, 120.0, 12.0),
            ),
            dtype=np.float32,
        )
        static_clusters = np.asarray(
            ((0.0, 2.0, 0.0, 1.0), (0.0, 4.0, 0.0, 1.0)),
            dtype=np.float32,
        )
        doppler_cube = np.zeros((8, 1, 1, 5), dtype=np.complex64)

        with (
            patch(
                "rawdatacapture.livedatacapture.compute_range_doppler_fft",
                return_value=doppler_cube,
            ),
            patch(
                "rawdatacapture.livedatacapture.compute_point_cloud",
                return_value=dynamic_points,
            ),
            patch(
                "rawdatacapture.livedatacapture.compute_static_point_cloud",
                return_value=static_points,
            ),
            patch(
                "rawdatacapture.livedatacapture.cluster_point_cloud",
                return_value=dynamic_clusters,
            ),
            patch(
                "rawdatacapture.livedatacapture."
                "cluster_point_cloud_with_labels",
                return_value=(
                    static_clusters,
                    np.asarray((0, 1), dtype=np.intp),
                ),
            ),
        ):
            payload, returned_cube = sink._compute_point_cloud_payload(
                np.empty((0,), dtype=np.complex64),
                np.arange(5),
            )

        self.assertIs(returned_cube, doppler_cube)
        self.assertEqual(payload.target_source, "dynamic")
        assert payload.target_track is not None
        np.testing.assert_allclose(payload.target_track.position_m, (0.0, 1.0, 0.0))
        self.assertEqual(payload.static_points.shape, (0, 5))
        self.assertEqual(payload.static_validation, "warming")

    def test_motion_qualified_cluster_hands_off_exact_members_only(self) -> None:
        sink = DisplayPayloadSink(
            COMBINED_DISPLAY_MODE,
            1,
            None,
            SimpleNamespace(),
        )
        motion_handoff = Mock()
        motion_handoff.update.return_value = np.asarray((0.0, 2.0, 0.0))
        motion_handoff.protection_position_m = np.asarray((0.0, 2.0, 0.0))
        sink.motion_handoff = motion_handoff
        dynamic_points = np.empty((0, 4), dtype=np.float32)
        static_points = np.asarray(
            (
                (0.0, 2.0, 0.0, 70.0, 9.0),
                (0.1, 2.0, 0.0, 68.0, 8.5),
                (-0.1, 2.0, 0.0, 66.0, 8.0),
                (0.2, 2.0, 0.0, 120.0, 15.0),
            ),
            dtype=np.float32,
        )
        static_clusters = np.asarray(
            ((0.0, 2.0, 0.0, 3.0),),
            dtype=np.float32,
        )
        static_labels = np.asarray((0, 0, 0, -1), dtype=np.intp)
        doppler_cube = np.zeros((8, 1, 1, 5), dtype=np.complex64)

        with (
            patch(
                "rawdatacapture.livedatacapture.compute_range_doppler_fft",
                return_value=doppler_cube,
            ),
            patch(
                "rawdatacapture.livedatacapture.compute_point_cloud",
                return_value=dynamic_points,
            ),
            patch(
                "rawdatacapture.livedatacapture.compute_static_point_cloud",
                return_value=static_points,
            ),
            patch(
                "rawdatacapture.livedatacapture.cluster_point_cloud",
                return_value=dynamic_points,
            ),
            patch(
                "rawdatacapture.livedatacapture."
                "cluster_point_cloud_with_labels",
                return_value=(static_clusters, static_labels),
            ),
        ):
            for _ in range(3):
                payload, _cube = sink._compute_point_cloud_payload(
                    np.empty((0,), dtype=np.complex64),
                    np.arange(5),
                )

        self.assertEqual(payload.target_source, "static")
        self.assertEqual(payload.static_validation, "validated")
        self.assertEqual(payload.static_candidate_count, 4)
        self.assertEqual(payload.static_points.shape, (3, 5))
        self.assertFalse(np.any(payload.static_points[:, 3] == 120.0))

    def test_single_static_maximum_requires_three_temporal_hits(self) -> None:
        sink = DisplayPayloadSink(
            COMBINED_DISPLAY_MODE,
            1,
            None,
            SimpleNamespace(),
        )
        self.assertEqual(sink.static_cluster_min_samples, 1)
        motion_handoff = Mock()
        motion_handoff.update.return_value = np.asarray((0.0, 2.0, 0.0))
        motion_handoff.protection_position_m = np.asarray((0.0, 2.0, 0.0))
        sink.motion_handoff = motion_handoff
        empty_points = np.empty((0, 4), dtype=np.float32)
        static_point = np.asarray(
            ((0.0, 2.0, 0.0, 70.0, 9.0),),
            dtype=np.float32,
        )
        doppler_cube = np.zeros((8, 1, 1, 5), dtype=np.complex64)

        with (
            patch(
                "rawdatacapture.livedatacapture.compute_range_doppler_fft",
                return_value=doppler_cube,
            ),
            patch(
                "rawdatacapture.livedatacapture.compute_point_cloud",
                return_value=empty_points,
            ),
            patch(
                "rawdatacapture.livedatacapture.compute_static_point_cloud",
                return_value=static_point,
            ),
            patch(
                "rawdatacapture.livedatacapture.cluster_point_cloud",
                return_value=empty_points,
            ),
        ):
            for _ in range(2):
                payload, _cube = sink._compute_point_cloud_payload(
                    np.empty((0,), dtype=np.complex64),
                    np.arange(5),
                )
                self.assertIsNone(payload.target_source)
                self.assertEqual(payload.static_validation, "handoff_pending")

            payload, _cube = sink._compute_point_cloud_payload(
                np.empty((0,), dtype=np.complex64),
                np.arange(5),
            )

        self.assertEqual(payload.target_source, "static")
        self.assertEqual(payload.static_validation, "validated")
        self.assertEqual(payload.static_points.shape, (1, 5))
        self.assertEqual(payload.static_clusters.shape, (1, 4))

    def test_strict_static_cluster_override_rejects_single_maximum(self) -> None:
        sink = DisplayPayloadSink(
            COMBINED_DISPLAY_MODE,
            1,
            None,
            SimpleNamespace(),
            static_cluster_min_samples=3,
        )
        motion_handoff = Mock()
        motion_handoff.update.return_value = np.asarray((0.0, 2.0, 0.0))
        motion_handoff.protection_position_m = np.asarray((0.0, 2.0, 0.0))
        sink.motion_handoff = motion_handoff
        empty_points = np.empty((0, 4), dtype=np.float32)
        static_point = np.asarray(
            ((0.0, 2.0, 0.0, 70.0, 9.0),),
            dtype=np.float32,
        )
        doppler_cube = np.zeros((8, 1, 1, 5), dtype=np.complex64)

        with (
            patch(
                "rawdatacapture.livedatacapture.compute_range_doppler_fft",
                return_value=doppler_cube,
            ),
            patch(
                "rawdatacapture.livedatacapture.compute_point_cloud",
                return_value=empty_points,
            ),
            patch(
                "rawdatacapture.livedatacapture.compute_static_point_cloud",
                return_value=static_point,
            ),
            patch(
                "rawdatacapture.livedatacapture.cluster_point_cloud",
                return_value=empty_points,
            ),
        ):
            for _ in range(3):
                payload, _cube = sink._compute_point_cloud_payload(
                    np.empty((0,), dtype=np.complex64),
                    np.arange(5),
                )

        self.assertIsNone(payload.target_source)
        self.assertEqual(payload.static_validation, "handoff_pending")
        self.assertEqual(payload.static_points.shape, (0, 5))

    def test_static_maximum_outside_handoff_gate_is_rejected(self) -> None:
        sink = DisplayPayloadSink(
            COMBINED_DISPLAY_MODE,
            1,
            None,
            SimpleNamespace(),
        )
        motion_handoff = Mock()
        motion_handoff.update.return_value = np.asarray((0.0, 2.0, 0.0))
        motion_handoff.protection_position_m = np.asarray((0.0, 2.0, 0.0))
        sink.motion_handoff = motion_handoff
        empty_points = np.empty((0, 4), dtype=np.float32)
        static_point = np.asarray(
            ((0.0, 3.0, 0.0, 70.0, 9.0),),
            dtype=np.float32,
        )
        doppler_cube = np.zeros((8, 1, 1, 5), dtype=np.complex64)

        with (
            patch(
                "rawdatacapture.livedatacapture.compute_range_doppler_fft",
                return_value=doppler_cube,
            ),
            patch(
                "rawdatacapture.livedatacapture.compute_point_cloud",
                return_value=empty_points,
            ),
            patch(
                "rawdatacapture.livedatacapture.compute_static_point_cloud",
                return_value=static_point,
            ),
            patch(
                "rawdatacapture.livedatacapture.cluster_point_cloud",
                return_value=empty_points,
            ),
        ):
            for _ in range(3):
                payload, _cube = sink._compute_point_cloud_payload(
                    np.empty((0,), dtype=np.complex64),
                    np.arange(5),
                )

        self.assertIsNone(payload.target_source)
        self.assertEqual(payload.static_validation, "handoff_pending")
        self.assertEqual(payload.static_points.shape, (0, 5))

    def test_disabling_static_detection_skips_static_processing(self) -> None:
        sink = DisplayPayloadSink(
            "point-cloud",
            1,
            None,
            SimpleNamespace(),
            static_detection=False,
        )
        sink.dynamic_target_tracker = SingleTargetTracker(confirmation_hits=1)
        points = np.asarray(((0.0, 2.0, 0.0, 80.0),), dtype=np.float32)
        clusters = np.asarray(((0.0, 2.0, 0.0, 1.0),), dtype=np.float32)
        doppler_cube = np.zeros((8, 1, 1, 5), dtype=np.complex64)

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
                "rawdatacapture.livedatacapture.compute_static_point_cloud",
            ) as static_point_cloud,
        ):
            payload, _cube = sink._compute_point_cloud_payload(
                np.empty((0,), dtype=np.complex64),
                np.arange(5),
            )

        static_point_cloud.assert_not_called()
        self.assertFalse(payload.static_reference.enabled)
        self.assertEqual(payload.static_points.shape, (0, 5))
        self.assertEqual(payload.target_source, "dynamic")

    def test_static_detection_runs_on_every_point_cloud_update(self) -> None:
        sink = DisplayPayloadSink(
            "point-cloud",
            1,
            None,
            SimpleNamespace(),
        )
        doppler_cube = np.zeros((8, 1, 1, 5), dtype=np.complex64)
        empty_points = np.empty((0, 4), dtype=np.float32)
        empty_static_points = np.empty((0, 5), dtype=np.float32)

        with (
            patch(
                "rawdatacapture.livedatacapture.compute_range_doppler_fft",
                return_value=doppler_cube,
            ),
            patch(
                "rawdatacapture.livedatacapture.compute_point_cloud",
                return_value=empty_points,
            ),
            patch(
                "rawdatacapture.livedatacapture.compute_static_point_cloud",
                return_value=empty_static_points,
            ) as static_point_cloud,
            patch(
                "rawdatacapture.livedatacapture.cluster_point_cloud",
                return_value=empty_points,
            ),
        ):
            for _ in range(5):
                sink._compute_point_cloud_payload(
                    np.empty((0,), dtype=np.complex64),
                    np.arange(5),
                )

        self.assertEqual(static_point_cloud.call_count, 5)

    def test_micro_doppler_history_survives_brief_track_gap(self) -> None:
        sink = DisplayPayloadSink(
            COMBINED_DISPLAY_MODE,
            1,
            None,
            SimpleNamespace(),
            static_detection=False,
        )
        first_track = TargetTrack(
            (0.0, 2.0, 0.0),
            (0.0, 0.0, 0.0),
            4,
            4,
            0,
            True,
        )
        reacquired_track = TargetTrack(
            (0.1, 2.1, 0.0),
            (0.0, 0.0, 0.0),
            3,
            3,
            0,
            True,
        )
        empty_points = np.empty((0, 4), dtype=np.float32)
        empty_static_points = np.empty((0, 5), dtype=np.float32)
        status = StaticReferenceStatus(False, False, 0, 0)

        def payload(track, source) -> PointCloudDisplayPayload:
            return PointCloudDisplayPayload(
                empty_points,
                empty_points,
                empty_static_points,
                empty_points,
                track,
                source,
                status,
            )

        doppler_cube = np.zeros((8, 1, 1, 5), dtype=np.complex64)
        spectra = np.arange(12, dtype=np.float32).reshape(4, 3)
        with (
            patch.object(
                sink,
                "_compute_point_cloud_payload",
                side_effect=(
                    (payload(first_track, "dynamic"), doppler_cube),
                    (payload(None, None), doppler_cube),
                    (payload(reacquired_track, "dynamic"), doppler_cube),
                ),
            ),
            patch(
                "rawdatacapture.livedatacapture.compute_micro_doppler_spectrum",
                side_effect=(
                    (np.ones(8, dtype=np.float32), 2.0),
                    (np.ones(8, dtype=np.float32), 2.1),
                ),
            ),
            patch(
                "rawdatacapture.livedatacapture."
                "compute_per_tx_micro_doppler_spectrogram",
                return_value=spectra,
            ),
        ):
            first = sink._compute_combined_payload(
                np.empty((0,), dtype=np.complex64),
                np.arange(5),
            )
            gap = sink._compute_combined_payload(
                np.empty((0,), dtype=np.complex64),
                np.arange(5),
            )
            resumed = sink._compute_combined_payload(
                np.empty((0,), dtype=np.complex64),
                np.arange(5),
            )

        self.assertEqual(first.point_cloud.target_source, "dynamic")
        self.assertIsNone(gap.point_cloud.target_source)
        self.assertEqual(gap.spectrogram_db.shape[1], 3)
        self.assertEqual(resumed.point_cloud.target_source, "dynamic")
        self.assertEqual(len(sink.micro_doppler_history), 6)

    def test_micro_doppler_history_survives_nearby_static_handoff(self) -> None:
        sink = DisplayPayloadSink(
            COMBINED_DISPLAY_MODE,
            1,
            None,
            SimpleNamespace(),
            static_detection=False,
        )
        dynamic_track = TargetTrack(
            (0.0, 2.0, 0.0),
            (0.0, 0.0, 0.0),
            4,
            4,
            0,
            True,
        )
        static_track = TargetTrack(
            (0.1, 2.4, 0.0),
            (0.0, 0.0, 0.0),
            4,
            4,
            0,
            True,
        )
        empty_points = np.empty((0, 4), dtype=np.float32)
        empty_static_points = np.empty((0, 5), dtype=np.float32)
        status = StaticReferenceStatus(False, False, 0, 0)

        def payload(track, source) -> PointCloudDisplayPayload:
            return PointCloudDisplayPayload(
                empty_points,
                empty_points,
                empty_static_points,
                empty_points,
                track,
                source,
                status,
            )

        doppler_cube = np.zeros((8, 1, 1, 5), dtype=np.complex64)
        spectra = np.arange(12, dtype=np.float32).reshape(4, 3)
        with (
            patch.object(
                sink,
                "_compute_point_cloud_payload",
                side_effect=(
                    (payload(dynamic_track, "dynamic"), doppler_cube),
                    (payload(static_track, "static"), doppler_cube),
                ),
            ),
            patch(
                "rawdatacapture.livedatacapture.compute_micro_doppler_spectrum",
                side_effect=(
                    (np.ones(8, dtype=np.float32), 2.0),
                    (np.ones(8, dtype=np.float32), 2.4),
                ),
            ),
            patch(
                "rawdatacapture.livedatacapture."
                "compute_per_tx_micro_doppler_spectrogram",
                return_value=spectra,
            ),
        ):
            sink._compute_combined_payload(
                np.empty((0,), dtype=np.complex64),
                np.arange(5),
            )
            handoff = sink._compute_combined_payload(
                np.empty((0,), dtype=np.complex64),
                np.arange(5),
            )

        self.assertEqual(handoff.point_cloud.target_source, "static")
        self.assertEqual(len(sink.micro_doppler_history), 6)

    def test_micro_doppler_history_survives_continuous_large_motion(self) -> None:
        sink = DisplayPayloadSink(
            COMBINED_DISPLAY_MODE,
            1,
            None,
            SimpleNamespace(),
            static_detection=False,
        )
        first_track = TargetTrack(
            (0.0, 2.0, 0.0),
            (0.0, 0.0, 0.0),
            4,
            4,
            0,
            True,
        )
        moved_track = TargetTrack(
            (0.0, 3.0, 0.0),
            (0.0, 0.0, 0.0),
            5,
            5,
            0,
            True,
        )
        spectrum = np.ones(4, dtype=np.float32)

        self.assertFalse(sink._update_micro_doppler_history_owner(first_track))
        sink.micro_doppler_history.append(spectrum)

        self.assertFalse(sink._update_micro_doppler_history_owner(moved_track))
        self.assertEqual(len(sink.micro_doppler_history), 1)

    def test_micro_doppler_history_resets_for_distant_reacquisition(self) -> None:
        sink = DisplayPayloadSink(
            COMBINED_DISPLAY_MODE,
            1,
            None,
            SimpleNamespace(),
            static_detection=False,
        )
        first_track = TargetTrack(
            (0.0, 2.0, 0.0),
            (0.0, 0.0, 0.0),
            4,
            4,
            0,
            True,
        )
        distant_track = TargetTrack(
            (0.0, 3.0, 0.0),
            (0.0, 0.0, 0.0),
            3,
            3,
            0,
            True,
        )
        spectra = np.arange(12, dtype=np.float32).reshape(4, 3)

        self.assertFalse(sink._update_micro_doppler_history_owner(first_track))
        sink.micro_doppler_history.extend(spectra[:, index] for index in range(3))
        self.assertFalse(sink._update_micro_doppler_history_owner(None))

        self.assertTrue(sink._update_micro_doppler_history_owner(distant_track))
        self.assertEqual(len(sink.micro_doppler_history), 0)

    def test_micro_doppler_history_expires_after_long_gap(self) -> None:
        sink = DisplayPayloadSink(
            COMBINED_DISPLAY_MODE,
            1,
            None,
            SimpleNamespace(),
            static_detection=False,
        )
        track = TargetTrack(
            (0.0, 2.0, 0.0),
            (0.0, 0.0, 0.0),
            4,
            4,
            0,
            True,
        )

        self.assertFalse(sink._update_micro_doppler_history_owner(track))
        sink.micro_doppler_history.append(np.ones(4, dtype=np.float32))
        for _ in range(31):
            self.assertFalse(sink._update_micro_doppler_history_owner(None))

        self.assertTrue(sink._update_micro_doppler_history_owner(track))
        self.assertEqual(len(sink.micro_doppler_history), 0)

    def test_micro_doppler_history_accepts_thirty_update_gap(self) -> None:
        sink = DisplayPayloadSink(
            COMBINED_DISPLAY_MODE,
            1,
            None,
            SimpleNamespace(),
            static_detection=False,
        )
        track = TargetTrack(
            (0.0, 2.0, 0.0),
            (0.0, 0.0, 0.0),
            4,
            4,
            0,
            True,
        )

        self.assertFalse(sink._update_micro_doppler_history_owner(track))
        sink.micro_doppler_history.append(np.ones(4, dtype=np.float32))
        for _ in range(30):
            self.assertFalse(sink._update_micro_doppler_history_owner(None))

        self.assertFalse(sink._update_micro_doppler_history_owner(track))
        self.assertEqual(len(sink.micro_doppler_history), 1)


class ClassificationIntegrationTests(unittest.TestCase):
    @staticmethod
    def _result(label: str = "unknown") -> InferenceResult:
        return InferenceResult(
            label=label,
            p_drone=0.99 if label == "drone" else None,
            threshold=0.98,
            status="ready" if label != "unknown" else "waiting",
            reason=None if label != "unknown" else "insufficient_history",
            valid_steps=48 if label != "unknown" else 1,
        )

    def test_fixed_range_is_converted_to_nearest_range_bin(self) -> None:
        engine = Mock()
        engine.unknown.return_value = self._result()
        engine.update.return_value = self._result("drone")
        sink = DisplayPayloadSink(
            "none",
            1,
            None,
            SimpleNamespace(),
            inference_engine=engine,
        )
        doppler_cube = np.zeros((128, 3, 4, 64), dtype=np.complex64)
        range_axis = np.arange(64, dtype=np.float32) * 0.2

        sink._classify_fixed_range(doppler_cube, range_axis, 2.05)

        engine.update.assert_called_once_with(doppler_cube, 10)
        self.assertEqual(sink.latest_classification.label, "drone")

    def test_predicted_track_resets_classification_history(self) -> None:
        engine = Mock()
        engine.unknown.return_value = self._result()
        engine.reset.return_value = self._result()
        sink = DisplayPayloadSink(
            "none",
            1,
            None,
            SimpleNamespace(),
            inference_engine=engine,
        )
        predicted = TargetTrack(
            position_m=(0.0, 2.0, 0.0),
            velocity_m_per_update=(0.0, 0.0, 0.0),
            age_updates=10,
            hits=9,
            missed_updates=1,
            confirmed=True,
        )

        sink._classify_tracked_target(
            np.zeros((128, 3, 4, 64), dtype=np.complex64),
            np.arange(64, dtype=np.float32),
            predicted,
            target_changed=False,
        )

        engine.reset.assert_called_once_with("predicted_target")
        engine.update.assert_not_called()


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

    def test_shared_magnitude_colorbar_starts_at_sixty_db(self) -> None:
        self.assertEqual(POINT_CLOUD_MAGNITUDE_DB_MIN, 60.0)

    def test_combined_point_cloud_keeps_full_rate_rendering(self) -> None:
        self.assertEqual(COMBINED_POINT_CLOUD_UPDATE_EVERY, 1)
        self.assertEqual(
            [
                _combined_point_cloud_update_due(update_count)
                for update_count in range(1, 6)
            ],
            [True, True, True, True, True],
        )

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

    def test_dedicated_rotor_defaults_prioritize_flash_timing(self) -> None:
        self.assertEqual(ROTOR_DISPLAY_MODE, "micro-doppler")
        self.assertEqual(DEFAULT_ROTOR_BLADES, 2)
        self.assertEqual(DEFAULT_ROTOR_RPM_MAX, 10_700.0)
        self.assertEqual(ROTOR_DISPLAY_MAX_RATE_HZ, 60.0)
        self.assertEqual(ROTOR_MICRO_DOPPLER_WINDOW_LOOPS, 16)
        self.assertEqual(ROTOR_MICRO_DOPPLER_HOP_LOOPS, 2)
        self.assertEqual(ROTOR_DISPLAY_HISTORY_SECONDS, 0.2)
        self.assertEqual(ROTOR_MICRO_DOPPLER_HISTORY_SECONDS, 2.0)
        self.assertEqual(
            2 * ROTOR_MICRO_DOPPLER_RANGE_HALF_WIDTH_BINS + 1,
            3,
        )

    def test_rotor_time_resolution_separates_max_rpm_blade_passages(
        self,
    ) -> None:
        config = RadarCaptureConfig.from_file(
            Path(__file__).with_name("profile.cfg")
        )
        slow_time_interval_s = float(config.slow_time_interval_s)
        window_span_s = (
            ROTOR_MICRO_DOPPLER_WINDOW_LOOPS * slow_time_interval_s
        )
        hop_s = ROTOR_MICRO_DOPPLER_HOP_LOOPS * slow_time_interval_s
        blade_passage_interval_s = 60.0 / (
            DEFAULT_ROTOR_BLADES * DEFAULT_ROTOR_RPM_MAX
        )

        self.assertLess(window_span_s, blade_passage_interval_s)
        self.assertLess(hop_s, blade_passage_interval_s / 4.0)
        self.assertLess(
            ROTOR_DISPLAY_HISTORY_SECONDS / ROTOR_DISPLAY_TIME_BINS,
            blade_passage_interval_s / 2.0,
        )

    def test_rotor_draw_uses_relative_scale_velocity_and_rpm_status(self) -> None:
        axis = Mock()
        image = Mock()
        flash_axis = Mock()
        flash_line = Mock()
        status_text = Mock()
        result = MicroDopplerResult(
            raw_spectrogram_db=np.ones((8, 3), dtype=np.float32),
            enhanced_spectrogram_db=np.ones((8, 3), dtype=np.float32) * 9.0,
            window_times_s=np.asarray((0.001, 0.002, 0.011)),
            velocity_axis_m_s=np.linspace(-4.0, 3.0, 8, dtype=np.float32),
            flash_scores_db=np.asarray((2.0, 5.0, 3.0), dtype=np.float32),
            noise_floor_db=np.ones(3, dtype=np.float32),
            selected_range_m=2.15,
            nominal_hop_s=0.001,
            unambiguous_velocity_m_s=4.0,
            rotor_estimates=(
                RotorEstimate(
                    blade_passage_hz=200.0,
                    rpm=6000.0,
                    confidence=0.9,
                ),
            ),
        )

        _draw_rotor_micro_doppler(
            axis,
            image,
            flash_axis,
            flash_line,
            status_text,
            result,
        )

        axis.set_ylim.assert_called_once_with(-4.0, 3.0)
        displayed = image.set_data.call_args.args[0]
        self.assertEqual(displayed.shape, (8, ROTOR_DISPLAY_TIME_BINS))
        self.assertTrue(np.isfinite(displayed[0]).all())
        self.assertIn("6000 RPM", status_text.set_text.call_args.args[0])
        self.assertIn("Gate: 2.15 m", status_text.set_text.call_args.args[0])

    def test_rotor_display_frame_includes_notch_overlay_bounds(self) -> None:
        result = MicroDopplerResult(
            raw_spectrogram_db=np.empty((0, 0), dtype=np.float32),
            enhanced_spectrogram_db=np.ones((8, 3), dtype=np.float32) * 9.0,
            window_times_s=np.asarray((0.001, 0.002, 0.011)),
            velocity_axis_m_s=np.linspace(-4.0, 3.0, 8, dtype=np.float32),
            flash_scores_db=np.asarray((2.0, 5.0, 3.0), dtype=np.float32),
            noise_floor_db=np.empty(0, dtype=np.float32),
            selected_range_m=2.15,
            nominal_hop_s=0.001,
            unambiguous_velocity_m_s=4.0,
        )

        display_frame = _prepare_rotor_display_frame(result)

        self.assertIsNotNone(display_frame)
        assert display_frame is not None
        self.assertEqual(
            display_frame.spectrogram_db.shape,
            (8, ROTOR_DISPLAY_TIME_BINS),
        )
        self.assertTrue(np.isnan(display_frame.spectrogram_db[2:7]).all())
        self.assertEqual(display_frame.dc_notch_velocity_bounds, (-2.5, 2.5))

    def test_turbo_lookup_table_is_compact_uint8_rgb(self) -> None:
        lookup_table = _turbo_lookup_table()

        self.assertEqual(lookup_table.shape, (256, 3))
        self.assertEqual(lookup_table.dtype, np.uint8)
        self.assertGreater(int(lookup_table[-1, 0]), int(lookup_table[-1, 2]))

    def test_rotor_colorization_produces_finite_direct_rgba_image(self) -> None:
        lookup_table = _turbo_lookup_table()

        colored = _colorize_rotor_spectrogram(
            np.asarray(((np.nan, 0.0, 15.0, 30.0, np.inf),)),
            lookup_table,
        )

        self.assertEqual(colored.shape, (1, 5, 4))
        self.assertEqual(colored.dtype, np.uint8)
        self.assertTrue(colored.flags.c_contiguous)
        np.testing.assert_array_equal(colored[0, 0, :3], lookup_table[0])
        np.testing.assert_array_equal(colored[0, 2, :3], lookup_table[128])
        np.testing.assert_array_equal(colored[0, 4, :3], lookup_table[-1])
        np.testing.assert_array_equal(colored[:, :, 3], 255)

    @patch("rawdatacapture.livedatacapture._run_rotor_pyqtgraph_display")
    def test_rotor_display_process_dispatches_to_pyqtgraph(
        self,
        rotor_renderer: Mock,
    ) -> None:
        payload_queue = Mock()
        stop_event = Mock()
        rendered_updates = Mock()
        skipped_updates = Mock()
        startup_status_queue = Mock()

        _run_display_process(
            ROTOR_DISPLAY_MODE,
            0.03,
            10.0,
            10.0,
            60.0,
            payload_queue,
            stop_event,
            rendered_updates,
            skipped_updates,
            startup_status_queue,
        )

        rotor_renderer.assert_called_once_with(
            0.03,
            payload_queue,
            stop_event,
            rendered_updates,
            skipped_updates,
            startup_status_queue,
        )

    def test_display_startup_status_reports_ready_backend(self) -> None:
        startup_status_queue = queue.Queue(maxsize=1)

        _report_display_startup(
            startup_status_queue,
            "ready",
            "Live rotor display ready: backend=PyQtGraph/xcb.",
        )

        self.assertEqual(
            startup_status_queue.get_nowait(),
            {
                "state": "ready",
                "message": (
                    "Live rotor display ready: backend=PyQtGraph/xcb."
                ),
            },
        )

    def test_rotor_qt_arguments_exclude_radar_display_option(self) -> None:
        with patch(
            "sys.argv",
            (
                "livedatacapture.py",
                "--display",
                "micro-doppler",
            ),
        ):
            qt_arguments = _rotor_qt_application_arguments()

        self.assertEqual(qt_arguments, ["radar-rotor-display"])
        self.assertNotIn("--display", qt_arguments)
        self.assertNotIn("micro-doppler", qt_arguments)

    def test_rotor_dependency_error_explains_missing_pyqtgraph(self) -> None:
        with patch(
            "rawdatacapture.livedatacapture.importlib.util.find_spec",
            return_value=None,
        ):
            message = _display_dependency_error(ROTOR_DISPLAY_MODE)

        self.assertIsNotNone(message)
        self.assertIn("PyQtGraph", str(message))

    def test_rotor_dependency_error_explains_missing_xcb_cursor(self) -> None:
        with (
            patch(
                "rawdatacapture.livedatacapture.importlib.util.find_spec",
                return_value=object(),
            ),
            patch(
                "rawdatacapture.livedatacapture."
                "_bundled_xcb_cursor_library",
                return_value=None,
            ),
            patch(
                "rawdatacapture.livedatacapture.sys.platform",
                "linux",
            ),
            patch("ctypes.util.find_library", return_value=None),
            patch.dict("os.environ", {"XDG_SESSION_TYPE": "x11"}),
        ):
            message = _display_dependency_error(ROTOR_DISPLAY_MODE)

        self.assertIsNotNone(message)
        self.assertIn("libxcb-cursor0", str(message))

    def test_rotor_raster_is_bounded_and_max_pooling_preserves_flashes(
        self,
    ) -> None:
        enhanced = np.asarray(
            (
                (3.0, 9.0, 4.0, 7.0),
                (1.0, 2.0, 8.0, 6.0),
            ),
            dtype=np.float32,
        )

        raster, extent = _rasterize_rotor_spectrogram(
            enhanced,
            np.asarray((0.0, 0.1, 1.9, 2.0)),
            np.asarray((-1.0, 1.0)),
            history_seconds=2.0,
            time_bins=4,
        )

        self.assertEqual(raster.shape, (2, 4))
        self.assertEqual(extent, (-2.0, 0.0, -1.0, 1.0))
        self.assertEqual(float(raster[0, 0]), 9.0)
        self.assertEqual(float(raster[1, 3]), 8.0)
        self.assertTrue(np.isnan(raster[:, 1:3]).all())

    def test_rotor_raster_defaults_to_active_acquisition_span(self) -> None:
        raster, extent = _rasterize_rotor_spectrogram(
            np.ones((2, 2), dtype=np.float32),
            np.asarray((0.0, ROTOR_DISPLAY_HISTORY_SECONDS)),
            np.asarray((-1.0, 1.0)),
        )

        self.assertEqual(raster.shape, (2, ROTOR_DISPLAY_TIME_BINS))
        self.assertEqual(
            extent,
            (-ROTOR_DISPLAY_HISTORY_SECONDS, 0.0, -1.0, 1.0),
        )

    def test_rotor_display_concatenates_only_active_window_intervals(
        self,
    ) -> None:
        physical_times = np.asarray(
            (0.001, 0.001234, 0.034207, 0.034441),
            dtype=np.float64,
        )

        active_times = _concatenated_active_window_times(
            physical_times.size,
            0.000234,
        )

        np.testing.assert_allclose(
            active_times,
            (-0.000702, -0.000468, -0.000234, 0.0),
        )
        np.testing.assert_allclose(np.diff(active_times), 0.000234)
        self.assertGreater(float(np.max(np.diff(physical_times))), 0.03)

    def test_rotor_display_fills_time_gaps_from_nearest_spectrum(self) -> None:
        raster = np.asarray(
            (
                (3.0, np.nan, np.nan, 7.0),
                (1.0, np.nan, np.nan, 6.0),
            ),
            dtype=np.float32,
        )

        filled = _fill_rotor_display_time_gaps(raster)

        np.testing.assert_array_equal(filled[:, 0], raster[:, 0])
        np.testing.assert_array_equal(filled[:, 1], raster[:, 0])
        np.testing.assert_array_equal(filled[:, 2], raster[:, 3])
        np.testing.assert_array_equal(filled[:, 3], raster[:, 3])
        self.assertTrue(np.isfinite(filled).all())
        self.assertTrue(np.isnan(raster[:, 1:3]).all())

    def test_gap_aware_series_inserts_nan_at_frame_gap(self) -> None:
        times, values = _gap_aware_series(
            np.asarray((0.0, 0.001, 0.010)),
            np.asarray((1.0, 2.0, 3.0)),
            maximum_gap_s=0.0025,
        )

        self.assertEqual(times.shape, (4,))
        self.assertTrue(np.isnan(values[2]))

if __name__ == "__main__":
    unittest.main()
