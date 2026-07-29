"""Live DCA1000 capture for the Mini4 Phase 1 PMM tracker."""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import queue
import signal
import socket
import sys
import threading
import time
from dataclasses import asdict, dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Any, BinaryIO, Callable, Iterable, Optional, TextIO

import numpy as np

if __package__ in {None, ""}:
    repository_root = str(Path(__file__).resolve().parent.parent)
    if repository_root not in sys.path:
        sys.path.insert(0, repository_root)
    from rawdatacapture.dsp import (
        compute_range_doppler_fft,
        compute_range_doppler_heatmap,
        compute_range_fft,
        compute_range_profile,
        frame_bytes_to_radar_cube,
        range_axis_m,
        range_resolution_m,
    )
    from rawdatacapture.pmm import (
        PmmConfig,
        PmmTrackResult,
        PmmTracker,
        validate_mini4_profile,
    )
else:
    from .dsp import (
        compute_range_doppler_fft,
        compute_range_doppler_heatmap,
        compute_range_fft,
        compute_range_profile,
        frame_bytes_to_radar_cube,
        range_axis_m,
        range_resolution_m,
    )
    from .pmm import (
        PmmConfig,
        PmmTrackResult,
        PmmTracker,
        validate_mini4_profile,
    )


UDP_IP = "192.168.33.30"
UDP_PORT = 4098
BUFFER_SIZE = 65535
DCA1000_HEADER_SIZE = 10
UINT32_MODULO = 2**32
SOCKET_TIMEOUT_SECONDS = 0.5
DEFAULT_PACKET_QUEUE_SIZE = 8192
DEFAULT_PROCESSING_QUEUE_SIZE = 32
DEFAULT_SOCKET_RECV_BUFFER_BYTES = 4 * 1024 * 1024
DEFAULT_LOG_PATH = Path(__file__).with_suffix(".log")
DEFAULT_CONFIG_PATH = Path(__file__).with_name("profile-mini4-20m.cfg")
DEFAULT_SETUP_PATH = Path(__file__).with_name("setup.json")
DEFAULT_MAX_RANGE_M = 20.0
DEFAULT_POINT_CLOUD_FOV_DEG = 60.0
COMBINED_DISPLAY_MODE = "combined"
DISPLAY_CHOICES = (
    "none",
    "range",
    "range-doppler",
    "point-cloud",
    COMBINED_DISPLAY_MODE,
)
PROCESSED_OUTPUT_BUFFER_BYTES = 1024 * 1024
_LOG_FILE: Optional[TextIO] = None
EmitFunc = Callable[[str], None]


class CaptureStartupError(RuntimeError):
    pass


@dataclass(frozen=True)
class DCA1000PacketHeader:
    sequence_number: int
    byte_count: int

    @classmethod
    def parse(
        cls,
        packet: bytes | bytearray | memoryview,
    ) -> "DCA1000PacketHeader":
        if len(packet) < DCA1000_HEADER_SIZE:
            raise ValueError("DCA1000 packet is shorter than its 10-byte header")
        return cls(
            sequence_number=int.from_bytes(packet[:4], "little"),
            byte_count=int.from_bytes(packet[4:10], "little"),
        )


@dataclass
class CaptureStats:
    packets_received: int = 0
    frames_emitted: int = 0
    malformed_packets: int = 0
    lost_packets: int = 0
    out_of_order_packets: int = 0
    duplicate_packets: int = 0
    byte_gaps: int = 0
    byte_gap_bytes: int = 0
    stream_resyncs: int = 0
    byte_overlaps: int = 0
    byte_overlap_bytes: int = 0
    invalid_frames: int = 0
    receiver_queue_drops: int = 0
    processing_frames_dropped: int = 0


@dataclass(frozen=True)
class CapturedFrame:
    data: bytes
    gap_bytes: int
    first_byte_at_s: float
    capture_diagnostics: Optional[dict[str, int]] = None

    @property
    def is_valid(self) -> bool:
        return self.gap_bytes == 0


@dataclass(frozen=True)
class TargetTrack:
    position_m: tuple[float, float, float]


@dataclass(frozen=True)
class PointCloudDisplayPayload:
    target_track: Optional[TargetTrack]
    tracking_status: str


@dataclass(frozen=True)
class CombinedDisplayPayload:
    point_cloud: PointCloudDisplayPayload
    spectrogram_db: np.ndarray


@dataclass(frozen=True)
class RadarCaptureConfig:
    num_adc_samples: int
    num_rx_channels: int
    num_chirps_per_frame: int
    bytes_per_frame: int
    iq_swap: bool
    channel_interleave: bool
    lvds_lanes: int
    num_loops: Optional[int] = None
    num_chirps_per_loop: Optional[int] = None
    tx_channel_masks: Optional[tuple[int, ...]] = None
    sample_rate_ksps: Optional[float] = None
    frequency_slope_mhz_per_us: Optional[float] = None
    start_frequency_ghz: Optional[float] = None
    idle_time_us: Optional[float] = None
    adc_start_time_us: Optional[float] = None
    ramp_end_time_us: Optional[float] = None
    frame_periodicity_ms: Optional[float] = None

    @classmethod
    def from_dimensions(
        cls,
        *,
        num_adc_samples: int,
        num_rx_channels: int,
        num_chirps_per_frame: int,
        iq_swap: bool = False,
        channel_interleave: bool = False,
        lvds_lanes: int = 2,
        num_loops: Optional[int] = None,
        num_chirps_per_loop: Optional[int] = None,
        tx_channel_masks: Optional[Iterable[int]] = None,
        sample_rate_ksps: Optional[float] = None,
        frequency_slope_mhz_per_us: Optional[float] = None,
        start_frequency_ghz: Optional[float] = None,
        idle_time_us: Optional[float] = None,
        adc_start_time_us: Optional[float] = None,
        ramp_end_time_us: Optional[float] = None,
        frame_periodicity_ms: Optional[float] = None,
    ) -> "RadarCaptureConfig":
        bytes_per_frame = (
            int(num_adc_samples)
            * int(num_rx_channels)
            * int(num_chirps_per_frame)
            * 4
        )
        return cls(
            num_adc_samples=int(num_adc_samples),
            num_rx_channels=int(num_rx_channels),
            num_chirps_per_frame=int(num_chirps_per_frame),
            bytes_per_frame=bytes_per_frame,
            iq_swap=bool(iq_swap),
            channel_interleave=bool(channel_interleave),
            lvds_lanes=int(lvds_lanes),
            num_loops=num_loops,
            num_chirps_per_loop=num_chirps_per_loop,
            tx_channel_masks=(
                tuple(int(mask) for mask in tx_channel_masks)
                if tx_channel_masks is not None
                else None
            ),
            sample_rate_ksps=sample_rate_ksps,
            frequency_slope_mhz_per_us=frequency_slope_mhz_per_us,
            start_frequency_ghz=start_frequency_ghz,
            idle_time_us=idle_time_us,
            adc_start_time_us=adc_start_time_us,
            ramp_end_time_us=ramp_end_time_us,
            frame_periodicity_ms=frame_periodicity_ms,
        )

    @classmethod
    def from_file(cls, config_path: Path) -> "RadarCaptureConfig":
        resolved = _resolve_config_path(config_path)
        if resolved.suffix.lower() != ".cfg":
            raise ValueError("Phase 1 accepts only an SDK CLI .cfg radar profile")
        return _config_from_cfg_lines(
            resolved.read_text(encoding="utf-8").splitlines()
        )

    @property
    def range_resolution_m(self) -> Optional[float]:
        return range_resolution_m(self)

    def range_axis_m(self) -> Optional[np.ndarray]:
        return range_axis_m(self)

    @property
    def chirp_period_s(self) -> Optional[float]:
        if self.idle_time_us is None or self.ramp_end_time_us is None:
            return None
        return (self.idle_time_us + self.ramp_end_time_us) * 1e-6

    @property
    def slow_time_interval_s(self) -> Optional[float]:
        if self.chirp_period_s is None:
            return None
        return self.chirp_period_s * int(self.num_chirps_per_loop or 1)

    @property
    def slow_time_rate_hz(self) -> Optional[float]:
        interval = self.slow_time_interval_s
        return None if not interval else 1.0 / interval

    @property
    def frame_duty_cycle(self) -> Optional[float]:
        if self.chirp_period_s is None or not self.frame_periodicity_ms:
            return None
        return (
            self.num_chirps_per_frame
            * self.chirp_period_s
            / (self.frame_periodicity_ms * 1e-3)
        )


@dataclass(frozen=True)
class CaptureSetupConfig:
    packet_sequence_enable: bool
    packet_delay_us: Optional[int] = None
    capture_hardware: Optional[str] = None

    @classmethod
    def from_file(cls, setup_path: Path) -> "CaptureSetupConfig":
        data = json.loads(
            _resolve_config_path(setup_path).read_text(encoding="utf-8")
        )
        dca = data.get("DCA1000Config", {}) if isinstance(data, dict) else {}
        packet_sequence = dca.get("packetSequenceEnable", 1)
        return cls(
            packet_sequence_enable=bool(int(packet_sequence)),
            packet_delay_us=(
                int(dca["packetDelay_us"])
                if "packetDelay_us" in dca
                else None
            ),
            capture_hardware=data.get("captureHardware"),
        )


class SequenceTracker:
    def __init__(self, stats: CaptureStats) -> None:
        self.stats = stats
        self.expected_sequence: Optional[int] = None

    def observe(self, sequence_number: int) -> None:
        if self.expected_sequence is None:
            self.expected_sequence = (sequence_number + 1) % UINT32_MODULO
            return
        if sequence_number == self.expected_sequence:
            self.expected_sequence = (self.expected_sequence + 1) % UINT32_MODULO
            return
        if sequence_number == (self.expected_sequence - 1) % UINT32_MODULO:
            self.stats.duplicate_packets += 1
            return
        delta = (sequence_number - self.expected_sequence) % UINT32_MODULO
        if 0 < delta < UINT32_MODULO // 2:
            self.stats.lost_packets += delta
            self.expected_sequence = (sequence_number + 1) % UINT32_MODULO
            return
        self.stats.out_of_order_packets += 1


class FrameBuffer:
    def __init__(self, bytes_per_frame: int, stats: CaptureStats) -> None:
        self.bytes_per_frame = int(bytes_per_frame)
        self.stats = stats
        self.base_byte_count: Optional[int] = None
        self.next_stream_offset = 0
        self.buffer = bytearray()
        self.gap_markers: list[tuple[int, int]] = []
        self.arrival_markers: list[tuple[int, int, float]] = []

    def add_payload(
        self,
        header: DCA1000PacketHeader,
        payload: bytes | bytearray | memoryview,
        payload_received_at_s: float,
    ) -> list[CapturedFrame]:
        if self.base_byte_count is None:
            self.base_byte_count = header.byte_count
        payload_start = header.byte_count - self.base_byte_count
        if payload_start < 0:
            self.stats.out_of_order_packets += 1
            return []
        if payload_start + len(payload) <= self.next_stream_offset:
            self.stats.duplicate_packets += 1
            return []
        if payload_start > self.next_stream_offset:
            gap = payload_start - self.next_stream_offset
            if gap > self.bytes_per_frame:
                self._resynchronize(header.byte_count)
                payload_start = 0
                gap = 0
            if gap:
                gap_start = len(self.buffer)
                self.buffer.extend(b"\x00" * gap)
                self.gap_markers.append((gap_start, gap_start + gap))
                self.next_stream_offset += gap
                self.stats.byte_gaps += 1
                self.stats.byte_gap_bytes += gap
        if payload_start < self.next_stream_offset:
            overlap = self.next_stream_offset - payload_start
            payload = payload[overlap:]
            self.stats.byte_overlaps += 1
            self.stats.byte_overlap_bytes += overlap
        if payload:
            start = len(self.buffer)
            self.buffer.extend(payload)
            self.arrival_markers.append(
                (start, start + len(payload), payload_received_at_s)
            )
            self.next_stream_offset += len(payload)

        frames: list[CapturedFrame] = []
        while len(self.buffer) >= self.bytes_per_frame:
            gap_bytes = sum(
                max(0, min(end, self.bytes_per_frame) - max(start, 0))
                for start, end in self.gap_markers
            )
            received_at = next(
                (
                    timestamp
                    for start, end, timestamp in self.arrival_markers
                    if end > 0 and start < self.bytes_per_frame
                ),
                payload_received_at_s,
            )
            frames.append(
                CapturedFrame(
                    data=bytes(self.buffer[: self.bytes_per_frame]),
                    gap_bytes=gap_bytes,
                    first_byte_at_s=received_at,
                )
            )
            if gap_bytes:
                self.stats.invalid_frames += 1
            del self.buffer[: self.bytes_per_frame]
            self.gap_markers = [
                (max(start - self.bytes_per_frame, 0), end - self.bytes_per_frame)
                for start, end in self.gap_markers
                if end > self.bytes_per_frame
            ]
            self.arrival_markers = [
                (
                    max(start - self.bytes_per_frame, 0),
                    end - self.bytes_per_frame,
                    timestamp,
                )
                for start, end, timestamp in self.arrival_markers
                if end > self.bytes_per_frame
            ]
            self.stats.frames_emitted += 1
        return frames

    def _resynchronize(self, byte_count: int) -> None:
        self.base_byte_count = byte_count
        self.next_stream_offset = 0
        self.buffer.clear()
        self.gap_markers.clear()
        self.arrival_markers.clear()
        self.stats.stream_resyncs += 1


class UdpPacketReceiver:
    def __init__(
        self,
        sock: socket.socket,
        packet_queue: queue.Queue[tuple[bytes, float]],
        buffer_size: int,
        stats: CaptureStats,
    ) -> None:
        self.sock = sock
        self.packet_queue = packet_queue
        self.buffer_size = buffer_size
        self.stats = stats
        self.stop_event = threading.Event()
        self.error: Optional[OSError] = None
        self.thread = threading.Thread(
            target=self._receive_loop,
            name="RadarUdpReceiver",
            daemon=True,
        )

    def start(self) -> None:
        self.thread.start()

    def stop(self) -> None:
        self.stop_event.set()

    def join(self, timeout: Optional[float] = None) -> None:
        self.thread.join(timeout)

    @property
    def is_alive(self) -> bool:
        return self.thread.is_alive()

    def _receive_loop(self) -> None:
        while not self.stop_event.is_set():
            try:
                packet = self.sock.recv(self.buffer_size)
            except socket.timeout:
                continue
            except OSError as exc:
                if not self.stop_event.is_set():
                    self.error = exc
                return
            try:
                self.packet_queue.put_nowait((packet, time.perf_counter()))
            except queue.Full:
                self.stats.receiver_queue_drops += 1


class ProcessingTimingStats:
    STAGES = ("range_fft", "doppler_fft", "pmm_tracking", "serialization", "total")

    def __init__(self) -> None:
        self.samples: dict[str, list[float]] = {
            stage: [] for stage in self.STAGES
        }

    def add_ms(self, stage: str, elapsed_ms: float) -> None:
        if stage in self.samples:
            self.samples[stage].append(float(elapsed_ms))

    def summary(self) -> dict[str, dict[str, float]]:
        result: dict[str, dict[str, float]] = {}
        for stage, values in self.samples.items():
            if not values:
                continue
            result[stage] = {
                "p50_ms": float(np.percentile(values, 50)),
                "p95_ms": float(np.percentile(values, 95)),
                "max_ms": float(np.max(values)),
            }
        return result


class RawFrameWriter:
    def __init__(
        self,
        output_path: Optional[Path],
        metadata_path: Optional[Path],
        config: RadarCaptureConfig,
        emit_func: EmitFunc,
    ) -> None:
        self.output_path = _resolve_output_path(output_path) if output_path else None
        self.metadata_path = (
            _resolve_output_path(metadata_path)
            if metadata_path
            else (
                Path(f"{self.output_path}.json")
                if self.output_path is not None
                else None
            )
        )
        self.config = config
        self.emit = emit_func
        self.file: Optional[BinaryIO] = None
        self.frames_saved = 0
        self.invalid_frames_skipped = 0
        if self.output_path is not None:
            self.output_path.parent.mkdir(parents=True, exist_ok=True)
            self.file = self.output_path.open("wb")
            self.emit(f"Saving valid raw frames to {self.output_path}")

    def write_frame(self, frame: CapturedFrame) -> None:
        if self.file is None:
            return
        if not frame.is_valid:
            self.invalid_frames_skipped += 1
            return
        self.file.write(frame.data)
        self.frames_saved += 1

    def close(self) -> None:
        if self.file is None:
            return
        self.file.close()
        self.file = None
        if self.metadata_path is not None:
            self.metadata_path.parent.mkdir(parents=True, exist_ok=True)
            self.metadata_path.write_text(
                json.dumps(
                    {
                        "format": "dca1000-valid-frames",
                        "frames_saved": self.frames_saved,
                        "invalid_frames_skipped": self.invalid_frames_skipped,
                        "bytes_per_frame": self.config.bytes_per_frame,
                        "radar_config": asdict(self.config),
                    },
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
        self.emit(
            f"Raw capture saved: frames={self.frames_saved}, "
            f"path={self.output_path}"
        )


class ProcessedOutputWriter:
    def __init__(
        self,
        output_path: Optional[Path],
        config: RadarCaptureConfig,
        emit_func: Optional[EmitFunc] = None,
        *,
        pmm_metadata: Optional[dict[str, Any]] = None,
    ) -> None:
        self.emit = emit_func or emit
        self.output_path = _resolve_output_path(output_path) if output_path else None
        self.file: Optional[TextIO] = None
        self.updates_saved = 0
        if self.output_path is None:
            return
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.file = self.output_path.open(
            "w",
            encoding="utf-8",
            buffering=PROCESSED_OUTPUT_BUFFER_BYTES,
        )
        metadata = {
            "record_type": "metadata",
            "format": "mini4-pmm-jsonl",
            "version": 1,
            "created_at": _timestamp(),
            "label": "PMM target",
            "pmm_tracking": pmm_metadata,
            "radar_config": asdict(config),
        }
        self.file.write(json.dumps(metadata, separators=(",", ":")) + "\n")
        self.emit(f"Saving PMM output to {self.output_path}")

    @property
    def enabled(self) -> bool:
        return self.file is not None

    def write_update(
        self,
        *,
        frame_index: int,
        pmm_result: PmmTrackResult,
        doppler_time_db: np.ndarray,
        diagnostics: dict[str, Any],
    ) -> None:
        if self.file is None:
            return
        record = {
            "record_type": "update",
            "update_index": self.updates_saved,
            "processed_frame_index": int(frame_index),
            "recorded_at": datetime.now().astimezone().isoformat(
                timespec="milliseconds"
            ),
            "pmm_tracking": pmm_result.to_dict(),
            "doppler_time_db": np.asarray(
                doppler_time_db,
                dtype=np.float32,
            ).T.tolist(),
            "diagnostics": diagnostics,
        }
        self.file.write(json.dumps(record, separators=(",", ":")) + "\n")
        self.updates_saved += 1

    def close(self) -> None:
        if self.file is None:
            return
        self.file.close()
        self.file = None
        self.emit(
            f"PMM output saved: updates={self.updates_saved}, "
            f"path={self.output_path}"
        )


def _target_track(result: PmmTrackResult) -> Optional[TargetTrack]:
    if not result.has_track or result.range_m is None:
        return None
    azimuth = np.deg2rad(result.azimuth_deg or 0.0)
    elevation = np.deg2rad(result.elevation_deg or 0.0)
    horizontal = result.range_m * np.cos(elevation)
    return TargetTrack(
        position_m=(
            float(horizontal * np.sin(azimuth)),
            float(horizontal * np.cos(azimuth)),
            float(result.range_m * np.sin(elevation)),
        ),
    )


class DisplayPayloadSink:
    def __init__(
        self,
        mode: str,
        update_every: int,
        payload_queue: Optional[mp.Queue],
        config: RadarCaptureConfig,
        tracker: PmmTracker,
        processed_writer: ProcessedOutputWriter,
        display_skipped_counter: Optional[Any] = None,
    ) -> None:
        self.mode = mode
        self.update_every = max(int(update_every), 1)
        self.payload_queue = payload_queue
        self.config = config
        self.tracker = tracker
        self.processed_writer = processed_writer
        self.display_skipped_counter = display_skipped_counter
        self.frame_count = 0
        self.capture_diagnostics: dict[str, int] = {}
        self.latest_range_fft_ms = 0.0
        self.timings = ProcessingTimingStats()

    def update(
        self,
        range_fft: np.ndarray,
        range_axis: np.ndarray,
        *,
        captured_at_s: Optional[float] = None,
    ) -> None:
        total_started = time.perf_counter()
        self.frame_count += 1
        started = time.perf_counter()
        doppler_cube = compute_range_doppler_fft(
            range_fft,
            self.config,
            fft_size=64,
        )
        doppler_ms = (time.perf_counter() - started) * 1_000.0
        started = time.perf_counter()
        result = self.tracker.update(
            doppler_cube,
            range_fft,
            range_axis,
            timestamp_s=captured_at_s,
        )
        tracking_ms = (time.perf_counter() - started) * 1_000.0
        result = replace(
            result,
            processing_ms={
                "range_fft": self.latest_range_fft_ms,
                "doppler_fft": doppler_ms,
                "pmm_tracking": tracking_ms,
                "total": self.latest_range_fft_ms + doppler_ms + tracking_ms,
            },
        )
        self.tracker.latest_result = result
        self.timings.add_ms("doppler_fft", doppler_ms)
        self.timings.add_ms("pmm_tracking", tracking_ms)

        serialization_started = time.perf_counter()
        self.processed_writer.write_update(
            frame_index=self.frame_count,
            pmm_result=result,
            doppler_time_db=self.tracker.spectrogram_db,
            diagnostics={
                "stage_latency_ms": result.processing_ms,
                "capture": dict(self.capture_diagnostics),
            },
        )
        self.timings.add_ms(
            "serialization",
            (time.perf_counter() - serialization_started) * 1_000.0,
        )

        if (
            self.payload_queue is not None
            and self.mode != "none"
            and self.frame_count % self.update_every == 0
        ):
            gate = (range_axis >= 0.3) & (range_axis <= 20.0)
            if self.mode == "range":
                payload: Any = (
                    range_axis[gate],
                    compute_range_profile(range_fft)[gate],
                )
            elif self.mode == "range-doppler":
                heatmap = compute_range_doppler_heatmap(range_fft, self.config)
                payload = (range_axis[gate], heatmap[:, gate])
            else:
                score_text = (
                    "—"
                    if result.pmm_score is None
                    else f"{result.pmm_score:,.0f}"
                )
                range_text = (
                    "—"
                    if result.range_m is None
                    else f"{result.range_m:.2f} m"
                )
                folding_text = (
                    "—"
                    if result.folding_size is None
                    else str(result.folding_size)
                )
                point = PointCloudDisplayPayload(
                    target_track=_target_track(result),
                    tracking_status=(
                        f"PMM target — {result.state} | calibration "
                        f"{result.calibration_frames_seen}/"
                        f"{result.calibration_frames_required} | score "
                        f"{score_text}/{result.threshold:,.0f} | range "
                        f"{range_text} | fold {folding_text}"
                    ),
                )
                payload = (
                    point
                    if self.mode == "point-cloud"
                    else CombinedDisplayPayload(
                        point_cloud=point,
                        spectrogram_db=self.tracker.spectrogram_db,
                    )
                )
            skipped = _put_latest_queue_payload(self.payload_queue, payload)
            _increment_shared_counter(self.display_skipped_counter, skipped)
        self.timings.add_ms(
            "total",
            (time.perf_counter() - total_started) * 1_000.0,
        )


def process_complete_frame(
    frame: CapturedFrame,
    config: RadarCaptureConfig,
    display: DisplayPayloadSink,
    raw_writer: RawFrameWriter,
    emit_func: Optional[EmitFunc] = None,
) -> None:
    emit_func = emit_func or emit
    if not frame.is_valid:
        raw_writer.write_frame(frame)
        emit_func(
            "Dropped incomplete frame: "
            f"gap_bytes={frame.gap_bytes}, "
            f"bytes_per_frame={config.bytes_per_frame}"
        )
        return
    raw_writer.write_frame(frame)
    display.capture_diagnostics = dict(frame.capture_diagnostics or {})
    radar_cube = frame_bytes_to_radar_cube(frame.data, config)
    started = time.perf_counter()
    range_fft = compute_range_fft(radar_cube)
    range_fft_ms = (time.perf_counter() - started) * 1_000.0
    display.latest_range_fft_ms = range_fft_ms
    display.timings.add_ms("range_fft", range_fft_ms)
    range_axis = config.range_axis_m()
    if range_axis is None:
        raise ValueError("Mini4 profile has no physical range axis")
    display.update(
        range_fft,
        range_axis,
        captured_at_s=frame.first_byte_at_s,
    )


def _run_frame_processor(
    *,
    config: RadarCaptureConfig,
    pmm_config: PmmConfig,
    frame_queue: mp.Queue,
    log_queue: mp.Queue,
    processed_frames_counter: Any,
    display_payload_queue: Optional[mp.Queue],
    display_skipped_counter: Optional[Any],
    raw_output: Optional[Path],
    raw_metadata: Optional[Path],
    processed_output: Optional[Path],
    display_mode: str,
    display_update_every: int,
    startup_status_queue: mp.Queue,
) -> None:
    signal.signal(signal.SIGINT, signal.SIG_IGN)

    def worker_emit(message: str) -> None:
        _queue_emit(log_queue, message)

    raw_writer: Optional[RawFrameWriter] = None
    processed_writer: Optional[ProcessedOutputWriter] = None
    display: Optional[DisplayPayloadSink] = None
    ready_reported = False
    try:
        validate_mini4_profile(config)
        tracker = PmmTracker(config, pmm_config)
        raw_writer = RawFrameWriter(
            raw_output,
            raw_metadata,
            config,
            worker_emit,
        )
        processed_writer = ProcessedOutputWriter(
            processed_output,
            config,
            worker_emit,
            pmm_metadata=tracker.metadata,
        )
        display = DisplayPayloadSink(
            display_mode,
            display_update_every,
            display_payload_queue,
            config,
            tracker,
            processed_writer,
            display_skipped_counter,
        )
        worker_emit(
            "PMM tracking enabled: "
            f"calibration={pmm_config.background_calibration_seconds:g}s, "
            f"threshold={pmm_config.detection_threshold:g}, "
            f"folding={pmm_config.folding_size_min}-"
            f"{pmm_config.folding_size_max}, "
            f"history={pmm_config.history_seconds:g}s"
        )
        startup_status_queue.put(
            {"state": "ready", "message": "Radar frame processor ready."},
            timeout=1.0,
        )
        ready_reported = True
        while True:
            frame = frame_queue.get()
            if frame is None:
                break
            process_complete_frame(
                frame,
                config,
                display,
                raw_writer,
                worker_emit,
            )
            _increment_shared_counter(processed_frames_counter)
    except Exception as exc:
        message = f"Frame processor failed: {type(exc).__name__}: {exc}"
        worker_emit(message)
        if not ready_reported:
            try:
                startup_status_queue.put(
                    {"state": "failed", "message": message},
                    timeout=1.0,
                )
            except queue.Full:
                pass
    finally:
        if raw_writer is not None:
            raw_writer.close()
        if processed_writer is not None:
            processed_writer.close()
        if display is not None:
            worker_emit(
                "Processing timing summary: "
                + json.dumps(display.timings.summary(), separators=(",", ":"))
            )


def _image_levels(values: np.ndarray) -> tuple[float, float]:
    finite = np.asarray(values, dtype=np.float32)
    finite = finite[np.isfinite(finite)]
    if not finite.size:
        return 0.0, 1.0
    low, high = np.percentile(finite, (2.0, 99.5))
    if high <= low:
        high = low + 1.0
    return float(low), float(high)


class _PyQtGraphDisplay:
    """PyQtGraph widgets and latest-only queue polling for every GUI mode."""

    def __init__(
        self,
        *,
        mode: str,
        max_range_m: float,
        fov_deg: float,
        payload_queue: Any,
        stop_event: Any,
        rendered_updates: Any,
        pg: Any,
        gl: Any,
        QtCore: Any,
        QtWidgets: Any,
    ) -> None:
        self.mode = mode
        self.max_range_m = float(max_range_m)
        self.fov_deg = float(fov_deg)
        self.payload_queue = payload_queue
        self.stop_event = stop_event
        self.rendered_updates = rendered_updates
        self.pg = pg
        self.gl = gl
        self.QtCore = QtCore
        self.QtWidgets = QtWidgets
        self.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
        self.app.setApplicationName("Mini4 PMM Tracker")
        self.app.setQuitOnLastWindowClosed(True)
        self.window: Any = None
        self.range_curve: Any = None
        self.range_doppler_image: Any = None
        self.point_status: Any = None
        self.target_scatter: Any = None
        self.doppler_time_image: Any = None
        self._build_window()
        self.timer = QtCore.QTimer()
        self.timer.setInterval(20)
        self.timer.timeout.connect(self._poll)
        self.timer.start()

    def _build_window(self) -> None:
        if self.mode == "range":
            self.window = self.pg.GraphicsLayoutWidget(
                title="Mini4 PMM — Range"
            )
            plot = self.window.addPlot(title="Live Range Profile")
            plot.setLabel("bottom", "Range", units="m")
            plot.setLabel("left", "Linear magnitude")
            plot.setXRange(0.3, self.max_range_m, padding=0.0)
            plot.showGrid(x=True, y=True, alpha=0.25)
            self.range_curve = plot.plot(
                pen=self.pg.mkPen("#55d6ff", width=2)
            )
            self.window.resize(1000, 620)
        elif self.mode == "range-doppler":
            self.window = self.pg.GraphicsLayoutWidget(
                title="Mini4 PMM — Range–Doppler"
            )
            plot = self.window.addPlot(title="Live Range–Doppler")
            plot.setLabel("bottom", "Range", units="m")
            plot.setLabel("left", "Centered Doppler bin")
            plot.setXRange(0.3, self.max_range_m, padding=0.0)
            plot.setYRange(-32.0, 31.0, padding=0.0)
            self.range_doppler_image = self.pg.ImageItem(
                axisOrder="row-major"
            )
            self.range_doppler_image.setLookupTable(
                self.pg.colormap.get("viridis").getLookupTable(nPts=256)
            )
            plot.addItem(self.range_doppler_image)
            self.window.resize(1050, 650)
        elif self.mode == "point-cloud":
            self.window, _point_view = self._make_point_panel()
            self.window.setWindowTitle("Mini4 PMM — Target")
            self.window.resize(850, 700)
        elif self.mode == COMBINED_DISPLAY_MODE:
            self.window = self.QtWidgets.QWidget()
            self.window.setWindowTitle("Mini4 PMM — Combined")
            layout = self.QtWidgets.QHBoxLayout(self.window)
            point_panel, _point_view = self._make_point_panel()
            doppler_panel = self.pg.GraphicsLayoutWidget()
            doppler_plot = doppler_panel.addPlot(
                title="Target-gated Doppler–Time"
            )
            doppler_plot.setLabel("bottom", "History frame")
            doppler_plot.setLabel("left", "Centered Doppler bin")
            doppler_plot.setYRange(-32.0, 31.0, padding=0.0)
            self.doppler_time_image = self.pg.ImageItem(
                axisOrder="row-major"
            )
            self.doppler_time_image.setLookupTable(
                self.pg.colormap.get("viridis").getLookupTable(nPts=256)
            )
            doppler_plot.addItem(self.doppler_time_image)
            layout.addWidget(point_panel, 1)
            layout.addWidget(doppler_panel, 1)
            self.window.resize(1400, 720)
        else:
            raise ValueError(f"Unsupported PyQtGraph display mode: {self.mode}")
        self.window.show()

    def _make_point_panel(self) -> tuple[Any, Any]:
        panel = self.QtWidgets.QWidget()
        layout = self.QtWidgets.QVBoxLayout(panel)
        self.point_status = self.QtWidgets.QLabel("PMM target — starting")
        self.point_status.setAlignment(self.QtCore.Qt.AlignmentFlag.AlignCenter)
        self.point_status.setStyleSheet(
            "font-size: 16px; font-weight: 600; padding: 6px;"
        )
        view = self.gl.GLViewWidget()
        view.setBackgroundColor((15, 18, 24, 255))
        view.setCameraPosition(
            distance=max(self.max_range_m * 1.8, 5.0),
            elevation=20.0,
            azimuth=-90.0,
        )
        lateral_limit = max(
            self.max_range_m * np.sin(np.deg2rad(self.fov_deg)),
            1.0,
        )
        ground_grid = self.gl.GLGridItem()
        ground_grid.setSize(
            x=2.0 * lateral_limit,
            y=self.max_range_m,
            z=1.0,
        )
        ground_grid.setSpacing(x=2.0, y=2.0, z=1.0)
        ground_grid.translate(0.0, self.max_range_m / 2.0, 0.0)
        view.addItem(ground_grid)
        axes = self.gl.GLAxisItem()
        axes.setSize(
            x=lateral_limit,
            y=self.max_range_m,
            z=lateral_limit,
        )
        view.addItem(axes)
        self.target_scatter = self.gl.GLScatterPlotItem(
            pos=np.empty((0, 3), dtype=np.float32),
            color=(0.2, 1.0, 0.35, 1.0),
            size=18.0,
            pxMode=True,
        )
        view.addItem(self.target_scatter)
        layout.addWidget(self.point_status)
        layout.addWidget(view, 1)
        return panel, view

    def _poll(self) -> None:
        if self.stop_event.is_set():
            self.timer.stop()
            self.app.quit()
            return
        latest = None
        while True:
            try:
                latest = self.payload_queue.get_nowait()
            except queue.Empty:
                break
        if latest is None:
            return
        self._render(latest)
        _increment_shared_counter(self.rendered_updates)

    def _render(self, payload: Any) -> None:
        if self.mode == "range":
            range_axis, values = payload
            self.range_curve.setData(
                np.asarray(range_axis, dtype=np.float32),
                np.asarray(values, dtype=np.float32),
            )
            return
        if self.mode == "range-doppler":
            range_axis, heatmap = payload
            axis = np.asarray(range_axis, dtype=np.float32)
            image = np.asarray(heatmap, dtype=np.float32)
            self.range_doppler_image.setImage(
                image,
                autoLevels=False,
                levels=_image_levels(image),
            )
            if axis.size:
                bin_width = (
                    float(np.median(np.diff(axis))) if axis.size > 1 else 0.1
                )
                self.range_doppler_image.setRect(
                    self.QtCore.QRectF(
                        float(axis[0]) - bin_width / 2.0,
                        -32.5,
                        float(axis[-1] - axis[0]) + bin_width,
                        64.0,
                    )
                )
            return
        point = (
            payload.point_cloud
            if isinstance(payload, CombinedDisplayPayload)
            else payload
        )
        self.point_status.setText(point.tracking_status)
        positions = (
            np.asarray(
                [point.target_track.position_m],
                dtype=np.float32,
            )
            if point.target_track is not None
            else np.empty((0, 3), dtype=np.float32)
        )
        self.target_scatter.setData(
            pos=positions,
            color=(0.2, 1.0, 0.35, 1.0),
            size=18.0,
            pxMode=True,
        )
        if (
            isinstance(payload, CombinedDisplayPayload)
            and self.doppler_time_image is not None
        ):
            spectrum = np.asarray(payload.spectrogram_db, dtype=np.float32)
            if spectrum.size:
                self.doppler_time_image.setImage(
                    spectrum,
                    autoLevels=False,
                    levels=_image_levels(spectrum),
                )
                self.doppler_time_image.setRect(
                    self.QtCore.QRectF(
                        -0.5,
                        -32.5,
                        float(spectrum.shape[1]),
                        64.0,
                    )
                )

    def run(self) -> int:
        return int(self.app.exec())


def _run_display_process(
    mode: str,
    max_range_m: float,
    fov_deg: float,
    payload_queue: mp.Queue,
    stop_event: mp.Event,
    rendered_updates: Any,
    startup_status_queue: mp.Queue,
) -> None:
    try:
        import pyqtgraph as pg
        import pyqtgraph.opengl as gl
        from PySide6 import QtCore, QtWidgets

        pg.setConfigOptions(
            antialias=True,
            imageAxisOrder="row-major",
            background="#0f1218",
            foreground="#e6edf3",
        )
        display = _PyQtGraphDisplay(
            mode=mode,
            max_range_m=max_range_m,
            fov_deg=fov_deg,
            payload_queue=payload_queue,
            stop_event=stop_event,
            rendered_updates=rendered_updates,
            pg=pg,
            gl=gl,
            QtCore=QtCore,
            QtWidgets=QtWidgets,
        )
        startup_status_queue.put(
            {"state": "ready", "message": "PyQtGraph radar display ready."},
            timeout=1.0,
        )
        display.run()
    except Exception as exc:
        try:
            startup_status_queue.put(
                {
                    "state": "failed",
                    "message": f"Display failed: {type(exc).__name__}: {exc}",
                },
                timeout=1.0,
            )
        except queue.Full:
            pass


class LiveDisplay:
    def __init__(
        self,
        mode: str,
        max_range_m: float,
        fov_deg: float,
        process_context: Any,
    ) -> None:
        self.mode = mode
        self.payload_queue: Optional[mp.Queue] = None
        self.skipped_updates: Optional[Any] = None
        self.rendered_updates: Optional[Any] = None
        self.stop_event: Optional[Any] = None
        self.process: Optional[mp.Process] = None
        if mode == "none":
            return
        self.payload_queue = process_context.Queue(maxsize=1)
        self.skipped_updates = process_context.Value("Q", 0)
        self.rendered_updates = process_context.Value("Q", 0)
        self.stop_event = process_context.Event()
        status_queue = process_context.Queue(maxsize=1)
        self.process = process_context.Process(
            target=_run_display_process,
            args=(
                mode,
                max_range_m,
                fov_deg,
                self.payload_queue,
                self.stop_event,
                self.rendered_updates,
                status_queue,
            ),
            name="RadarLiveDisplay",
        )
        self.process.start()
        try:
            status = status_queue.get(timeout=30.0)
        except queue.Empty as exc:
            raise CaptureStartupError("Display did not become ready") from exc
        if status.get("state") != "ready":
            raise CaptureStartupError(status.get("message", "Display failed"))

    def close(self) -> None:
        if self.stop_event is not None:
            self.stop_event.set()
        if self.process is not None:
            self.process.join(timeout=3.0)
            if self.process.is_alive():
                self.process.terminate()
                self.process.join(timeout=1.0)


def listen_for_frames(
    *,
    host_ip: str,
    data_port: int,
    config: RadarCaptureConfig,
    setup_config: CaptureSetupConfig,
    buffer_size: int,
    socket_recv_buffer_bytes: int,
    socket_timeout_seconds: float,
    packet_queue_size: int,
    frame_queue: mp.Queue,
    log_queue: mp.Queue,
) -> CaptureStats:
    stats = CaptureStats()
    sequence_tracker = SequenceTracker(stats)
    frame_buffer = FrameBuffer(config.bytes_per_frame, stats)
    packet_queue: queue.Queue[tuple[bytes, float]] = queue.Queue(
        maxsize=max(packet_queue_size, 1)
    )
    synthetic_sequence = 0
    synthetic_byte_count = 0
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    if socket_recv_buffer_bytes > 0:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, socket_recv_buffer_bytes)
    sock.settimeout(socket_timeout_seconds)
    try:
        sock.bind((host_ip, data_port))
    except OSError as exc:
        sock.close()
        raise CaptureStartupError(
            f"Could not bind UDP data socket on {host_ip}:{data_port}: {exc}"
        ) from exc
    emit(
        "Listening for live radar stream "
        f"on {host_ip}:{data_port}; bytes_per_frame={config.bytes_per_frame}"
    )
    receiver = UdpPacketReceiver(sock, packet_queue, buffer_size, stats)
    receiver.start()
    try:
        while True:
            try:
                packet, received_at = packet_queue.get(
                    timeout=socket_timeout_seconds
                )
            except queue.Empty:
                _drain_log_queue(log_queue)
                if receiver.error is not None:
                    raise receiver.error
                continue
            packet_view = memoryview(packet)
            if setup_config.packet_sequence_enable:
                try:
                    header = DCA1000PacketHeader.parse(packet_view)
                except ValueError:
                    stats.malformed_packets += 1
                    continue
                payload = packet_view[DCA1000_HEADER_SIZE:]
                sequence_tracker.observe(header.sequence_number)
            else:
                header = DCA1000PacketHeader(
                    synthetic_sequence,
                    synthetic_byte_count,
                )
                payload = packet_view
                synthetic_sequence = (synthetic_sequence + 1) % UINT32_MODULO
                synthetic_byte_count += len(payload)
            stats.packets_received += 1
            for frame in frame_buffer.add_payload(header, payload, received_at):
                if not frame.is_valid:
                    continue
                try:
                    processing_occupancy = int(frame_queue.qsize())
                except (NotImplementedError, OSError):
                    processing_occupancy = -1
                frame = replace(
                    frame,
                    capture_diagnostics={
                        "packets_received": stats.packets_received,
                        "frames_emitted": stats.frames_emitted,
                        "invalid_frames": stats.invalid_frames,
                        "lost_packets": stats.lost_packets,
                        "receiver_queue_drops": stats.receiver_queue_drops,
                        "processing_frames_dropped": (
                            stats.processing_frames_dropped
                        ),
                        "packet_queue_occupancy": packet_queue.qsize(),
                        "processing_queue_occupancy": processing_occupancy,
                    },
                )
                try:
                    frame_queue.put_nowait(frame)
                except queue.Full:
                    stats.processing_frames_dropped += 1
            if stats.packets_received % 200 == 0:
                _drain_log_queue(log_queue)
    except KeyboardInterrupt:
        emit("Streaming stopped.")
    finally:
        receiver.stop()
        sock.close()
        receiver.join(timeout=max(socket_timeout_seconds + 0.5, 1.0))
        _drain_log_queue(log_queue)
        emit(_format_capture_summary(stats))
    return stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Receive DCA1000 ADC packets and track one PMM target."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--setup", type=Path, default=DEFAULT_SETUP_PATH)
    parser.add_argument("--host-ip", default=UDP_IP)
    parser.add_argument("--data-port", type=int, default=UDP_PORT)
    parser.add_argument("--buffer-size", type=int, default=BUFFER_SIZE)
    parser.add_argument(
        "--socket-recv-buffer",
        type=int,
        default=DEFAULT_SOCKET_RECV_BUFFER_BYTES,
    )
    parser.add_argument(
        "--packet-queue-size",
        type=int,
        default=DEFAULT_PACKET_QUEUE_SIZE,
    )
    parser.add_argument(
        "--processing-queue-size",
        type=int,
        default=DEFAULT_PROCESSING_QUEUE_SIZE,
    )
    parser.add_argument("--socket-timeout", type=float, default=SOCKET_TIMEOUT_SECONDS)
    parser.add_argument("--log-file", type=Path, default=DEFAULT_LOG_PATH)
    parser.add_argument("--raw-output", type=Path)
    parser.add_argument("--raw-metadata", type=Path)
    parser.add_argument("--processed-output", type=Path)
    parser.add_argument("--display", choices=DISPLAY_CHOICES, default="none")
    parser.add_argument("--display-update-every", type=int, default=1)
    parser.add_argument("--max-range-m", type=float, default=DEFAULT_MAX_RANGE_M)
    parser.add_argument(
        "--point-cloud-fov-deg",
        type=float,
        default=DEFAULT_POINT_CLOUD_FOV_DEG,
    )
    parser.add_argument(
        "--pmm-background-calibration-seconds",
        type=float,
        default=30.0,
    )
    parser.add_argument("--pmm-max-target-speed-m-s", type=float, default=4.0)
    parser.add_argument("--pmm-folding-size-min", type=int, default=2)
    parser.add_argument("--pmm-folding-size-max", type=int, default=20)
    parser.add_argument("--pmm-detection-threshold", type=float, default=30_000.0)
    parser.add_argument("--pmm-history-seconds", type=float, default=3.6)
    parser.add_argument("--pmm-provisional-frames", type=int, default=5)
    parser.add_argument("--pmm-confirmation-window-frames", type=int, default=10)
    parser.add_argument("--pmm-confirmation-hits", type=int, default=7)
    parser.add_argument("--pmm-coast-frames", type=int, default=10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_terminal_log(args.log_file)
    context = mp.get_context("spawn")
    frame_queue: Optional[mp.Queue] = None
    log_queue: Optional[mp.Queue] = None
    processor: Optional[mp.Process] = None
    display: Optional[LiveDisplay] = None
    try:
        config = RadarCaptureConfig.from_file(args.config)
        validate_mini4_profile(config)
        setup = CaptureSetupConfig.from_file(args.setup)
        pmm_config = PmmConfig(
            background_calibration_seconds=(
                args.pmm_background_calibration_seconds
            ),
            maximum_target_speed_m_s=args.pmm_max_target_speed_m_s,
            folding_size_min=args.pmm_folding_size_min,
            folding_size_max=args.pmm_folding_size_max,
            detection_threshold=args.pmm_detection_threshold,
            history_seconds=args.pmm_history_seconds,
            provisional_frames=args.pmm_provisional_frames,
            confirmation_window_frames=(
                args.pmm_confirmation_window_frames
            ),
            confirmation_hits=args.pmm_confirmation_hits,
            coast_frames=args.pmm_coast_frames,
        )
        emit(f"Loaded Mini4 profile: {config}")
        display = LiveDisplay(
            args.display,
            min(max(args.max_range_m, 0.3), 20.0),
            min(max(args.point_cloud_fov_deg, 0.0), 60.0),
            context,
        )
        frame_queue = context.Queue(maxsize=max(args.processing_queue_size, 1))
        log_queue = context.Queue(maxsize=1000)
        status_queue = context.Queue(maxsize=1)
        processed_counter = context.Value("Q", 0)
        processor = context.Process(
            target=_run_frame_processor,
            kwargs={
                "config": config,
                "pmm_config": pmm_config,
                "frame_queue": frame_queue,
                "log_queue": log_queue,
                "processed_frames_counter": processed_counter,
                "display_payload_queue": (
                    display.payload_queue if display is not None else None
                ),
                "display_skipped_counter": (
                    display.skipped_updates if display is not None else None
                ),
                "raw_output": args.raw_output,
                "raw_metadata": args.raw_metadata,
                "processed_output": args.processed_output,
                "display_mode": args.display,
                "display_update_every": args.display_update_every,
                "startup_status_queue": status_queue,
            },
            name="RadarFrameProcessor",
        )
        processor.start()
        try:
            status = status_queue.get(timeout=30.0)
        except queue.Empty as exc:
            raise CaptureStartupError("Frame processor did not become ready") from exc
        _drain_log_queue(log_queue)
        if status.get("state") != "ready":
            raise CaptureStartupError(status.get("message", "Processor failed"))
        emit(status["message"])
        listen_for_frames(
            host_ip=args.host_ip,
            data_port=args.data_port,
            config=config,
            setup_config=setup,
            buffer_size=args.buffer_size,
            socket_recv_buffer_bytes=max(args.socket_recv_buffer, 0),
            socket_timeout_seconds=max(args.socket_timeout, 0.05),
            packet_queue_size=max(args.packet_queue_size, 1),
            frame_queue=frame_queue,
            log_queue=log_queue,
        )
    except CaptureStartupError as exc:
        emit(f"Capture startup failed: {exc}")
    except ValueError as exc:
        emit(f"Capture configuration failed: {exc}")
    finally:
        if frame_queue is not None:
            try:
                frame_queue.put(None, timeout=1.0)
            except queue.Full:
                pass
        if processor is not None:
            processor.join(timeout=10.0)
            if processor.is_alive():
                processor.terminate()
                processor.join(timeout=1.0)
        if log_queue is not None:
            _drain_log_queue(log_queue)
        if display is not None:
            display.close()
        close_terminal_log()


def _config_from_cfg_lines(lines: Iterable[str]) -> RadarCaptureConfig:
    values: dict[str, Any] = {
        "iq_swap": False,
        "channel_interleave": False,
        "lvds_lanes": 2,
    }
    chirp_masks: dict[int, int] = {}
    for raw_line in lines:
        line = raw_line.split("%", 1)[0].split("#", 1)[0].strip()
        if not line:
            continue
        tokens = line.split()
        command = tokens[0]
        if command == "profileCfg" and len(tokens) >= 12:
            values.update(
                start_frequency_ghz=float(tokens[2]),
                idle_time_us=float(tokens[3]),
                adc_start_time_us=float(tokens[4]),
                ramp_end_time_us=float(tokens[5]),
                frequency_slope_mhz_per_us=float(tokens[8]),
                num_adc_samples=int(float(tokens[10])),
                sample_rate_ksps=float(tokens[11]),
            )
        elif command == "channelCfg" and len(tokens) >= 2:
            values["num_rx_channels"] = int(tokens[1], 0).bit_count()
        elif command == "frameCfg" and len(tokens) >= 6:
            values.update(
                chirp_start=int(tokens[1]),
                chirp_end=int(tokens[2]),
                num_loops=int(tokens[3]),
                frame_periodicity_ms=float(tokens[5]),
            )
        elif command == "chirpCfg" and len(tokens) >= 9:
            for chirp_index in range(int(tokens[1]), int(tokens[2]) + 1):
                chirp_masks[chirp_index] = int(tokens[8], 0)
        elif command == "adcbufCfg" and len(tokens) >= 5:
            values["iq_swap"] = bool(int(tokens[3]))
            values["channel_interleave"] = int(tokens[4]) == 0
        elif command in {"lvdsLaneCfg", "laneCfg"} and len(tokens) >= 2:
            values["lvds_lanes"] = int(tokens[1], 0).bit_count()
    required = (
        "num_adc_samples",
        "num_rx_channels",
        "chirp_start",
        "chirp_end",
        "num_loops",
    )
    missing = [name for name in required if name not in values]
    if missing:
        raise ValueError(f"Radar profile is missing: {', '.join(missing)}")
    chirps_per_loop = values["chirp_end"] - values["chirp_start"] + 1
    return RadarCaptureConfig.from_dimensions(
        num_adc_samples=values["num_adc_samples"],
        num_rx_channels=values["num_rx_channels"],
        num_chirps_per_frame=values["num_loops"] * chirps_per_loop,
        iq_swap=values["iq_swap"],
        channel_interleave=values["channel_interleave"],
        lvds_lanes=values["lvds_lanes"],
        num_loops=values["num_loops"],
        num_chirps_per_loop=chirps_per_loop,
        tx_channel_masks=tuple(
            chirp_masks.get(index, 0)
            for index in range(values["chirp_start"], values["chirp_end"] + 1)
        ),
        sample_rate_ksps=values.get("sample_rate_ksps"),
        frequency_slope_mhz_per_us=values.get(
            "frequency_slope_mhz_per_us"
        ),
        start_frequency_ghz=values.get("start_frequency_ghz"),
        idle_time_us=values.get("idle_time_us"),
        adc_start_time_us=values.get("adc_start_time_us"),
        ramp_end_time_us=values.get("ramp_end_time_us"),
        frame_periodicity_ms=values.get("frame_periodicity_ms"),
    )


def _put_latest_queue_payload(payload_queue: mp.Queue, payload: Any) -> int:
    skipped = 0
    try:
        payload_queue.put_nowait(payload)
        return skipped
    except queue.Full:
        pass
    try:
        payload_queue.get_nowait()
        skipped += 1
    except queue.Empty:
        pass
    try:
        payload_queue.put_nowait(payload)
    except queue.Full:
        skipped += 1
    return skipped


def _increment_shared_counter(counter: Optional[Any], amount: int = 1) -> None:
    if counter is None or amount <= 0:
        return
    with counter.get_lock():
        counter.value += amount


def _queue_emit(log_queue: mp.Queue, message: str) -> None:
    try:
        log_queue.put_nowait(message)
    except queue.Full:
        pass


def _drain_log_queue(log_queue: mp.Queue) -> None:
    while True:
        try:
            emit(log_queue.get_nowait())
        except queue.Empty:
            return


def _format_capture_summary(stats: CaptureStats) -> str:
    return (
        "Capture summary: "
        f"packets={stats.packets_received}, "
        f"frames={stats.frames_emitted}, "
        f"invalid_frames={stats.invalid_frames}, "
        f"lost_packets={stats.lost_packets}, "
        f"receiver_queue_drops={stats.receiver_queue_drops}, "
        f"processing_drops={stats.processing_frames_dropped}, "
        f"stream_resyncs={stats.stream_resyncs}"
    )


def _resolve_config_path(path: Path) -> Path:
    if path.exists():
        return path
    if not path.is_absolute():
        beside_script = Path(__file__).resolve().parent / path.name
        if beside_script.exists():
            return beside_script
    raise FileNotFoundError(f"Config file not found: {path}")


def _resolve_output_path(path: Optional[Path]) -> Optional[Path]:
    if path is None or path.is_absolute():
        return path
    return Path.cwd() / path


def setup_terminal_log(path: Path) -> None:
    global _LOG_FILE
    resolved = _resolve_output_path(path)
    assert resolved is not None
    resolved.parent.mkdir(parents=True, exist_ok=True)
    _LOG_FILE = resolved.open("a", encoding="utf-8", buffering=1)


def close_terminal_log() -> None:
    global _LOG_FILE
    if _LOG_FILE is not None:
        _LOG_FILE.close()
        _LOG_FILE = None


def emit(message: str) -> None:
    print(message, flush=True)
    if _LOG_FILE is not None:
        _LOG_FILE.write(message + "\n")


def _timestamp() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


if __name__ == "__main__":
    main()
