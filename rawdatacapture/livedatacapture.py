import argparse
import json
import multiprocessing as mp
import queue
import socket
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, BinaryIO, Callable, Iterable, Optional, TextIO

import numpy as np

try:
    from .dsp import (
        compute_range_doppler_heatmap,
        compute_point_cloud,
        compute_range_fft,
        compute_range_profile,
        frame_bytes_to_radar_cube,
        range_axis_m,
        range_resolution_m,
        validate_openradar_backend,
    )
except ImportError:
    from dsp import (
        compute_range_doppler_heatmap,
        compute_point_cloud,
        compute_range_fft,
        compute_range_profile,
        frame_bytes_to_radar_cube,
        range_axis_m,
        range_resolution_m,
        validate_openradar_backend,
    )


# Default DCA1000 network parameters.
UDP_IP = "192.168.33.30"  # Host/laptop static Ethernet IP
UDP_PORT = 4098           # DCA1000 raw ADC data port
BUFFER_SIZE = 65535       # Max UDP packet payload allocation

DCA1000_HEADER_SIZE = 10
UINT32_MODULO = 2**32
SOCKET_TIMEOUT_SECONDS = 0.5
DEFAULT_PROCESSING_QUEUE_SIZE = 4
DEFAULT_LOG_PATH = Path(__file__).with_suffix(".log")
DEFAULT_CONFIG_PATH = Path(__file__).with_name("mmwave.json")
DEFAULT_SETUP_PATH = Path(__file__).with_name("setup.json")
DEFAULT_MAX_RANGE_M = 10.0
DEFAULT_POINT_CLOUD_FOV_DEG = 60.0
POINT_CLOUD_MAGNITUDE_DB_MIN = 40.0
POINT_CLOUD_MAGNITUDE_DB_MAX = 120.0
DEFAULT_SOCKET_RECV_BUFFER_BYTES = 4 * 1024 * 1024
_LOG_FILE: Optional[TextIO] = None
EmitFunc = Callable[[str], None]


class CaptureStartupError(RuntimeError):
    """Raised for expected startup failures that should not print a traceback."""


@dataclass(frozen=True)
class DCA1000PacketHeader:
    """DCA1000 inline packet header: uint32 sequence + uint48 byte count."""

    sequence_number: int
    byte_count: int

    @classmethod
    def parse(cls, packet: bytes) -> "DCA1000PacketHeader":
        if len(packet) < DCA1000_HEADER_SIZE:
            raise ValueError(
                f"DCA1000 packet is too short for a {DCA1000_HEADER_SIZE}-byte header"
            )

        return cls(
            sequence_number=int.from_bytes(packet[0:4], byteorder="little", signed=False),
            byte_count=int.from_bytes(packet[4:10], byteorder="little", signed=False),
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
    processing_frames_dropped: int = 0


@dataclass(frozen=True)
class CapturedFrame:
    data: bytes
    gap_bytes: int
    first_byte_at_s: float

    @property
    def is_valid(self) -> bool:
        return self.gap_bytes == 0


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

        previous_sequence = (self.expected_sequence - 1) % UINT32_MODULO
        if sequence_number == previous_sequence:
            self.stats.duplicate_packets += 1
            return

        delta = (sequence_number - self.expected_sequence) % UINT32_MODULO
        if 0 < delta < (UINT32_MODULO // 2):
            self.stats.lost_packets += delta
            self.expected_sequence = (sequence_number + 1) % UINT32_MODULO
            return

        self.stats.out_of_order_packets += 1


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
    ) -> "RadarCaptureConfig":
        bytes_per_complex_sample = 4  # int16 I + int16 Q
        bytes_per_frame = (
            num_adc_samples
            * num_rx_channels
            * num_chirps_per_frame
            * bytes_per_complex_sample
        )
        return cls(
            num_adc_samples=num_adc_samples,
            num_rx_channels=num_rx_channels,
            num_chirps_per_frame=num_chirps_per_frame,
            bytes_per_frame=bytes_per_frame,
            iq_swap=iq_swap,
            channel_interleave=channel_interleave,
            lvds_lanes=lvds_lanes,
            num_loops=num_loops,
            num_chirps_per_loop=num_chirps_per_loop,
            tx_channel_masks=(
                tuple(int(mask) for mask in tx_channel_masks)
                if tx_channel_masks is not None
                else None
            ),
            sample_rate_ksps=sample_rate_ksps,
            frequency_slope_mhz_per_us=frequency_slope_mhz_per_us,
        )

    @classmethod
    def from_file(cls, config_path: Path) -> "RadarCaptureConfig":
        config_path = _resolve_config_path(config_path)
        suffix = config_path.suffix.lower()
        if suffix == ".cfg":
            return cls.from_mmwave_cfg(config_path)
        if suffix == ".json":
            return cls.from_mmwave_json(config_path)
        raise ValueError(f"Unsupported config file extension: {config_path.suffix}")

    @classmethod
    def from_mmwave_cfg(cls, config_path: Path) -> "RadarCaptureConfig":
        return _config_from_cfg_lines(config_path.read_text(encoding="utf-8").splitlines())

    @classmethod
    def from_mmwave_json(cls, config_path: Path) -> "RadarCaptureConfig":
        data = json.loads(config_path.read_text(encoding="utf-8"))

        studio_config = _config_from_mmwave_studio_json(data)
        if studio_config is not None:
            return studio_config

        command_config = _config_from_json_command_lines(data)
        if command_config is not None:
            return command_config

        return _config_from_mapping(data, source_name="JSON")

    @property
    def range_resolution_m(self) -> Optional[float]:
        return range_resolution_m(self)

    def range_axis_m(self) -> Optional[np.ndarray]:
        return range_axis_m(self)


@dataclass(frozen=True)
class CaptureSetupConfig:
    packet_sequence_enable: bool
    packet_delay_us: Optional[int] = None
    capture_hardware: Optional[str] = None
    config_used: Optional[str] = None

    @classmethod
    def from_file(cls, setup_path: Path) -> "CaptureSetupConfig":
        setup_path = _resolve_config_path(setup_path)
        data = json.loads(setup_path.read_text(encoding="utf-8"))
        dca_config = data.get("DCA1000Config", {}) if isinstance(data, dict) else {}

        packet_sequence_value = (
            _optional_int(dca_config, "packetSequenceEnable")
            if isinstance(dca_config, dict)
            else None
        )
        packet_sequence_enable = (
            True if packet_sequence_value is None else bool(packet_sequence_value)
        )
        packet_delay_us = (
            _optional_int(dca_config, "packetDelay_us")
            if isinstance(dca_config, dict)
            else None
        )
        capture_hardware = _optional_string(data, "captureHardware")
        config_used = _optional_string(data, "configUsed")

        return cls(
            packet_sequence_enable=packet_sequence_enable,
            packet_delay_us=packet_delay_us,
            capture_hardware=capture_hardware,
            config_used=config_used,
        )


class FrameBuffer:
    def __init__(self, bytes_per_frame: int, stats: CaptureStats) -> None:
        self.bytes_per_frame = bytes_per_frame
        self.stats = stats
        self.base_byte_count: Optional[int] = None
        self.next_stream_offset = 0
        self.buffer = bytearray()
        self.gap_markers: list[tuple[int, int]] = []
        self.arrival_markers: list[tuple[int, int, float]] = []

    def add_payload(
        self,
        header: DCA1000PacketHeader,
        payload: bytes,
        payload_received_at_s: float,
    ) -> list[CapturedFrame]:
        if self.base_byte_count is None:
            self.base_byte_count = header.byte_count

        payload_start = header.byte_count - self.base_byte_count
        if payload_start < 0:
            self.stats.out_of_order_packets += 1
            return []

        payload_end = payload_start + len(payload)
        if payload_end <= self.next_stream_offset:
            self.stats.duplicate_packets += 1
            return []

        if payload_start > self.next_stream_offset:
            gap = payload_start - self.next_stream_offset
            if gap > self.bytes_per_frame:
                self._resynchronize(header.byte_count)
                payload_start = 0
                gap = 0
            gap_start = len(self.buffer)
            if gap:
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
            payload_buffer_start = len(self.buffer)
            self.buffer.extend(payload)
            self.arrival_markers.append(
                (
                    payload_buffer_start,
                    payload_buffer_start + len(payload),
                    payload_received_at_s,
                )
            )
            self.next_stream_offset += len(payload)

        frames: list[CapturedFrame] = []
        while len(self.buffer) >= self.bytes_per_frame:
            gap_bytes = self._gap_bytes_in_next_frame()
            first_byte_at_s = self._first_byte_time_in_next_frame()
            frames.append(
                CapturedFrame(
                    data=bytes(self.buffer[: self.bytes_per_frame]),
                    gap_bytes=gap_bytes,
                    first_byte_at_s=first_byte_at_s,
                )
            )
            if gap_bytes:
                self.stats.invalid_frames += 1
            del self.buffer[: self.bytes_per_frame]
            self._advance_frame_markers(self.bytes_per_frame)
            self.stats.frames_emitted += 1

        return frames

    def _resynchronize(self, byte_count: int) -> None:
        """Discard partial assembly after an implausibly large stream gap."""
        self.base_byte_count = byte_count
        self.next_stream_offset = 0
        self.buffer.clear()
        self.gap_markers.clear()
        self.arrival_markers.clear()
        self.stats.stream_resyncs += 1

    def _gap_bytes_in_next_frame(self) -> int:
        frame_end = self.bytes_per_frame
        total = 0
        for start, end in self.gap_markers:
            overlap_start = max(start, 0)
            overlap_end = min(end, frame_end)
            if overlap_end > overlap_start:
                total += overlap_end - overlap_start
        return total

    def _first_byte_time_in_next_frame(self) -> float:
        frame_end = self.bytes_per_frame
        for start, end, received_at_s in self.arrival_markers:
            if end > 0 and start < frame_end:
                return received_at_s
        return time.perf_counter()

    def _advance_frame_markers(self, consumed_bytes: int) -> None:
        advanced_markers = []
        for start, end in self.gap_markers:
            start -= consumed_bytes
            end -= consumed_bytes
            if end > 0:
                advanced_markers.append((max(start, 0), end))
        self.gap_markers = advanced_markers

        advanced_arrival_markers = []
        for start, end, received_at_s in self.arrival_markers:
            start -= consumed_bytes
            end -= consumed_bytes
            if end > 0:
                advanced_arrival_markers.append((max(start, 0), end, received_at_s))
        self.arrival_markers = advanced_arrival_markers


class LiveDisplay:
    def __init__(
        self,
        mode: str,
        pause_seconds: float,
        max_range_m: float,
        point_cloud_range_m: Optional[float] = None,
        point_cloud_fov_deg: float = DEFAULT_POINT_CLOUD_FOV_DEG,
        emit_func: Optional[EmitFunc] = None,
    ) -> None:
        self.mode = mode
        self.pause_seconds = max(pause_seconds, 0.001)
        self.max_range_m = max(max_range_m, 0.0)
        self.point_cloud_range_m = max(
            point_cloud_range_m
            if point_cloud_range_m is not None
            else self.max_range_m,
            1.0,
        )
        self.point_cloud_fov_deg = min(max(float(point_cloud_fov_deg), 0.0), 90.0)
        self.emit = emit_func or emit
        self.stop_event: Optional[mp.Event] = None
        self.payload_queue: Optional[mp.Queue] = None
        self.latency_queue: Optional[mp.Queue] = None
        self.process: Optional[mp.Process] = None

        if self.mode == "none":
            return

        self.stop_event = mp.Event()
        self.payload_queue = mp.Queue(maxsize=1)
        self.latency_queue = mp.Queue(maxsize=100)
        self.process = mp.Process(
            target=_run_display_process,
            args=(
                self.mode,
                self.pause_seconds,
                self.max_range_m,
                self.point_cloud_range_m,
                self.point_cloud_fov_deg,
                self.payload_queue,
                self.latency_queue,
                self.stop_event,
            ),
            name="RadarLiveDisplay",
            daemon=True,
        )
        self.process.start()

    def close(self) -> None:
        self.emit_latency_messages()
        if self.stop_event is not None:
            self.stop_event.set()
        if self.process is not None:
            self.process.join(timeout=2.0)
            if self.process.is_alive():
                self.process.terminate()
                self.process.join(timeout=1.0)
        self.emit_latency_messages()
        if self.payload_queue is not None:
            self.payload_queue.close()
            self.payload_queue.join_thread()
        if self.latency_queue is not None:
            self.latency_queue.close()
            self.latency_queue.join_thread()

    def emit_latency_messages(self) -> None:
        if self.latency_queue is None:
            return

        while True:
            try:
                self.emit(self.latency_queue.get_nowait())
            except queue.Empty:
                return


class DisplayPayloadSink:
    def __init__(
        self,
        mode: str,
        update_every: int,
        payload_queue: Optional[mp.Queue],
        config: RadarCaptureConfig,
        max_range_m: float = DEFAULT_MAX_RANGE_M,
        point_cloud_fov_deg: float = DEFAULT_POINT_CLOUD_FOV_DEG,
    ) -> None:
        self.mode = mode
        self.update_every = max(update_every, 1)
        self.payload_queue = payload_queue
        self.config = config
        self.max_range_m = max(float(max_range_m), 0.0)
        self.point_cloud_fov_deg = min(max(float(point_cloud_fov_deg), 0.0), 90.0)
        self.frame_count = 0

    def update(
        self,
        range_fft: np.ndarray,
        range_axis_m: Optional[np.ndarray],
        frame_first_byte_at_s: float,
    ) -> None:
        if self.mode == "none" or self.payload_queue is None:
            return

        self.frame_count += 1
        if self.frame_count % self.update_every != 0:
            return

        if self.mode == "range":
            payload = (
                frame_first_byte_at_s,
                range_axis_m,
                compute_range_profile(range_fft),
            )
        elif self.mode == "range-doppler":
            payload = (
                frame_first_byte_at_s,
                range_axis_m,
                compute_range_doppler_heatmap(range_fft, self.config),
            )
        elif self.mode == "point-cloud":
            payload = (
                frame_first_byte_at_s,
                compute_point_cloud(
                    range_fft,
                    range_axis_m,
                    self.config,
                    max_range_m=self.max_range_m,
                    azimuth_fov_deg=self.point_cloud_fov_deg,
                    elevation_fov_deg=self.point_cloud_fov_deg,
                ),
            )
        else:
            return

        _put_latest_queue_payload(self.payload_queue, payload)


def _put_latest_queue_payload(payload_queue: mp.Queue, payload: Any) -> None:
    try:
        payload_queue.put_nowait(payload)
        return
    except queue.Full:
        pass

    try:
        payload_queue.get_nowait()
    except queue.Empty:
        pass

    try:
        payload_queue.put_nowait(payload)
    except queue.Full:
        pass


class RawFrameWriter:
    def __init__(
        self,
        output_path: Optional[Path],
        metadata_path: Optional[Path],
        config: RadarCaptureConfig,
        emit_func: Optional[EmitFunc] = None,
    ) -> None:
        self.config = config
        self.emit = emit_func or emit
        self.output_path = _resolve_output_path(output_path) if output_path else None
        self.metadata_path = (
            _resolve_output_path(metadata_path)
            if metadata_path
            else _default_metadata_path(self.output_path)
        )
        self.file: Optional[BinaryIO] = None
        self.frames_saved = 0
        self.bytes_saved = 0
        self.invalid_frames_skipped = 0
        self.started_at: Optional[str] = None

        if self.output_path is None:
            return

        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.file = self.output_path.open("wb")
        self.started_at = _timestamp()
        self.emit(f"Saving valid raw frames to {self.output_path}")
        if self.metadata_path is not None:
            self.emit(f"Raw capture metadata will be written to {self.metadata_path}")

    @property
    def enabled(self) -> bool:
        return self.file is not None

    def write_frame(self, frame: CapturedFrame) -> None:
        if not self.enabled:
            return
        if not frame.is_valid:
            self.invalid_frames_skipped += 1
            return

        assert self.file is not None
        self.file.write(frame.data)
        self.frames_saved += 1
        self.bytes_saved += len(frame.data)

    def close(self) -> None:
        if self.file is None:
            return

        self.file.close()
        self.file = None
        self.emit(
            "Raw capture saved "
            f"frames={self.frames_saved}, bytes={self.bytes_saved}"
        )
        self._write_metadata()

    def _write_metadata(self) -> None:
        if self.output_path is None or self.metadata_path is None:
            return

        self.metadata_path.parent.mkdir(parents=True, exist_ok=True)
        metadata = {
            "created_at": self.started_at,
            "closed_at": _timestamp(),
            "raw_output_path": str(self.output_path),
            "frames_saved": self.frames_saved,
            "bytes_saved": self.bytes_saved,
            "bytes_per_frame": self.config.bytes_per_frame,
            "invalid_frames_skipped": self.invalid_frames_skipped,
            "frame_shape": {
                "num_chirps_per_frame": self.config.num_chirps_per_frame,
                "num_loops": self.config.num_loops,
                "num_chirps_per_loop": self.config.num_chirps_per_loop,
                "tx_channel_masks": self.config.tx_channel_masks,
                "num_rx_channels": self.config.num_rx_channels,
                "num_adc_samples": self.config.num_adc_samples,
            },
            "sample_format": {
                "adc_bits": 16,
                "complex": True,
                "i_dtype": "int16",
                "q_dtype": "int16",
                "byte_order": "little",
                "iq_swap": self.config.iq_swap,
                "channel_interleave": self.config.channel_interleave,
                "lvds_lanes": self.config.lvds_lanes,
            },
            "range_processing": {
                "sample_rate_ksps": self.config.sample_rate_ksps,
                "frequency_slope_mhz_per_us": self.config.frequency_slope_mhz_per_us,
                "range_resolution_m": self.config.range_resolution_m,
            },
        }
        self.metadata_path.write_text(
            json.dumps(metadata, indent=2) + "\n",
            encoding="utf-8",
        )
        self.emit(f"Raw capture metadata saved to {self.metadata_path}")


def _run_display_process(
    mode: str,
    pause_seconds: float,
    max_range_m: float,
    point_cloud_range_m: float,
    point_cloud_fov_deg: float,
    payload_queue: mp.Queue,
    latency_queue: mp.Queue,
    stop_event: mp.Event,
) -> None:
    try:
        import signal

        signal.signal(signal.SIGINT, signal.SIG_IGN)
    except Exception:
        pass

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("Live display disabled: matplotlib is not installed.")
        return

    plt.ion()
    figure = plt.figure()
    if mode == "point-cloud":
        axis = figure.add_subplot(111, projection="3d")
    else:
        axis = figure.add_subplot(111)
    line = None
    image = None
    scatter = None

    if mode == "range":
        line = axis.plot([], [], lw=1.5)[0]
        axis.set_title("Live Range Profile")
        axis.set_xlabel("Range (m)")
        axis.set_ylabel("Magnitude")
        axis.grid(True, alpha=0.3)
    elif mode == "range-doppler":
        image = axis.imshow(
            np.zeros((1, 1)),
            aspect="auto",
            origin="lower",
            interpolation="nearest",
        )
        figure.colorbar(image, ax=axis, label="Magnitude (dB)")
        axis.set_title("Live Range-Doppler Heatmap")
        axis.set_xlabel("Range (m)")
        axis.set_ylabel("Doppler bin")
    elif mode == "point-cloud":
        scatter = axis.scatter([], [], [], c=[], s=18, cmap="viridis")
        scatter.set_clim(POINT_CLOUD_MAGNITUDE_DB_MIN, POINT_CLOUD_MAGNITUDE_DB_MAX)
        figure.colorbar(scatter, ax=axis, label="Magnitude (dB)", pad=0.12)
        axis.set_title(
            "Live 3D Point Cloud "
            f"(±{point_cloud_fov_deg:g}° FOV, {point_cloud_range_m:g} m)"
        )
        axis.set_xlabel("X left/right (m)")
        axis.set_ylabel("Y forward (m)")
        axis.set_zlabel("Z elevation (m)")
        _set_point_cloud_axes(axis, point_cloud_range_m, point_cloud_fov_deg)
        axis.view_init(elev=24, azim=-60)
        axis.grid(True, alpha=0.3)
    figure.tight_layout()
    plt.show(block=False)

    try:
        while not stop_event.is_set():
            try:
                payload = payload_queue.get(timeout=pause_seconds)
                while True:
                    try:
                        payload = payload_queue.get_nowait()
                    except queue.Empty:
                        break

                if mode == "range" and line is not None:
                    frame_first_byte_at_s, range_axis_m, range_profile = payload
                    _draw_range_profile(
                        axis,
                        line,
                        range_axis_m,
                        range_profile,
                        max_range_m,
                    )
                elif mode == "range-doppler" and image is not None:
                    frame_first_byte_at_s, range_axis_m, heatmap = payload
                    _draw_range_doppler(axis, image, range_axis_m, heatmap, max_range_m)
                elif mode == "point-cloud" and scatter is not None:
                    frame_first_byte_at_s, points = payload
                    _draw_point_cloud(
                        axis,
                        scatter,
                        points,
                        point_cloud_range_m,
                        point_cloud_fov_deg,
                    )
                else:
                    frame_first_byte_at_s = None

                figure.canvas.draw()
                if frame_first_byte_at_s is not None:
                    display_latency_ms = (
                        time.perf_counter() - frame_first_byte_at_s
                    ) * 1000.0
                    latency_message = (
                        "display latency: "
                        f"frame_first_byte_to_canvas_draw_ms={display_latency_ms:.1f}"
                    )
                    try:
                        latency_queue.put_nowait(latency_message)
                    except queue.Full:
                        pass
            except queue.Empty:
                pass
            except KeyboardInterrupt:
                break

            try:
                plt.pause(pause_seconds)
            except KeyboardInterrupt:
                break
    finally:
        plt.close(figure)


def _draw_range_profile(
    axis: Any,
    line: Any,
    range_axis_m: Optional[np.ndarray],
    range_profile: np.ndarray,
    max_range_m: float,
) -> None:
    x_axis = _range_plot_axis(range_axis_m, range_profile.size)
    line.set_data(x_axis, range_profile)
    axis.set_xlim(float(x_axis[0]), _range_plot_xmax(x_axis, max_range_m))
    profile_max = float(np.max(range_profile)) if range_profile.size else 1.0
    axis.set_ylim(0, max(profile_max * 1.1, 1.0))


def _draw_range_doppler(
    axis: Any,
    image: Any,
    range_axis_m: Optional[np.ndarray],
    heatmap: np.ndarray,
    max_range_m: float,
) -> None:
    x_axis = _range_plot_axis(range_axis_m, heatmap.shape[1])
    image.set_data(heatmap)
    image.set_extent((float(x_axis[0]), float(x_axis[-1]), 0, heatmap.shape[0] - 1))
    image.set_clim(float(np.min(heatmap)), float(np.max(heatmap)))
    axis.set_xlim(float(x_axis[0]), _range_plot_xmax(x_axis, max_range_m))
    axis.set_ylim(0, max(heatmap.shape[0] - 1, 1))


def _draw_point_cloud(
    axis: Any,
    scatter: Any,
    points: np.ndarray,
    point_cloud_range_m: float,
    point_cloud_fov_deg: float,
) -> None:
    empty = np.array([], dtype=np.float32)
    if points.size == 0:
        scatter._offsets3d = (empty, empty, empty)
        scatter.set_array(empty)
    else:
        scatter._offsets3d = (points[:, 0], points[:, 1], points[:, 2])
        scatter.set_array(points[:, 3])
        scatter.set_clim(
            POINT_CLOUD_MAGNITUDE_DB_MIN,
            POINT_CLOUD_MAGNITUDE_DB_MAX,
        )

    _set_point_cloud_axes(axis, point_cloud_range_m, point_cloud_fov_deg)


def _set_point_cloud_axes(
    axis: Any,
    point_cloud_range_m: float,
    point_cloud_fov_deg: float,
) -> None:
    range_limit_m = max(float(point_cloud_range_m), 1.0)
    fov_deg = min(max(float(point_cloud_fov_deg), 0.0), 90.0)
    cross_range_limit_m = max(range_limit_m * np.sin(np.deg2rad(fov_deg)), 0.5)
    axis.set_xlim(-cross_range_limit_m, cross_range_limit_m)
    axis.set_ylim(0.0, range_limit_m)
    axis.set_zlim(-cross_range_limit_m, cross_range_limit_m)
    axis.set_box_aspect(
        (2.0 * cross_range_limit_m, range_limit_m, 2.0 * cross_range_limit_m)
    )


def _point_cloud_range_limit_m(config: RadarCaptureConfig) -> Optional[float]:
    """Return a one-bin-padded limit that contains every DSP range bin."""
    resolution_m = config.range_resolution_m
    if resolution_m is None:
        return None
    return config.num_adc_samples * resolution_m


def _range_plot_axis(
    range_axis_m: Optional[np.ndarray],
    fallback_size: int,
) -> np.ndarray:
    if range_axis_m is not None and range_axis_m.size == fallback_size:
        return range_axis_m
    return np.arange(fallback_size, dtype=np.float32)


def _range_plot_xmax(x_axis: np.ndarray, max_range_m: float) -> float:
    if max_range_m > 0:
        return max(max_range_m, float(x_axis[0] + 1.0))
    return float(max(x_axis[-1], x_axis[0] + 1.0))


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
            "Dropped frame: incomplete payload, "
            f"gap_bytes={frame.gap_bytes}, bytes_per_frame={config.bytes_per_frame}"
        )
        return

    raw_writer.write_frame(frame)
    radar_cube = frame_bytes_to_radar_cube(frame.data, config)
    range_fft = compute_range_fft(radar_cube)
    range_profile = compute_range_profile(range_fft)
    peak_range_bin = int(np.argmax(range_profile))
    range_axis_m = config.range_axis_m()
    peak_range_m = (
        float(range_axis_m[peak_range_bin]) if range_axis_m is not None else None
    )
    peak_range_text = (
        f"peak_range_m={peak_range_m:.3f}"
        if peak_range_m is not None
        else f"peak_range_bin={peak_range_bin}"
    )

    emit_func(
        "Complete frame "
        f"cube_shape={radar_cube.shape}, "
        f"{peak_range_text}, "
        f"peak_range_bin={peak_range_bin}, "
        f"peak_magnitude={range_profile[peak_range_bin]:.2f}"
    )
    display.update(range_fft, range_axis_m, frame.first_byte_at_s)


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


def _request_processor_stop(frame_queue: mp.Queue) -> None:
    try:
        frame_queue.put(None, timeout=0.2)
        return
    except queue.Full:
        pass

    try:
        frame_queue.get(timeout=0.2)
    except queue.Empty:
        pass

    try:
        frame_queue.put(None, timeout=0.2)
    except queue.Full:
        pass


def _run_frame_processor(
    config: RadarCaptureConfig,
    frame_queue: mp.Queue,
    log_queue: mp.Queue,
    display_payload_queue: Optional[mp.Queue],
    raw_output: Optional[Path],
    raw_metadata: Optional[Path],
    display_mode: str,
    display_update_every: int,
    max_range_m: float,
    point_cloud_fov_deg: float,
) -> None:
    def worker_emit(message: str) -> None:
        _queue_emit(log_queue, message)

    raw_writer = RawFrameWriter(raw_output, raw_metadata, config, worker_emit)
    display = DisplayPayloadSink(
        display_mode,
        display_update_every,
        display_payload_queue,
        config,
        max_range_m,
        point_cloud_fov_deg,
    )

    try:
        while True:
            frame = frame_queue.get()
            if frame is None:
                break

            process_complete_frame(frame, config, display, raw_writer, worker_emit)
    except KeyboardInterrupt:
        pass
    except Exception as exc:
        worker_emit(f"Frame processor stopped after error: {exc!r}")
    finally:
        raw_writer.close()


def listen_for_frames(
    *,
    host_ip: str,
    data_port: int,
    config: RadarCaptureConfig,
    setup_config: CaptureSetupConfig,
    buffer_size: int,
    socket_recv_buffer_bytes: int,
    socket_timeout_seconds: float,
    frame_queue: mp.Queue,
    log_queue: mp.Queue,
    display: LiveDisplay,
) -> CaptureStats:
    stats = CaptureStats()
    sequence_tracker = SequenceTracker(stats)
    frame_buffer = FrameBuffer(config.bytes_per_frame, stats)
    synthetic_sequence_number = 0
    synthetic_byte_count = 0

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    if socket_recv_buffer_bytes > 0:
        try:
            sock.setsockopt(
                socket.SOL_SOCKET,
                socket.SO_RCVBUF,
                socket_recv_buffer_bytes,
            )
        except OSError as exc:
            emit(
                "Could not set UDP receive buffer to "
                f"{socket_recv_buffer_bytes} bytes: {exc}"
            )

    actual_recv_buffer_bytes = sock.getsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF)
    sock.settimeout(socket_timeout_seconds)
    try:
        sock.bind((host_ip, data_port))
    except OSError as exc:
        sock.close()
        raise CaptureStartupError(
            "Could not bind UDP data socket to "
            f"{host_ip}:{data_port}: {exc}. "
            "Check that this IP address is assigned to your PC Ethernet adapter, "
            "or pass the correct adapter IP with --host-ip."
        ) from exc

    emit(
        "Listening for live radar stream "
        f"on {host_ip}:{data_port}; bytes_per_frame={config.bytes_per_frame}"
    )
    emit(
        "DCA1000 setup: "
        f"packet_sequence_enable={setup_config.packet_sequence_enable}, "
        f"packet_delay_us={setup_config.packet_delay_us}"
    )
    emit(
        "UDP socket receive buffer: "
        f"requested={socket_recv_buffer_bytes}B, actual={actual_recv_buffer_bytes}B"
    )
    emit("Trigger frames now. Press Ctrl+C to stop.")

    try:
        while True:
            try:
                packet, _addr = sock.recvfrom(buffer_size)
                packet_received_at_s = time.perf_counter()
            except socket.timeout:
                display.emit_latency_messages()
                _drain_log_queue(log_queue)
                continue

            if setup_config.packet_sequence_enable:
                try:
                    header = DCA1000PacketHeader.parse(packet)
                except ValueError:
                    stats.malformed_packets += 1
                    continue

                payload = packet[DCA1000_HEADER_SIZE:]
                sequence_tracker.observe(header.sequence_number)
            else:
                payload = packet
                header = DCA1000PacketHeader(
                    sequence_number=synthetic_sequence_number,
                    byte_count=synthetic_byte_count,
                )
                synthetic_sequence_number = (
                    synthetic_sequence_number + 1
                ) % UINT32_MODULO
                synthetic_byte_count += len(payload)

            stats.packets_received += 1

            for frame in frame_buffer.add_payload(
                header,
                payload,
                packet_received_at_s,
            ):
                if not frame.is_valid:
                    emit(
                        "Dropped frame: incomplete payload, "
                        f"gap_bytes={frame.gap_bytes}, "
                        f"bytes_per_frame={config.bytes_per_frame}"
                    )
                    continue

                try:
                    frame_queue.put_nowait(frame)
                except queue.Full:
                    stats.processing_frames_dropped += 1
                    emit(
                        "Dropped frame: processing queue full, "
                        f"bytes_per_frame={config.bytes_per_frame}"
                    )

            if stats.packets_received % 200 == 0:
                display.emit_latency_messages()
                _drain_log_queue(log_queue)
                emit(_format_stats(stats))
    except KeyboardInterrupt:
        display.emit_latency_messages()
        _drain_log_queue(log_queue)
        emit("Streaming stopped.")
        emit(_format_stats(stats))
    finally:
        sock.close()
        display.emit_latency_messages()
        _drain_log_queue(log_queue)
    return stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Receive DCA1000 UDP ADC packets and emit complete radar frames."
    )
    parser.add_argument(
        "--config",
        default=DEFAULT_CONFIG_PATH,
        type=Path,
        help=(
            "Radar .cfg or mmWave Studio JSON used to derive frame size. "
            f"Defaults to {DEFAULT_CONFIG_PATH}."
        ),
    )
    parser.add_argument(
        "--setup",
        default=DEFAULT_SETUP_PATH,
        type=Path,
        help=(
            "mmWave Studio setup JSON with DCA1000 capture settings. "
            f"Defaults to {DEFAULT_SETUP_PATH}."
        ),
    )
    parser.add_argument("--host-ip", default=UDP_IP, help="Host Ethernet IP to bind.")
    parser.add_argument("--data-port", type=int, default=UDP_PORT)
    parser.add_argument("--buffer-size", type=int, default=BUFFER_SIZE)
    parser.add_argument(
        "--socket-recv-buffer",
        type=int,
        default=DEFAULT_SOCKET_RECV_BUFFER_BYTES,
        help=(
            "Requested UDP socket receive buffer size in bytes. "
            "Use 0 to keep the Windows default. The actual granted size is logged."
        ),
    )
    parser.add_argument(
        "--processing-queue-size",
        type=int,
        default=DEFAULT_PROCESSING_QUEUE_SIZE,
        help=(
            "Maximum complete valid frames waiting for FFT/display processing. "
            "If this fills, frames are dropped instead of blocking UDP receive."
        ),
    )
    parser.add_argument(
        "--socket-timeout",
        type=float,
        default=SOCKET_TIMEOUT_SECONDS,
        help="Seconds between socket polls so Ctrl+C can stop the receiver.",
    )
    parser.add_argument(
        "--log-file",
        type=Path,
        default=DEFAULT_LOG_PATH,
        help="Append terminal status output to this log file.",
    )
    parser.add_argument(
        "--raw-output",
        type=Path,
        help=(
            "Write valid complete raw ADC frames to this binary file. "
            "Frames are stored consecutively without DCA1000 packet headers."
        ),
    )
    parser.add_argument(
        "--raw-metadata",
        type=Path,
        help=(
            "Write raw capture metadata to this JSON file. "
            "Defaults to '<raw-output>.json' when --raw-output is used."
        ),
    )
    parser.add_argument(
        "--display",
        choices=("none", "range", "range-doppler", "point-cloud"),
        default="none",
        help="Optional live display mode.",
    )
    parser.add_argument(
        "--display-update-every",
        type=int,
        default=1,
        help="Update the live display every N valid frames.",
    )
    parser.add_argument(
        "--display-pause",
        type=float,
        default=0.03,
        help="Seconds to yield to the GUI event loop when display is enabled.",
    )
    parser.add_argument(
        "--max-range-m",
        type=float,
        default=DEFAULT_MAX_RANGE_M,
        help=(
            "Maximum range in meters for all live displays. "
            "Defaults to 10 m; use 0 for the full computed range."
        ),
    )
    parser.add_argument(
        "--point-cloud-fov-deg",
        type=float,
        default=DEFAULT_POINT_CLOUD_FOV_DEG,
        help=(
            "Point-cloud azimuth/elevation half-FOV in degrees. "
            "Defaults to ±60 degrees."
        ),
    )
    return parser.parse_args()


def _format_stats(stats: CaptureStats) -> str:
    return (
        "stats: "
        f"packets={stats.packets_received}, "
        f"frames={stats.frames_emitted}, "
        f"invalid_frames={stats.invalid_frames}, "
        f"lost_packets={stats.lost_packets}, "
        f"out_of_order={stats.out_of_order_packets}, "
        f"duplicates={stats.duplicate_packets}, "
        f"byte_gaps={stats.byte_gaps}/{stats.byte_gap_bytes}B, "
        f"stream_resyncs={stats.stream_resyncs}, "
        f"byte_overlaps={stats.byte_overlaps}/{stats.byte_overlap_bytes}B, "
        f"malformed={stats.malformed_packets}, "
        f"processing_drops={stats.processing_frames_dropped}"
    )


def _resolve_config_path(config_path: Path) -> Path:
    if config_path.exists():
        return config_path

    if not config_path.is_absolute():
        script_relative_path = Path(__file__).resolve().parent / config_path.name
        if script_relative_path.exists():
            return script_relative_path

    raise FileNotFoundError(
        "Config file not found: "
        f"{config_path}. Current directory is {Path.cwd()}. "
        "Pass the full path, or place the file beside livedatacapture.py."
    )


def _resolve_output_path(output_path: Path) -> Path:
    if output_path.is_absolute():
        return output_path
    return Path.cwd() / output_path


def _default_metadata_path(output_path: Optional[Path]) -> Optional[Path]:
    if output_path is None:
        return None
    return Path(f"{output_path}.json")


def _config_from_json_command_lines(data: Any) -> Optional[RadarCaptureConfig]:
    command_lines = list(_iter_json_command_lines(data))
    if not command_lines:
        return None

    try:
        return _config_from_cfg_lines(command_lines)
    except ValueError:
        return None


def _config_from_mmwave_studio_json(data: Any) -> Optional[RadarCaptureConfig]:
    if not isinstance(data, dict):
        return None

    devices = data.get("mmWaveDevices")
    if not isinstance(devices, list) or not devices:
        return None

    device = devices[0]
    if not isinstance(device, dict):
        return None

    rf_config = _as_mapping(device.get("rfConfig"))
    if rf_config is None:
        return None

    channel_config = _as_mapping(rf_config.get("rlChanCfg_t"))
    frame_config = _as_mapping(rf_config.get("rlFrameCfg_t"))
    profile_config = _first_nested_mapping(
        rf_config.get("rlProfiles"),
        "rlProfileCfg_t",
    )

    if channel_config is None or frame_config is None or profile_config is None:
        return None

    raw_capture_config = _as_mapping(device.get("rawDataCaptureConfig")) or {}
    data_format_config = _as_mapping(raw_capture_config.get("rlDevDataFmtCfg_t")) or {}
    lane_enable_config = _as_mapping(raw_capture_config.get("rlDevLaneEnable_t")) or {}

    rx_channel_mask = _required_int(channel_config, "rxChannelEn")
    chirp_start_idx = _required_int(frame_config, "chirpStartIdx")
    chirp_end_idx = _required_int(frame_config, "chirpEndIdx")
    num_loops = _required_int(frame_config, "numLoops")

    iq_swap = bool(_optional_int(data_format_config, "iqSwapSel", "iqSwap") or 0)
    channel_interleave_value = _optional_int(
        data_format_config,
        "chInterleave",
        "channelInterleave",
    )
    channel_interleave = (
        False if channel_interleave_value is None else channel_interleave_value == 0
    )

    lane_mask = _optional_int(lane_enable_config, "laneEn")
    lvds_lanes = _bit_count(lane_mask) if lane_mask is not None else 2

    return RadarCaptureConfig.from_dimensions(
        num_adc_samples=_required_int(profile_config, "numAdcSamples"),
        num_rx_channels=_bit_count(rx_channel_mask),
        num_chirps_per_frame=num_loops * (chirp_end_idx - chirp_start_idx + 1),
        iq_swap=iq_swap,
        channel_interleave=channel_interleave,
        lvds_lanes=lvds_lanes,
        num_loops=num_loops,
        num_chirps_per_loop=chirp_end_idx - chirp_start_idx + 1,
        sample_rate_ksps=_optional_float(profile_config, "digOutSampleRate"),
        frequency_slope_mhz_per_us=_optional_float(
            profile_config,
            "freqSlopeConst_MHz_usec",
            "freqSlopeConst",
        ),
    )


def _config_from_mapping(data: Any, *, source_name: str) -> RadarCaptureConfig:
    num_adc_samples = _required_int(
        data,
        "num_adc_samples",
        "numAdcSamples",
        "NumAdcSamples",
        "NumOfAdcSamples",
        "numADCSamples",
        "adcSamples",
    )

    num_rx_channels = _optional_int(
        data,
        "num_rx_channels",
        "numRxChannels",
        "NumRxChannels",
        "NumOfRxChannels",
        "numRxAntennas",
        "NumRxAntennas",
    )
    if num_rx_channels is None:
        rx_channel_mask = _optional_int(
            data, "rxChannelEn", "RxChannelEn", "rxChannelEnable", "rxChanEn"
        )
        num_rx_channels = (
            _bit_count(rx_channel_mask)
            if rx_channel_mask is not None
            else _enabled_count(data, "rx0En", "rx1En", "rx2En", "rx3En")
        )

    num_chirps_per_frame = _optional_int(
        data,
        "num_chirps_per_frame",
        "numChirpsPerFrame",
        "NumChirpsPerFrame",
        "NumOfChirpsPerFrame",
    )
    if num_chirps_per_frame is None:
        chirp_start_idx = _required_int(
            data,
            "fchirpStartIdx",
            "frameChirpStartIdx",
            "chirpStartIdx",
            "ChirpStartIdx",
            "chirpStartIndex",
        )
        chirp_end_idx = _required_int(
            data,
            "fchirpEndIdx",
            "frameChirpEndIdx",
            "chirpEndIdx",
            "ChirpEndIdx",
            "chirpEndIndex",
        )
        num_loops = _required_int(
            data, "numLoops", "NumLoops", "numOfLoops", "loopCount"
        )
        num_chirps_per_frame = num_loops * (chirp_end_idx - chirp_start_idx + 1)

    iq_swap = bool(_optional_int(data, "iqSwap", "IQSwap", "sampleSwap") or 0)
    sample_rate_ksps = _optional_float(
        data, "digOutSampleRate", "sampleRateKsps", "sample_rate_ksps"
    )
    frequency_slope_mhz_per_us = _optional_float(
        data,
        "freqSlopeConst",
        "freqSlopeConst_MHz_usec",
        "frequencySlopeMhzPerUs",
        "frequency_slope_mhz_per_us",
    )
    channel_interleave_value = _optional_int(
        data, "channelInterleave", "ChannelInterleave", "chInterleave"
    )
    channel_interleave = (
        False if channel_interleave_value is None else channel_interleave_value == 0
    )

    lvds_lanes = _optional_int(
        data, "lvds_lanes", "lvdsLanes", "NumOfLanes", "numLanes"
    )
    if lvds_lanes is None:
        lane_mask = _optional_int(data, "laneEn", "LaneEn", "lvdsLaneEn", "laneEnable")
        if lane_mask is not None:
            lvds_lanes = _bit_count(lane_mask)
        else:
            lvds_lanes = _enabled_count(
                data, "lane1En", "lane2En", "lane3En", "lane4En", default=2
            )

    if num_adc_samples <= 0 or num_rx_channels <= 0 or num_chirps_per_frame <= 0:
        raise ValueError(f"{source_name} radar dimensions must be positive")

    return RadarCaptureConfig.from_dimensions(
        num_adc_samples=num_adc_samples,
        num_rx_channels=num_rx_channels,
        num_chirps_per_frame=num_chirps_per_frame,
        iq_swap=iq_swap,
        channel_interleave=channel_interleave,
        lvds_lanes=lvds_lanes,
        num_loops=num_loops if "num_loops" in locals() else None,
        num_chirps_per_loop=(
            chirp_end_idx - chirp_start_idx + 1
            if "chirp_start_idx" in locals() and "chirp_end_idx" in locals()
            else None
        ),
        sample_rate_ksps=sample_rate_ksps,
        frequency_slope_mhz_per_us=frequency_slope_mhz_per_us,
    )


def _config_from_cfg_lines(lines: Iterable[str]) -> RadarCaptureConfig:
    profile_adc_samples: Optional[int] = None
    rx_channel_mask: Optional[int] = None
    chirp_start_idx: Optional[int] = None
    chirp_end_idx: Optional[int] = None
    num_loops: Optional[int] = None
    iq_swap = False
    channel_interleave = False
    lvds_lanes = 2
    sample_rate_ksps: Optional[float] = None
    frequency_slope_mhz_per_us: Optional[float] = None
    chirp_tx_masks: dict[int, int] = {}

    for raw_line in lines:
        line = raw_line.split("%", 1)[0].split("#", 1)[0].strip()
        if not line:
            continue

        tokens = line.split()
        command = tokens[0]
        if command == "profileCfg" and len(tokens) > 10:
            frequency_slope_mhz_per_us = float(tokens[8])
            profile_adc_samples = int(float(tokens[10]))
            sample_rate_ksps = float(tokens[11])
        elif command == "channelCfg" and len(tokens) > 1:
            rx_channel_mask = int(tokens[1], 0)
        elif command == "frameCfg" and len(tokens) > 3:
            chirp_start_idx = int(tokens[1])
            chirp_end_idx = int(tokens[2])
            num_loops = int(tokens[3])
        elif command == "chirpCfg" and len(tokens) > 8:
            chirp_cfg_start_idx = int(tokens[1])
            chirp_cfg_end_idx = int(tokens[2])
            tx_enable_mask = int(tokens[8], 0)
            for chirp_idx in range(chirp_cfg_start_idx, chirp_cfg_end_idx + 1):
                chirp_tx_masks[chirp_idx] = tx_enable_mask
        elif command == "adcbufCfg" and len(tokens) > 4:
            iq_swap = bool(int(tokens[3]))
            # TI adcbufCfg uses 0 for interleaved and 1 for non-interleaved.
            channel_interleave = int(tokens[4]) == 0
        elif command in {"lvdsLaneCfg", "laneCfg"} and len(tokens) > 1:
            lvds_lanes = _bit_count(int(tokens[1], 0))

    missing = []
    if profile_adc_samples is None:
        missing.append("profileCfg numAdcSamples")
    if rx_channel_mask is None:
        missing.append("channelCfg rxChannelEn")
    if chirp_start_idx is None or chirp_end_idx is None or num_loops is None:
        missing.append("frameCfg chirpStartIdx/chirpEndIdx/numLoops")
    if missing:
        raise ValueError(f"Missing required config fields: {', '.join(missing)}")

    return RadarCaptureConfig.from_dimensions(
        num_adc_samples=profile_adc_samples,
        num_rx_channels=_bit_count(rx_channel_mask),
        num_chirps_per_frame=num_loops * (chirp_end_idx - chirp_start_idx + 1),
        iq_swap=iq_swap,
        channel_interleave=channel_interleave,
        lvds_lanes=lvds_lanes or 2,
        num_loops=num_loops,
        num_chirps_per_loop=chirp_end_idx - chirp_start_idx + 1,
        tx_channel_masks=tuple(
            chirp_tx_masks.get(chirp_idx, 1 << offset)
            for offset, chirp_idx in enumerate(range(chirp_start_idx, chirp_end_idx + 1))
        ),
        sample_rate_ksps=sample_rate_ksps,
        frequency_slope_mhz_per_us=frequency_slope_mhz_per_us,
    )


def _iter_json_command_lines(value: Any) -> Iterable[str]:
    known_commands = {
        "profileCfg",
        "channelCfg",
        "frameCfg",
        "adcbufCfg",
        "lvdsLaneCfg",
        "laneCfg",
    }
    if isinstance(value, str):
        stripped = value.strip()
        if stripped and stripped.split(maxsplit=1)[0] in known_commands:
            yield stripped
    elif isinstance(value, list):
        if value and isinstance(value[0], str) and value[0] in known_commands:
            yield " ".join(str(item) for item in value)
        for item in value:
            yield from _iter_json_command_lines(item)
    elif isinstance(value, dict):
        for item in value.values():
            yield from _iter_json_command_lines(item)


def _required_int(data: Any, *names: str) -> int:
    value = _optional_value(data, *names)
    if value is None:
        raise ValueError(f"Missing required JSON field; tried: {', '.join(names)}")
    return _to_int(value)


def _optional_int(data: Any, *names: str) -> Optional[int]:
    value = _optional_value(data, *names)
    return None if value is None else _to_int(value)


def _optional_float(data: Any, *names: str) -> Optional[float]:
    value = _optional_value(data, *names)
    return None if value is None else float(value)


def _optional_string(data: Any, *names: str) -> Optional[str]:
    value = _optional_value(data, *names)
    return None if value is None else str(value)


def _optional_value(data: Any, *names: str) -> Any:
    normalized_names = {_normalize_key(name) for name in names}
    for key, value in _walk_json(data):
        if _normalize_key(key) in normalized_names:
            return value
    return None


def _as_mapping(value: Any) -> Optional[dict[str, Any]]:
    return value if isinstance(value, dict) else None


def _first_nested_mapping(value: Any, key: str) -> Optional[dict[str, Any]]:
    if isinstance(value, dict):
        return _as_mapping(value.get(key))

    if isinstance(value, list):
        for item in value:
            if not isinstance(item, dict):
                continue
            nested = _as_mapping(item.get(key))
            if nested is not None:
                return nested

    return None


def _enabled_count(data: Any, *names: str, default: Optional[int] = None) -> int:
    values = [_optional_int(data, name) for name in names]
    present_values = [value for value in values if value is not None]
    if not present_values:
        if default is not None:
            return default
        raise ValueError(f"Missing enable fields; tried: {', '.join(names)}")
    return sum(1 for value in present_values if value != 0)


def _walk_json(value: Any) -> Iterable[tuple[str, Any]]:
    if isinstance(value, dict):
        for key, child in value.items():
            yield str(key), child
            yield from _walk_json(child)
    elif isinstance(value, list):
        for child in value:
            yield from _walk_json(child)


def _normalize_key(value: str) -> str:
    return "".join(character for character in value.lower() if character.isalnum())


def _to_int(value: Any) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)

    text = str(value).strip()
    try:
        return int(text, 0)
    except ValueError:
        return int(float(text))


def _bit_count(mask: int) -> int:
    return int(mask).bit_count()


def setup_terminal_log(log_path: Path) -> None:
    global _LOG_FILE

    resolved_path = log_path
    if not resolved_path.is_absolute():
        resolved_path = Path.cwd() / resolved_path
    resolved_path.parent.mkdir(parents=True, exist_ok=True)

    _LOG_FILE = resolved_path.open("a", encoding="utf-8", buffering=1)
    emit("")
    emit(f"--- Live capture log started at {_timestamp()}: {resolved_path} ---")


def close_terminal_log() -> None:
    global _LOG_FILE

    if _LOG_FILE is not None:
        emit(f"--- Live capture log ended at {_timestamp()} ---")
        _LOG_FILE.close()
        _LOG_FILE = None


def emit(message: str) -> None:
    print(message)
    if _LOG_FILE is not None:
        _LOG_FILE.write(f"[{_timestamp()}] {message}\n")


def _timestamp() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def main() -> None:
    args = parse_args()
    setup_terminal_log(args.log_file)
    frame_queue: Optional[mp.Queue] = None
    log_queue: Optional[mp.Queue] = None
    processor: Optional[mp.Process] = None
    display: Optional[LiveDisplay] = None
    try:
        config = RadarCaptureConfig.from_file(args.config)
        emit(f"Loaded radar config: {config}")
        try:
            dsp_backend = validate_openradar_backend()
        except RuntimeError as exc:
            raise CaptureStartupError(str(exc)) from exc
        emit(f"DSP backend: {dsp_backend}")
        setup_config = CaptureSetupConfig.from_file(args.setup)
        emit(f"Loaded capture setup: {setup_config}")
        display = LiveDisplay(
            args.display,
            args.display_pause,
            args.max_range_m,
            point_cloud_range_m=(
                args.max_range_m
                if args.max_range_m > 0.0
                else _point_cloud_range_limit_m(config)
            ),
            point_cloud_fov_deg=args.point_cloud_fov_deg,
        )
        frame_queue = mp.Queue(maxsize=max(args.processing_queue_size, 1))
        log_queue = mp.Queue(maxsize=1000)
        processor = mp.Process(
            target=_run_frame_processor,
            args=(
                config,
                frame_queue,
                log_queue,
                display.payload_queue if display is not None else None,
                args.raw_output,
                args.raw_metadata,
                args.display,
                args.display_update_every,
                args.max_range_m,
                args.point_cloud_fov_deg,
            ),
            name="RadarFrameProcessor",
        )
        processor.start()
        listen_for_frames(
            host_ip=args.host_ip,
            data_port=args.data_port,
            config=config,
            setup_config=setup_config,
            buffer_size=args.buffer_size,
            socket_recv_buffer_bytes=args.socket_recv_buffer,
            socket_timeout_seconds=args.socket_timeout,
            frame_queue=frame_queue,
            log_queue=log_queue,
            display=display,
        )
    except CaptureStartupError as exc:
        emit(f"Capture startup failed: {exc}")
    finally:
        emit("Shutting down capture pipeline...")
        if frame_queue is not None:
            _request_processor_stop(frame_queue)

        if processor is not None:
            processor.join(timeout=5.0)
            if processor.is_alive():
                emit("Frame processor did not stop in time; terminating.")
                processor.terminate()
                processor.join(timeout=1.0)

        if log_queue is not None:
            _drain_log_queue(log_queue)

        if display is not None:
            display.close()
            emit("Live display closed.")

        if frame_queue is not None:
            frame_queue.close()
            frame_queue.join_thread()
        if log_queue is not None:
            log_queue.close()
            log_queue.join_thread()

        emit("Capture pipeline shutdown complete.")
        close_terminal_log()


if __name__ == "__main__":
    main()
