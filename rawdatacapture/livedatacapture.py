import argparse
import importlib.util
import json
import multiprocessing as mp
import os
import queue
import signal
import socket
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, BinaryIO, Callable, Iterable, Optional, TextIO

import numpy as np

if __package__ in {None, ""}:
    repository_root = str(Path(__file__).resolve().parent.parent)
    if repository_root not in sys.path:
        sys.path.insert(0, repository_root)
    from rawdatacapture.dsp import (
        AdaptiveClutterMap,
        DEFAULT_ROTOR_NOISE_GATE_MAX_DB,
        DEFAULT_ROTOR_NOISE_GATE_MIN_DB,
        DEFAULT_ROTOR_NOISE_MIN_SUPPORT,
        DEFAULT_ROTOR_NOISE_SIGMA_MULTIPLIER,
        DEFAULT_ROTOR_NOISE_SUPPORT_SHAPE,
        StaticSceneMap,
        MicroDopplerResult,
        RotorEstimate,
        cluster_point_cloud,
        cluster_point_cloud_with_labels,
        compute_micro_doppler_spectrum,
        compute_per_tx_micro_doppler_spectrogram,
        compute_range_doppler_heatmap,
        compute_range_doppler_fft,
        compute_point_cloud,
        compute_range_fft,
        compute_range_profile,
        compute_rotor_micro_doppler_frame,
        compute_static_point_cloud,
        estimate_rotor_rpm,
        frame_bytes_to_radar_cube,
        range_axis_m,
        range_resolution_m,
        static_target_protection_mask,
        validate_openradar_backend,
    )
else:
    from .dsp import (
        AdaptiveClutterMap,
        DEFAULT_ROTOR_NOISE_GATE_MAX_DB,
        DEFAULT_ROTOR_NOISE_GATE_MIN_DB,
        DEFAULT_ROTOR_NOISE_MIN_SUPPORT,
        DEFAULT_ROTOR_NOISE_SIGMA_MULTIPLIER,
        DEFAULT_ROTOR_NOISE_SUPPORT_SHAPE,
        StaticSceneMap,
        MicroDopplerResult,
        RotorEstimate,
        cluster_point_cloud,
        cluster_point_cloud_with_labels,
        compute_micro_doppler_spectrum,
        compute_per_tx_micro_doppler_spectrogram,
        compute_range_doppler_heatmap,
        compute_range_doppler_fft,
        compute_point_cloud,
        compute_range_fft,
        compute_range_profile,
        compute_rotor_micro_doppler_frame,
        compute_static_point_cloud,
        estimate_rotor_rpm,
        frame_bytes_to_radar_cube,
        range_axis_m,
        range_resolution_m,
        static_target_protection_mask,
        validate_openradar_backend,
    )

from inference import (
    DroneBirdInference,
    InferenceResult,
    doppler_cube_to_feature_step,
)
from tensorrt_inference import (
    create_inference_engine,
    resolve_classification_device,
)


# Default DCA1000 network parameters.
UDP_IP = "192.168.33.30"  # Host/laptop static Ethernet IP
UDP_PORT = 4098           # DCA1000 raw ADC data port
BUFFER_SIZE = 65535       # Max UDP packet payload allocation

DCA1000_HEADER_SIZE = 10
UINT32_MODULO = 2**32
SOCKET_TIMEOUT_SECONDS = 0.5
DEFAULT_PACKET_QUEUE_SIZE = 8192
DEFAULT_PROCESSING_QUEUE_SIZE = 32
DEFAULT_LOG_PATH = Path(__file__).with_suffix(".log")
DEFAULT_CONFIG_PATH = Path(__file__).with_name("mmwave.json")
DEFAULT_SETUP_PATH = Path(__file__).with_name("setup.json")
DEFAULT_CLASSIFICATION_ARTIFACT_DIR = (
    Path(__file__).resolve().parent.parent
    / "Radar-UREx-output"
    / "artifacts"
)
DEFAULT_MAX_RANGE_M = 10.0
DEFAULT_POINT_CLOUD_FOV_DEG = 60.0
DEFAULT_CLUSTER_EPS_M = 0.4
DEFAULT_CLUSTER_MIN_SAMPLES = 2
DEFAULT_CLUTTER_MAP_UPDATE_RATE = 0.02
DEFAULT_CLUTTER_MAP_WARMUP_FRAMES = 30
DEFAULT_CLUTTER_MAP_MIN_SNR_DB = 3.0
DEFAULT_STATIC_DETECTION = True
DEFAULT_STATIC_WARMUP_FRAMES = 30
DEFAULT_STATIC_REFERENCE_FRAMES = 90
DEFAULT_STATIC_MIN_CHANGE_DB = 3.0
DEFAULT_STATIC_BACKGROUND_UPDATE_RATE = 0.01
DEFAULT_STATIC_CLUSTER_MIN_SAMPLES = 1
STATIC_MOTION_HISTORY_UPDATES = 30
STATIC_MOTION_MIN_DISPLACEMENT_M = 0.3
STATIC_HANDOFF_WINDOW_UPDATES = 60
STATIC_HANDOFF_DISTANCE_M = 0.75
STATIC_PROTECTION_CELLS = 2
STATIC_TRACK_MAX_MISSED_UPDATES = 30
DEFAULT_TRACK_ASSOCIATION_DISTANCE_M = 0.75
DEFAULT_TRACK_MAX_MISSED_UPDATES = 10
DEFAULT_TRACK_CONFIRMATION_HITS = 3
COMBINED_DISPLAY_MODE = "point-cloud-micro-doppler"
ROTOR_DISPLAY_MODE = "micro-doppler"
CLASSIFICATION_RESULT_PREFIX = "CLASSIFICATION_RESULT "
COMBINED_POINT_CLOUD_UPDATE_EVERY = 1
MICRO_DOPPLER_HISTORY_UPDATES = 150
MICRO_DOPPLER_RANGE_HALF_WIDTH_BINS = 2
MICRO_DOPPLER_WINDOW_LOOPS = 64
MICRO_DOPPLER_HOP_LOOPS = 32
MICRO_DOPPLER_FFT_SIZE = 128
MICRO_DOPPLER_HISTORY_MAX_GAP_UPDATES = 30
MICRO_DOPPLER_HISTORY_ASSOCIATION_DISTANCE_M = (
    DEFAULT_TRACK_ASSOCIATION_DISTANCE_M
)
ROTOR_MICRO_DOPPLER_RANGE_HALF_WIDTH_BINS = 1
ROTOR_MICRO_DOPPLER_WINDOW_LOOPS = 16
ROTOR_MICRO_DOPPLER_HOP_LOOPS = 2
ROTOR_MICRO_DOPPLER_FFT_SIZE = 128
ROTOR_MICRO_DOPPLER_DC_NOTCH_BINS = 2
ROTOR_MICRO_DOPPLER_HISTORY_SECONDS = 2.0
ROTOR_DISPLAY_HISTORY_SECONDS = 0.2
ROTOR_MICRO_DOPPLER_RELATIVE_DB_MAX = 30.0
DEFAULT_ROTOR_BLADES = 2
DEFAULT_ROTOR_RPM_MAX = 10_700.0
ROTOR_RPM_ESTIMATE_UPDATE_SECONDS = 1.0
ROTOR_DISPLAY_MAX_RATE_HZ = 60.0
ROTOR_DISPLAY_TIME_BINS = 512
PROCESSED_OUTPUT_BUFFER_BYTES = 1024 * 1024
ROTOR_SPECTRUM_OUTPUT_DECIMALS = 2
MAGNITUDE_COLORMAP = "turbo"
POINT_CLOUD_MAGNITUDE_DB_MIN = 60.0
POINT_CLOUD_MAGNITUDE_DB_MAX = 120.0
DEFAULT_SOCKET_RECV_BUFFER_BYTES = 4 * 1024 * 1024
_LOG_FILE: Optional[TextIO] = None
EmitFunc = Callable[[str], None]


class CaptureStartupError(RuntimeError):
    """Raised for expected startup failures that should not print a traceback."""


@dataclass(frozen=True)
class RotorPostprocessItem:
    """Ordered compact handoff from radar DSP to inference/output processing."""

    frame_index: int
    save_update: bool
    feature_step: np.ndarray
    rotor_result: MicroDopplerResult


@dataclass(frozen=True)
class DCA1000PacketHeader:
    """DCA1000 inline packet header: uint32 sequence + uint48 byte count."""

    sequence_number: int
    byte_count: int

    @classmethod
    def parse(
        cls,
        packet: bytes | bytearray | memoryview,
    ) -> "DCA1000PacketHeader":
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
    receiver_queue_drops: int = 0
    processing_frames_dropped: int = 0


class UdpPacketReceiver:
    """Drain UDP datagrams into a bounded in-process queue."""

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
        self.thread.join(timeout=timeout)

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

            packet_received_at_s = time.perf_counter()
            try:
                self.packet_queue.put_nowait((packet, packet_received_at_s))
            except queue.Full:
                self.stats.receiver_queue_drops += 1


@dataclass(frozen=True)
class CapturedFrame:
    data: bytes
    gap_bytes: int
    first_byte_at_s: float

    @property
    def is_valid(self) -> bool:
        return self.gap_bytes == 0


@dataclass(frozen=True)
class TargetTrack:
    """Display-safe snapshot of the one persistent 3D target track."""

    position_m: tuple[float, float, float]
    velocity_m_per_update: tuple[float, float, float]
    age_updates: int
    hits: int
    missed_updates: int
    confirmed: bool

    @property
    def range_m(self) -> float:
        return float(np.linalg.norm(self.position_m))

    @property
    def is_predicted(self) -> bool:
        return self.missed_updates > 0


@dataclass(frozen=True)
class StaticReferenceStatus:
    enabled: bool
    ready: bool
    frames_seen: int
    required_frames: int
    warmup_frames_seen: int = 0
    warmup_frames_required: int = 0
    adaptive: bool = False

    @property
    def label(self) -> str:
        if not self.enabled:
            return "Static detection disabled"
        if self.warmup_frames_seen < self.warmup_frames_required:
            return (
                "Warming static detector "
                f"{self.warmup_frames_seen}/{self.warmup_frames_required}"
            )
        if self.ready:
            return "Static reference ready (adaptive)"
        return (
            "Calibrating static reference "
            f"{self.frames_seen}/{self.required_frames}"
        )


@dataclass(frozen=True)
class PointCloudDisplayPayload:
    points: np.ndarray
    clusters: np.ndarray
    static_points: np.ndarray
    static_clusters: np.ndarray
    target_track: Optional[TargetTrack]
    target_source: Optional[str]
    static_reference: StaticReferenceStatus
    static_candidate_count: int = 0
    static_validation: str = "disabled"


@dataclass(frozen=True)
class CombinedDisplayPayload:
    point_cloud: PointCloudDisplayPayload
    spectrogram_db: np.ndarray
    selected_range_m: Optional[float]


@dataclass(frozen=True)
class RotorDisplayPayload:
    result: MicroDopplerResult


@dataclass(frozen=True)
class RotorDisplayFrame:
    spectrogram_db: np.ndarray
    extent: tuple[float, float, float, float]
    flash_times_s: np.ndarray
    flash_scores_db: np.ndarray
    status_text: str
    dc_notch_velocity_bounds: tuple[float, float]


class ProcessingTimingStats:
    """Collect low-overhead per-stage timing samples for the final summary."""

    STAGE_ORDER = (
        "range_fft",
        "doppler",
        "dynamic_detection",
        "static_detection",
        "clustering",
        "micro_doppler",
        "classification_feature",
        "classification",
        "serialization",
        "total",
    )

    def __init__(self) -> None:
        self.samples_ms = {stage: [] for stage in self.STAGE_ORDER}

    def add(self, stage: str, elapsed_seconds: float) -> None:
        if stage in self.samples_ms:
            self.samples_ms[stage].append(max(float(elapsed_seconds), 0.0) * 1000.0)

    def format_summary(self) -> str:
        fields = []
        for stage in self.STAGE_ORDER:
            samples = self.samples_ms[stage]
            if not samples:
                continue
            values = np.asarray(samples, dtype=np.float64)
            fields.append(
                f"{stage}=p50:{np.percentile(values, 50):.2f}ms/"
                f"p95:{np.percentile(values, 95):.2f}ms/"
                f"max:{np.max(values):.2f}ms"
            )
        return "Processing timing summary: " + ", ".join(fields)


class SingleTargetTracker:
    """Track one 3D point-cloud target with gated nearest-neighbor updates.

    Positions and velocity are expressed per processed display update because
    the current capture configuration does not expose frame timestamps to this
    stage. The tracker acquires the strongest candidate, then prioritizes
    spatial continuity over candidate strength until the track is lost.
    """

    def __init__(
        self,
        *,
        association_distance_m: float = DEFAULT_TRACK_ASSOCIATION_DISTANCE_M,
        max_missed_updates: int = DEFAULT_TRACK_MAX_MISSED_UPDATES,
        confirmation_hits: int = DEFAULT_TRACK_CONFIRMATION_HITS,
        position_gain: float = 0.65,
        velocity_gain: float = 0.35,
        acquisition_policy: str = "strongest",
    ) -> None:
        if association_distance_m <= 0.0:
            raise ValueError("Track association distance must be positive")
        if max_missed_updates < 0:
            raise ValueError("Maximum missed track updates cannot be negative")
        if confirmation_hits < 1:
            raise ValueError("Track confirmation hits must be at least one")
        if not 0.0 < position_gain <= 1.0:
            raise ValueError("Track position gain must be in (0, 1]")
        if not 0.0 <= velocity_gain <= 1.0:
            raise ValueError("Track velocity gain must be in [0, 1]")
        if acquisition_policy not in {"strongest", "nearest"}:
            raise ValueError(
                "Track acquisition policy must be 'strongest' or 'nearest'"
            )

        self.association_distance_m = float(association_distance_m)
        self.max_missed_updates = int(max_missed_updates)
        self.confirmation_hits = int(confirmation_hits)
        self.position_gain = float(position_gain)
        self.velocity_gain = float(velocity_gain)
        self.acquisition_policy = acquisition_policy
        self._position_m: Optional[np.ndarray] = None
        self._velocity_m_per_update = np.zeros(3, dtype=np.float64)
        self._age_updates = 0
        self._hits = 0
        self._consecutive_hits = 0
        self._missed_updates = 0
        self._confirmed = False

    @property
    def is_active(self) -> bool:
        return self._position_m is not None

    @property
    def is_confirmed(self) -> bool:
        return self._confirmed

    def update(self, candidates: np.ndarray) -> Optional[TargetTrack]:
        candidates = np.asarray(candidates, dtype=np.float64)
        if candidates.ndim != 2 or (candidates.size and candidates.shape[1] < 4):
            raise ValueError("Track candidates must have shape [candidate, >=4]")

        if self._position_m is None:
            if candidates.size == 0:
                return None
            if self.acquisition_policy == "nearest":
                acquisition_index = int(
                    np.argmin(np.linalg.norm(candidates[:, :3], axis=1))
                )
            else:
                acquisition_index = int(np.argmax(candidates[:, 3]))
            self._position_m = candidates[acquisition_index, :3].copy()
            self._velocity_m_per_update.fill(0.0)
            self._age_updates = 1
            self._hits = 1
            self._consecutive_hits = 1
            self._missed_updates = 0
            self._confirmed = self.confirmation_hits == 1
            return self._snapshot()

        predicted_position = self._position_m + self._velocity_m_per_update
        matched_position: Optional[np.ndarray] = None
        if candidates.size:
            distances_m = np.linalg.norm(candidates[:, :3] - predicted_position, axis=1)
            match_index = int(np.argmin(distances_m))
            expanded_gate_m = self.association_distance_m * (
                1.0 + 0.25 * self._missed_updates
            )
            if distances_m[match_index] <= expanded_gate_m:
                matched_position = candidates[match_index, :3]

        self._age_updates += 1
        if matched_position is None:
            self._position_m = predicted_position
            self._missed_updates += 1
            self._consecutive_hits = 0
            if self._missed_updates >= self.max_missed_updates:
                self.reset()
                return None
            return self._snapshot()

        previous_position = self._position_m.copy()
        innovation = matched_position - predicted_position
        self._position_m = predicted_position + self.position_gain * innovation
        measured_velocity = self._position_m - previous_position
        self._velocity_m_per_update = (
            (1.0 - self.velocity_gain) * self._velocity_m_per_update
            + self.velocity_gain * measured_velocity
        )
        self._hits += 1
        self._consecutive_hits += 1
        self._missed_updates = 0
        if self._consecutive_hits >= self.confirmation_hits:
            self._confirmed = True
        return self._snapshot()

    def reset(self) -> None:
        self._position_m = None
        self._velocity_m_per_update.fill(0.0)
        self._age_updates = 0
        self._hits = 0
        self._consecutive_hits = 0
        self._missed_updates = 0
        self._confirmed = False

    def _snapshot(self) -> TargetTrack:
        assert self._position_m is not None
        return TargetTrack(
            position_m=tuple(float(value) for value in self._position_m),
            velocity_m_per_update=tuple(
                float(value) for value in self._velocity_m_per_update
            ),
            age_updates=self._age_updates,
            hits=self._hits,
            missed_updates=self._missed_updates,
            confirmed=self._confirmed,
        )


class MotionHandoffQualifier:
    """Retain a short handoff window after a genuinely moving target."""

    def __init__(
        self,
        *,
        history_updates: int = STATIC_MOTION_HISTORY_UPDATES,
        minimum_displacement_m: float = STATIC_MOTION_MIN_DISPLACEMENT_M,
        handoff_window_updates: int = STATIC_HANDOFF_WINDOW_UPDATES,
        protection_missed_updates: int = STATIC_TRACK_MAX_MISSED_UPDATES,
    ) -> None:
        self.history: deque[Optional[np.ndarray]] = deque(
            maxlen=max(int(history_updates), 2)
        )
        self.minimum_displacement_m = max(float(minimum_displacement_m), 0.0)
        self.handoff_window_updates = max(int(handoff_window_updates), 1)
        self.protection_missed_updates = max(
            int(protection_missed_updates),
            0,
        )
        self.remaining_updates = 0
        self.consecutive_missed_updates = 0
        self.last_position_m: Optional[np.ndarray] = None

    def update(self, dynamic_track: Optional[TargetTrack]) -> Optional[np.ndarray]:
        if self.remaining_updates > 0:
            self.remaining_updates -= 1

        measured_position: Optional[np.ndarray] = None
        if (
            dynamic_track is not None
            and dynamic_track.confirmed
            and not dynamic_track.is_predicted
        ):
            self.consecutive_missed_updates = 0
            measured_position = np.asarray(
                dynamic_track.position_m,
                dtype=np.float64,
            )
            if any(
                previous is not None
                and np.linalg.norm(measured_position - previous)
                >= self.minimum_displacement_m
                for previous in self.history
            ):
                self.remaining_updates = self.handoff_window_updates
            if self.remaining_updates > 0:
                self.last_position_m = measured_position.copy()
        else:
            self.consecutive_missed_updates += 1
        self.history.append(
            measured_position.copy()
            if measured_position is not None
            else None
        )

        if self.remaining_updates <= 0:
            self.last_position_m = None
            return None
        assert self.last_position_m is not None
        return self.last_position_m.copy()

    @property
    def protection_position_m(self) -> Optional[np.ndarray]:
        """Protect recent motion until 30 consecutive detection misses."""
        if (
            self.last_position_m is None
            or self.consecutive_missed_updates
            >= self.protection_missed_updates
        ):
            return None
        return self.last_position_m.copy()


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
    start_frequency_ghz: Optional[float] = None
    idle_time_us: Optional[float] = None
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
        ramp_end_time_us: Optional[float] = None,
        frame_periodicity_ms: Optional[float] = None,
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
            start_frequency_ghz=start_frequency_ghz,
            idle_time_us=idle_time_us,
            ramp_end_time_us=ramp_end_time_us,
            frame_periodicity_ms=frame_periodicity_ms,
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

    @property
    def chirp_period_s(self) -> Optional[float]:
        if self.idle_time_us is None or self.ramp_end_time_us is None:
            return None
        return (self.idle_time_us + self.ramp_end_time_us) * 1e-6

    @property
    def slow_time_interval_s(self) -> Optional[float]:
        if self.chirp_period_s is None:
            return None
        return self.chirp_period_s * (self.num_chirps_per_loop or 1)

    @property
    def slow_time_rate_hz(self) -> Optional[float]:
        if not self.slow_time_interval_s:
            return None
        return 1.0 / self.slow_time_interval_s

    @property
    def frame_duty_cycle(self) -> Optional[float]:
        if self.chirp_period_s is None or not self.frame_periodicity_ms:
            return None
        active_time_s = self.num_chirps_per_frame * self.chirp_period_s
        return active_time_s / (self.frame_periodicity_ms * 1e-3)


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
        payload: bytes | bytearray | memoryview,
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
        process_context: Optional[Any] = None,
    ) -> None:
        context = process_context or mp
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
        self.stop_event: Optional[mp.Event] = None
        self.payload_queue: Optional[mp.Queue] = None
        self.process: Optional[mp.Process] = None
        self.rendered_updates: Optional[Any] = None
        self.skipped_updates: Optional[Any] = None
        self.startup_status_queue: Optional[mp.Queue] = None

        if self.mode == "none":
            return
        dependency_error = _display_dependency_error(self.mode)
        if dependency_error is not None:
            raise CaptureStartupError(dependency_error)

        self.stop_event = context.Event()
        self.payload_queue = context.Queue(maxsize=1)
        self.rendered_updates = context.Value("Q", 0)
        self.skipped_updates = context.Value("Q", 0)
        self.startup_status_queue = context.Queue(maxsize=1)
        self.process = context.Process(
            target=_run_display_process,
            args=(
                self.mode,
                self.pause_seconds,
                self.max_range_m,
                self.point_cloud_range_m,
                self.point_cloud_fov_deg,
                self.payload_queue,
                self.stop_event,
                self.rendered_updates,
                self.skipped_updates,
                self.startup_status_queue,
            ),
            name="RadarLiveDisplay",
            daemon=True,
        )
        self.process.start()
        try:
            startup_status = self.startup_status_queue.get(timeout=8.0)
        except queue.Empty as exc:
            exit_code = self.process.exitcode
            self.close()
            detail = (
                f" (display process exited with code {exit_code})"
                if exit_code is not None
                else ""
            )
            raise CaptureStartupError(
                "Live display did not report readiness within 8 seconds"
                f"{detail}."
            ) from exc
        if startup_status.get("state") != "ready":
            self.close()
            raise CaptureStartupError(
                startup_status.get(
                    "message",
                    "Live display failed during startup.",
                )
            )
        emit(startup_status["message"])

    def close(self) -> None:
        if self.stop_event is not None:
            self.stop_event.set()
        if self.process is not None:
            self.process.join(timeout=2.0)
            if self.process.is_alive():
                self.process.terminate()
                self.process.join(timeout=1.0)
        if self.payload_queue is not None:
            self.payload_queue.close()
            self.payload_queue.join_thread()
        if self.startup_status_queue is not None:
            self.startup_status_queue.close()
            self.startup_status_queue.join_thread()

    @property
    def rendered_update_count(self) -> int:
        return _shared_counter_value(self.rendered_updates)

    @property
    def skipped_update_count(self) -> int:
        return _shared_counter_value(self.skipped_updates)


def _point_cloud_track_candidates(
    points: np.ndarray,
    clusters: np.ndarray,
    *,
    assignment_distance_m: Optional[float] = None,
) -> np.ndarray:
    """Return XYZ candidates with a magnitude score for track acquisition."""
    if clusters.size == 0:
        if points.size == 0:
            return np.empty((0, 4), dtype=np.float32)
        return np.asarray(points[:, :4], dtype=np.float32)

    candidates = np.empty((clusters.shape[0], 4), dtype=np.float32)
    candidates[:, :3] = clusters[:, :3]
    if points.size == 0:
        candidates[:, 3] = clusters[:, 3]
        return candidates

    distances = np.linalg.norm(
        points[:, np.newaxis, :3] - clusters[np.newaxis, :, :3],
        axis=2,
    )
    assigned_clusters = np.argmin(distances, axis=1)
    for cluster_index in range(clusters.shape[0]):
        assigned = assigned_clusters == cluster_index
        if assignment_distance_m is not None:
            assigned &= distances[:, cluster_index] <= assignment_distance_m
        member_scores = points[assigned, 3]
        candidates[cluster_index, 3] = (
            float(np.max(member_scores))
            if member_scores.size
            else float(clusters[cluster_index, 3])
        )
    return candidates


class ProcessedOutputWriter:
    """Stream processed point-cloud and micro-Doppler updates as JSON Lines."""

    def __init__(
        self,
        output_path: Optional[Path],
        config: RadarCaptureConfig,
        update_every: int,
        emit_func: Optional[EmitFunc] = None,
        *,
        static_detection: bool = DEFAULT_STATIC_DETECTION,
        static_warmup_frames: int = DEFAULT_STATIC_WARMUP_FRAMES,
        static_reference_frames: int = DEFAULT_STATIC_REFERENCE_FRAMES,
        static_min_change_db: float = DEFAULT_STATIC_MIN_CHANGE_DB,
        static_background_update_rate: float = (
            DEFAULT_STATIC_BACKGROUND_UPDATE_RATE
        ),
        static_cluster_min_samples: int = DEFAULT_STATIC_CLUSTER_MIN_SAMPLES,
        rotor_processing: Optional[dict[str, Any]] = None,
        classification_metadata: Optional[dict[str, Any]] = None,
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
            "format": "radar-processed-jsonl",
            "version": 5,
            "created_at": _timestamp(),
            "display_update_every": max(int(update_every), 1),
            "point_columns": ["x_m", "y_m", "z_m", "magnitude_db"],
            "cluster_columns": ["x_m", "y_m", "z_m", "point_count"],
            "static_point_columns": [
                "x_m",
                "y_m",
                "z_m",
                "magnitude_db",
                "change_db",
            ],
            "static_cluster_columns": [
                "x_m",
                "y_m",
                "z_m",
                "point_count",
            ],
            "static_detection": {
                "enabled": bool(static_detection),
                "warmup_frames": max(int(static_warmup_frames), 0),
                "reference_frames": max(int(static_reference_frames), 1),
                "minimum_change_db": max(float(static_min_change_db), 0.0),
                "background_update_rate": min(
                    max(float(static_background_update_rate), 0.0),
                    1.0,
                ),
                "cluster_min_samples": max(
                    int(static_cluster_min_samples),
                    1,
                ),
                "reference_update_unit": "processed detection update",
                "reference_policy": (
                    "startup median, adaptive except around motion-qualified "
                    "and validated targets"
                ),
                "noise_policy": (
                    "per-cell calibration variability with common-mode "
                    "gain suppression and temporal smoothing"
                ),
                "validation_policy": (
                    "recent dynamic motion followed by a persistent nearby "
                    "static local maximum or cluster across three associated "
                    "updates"
                ),
            },
            "micro_doppler_units": "dB power",
            "micro_doppler_axis": "centered Doppler bin",
            "micro_doppler_windows_layout": ["window", "centered_doppler_bin"],
            "micro_doppler_processing": {
                "mode": "per-TX TDM STFT",
                "window": "Hann",
                "window_loops": MICRO_DOPPLER_WINDOW_LOOPS,
                "hop_loops": MICRO_DOPPLER_HOP_LOOPS,
                "fft_size": MICRO_DOPPLER_FFT_SIZE,
                "tx_rx_combination": "incoherent power sum",
            },
            "rotor_micro_doppler_processing": rotor_processing,
            "classification": (
                classification_metadata
                if classification_metadata is not None
                else {"enabled": False}
            ),
            "radar_config": {
                "num_adc_samples": config.num_adc_samples,
                "num_rx_channels": config.num_rx_channels,
                "num_chirps_per_frame": config.num_chirps_per_frame,
                "num_loops": config.num_loops,
                "num_chirps_per_loop": config.num_chirps_per_loop,
                "tx_channel_masks": config.tx_channel_masks,
                "sample_rate_ksps": config.sample_rate_ksps,
                "frequency_slope_mhz_per_us": config.frequency_slope_mhz_per_us,
                "start_frequency_ghz": config.start_frequency_ghz,
                "idle_time_us": config.idle_time_us,
                "ramp_end_time_us": config.ramp_end_time_us,
                "frame_periodicity_ms": config.frame_periodicity_ms,
                "slow_time_rate_hz": config.slow_time_rate_hz,
                "frame_duty_cycle": config.frame_duty_cycle,
                "range_resolution_m": config.range_resolution_m,
            },
        }
        self.file.write(json.dumps(metadata, separators=(",", ":")) + "\n")
        self.emit(f"Saving processed radar output to {self.output_path}")

    @property
    def enabled(self) -> bool:
        return self.file is not None

    def write_update(
        self,
        *,
        frame_index: int,
        points: np.ndarray,
        clusters: np.ndarray,
        target_track: Optional[TargetTrack],
        micro_doppler_db: np.ndarray,
        selected_range_m: Optional[float],
        micro_doppler_windows_db: Optional[np.ndarray] = None,
        static_points: Optional[np.ndarray] = None,
        static_clusters: Optional[np.ndarray] = None,
        static_reference: Optional[StaticReferenceStatus] = None,
        target_source: Optional[str] = None,
        static_candidate_count: int = 0,
        static_validation: str = "disabled",
        rotor_micro_doppler: Optional[MicroDopplerResult] = None,
        classification: Optional[InferenceResult] = None,
    ) -> None:
        if self.file is None:
            return

        track = None
        if target_track is not None:
            track = {
                "position_m": list(target_track.position_m),
                "velocity_m_per_update": list(target_track.velocity_m_per_update),
                "age_updates": target_track.age_updates,
                "hits": target_track.hits,
                "missed_updates": target_track.missed_updates,
                "confirmed": target_track.confirmed,
                "predicted": target_track.is_predicted,
            }
        record = {
            "record_type": "update",
            "update_index": self.updates_saved,
            "processed_frame_index": int(frame_index),
            "recorded_at": datetime.now().astimezone().isoformat(
                timespec="milliseconds"
            ),
            "points": np.asarray(points, dtype=np.float32).tolist(),
            "clusters": np.asarray(clusters, dtype=np.float32).tolist(),
            "static_points": np.asarray(
                static_points
                if static_points is not None
                else np.empty((0, 5), dtype=np.float32),
                dtype=np.float32,
            ).tolist(),
            "static_clusters": np.asarray(
                static_clusters
                if static_clusters is not None
                else np.empty((0, 4), dtype=np.float32),
                dtype=np.float32,
            ).tolist(),
            "static_reference": (
                {
                    "enabled": static_reference.enabled,
                    "ready": static_reference.ready,
                    "frames_seen": static_reference.frames_seen,
                    "required_frames": static_reference.required_frames,
                    "warmup_frames_seen": static_reference.warmup_frames_seen,
                    "warmup_frames_required": (
                        static_reference.warmup_frames_required
                    ),
                    "adaptive": static_reference.adaptive,
                }
                if static_reference is not None
                else None
            ),
            "target_track": track,
            "target_source": target_source,
            "static_candidate_count": max(int(static_candidate_count), 0),
            "static_validation": str(static_validation),
            "micro_doppler_db": np.asarray(
                micro_doppler_db,
                dtype=np.float32,
            ).tolist(),
            "micro_doppler_windows_db": np.asarray(
                micro_doppler_windows_db
                if micro_doppler_windows_db is not None
                else np.asarray(micro_doppler_db)[:, np.newaxis],
                dtype=np.float32,
            ).T.tolist(),
            "selected_range_m": (
                float(selected_range_m) if selected_range_m is not None else None
            ),
            "rotor_micro_doppler": (
                {
                    "raw_spectrogram_db": np.asarray(
                        np.round(
                            np.asarray(
                                rotor_micro_doppler.raw_spectrogram_db,
                                dtype=np.float64,
                            ),
                            decimals=ROTOR_SPECTRUM_OUTPUT_DECIMALS,
                        ),
                        dtype=np.float64,
                    ).T.tolist(),
                    "enhanced_spectrogram_db": np.asarray(
                        np.round(
                            np.asarray(
                                rotor_micro_doppler.enhanced_spectrogram_db,
                                dtype=np.float64,
                            ),
                            decimals=ROTOR_SPECTRUM_OUTPUT_DECIMALS,
                        ),
                        dtype=np.float64,
                    ).T.tolist(),
                    "window_times_s": np.asarray(
                        rotor_micro_doppler.window_times_s,
                        dtype=np.float64,
                    ).tolist(),
                    "velocity_axis_m_s": np.asarray(
                        rotor_micro_doppler.velocity_axis_m_s,
                        dtype=np.float32,
                    ).tolist(),
                    "flash_scores_db": np.asarray(
                        rotor_micro_doppler.flash_scores_db,
                        dtype=np.float32,
                    ).tolist(),
                    "noise_floor_db": np.asarray(
                        rotor_micro_doppler.noise_floor_db,
                        dtype=np.float32,
                    ).tolist(),
                    "noise_gate_db": np.asarray(
                        rotor_micro_doppler.noise_gate_db
                        if rotor_micro_doppler.noise_gate_db is not None
                        else np.empty((0,), dtype=np.float32),
                        dtype=np.float32,
                    ).tolist(),
                    "selected_range_m": float(
                        rotor_micro_doppler.selected_range_m
                    ),
                    "nominal_hop_s": float(
                        rotor_micro_doppler.nominal_hop_s
                    ),
                    "unambiguous_velocity_m_s": (
                        float(rotor_micro_doppler.unambiguous_velocity_m_s)
                        if rotor_micro_doppler.unambiguous_velocity_m_s
                        is not None
                        else None
                    ),
                    "velocity_alias_risk": bool(
                        rotor_micro_doppler.velocity_alias_risk
                    ),
                    "alias_warning": rotor_micro_doppler.alias_warning,
                    "rotor_estimates": [
                        {
                            "blade_passage_hz": estimate.blade_passage_hz,
                            "rpm": estimate.rpm,
                            "confidence": estimate.confidence,
                            "harmonic_rank": estimate.harmonic_rank,
                            "velocity_alias_risk": (
                                estimate.velocity_alias_risk
                            ),
                        }
                        for estimate in rotor_micro_doppler.rotor_estimates
                    ],
                }
                if rotor_micro_doppler is not None
                else None
            ),
            "classification": (
                classification.to_dict()
                if classification is not None
                else None
            ),
        }
        self.file.write(json.dumps(record, separators=(",", ":")) + "\n")
        self.updates_saved += 1

    def close(self) -> None:
        if self.file is None:
            return
        self.file.close()
        self.file = None
        self.emit(
            f"Processed radar output saved: updates={self.updates_saved}, "
            f"path={self.output_path}"
        )


class DisplayPayloadSink:
    def __init__(
        self,
        mode: str,
        update_every: int,
        payload_queue: Optional[mp.Queue],
        config: RadarCaptureConfig,
        max_range_m: float = DEFAULT_MAX_RANGE_M,
        point_cloud_fov_deg: float = DEFAULT_POINT_CLOUD_FOV_DEG,
        cluster_eps_m: float = DEFAULT_CLUSTER_EPS_M,
        cluster_min_samples: int = DEFAULT_CLUSTER_MIN_SAMPLES,
        clutter_map_update_rate: float = DEFAULT_CLUTTER_MAP_UPDATE_RATE,
        clutter_map_warmup_frames: int = DEFAULT_CLUTTER_MAP_WARMUP_FRAMES,
        clutter_map_min_snr_db: float = DEFAULT_CLUTTER_MAP_MIN_SNR_DB,
        processed_writer: Optional[ProcessedOutputWriter] = None,
        display_skipped_counter: Optional[Any] = None,
        static_detection: bool = DEFAULT_STATIC_DETECTION,
        static_warmup_frames: int = DEFAULT_STATIC_WARMUP_FRAMES,
        static_reference_frames: int = DEFAULT_STATIC_REFERENCE_FRAMES,
        static_min_change_db: float = DEFAULT_STATIC_MIN_CHANGE_DB,
        static_background_update_rate: float = (
            DEFAULT_STATIC_BACKGROUND_UPDATE_RATE
        ),
        static_cluster_min_samples: int = DEFAULT_STATIC_CLUSTER_MIN_SAMPLES,
        micro_doppler_range_m: Optional[float] = None,
        micro_doppler_range_half_width_bins: int = (
            ROTOR_MICRO_DOPPLER_RANGE_HALF_WIDTH_BINS
        ),
        rotor_blades: int = DEFAULT_ROTOR_BLADES,
        rotor_count: int = 1,
        rotor_radius_m: Optional[float] = None,
        rotor_rpm_min: float = 500.0,
        rotor_rpm_max: float = DEFAULT_ROTOR_RPM_MAX,
        inference_engine: Optional[DroneBirdInference] = None,
        classification_emit_func: Optional[EmitFunc] = None,
        rotor_post_queue: Optional[mp.Queue] = None,
        rotor_post_failure_event: Optional[Any] = None,
        rotor_post_queue_high_water: Optional[Any] = None,
    ) -> None:
        self.mode = mode
        self.update_every = max(update_every, 1)
        self.payload_queue = payload_queue
        self.config = config
        self.max_range_m = max(float(max_range_m), 0.0)
        self.point_cloud_fov_deg = min(max(float(point_cloud_fov_deg), 0.0), 90.0)
        self.cluster_eps_m = max(float(cluster_eps_m), 0.0)
        self.cluster_min_samples = max(int(cluster_min_samples), 1)
        self.processed_writer = processed_writer
        self.display_skipped_counter = display_skipped_counter
        self.rotor_post_queue = rotor_post_queue
        self.rotor_post_failure_event = rotor_post_failure_event
        self.rotor_post_queue_high_water = rotor_post_queue_high_water
        self.clutter_map = (
            AdaptiveClutterMap(
                update_rate=clutter_map_update_rate,
                warmup_frames=clutter_map_warmup_frames,
                minimum_snr_db=clutter_map_min_snr_db,
            )
            if clutter_map_update_rate > 0.0
            else None
        )
        self.static_scene_map = (
            StaticSceneMap(
                warmup_frames=static_warmup_frames,
                reference_frames=static_reference_frames,
                minimum_change_db=static_min_change_db,
                background_update_rate=static_background_update_rate,
            )
            if static_detection
            else None
        )
        self.dynamic_target_tracker = SingleTargetTracker()
        self.static_target_tracker = SingleTargetTracker(
            acquisition_policy="nearest",
            max_missed_updates=STATIC_TRACK_MAX_MISSED_UPDATES,
        )
        self.motion_handoff = MotionHandoffQualifier()
        self.static_cluster_min_samples = max(
            int(static_cluster_min_samples),
            1,
        )
        # Retain the old attribute for callers that inspect the dynamic tracker.
        self.target_tracker = self.dynamic_target_tracker
        self.micro_doppler_history = deque(maxlen=MICRO_DOPPLER_HISTORY_UPDATES)
        self.micro_doppler_history_target_position_m: Optional[np.ndarray] = None
        self.micro_doppler_history_gap_updates = 0
        self.latest_micro_doppler_db = np.empty((0,), dtype=np.float32)
        self.latest_micro_doppler_windows_db = np.empty((0, 0), dtype=np.float32)
        self.frame_count = 0
        self.timings = ProcessingTimingStats()
        self.static_candidate_total = 0
        self.static_validated_updates = 0
        self.static_handoff_pending_updates = 0
        self.micro_doppler_range_m = (
            float(micro_doppler_range_m)
            if micro_doppler_range_m is not None
            else None
        )
        self.micro_doppler_range_half_width_bins = max(
            int(micro_doppler_range_half_width_bins),
            0,
        )
        self.rotor_blades = max(int(rotor_blades), 1)
        self.rotor_count = max(int(rotor_count), 1)
        self.rotor_radius_m = (
            float(rotor_radius_m) if rotor_radius_m is not None else None
        )
        self.rotor_rpm_min = max(float(rotor_rpm_min), 1.0)
        self.rotor_rpm_max = max(float(rotor_rpm_max), self.rotor_rpm_min + 1.0)
        self.rotor_raw_history: deque[np.ndarray] = deque()
        self.rotor_enhanced_history: deque[np.ndarray] = deque()
        self.rotor_time_history: deque[float] = deque()
        self.rotor_flash_history: deque[float] = deque()
        self.rotor_noise_history: deque[float] = deque()
        self.capture_time_origin_s: Optional[float] = None
        self.latest_rotor_estimates: tuple[RotorEstimate, ...] = ()
        self.last_rotor_estimate_time_s: Optional[float] = None
        self.inference_engine = inference_engine
        self.classification_emit = classification_emit_func
        self.latest_classification: Optional[InferenceResult] = (
            inference_engine.unknown("no_target")
            if inference_engine is not None
            else None
        )
        self._last_classification_emit_s = 0.0
        self._last_classification_signature: Optional[
            tuple[str, Optional[str]]
        ] = None
        self._classification_owner_position_m: Optional[np.ndarray] = None

    def update(
        self,
        range_fft: np.ndarray,
        range_axis_m: Optional[np.ndarray],
        *,
        captured_at_s: Optional[float] = None,
    ) -> None:
        save_processed = bool(
            self.processed_writer is not None and self.processed_writer.enabled
        )
        if (
            self.mode == "none"
            and not save_processed
            and self.inference_engine is None
            and self.rotor_post_queue is None
        ):
            return
        if (
            self.payload_queue is None
            and not save_processed
            and self.inference_engine is None
            and self.rotor_post_queue is None
        ):
            return

        self.frame_count += 1
        publish_update = self.frame_count % self.update_every == 0
        if (
            not publish_update
            and self.inference_engine is None
            and self.rotor_post_queue is None
        ):
            return

        if self.mode == ROTOR_DISPLAY_MODE:
            frame_period_s = (
                float(self.config.frame_periodicity_ms) * 1e-3
                if self.config.frame_periodicity_ms is not None
                else 0.0
            )
            rotor_display_every = (
                max(
                    int(round(1.0 / (frame_period_s * ROTOR_DISPLAY_MAX_RATE_HZ))),
                    1,
                )
                if frame_period_s > 0.0
                else 1
            )
            publish_rotor_display = (
                publish_update
                and self.payload_queue is not None
                and (
                    self.frame_count == 1
                    or self.frame_count % rotor_display_every == 0
                )
            )
            rotor_payload, frame_result = self._compute_rotor_payload(
                range_fft,
                range_axis_m,
                captured_at_s=captured_at_s,
                build_display_payload=publish_rotor_display,
            )
            if self.rotor_post_queue is not None:
                doppler_started = time.perf_counter()
                doppler_cube = compute_range_doppler_fft(
                    range_fft,
                    self.config,
                )
                self.timings.add(
                    "doppler",
                    time.perf_counter() - doppler_started,
                )
                feature_started = time.perf_counter()
                feature_step = self._fixed_range_feature_step(
                    doppler_cube,
                    range_axis_m,
                    self.micro_doppler_range_m,
                )
                self.timings.add(
                    "classification_feature",
                    time.perf_counter() - feature_started,
                )
                self._enqueue_rotor_postprocess(
                    RotorPostprocessItem(
                        frame_index=self.frame_count,
                        save_update=publish_update,
                        feature_step=feature_step,
                        rotor_result=frame_result,
                    )
                )
            elif self.inference_engine is not None:
                doppler_started = time.perf_counter()
                doppler_cube = compute_range_doppler_fft(
                    range_fft,
                    self.config,
                )
                self.timings.add(
                    "doppler",
                    time.perf_counter() - doppler_started,
                )
                self._classify_fixed_range(
                    doppler_cube,
                    range_axis_m,
                    self.micro_doppler_range_m,
                )
            if not publish_update:
                return
            if save_processed:
                assert self.processed_writer is not None
                serialization_started = time.perf_counter()
                self.processed_writer.write_update(
                    frame_index=self.frame_count,
                    points=np.empty((0, 4), dtype=np.float32),
                    clusters=np.empty((0, 4), dtype=np.float32),
                    target_track=None,
                    micro_doppler_db=(
                        frame_result.raw_spectrogram_db[:, -1]
                        if frame_result.raw_spectrogram_db.size
                        else np.empty((0,), dtype=np.float32)
                    ),
                    micro_doppler_windows_db=(
                        frame_result.raw_spectrogram_db[:, -1:]
                        if frame_result.raw_spectrogram_db.size
                        else np.empty((0, 0), dtype=np.float32)
                    ),
                    selected_range_m=frame_result.selected_range_m,
                    rotor_micro_doppler=frame_result,
                    classification=self.latest_classification,
                )
                self.timings.add(
                    "serialization",
                    time.perf_counter() - serialization_started,
                )
            if self.payload_queue is not None and rotor_payload is not None:
                skipped_updates = _put_latest_queue_payload(
                    self.payload_queue,
                    rotor_payload,
                )
                _increment_shared_counter(
                    self.display_skipped_counter,
                    skipped_updates,
                )
            return

        combined_payload = None
        if (
            save_processed
            or self.mode == COMBINED_DISPLAY_MODE
            or self.inference_engine is not None
        ):
            combined_payload = self._compute_combined_payload(
                range_fft,
                range_axis_m,
                build_micro_doppler=publish_update,
            )
            if not publish_update:
                return
            if save_processed:
                assert self.processed_writer is not None
                point_cloud = combined_payload.point_cloud
                serialization_started = time.perf_counter()
                self.processed_writer.write_update(
                    frame_index=self.frame_count,
                    points=point_cloud.points,
                    clusters=point_cloud.clusters,
                    static_points=point_cloud.static_points,
                    static_clusters=point_cloud.static_clusters,
                    static_reference=point_cloud.static_reference,
                    target_track=point_cloud.target_track,
                    target_source=point_cloud.target_source,
                    static_candidate_count=point_cloud.static_candidate_count,
                    static_validation=point_cloud.static_validation,
                    micro_doppler_db=self.latest_micro_doppler_db,
                    micro_doppler_windows_db=self.latest_micro_doppler_windows_db,
                    selected_range_m=combined_payload.selected_range_m,
                    classification=self.latest_classification,
                )
                self.timings.add(
                    "serialization",
                    time.perf_counter() - serialization_started,
                )

        if self.mode == "none":
            return
        if self.mode == "range":
            payload = (
                range_axis_m,
                compute_range_profile(range_fft),
            )
        elif self.mode == "range-doppler":
            payload = (
                range_axis_m,
                compute_range_doppler_heatmap(range_fft, self.config),
            )
        elif self.mode == "point-cloud":
            if combined_payload is not None:
                payload = combined_payload.point_cloud
            else:
                payload, _doppler_cube = self._compute_point_cloud_payload(
                    range_fft,
                    range_axis_m,
                )
        elif self.mode == COMBINED_DISPLAY_MODE:
            assert combined_payload is not None
            payload = combined_payload
        else:
            return

        if self.payload_queue is not None:
            skipped_updates = _put_latest_queue_payload(
                self.payload_queue,
                payload,
            )
            _increment_shared_counter(
                self.display_skipped_counter,
                skipped_updates,
            )

    def _fixed_range_feature_step(
        self,
        doppler_cube: np.ndarray,
        range_axis_m: Optional[np.ndarray],
        target_range_m: Optional[float],
    ) -> np.ndarray:
        range_axis = (
            np.asarray(range_axis_m, dtype=np.float64)
            if range_axis_m is not None
            else np.empty((0,), dtype=np.float64)
        )
        if (
            target_range_m is None
            or not np.isfinite(target_range_m)
            or range_axis.size != doppler_cube.shape[-1]
            or not np.isfinite(range_axis).all()
        ):
            raise ValueError("Dedicated classification target range is invalid")
        target_range_bin = int(
            np.argmin(np.abs(range_axis - float(target_range_m)))
        )
        return doppler_cube_to_feature_step(doppler_cube, target_range_bin)

    def _enqueue_rotor_postprocess(self, item: RotorPostprocessItem) -> None:
        if self.rotor_post_queue is None:
            return
        while True:
            if (
                self.rotor_post_failure_event is not None
                and self.rotor_post_failure_event.is_set()
            ):
                raise RuntimeError("Rotor post-processing worker failed")
            try:
                self.rotor_post_queue.put(item, timeout=0.1)
                break
            except queue.Full:
                continue
        if self.rotor_post_queue_high_water is None:
            return
        try:
            depth = max(int(self.rotor_post_queue.qsize()), 0)
        except (AttributeError, NotImplementedError):
            return
        with self.rotor_post_queue_high_water.get_lock():
            self.rotor_post_queue_high_water.value = max(
                int(self.rotor_post_queue_high_water.value),
                depth,
            )

    def _compute_rotor_payload(
        self,
        range_fft: np.ndarray,
        range_axis_m: Optional[np.ndarray],
        *,
        captured_at_s: Optional[float],
        build_display_payload: bool = True,
    ) -> tuple[Optional[RotorDisplayPayload], MicroDopplerResult]:
        if self.micro_doppler_range_m is None:
            raise ValueError(
                "Dedicated micro-Doppler mode requires --micro-doppler-range-m"
            )
        if captured_at_s is not None:
            if self.capture_time_origin_s is None:
                self.capture_time_origin_s = float(captured_at_s)
            frame_time_s = float(captured_at_s) - self.capture_time_origin_s
        else:
            frame_period_s = (
                float(self.config.frame_periodicity_ms) * 1e-3
                if self.config.frame_periodicity_ms is not None
                else 0.0
            )
            frame_time_s = (self.frame_count - 1) * frame_period_s

        micro_doppler_started = time.perf_counter()
        frame_result = compute_rotor_micro_doppler_frame(
            range_fft,
            range_axis_m,
            self.config,
            target_range_m=self.micro_doppler_range_m,
            frame_time_s=frame_time_s,
            range_half_width_bins=self.micro_doppler_range_half_width_bins,
            window_loops=ROTOR_MICRO_DOPPLER_WINDOW_LOOPS,
            hop_loops=ROTOR_MICRO_DOPPLER_HOP_LOOPS,
            fft_size=ROTOR_MICRO_DOPPLER_FFT_SIZE,
            dc_notch_bins=ROTOR_MICRO_DOPPLER_DC_NOTCH_BINS,
            rotor_radius_m=self.rotor_radius_m,
            rotor_rpm_max=self.rotor_rpm_max,
        )
        for index, window_time_s in enumerate(frame_result.window_times_s):
            self.rotor_raw_history.append(
                frame_result.raw_spectrogram_db[:, index]
            )
            self.rotor_enhanced_history.append(
                frame_result.enhanced_spectrogram_db[:, index]
            )
            self.rotor_time_history.append(float(window_time_s))
            self.rotor_flash_history.append(
                float(frame_result.flash_scores_db[index])
            )
            self.rotor_noise_history.append(
                float(frame_result.noise_floor_db[index])
            )
        if self.rotor_time_history:
            cutoff_s = (
                self.rotor_time_history[-1]
                - ROTOR_MICRO_DOPPLER_HISTORY_SECONDS
            )
            while (
                self.rotor_time_history
                and self.rotor_time_history[0] < cutoff_s
            ):
                self.rotor_raw_history.popleft()
                self.rotor_enhanced_history.popleft()
                self.rotor_time_history.popleft()
                self.rotor_flash_history.popleft()
                self.rotor_noise_history.popleft()

        history_times = np.asarray(self.rotor_time_history, dtype=np.float64)
        history_scores = np.asarray(
            self.rotor_flash_history,
            dtype=np.float32,
        )
        newest_history_time_s = (
            float(history_times[-1]) if history_times.size else None
        )
        estimate_due = (
            newest_history_time_s is not None
            and (
                self.last_rotor_estimate_time_s is None
                or newest_history_time_s - self.last_rotor_estimate_time_s
                >= ROTOR_RPM_ESTIMATE_UPDATE_SECONDS
            )
        )
        if estimate_due:
            self.latest_rotor_estimates = estimate_rotor_rpm(
                history_times,
                history_scores,
                blade_count=self.rotor_blades,
                rotor_count=self.rotor_count,
                rpm_min=self.rotor_rpm_min,
                rpm_max=self.rotor_rpm_max,
                velocity_alias_risk=frame_result.velocity_alias_risk,
            )
            self.last_rotor_estimate_time_s = newest_history_time_s
        rotor_payload = None
        if build_display_payload:
            display_window_limit = max(
                int(
                    np.floor(
                        ROTOR_DISPLAY_HISTORY_SECONDS
                        / max(frame_result.nominal_hop_s, 1e-9)
                    )
                )
                + 1,
                2,
            )
            display_history_start = max(
                int(history_times.size) - display_window_limit,
                0,
            )
            enhanced_history = tuple(self.rotor_enhanced_history)
            display_window_count = (
                int(history_times.size) - display_history_start
            )
            history_result = MicroDopplerResult(
                # The GUI only draws the enhanced spectrum. Raw spectra and
                # noise floors remain in processed output but are deliberately
                # omitted from the cross-process display payload.
                raw_spectrogram_db=np.empty((0, 0), dtype=np.float32),
                enhanced_spectrogram_db=(
                    np.stack(
                        enhanced_history[display_history_start:],
                        axis=1,
                    )
                    if enhanced_history
                    else np.empty((0, 0), dtype=np.float32)
                ),
                # The GUI intentionally concatenates only measured STFT
                # centres. Physical capture timestamps remain in frame_result
                # for RPM estimation and processed output.
                window_times_s=_concatenated_active_window_times(
                    display_window_count,
                    frame_result.nominal_hop_s,
                ),
                velocity_axis_m_s=frame_result.velocity_axis_m_s,
                flash_scores_db=history_scores[display_history_start:],
                noise_floor_db=np.empty((0,), dtype=np.float32),
                selected_range_m=frame_result.selected_range_m,
                nominal_hop_s=frame_result.nominal_hop_s,
                unambiguous_velocity_m_s=(
                    frame_result.unambiguous_velocity_m_s
                ),
                noise_gate_db=None,
                rotor_estimates=self.latest_rotor_estimates,
                velocity_alias_risk=frame_result.velocity_alias_risk,
                alias_warning=frame_result.alias_warning,
            )
            rotor_payload = RotorDisplayPayload(history_result)
        frame_result_with_estimates = MicroDopplerResult(
            raw_spectrogram_db=frame_result.raw_spectrogram_db,
            enhanced_spectrogram_db=frame_result.enhanced_spectrogram_db,
            window_times_s=frame_result.window_times_s,
            velocity_axis_m_s=frame_result.velocity_axis_m_s,
            flash_scores_db=frame_result.flash_scores_db,
            noise_floor_db=frame_result.noise_floor_db,
            selected_range_m=frame_result.selected_range_m,
            nominal_hop_s=frame_result.nominal_hop_s,
            unambiguous_velocity_m_s=(
                frame_result.unambiguous_velocity_m_s
            ),
            noise_gate_db=frame_result.noise_gate_db,
            rotor_estimates=self.latest_rotor_estimates,
            velocity_alias_risk=frame_result.velocity_alias_risk,
            alias_warning=frame_result.alias_warning,
        )
        self.timings.add(
            "micro_doppler",
            time.perf_counter() - micro_doppler_started,
        )
        return rotor_payload, frame_result_with_estimates

    def _compute_combined_payload(
        self,
        range_fft: np.ndarray,
        range_axis_m: Optional[np.ndarray],
        *,
        build_micro_doppler: bool = True,
    ) -> CombinedDisplayPayload:
        point_cloud, doppler_cube = self._compute_point_cloud_payload(
            range_fft,
            range_axis_m,
        )
        target_track = point_cloud.target_track
        target_changed = self._update_micro_doppler_history_owner(target_track)
        self._classify_tracked_target(
            doppler_cube,
            range_axis_m,
            target_track,
            target_changed=target_changed,
        )
        if not build_micro_doppler:
            return CombinedDisplayPayload(
                point_cloud=point_cloud,
                spectrogram_db=np.empty((0, 0), dtype=np.float32),
                selected_range_m=(
                    target_track.range_m
                    if target_track is not None
                    else None
                ),
            )

        micro_doppler_started = time.perf_counter()
        selected_range_m = None
        if target_track is not None:
            _spectrum_db, selected_range_m = compute_micro_doppler_spectrum(
                doppler_cube,
                range_axis_m,
                target_range_m=target_track.range_m,
                range_half_width_bins=MICRO_DOPPLER_RANGE_HALF_WIDTH_BINS,
                max_range_m=self.max_range_m,
            )
        short_time_spectra = np.empty((0, 0), dtype=np.float32)
        if selected_range_m is not None:
            short_time_spectra = compute_per_tx_micro_doppler_spectrogram(
                range_fft,
                range_axis_m,
                self.config,
                target_range_m=selected_range_m,
                range_half_width_bins=MICRO_DOPPLER_RANGE_HALF_WIDTH_BINS,
                window_loops=MICRO_DOPPLER_WINDOW_LOOPS,
                hop_loops=MICRO_DOPPLER_HOP_LOOPS,
                fft_size=MICRO_DOPPLER_FFT_SIZE,
            )
        if short_time_spectra.size:
            self.latest_micro_doppler_db = short_time_spectra[:, -1]
            self.latest_micro_doppler_windows_db = short_time_spectra
            if (
                self.micro_doppler_history
                and self.micro_doppler_history[0].shape
                != self.latest_micro_doppler_db.shape
            ):
                self.micro_doppler_history.clear()
            self.micro_doppler_history.extend(
                short_time_spectra[:, index]
                for index in range(short_time_spectra.shape[1])
            )
        else:
            self.latest_micro_doppler_db = np.empty((0,), dtype=np.float32)
            self.latest_micro_doppler_windows_db = np.empty((0, 0), dtype=np.float32)
        spectrogram_db = (
            np.stack(tuple(self.micro_doppler_history), axis=1)
            if self.micro_doppler_history
            else np.empty((0, 0), dtype=np.float32)
        )
        self.timings.add(
            "micro_doppler",
            time.perf_counter() - micro_doppler_started,
        )
        return CombinedDisplayPayload(
            point_cloud=point_cloud,
            spectrogram_db=spectrogram_db,
            selected_range_m=selected_range_m,
        )

    def _classify_tracked_target(
        self,
        doppler_cube: np.ndarray,
        range_axis_m: Optional[np.ndarray],
        target_track: Optional[TargetTrack],
        *,
        target_changed: bool,
    ) -> None:
        if self.inference_engine is None:
            return
        if target_track is None:
            self._classification_owner_position_m = None
            self._set_classification(
                self.inference_engine.reset("no_confirmed_target")
            )
            return
        if target_track.is_predicted:
            self._classification_owner_position_m = None
            self._set_classification(
                self.inference_engine.reset("predicted_target")
            )
            return
        target_position_m = np.asarray(
            target_track.position_m,
            dtype=np.float64,
        )
        if self._classification_owner_position_m is not None:
            target_changed = target_changed or (
                float(
                    np.linalg.norm(
                        target_position_m
                        - self._classification_owner_position_m
                    )
                )
                > MICRO_DOPPLER_HISTORY_ASSOCIATION_DISTANCE_M
            )
        if target_changed:
            self._classification_owner_position_m = target_position_m.copy()
            self._set_classification(
                self.inference_engine.reset("target_changed")
            )
            return
        self._classification_owner_position_m = target_position_m.copy()
        self._classify_fixed_range(
            doppler_cube,
            range_axis_m,
            target_track.range_m,
        )

    def _classify_fixed_range(
        self,
        doppler_cube: np.ndarray,
        range_axis_m: Optional[np.ndarray],
        target_range_m: Optional[float],
    ) -> None:
        if self.inference_engine is None:
            return
        range_axis = (
            np.asarray(range_axis_m, dtype=np.float64)
            if range_axis_m is not None
            else np.empty((0,), dtype=np.float64)
        )
        if (
            target_range_m is None
            or not np.isfinite(target_range_m)
            or range_axis.size != doppler_cube.shape[-1]
            or not np.isfinite(range_axis).all()
        ):
            self._set_classification(
                self.inference_engine.reset("invalid_target_range")
            )
            return
        target_range_bin = int(
            np.argmin(np.abs(range_axis - float(target_range_m)))
        )
        self._set_classification(
            self.inference_engine.update(
                doppler_cube,
                target_range_bin,
            )
        )

    def _set_classification(self, result: InferenceResult) -> None:
        self.latest_classification = result
        if self.classification_emit is None:
            return
        signature = (result.label, result.reason)
        now = time.monotonic()
        if (
            signature == self._last_classification_signature
            and now - self._last_classification_emit_s < 1.0
        ):
            return
        self.classification_emit(
            CLASSIFICATION_RESULT_PREFIX
            + json.dumps(result.to_dict(), separators=(",", ":"))
        )
        self._last_classification_signature = signature
        self._last_classification_emit_s = now

    def _update_micro_doppler_history_owner(
        self,
        target_track: Optional[TargetTrack],
    ) -> bool:
        """Update history ownership and report a discontinuous target change."""
        if target_track is None:
            if self.micro_doppler_history_target_position_m is not None:
                self.micro_doppler_history_gap_updates += 1
            return False

        target_position_m = np.asarray(
            target_track.position_m,
            dtype=np.float64,
        )
        owner_position_m = self.micro_doppler_history_target_position_m
        target_changed = False
        reacquired_after_gap = (
            owner_position_m is not None
            and self.micro_doppler_history_gap_updates > 0
        )
        if reacquired_after_gap:
            gap_expired = (
                self.micro_doppler_history_gap_updates
                > MICRO_DOPPLER_HISTORY_MAX_GAP_UPDATES
            )
            target_distance_m = float(
                np.linalg.norm(target_position_m - owner_position_m)
            )
            target_changed = (
                gap_expired
                or target_distance_m
                > MICRO_DOPPLER_HISTORY_ASSOCIATION_DISTANCE_M
            )

        if target_changed:
            self.micro_doppler_history.clear()
            self.latest_micro_doppler_db = np.empty((0,), dtype=np.float32)
            self.latest_micro_doppler_windows_db = np.empty(
                (0, 0),
                dtype=np.float32,
            )
        self.micro_doppler_history_target_position_m = target_position_m.copy()
        self.micro_doppler_history_gap_updates = 0
        return target_changed

    def _compute_point_cloud_payload(
        self,
        range_fft: np.ndarray,
        range_axis_m: Optional[np.ndarray],
    ) -> tuple[PointCloudDisplayPayload, np.ndarray]:
        doppler_started = time.perf_counter()
        doppler_cube = compute_range_doppler_fft(range_fft, self.config)
        self.timings.add("doppler", time.perf_counter() - doppler_started)
        dynamic_started = time.perf_counter()
        points = compute_point_cloud(
            range_fft,
            range_axis_m,
            self.config,
            doppler_cube=doppler_cube,
            clutter_map=self.clutter_map,
            max_range_m=self.max_range_m,
            azimuth_fov_deg=self.point_cloud_fov_deg,
            elevation_fov_deg=self.point_cloud_fov_deg,
        )
        self.timings.add(
            "dynamic_detection",
            time.perf_counter() - dynamic_started,
        )
        clustering_started = time.perf_counter()
        clusters = cluster_point_cloud(
            points,
            eps_m=self.cluster_eps_m,
            min_samples=self.cluster_min_samples,
        )
        dynamic_track = self.dynamic_target_tracker.update(
            _point_cloud_track_candidates(
                points,
                clusters,
                assignment_distance_m=max(2.0 * self.cluster_eps_m, 1e-6),
            )
        )
        clustering_elapsed = time.perf_counter() - clustering_started

        raw_static_points = np.empty((0, 5), dtype=np.float32)
        raw_static_clusters = np.empty((0, 4), dtype=np.float32)
        static_points = np.empty((0, 5), dtype=np.float32)
        static_clusters = np.empty((0, 4), dtype=np.float32)
        static_track = None
        static_candidate_count = 0
        static_validation = "disabled"
        handoff_position = self.motion_handoff.update(dynamic_track)
        if self.static_scene_map is not None:
            static_started = time.perf_counter()
            raw_static_points = compute_static_point_cloud(
                doppler_cube,
                range_axis_m,
                self.config,
                self.static_scene_map,
                max_range_m=self.max_range_m,
                azimuth_fov_deg=self.point_cloud_fov_deg,
                elevation_fov_deg=self.point_cloud_fov_deg,
            )
            self.timings.add(
                "static_detection",
                time.perf_counter() - static_started,
            )
            static_candidate_count = int(raw_static_points.shape[0])
            self.static_candidate_total += static_candidate_count
            clustering_started = time.perf_counter()
            raw_static_clusters, raw_static_labels = (
                cluster_point_cloud_with_labels(
                    raw_static_points,
                    eps_m=self.cluster_eps_m,
                    min_samples=self.static_cluster_min_samples,
                )
            )
            clustering_elapsed += time.perf_counter() - clustering_started
            cluster_candidates = np.empty((0, 4), dtype=np.float32)
            if raw_static_clusters.size:
                cluster_candidates = _point_cloud_track_candidates(
                    raw_static_points,
                    raw_static_clusters,
                    assignment_distance_m=max(
                        2.0 * self.cluster_eps_m,
                        1e-6,
                    ),
                )
            if not self.static_target_tracker.is_confirmed:
                if handoff_position is None or not cluster_candidates.size:
                    cluster_candidates = np.empty((0, 4), dtype=np.float32)
                else:
                    handoff_distances = np.linalg.norm(
                        cluster_candidates[:, :3] - handoff_position,
                        axis=1,
                    )
                    cluster_candidates = cluster_candidates[
                        handoff_distances <= STATIC_HANDOFF_DISTANCE_M
                    ]

            static_track = self.static_target_tracker.update(cluster_candidates)
            if static_track is not None and static_track.confirmed:
                static_validation = "validated"
                self.static_validated_updates += 1
                if raw_static_clusters.size and not static_track.is_predicted:
                    track_position = np.asarray(
                        static_track.position_m,
                        dtype=np.float64,
                    )
                    cluster_distances = np.linalg.norm(
                        raw_static_clusters[:, :3] - track_position,
                        axis=1,
                    )
                    selected_cluster_index = int(np.argmin(cluster_distances))
                    if (
                        cluster_distances[selected_cluster_index]
                        <= max(2.0 * self.cluster_eps_m, 1e-6)
                    ):
                        static_clusters = raw_static_clusters[
                            selected_cluster_index : selected_cluster_index + 1
                        ]
                        static_points = raw_static_points[
                            raw_static_labels == selected_cluster_index
                        ]
            elif handoff_position is not None:
                static_validation = "handoff_pending"
                self.static_handoff_pending_updates += 1
            elif self.static_scene_map.is_ready:
                static_validation = "background"
            elif self.static_scene_map.is_warming_up:
                static_validation = "warming"
            else:
                static_validation = "calibrating"

            protected_positions = []
            motion_protection_position = (
                self.motion_handoff.protection_position_m
            )
            if motion_protection_position is not None:
                protected_positions.append(motion_protection_position)
            if static_track is not None and static_track.confirmed:
                protected_positions.append(
                    np.asarray(static_track.position_m, dtype=np.float64)
                )
            protected_cells = None
            reference_shape = self.static_scene_map.reference_shape
            if reference_shape is not None and protected_positions:
                protected_cells = static_target_protection_mask(
                    (
                        int(reference_shape[0]),
                        int(reference_shape[1]),
                        int(reference_shape[2]),
                    ),
                    np.stack(protected_positions),
                    range_axis_m,
                    neighborhood_cells=STATIC_PROTECTION_CELLS,
                )
            self.static_scene_map.adapt(protected_cells)
        self.timings.add("clustering", clustering_elapsed)

        target_track = None
        target_source = None
        if (
            dynamic_track is not None
            and dynamic_track.confirmed
            and not dynamic_track.is_predicted
        ):
            target_track = dynamic_track
            target_source = "dynamic"
        elif static_track is not None and static_track.confirmed:
            target_track = static_track
            target_source = "static"
        elif dynamic_track is not None and dynamic_track.confirmed:
            target_track = dynamic_track
            target_source = "dynamic"

        static_reference = self._static_reference_status()
        return (
            PointCloudDisplayPayload(
                points=points,
                clusters=clusters,
                static_points=static_points,
                static_clusters=static_clusters,
                target_track=target_track,
                target_source=target_source,
                static_reference=static_reference,
                static_candidate_count=static_candidate_count,
                static_validation=static_validation,
            ),
            doppler_cube,
        )

    def _static_reference_status(self) -> StaticReferenceStatus:
        if self.static_scene_map is None:
            return StaticReferenceStatus(
                enabled=False,
                ready=False,
                frames_seen=0,
                required_frames=0,
            )
        return StaticReferenceStatus(
            enabled=True,
            ready=self.static_scene_map.is_ready,
            frames_seen=self.static_scene_map.frames_seen,
            required_frames=self.static_scene_map.reference_frames,
            warmup_frames_seen=self.static_scene_map.warmup_frames_seen,
            warmup_frames_required=self.static_scene_map.warmup_frames,
            adaptive=self.static_scene_map.background_update_rate > 0.0,
        )

    def format_static_summary(self) -> str:
        return (
            "Static detection summary: "
            f"candidate_points={self.static_candidate_total}, "
            f"handoff_pending_updates={self.static_handoff_pending_updates}, "
            f"validated_updates={self.static_validated_updates}"
        )


def _put_latest_queue_payload(payload_queue: mp.Queue, payload: Any) -> int:
    skipped_updates = 0
    try:
        payload_queue.put_nowait(payload)
        return skipped_updates
    except queue.Full:
        pass

    try:
        payload_queue.get_nowait()
        skipped_updates += 1
    except queue.Empty:
        pass

    try:
        payload_queue.put_nowait(payload)
    except queue.Full:
        skipped_updates += 1
    return skipped_updates


def _increment_shared_counter(counter: Optional[Any], amount: int = 1) -> None:
    if counter is None or amount <= 0:
        return
    with counter.get_lock():
        counter.value += amount


def _shared_counter_value(counter: Optional[Any]) -> int:
    if counter is None:
        return 0
    with counter.get_lock():
        return int(counter.value)


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
            "slow_time_processing": {
                "start_frequency_ghz": self.config.start_frequency_ghz,
                "idle_time_us": self.config.idle_time_us,
                "ramp_end_time_us": self.config.ramp_end_time_us,
                "chirp_period_s": self.config.chirp_period_s,
                "frame_periodicity_ms": self.config.frame_periodicity_ms,
                "slow_time_rate_hz": self.config.slow_time_rate_hz,
                "frame_duty_cycle": self.config.frame_duty_cycle,
            },
        }
        self.metadata_path.write_text(
            json.dumps(metadata, indent=2) + "\n",
            encoding="utf-8",
        )
        self.emit(f"Raw capture metadata saved to {self.metadata_path}")


def _display_dependency_error(mode: str) -> Optional[str]:
    if mode == "none":
        return None
    if importlib.util.find_spec("pyqtgraph") is None:
        return (
            "Live display requires PyQtGraph. Install "
            "'pyqtgraph>=0.13.7,<0.15' and 'PySide6>=6.7,<7'."
        )
    qt_bindings = ("PySide6", "PyQt6", "PySide2", "PyQt5")
    if not any(importlib.util.find_spec(name) is not None for name in qt_bindings):
        return (
            "Live display requires a Qt binding. Install "
            "'PySide6>=6.7,<7'."
        )
    if mode in {"point-cloud", COMBINED_DISPLAY_MODE}:
        try:
            pyqtgraph_opengl = importlib.util.find_spec("pyqtgraph.opengl")
        except (ImportError, ModuleNotFoundError):
            pyqtgraph_opengl = None
        if pyqtgraph_opengl is None or importlib.util.find_spec("OpenGL") is None:
            return (
                "3D live display requires PyQtGraph OpenGL support. Install "
                "'PyOpenGL>=3.1.7'."
            )
    if (
        sys.platform.startswith("linux")
        and os.environ.get("XDG_SESSION_TYPE", "").lower() == "x11"
    ):
        from ctypes.util import find_library

        if (
            find_library("xcb-cursor") is None
            and _bundled_xcb_cursor_library() is None
        ):
            return (
                "PySide6 cannot open an X11 window because libxcb-cursor0 is "
                "missing. Install it with 'sudo apt-get install "
                "libxcb-cursor0'."
            )
    return None


def _bundled_xcb_cursor_library() -> Optional[Path]:
    if not sys.platform.startswith("linux"):
        return None
    candidate = Path(sys.prefix) / "lib" / "libxcb-cursor.so.0"
    return candidate if candidate.is_file() else None


def _preload_bundled_xcb_cursor_library() -> Optional[str]:
    library_path = _bundled_xcb_cursor_library()
    if library_path is None:
        return None
    try:
        import ctypes

        ctypes.CDLL(str(library_path), mode=ctypes.RTLD_GLOBAL)
    except OSError as exc:
        return f"Unable to preload {library_path}: {exc}"
    return None


def _report_display_startup(
    startup_status_queue: mp.Queue,
    state: str,
    message: str,
) -> None:
    try:
        startup_status_queue.put_nowait(
            {"state": state, "message": message}
        )
    except queue.Full:
        pass


def _qt_application_arguments() -> list[str]:
    """Keep radar's --display option away from Qt's X11 argument parser."""
    return ["radar-live-display"]


def _run_rotor_pyqtgraph_display(
    pause_seconds: float,
    payload_queue: mp.Queue,
    stop_event: mp.Event,
    rendered_updates: Any,
    skipped_updates: Any,
    startup_status_queue: mp.Queue,
) -> None:
    """Run the dedicated high-rate rotor renderer in its display process."""
    preload_error = _preload_bundled_xcb_cursor_library()
    if preload_error is not None:
        print(f"Rotor display runtime warning: {preload_error}")
    try:
        import pyqtgraph as pg
        from pyqtgraph.Qt import QtCore, QtWidgets
    except (ImportError, RuntimeError) as exc:
        message = f"Rotor display could not load PyQtGraph/Qt: {exc}"
        print(message)
        _report_display_startup(startup_status_queue, "error", message)
        return

    pg.setConfigOption("imageAxisOrder", "row-major")
    application = QtWidgets.QApplication.instance()
    if application is None:
        application = QtWidgets.QApplication(
            _qt_application_arguments()
        )
    application.setApplicationName("Radar Rotor Micro-Doppler")
    qt_platform = application.platformName().lower()
    if qt_platform in {"offscreen", "minimal", "minimalegl"}:
        message = (
            "Rotor display cannot open a visible window: Qt selected the "
            f"'{qt_platform}' platform. On Ubuntu/X11 install "
            "'libxcb-cursor0'."
        )
        print(message)
        _report_display_startup(startup_status_queue, "error", message)
        application.quit()
        return
    window = pg.GraphicsLayoutWidget(
        title="Live Rotor Micro-Doppler",
        show=False,
    )
    window.resize(1200, 700)
    status_label = window.addLabel(
        "RPM: collecting history...",
        row=0,
        col=0,
        colspan=2,
        justify="left",
        color="#f0f0f0",
    )

    spectrogram_plot = window.addPlot(row=1, col=0)
    spectrogram_plot.setLabel("left", "Radial velocity", units="m/s")
    spectrogram_plot.setTitle("Enhanced rotor micro-Doppler")
    spectrogram_plot.setXRange(
        -ROTOR_DISPLAY_HISTORY_SECONDS,
        0.0,
        padding=0.0,
    )
    spectrogram_plot.hideAxis("bottom")
    spectrogram_plot.showGrid(x=False, y=True, alpha=0.18)
    spectrogram_plot.setMouseEnabled(x=False, y=False)
    spectrogram_plot.hideButtons()

    initial_image = np.zeros(
        (ROTOR_MICRO_DOPPLER_FFT_SIZE, ROTOR_DISPLAY_TIME_BINS, 4),
        dtype=np.uint8,
    )
    initial_image[:, :, 3] = 255
    image_item = pg.ImageItem(
        initial_image,
        axisOrder="row-major",
        levels=None,
        autoDownsample=False,
    )
    spectrogram_plot.addItem(image_item)
    image_item.setRect(
        QtCore.QRectF(
            -ROTOR_DISPLAY_HISTORY_SECONDS,
            -1.0,
            ROTOR_DISPLAY_HISTORY_SECONDS,
            2.0,
        )
    )

    rotor_lut = _turbo_lookup_table()
    rotor_colormap = pg.ColorMap(
        np.linspace(0.0, 1.0, rotor_lut.shape[0]),
        rotor_lut,
    )
    colorbar_source = pg.ImageItem(
        np.asarray(((0.0, ROTOR_MICRO_DOPPLER_RELATIVE_DB_MAX),)),
        levels=(0.0, ROTOR_MICRO_DOPPLER_RELATIVE_DB_MAX),
    )
    spectrogram_plot.addColorBar(
        colorbar_source,
        colorMap=rotor_colormap,
        values=(0.0, ROTOR_MICRO_DOPPLER_RELATIVE_DB_MAX),
        interactive=False,
        colorMapMenu=False,
        label="Relative power above robust floor (dB)",
    )

    dc_notch_region = pg.LinearRegionItem(
        values=(-0.05, 0.05),
        orientation="horizontal",
        movable=False,
        brush=pg.mkBrush(90, 90, 90, 230),
        pen=pg.mkPen(None),
    )
    dc_notch_region.setZValue(10.0)
    spectrogram_plot.addItem(dc_notch_region)

    flash_plot = window.addPlot(row=2, col=0)
    flash_plot.setLabel(
        "bottom",
        "Active acquisition time relative to newest window",
        units="s",
    )
    flash_plot.setLabel("left", "Flash score", units="dB")
    flash_plot.setXRange(
        -ROTOR_DISPLAY_HISTORY_SECONDS,
        0.0,
        padding=0.0,
    )
    flash_plot.setYRange(
        0.0,
        ROTOR_MICRO_DOPPLER_RELATIVE_DB_MAX,
        padding=0.0,
    )
    flash_plot.setXLink(spectrogram_plot)
    flash_plot.showGrid(x=True, y=True, alpha=0.25)
    flash_plot.setMouseEnabled(x=False, y=False)
    flash_plot.hideButtons()
    flash_curve = flash_plot.plot(
        [],
        [],
        pen=pg.mkPen("#ff9f1c", width=2),
    )
    window.ci.layout.setRowStretchFactor(1, 3)
    window.ci.layout.setRowStretchFactor(2, 1)

    timer = QtCore.QTimer()
    last_extent: Optional[tuple[float, float, float, float]] = None
    last_status_text: Optional[str] = None

    def poll_latest_payload() -> None:
        nonlocal last_extent, last_status_text
        if stop_event.is_set():
            timer.stop()
            window.close()
            application.quit()
            return

        try:
            payload = payload_queue.get_nowait()
        except queue.Empty:
            return

        stale_payloads = 0
        while True:
            try:
                payload = payload_queue.get_nowait()
                stale_payloads += 1
            except queue.Empty:
                break
        _increment_shared_counter(skipped_updates, stale_payloads)

        try:
            display_frame = _prepare_rotor_display_frame(payload.result)
            if display_frame is None:
                return
            colored_image = _colorize_rotor_spectrogram(
                display_frame.spectrogram_db,
                rotor_lut,
            )
            image_item.setImage(
                colored_image,
                autoLevels=False,
                levels=None,
            )
            if display_frame.extent != last_extent:
                x_min, x_max, velocity_min, velocity_max = display_frame.extent
                image_item.setRect(
                    QtCore.QRectF(
                        x_min,
                        velocity_min,
                        x_max - x_min,
                        velocity_max - velocity_min,
                    )
                )
                spectrogram_plot.setYRange(
                    velocity_min,
                    velocity_max,
                    padding=0.0,
                )
                dc_notch_region.setRegion(
                    display_frame.dc_notch_velocity_bounds
                )
                last_extent = display_frame.extent
            flash_curve.setData(
                display_frame.flash_times_s,
                display_frame.flash_scores_db,
                connect="finite",
            )
            if display_frame.status_text != last_status_text:
                status_label.setText(
                    display_frame.status_text.replace("\n", " · ")
                )
                last_status_text = display_frame.status_text
            _increment_shared_counter(rendered_updates)
        except Exception as exc:
            print(f"Rotor PyQtGraph update failed: {exc}")
            timer.stop()
            window.close()
            application.quit()

    timer.timeout.connect(poll_latest_payload)
    polling_interval_ms = max(
        1,
        min(
            int(round(max(pause_seconds, 0.001) * 1000.0)),
            int(round(1000.0 / ROTOR_DISPLAY_MAX_RATE_HZ)),
        ),
    )
    timer.start(polling_interval_ms)
    primary_screen = application.primaryScreen()
    if primary_screen is not None:
        available_geometry = primary_screen.availableGeometry()
        window.move(
            available_geometry.center() - window.rect().center()
        )
    window.showNormal()
    window.raise_()
    window.activateWindow()
    application.alert(window, 5000)
    application.processEvents()
    if not window.isVisible():
        message = "Rotor display window was created but is not visible."
        _report_display_startup(startup_status_queue, "error", message)
        timer.stop()
        window.close()
        application.quit()
        return
    screen_name = (
        window.screen().name()
        if window.screen() is not None
        else "unknown"
    )
    geometry = window.geometry()
    _report_display_startup(
        startup_status_queue,
        "ready",
        (
            "Live rotor display ready: "
            f"backend=PyQtGraph/{qt_platform}, "
            f"DISPLAY={os.environ.get('DISPLAY', '<unset>')}, "
            f"screen={screen_name}, "
            f"geometry={geometry.width()}x{geometry.height()}"
            f"+{geometry.x()}+{geometry.y()}."
        ),
    )
    try:
        application.exec()
    finally:
        timer.stop()
        window.close()


def _turbo_lookup_table(size: int = 256) -> np.ndarray:
    """Return Google's polynomial approximation of the Turbo color map."""
    sample_count = max(int(size), 2)
    positions = np.linspace(0.0, 1.0, sample_count, dtype=np.float64)
    powers = np.stack(
        tuple(positions**power for power in range(6)),
        axis=1,
    )
    coefficients = np.asarray(
        (
            (
                0.13572138,
                4.61539260,
                -42.66032258,
                132.13108234,
                -152.94239396,
                59.28637943,
            ),
            (
                0.09140261,
                2.19418839,
                4.84296658,
                -14.18503333,
                4.27729857,
                2.82956604,
            ),
            (
                0.10667330,
                12.64194608,
                -60.58204836,
                110.36276771,
                -89.90310912,
                27.34824973,
            ),
        ),
        dtype=np.float64,
    )
    colors = np.clip(powers @ coefficients.T, 0.0, 1.0)
    return np.rint(colors * 255.0).astype(np.uint8)


def _colorize_rotor_spectrogram(
    spectrogram_db: np.ndarray,
    lookup_table: np.ndarray,
) -> np.ndarray:
    """Map the relative-dB raster to direct-rendering uint8 RGBA pixels."""
    relative_db = np.nan_to_num(
        np.asarray(spectrogram_db, dtype=np.float32),
        nan=0.0,
        posinf=ROTOR_MICRO_DOPPLER_RELATIVE_DB_MAX,
        neginf=0.0,
    )
    color_indices = np.rint(
        np.clip(
            relative_db / ROTOR_MICRO_DOPPLER_RELATIVE_DB_MAX,
            0.0,
            1.0,
        )
        * (lookup_table.shape[0] - 1)
    ).astype(np.intp)
    rgb = lookup_table[color_indices]
    rgba = np.empty((*rgb.shape[:2], 4), dtype=np.uint8)
    rgba[:, :, :3] = rgb
    rgba[:, :, 3] = 255
    return np.ascontiguousarray(rgba)


def _run_display_process(
    mode: str,
    pause_seconds: float,
    max_range_m: float,
    point_cloud_range_m: float,
    point_cloud_fov_deg: float,
    payload_queue: mp.Queue,
    stop_event: mp.Event,
    rendered_updates: Any,
    skipped_updates: Any,
    startup_status_queue: mp.Queue,
) -> None:
    try:
        import signal

        signal.signal(signal.SIGINT, signal.SIG_IGN)
    except Exception:
        pass

    if mode == ROTOR_DISPLAY_MODE:
        _run_rotor_pyqtgraph_display(
            pause_seconds,
            payload_queue,
            stop_event,
            rendered_updates,
            skipped_updates,
            startup_status_queue,
        )
        return

    _run_pyqtgraph_display(
        mode,
        pause_seconds,
        max_range_m,
        point_cloud_range_m,
        point_cloud_fov_deg,
        payload_queue,
        stop_event,
        rendered_updates,
        skipped_updates,
        startup_status_queue,
    )


def _run_pyqtgraph_display(
    mode: str,
    pause_seconds: float,
    max_range_m: float,
    point_cloud_range_m: float,
    point_cloud_fov_deg: float,
    payload_queue: mp.Queue,
    stop_event: mp.Event,
    rendered_updates: Any,
    skipped_updates: Any,
    startup_status_queue: mp.Queue,
) -> None:
    """Run range, heatmap, point-cloud, and combined PyQtGraph displays."""
    preload_error = _preload_bundled_xcb_cursor_library()
    if preload_error is not None:
        print(f"Live display runtime warning: {preload_error}")
    try:
        import pyqtgraph as pg
        from pyqtgraph.Qt import QtCore, QtGui, QtWidgets
        if mode in {"point-cloud", COMBINED_DISPLAY_MODE}:
            import pyqtgraph.opengl as gl
        else:
            gl = None
    except (ImportError, RuntimeError) as exc:
        message = f"Live display could not load PyQtGraph/Qt: {exc}"
        print(message)
        _report_display_startup(startup_status_queue, "error", message)
        return

    pg.setConfigOption("imageAxisOrder", "row-major")
    application = QtWidgets.QApplication.instance()
    if application is None:
        application = QtWidgets.QApplication(_qt_application_arguments())
    application.setApplicationName("Radar Live Display")
    qt_platform = application.platformName().lower()
    if qt_platform in {"offscreen", "minimal", "minimalegl"}:
        message = (
            "Live display cannot open a visible window: Qt selected the "
            f"'{qt_platform}' platform. On Ubuntu/X11 install "
            "'libxcb-cursor0'."
        )
        _report_display_startup(startup_status_queue, "error", message)
        application.quit()
        return

    turbo_lut = _turbo_lookup_table()
    turbo_colormap = pg.ColorMap(
        np.linspace(0.0, 1.0, turbo_lut.shape[0]),
        turbo_lut,
    )
    window: Any
    plot = None
    curve = None
    image_item = None
    point_cloud_view = None
    point_cloud_status = None
    scatter_items: dict[str, Any] = {}
    micro_doppler_plot = None
    micro_doppler_image = None
    micro_doppler_status = None

    if mode in {"range", "range-doppler"}:
        window = pg.GraphicsLayoutWidget(
            title="Live Radar Display",
            show=False,
        )
        plot = window.addPlot(row=0, col=0)
        plot.showGrid(x=True, y=True, alpha=0.25)
        plot.setMouseEnabled(x=False, y=False)
        plot.hideButtons()
        if mode == "range":
            plot.setTitle("Live Range Profile")
            plot.setLabel("bottom", "Range", units="m")
            plot.setLabel("left", "Magnitude")
            curve = plot.plot(pen=pg.mkPen("#4da3ff", width=2))
        else:
            plot.setTitle("Live Range-Doppler Heatmap")
            plot.setLabel("bottom", "Range", units="m")
            plot.setLabel("left", "Doppler bin")
            image_item = pg.ImageItem(
                np.zeros((1, 1), dtype=np.float32),
                axisOrder="row-major",
                autoDownsample=True,
            )
            plot.addItem(image_item)
            plot.addColorBar(
                image_item,
                colorMap=turbo_colormap,
                interactive=False,
                colorMapMenu=False,
                label="Magnitude (dB)",
            )
    else:
        assert gl is not None
        window = QtWidgets.QWidget()
        window.setWindowTitle("Live Radar Display")
        outer_layout = QtWidgets.QHBoxLayout(window)
        point_cloud_panel = QtWidgets.QWidget()
        point_cloud_layout = QtWidgets.QVBoxLayout(point_cloud_panel)
        point_cloud_status = QtWidgets.QLabel()
        point_cloud_status.setWordWrap(True)
        point_cloud_layout.addWidget(point_cloud_status)
        point_cloud_view = gl.GLViewWidget()
        point_cloud_layout.addWidget(point_cloud_view, 1)
        outer_layout.addWidget(point_cloud_panel, 1)

        cross_range_m = _point_cloud_cross_range_limit(
            point_cloud_range_m,
            point_cloud_fov_deg,
        )
        ground_grid = gl.GLGridItem()
        ground_grid.setSize(
            x=2.0 * cross_range_m,
            y=point_cloud_range_m,
            z=1.0,
        )
        ground_grid.setSpacing(x=1.0, y=1.0, z=1.0)
        ground_grid.translate(0.0, point_cloud_range_m / 2.0, 0.0)
        point_cloud_view.addItem(ground_grid)
        axis_item = gl.GLAxisItem()
        axis_item.setSize(
            x=cross_range_m,
            y=point_cloud_range_m,
            z=cross_range_m,
        )
        point_cloud_view.addItem(axis_item)
        point_cloud_view.setCameraPosition(
            pos=QtGui.QVector3D(0.0, point_cloud_range_m / 2.0, 0.0),
            distance=max(point_cloud_range_m * 2.2, 5.0),
            elevation=24.0,
            azimuth=-60.0,
        )
        point_cloud_view.setBackgroundColor("#101418")
        for name, color, size in (
            ("points", (1.0, 1.0, 1.0, 1.0), 5.0),
            ("clusters", (1.0, 0.15, 0.15, 1.0), 10.0),
            ("static_points", (1.0, 0.55, 0.05, 0.9), 7.0),
            ("static_clusters", (0.0, 0.9, 0.9, 1.0), 10.0),
            ("target", (0.3, 1.0, 0.2, 1.0), 16.0),
        ):
            item = gl.GLScatterPlotItem(
                pos=np.empty((0, 3), dtype=np.float32),
                color=color,
                size=size,
                pxMode=True,
            )
            point_cloud_view.addItem(item)
            scatter_items[name] = item

        if mode == COMBINED_DISPLAY_MODE:
            graph_panel = pg.GraphicsLayoutWidget(show=False)
            micro_doppler_status = graph_panel.addLabel(
                "Refresh rate: measuring...",
                row=0,
                col=0,
                justify="left",
            )
            micro_doppler_plot = graph_panel.addPlot(row=1, col=0)
            micro_doppler_plot.setTitle("Live Micro-Doppler Spectrogram")
            micro_doppler_plot.setLabel(
                "bottom",
                "STFT windows (newest at 0)",
            )
            micro_doppler_plot.setLabel("left", "Centered Doppler bin")
            micro_doppler_plot.setMouseEnabled(x=False, y=False)
            micro_doppler_plot.hideButtons()
            micro_doppler_image = pg.ImageItem(
                np.zeros(
                    (MICRO_DOPPLER_FFT_SIZE, MICRO_DOPPLER_HISTORY_UPDATES),
                    dtype=np.float32,
                ),
                axisOrder="row-major",
                levels=(
                    POINT_CLOUD_MAGNITUDE_DB_MIN,
                    POINT_CLOUD_MAGNITUDE_DB_MAX,
                ),
                autoDownsample=True,
            )
            micro_doppler_plot.addItem(micro_doppler_image)
            micro_doppler_plot.addColorBar(
                micro_doppler_image,
                colorMap=turbo_colormap,
                values=(
                    POINT_CLOUD_MAGNITUDE_DB_MIN,
                    POINT_CLOUD_MAGNITUDE_DB_MAX,
                ),
                interactive=False,
                colorMapMenu=False,
                label="Magnitude (dB)",
            )
            outer_layout.addWidget(graph_panel, 1)

    window.resize(1200 if mode == COMBINED_DISPLAY_MODE else 900, 700)
    display_rate_events: deque[tuple[float, float]] = deque(maxlen=120)
    display_refresh_rate_hz: Optional[float] = None
    point_cloud_rate_events: deque[tuple[float, float]] = deque(maxlen=120)
    point_cloud_refresh_rate_hz: Optional[float] = None
    combined_update_count = 0
    timer = QtCore.QTimer()

    def poll_latest_payload() -> None:
        nonlocal display_refresh_rate_hz
        nonlocal point_cloud_refresh_rate_hz
        nonlocal combined_update_count
        if stop_event.is_set():
            timer.stop()
            window.close()
            application.quit()
            return
        try:
            payload = payload_queue.get_nowait()
        except queue.Empty:
            return
        stale_payloads = 0
        while True:
            try:
                payload = payload_queue.get_nowait()
                stale_payloads += 1
            except queue.Empty:
                break
        _increment_shared_counter(skipped_updates, stale_payloads)

        point_cloud_refreshed = False
        try:
            if mode == "range" and plot is not None and curve is not None:
                range_values, range_profile = payload
                _draw_range_profile(
                    plot,
                    curve,
                    range_values,
                    range_profile,
                    max_range_m,
                    display_refresh_rate_hz,
                )
            elif (
                mode == "range-doppler"
                and plot is not None
                and image_item is not None
            ):
                range_values, heatmap = payload
                _draw_range_doppler(
                    plot,
                    image_item,
                    range_values,
                    heatmap,
                    max_range_m,
                    display_refresh_rate_hz,
                )
            elif mode == "point-cloud":
                _draw_point_cloud_pyqtgraph(
                    scatter_items,
                    point_cloud_status,
                    payload,
                    point_cloud_range_m,
                    point_cloud_fov_deg,
                    display_refresh_rate_hz,
                    turbo_lut,
                )
                point_cloud_refreshed = True
            elif (
                mode == COMBINED_DISPLAY_MODE
                and micro_doppler_plot is not None
                and micro_doppler_image is not None
            ):
                combined_update_count += 1
                point_cloud_refreshed = _combined_point_cloud_update_due(
                    combined_update_count
                )
                if point_cloud_refreshed:
                    _draw_point_cloud_pyqtgraph(
                        scatter_items,
                        point_cloud_status,
                        payload.point_cloud,
                        point_cloud_range_m,
                        point_cloud_fov_deg,
                        point_cloud_refresh_rate_hz,
                        turbo_lut,
                    )
                _set_rate_indicator(
                    micro_doppler_status,
                    display_refresh_rate_hz,
                    range_gate_m=payload.selected_range_m,
                )
                _draw_micro_doppler(
                    micro_doppler_plot,
                    micro_doppler_image,
                    payload.spectrogram_db,
                    payload.selected_range_m,
                    update_axes=True,
                    update_title=False,
                )
            refreshed_at_s = time.perf_counter()
            display_refresh_rate_hz = _record_event_rate(
                display_rate_events,
                refreshed_at_s,
                1.0,
            )
            if point_cloud_refreshed:
                point_cloud_refresh_rate_hz = _record_event_rate(
                    point_cloud_rate_events,
                    refreshed_at_s,
                    1.0,
                )
            _increment_shared_counter(rendered_updates)
        except Exception as exc:
            print(f"PyQtGraph display update failed: {exc}")
            timer.stop()
            window.close()
            application.quit()

    timer.timeout.connect(poll_latest_payload)
    timer.start(max(1, int(round(max(pause_seconds, 0.001) * 1000.0))))
    primary_screen = application.primaryScreen()
    if primary_screen is not None:
        window.move(
            primary_screen.availableGeometry().center()
            - window.rect().center()
        )
    window.showNormal()
    window.raise_()
    window.activateWindow()
    application.processEvents()
    if not window.isVisible():
        message = "Live display window was created but is not visible."
        _report_display_startup(startup_status_queue, "error", message)
        timer.stop()
        window.close()
        application.quit()
        return
    screen_name = (
        window.screen().name()
        if window.screen() is not None
        else "unknown"
    )
    geometry = window.geometry()
    _report_display_startup(
        startup_status_queue,
        "ready",
        (
            "Live display ready: "
            f"backend=PyQtGraph/{qt_platform}, "
            f"DISPLAY={os.environ.get('DISPLAY', '<unset>')}, "
            f"screen={screen_name}, "
            f"geometry={geometry.width()}x{geometry.height()}"
            f"+{geometry.x()}+{geometry.y()}."
        ),
    )
    try:
        application.exec()
    finally:
        timer.stop()
        window.close()


def _record_event_rate(
    events: deque[tuple[float, float]],
    occurred_at_s: float,
    event_units: float,
) -> Optional[float]:
    """Record an event and return its rolling unit rate over two seconds."""
    events.append((float(occurred_at_s), max(float(event_units), 0.0)))
    cutoff_s = occurred_at_s - 2.0
    while len(events) > 2 and events[0][0] < cutoff_s:
        events.popleft()
    if len(events) < 2:
        return None
    elapsed_s = events[-1][0] - events[0][0]
    if elapsed_s <= 0.0:
        return None
    return sum(units for _timestamp_s, units in tuple(events)[1:]) / elapsed_s


def _combined_point_cloud_update_due(update_count: int) -> bool:
    """Return whether the combined display should redraw its 3D panel."""
    return (
        update_count > 0
        and (update_count - 1) % COMBINED_POINT_CLOUD_UPDATE_EVERY == 0
    )


def _set_rate_indicator(
    artist: Any,
    display_refresh_rate_hz: Optional[float],
    *,
    range_gate_m: Optional[float] = None,
) -> None:
    display_rate_text = (
        f"{display_refresh_rate_hz:.1f} Hz"
        if display_refresh_rate_hz is not None
        else "measuring..."
    )
    text = (
        f"Gate: {range_gate_m:.2f} m\n" if range_gate_m is not None else ""
    )
    text += f"Refresh rate: {display_rate_text}"
    if hasattr(artist, "setText"):
        artist.setText(text)
    else:
        artist.set_text(text)


def _draw_range_profile(
    plot: Any,
    curve: Any,
    range_axis_m: Optional[np.ndarray],
    range_profile: np.ndarray,
    max_range_m: float,
    update_rate_hz: Optional[float] = None,
) -> None:
    x_axis = _range_plot_axis(range_axis_m, range_profile.size)
    curve.setData(x_axis, range_profile)
    plot.setXRange(
        float(x_axis[0]),
        _range_plot_xmax(x_axis, max_range_m),
        padding=0.0,
    )
    profile_max = float(np.max(range_profile)) if range_profile.size else 1.0
    plot.setYRange(0.0, max(profile_max * 1.1, 1.0), padding=0.0)
    rate_text = f" — {update_rate_hz:.1f} Hz" if update_rate_hz is not None else ""
    plot.setTitle(f"Live Range Profile{rate_text}")


def _draw_range_doppler(
    plot: Any,
    image: Any,
    range_axis_m: Optional[np.ndarray],
    heatmap: np.ndarray,
    max_range_m: float,
    update_rate_hz: Optional[float] = None,
) -> None:
    x_axis = _range_plot_axis(range_axis_m, heatmap.shape[1])
    minimum = float(np.min(heatmap))
    maximum = float(np.max(heatmap))
    if maximum <= minimum:
        maximum = minimum + 1.0
    image.setImage(
        np.ascontiguousarray(heatmap),
        autoLevels=False,
        levels=(minimum, maximum),
    )
    image.setRect(
        float(x_axis[0]),
        0.0,
        max(float(x_axis[-1] - x_axis[0]), 1e-6),
        float(max(heatmap.shape[0] - 1, 1)),
    )
    plot.setXRange(
        float(x_axis[0]),
        _range_plot_xmax(x_axis, max_range_m),
        padding=0.0,
    )
    plot.setYRange(0.0, float(max(heatmap.shape[0] - 1, 1)), padding=0.0)
    rate_text = f" — {update_rate_hz:.1f} Hz" if update_rate_hz is not None else ""
    plot.setTitle(f"Live Range-Doppler Heatmap{rate_text}")


def _draw_micro_doppler(
    plot: Any,
    image: Any,
    spectrogram_db: np.ndarray,
    selected_range_m: Optional[float],
    display_update_rate_hz: Optional[float] = None,
    *,
    update_axes: bool = True,
    update_title: bool = True,
) -> None:
    if spectrogram_db.ndim != 2 or spectrogram_db.size == 0:
        return

    extent = _micro_doppler_extent(*spectrogram_db.shape)
    display_data = spectrogram_db
    if not update_axes and spectrogram_db.shape[1] < MICRO_DOPPLER_HISTORY_UPDATES:
        display_data = np.full(
            (spectrogram_db.shape[0], MICRO_DOPPLER_HISTORY_UPDATES),
            POINT_CLOUD_MAGNITUDE_DB_MIN,
            dtype=np.float32,
        )
        display_data[:, -spectrogram_db.shape[1] :] = spectrogram_db
    image.setImage(
        np.ascontiguousarray(display_data),
        autoLevels=False,
        levels=(
            POINT_CLOUD_MAGNITUDE_DB_MIN,
            POINT_CLOUD_MAGNITUDE_DB_MAX,
        ),
    )
    if update_axes:
        image.setRect(
            float(extent[0]),
            float(extent[2]),
            float(extent[1] - extent[0]),
            float(extent[3] - extent[2]),
        )
        image.setLevels(
            (
                POINT_CLOUD_MAGNITUDE_DB_MIN,
                POINT_CLOUD_MAGNITUDE_DB_MAX,
            )
        )
        plot.setXRange(*extent[:2], padding=0.0)
        plot.setYRange(*extent[2:], padding=0.0)
    if update_title:
        range_text = (
            f" — gate {selected_range_m:.2f} m"
            if selected_range_m is not None
            else ""
        )
        rate_text = (
            f" — {display_update_rate_hz:.1f} Hz"
            if display_update_rate_hz is not None
            else ""
        )
        plot.setTitle(f"Live Micro-Doppler Spectrogram{range_text}{rate_text}")


def _prepare_rotor_display_frame(
    result: MicroDopplerResult,
) -> Optional[RotorDisplayFrame]:
    if (
        result.enhanced_spectrogram_db.ndim != 2
        or result.enhanced_spectrogram_db.size == 0
        or result.window_times_s.size == 0
        or result.velocity_axis_m_s.size == 0
    ):
        return None

    hop_s = max(float(result.nominal_hop_s), 1e-6)
    newest_time_s = float(result.window_times_s[-1])
    display_data, extent = _rasterize_rotor_spectrogram(
        result.enhanced_spectrogram_db,
        result.window_times_s,
        result.velocity_axis_m_s,
    )
    display_data = _fill_rotor_display_time_gaps(display_data)
    fft_center = display_data.shape[0] // 2
    notch_start = max(
        fft_center - ROTOR_MICRO_DOPPLER_DC_NOTCH_BINS,
        0,
    )
    notch_stop = min(
        fft_center + ROTOR_MICRO_DOPPLER_DC_NOTCH_BINS + 1,
        display_data.shape[0],
    )
    display_data[
        notch_start:notch_stop,
        :,
    ] = np.nan

    velocity_min = float(result.velocity_axis_m_s[0])
    velocity_max = float(result.velocity_axis_m_s[-1])
    if result.velocity_axis_m_s.size > 1:
        velocity_bin_width = float(
            np.median(np.abs(np.diff(result.velocity_axis_m_s)))
        )
    else:
        velocity_bin_width = max(velocity_max - velocity_min, 1e-6)
    dc_notch_velocity_bounds = (
        max(
            velocity_min,
            float(result.velocity_axis_m_s[notch_start])
            - 0.5 * velocity_bin_width,
        ),
        min(
            velocity_max,
            float(result.velocity_axis_m_s[notch_stop - 1])
            + 0.5 * velocity_bin_width,
        ),
    )

    flash_times_s, flash_scores_db = _gap_aware_series(
        result.window_times_s - newest_time_s,
        result.flash_scores_db,
        maximum_gap_s=hop_s * 2.5,
    )

    estimate_text = "RPM: collecting history..."
    if result.rotor_estimates:
        estimates = []
        for estimate in result.rotor_estimates:
            harmonic = (
                f", harmonic ×{estimate.harmonic_rank}"
                if estimate.harmonic_rank > 1
                else ""
            )
            estimates.append(
                f"{estimate.rpm:.0f} RPM "
                f"({estimate.confidence:.0%} confidence{harmonic})"
            )
        estimate_text = "RPM: " + "; ".join(estimates)
    alias_text = (
        "\nALIAS RISK: " + (result.alias_warning or "velocity may be aliased")
        if result.velocity_alias_risk
        else ""
    )
    return RotorDisplayFrame(
        spectrogram_db=display_data,
        extent=extent,
        flash_times_s=flash_times_s,
        flash_scores_db=flash_scores_db,
        status_text=(
            f"Gate: {result.selected_range_m:.2f} m\n"
            f"{estimate_text}{alias_text}"
        ),
        dc_notch_velocity_bounds=dc_notch_velocity_bounds,
    )


def _rasterize_rotor_spectrogram(
    enhanced_spectrogram_db: np.ndarray,
    window_times_s: np.ndarray,
    velocity_axis_m_s: np.ndarray,
    *,
    history_seconds: float = ROTOR_DISPLAY_HISTORY_SECONDS,
    time_bins: int = ROTOR_DISPLAY_TIME_BINS,
) -> tuple[np.ndarray, tuple[float, float, float, float]]:
    """Build a bounded, gap-preserving display grid using temporal max pooling."""
    spectra = np.asarray(enhanced_spectrogram_db, dtype=np.float32)
    times = np.asarray(window_times_s, dtype=np.float64)
    velocity = np.asarray(velocity_axis_m_s, dtype=np.float32)
    bin_count = max(int(time_bins), 2)
    span_s = max(float(history_seconds), 1e-6)
    velocity_min = float(velocity[0])
    velocity_max = float(velocity[-1])
    extent = (-span_s, 0.0, velocity_min, velocity_max)
    raster = np.full((spectra.shape[0], bin_count), np.nan, dtype=np.float32)
    if spectra.size == 0 or times.size == 0:
        return raster, extent

    newest_time_s = float(times[-1])
    relative_times_s = times - newest_time_s
    valid = (
        np.isfinite(relative_times_s)
        & (relative_times_s >= -span_s)
        & (relative_times_s <= 0.0)
    )
    if not np.any(valid):
        return raster, extent

    normalized = (relative_times_s[valid] + span_s) / span_s
    column_indices = np.minimum(
        np.floor(normalized * bin_count).astype(np.intp),
        bin_count - 1,
    )
    pooled = np.full(raster.shape, -np.inf, dtype=np.float32)
    frequency_indices = np.arange(spectra.shape[0], dtype=np.intp)[:, None]
    np.maximum.at(
        pooled,
        (frequency_indices, column_indices[None, :]),
        spectra[:, valid],
    )
    occupied = np.isfinite(pooled)
    raster[occupied] = pooled[occupied]
    return raster, extent


def _concatenated_active_window_times(
    window_count: int,
    nominal_hop_s: float,
) -> np.ndarray:
    """Return a uniform active-time axis ending at zero for display only."""
    count = max(int(window_count), 0)
    if count == 0:
        return np.empty((0,), dtype=np.float64)
    hop_s = max(float(nominal_hop_s), 1e-9)
    return (
        np.arange(count, dtype=np.float64) - float(count - 1)
    ) * hop_s


def _fill_rotor_display_time_gaps(raster: np.ndarray) -> np.ndarray:
    """Fill unmeasured display columns from the nearest measured spectrum."""
    filled = np.asarray(raster, dtype=np.float32).copy()
    if filled.ndim != 2 or filled.shape[1] == 0:
        return filled

    measured_columns = np.flatnonzero(np.any(np.isfinite(filled), axis=0))
    if measured_columns.size == 0:
        return filled

    columns = np.arange(filled.shape[1], dtype=np.intp)
    insertion_points = np.searchsorted(measured_columns, columns, side="left")
    right_columns = measured_columns[
        np.minimum(insertion_points, measured_columns.size - 1)
    ]
    left_columns = measured_columns[np.maximum(insertion_points - 1, 0)]
    nearest_columns = np.where(
        np.abs(right_columns - columns) < np.abs(columns - left_columns),
        right_columns,
        left_columns,
    )
    missing_columns = ~np.any(np.isfinite(filled), axis=0)
    filled[:, missing_columns] = filled[:, nearest_columns[missing_columns]]
    return filled


def _gap_aware_series(
    times_s: np.ndarray,
    values: np.ndarray,
    *,
    maximum_gap_s: float,
) -> tuple[np.ndarray, np.ndarray]:
    times = np.asarray(times_s, dtype=np.float64)
    samples = np.asarray(values, dtype=np.float32)
    if times.size <= 1:
        return times, samples
    gap_indices = np.flatnonzero(np.diff(times) > maximum_gap_s)
    if gap_indices.size == 0:
        return times, samples
    expanded_times = []
    expanded_samples = []
    gap_set = set(int(index) for index in gap_indices)
    for index, (sample_time_s, sample) in enumerate(zip(times, samples)):
        expanded_times.append(float(sample_time_s))
        expanded_samples.append(float(sample))
        if index in gap_set:
            expanded_times.append(float(sample_time_s))
            expanded_samples.append(np.nan)
    return (
        np.asarray(expanded_times, dtype=np.float64),
        np.asarray(expanded_samples, dtype=np.float32),
    )


def _micro_doppler_extent(
    doppler_bins: int,
    history_updates: int,
) -> tuple[int, int, int, int]:
    first_update = -max(history_updates - 1, 1)
    first_doppler_bin = -(doppler_bins // 2)
    last_doppler_bin = first_doppler_bin + doppler_bins - 1
    return (
        first_update,
        0,
        first_doppler_bin,
        max(last_doppler_bin, first_doppler_bin + 1),
    )


def _point_cloud_cross_range_limit(
    point_cloud_range_m: float,
    point_cloud_fov_deg: float,
) -> float:
    range_limit_m = max(float(point_cloud_range_m), 1.0)
    fov_deg = min(max(float(point_cloud_fov_deg), 0.0), 90.0)
    return max(
        range_limit_m * np.sin(np.deg2rad(fov_deg)),
        0.5,
    )


def _set_gl_scatter_data(
    item: Any,
    positions: np.ndarray,
    colors: np.ndarray | tuple[float, float, float, float],
    sizes: np.ndarray | float,
) -> None:
    item.setData(
        pos=np.ascontiguousarray(positions, dtype=np.float32).reshape((-1, 3)),
        color=colors,
        size=sizes,
        pxMode=True,
    )


def _magnitude_colors(
    magnitudes_db: np.ndarray,
    lookup_table: np.ndarray,
) -> np.ndarray:
    normalized = np.clip(
        (
            np.asarray(magnitudes_db, dtype=np.float32)
            - POINT_CLOUD_MAGNITUDE_DB_MIN
        )
        / (POINT_CLOUD_MAGNITUDE_DB_MAX - POINT_CLOUD_MAGNITUDE_DB_MIN),
        0.0,
        1.0,
    )
    indices = np.rint(normalized * (lookup_table.shape[0] - 1)).astype(np.intp)
    colors = np.empty((indices.size, 4), dtype=np.float32)
    colors[:, :3] = lookup_table[indices].astype(np.float32) / 255.0
    colors[:, 3] = 1.0
    return colors


def _draw_point_cloud_pyqtgraph(
    scatter_items: dict[str, Any],
    status_label: Any,
    point_cloud: PointCloudDisplayPayload,
    point_cloud_range_m: float,
    point_cloud_fov_deg: float,
    update_rate_hz: Optional[float],
    lookup_table: np.ndarray,
) -> None:
    empty = np.empty((0, 3), dtype=np.float32)
    points = np.asarray(point_cloud.points)
    if points.size:
        _set_gl_scatter_data(
            scatter_items["points"],
            points[:, :3],
            _magnitude_colors(points[:, 3], lookup_table),
            5.0,
        )
    else:
        _set_gl_scatter_data(
            scatter_items["points"],
            empty,
            np.empty((0, 4), dtype=np.float32),
            5.0,
        )

    clusters = np.asarray(point_cloud.clusters)
    _set_gl_scatter_data(
        scatter_items["clusters"],
        clusters[:, :3] if clusters.size else empty,
        (1.0, 0.15, 0.15, 1.0),
        (
            np.clip(7.0 + clusters[:, 3] * 1.5, 8.0, 18.0)
            if clusters.size
            else 10.0
        ),
    )
    static_points = np.asarray(point_cloud.static_points)
    _set_gl_scatter_data(
        scatter_items["static_points"],
        static_points[:, :3] if static_points.size else empty,
        (1.0, 0.55, 0.05, 0.9),
        7.0,
    )
    static_clusters = np.asarray(point_cloud.static_clusters)
    _set_gl_scatter_data(
        scatter_items["static_clusters"],
        static_clusters[:, :3] if static_clusters.size else empty,
        (0.0, 0.9, 0.9, 1.0),
        (
            np.clip(8.0 + static_clusters[:, 3] * 1.5, 9.0, 19.0)
            if static_clusters.size
            else 10.0
        ),
    )
    if point_cloud.target_track is None:
        target_position = empty
        target_color = (0.3, 1.0, 0.2, 1.0)
        target_size = 16.0
    else:
        target_position = np.asarray(
            point_cloud.target_track.position_m,
            dtype=np.float32,
        ).reshape((1, 3))
        alpha = 0.4 if point_cloud.target_track.is_predicted else 1.0
        target_color = (
            (1.0, 0.55, 0.05, alpha)
            if point_cloud.target_source == "static"
            else (0.3, 1.0, 0.2, alpha)
        )
        target_size = 13.0 if point_cloud.target_track.is_predicted else 16.0
    _set_gl_scatter_data(
        scatter_items["target"],
        target_position,
        target_color,
        target_size,
    )

    rate_text = (
        f"{update_rate_hz:.1f} Hz"
        if update_rate_hz is not None
        else "measuring..."
    )
    reference_text = (
        point_cloud.static_reference.label
        if point_cloud.static_reference is not None
        else ""
    )
    status = (
        "Live 3D Point Cloud "
        f"(±{point_cloud_fov_deg:g}° FOV, {point_cloud_range_m:g} m) · "
        f"Refresh rate: {rate_text}\n"
        "Axes: X left/right · Y forward · Z elevation"
    )
    if reference_text:
        status += f"\n{reference_text}"
    status_label.setText(status)


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
    total_started = time.perf_counter()
    if not frame.is_valid:
        raw_writer.write_frame(frame)
        emit_func(
            "Dropped frame: incomplete payload, "
            f"gap_bytes={frame.gap_bytes}, bytes_per_frame={config.bytes_per_frame}"
        )
        return

    raw_writer.write_frame(frame)
    radar_cube = frame_bytes_to_radar_cube(frame.data, config)
    range_fft_started = time.perf_counter()
    range_fft = compute_range_fft(radar_cube)
    display.timings.add(
        "range_fft",
        time.perf_counter() - range_fft_started,
    )
    range_axis_m = config.range_axis_m()
    display.update(
        range_fft,
        range_axis_m,
        captured_at_s=frame.first_byte_at_s,
    )
    display.timings.add("total", time.perf_counter() - total_started)


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


def _request_processor_stop(
    frame_queue: mp.Queue,
    timeout_seconds: float = 5.0,
) -> bool:
    try:
        frame_queue.put(None, timeout=timeout_seconds)
        return True
    except queue.Full:
        return False


def _rotor_processing_metadata(
    *,
    micro_doppler_range_half_width_bins: int,
    micro_doppler_range_m: Optional[float],
    rotor_blades: int,
    rotor_count: int,
    rotor_radius_m: Optional[float],
    rotor_rpm_min: float,
    rotor_rpm_max: float,
) -> dict[str, Any]:
    return {
        "mode": "per-TX clutter-rejected STFT",
        "window": "Hann",
        "window_loops": ROTOR_MICRO_DOPPLER_WINDOW_LOOPS,
        "hop_loops": ROTOR_MICRO_DOPPLER_HOP_LOOPS,
        "fft_size": ROTOR_MICRO_DOPPLER_FFT_SIZE,
        "range_half_width_bins": micro_doppler_range_half_width_bins,
        "dc_notch_bins": ROTOR_MICRO_DOPPLER_DC_NOTCH_BINS,
        "relative_display_db": [
            0.0,
            ROTOR_MICRO_DOPPLER_RELATIVE_DB_MAX,
        ],
        "estimator_history_seconds": ROTOR_MICRO_DOPPLER_HISTORY_SECONDS,
        "display_history_seconds": ROTOR_DISPLAY_HISTORY_SECONDS,
        "display_time_basis": "concatenated active acquisition intervals",
        "estimator_time_basis": "physical capture timestamps",
        "display_time_bins": ROTOR_DISPLAY_TIME_BINS,
        "processed_spectrum_db_decimals": ROTOR_SPECTRUM_OUTPUT_DECIMALS,
        "legacy_windows_policy": (
            "latest window only; full windows are in rotor_micro_doppler"
        ),
        "weighted_complex_mean_cancellation": True,
        "adaptive_noise_filter": {
            "minimum_gate_db": DEFAULT_ROTOR_NOISE_GATE_MIN_DB,
            "maximum_gate_db": DEFAULT_ROTOR_NOISE_GATE_MAX_DB,
            "robust_sigma_multiplier": DEFAULT_ROTOR_NOISE_SIGMA_MULTIPLIER,
            "mad_scale": 1.4826,
            "mad_policy": "lower half below median to avoid positive blade bias",
            "support_shape": list(DEFAULT_ROTOR_NOISE_SUPPORT_SHAPE),
            "minimum_support_cells": DEFAULT_ROTOR_NOISE_MIN_SUPPORT,
            "amplitude_policy": (
                "preserve retained relative dB; blank rejected cells"
            ),
        },
        "target_range_m": micro_doppler_range_m,
        "rotor_blades": rotor_blades,
        "rotor_count": rotor_count,
        "rotor_radius_m": rotor_radius_m,
        "rotor_rpm_min": rotor_rpm_min,
        "rotor_rpm_max": rotor_rpm_max,
    }


def _run_rotor_postprocessor(
    config: RadarCaptureConfig,
    post_queue: mp.Queue,
    log_queue: mp.Queue,
    startup_status_queue: mp.Queue,
    failure_event: Any,
    postprocessed_frames_counter: Any,
    processed_output: Optional[Path],
    display_update_every: int,
    classification_artifact_dir: Path,
    classification_profile_path: Path,
    classification_device: str,
    micro_doppler_range_m: Optional[float],
    micro_doppler_range_half_width_bins: int,
    rotor_blades: int,
    rotor_count: int,
    rotor_radius_m: Optional[float],
    rotor_rpm_min: float,
    rotor_rpm_max: float,
) -> None:
    signal.signal(signal.SIGINT, signal.SIG_IGN)

    def worker_emit(message: str) -> None:
        _queue_emit(log_queue, message)

    ready_reported = False
    inference_engine: Optional[Any] = None
    processed_writer: Optional[ProcessedOutputWriter] = None
    timings = ProcessingTimingStats()
    last_classification_emit_s = 0.0
    last_classification_signature: Optional[tuple[str, Optional[str]]] = None
    try:
        inference_engine = create_inference_engine(
            classification_artifact_dir,
            config,
            classification_profile_path,
            device=classification_device,
        )
        processed_writer = ProcessedOutputWriter(
            processed_output,
            config,
            display_update_every,
            worker_emit,
            static_detection=False,
            rotor_processing=_rotor_processing_metadata(
                micro_doppler_range_half_width_bins=(
                    micro_doppler_range_half_width_bins
                ),
                micro_doppler_range_m=micro_doppler_range_m,
                rotor_blades=rotor_blades,
                rotor_count=rotor_count,
                rotor_radius_m=rotor_radius_m,
                rotor_rpm_min=rotor_rpm_min,
                rotor_rpm_max=rotor_rpm_max,
            ),
            classification_metadata=inference_engine.metadata,
        )
        metadata = inference_engine.metadata
        backend = metadata.get("backend", "pytorch")
        device = metadata.get("device", "cpu")
        startup_message = (
            "Rotor post-processor ready: "
            f"classification_backend={backend}, device={device}"
        )
        if backend == "tensorrt":
            gpu = metadata.get("gpu", {})
            benchmark = metadata.get("benchmark", {})
            startup_message += (
                f", gpu={gpu.get('name', 'unknown')}, "
                f"precision={metadata.get('precision')}, "
                f"device_buffers={metadata.get('device_allocation_bytes', 0)}B, "
                f"inference_p50={float(benchmark.get('p50_ms', 0.0)):.3f}ms, "
                f"p95={float(benchmark.get('p95_ms', 0.0)):.3f}ms"
            )
        startup_status_queue.put(
            {"state": "ready", "message": startup_message},
            timeout=1.0,
        )
        ready_reported = True

        while True:
            item = post_queue.get()
            if item is None:
                break
            if not isinstance(item, RotorPostprocessItem):
                raise TypeError(
                    f"Unexpected rotor post-processing item: {type(item).__name__}"
                )
            total_started = time.perf_counter()
            inference_started = time.perf_counter()
            classification = inference_engine.update_feature_step(
                item.feature_step
            )
            timings.add(
                "classification",
                time.perf_counter() - inference_started,
            )
            signature = (classification.label, classification.reason)
            now = time.monotonic()
            if (
                signature != last_classification_signature
                or now - last_classification_emit_s >= 1.0
            ):
                worker_emit(
                    CLASSIFICATION_RESULT_PREFIX
                    + json.dumps(
                        classification.to_dict(),
                        separators=(",", ":"),
                    )
                )
                last_classification_signature = signature
                last_classification_emit_s = now

            if item.save_update and processed_writer.enabled:
                result = item.rotor_result
                serialization_started = time.perf_counter()
                processed_writer.write_update(
                    frame_index=item.frame_index,
                    points=np.empty((0, 4), dtype=np.float32),
                    clusters=np.empty((0, 4), dtype=np.float32),
                    target_track=None,
                    micro_doppler_db=(
                        result.raw_spectrogram_db[:, -1]
                        if result.raw_spectrogram_db.size
                        else np.empty((0,), dtype=np.float32)
                    ),
                    micro_doppler_windows_db=(
                        result.raw_spectrogram_db[:, -1:]
                        if result.raw_spectrogram_db.size
                        else np.empty((0, 0), dtype=np.float32)
                    ),
                    selected_range_m=result.selected_range_m,
                    rotor_micro_doppler=result,
                    classification=classification,
                )
                timings.add(
                    "serialization",
                    time.perf_counter() - serialization_started,
                )
            timings.add("total", time.perf_counter() - total_started)
            _increment_shared_counter(postprocessed_frames_counter)
    except Exception as exc:
        failure_event.set()
        message = (
            "Rotor post-processor failed: "
            f"{type(exc).__name__}: {exc}"
        )
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
        if processed_writer is not None:
            processed_writer.close()
        if inference_engine is not None and hasattr(inference_engine, "close"):
            inference_engine.close()
        worker_emit(
            timings.format_summary().replace(
                "Processing timing summary:",
                "Post-processing timing summary:",
                1,
            )
        )


def _run_frame_processor_impl(
    config: RadarCaptureConfig,
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
    max_range_m: float,
    point_cloud_fov_deg: float,
    cluster_eps_m: float,
    cluster_min_samples: int,
    clutter_map_update_rate: float,
    clutter_map_warmup_frames: int,
    clutter_map_min_snr_db: float,
    static_detection: bool,
    static_warmup_frames: int,
    static_reference_frames: int,
    static_min_change_db: float,
    static_background_update_rate: float,
    static_cluster_min_samples: int,
    micro_doppler_range_m: Optional[float],
    micro_doppler_range_half_width_bins: int,
    rotor_blades: int,
    rotor_count: int,
    rotor_radius_m: Optional[float],
    rotor_rpm_min: float,
    rotor_rpm_max: float,
    classification_enabled: bool = False,
    classification_artifact_dir: Optional[Path] = None,
    classification_profile_path: Optional[Path] = None,
    classification_device: str = "auto",
    rotor_post_queue: Optional[mp.Queue] = None,
    rotor_post_failure_event: Optional[Any] = None,
    rotor_post_queue_high_water: Optional[Any] = None,
    startup_status_queue: Optional[mp.Queue] = None,
) -> None:
    signal.signal(signal.SIGINT, signal.SIG_IGN)

    def worker_emit(message: str) -> None:
        _queue_emit(log_queue, message)

    use_rotor_postprocess = bool(
        classification_enabled
        and display_mode == ROTOR_DISPLAY_MODE
        and rotor_post_queue is not None
    )
    inference_engine = None
    if classification_enabled and not use_rotor_postprocess:
        if (
            classification_artifact_dir is None
            or classification_profile_path is None
        ):
            raise ValueError(
                "Classification requires artifact and radar profile paths"
            )
        inference_engine = create_inference_engine(
            classification_artifact_dir,
            config,
            classification_profile_path,
            device=classification_device,
        )
        worker_emit(
            "CNN classification enabled: "
            f"artifacts={classification_artifact_dir}, "
            f"device={resolve_classification_device(classification_device)}, "
            f"threshold={inference_engine.threshold:.6f}, "
            "negative_training_class=bionic_bird"
        )

    raw_writer = RawFrameWriter(raw_output, raw_metadata, config, worker_emit)
    processed_writer = ProcessedOutputWriter(
        None if use_rotor_postprocess else processed_output,
        config,
        display_update_every,
        worker_emit,
        static_detection=static_detection,
        static_warmup_frames=static_warmup_frames,
        static_reference_frames=static_reference_frames,
        static_min_change_db=static_min_change_db,
        static_background_update_rate=static_background_update_rate,
        static_cluster_min_samples=static_cluster_min_samples,
        rotor_processing=(
            _rotor_processing_metadata(
                micro_doppler_range_half_width_bins=(
                    micro_doppler_range_half_width_bins
                ),
                micro_doppler_range_m=micro_doppler_range_m,
                rotor_blades=rotor_blades,
                rotor_count=rotor_count,
                rotor_radius_m=rotor_radius_m,
                rotor_rpm_min=rotor_rpm_min,
                rotor_rpm_max=rotor_rpm_max,
            )
            if display_mode == ROTOR_DISPLAY_MODE
            else None
        ),
        classification_metadata=(
            inference_engine.metadata
            if inference_engine is not None
            else {"enabled": False}
        ),
    )
    display = DisplayPayloadSink(
        display_mode,
        display_update_every,
        display_payload_queue,
        config,
        max_range_m,
        point_cloud_fov_deg,
        cluster_eps_m,
        cluster_min_samples,
        clutter_map_update_rate,
        clutter_map_warmup_frames,
        clutter_map_min_snr_db,
        processed_writer,
        display_skipped_counter,
        static_detection=static_detection,
        static_warmup_frames=static_warmup_frames,
        static_reference_frames=static_reference_frames,
        static_min_change_db=static_min_change_db,
        static_background_update_rate=static_background_update_rate,
        static_cluster_min_samples=static_cluster_min_samples,
        micro_doppler_range_m=micro_doppler_range_m,
        micro_doppler_range_half_width_bins=(
            micro_doppler_range_half_width_bins
        ),
        rotor_blades=rotor_blades,
        rotor_count=rotor_count,
        rotor_radius_m=rotor_radius_m,
        rotor_rpm_min=rotor_rpm_min,
        rotor_rpm_max=rotor_rpm_max,
        inference_engine=inference_engine,
        classification_emit_func=worker_emit,
        rotor_post_queue=(rotor_post_queue if use_rotor_postprocess else None),
        rotor_post_failure_event=(
            rotor_post_failure_event if use_rotor_postprocess else None
        ),
        rotor_post_queue_high_water=(
            rotor_post_queue_high_water if use_rotor_postprocess else None
        ),
    )
    if display.clutter_map is not None and (
        display_mode in {"point-cloud", COMBINED_DISPLAY_MODE}
        or processed_writer.enabled
    ):
        worker_emit(
            "Adaptive clutter map enabled: "
            f"update_rate={clutter_map_update_rate:g}, "
            f"warmup={clutter_map_warmup_frames} updates, "
            f"minimum_snr={clutter_map_min_snr_db:g} dB"
        )
    if display.static_scene_map is not None and (
        display_mode in {"point-cloud", COMBINED_DISPLAY_MODE}
        or processed_writer.enabled
    ):
        worker_emit(
            "Static scene detection enabled: "
            f"warmup={static_warmup_frames} updates, "
            f"reference={static_reference_frames} updates, "
            f"minimum_change={static_min_change_db:g} dB, "
            f"background_update_rate={static_background_update_rate:g}. "
            "Keep the target absent until calibration is complete."
    )
    if display_mode == ROTOR_DISPLAY_MODE:
        chirps_per_loop = config.num_chirps_per_loop or 1
        slow_time_rate_hz = config.slow_time_rate_hz
        slow_time_interval_s = config.slow_time_interval_s
        slow_time_text = (
            f"per-TX slow_time_rate={slow_time_rate_hz:.2f} Hz, "
            if slow_time_rate_hz is not None
            else ""
        )
        resolution_text = (
            "STFT time span/hop="
            f"{ROTOR_MICRO_DOPPLER_WINDOW_LOOPS * slow_time_interval_s * 1e3:.3f}/"
            f"{ROTOR_MICRO_DOPPLER_HOP_LOOPS * slow_time_interval_s * 1e3:.3f} ms, "
            if slow_time_interval_s is not None
            else ""
        )
        worker_emit(
            "Dedicated rotor micro-Doppler enabled: "
            f"gate={micro_doppler_range_m:.2f} m, "
            f"range_bins={2 * micro_doppler_range_half_width_bins + 1}, "
            f"window/hop={ROTOR_MICRO_DOPPLER_WINDOW_LOOPS}/"
            f"{ROTOR_MICRO_DOPPLER_HOP_LOOPS} loops, "
            f"chirps_per_loop={chirps_per_loop}, "
            f"{slow_time_text}"
            f"{resolution_text}"
            f"active_display_span={ROTOR_DISPLAY_HISTORY_SECONDS:g} s, "
            f"blades={rotor_blades}, rotors={rotor_count}, "
            f"RPM band={rotor_rpm_min:g}-{rotor_rpm_max:g}."
        )

    if startup_status_queue is not None:
        startup_status_queue.put(
            {
                "state": "ready",
                "message": "Radar frame processor ready.",
            },
            timeout=1.0,
        )

    try:
        while True:
            frame = frame_queue.get()
            if frame is None:
                break

            process_complete_frame(frame, config, display, raw_writer, worker_emit)
            _increment_shared_counter(processed_frames_counter)
    except KeyboardInterrupt:
        pass
    except Exception as exc:
        if rotor_post_failure_event is not None:
            rotor_post_failure_event.set()
        worker_emit(f"Frame processor stopped after error: {exc!r}")
    finally:
        if use_rotor_postprocess and rotor_post_queue is not None:
            while not (
                rotor_post_failure_event is not None
                and rotor_post_failure_event.is_set()
            ):
                try:
                    rotor_post_queue.put(None, timeout=0.1)
                    break
                except queue.Full:
                    continue
        raw_writer.close()
        processed_writer.close()
        worker_emit(display.format_static_summary())
        worker_emit(display.timings.format_summary())


def _run_frame_processor(*args: Any, **kwargs: Any) -> None:
    """Run the DSP worker and report failures that occur during initialization."""
    log_queue = kwargs.get("log_queue")
    if log_queue is None and len(args) >= 3:
        log_queue = args[2]
    startup_status_queue = kwargs.get("startup_status_queue")
    if startup_status_queue is None and len(args) >= 39:
        startup_status_queue = args[38]
    try:
        _run_frame_processor_impl(*args, **kwargs)
    except KeyboardInterrupt:
        pass
    except Exception as exc:
        message = f"Frame processor failed during startup: {exc!r}"
        if log_queue is None:
            raise
        _queue_emit(log_queue, message)
        if startup_status_queue is not None:
            try:
                startup_status_queue.put(
                    {"state": "failed", "message": message},
                    timeout=1.0,
                )
            except queue.Full:
                pass


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
    display: LiveDisplay,
    pipeline_failure_event: Optional[Any] = None,
    pipeline_process: Optional[Any] = None,
) -> CaptureStats:
    stats = CaptureStats()
    sequence_tracker = SequenceTracker(stats)
    frame_buffer = FrameBuffer(config.bytes_per_frame, stats)
    synthetic_sequence_number = 0
    synthetic_byte_count = 0
    reported_error_counts = _capture_error_counts(stats)
    packet_queue: queue.Queue[tuple[bytes, float]] = queue.Queue(
        maxsize=max(packet_queue_size, 1)
    )

    def emit_new_error_stats() -> None:
        nonlocal reported_error_counts
        reported_error_counts = _report_new_error_stats(
            stats,
            reported_error_counts,
            emit,
        )

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
    if (
        socket_recv_buffer_bytes > 0
        and actual_recv_buffer_bytes < socket_recv_buffer_bytes
    ):
        emit(
            "WARNING: The operating system granted less UDP receive buffering "
            "than requested. On Linux, raise the limit before capture with "
            f"'sudo sysctl -w net.core.rmem_max={socket_recv_buffer_bytes}'."
        )
    emit(f"UDP packet queue capacity: {max(packet_queue_size, 1)} datagrams")
    emit("Trigger frames now. Press Ctrl+C to stop.")

    receiver = UdpPacketReceiver(
        sock,
        packet_queue,
        buffer_size,
        stats,
    )
    receiver.start()
    try:
        while True:
            if pipeline_process is not None and not pipeline_process.is_alive():
                if pipeline_failure_event is not None:
                    pipeline_failure_event.set()
                emit(
                    "Processing pipeline worker exited unexpectedly; "
                    "stopping capture."
                )
                break
            if (
                pipeline_failure_event is not None
                and pipeline_failure_event.is_set()
            ):
                emit("Processing pipeline failed; stopping capture.")
                break
            try:
                packet, packet_received_at_s = packet_queue.get(
                    timeout=socket_timeout_seconds
                )
            except queue.Empty:
                _drain_log_queue(log_queue)
                emit_new_error_stats()
                if receiver.error is not None:
                    emit(f"UDP receiver stopped after error: {receiver.error}")
                    break
                continue

            packet_view = memoryview(packet)
            if setup_config.packet_sequence_enable:
                try:
                    header = DCA1000PacketHeader.parse(packet_view)
                except ValueError:
                    stats.malformed_packets += 1
                    emit_new_error_stats()
                    continue

                payload = packet_view[DCA1000_HEADER_SIZE:]
                sequence_tracker.observe(header.sequence_number)
            else:
                payload = packet_view
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
                    continue

                try:
                    frame_queue.put_nowait(frame)
                except queue.Full:
                    stats.processing_frames_dropped += 1

            emit_new_error_stats()

            if stats.packets_received % 200 == 0:
                _drain_log_queue(log_queue)
    except KeyboardInterrupt:
        _drain_log_queue(log_queue)
        emit_new_error_stats()
        emit("Streaming stopped.")
    finally:
        receiver.stop()
        sock.close()
        receiver.join(timeout=max(socket_timeout_seconds + 0.5, 1.0))
        if receiver.is_alive:
            emit("WARNING: UDP receiver thread did not stop cleanly.")
        _drain_log_queue(log_queue)
        emit_new_error_stats()
        emit(_format_capture_summary(stats))
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
        "--packet-queue-size",
        type=int,
        default=DEFAULT_PACKET_QUEUE_SIZE,
        help=(
            "Maximum UDP datagrams buffered between the receiver thread and "
            f"frame assembly. Defaults to {DEFAULT_PACKET_QUEUE_SIZE}."
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
        "--processed-output",
        type=Path,
        help=(
            "Stream processed 3D point clouds and micro-Doppler spectra to "
            "this JSONL file."
        ),
    )
    parser.add_argument(
        "--classification",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Run the trained drone/not-drone CNN. Enabled by default; use "
            "--no-classification to run without PyTorch."
        ),
    )
    parser.add_argument(
        "--classification-artifacts",
        type=Path,
        default=DEFAULT_CLASSIFICATION_ARTIFACT_DIR,
        help=(
            "Directory containing the trained CNN state, calibration bundle, "
            "and model card."
        ),
    )
    parser.add_argument(
        "--classification-device",
        choices=("auto", "cuda", "cpu"),
        default="auto",
        help=(
            "Inference device. On Jetson, auto requires the TensorRT CUDA "
            "backend and never silently falls back to CPU."
        ),
    )
    parser.add_argument(
        "--display",
        choices=(
            "none",
            "range",
            "range-doppler",
            "point-cloud",
            ROTOR_DISPLAY_MODE,
            COMBINED_DISPLAY_MODE,
        ),
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
        "--micro-doppler-range-m",
        type=float,
        help=(
            "Required fixed range gate in meters for the dedicated "
            "micro-doppler display."
        ),
    )
    parser.add_argument(
        "--micro-doppler-range-half-width-bins",
        type=int,
        default=ROTOR_MICRO_DOPPLER_RANGE_HALF_WIDTH_BINS,
        help="Range bins on each side of the dedicated rotor gate. Defaults to 1.",
    )
    parser.add_argument(
        "--rotor-blades",
        type=int,
        default=DEFAULT_ROTOR_BLADES,
        help=(
            "Blade count per rotor used to convert blade-passage rate to RPM. "
            "Defaults to 2."
        ),
    )
    parser.add_argument(
        "--rotor-count",
        type=int,
        default=1,
        help="Number of separated rotor-rate peaks to report.",
    )
    parser.add_argument(
        "--rotor-radius-m",
        type=float,
        help="Rotor radius in meters for velocity-alias diagnostics.",
    )
    parser.add_argument(
        "--rotor-rpm-min",
        type=float,
        default=500.0,
        help="Minimum RPM searched by the blade-passage estimator.",
    )
    parser.add_argument(
        "--rotor-rpm-max",
        type=float,
        default=DEFAULT_ROTOR_RPM_MAX,
        help=(
            "Maximum RPM searched by the blade-passage estimator. "
            "Defaults to 10700."
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
    parser.add_argument(
        "--cluster-eps-m",
        type=float,
        default=DEFAULT_CLUSTER_EPS_M,
        help=(
            "DBSCAN XYZ neighborhood radius in meters for point-cloud clustering. "
            "Defaults to 0.4 m; use 0 to disable clustering."
        ),
    )
    parser.add_argument(
        "--cluster-min-samples",
        type=int,
        default=DEFAULT_CLUSTER_MIN_SAMPLES,
        help="Minimum points required for a DBSCAN cluster. Defaults to 2.",
    )
    parser.add_argument(
        "--clutter-map-update-rate",
        type=float,
        default=DEFAULT_CLUTTER_MAP_UPDATE_RATE,
        help=(
            "Adaptive clutter-map EMA update rate. Defaults to 0.02; "
            "use 0 to disable clutter suppression."
        ),
    )
    parser.add_argument(
        "--clutter-map-warmup-frames",
        type=int,
        default=DEFAULT_CLUTTER_MAP_WARMUP_FRAMES,
        help=(
            "Frames used to learn the initial clutter map before detections. "
            "Defaults to 30."
        ),
    )
    parser.add_argument(
        "--clutter-map-min-snr-db",
        type=float,
        default=DEFAULT_CLUTTER_MAP_MIN_SNR_DB,
        help=(
            "Minimum target-to-background power ratio in dB after clutter "
            "normalization. Defaults to 3 dB."
        ),
    )
    parser.add_argument(
        "--static-detection",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_STATIC_DETECTION,
        help=(
            "Detect motion-qualified stationary changes against an adaptive "
            "startup reference. "
            "Enabled by default; use --no-static-detection to disable."
        ),
    )
    parser.add_argument(
        "--static-warmup-frames",
        type=int,
        default=DEFAULT_STATIC_WARMUP_FRAMES,
        help=(
            "Processed updates discarded before static calibration. "
            f"Defaults to {DEFAULT_STATIC_WARMUP_FRAMES}."
        ),
    )
    parser.add_argument(
        "--static-reference-frames",
        type=int,
        default=DEFAULT_STATIC_REFERENCE_FRAMES,
        help=(
            "Processed detection updates used for the startup static-scene "
            f"reference. Defaults to {DEFAULT_STATIC_REFERENCE_FRAMES}."
        ),
    )
    parser.add_argument(
        "--static-min-change-db",
        type=float,
        default=DEFAULT_STATIC_MIN_CHANGE_DB,
        help=(
            "Minimum positive range-angle power change for a static detection. "
            f"Defaults to {DEFAULT_STATIC_MIN_CHANGE_DB:g} dB."
        ),
    )
    parser.add_argument(
        "--static-background-update-rate",
        type=float,
        default=DEFAULT_STATIC_BACKGROUND_UPDATE_RATE,
        help=(
            "Adaptive update rate for unprotected static background cells. "
            f"Defaults to {DEFAULT_STATIC_BACKGROUND_UPDATE_RATE:g}."
        ),
    )
    parser.add_argument(
        "--static-cluster-min-samples",
        type=int,
        default=DEFAULT_STATIC_CLUSTER_MIN_SAMPLES,
        help=(
            "Minimum same-frame points in a static handoff cluster. "
            f"Defaults to {DEFAULT_STATIC_CLUSTER_MIN_SAMPLES}; temporal "
            "tracking still requires 3 consecutive associated updates."
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
        f"receiver_queue_drops={stats.receiver_queue_drops}, "
        f"processing_drops={stats.processing_frames_dropped}"
    )


def _format_capture_summary(stats: CaptureStats) -> str:
    valid_frames = max(stats.frames_emitted - stats.invalid_frames, 0)
    queued_frames = max(valid_frames - stats.processing_frames_dropped, 0)
    return (
        "Capture summary: "
        f"packets={stats.packets_received}, "
        f"frames={stats.frames_emitted}, "
        f"valid_frames={valid_frames}, "
        f"invalid_frames={stats.invalid_frames}, "
        f"queued_frames={queued_frames}, "
        f"lost_packets={stats.lost_packets}, "
        f"receiver_queue_drops={stats.receiver_queue_drops}, "
        f"processing_drops={stats.processing_frames_dropped}, "
        f"stream_resyncs={stats.stream_resyncs}"
    )


def _capture_error_counts(stats: CaptureStats) -> tuple[int, ...]:
    """Return only counters that indicate capture or processing errors."""
    return (
        stats.invalid_frames,
        stats.lost_packets,
        stats.out_of_order_packets,
        stats.duplicate_packets,
        stats.byte_gaps,
        stats.byte_gap_bytes,
        stats.stream_resyncs,
        stats.byte_overlaps,
        stats.byte_overlap_bytes,
        stats.malformed_packets,
        stats.receiver_queue_drops,
        stats.processing_frames_dropped,
    )


def _report_new_error_stats(
    stats: CaptureStats,
    reported_error_counts: tuple[int, ...],
    emit_func: EmitFunc,
) -> tuple[int, ...]:
    """Emit immediately when any capture or processing error counter changes."""
    error_counts = _capture_error_counts(stats)
    if error_counts != reported_error_counts:
        emit_func(_format_stats(stats))
    return error_counts


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
        start_frequency_ghz=_optional_float(
            profile_config,
            "startFreqConst_GHz",
            "startFreq",
        ),
        idle_time_us=_optional_float(
            profile_config,
            "idleTimeConst_usec",
            "idleTime",
        ),
        ramp_end_time_us=_optional_float(
            profile_config,
            "rampEndTime_usec",
            "rampEndTime",
        ),
        frame_periodicity_ms=_optional_float(
            frame_config,
            "framePeriodicity_msec",
            "framePeriodicity",
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
    start_frequency_ghz = _optional_float(
        data,
        "startFreq",
        "startFreqConst_GHz",
        "start_frequency_ghz",
    )
    idle_time_us = _optional_float(
        data,
        "idleTime",
        "idleTimeConst_usec",
        "idle_time_us",
    )
    ramp_end_time_us = _optional_float(
        data,
        "rampEndTime",
        "rampEndTime_usec",
        "ramp_end_time_us",
    )
    frame_periodicity_ms = _optional_float(
        data,
        "framePeriodicity",
        "framePeriodicity_msec",
        "frame_periodicity_ms",
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
        start_frequency_ghz=start_frequency_ghz,
        idle_time_us=idle_time_us,
        ramp_end_time_us=ramp_end_time_us,
        frame_periodicity_ms=frame_periodicity_ms,
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
    start_frequency_ghz: Optional[float] = None
    idle_time_us: Optional[float] = None
    ramp_end_time_us: Optional[float] = None
    frame_periodicity_ms: Optional[float] = None
    chirp_tx_masks: dict[int, int] = {}

    for raw_line in lines:
        line = raw_line.split("%", 1)[0].split("#", 1)[0].strip()
        if not line:
            continue

        tokens = line.split()
        command = tokens[0]
        if command == "profileCfg" and len(tokens) > 10:
            start_frequency_ghz = float(tokens[2])
            idle_time_us = float(tokens[3])
            ramp_end_time_us = float(tokens[5])
            frequency_slope_mhz_per_us = float(tokens[8])
            profile_adc_samples = int(float(tokens[10]))
            sample_rate_ksps = float(tokens[11])
        elif command == "channelCfg" and len(tokens) > 1:
            rx_channel_mask = int(tokens[1], 0)
        elif command == "frameCfg" and len(tokens) > 3:
            chirp_start_idx = int(tokens[1])
            chirp_end_idx = int(tokens[2])
            num_loops = int(tokens[3])
            if len(tokens) > 5:
                frame_periodicity_ms = float(tokens[5])
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
        start_frequency_ghz=start_frequency_ghz,
        idle_time_us=idle_time_us,
        ramp_end_time_us=ramp_end_time_us,
        frame_periodicity_ms=frame_periodicity_ms,
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
    process_context = mp.get_context("spawn")
    frame_queue: Optional[mp.Queue] = None
    log_queue: Optional[mp.Queue] = None
    processor_status_queue: Optional[mp.Queue] = None
    processor: Optional[mp.Process] = None
    rotor_post_queue: Optional[mp.Queue] = None
    rotor_post_status_queue: Optional[mp.Queue] = None
    rotor_post_failure_event: Optional[Any] = None
    rotor_postprocessor: Optional[mp.Process] = None
    postprocessed_frames_counter: Optional[Any] = None
    rotor_post_queue_high_water: Optional[Any] = None
    display: Optional[LiveDisplay] = None
    capture_stats: Optional[CaptureStats] = None
    processed_frames_counter: Optional[Any] = None
    try:
        if args.classification and importlib.util.find_spec("torch") is None:
            raise CaptureStartupError(
                "CNN classification requires PyTorch 2.6 or newer; install "
                "'torch>=2.6,<3' or run with --no-classification"
            )
        if args.display == ROTOR_DISPLAY_MODE:
            if (
                args.micro_doppler_range_m is None
                or not np.isfinite(args.micro_doppler_range_m)
                or args.micro_doppler_range_m <= 0.0
            ):
                raise CaptureStartupError(
                    "--micro-doppler-range-m must be a finite positive value "
                    "for the dedicated micro-doppler display"
                )
            if args.rotor_blades < 1 or args.rotor_count < 1:
                raise CaptureStartupError(
                    "--rotor-blades and --rotor-count must be positive"
                )
            if (
                not np.isfinite(args.rotor_rpm_min)
                or not np.isfinite(args.rotor_rpm_max)
                or args.rotor_rpm_min <= 0.0
                or args.rotor_rpm_max <= args.rotor_rpm_min
            ):
                raise CaptureStartupError(
                    "Rotor RPM bounds must be finite, positive, and increasing"
                )
            args.static_detection = False
        resolved_config_path = _resolve_config_path(args.config)
        config = RadarCaptureConfig.from_file(resolved_config_path)
        classification_artifact_dir = _resolve_output_path(
            args.classification_artifacts
        )
        resolved_classification_device = "disabled"
        if args.classification:
            try:
                resolved_classification_device = resolve_classification_device(
                    args.classification_device
                )
            except ValueError as exc:
                raise CaptureStartupError(
                    f"Invalid classification device: {exc}"
                ) from exc
            emit(
                "Classification requested: "
                f"device={resolved_classification_device}, "
                f"artifacts={classification_artifact_dir}"
            )
        if args.display == ROTOR_DISPLAY_MODE and (
            config.start_frequency_ghz is None
            or config.slow_time_interval_s is None
        ):
            raise CaptureStartupError(
                "Dedicated micro-Doppler mode requires profile start "
                "frequency, idle time, and ramp-end time"
            )
        if args.display == ROTOR_DISPLAY_MODE and (
            config.num_chirps_per_loop is None
            or config.num_chirps_per_loop <= 0
            or config.tx_channel_masks is None
            or not config.tx_channel_masks
            or any(mask <= 0 for mask in config.tx_channel_masks)
        ):
            raise CaptureStartupError(
                "Dedicated micro-Doppler mode requires a valid TX chirp "
                "schedule in the radar profile"
            )
        if (
            args.display == ROTOR_DISPLAY_MODE
            and config.frame_duty_cycle is not None
            and config.frame_duty_cycle > 0.505
        ):
            raise CaptureStartupError(
                "Radar profile exceeds the dedicated rotor mode's "
                "50% frame-duty target"
            )
        if args.display == ROTOR_DISPLAY_MODE:
            configured_range_axis = config.range_axis_m()
            if (
                configured_range_axis is None
                or configured_range_axis.size == 0
                or args.micro_doppler_range_m
                > float(configured_range_axis[-1])
            ):
                raise CaptureStartupError(
                    "The requested micro-Doppler range is outside the "
                    "configured range axis"
                )
        emit(f"Loaded radar config: {config}")
        try:
            dsp_backend = validate_openradar_backend()
        except RuntimeError as exc:
            raise CaptureStartupError(str(exc)) from exc
        emit(f"DSP backend: {dsp_backend}")
        setup_config = CaptureSetupConfig.from_file(args.setup)
        emit(f"Loaded capture setup: {setup_config}")

        log_queue = process_context.Queue(maxsize=1000)
        use_rotor_postprocess = bool(
            args.classification and args.display == ROTOR_DISPLAY_MODE
        )
        if use_rotor_postprocess:
            rotor_post_queue = process_context.Queue(
                maxsize=max(args.processing_queue_size, 1)
            )
            rotor_post_status_queue = process_context.Queue(maxsize=1)
            rotor_post_failure_event = process_context.Event()
            postprocessed_frames_counter = process_context.Value("Q", 0)
            rotor_post_queue_high_water = process_context.Value("Q", 0)
            rotor_postprocessor = process_context.Process(
                target=_run_rotor_postprocessor,
                args=(
                    config,
                    rotor_post_queue,
                    log_queue,
                    rotor_post_status_queue,
                    rotor_post_failure_event,
                    postprocessed_frames_counter,
                    args.processed_output,
                    args.display_update_every,
                    classification_artifact_dir,
                    resolved_config_path,
                    args.classification_device,
                    args.micro_doppler_range_m,
                    max(args.micro_doppler_range_half_width_bins, 0),
                    max(args.rotor_blades, 1),
                    max(args.rotor_count, 1),
                    (
                        max(args.rotor_radius_m, 0.0)
                        if args.rotor_radius_m is not None
                        else None
                    ),
                    max(args.rotor_rpm_min, 1.0),
                    max(args.rotor_rpm_max, args.rotor_rpm_min + 1.0),
                ),
                name="RotorPostProcessor",
            )
            rotor_postprocessor.start()
            try:
                rotor_post_status = rotor_post_status_queue.get(timeout=180.0)
            except queue.Empty as exc:
                raise CaptureStartupError(
                    "Rotor GPU post-processor did not report readiness within "
                    "180 seconds"
                ) from exc
            _drain_log_queue(log_queue)
            if rotor_post_status.get("state") != "ready":
                raise CaptureStartupError(
                    rotor_post_status.get(
                        "message",
                        "Rotor GPU post-processor failed during startup",
                    )
                )
            emit(rotor_post_status["message"])

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
            process_context=process_context,
        )
        frame_queue = process_context.Queue(
            maxsize=max(args.processing_queue_size, 1)
        )
        processed_frames_counter = process_context.Value("Q", 0)
        processor_status_queue = process_context.Queue(maxsize=1)
        processor = process_context.Process(
            target=_run_frame_processor,
            args=(
                config,
                frame_queue,
                log_queue,
                processed_frames_counter,
                display.payload_queue if display is not None else None,
                display.skipped_updates if display is not None else None,
                args.raw_output,
                args.raw_metadata,
                args.processed_output,
                args.display,
                args.display_update_every,
                args.max_range_m,
                args.point_cloud_fov_deg,
                args.cluster_eps_m,
                args.cluster_min_samples,
                args.clutter_map_update_rate,
                args.clutter_map_warmup_frames,
                args.clutter_map_min_snr_db,
                args.static_detection,
                max(args.static_warmup_frames, 0),
                max(args.static_reference_frames, 1),
                max(args.static_min_change_db, 0.0),
                min(max(args.static_background_update_rate, 0.0), 1.0),
                max(args.static_cluster_min_samples, 1),
                args.micro_doppler_range_m,
                max(args.micro_doppler_range_half_width_bins, 0),
                max(args.rotor_blades, 1),
                max(args.rotor_count, 1),
                (
                    max(args.rotor_radius_m, 0.0)
                    if args.rotor_radius_m is not None
                    else None
                ),
                max(args.rotor_rpm_min, 1.0),
                max(args.rotor_rpm_max, args.rotor_rpm_min + 1.0),
                args.classification,
                classification_artifact_dir,
                resolved_config_path,
                args.classification_device,
                rotor_post_queue,
                rotor_post_failure_event,
                rotor_post_queue_high_water,
                processor_status_queue,
            ),
            name="RadarFrameProcessor",
        )
        processor.start()
        processor_startup_timeout_s = (
            180.0 if resolved_classification_device == "cuda" else 30.0
        )
        try:
            processor_status = processor_status_queue.get(
                timeout=processor_startup_timeout_s
            )
        except queue.Empty as exc:
            raise CaptureStartupError(
                "Radar frame processor did not report readiness within "
                f"{processor_startup_timeout_s:g} seconds"
            ) from exc
        _drain_log_queue(log_queue)
        if processor_status.get("state") != "ready":
            raise CaptureStartupError(
                processor_status.get(
                    "message",
                    "Radar frame processor failed during startup",
                )
            )
        emit(processor_status["message"])
        capture_stats = listen_for_frames(
            host_ip=args.host_ip,
            data_port=args.data_port,
            config=config,
            setup_config=setup_config,
            buffer_size=args.buffer_size,
            socket_recv_buffer_bytes=args.socket_recv_buffer,
            socket_timeout_seconds=args.socket_timeout,
            packet_queue_size=args.packet_queue_size,
            frame_queue=frame_queue,
            log_queue=log_queue,
            display=display,
            pipeline_failure_event=rotor_post_failure_event,
            pipeline_process=rotor_postprocessor,
        )
    except CaptureStartupError as exc:
        emit(f"Capture startup failed: {exc}")
    finally:
        emit("Shutting down capture pipeline...")
        if frame_queue is not None:
            if not _request_processor_stop(frame_queue):
                emit(
                    "Frame processor stop marker could not be queued; "
                    "no queued frame was discarded."
                )

        if processor is not None:
            processor.join(timeout=10.0)
            if processor.is_alive():
                emit("Frame processor did not stop in time; terminating.")
                processor.terminate()
                processor.join(timeout=1.0)

        if rotor_postprocessor is not None:
            if rotor_postprocessor.is_alive() and rotor_post_queue is not None:
                try:
                    rotor_post_queue.put(None, timeout=1.0)
                except queue.Full:
                    pass
            rotor_postprocessor.join(timeout=30.0)
            if rotor_postprocessor.is_alive():
                emit("Rotor post-processor did not stop in time; terminating.")
                rotor_postprocessor.terminate()
                rotor_postprocessor.join(timeout=1.0)

        if log_queue is not None:
            _drain_log_queue(log_queue)

        if capture_stats is not None and processed_frames_counter is not None:
            valid_frames = max(
                capture_stats.frames_emitted - capture_stats.invalid_frames,
                0,
            )
            queued_frames = max(
                valid_frames - capture_stats.processing_frames_dropped,
                0,
            )
            processed_frames = _shared_counter_value(processed_frames_counter)
            emit(
                "Processing summary: "
                f"valid_frames={valid_frames}, "
                f"queued_frames={queued_frames}, "
                f"processed_frames={processed_frames}, "
                f"processing_drops={capture_stats.processing_frames_dropped}"
            )
            if postprocessed_frames_counter is not None:
                postprocessed_frames = _shared_counter_value(
                    postprocessed_frames_counter
                )
                postprocessing_drops = max(
                    processed_frames - postprocessed_frames,
                    0,
                )
                emit(
                    "Post-processing summary: "
                    f"processed_frames={processed_frames}, "
                    f"postprocessed_frames={postprocessed_frames}, "
                    f"queue_high_water="
                    f"{_shared_counter_value(rotor_post_queue_high_water)}, "
                    f"postprocessing_drops={postprocessing_drops}"
                )

        if display is not None:
            display.close()
            emit("Live display closed.")
            if capture_stats is not None and display.mode != "none":
                rendered_updates = display.rendered_update_count
                skipped_updates = display.skipped_update_count
                not_rendered = max(
                    capture_stats.frames_emitted - rendered_updates,
                    0,
                )
                not_rendered_rate = (
                    (100.0 * not_rendered / capture_stats.frames_emitted)
                    if capture_stats.frames_emitted
                    else 0.0
                )
                redraw_coverage = 100.0 - not_rendered_rate
                emit(
                    "Display summary: "
                    f"rendered_updates={rendered_updates}, "
                    f"coalesced_updates={skipped_updates}, "
                    f"radar_frame_redraw_coverage={redraw_coverage:.3f}%, "
                    f"radar_frames_without_dedicated_redraw="
                    f"{not_rendered}/{capture_stats.frames_emitted}. "
                    "This is GUI cadence, not capture or processing loss."
                )

        if frame_queue is not None:
            frame_queue.close()
            frame_queue.join_thread()
        if rotor_post_queue is not None:
            rotor_post_queue.close()
            rotor_post_queue.join_thread()
        if rotor_post_status_queue is not None:
            rotor_post_status_queue.close()
            rotor_post_status_queue.join_thread()
        if processor_status_queue is not None:
            processor_status_queue.close()
            processor_status_queue.join_thread()
        if log_queue is not None:
            log_queue.close()
            log_queue.join_thread()

        emit("Capture pipeline shutdown complete.")
        close_terminal_log()


if __name__ == "__main__":
    main()
