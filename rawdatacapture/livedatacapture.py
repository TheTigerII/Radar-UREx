import argparse
import json
import socket
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional

import numpy as np


# Default DCA1000 network parameters.
UDP_IP = "192.168.33.30"  # Host/laptop static Ethernet IP
UDP_PORT = 4098           # DCA1000 raw ADC data port
BUFFER_SIZE = 65535       # Max UDP packet payload allocation

DCA1000_HEADER_SIZE = 10
UINT32_MODULO = 2**32


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
    payload_bytes_received: int = 0
    frames_emitted: int = 0
    malformed_packets: int = 0
    lost_packets: int = 0
    out_of_order_packets: int = 0
    duplicate_packets: int = 0
    byte_gaps: int = 0
    byte_gap_bytes: int = 0
    byte_overlaps: int = 0
    byte_overlap_bytes: int = 0


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

    @classmethod
    def from_dimensions(
        cls,
        *,
        num_adc_samples: int,
        num_rx_channels: int,
        num_chirps_per_frame: int,
        iq_swap: bool = False,
        channel_interleave: bool = True,
        lvds_lanes: int = 2,
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
        )

    @classmethod
    def from_file(cls, config_path: Path) -> "RadarCaptureConfig":
        suffix = config_path.suffix.lower()
        if suffix == ".cfg":
            return cls.from_mmwave_cfg(config_path)
        if suffix == ".json":
            return cls.from_mmwave_json(config_path)
        if suffix == ".xml":
            return cls.from_mmwave_xml(config_path)
        raise ValueError(f"Unsupported config file extension: {config_path.suffix}")

    @classmethod
    def from_mmwave_cfg(cls, config_path: Path) -> "RadarCaptureConfig":
        return _config_from_cfg_lines(config_path.read_text(encoding="utf-8").splitlines())

    @classmethod
    def from_mmwave_json(cls, config_path: Path) -> "RadarCaptureConfig":
        data = json.loads(config_path.read_text(encoding="utf-8"))

        command_config = _config_from_json_command_lines(data)
        if command_config is not None:
            return command_config

        return _config_from_mapping(data, source_name="JSON")

    @classmethod
    def from_mmwave_xml(cls, config_path: Path) -> "RadarCaptureConfig":
        data = _xml_to_config_data(config_path)

        command_config = _config_from_json_command_lines(data)
        if command_config is not None:
            return command_config

        return _config_from_mapping(data, source_name="XML")


class FrameBuffer:
    def __init__(self, bytes_per_frame: int, stats: CaptureStats) -> None:
        self.bytes_per_frame = bytes_per_frame
        self.stats = stats
        self.base_byte_count: Optional[int] = None
        self.next_stream_offset = 0
        self.buffer = bytearray()

    def add_payload(
        self, header: DCA1000PacketHeader, payload: bytes
    ) -> list[bytes]:
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
            self.buffer.extend(b"\x00" * gap)
            self.next_stream_offset += gap
            self.stats.byte_gaps += 1
            self.stats.byte_gap_bytes += gap

        if payload_start < self.next_stream_offset:
            overlap = self.next_stream_offset - payload_start
            payload = payload[overlap:]
            self.stats.byte_overlaps += 1
            self.stats.byte_overlap_bytes += overlap

        self.buffer.extend(payload)
        self.next_stream_offset += len(payload)

        frames: list[bytes] = []
        while len(self.buffer) >= self.bytes_per_frame:
            frames.append(bytes(self.buffer[: self.bytes_per_frame]))
            del self.buffer[: self.bytes_per_frame]
            self.stats.frames_emitted += 1

        return frames


def process_complete_frame(frame_bytes: bytes, config: RadarCaptureConfig) -> None:
    adc_samples = np.frombuffer(frame_bytes, dtype="<i2")
    print(
        "Complete frame "
        f"{len(adc_samples)} int16 samples "
        f"({config.num_chirps_per_frame} chirps, "
        f"{config.num_rx_channels} RX, "
        f"{config.num_adc_samples} ADC samples)"
    )


def listen_for_frames(
    *,
    host_ip: str,
    data_port: int,
    config: RadarCaptureConfig,
    buffer_size: int,
) -> None:
    stats = CaptureStats()
    sequence_tracker = SequenceTracker(stats)
    frame_buffer = FrameBuffer(config.bytes_per_frame, stats)

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((host_ip, data_port))

    print(
        "Listening for live radar stream "
        f"on {host_ip}:{data_port}; bytes_per_frame={config.bytes_per_frame}"
    )
    print("Trigger frames now. Press Ctrl+C to stop.")

    try:
        while True:
            packet, _addr = sock.recvfrom(buffer_size)
            try:
                header = DCA1000PacketHeader.parse(packet)
            except ValueError:
                stats.malformed_packets += 1
                continue

            payload = packet[DCA1000_HEADER_SIZE:]
            stats.packets_received += 1
            stats.payload_bytes_received += len(payload)

            sequence_tracker.observe(header.sequence_number)
            for frame in frame_buffer.add_payload(header, payload):
                process_complete_frame(frame, config)

            if stats.packets_received % 200 == 0:
                print(_format_stats(stats))
    except KeyboardInterrupt:
        print("Streaming stopped.")
        print(_format_stats(stats))
    finally:
        sock.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Receive DCA1000 UDP ADC packets and emit complete radar frames."
    )
    parser.add_argument(
        "--config",
        required=True,
        type=Path,
        help="Radar .cfg, mmWave Studio XML, or JSON used to derive frame size.",
    )
    parser.add_argument("--host-ip", default=UDP_IP, help="Host Ethernet IP to bind.")
    parser.add_argument("--data-port", type=int, default=UDP_PORT)
    parser.add_argument("--buffer-size", type=int, default=BUFFER_SIZE)
    return parser.parse_args()


def _format_stats(stats: CaptureStats) -> str:
    return (
        "stats: "
        f"packets={stats.packets_received}, "
        f"frames={stats.frames_emitted}, "
        f"lost_packets={stats.lost_packets}, "
        f"out_of_order={stats.out_of_order_packets}, "
        f"duplicates={stats.duplicate_packets}, "
        f"byte_gaps={stats.byte_gaps}/{stats.byte_gap_bytes}B, "
        f"byte_overlaps={stats.byte_overlaps}/{stats.byte_overlap_bytes}B, "
        f"malformed={stats.malformed_packets}"
    )


def _config_from_json_command_lines(data: Any) -> Optional[RadarCaptureConfig]:
    command_lines = list(_iter_json_command_lines(data))
    if not command_lines:
        return None

    try:
        return _config_from_cfg_lines(command_lines)
    except ValueError:
        return None


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
        rx_channel_mask = _required_int(
            data, "rxChannelEn", "RxChannelEn", "rxChannelEnable", "rxChanEn"
        )
        num_rx_channels = _bit_count(rx_channel_mask)

    num_chirps_per_frame = _optional_int(
        data,
        "num_chirps_per_frame",
        "numChirpsPerFrame",
        "NumChirpsPerFrame",
        "NumOfChirpsPerFrame",
    )
    if num_chirps_per_frame is None:
        chirp_start_idx = _required_int(
            data, "chirpStartIdx", "ChirpStartIdx", "chirpStartIndex"
        )
        chirp_end_idx = _required_int(
            data, "chirpEndIdx", "ChirpEndIdx", "chirpEndIndex"
        )
        num_loops = _required_int(data, "numLoops", "NumLoops", "numOfLoops")
        num_chirps_per_frame = num_loops * (chirp_end_idx - chirp_start_idx + 1)

    iq_swap = bool(_optional_int(data, "iqSwap", "IQSwap", "sampleSwap") or 0)
    channel_interleave_value = _optional_int(
        data, "channelInterleave", "ChannelInterleave", "chInterleave"
    )
    channel_interleave = (
        True if channel_interleave_value is None else channel_interleave_value == 0
    )

    lvds_lanes = _optional_int(
        data, "lvds_lanes", "lvdsLanes", "NumOfLanes", "numLanes"
    )
    if lvds_lanes is None:
        lane_mask = _optional_int(data, "laneEn", "LaneEn", "lvdsLaneEn", "laneEnable")
        lvds_lanes = _bit_count(lane_mask) if lane_mask is not None else 2

    if num_adc_samples <= 0 or num_rx_channels <= 0 or num_chirps_per_frame <= 0:
        raise ValueError(f"{source_name} radar dimensions must be positive")

    return RadarCaptureConfig.from_dimensions(
        num_adc_samples=num_adc_samples,
        num_rx_channels=num_rx_channels,
        num_chirps_per_frame=num_chirps_per_frame,
        iq_swap=iq_swap,
        channel_interleave=channel_interleave,
        lvds_lanes=lvds_lanes,
    )


def _xml_to_config_data(config_path: Path) -> dict[str, Any]:
    root = ET.parse(config_path).getroot()
    data: dict[str, Any] = {}
    command_lines: list[str] = []

    for element in root.iter():
        tag = _local_xml_name(element.tag)
        text = (element.text or "").strip()

        if text:
            data.setdefault(tag, text)
            if _looks_like_mmwave_command(text):
                command_lines.append(text)

        for key, value in element.attrib.items():
            clean_key = _local_xml_name(key)
            clean_value = value.strip()
            data.setdefault(clean_key, clean_value)
            if _looks_like_mmwave_command(clean_value):
                command_lines.append(clean_value)

        name = _first_attribute(element, "name", "Name", "key", "Key", "id", "Id")
        value = _first_attribute(element, "value", "Value", "val", "Val")
        if name is not None and value is not None:
            data.setdefault(name, value)

        if value is not None:
            data.setdefault(tag, value)

    if command_lines:
        data["commandLines"] = command_lines

    return data


def _config_from_cfg_lines(lines: Iterable[str]) -> RadarCaptureConfig:
    profile_adc_samples: Optional[int] = None
    rx_channel_mask: Optional[int] = None
    chirp_start_idx: Optional[int] = None
    chirp_end_idx: Optional[int] = None
    num_loops: Optional[int] = None
    iq_swap = False
    channel_interleave = True
    lvds_lanes = 2

    for raw_line in lines:
        line = raw_line.split("%", 1)[0].split("#", 1)[0].strip()
        if not line:
            continue

        tokens = line.split()
        command = tokens[0]
        if command == "profileCfg" and len(tokens) > 10:
            profile_adc_samples = int(float(tokens[10]))
        elif command == "channelCfg" and len(tokens) > 1:
            rx_channel_mask = int(tokens[1], 0)
        elif command == "frameCfg" and len(tokens) > 3:
            chirp_start_idx = int(tokens[1])
            chirp_end_idx = int(tokens[2])
            num_loops = int(tokens[3])
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


def _looks_like_mmwave_command(value: str) -> bool:
    known_commands = {
        "profileCfg",
        "channelCfg",
        "frameCfg",
        "adcbufCfg",
        "lvdsLaneCfg",
        "laneCfg",
    }
    stripped = value.strip()
    return bool(stripped) and stripped.split(maxsplit=1)[0] in known_commands


def _first_attribute(element: ET.Element, *names: str) -> Optional[str]:
    normalized_names = {_normalize_key(name) for name in names}
    for key, value in element.attrib.items():
        if _normalize_key(key) in normalized_names:
            return value.strip()
    return None


def _local_xml_name(value: str) -> str:
    return value.rsplit("}", 1)[-1]


def _required_int(data: Any, *names: str) -> int:
    value = _optional_value(data, *names)
    if value is None:
        raise ValueError(f"Missing required JSON field; tried: {', '.join(names)}")
    return _to_int(value)


def _optional_int(data: Any, *names: str) -> Optional[int]:
    value = _optional_value(data, *names)
    return None if value is None else _to_int(value)


def _optional_value(data: Any, *names: str) -> Any:
    normalized_names = {_normalize_key(name) for name in names}
    for key, value in _walk_json(data):
        if _normalize_key(key) in normalized_names:
            return value
    return None


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


def main() -> None:
    args = parse_args()
    config = RadarCaptureConfig.from_file(args.config)
    print(f"Loaded radar config: {config}")
    listen_for_frames(
        host_ip=args.host_ip,
        data_port=args.data_port,
        config=config,
        buffer_size=args.buffer_size,
    )


if __name__ == "__main__":
    main()
