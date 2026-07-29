"""Deterministic offline replay for Mini4-20m DCA1000 raw-frame recordings."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterator

if __package__ in {None, ""}:
    repository_root = str(Path(__file__).resolve().parent.parent)
    if repository_root not in sys.path:
        sys.path.insert(0, repository_root)

from rawdatacapture.dsp import (
    compute_range_doppler_fft,
    compute_range_fft,
    frame_bytes_to_radar_cube,
)
from rawdatacapture.livedatacapture import RadarCaptureConfig
from rawdatacapture.pmm import (
    MINI4_DEFAULT_DETECTION_THRESHOLD,
    PmmConfig,
    PmmTracker,
    validate_mini4_profile,
)


def replay_raw_frames(
    raw_path: Path,
    radar_config: RadarCaptureConfig,
    pmm_config: PmmConfig,
) -> Iterator[dict]:
    """Yield one JSON-safe PMM result for each complete raw ADC frame."""
    validate_mini4_profile(radar_config)
    frame_bytes = radar_config.bytes_per_frame
    range_axis = radar_config.range_axis_m()
    if range_axis is None:
        raise ValueError("Mini4 replay profile has no physical range axis")
    tracker = PmmTracker(radar_config, pmm_config)
    frame_period_s = float(radar_config.frame_periodicity_ms or 100.0) * 1e-3

    with Path(raw_path).open("rb") as source:
        frame_index = 0
        while True:
            payload = source.read(frame_bytes)
            if not payload:
                break
            if len(payload) != frame_bytes:
                raise ValueError(
                    "Raw replay ends with an incomplete frame: "
                    f"expected={frame_bytes}, observed={len(payload)}"
                )
            radar_cube = frame_bytes_to_radar_cube(payload, radar_config)
            range_fft = compute_range_fft(radar_cube)
            doppler_cube = compute_range_doppler_fft(
                range_fft,
                radar_config,
                fft_size=64,
            )
            result = tracker.update(
                doppler_cube,
                range_fft,
                range_axis,
                timestamp_s=frame_index * frame_period_s,
            )
            yield {
                "record_type": "pmm_replay",
                "frame_index": frame_index,
                "pmm_tracking": result.to_dict(),
            }
            frame_index += 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Replay complete Mini4-20m DCA1000 raw ADC frames."
    )
    parser.add_argument("raw_path", type=Path)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).with_name("profile-mini4-20m.cfg"),
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--calibration-seconds", type=float, default=30.0)
    parser.add_argument(
        "--threshold",
        type=float,
        default=MINI4_DEFAULT_DETECTION_THRESHOLD,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    radar_config = RadarCaptureConfig.from_file(args.config)
    pmm_config = PmmConfig(
        background_calibration_seconds=args.calibration_seconds,
        detection_threshold=args.threshold,
    )
    output = args.output.open("w", encoding="utf-8") if args.output else None
    try:
        for record in replay_raw_frames(args.raw_path, radar_config, pmm_config):
            line = json.dumps(record, separators=(",", ":"))
            if output is not None:
                output.write(line + "\n")
            else:
                print(line)
    finally:
        if output is not None:
            output.close()


if __name__ == "__main__":
    main()
