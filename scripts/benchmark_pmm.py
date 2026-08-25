"""Jetson-side synthetic worst-path benchmark for Phase 1 PMM tracking."""

from __future__ import annotations

import argparse
import json
import resource
import sys
import time
from pathlib import Path

import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from main.dsp import (  # noqa: E402
    compute_range_doppler_fft,
    compute_range_fft,
)
from main.livedatacapture import RadarCaptureConfig  # noqa: E402
from main.pmm import PmmConfig, PmmTracker  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames", type=int, default=55)
    parser.add_argument("--warmup-frames", type=int, default=5)
    parser.add_argument(
        "--config",
        type=Path,
        default=REPOSITORY_ROOT
        / "profiles"
        / "profile-mini4-20m.cfg",
    )
    args = parser.parse_args()

    frame_count = max(args.frames, args.warmup_frames + 1)
    radar_config = RadarCaptureConfig.from_file(args.config)
    random = np.random.default_rng(6843)
    radar_cube = (
        random.normal(0.0, 100.0, (96, 4, 256))
        + 1j * random.normal(0.0, 100.0, (96, 4, 256))
    ).astype(np.complex64)
    range_axis = radar_config.range_axis_m()
    assert range_axis is not None
    tracker = PmmTracker(
        radar_config,
        PmmConfig(
            background_calibration_seconds=0.1,
            detection_threshold=0.0,
            provisional_frames=1,
            confirmation_window_frames=1,
            confirmation_hits=1,
        ),
    )

    samples_ms = []
    for frame_index in range(frame_count):
        started = time.perf_counter()
        range_fft = compute_range_fft(radar_cube)
        doppler_cube = compute_range_doppler_fft(
            range_fft,
            radar_config,
            fft_size=64,
        )
        tracker.update(
            doppler_cube,
            range_fft,
            range_axis,
            timestamp_s=frame_index * 0.1,
        )
        elapsed_ms = (time.perf_counter() - started) * 1_000.0
        if frame_index >= args.warmup_frames:
            samples_ms.append(elapsed_ms)

    print(
        json.dumps(
            {
                "samples": len(samples_ms),
                "p50_ms": round(float(np.percentile(samples_ms, 50)), 2),
                "p95_ms": round(float(np.percentile(samples_ms, 95)), 2),
                "max_ms": round(float(np.max(samples_ms)), 2),
                "process_peak_rss_mib": round(
                    resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0,
                    1,
                ),
            },
            separators=(",", ":"),
        )
    )


if __name__ == "__main__":
    main()
