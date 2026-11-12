"""Re-measure the Mode 4 calibrated CFAR multipliers by Monte Carlo.

Mode 4 (``main/enhanced_detection.py``) scales Mode 1's nominal OS-CFAR
threshold by a factor chosen so that noise-only frames produce a known
per-cell false-alarm probability. This script regenerates those factors and,
optionally, the simulated detection gain relative to Mode 1.

Run it again whenever the conventional profile dimensions, clutter warm-up,
CFAR windows or beam bank change, then update
``CALIBRATED_THRESHOLD_MULTIPLIER`` in ``main/enhanced_detection.py``.

    python scripts/calibrate_enhanced_cfar.py --noise-frames 1200 --gain

The simulation is complex white Gaussian noise plus one point target on the
IWR6843ISK-ODS virtual array. It contains no clutter, multipath or hardware
impairments, so its gains are an upper bound, not a field prediction.
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
for entry in (ROOT, ROOT / "testcodes"):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

from main import dsp  # noqa: E402
from main import enhanced_detection as ed  # noqa: E402
from test_enhanced_detection import (  # noqa: E402
    DOPPLER_BINS,
    make_config,
    synthetic_range_fft,
)

CFAR = dict(
    false_alarm_rate=ed.NOMINAL_FALSE_ALARM_RATE,
    range_guard_cells=2,
    doppler_guard_cells=1,
    range_training_cells=4,
    doppler_training_cells=2,
)
CONFIGURATIONS = tuple(ed.CALIBRATED_THRESHOLD_MULTIPLIER)
TARGET_RANGE_BIN = 40
TARGET_DOPPLER_BINS = 9


def margin_map(frame, config, clutter, coherent):
    """Return (power / nominal threshold, valid mask) for one frame."""
    cube = dsp.compute_range_doppler_fft(frame, config)
    if coherent:
        cube = ed.tdm_doppler_compensation(cube)
        raw = ed.coherent_beam_power(cube, config)
    else:
        raw = np.mean(np.abs(cube) ** 2, axis=(1, 2))
    power = clutter.normalize(raw)
    threshold, valid = ed.os_cfar_thresholds(power, **CFAR)
    eligible = (
        valid
        & dsp.doppler_peak_mask(power)
        & (power >= clutter.minimum_snr_linear)
    )
    clutter.update(raw, np.zeros_like(eligible))
    return np.where(eligible, power / threshold, 0.0), valid


def calibrate(detection_power, reference, args, rng):
    config = make_config()
    coherent = detection_power == ed.DETECTION_POWER_COHERENT_BEAM
    clutter = ed.create_clutter_map(
        reference,
        update_rate=0.02,
        warmup_frames=args.warmup_frames,
        minimum_snr_db=3.0,
    )
    for _ in range(args.warmup_frames):
        margin_map(synthetic_range_fft(rng), config, clutter, coherent)
    values, cells = [], 0
    for _ in range(args.noise_frames):
        margins, valid = margin_map(synthetic_range_fft(rng), config, clutter, coherent)
        kept = margins[valid]
        values.append(kept[kept > 0.02])
        cells += int(valid.sum())
    ordered = np.sort(np.concatenate(values))[::-1]
    index = int(cells * args.target_pfa)
    if index >= ordered.size:
        raise SystemExit("Not enough noise frames for the requested Pfa")
    multiplier = float(ordered[index])

    detection = None
    if args.gain:
        detection = []
        center = DOPPLER_BINS // 2 + TARGET_DOPPLER_BINS
        for amplitude in args.amplitudes:
            hits = 0
            for _ in range(args.target_frames):
                frame = synthetic_range_fft(
                    rng,
                    amplitude=amplitude,
                    range_bin=TARGET_RANGE_BIN,
                    doppler_bins=TARGET_DOPPLER_BINS,
                    azimuth_u=rng.uniform(-0.75, 0.75),
                    elevation_u=rng.uniform(-0.6, 0.6),
                )
                margins, _ = margin_map(frame, config, clutter, coherent)
                window = margins[
                    center - 2 : center + 3,
                    TARGET_RANGE_BIN - 1 : TARGET_RANGE_BIN + 2,
                ]
                hits += bool((window > multiplier).any())
            detection.append(hits / args.target_frames)
    return multiplier, cells, detection


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--noise-frames", type=int, default=1200)
    parser.add_argument("--warmup-frames", type=int, default=30)
    parser.add_argument("--target-pfa", type=float, default=ed.CALIBRATED_PFA)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--gain", action="store_true", help="Also print Pd curves")
    parser.add_argument("--target-frames", type=int, default=60)
    parser.add_argument(
        "--amplitudes",
        type=float,
        nargs="+",
        default=[0.1, 0.14, 0.2, 0.28, 0.4, 0.56, 0.8, 1.1],
    )
    args = parser.parse_args()
    warnings.filterwarnings("ignore", category=RuntimeWarning)

    print(f"Target per-cell Pfa: {args.target_pfa:g}")
    for detection_power, reference in CONFIGURATIONS:
        rng = np.random.default_rng(args.seed)
        multiplier, cells, detection = calibrate(detection_power, reference, args, rng)
        stored = ed.CALIBRATED_THRESHOLD_MULTIPLIER[(detection_power, reference)]
        print(
            f"({detection_power!r}, {reference!r}): {multiplier:.3f}"
            f"   [stored {stored}; {cells:.1e} noise cells]"
        )
        if detection is not None:
            print(
                "    Pd at amplitudes "
                + " ".join(f"{a:g}" for a in args.amplitudes)
                + ": "
                + " ".join(f"{p:.2f}" for p in detection)
            )


if __name__ == "__main__":
    main()
