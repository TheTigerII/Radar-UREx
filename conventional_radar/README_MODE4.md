# Mode 4 — enhanced conventional detection

Mode 4 is Mode 1 (the original conventional range-Doppler/CNN pipeline) with
four opt-in detection stages. It uses the same radar profile
(`profiles/profile.cfg`), the same August 2026 range/channel calibration, the
same clustering, tracker and CNN. Only the step that decides which
range-Doppler cells are detections changes, so a range difference between
Mode 1 and Mode 4 on the same course is attributable to these stages.

Status: implemented and unit-tested on synthetic data. **Not yet run on the
radar.** No field range is claimed.

## Why Mode 1 leaves range unused

Reading `main/dsp.py` and simulating its detector on noise-only frames showed:

1. **The CFAR threshold is far above its own target.** OS-CFAR is configured
   for a false-alarm probability of 1e-3 per cell, but its multiplier assumes
   single-look exponential noise while the tested value is a 12-channel
   average. A cell must also pass the range test *and* the Doppler test, be a
   Doppler peak and exceed a 3 dB floor. In 8 million simulated noise cells
   the nominal threshold produced zero false alarms; the threshold needed for
   1e-6 is about one fifth of it (7–8 dB lower).
2. **The clutter reference is noisy.** `AdaptiveClutterMap` seeds itself with
   the first frame and applies a 0.02 exponential update for 30 frames. The
   first frame keeps ~56% of the weight, so every later frame is divided by a
   reference worth about three frames of averaging.
3. **Detection ignores the array.** Power is averaged across the 12 virtual
   channels without phase, so the array's directional gain is not used until
   after a cell has already been detected.
4. **Angle processing has no TDM phase correction.** TX slots fire 39 µs apart;
   a moving target's phase advances between them. This is small at walking
   speed but must be corrected before channels can be combined coherently.
5. **Every frame is judged alone.** A target just below threshold in every
   frame is never declared.

## The four stages

| Stage | Flag | What it does | Extra processing |
|---|---|---|---|
| Calibrated threshold | `--cfar-threshold calibrated` | Scales the nominal CFAR threshold to a measured noise-only false-alarm probability of 1e-6 per cell (about one stray cell every 5 s, removed by track confirmation) | None |
| Averaged clutter reference | `--clutter-reference mean` | Clutter warm-up reference is the equal-weight mean of the 30 warm-up frames | None |
| Coherent beam detection | `--detection-power coherent-beam` | Corrects TDM Doppler phase, applies the calibrated channel coefficients, forms 37 receive beams across ±60° on the 4×4 ODS virtual grid and uses the strongest beam as the detection value. The corrected samples are also used for the angle estimate | About +2 ms per frame on a laptop; **must be measured on the Jetson** |
| Multi-frame integration | `--tbd-window-frames 8` | Sums each cell's margin (power ÷ threshold) along paths that move at most 2 range bins and 2 Doppler bins per frame over 8 frames (~0.27 s). Declares a target when the path mean reaches 0.50 and the current frame reaches 0.25, and no ordinary detection is within 2 range bins | About +0.5 ms per frame on a laptop |

With no flags the capture command is identical to Mode 1 and `dsp.py` is not
modified. The implementation is `main/enhanced_detection.py`; the frame
processor calls it only when at least one stage is enabled.

## Simulated effect (upper bound)

Synthetic frames: complex white noise plus one moving point target at random
angles on the ODS virtual array, run through the real detection code. Values
are the reduction in target power needed for the stated detection
probability, relative to Mode 1. Range scales with the fourth root of power.

| Configuration | Pd = 0.5 | Pd = 0.9 | Range factor |
|---|---:|---:|---:|
| Mode 1 | 0 dB | 0 dB | 1.00 |
| Calibrated threshold | 8.3 dB | 7.4 dB | 1.5–1.6 |
| + averaged clutter reference | 9.2 dB | 9.0 dB | 1.7 |
| + coherent beams | 11.6 dB | 10.3 dB | 1.8–1.9 |
| + multi-frame integration | about 3.6 dB more at Pd = 0.5 | | about 2.2–2.4 |

This is thermal noise only. It contains no clutter, multipath, foliage or
hardware effects. Real scenes will produce more false detections at the lower
threshold and less gain than this. Treat the numbers as the most the stages
could deliver, and measure the real figure against Mode 1.

## Running it

GUI: choose **4 — Enhanced conventional (extended-range detection)**. The four
"Mode 4 stages" checkboxes are on by default. Untick a stage to remove it.

Command line (Mode 1 command plus flags):

```bash
python conventional_radar/main/run.py ... \
  --cfar-threshold calibrated --clutter-reference mean \
  --detection-power coherent-beam --tbd-window-frames 8
```

The capture log prints `Enhanced conventional detection (Mode 4): ...` and the
runtime-performance log metadata records the enabled stages under
`enhanced_detection`, so a run's configuration can be identified afterwards.

## First checks on the Jetson

1. Run the test suite: `python -m pytest -q testcodes` from
   `conventional_radar/` (270 tests; on Windows one unrelated path-separator
   test in `test_run.py` fails).
2. Run Mode 4 with all stages for a few minutes with the scene empty and read
   the runtime-performance log. Mode 1 already uses ~32 ms (p95) of its
   33.3 ms frame period. If period overruns rise, untick **Coherent
   12-channel beam detection** first: the two free stages carry most of the
   simulated gain.
3. With the scene still empty, watch for false tracks. If stray points form
   confirmed tracks, untick **Multi-frame integration**, then **Calibrated
   CFAR threshold**.

## Measuring the improvement

Use the Mode 1 / Mode 4 pair on the same marked course, same target, same
empty-scene warm-up:

1. Mode 1.
2. Mode 4 with only the two free stages.
3. Mode 4 with coherent beams added.
4. Mode 4 with all four stages.

Record maximum repeatable range, false tracks per minute and period overruns
for each. Three passes per configuration.

## Re-calibrating

The threshold multipliers in `CALIBRATED_THRESHOLD_MULTIPLIER` were measured
for 128 Doppler × 64 range bins, a 30-frame warm-up and Mode 1's CFAR windows.
If any of those change:

```bash
cd conventional_radar
python scripts/calibrate_enhanced_cfar.py --noise-frames 1200 --gain
```

## Limits

- The 12.5 m unambiguous range of the Mode 1 profile is a hard ceiling.
- The CNN is unchanged and still needs 48 valid frames; a longer detection
  range does not by itself mean the classifier is accurate there.
- Coherent beams rely on the stored channel calibration being right. A stale
  calibration reduces their gain; the other three stages do not depend on it.
- The static/hover detection route is unchanged.
