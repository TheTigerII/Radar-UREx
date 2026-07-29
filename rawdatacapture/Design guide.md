# Mini4 PMM Tracking Design

This repository contains one radar processing path: Phase 1 detection and
tracking of a single periodic-micro-motion (PMM) target. It does not identify
the reflector type. Every reported target is labelled `PMM target`.

## Signal path

```text
DCA1000 raw ADC
  -> validate and assemble complete frames
  -> 256-point range FFT
  -> 64-point Doppler FFT (32 samples/TX, zero-padded)
  -> PMM spectrum folding
  -> calibrated projection subtraction
  -> continuity-constrained range tracking
  -> particle-filter range/velocity smoothing
  -> 12-element ODS Capon angle search
  -> angle continuity tracking and particle smoothing
  -> JSONL and display
```

`livedatacapture.py` owns UDP receive, frame assembly, bounded queues, loss
accounting, raw recording, processing, display handoff, and processed output.
`pmm.py` owns PMM extraction and tracking. `dsp.py` contains only the FFT and
DCA1000 decoding primitives needed by this path.

## Fixed acquisition profile

Only `profile-mini4-20m.cfg` is accepted:

- 60.25 GHz start frequency.
- 3 TDM transmitters and 4 receivers.
- 256 ADC samples at 6.25 MS/s.
- 46.875 MHz/us slope.
- 6 us ADC start, 50 us ramp, and 850 us idle.
- 32 loops and 96 chirps per 100 ms frame.
- 393,216 raw bytes per frame.
- 256 range bins at approximately 7.8 cm spacing.
- 64 Doppler bins after zero-padding.

Preflight rejects any dimension, timing, slope, sampling-rate, TX order, or
frame-period mismatch. Processing and display are gated to 0.3–20 m.
The profile's `adcbufCfg` selects the IWR68xx non-interleaved ADC layout, so
each receiver's samples are decoded contiguously within a chirp.

## Capture integrity

DCA1000 packet sequence and byte counters are checked independently. Missing
bytes invalidate the affected frame; invalid frames are never processed.
Packets and complete frames move through bounded queues, and overflow counters
are included in diagnostics. A large discontinuity resynchronizes frame
ownership. Raw capture remains optional and stores complete frames with sidecar
metadata.

## PMM extraction and calibration

For every range bin, the linear-magnitude Doppler spectrum is folded for
integer sizes 2 through 20. Each folding column is averaged as specified by
Equation 10 in the paper. The maximum score and its folding size are retained.
The tracker keeps at most 36 frames (3.6 seconds at 10 Hz), including while it
is searching.

Startup requires 300 valid target-free frames by default. Their mean PMM
spectrum becomes the background. Projection-based subtraction estimates the
background gain and subtracts the projected spectrum. Until calibration
finishes, the only state is `calibrating` and no detection is emitted.
Calibration is tied to profile and feature fingerprints and is reset by a
profile mismatch or observation discontinuity.

The default score threshold is 750. It is a runtime setting derived from the
first local target-free and Mini 4 Pro recordings after correcting the ADC
layout and folding implementation. In that pair, the target-free maximum was
about 615 and the target recording's median post-calibration score was about
1,145. More target-free and controlled-flight recordings are still required
before treating 750 as a site-independent threshold.

The paper's value of 30,000 is not used here: it gates complete 3.6-second
Doppler-Time segments before the paper's identification stage, rather than
individual tracking frames. Phase 1 has no identification stage.

## Tracking

Dynamic programming maximizes cumulative PMM score while limiting each
range-bin transition. The default 4 m/s speed limit is converted to bins from
the measured frame interval and 7.8 cm range spacing. A path becomes
provisional after five valid observations.

A 5,000-particle constant-velocity filter smooths range and estimates radial
velocity. At the filtered range, Capon beamforming searches the 12-element ODS
virtual array from -60 to +60 degrees in azimuth and elevation on a 2-degree
grid. PMM folding, continuity constraints, and particle filters are also
applied to the angle estimates. Positive elevation and positive display Z both
mean upward; the ODS vertical antenna coordinate is sign-corrected to maintain
that convention.

Track states are:

- `calibrating`
- `searching`
- `tentative`
- `confirmed`
- `coasting`
- `lost`

Confirmation requires at least seven threshold hits in ten valid frames and a
valid dynamic-programming path. A confirmed track coasts for at most ten
missing frames. Timestamp discontinuity or distant reacquisition resets track
ownership.

## Output contract

Processed output is JSON Lines. The metadata record contains the exact profile
and feature fingerprints and PMM settings. Frame records contain:

- calibration progress;
- raw and background-subtracted PMM scores;
- winning folding size;
- selected range bin and filtered range;
- filtered radial velocity, azimuth, and elevation;
- track state, age, hits, misses, and predicted/measured status;
- dynamic-programming and particle-filter diagnostics;
- target-gated Doppler–Time history;
- stage latency, queue occupancy, and packet/frame-loss counters.

No object-type label or probability is produced. The target-gated
Doppler–Time history is recorded only as future research data.

## Process model

The main process receives UDP packets. A worker process performs all DSP and
serialization, and an optional PyQtGraph display process consumes a latest-only
queue. Range and image modes use native PyQtGraph plot items; target and
combined modes use `pyqtgraph.opengl` for the 3D PMM position.
The processing queue is bounded and records drops rather than allowing
unbounded memory growth. The display queue is also bounded and may discard
superseded drawings without affecting processing.

## Verification

Run:

```bash
python -m unittest discover -s . -p "test_*.py" -v
python -m unittest discover -s rawdatacapture -p "test_*.py" -v
python rawdatacapture/startup.py \
  --config rawdatacapture/profile-mini4-20m.cfg \
  --sdk-profile rawdatacapture/profile-mini4-20m.cfg \
  --setup rawdatacapture/setup.json \
  --preflight-only --skip-socket-preflight
python scripts/benchmark_pmm.py
```

The deterministic suite covers folding, subtraction, path constraints,
particles, state transitions, ODS Capon geometry, profile rejection, frame
integrity, JSONL labels, replay, and launcher wiring.
