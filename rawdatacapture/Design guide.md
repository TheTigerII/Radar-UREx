# Mini4 PMM Tracking Design

This repository contains one live radar processing path: Phase 1 detection and
tracking of a single periodic-micro-motion (PMM) target. The live path does not
identify the reflector type, and every runtime target is labelled `PMM target`.
The repository also contains an optional LSTM training notebook and a gated
real-time inference path. Classification is disabled unless explicitly enabled
and compatible weights are present.

## Signal path

```text
DCA1000 raw ADC
  -> validate and assemble complete frames
  -> 256-point range FFT
  -> per-TX/RX/range slow-time mean subtraction
  -> 64-point Doppler FFT (32 samples/TX, zero-padded)
  -> PMM spectrum folding
  -> calibrated projection subtraction
  -> continuity-constrained range tracking
  -> particle-filter range/velocity smoothing
  -> 12-element ODS Capon angle search
  -> angle continuity tracking and particle smoothing
  -> JSONL and display
```

The optional offline path is:

```text
labelled Mini4 PMM JSONL captures
  -> validated, non-overlapping 36-frame Doppler-Time segments
  -> PMM quality gate and strongest-bin alignment
  -> balanced 70/15/15 train/validation/test split
  -> two-layer LSTM training and locked-test evaluation
  -> PyTorch state, ONNX model, and manifest
```

When enabled, the live classifier consumes the same rolling Doppler-Time
history after PMM tracking:

```text
36-frame target-gated Doppler-Time history
  -> capture-threshold PMM quality gate
  -> training-identical alignment and normalization
  -> two-layer PyTorch LSTM
  -> other/UAV probabilities in JSONL and display status
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

Before Doppler windowing and transformation, each TX/RX/range slow-time vector
has its complex mean removed. This suppresses stationary zero-Doppler clutter
in PMM extraction and in the displayed range-Doppler and Doppler-time spectra.
It can also attenuate genuinely very slow target components.

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
individual tracking frames. The live Phase 1 path has no identification stage.
The offline notebook instead gates each segment using the compatible capture's
recorded `detection_threshold`, because the paper's value is not portable to
this acquisition profile or signal scale.

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

With classification disabled, no object-type label or probability is produced
by the live path. When enabled, metadata includes the model contract and each
update includes a `classification` object. Its status is `warming_up`,
`below_pmm_threshold`, or `classified`; inadequate histories and low-PMM
windows remain `unknown` rather than forcing an object label.

## Offline LSTM training

`training.ipynb` implements a repository-native adaptation of mmHawkeye's UAV
identification network. The paper does not disclose its exact tensor shape, so
the notebook uses the Mini4 contract: 36 frames at 10 Hz, with 64 centered
Doppler bins per frame. Input captures are discovered recursively under
`dataset/uav/` and `dataset/other/`; the parent directory supplies the binary
label, and each JSONL file represents one capture.

The notebook accepts only `mini4-pmm-jsonl` version 1 captures. It rejects
non-finite histories, wrong Doppler dimensions, changing per-frame thresholds,
and datasets that mix profile fingerprints, feature fingerprints, feature
versions, or capture thresholds. Full rolling histories are selected at
36-frame intervals so accepted examples do not overlap. A segment is retained
only if the maximum corrected PMM score in its 36 frames reaches the capture's
recorded threshold.

Slow-time complex mean subtraction in `dsp.py` is treated as the DC-removal
stage, so the notebook does not subtract DC again. Each spectrum is converted
from dB to linear amplitude; if its strongest bin is more than one bin from the
center, it is shifted to the center with linear interpolation and zero fill.
The aligned amplitude is compressed with `log1p`.

Before splitting, the majority class is deterministically undersampled to the
minority count. A seed-42, stratified segment-level split assigns approximately
70% to training, 15% to validation, and 15% to a locked test partition. Integer
rounding is resolved independently per class. At least seven usable segments
per class are required. Normalization mean and standard deviation are fitted
from training segments only. This deliberately follows the selected
segment-level policy; it does not prevent captures from contributing segments
to more than one partition.

The classifier has two stacked LSTM layers with input size 64 and hidden size
128, followed by a two-class fully connected layer. It uses cross-entropy,
Adam with a learning rate of 0.00005, batch size 10, and 100 fixed epochs. The
lowest validation-loss state is retained, and the test partition is used only
after model selection. Evaluation reports accuracy, precision, recall, F1,
the confusion matrix, the ROC curve, and ROC AUC.

The export cell writes `model_state.pt`, `model.onnx`, and `manifest.json`
under `training_output/mmhawkeye_lstm/`. The manifest records preprocessing,
labels, split membership, dataset hashes, radar/feature fingerprints,
hyperparameters, software versions, test metrics, and PyTorch/ONNX parity.
The ONNX interface is float32 `doppler_time` with shape `[batch, 36, 64]` and
two output logits. A trained `model_state.pt` must be copied into the
`model_weights/` directory before it can be enabled for live classification.

`inference.py` loads the checkpoint with PyTorch's weights-only loader and
rejects missing metadata, unexpected label order, architecture or input-shape
changes, invalid normalization, and mismatched profile/feature fingerprints.
It maintains the rolling PMM score window, returns `warming_up` until all 36
frames are present, and returns `unknown` if the maximum score is below the
runtime threshold. Otherwise it applies the notebook's dB-to-amplitude,
strongest-bin alignment, `log1p`, and stored normalization operations before
computing the two probabilities. Inference runs in the DSP worker rather than
the UDP receiver, and its latency is included in processing diagnostics.

All notebook cells are committed without execution counts or outputs. Training,
evaluation, and permanent artifact creation occur only when an operator runs
the notebook.

## Process model

The main process receives UDP packets. A worker process performs all DSP,
optional LSTM inference, and serialization, and an optional PyQtGraph display
process consumes a latest-only queue. Range and image modes use native
PyQtGraph plot items; target and combined modes use `pyqtgraph.opengl` for the
3D PMM position.
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
integrity, JSONL labels, replay, launcher prompts, model contract rejection,
rolling inference warm-up and gating, and classification output serialization.
