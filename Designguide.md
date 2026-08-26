# Mini4 PMM Tracking Design

This repository contains one normal live radar processing path for Phase 1
detection and tracking of a single periodic-micro-motion (PMM) target, plus
separate range/RX-channel and angular hardware-calibration modes. The PMM path
does not identify the reflector type, and every runtime target is labelled
`PMM target`. The repository also contains an optional LSTM training notebook
and a gated real-time inference path. Classification is disabled unless
explicitly enabled and compatible weights are present.

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
  -> identification branch: Doppler spectrum at tracked range
  -> center-bin DC baseline subtraction and body-peak alignment
  -> 12-element ODS Capon angle search
  -> calibrated A-PMM projection subtraction
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
  -> two-layer LSTM through TensorRT on the Jetson GPU
  -> other/UAV probabilities in JSONL and display status
```

`livedatacapture.py` owns UDP receive, frame assembly, bounded queues, loss
accounting, raw recording, processing, display handoff, and processed output.
`pmm.py` owns PMM extraction and tracking. `dsp.py` contains only the FFT and
DCA1000 decoding primitives needed by this path.

## Fixed acquisition profile

Normal PMM tracking accepts an SDK CLI `.cfg` whose acquisition fields match
the Mini4 contract. `profile-mini4-20m.cfg` is the default and reference
profile:

- 60.25 GHz start frequency.
- 3 TDM transmitters and 4 receivers.
- 256 ADC samples at 6.25 MS/s.
- 46.875 MHz/us slope.
- 6 us ADC start, 50 us ramp, and 850 us idle.
- 32 loops and 96 chirps per 100 ms frame.
- 393,216 raw bytes per frame.
- 256 range bins at approximately 7.8 cm spacing.
- 64 Doppler bins after zero-padding.

The slow-time vectors are Hann-windowed and transformed without complex-mean
subtraction. As in the paper, tracking suppresses static background only after
PMM folding by projecting each Range-Time-PMM or Angle-Time-PMM column onto its
target-free background spectrum and subtracting that projection. The same
non-demeaned Doppler spectrum remains available for identification, preserving
a hovering target's body-velocity peak.

The capture worker rejects any dimension, timing, slope, sampling-rate, TX
order, or frame-period mismatch. The startup preflight separately validates
positive dimensions, two-lane LVDS support, DCA1000 settings, serial settings,
the SDK command file, and (unless skipped) the UDP bind. Processing and display
are gated to 0.3–20 m after applying the configured range bias. The profile's
`adcbufCfg` selects the IWR68xx non-interleaved ADC layout, so each receiver's
samples are decoded contiguously within a chirp. Range/RX compensation and the
host angular-bias comment may change without changing the acquisition
contract.

## Capture integrity

DCA1000 packet sequence and byte counters are checked independently. Missing
bytes invalidate the affected frame; invalid frames are never processed.
Packets and complete frames move through bounded queues, and overflow counters
are included in diagnostics. A large discontinuity resynchronizes frame
ownership. Raw capture remains optional and stores complete frames with sidecar
metadata.

## Radar and DCA1000 startup control plane

`main/startup.py` configures the radar and DCA1000 hardware but does not
receive raw ADC packets. `main/run.py` is the recommended integrated launcher:
it starts `main/livedatacapture.py`, waits for the receiver and frame processor
to report readiness, and only then starts the hardware control process.

### Responsibilities and backends

The startup control plane:

- parses radar dimensions and commands from the SDK CLI `.cfg` profile;
- parses board and DCA1000 settings from `profiles/setup.json`;
- validates dimensions, two-lane LVDS support, ports, hardware settings, the
  direct-serial command profile, and the capture UDP bind address;
- configures DCA1000 over UDP or simulates it with the dry-run backend;
- configures SDK CLI firmware over the radar command UART or uses dry-run;
- defers `sensorStart` until DCA1000 recording is armed; and
- sends `sensorStop` and DCA1000 `RECORD_STOP` during shutdown.

It does not flash firmware. The radar backends are `dry-run` (the default) and
`direct-serial`. Direct serial uses pyserial, sends commands from
`--sdk-profile`, and defers the profile's `sensorStart`. Blank lines and `%` or
`#` comments are ignored. Command responses complete on `Done`, `Error`, the
`mmwDemo:/>` prompt, `Ignored`, or `Skipped`; an error response fails startup
unless cleanup is already in progress.

The DCA1000 backends are `dry-run` and `direct-udp`. Direct UDP binds local
port 4096 by default and sends commands to `192.168.33.180:4096` in this order:

```text
SYSTEM_CONNECT
RESET_FPGA
CONFIG_FPGA_GEN
CONFIG_RECORD
RECORD_START
```

Shutdown sends `RECORD_STOP` if recording was armed. Each command validates
its acknowledgement and honors `--dca-timeout` and `--dca-retries`. FPGA and
packet payloads are derived from the setup and radar configuration, with
optional hexadecimal overrides under `directUdpDCA1000.payloads` in
`profiles/setup.json`.

The only in-process `--capture-backend` is `dry-run`. It advances orchestration
state and reports the expected address and frame size but does not bind or read
the data stream. Live reception belongs to `main/livedatacapture.py`.

### Configuration and preflight

The radar configuration and SDK command-profile defaults both resolve to
`profiles/profile-mini4-20m.cfg`; capture-board settings resolve to
`profiles/setup.json`. With direct serial, if `--config` remains at its default,
the selected `--sdk-profile` also supplies the capture dimensions. Normally
the same `.cfg` should be used for both so hardware programming and frame
interpretation cannot diverge.

Preflight checks positive ADC, RX, chirp, and frame-byte dimensions; exactly
two LVDS lanes; supported DCA1000 settings; valid UDP ports; direct-serial port
and baud settings; an existing SDK profile containing `sensorStart`; and the
ability to bind `host_ip:data_port`. The socket probe closes immediately.
Because the integrated launcher already has the real receiver bound, it starts
the control process with `--skip-socket-preflight`.

`--preflight-only` stops after validation and never configures hardware.

### Orchestration and shutdown

```text
load configs
  -> CONFIGS_LOADED
preflight validation
  -> PREFLIGHT_PASSED
configure DCA1000
  -> DCA1000_READY
configure radar, excluding sensorStart
  -> RADAR_READY
start the capture backend
  -> RECEIVER_READY
arm DCA1000 and wait --readiness-delay (default 0.25 s)
  -> DCA1000_ARMED
send deferred sensorStart
  -> RADAR_STREAMING
```

On shutdown, cleanup attempts radar sensor stop, DCA1000 record stop and
control-socket close, then capture-backend close. Cleanup continues if one
step fails, reports all failures, and ends in `STOPPED`.

Current limitations are that firmware flashing is not implemented, the
startup module has no real capture backend, direct serial assumes compatible
SDK CLI firmware is already running, and DCA1000 direct-UDP compatibility
depends on its firmware accepting the generated or overridden payloads. Packet
and frame health are reported by `main/livedatacapture.py`, not by the startup
state monitor.

## Hardware calibration path

The `calibration`, `azimuth-calibration`, and `elevation-calibration` modes are
separate from PMM background calibration and normal processed output. The
launcher reads `profiles/profile_calibration.cfg`, creates a temporary runtime
profile that enables raw LVDS, disables GUI output and firmware range
calibration, and starts a range-FFT-only calibration worker. It writes a JSON
report after a stable result and changes the operational profile only after an
interactive confirmation.

Range/RX-channel calibration uses a laser-measured reflector distance. It
forms a Hann-weighted zero-Doppler response across loops, searches the requested
range window, interpolates the peak, and accumulates the 12 physical TX/RX
channels. The default acceptance contract is 16 warm-up frames followed by 64
stable frames, with at least 10 dB peak prominence, range standard deviation at
most 0.01 m, phase standard deviation at most 5 degrees, and channel-magnitude
coefficient of variation at most 0.05. Applying the result replaces the single
`compRangeBiasAndRxChanPhase` command atomically after creating a timestamped
backup.

Angular calibration uses the operational profile's range bias and RX-channel
coefficients as host-side corrections while the temporary calibration profile
runs on the radar. The bias-corrected range axis selects the reflector bin.
That bin is processed by the same compensated ODS geometry, Capon beamformer,
Doppler FFT, PMM folding range, and angle grid as runtime tracking, and 64
accepted angle estimates are accumulated to a standard deviation of at most
1 degree.
Applying an azimuth or elevation result updates an SDK-safe
`% hostAngleCalibration` comment while preserving the other angular bias,
again with a timestamped backup.

## PMM extraction and calibration

For every range bin, the linear-magnitude Doppler spectrum is folded for
integer sizes 2 through 32. Each folding column is averaged as specified by
Equation 10 in the paper. The maximum score and its folding size are retained.
The tracker keeps at most 36 frames (3.6 seconds at 10 Hz), including while it
is searching.

Startup requires 300 valid target-free frames by default. Their mean PMM
spectrum becomes the background. Projection-based subtraction estimates the
background gain and subtracts the projected spectrum. The tracker also applies
that final background to every stored calibration frame, then learns a frozen
threshold independently for each range bin as the residual median plus six
scaled median absolute deviations, with a default linear-score floor of 700.
In other words, each threshold is the larger of the calibrated value and 700.
These thresholds are never updated after
startup calibration. Adaptive tracking paths use score-to-threshold ratios so
that a naturally noisy range bin does not win merely because its raw residual
scale is larger. An optional fixed-score override remains available for
repeatability with legacy captures. During each target-free
frame, the strongest range-PMM bin is also passed through the Capon angle
search. The mean azimuth and elevation PMM marginals become the A-PMM
backgrounds used for paper-style projection subtraction before angle dynamic
programming. Until calibration finishes, the only state is `calibrating` and
no detection is emitted.
The metadata records profile and feature fingerprints for compatibility
checking. A changed range axis or incompatible Doppler-cube shape resets the
learned background calibration. A non-monotonic timestamp or a gap greater
than 1.5 frame periods resets track ownership and histories but preserves the
learned background.

The default detector uses the frozen adaptive thresholds. The sigma multiplier
is a site-dependent runtime setting rather than a universal constant, so
target-free and controlled-flight recordings are required before deploying it
at a new site.

The paper's value of 30,000 is not used here: it gates complete 3.6-second
Doppler-Time segments before the paper's identification stage, rather than
individual tracking frames. The core PMM tracker has no identification stage;
object classification is the separate optional path described below. The
offline notebook gates each segment using the compatible capture's recorded
`detection_threshold`, because the paper's value is not portable to this
acquisition profile or signal scale.

## Tracking

Dynamic programming maximizes cumulative PMM score while limiting each
range-bin transition. The default 4 m/s speed limit is converted to bins from
the measured frame interval and 7.8 cm range spacing. PMM path evaluation
begins after five corrected observations by default; a threshold hit can then
create a `tentative` track.

A 5,000-particle constant-velocity filter smooths range and estimates radial
velocity. At the filtered range, Capon beamforming searches the 12-element ODS
virtual array from -60 to +60 degrees in azimuth and elevation on a 2-degree
grid. The folded azimuth and elevation marginals undergo calibrated projection
subtraction before continuity constraints and particle filters are applied.
Positive elevation and positive display Z both mean upward; the ODS vertical
antenna coordinate is sign-corrected to maintain that convention.

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
and feature fingerprints and PMM settings. Update records contain:

- calibration progress;
- raw and background-subtracted PMM scores;
- range, azimuth, and elevation background-projection gains;
- winning folding size;
- selected range bin and filtered range;
- filtered radial velocity, azimuth, and elevation;
- track state, age, hits, misses, and predicted/measured status;
- dynamic-programming diagnostics and the configured particle count;
- target-gated Doppler–Time history;
- stage latency, queue occupancy, and packet/frame-loss counters.

With classification disabled, no object-type label or probability is produced
by the live path. When enabled, metadata includes the model contract and each
update includes a `classification` object. Its status is `warming_up`,
`below_pmm_threshold`, or `classified`; inadequate histories and low-PMM
windows remain `unknown` rather than forcing an object label.

Optional classification evaluation is implemented separately in
`main/classification_evaluation.py`. `run.py` prompts for this opt-in only after
classification is enabled; direct capture uses explicit `--inference-logging`,
`--inference-log`, and `--evaluation-label` arguments. Calibration modes and
classification-disabled runs reject evaluation logging.

The DSP worker owns the version 2 evaluation logger so inference attempts stay
aligned with their processed frame. It writes line-flushed metadata and one
record for every classifier call, including warm-up and low-PMM results. Native
`uav`/`other` results are preserved while evaluation maps them to the run-level
`drone`/`not_drone` truth vocabulary. A decrease in the 36-frame history is
recorded as a `tracking_history_restarted` reset.

Orderly shutdown appends run and aggregate summaries. Aggregation accepts only
completed labeled logs with matching format version, classifier/runtime
contract, deployed artifact hashes, profile hash and fingerprints, and PMM
configuration. This compatibility key deliberately excludes Eli's version 1
logs. Inference lines are flushed immediately; a hard-killed run can lack
summaries and is excluded from aggregation without losing its complete lines.

## Offline LSTM training

`machinelearning/training.ipynb` implements a repository-native adaptation of mmHawkeye's UAV
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

Each identification spectrum is converted from dB to linear amplitude. The
body-peak bin is located before DC removal. Center-bin values are averaged only
over frames whose body peak is more than one bin from the center, and that
average is subtracted from every center-bin value with a zero floor. If no such
frame exists, the center bin is left unchanged to preserve a hovering target.
Spectra with an off-center body peak are then shifted to the center with linear
interpolation and zero fill. The aligned amplitude is compressed with `log1p`.

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
two output logits. A trained `model.onnx`, its external tensor-data sidecar when
declared, and `manifest.json` must be copied into the `model_weights/` directory
before live classification can be enabled.

`inference.py` uses `model_weights/generated/model.fp16.engine` with TensorRT.
When that file is absent, it invokes `trtexec` and streams the compiler output
to the terminal before loading the generated engine. There is no PyTorch or
ONNX Runtime inference fallback. TensorRT loading, compilation, or execution
errors are reported. The runtime validates the engine tensor interface, and
the manifest validator rejects missing metadata, unexpected
label order, architecture or input-shape changes, invalid normalization, and
mismatched profile/feature fingerprints.
It maintains the rolling PMM score window, returns `warming_up` until both the
Doppler-Time history and PMM-score window contain 36 entries, and returns
`unknown` if the maximum score is below the runtime threshold. Otherwise it
applies the notebook's dB-to-amplitude,
strongest-bin alignment, `log1p`, and stored normalization operations before
computing the two probabilities. Inference runs in the DSP worker rather than
the UDP receiver, and its latency is included in processing diagnostics.

Training, evaluation, and permanent artifact creation occur only when an
operator runs the notebook. Any committed notebook output is a record of a
prior interactive run and is not consumed by the live application.

## Process model

After `run.py` launches the capture subprocess, that subprocess's main process
receives UDP packets and assembles frames. A spawned worker process performs
all DSP, optional LSTM inference, raw/processed writing, and serialization; an
optional PyQtGraph display process consumes a latest-only queue. Range and
image modes use native PyQtGraph plot items; target and combined modes use
`pyqtgraph.opengl` for the 3D PMM position.
The processing queue is bounded and records drops rather than allowing
unbounded memory growth. The display queue is also bounded and may discard
superseded drawings without affecting processing.

## Verification

Run:

```bash
python -m unittest discover -s testcodes -p "test_*.py" -v
python main/startup.py \
  --config profiles/profile-mini4-20m.cfg \
  --sdk-profile profiles/profile-mini4-20m.cfg \
  --setup profiles/setup.json \
  --preflight-only --skip-socket-preflight
python scripts/benchmark_pmm.py
```

The deterministic suite covers folding, subtraction, path constraints,
particles, state transitions, ODS Capon geometry, profile rejection, frame
integrity, JSONL labels, replay, launcher prompts, model contract rejection,
rolling inference warm-up and gating, and classification output serialization.
