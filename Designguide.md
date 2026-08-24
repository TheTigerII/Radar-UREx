# Radar Capture and Processing Design

This document describes the code currently implemented for live raw ADC capture
from a TI IWR6843ISK-ODS through a DCA1000EVM. It is an implementation guide,
not a description of TI UART TLV output or mmWave Studio post-processing.

Range FFT, TDM separation, and Doppler FFT use local complex64 SciPy kernels in
`dsp_kernels.py`. Independent NumPy-reference tests verify their layout and
numerical results. Hann windows are cached, complex64 precision is preserved,
and no unused log-magnitude intermediate is created. OS-CFAR uses a vectorized
ordered-window implementation with cached training indices and scale factors.
Raw DCA1000 decoding and the IWR6843ISK-ODS-specific planar antenna mapping are
also implemented locally.
All live displays run with PyQtGraph/PySide6 in an isolated display process.
Range and spectrogram modes use `PlotItem`/`ImageItem`; point-cloud modes use
PyQtGraph's OpenGL view.

## Runtime Components

The integrated entry point is `main/run.py`. It launches two independent programs:

```text
main/run.py
  +-- livedatacapture.py   UDP receive, frame assembly, DSP, display, raw saving
  +-- startup.py           radar serial control and DCA1000 UDP control
```

`main/run.py` prompts for a duration before initialization (5 minutes by default;
zero means unlimited), starts the capture pipeline first, and waits until it
reports that the UDP receiver is listening before starting the hardware
controller. The readiness wait also covers initialization of the selected
display, frame processor, and optional rotor post-processor; its timeout is
extended for CUDA/TensorRT startup. When the deadline expires or Ctrl+C is
pressed, `main/run.py` stops `startup.py` first so `sensorStop` and DCA1000
`RECORD_STOP` are attempted before capture is closed.

The programs can also be run manually in two terminals. `startup.py` does not
embed the real capture receiver: its `--capture-backend` currently supports
only `dry-run`.

## `main/` Package File Reference

The `main` directory contains the complete runtime package. The files and their
boundaries are:

| File | Responsibility and important contracts |
| --- | --- |
| `main/__init__.py` | Marks `main` as the radar capture, processing, calibration, and inference package. It exports no names and performs no initialization. |
| `main/run.py` | Recommended operator-facing entry point. It gathers interactive choices when arguments are omitted, resolves the radar command UART, creates output paths, launches `livedatacapture.py` before `startup.py`, relays capture readiness, and shuts the hardware controller down before the data plane. It also owns the end-to-end range, azimuth, and elevation calibration workflow. Its `main()` returns a process exit code. |
| `main/startup.py` | Hardware control plane. It loads the radar and DCA1000 configuration, runs preflight checks, sends DCA1000 commands over UDP, sends SDK CLI commands over the radar UART, and enforces the arm-before-`sensorStart` sequence. Its internal capture backend is deliberately dry-run only. Its `main()` returns an exit code. |
| `main/livedatacapture.py` | Live data plane and largest orchestration module. It parses `.cfg`/JSON capture dimensions, receives and assembles DCA1000 UDP data, starts the DSP, optional rotor-classification, and GUI processes, tracks losses and drops, and writes raw, metadata, processed JSONL, and terminal-log output. It can be run directly, but does not configure or stop the hardware. |
| `main/dsp.py` | Hardware-independent numerical processing and state. It converts LVDS frame bytes, computes range/Doppler transforms, performs dynamic and static detection, maps the ODS virtual antenna, estimates XYZ, clusters points, maintains clutter/static reference maps, computes both micro-Doppler variants, and estimates rotor RPM. It has no command-line entry point. |
| `main/dsp_kernels.py` | Optimized local SciPy range FFT, TDM Doppler FFT, and vectorized OS-CFAR kernels. It caches Hann windows, CFAR indices, and scale factors. |
| `main/calibrate.py` | Calibration domain layer. It validates and generates temporary raw-LVDS profiles, accumulates range/channel or angular observations, applies stability gates, serializes calibration reports, atomically updates the operational profile with backups, and supplies the calibration GUI child process. It has no standalone CLI; `run.py` and `livedatacapture.py` integrate it. |
| `main/inference.py` | Shared CNN feature contract and CPU/PyTorch backend. It creates one `[2, 64]` feature step from a selected three-bin range gate, retains 48 steps, validates the checkpoint/calibration/model-card/profile fingerprint, normalizes the `[2, 48, 64]` window, and returns calibrated `drone`, `not_drone`, or `unknown` results. It has no CLI. |
| `main/tensorrt_inference.py` | CUDA/TensorRT FP16 backend and device selector. It owns pinned host/device buffers and one CUDA stream, builds or validates the cached engine against ONNX, artifact, profile, TensorRT, and GPU metadata, requires parity within 0.005 probability with no label changes, benchmarks the engine, and implements the same stateful inference interface as the CPU backend. `auto` selects CUDA on Jetson and CPU elsewhere; requested CUDA failures are fatal rather than silently falling back. |
| `main/classification_evaluation.py` | Optional live-inference evaluation logger. It streams versioned per-attempt JSONL, records the 48-step history lifecycle and reset metadata, computes run-level classification, confidence, duration, latency, stability, and recovery metrics, fingerprints compatible deployments, and aggregates completed labeled logs. It has no CLI and is instantiated by `livedatacapture.py` only when explicitly enabled. |

The primary module dependencies are:

```text
run.py
  +-- calibrate.py
  +-- subprocess: livedatacapture.py
  `-- subprocess: startup.py

startup.py
  `-- livedatacapture.py (configuration data types and parsers only)

livedatacapture.py
  +-- dsp.py
  +-- calibrate.py
  +-- inference.py
  +-- tensorrt_inference.py
  `-- classification_evaluation.py

dsp.py
  `-- dsp_kernels.py (local FFT and CFAR kernels)

tensorrt_inference.py
  `-- inference.py (feature, result, and artifact contracts)
```

`run.py`, `startup.py`, and `livedatacapture.py` support direct script
execution. The remaining files are library modules and should be imported
through the `main` package. Optional dependencies are loaded only on the paths
that need them: PySerial for direct radar control, PyQtGraph/PySide6 for live
displays, PyTorch/joblib for CPU classification, and TensorRT/CUDA Python for
CUDA classification. SciPy and NumPy are core DSP dependencies. All hot FFT and
CFAR paths are local and require no external radar-DSP package.

## Startup Control Plane

`main/startup.py` configures the radar and DCA1000 but does not receive raw ADC
packets. Use `main/run.py` for the integrated workflow or start
`livedatacapture.py` separately before starting the control plane.

### Responsibilities and backends

The control plane parses radar dimensions from a mmWave Studio JSON or SDK CLI
`.cfg`, parses board/DCA1000 settings from `setup.json`, validates the inputs,
configures both devices, defers `sensorStart` until recording is armed, and
attempts `sensorStop` and `RECORD_STOP` during shutdown. It does not flash
firmware. `--load-firmware` only requires the configured MSS/BSS paths and
reports them through the dry-run radar backend.

Radar backends:

- `dry-run` (default) reports the intended radar operations.
- `direct-serial` opens the SDK CLI UART with PySerial, sends active commands
  from `--sdk-profile` other than `sensorStart`, and sends the deferred start
  only after the DCA1000 is armed. The profile must contain `sensorStart`;
  blank lines and `%`/`#` comments are ignored. Responses complete on `Done`,
  `Error`, the `mmwDemo:/>` prompt, `Ignored`, or `Skipped`. An error response
  fails normal startup; shutdown permits an error from `sensorStop` so the
  remaining cleanup can continue.

DCA1000 backends:

- `dry-run` (default) reports the intended configuration and arm operations.
- `direct-udp` binds the local DCA1000 control port (4096 by default) and sends
  commands to `192.168.33.180:4096`. Each acknowledgement is validated with
  the configured timeout and retry count.

Direct UDP sends the following commands in order:

```text
SYSTEM_CONNECT
RESET_FPGA
CONFIG_FPGA_GEN
CONFIG_RECORD
RECORD_START
```

If record start succeeded, shutdown sends `RECORD_STOP`. The FPGA and record
payloads are derived from the capture/setup configuration; `setup.json` can
provide hexadecimal overrides under `directUdpDCA1000.payloads`.

The only `--capture-backend` is `dry-run`. It advances orchestration state and
reports the expected address and frame size, but never binds the data port,
receives data, saves frames, or drives a display. Running `startup.py` alone
therefore starts a streaming sensor without a real data consumer.

### Configuration selection and preflight

Standalone defaults are `profiles/mmwave.json`, `profiles/setup.json`, and
`profiles/profile.cfg` for `--config`, `--setup`, and `--sdk-profile`
respectively. If `--radar-backend direct-serial` is selected while `--config`
still equals its JSON default, the SDK profile is also used as the dimension
source. Normally the same `.cfg` should be supplied for both options, which is
what `run.py` does.

Preflight validates:

- positive ADC sample, RX, chirp, and frame-byte dimensions;
- exactly two LVDS lanes;
- DCA1000 capture-hardware selection and, when present, a packet delay from
  5 to 500 us;
- data and DCA1000 control ports;
- a radar control port and positive baud rate from the CLI overrides or
  `setup.json` (the current implementation requires these even for dry-run);
- an existing SDK profile containing `sensorStart` for direct serial;
- existing MSS/BSS firmware paths only when `--load-firmware` is selected; and
- the ability to bind `host_ip:data_port`, unless
  `--skip-socket-preflight` is used.

The socket check is a short probe and closes immediately. In a two-terminal
workflow the real receiver already owns the port, so the control plane must be
given `--skip-socket-preflight`.

### Orchestration and shutdown

The startup state machine is:

```text
1. load configs                 -> CONFIGS_LOADED
2. run preflight                -> PREFLIGHT_PASSED
3. configure DCA1000            -> DCA1000_READY
4. configure radar, defer start -> RADAR_READY
5. start dry-run receiver       -> RECEIVER_READY
6. arm DCA1000, readiness delay -> DCA1000_ARMED
7. send sensorStart             -> RADAR_STREAMING
```

`--preflight-only` stops after step 2 without configuring hardware. A normal
standalone run then waits for Ctrl+C. Every exit after startup has begun enters
`STOPPING` and attempts cleanup in this order:

```text
radar sensor stop and serial close
DCA1000 record stop and control-socket close
dry-run capture close
```

Cleanup continues after an individual failure, reports the accumulated errors,
and finishes in `STOPPED`.

In the integrated workflow, `run.py` starts `livedatacapture.py` first and
waits for the capture-ready line produced only after the UDP socket, frame
processor, requested GUI, and optional rotor post-processor are ready. The
normal timeout is 30 seconds and the CUDA/TensorRT path receives 300 seconds
for a first-run engine build and parity validation. `run.py` then invokes the
real serial and UDP backends with `--skip-socket-preflight`, using the same
`.cfg` for capture dimensions and SDK commands. On a deadline, Ctrl+C, or child
exit, it stops `startup.py` before `livedatacapture.py`.

The control plane currently has no firmware-flashing implementation, no real
capture backend, and only state-transition health reporting. Direct serial
assumes compatible SDK CLI firmware is already running. DCA1000 control also
depends on its firmware accepting the generated or overridden payloads;
packet/frame health is reported by `livedatacapture.py`.

## Capture Data Flow

```text
DCA1000 UDP packets (host 192.168.33.30, port 4098)
  -> main capture process
       dedicated receiver thread drains the UDP socket
       bounded in-process packet queue absorbs short host stalls
       parse optional 10-byte sequence/byte-count header
       track packet order and loss
       assemble a continuous byte stream
       split stream into configured frame sizes
       reject frames containing ordinary byte gaps
       enqueue valid frames in a bounded processing queue
  -> RadarFrameProcessor process
       optionally save raw frame
       reshape int16 LVDS data to [chirp, rx, sample]
       range FFT, selected display DSP, and compact CNN feature extraction
       enqueue latest display result
       enqueue ordered dedicated-rotor results
  -> RotorPostProcessor process (dedicated rotor classification)
       run the fixed-shape CNN with PyTorch/CPU or TensorRT FP16/CUDA
       serialize processed JSONL in frame order
  -> RadarLiveDisplay process (when enabled)
       update persistent PyQtGraph curve, image, or OpenGL scatter items
```

Capture, processing, rotor post-processing, and display use bounded queues. The UDP receiver thread
never waits for frame assembly, FFT, logging, or plotting. A full packet queue
increments `receiver_queue_drops`; a full processing queue increments
`processing_frames_dropped`; the one-item display queue discards stale display
results in favor of the newest one.

### Processes and queue sizes

- UDP receiver thread: socket receive only.
- Main process: packet dequeue and `FrameBuffer` assembly.
- `RadarFrameProcessor`: frame conversion, DSP, logging, and optional raw write.
- `RotorPostProcessor`: spawned CNN inference and JSON serialization for
  classified dedicated-rotor capture. `auto` uses TensorRT FP16/CUDA on Jetson
  and PyTorch/CPU elsewhere; `--classification-device` can select the backend.
- `RadarLiveDisplay`: PyQtGraph UI for every display mode.
- Packet queue: `--packet-queue-size`, default 8,192 datagrams.
- Processing queue: `--processing-queue-size`, default 32 frames.
- Rotor post-processing queue: the same 32-frame capacity by default.
- Display payload queue: one result.
- Processor log queue: 1,000 messages.

Routine successful frames and display latency are silent. Capture statistics
are emitted immediately whenever an error counter changes. Shutdown emits
capture, processing, and display summaries, including the number of updates
actually rendered by the GUI. The processor ignores the parent process's
SIGINT, consumes the queue sentinel, and drains all frames queued before that
sentinel. Final DSP and post-processing reports include aggregate p50, p95,
and maximum timings for range FFT, Doppler, dynamic detection/CFAR, static
detection, clustering, micro-Doppler, classification feature extraction,
classification, serialization, and each process's total.

## Configuration

`livedatacapture.py` accepts either an SDK CLI `.cfg` or mmWave Studio `.json`
through `--config`. It derives:

```text
num_adc_samples
num_rx_channels
num_chirps_per_frame
num_loops
num_chirps_per_loop
tx_channel_masks (from .cfg when available)
iq_swap
channel_interleave
lvds_lanes
sample_rate_ksps
frequency_slope_mhz_per_us
start_frequency_ghz
idle_time_us
ramp_end_time_us
frame_periodicity_ms
range_bias_m
rx_channel_compensation
azimuth_bias_deg
elevation_bias_deg
bytes_per_frame
```

The frame-size calculation is:

```text
num_chirps_per_frame = num_loops * chirps_per_loop
bytes_per_frame = num_adc_samples * num_rx_channels
                  * num_chirps_per_frame * 4
```

The factor 4 is two bytes for I plus two bytes for Q. The current reshape code
supports complex, 16-bit, two-lane LVDS data.

`setup.json` supplies DCA1000 settings including `packetSequenceEnable` and
`packetDelay_us`. The current 50-us delay remains comfortably above the data
rate required by the 33.33-ms radar profile while smoothing Ethernet bursts.
The supported hardware range is validated as 5 to 500 us. When packet sequence
headers are disabled, the receiver uses synthetic sequence and byte counts; it
can assemble received bytes but cannot detect network loss from DCA1000
metadata.

## Packet and Frame Integrity

With sequence headers enabled, `SequenceTracker` records lost, duplicate, and
out-of-order packet numbers. `FrameBuffer` uses the DCA1000 byte count to detect
gaps and overlaps.

- A small forward gap is zero-filled only to preserve frame alignment. Any
  frame touched by those bytes is marked invalid and is not processed or saved.
- A forward discontinuity larger than one complete frame is treated as an
  implausible stream jump. The partial assembly is discarded and the buffer is
  resynchronized at the current packet. This bounds memory use and increments
  `stream_resyncs`.
- Fully duplicate payloads are ignored.
- Overlapping payload prefixes are trimmed.
- Backward byte counts are treated as out of order and ignored.

The buffer can return more than one frame for one packet, although normal
DCA1000 payloads are much smaller than a frame.

## Raw Data Conversion

`dsp.frame_bytes_to_radar_cube()` reads little-endian `int16` groups of four.
For normal IQ order, the first two values provide I samples and the next two Q
samples; `iq_swap` reverses those roles. The resulting complex stream is
reshaped to:

```text
[num_chirps_per_frame, num_rx_channels, num_adc_samples]
```

Both interleaved and non-interleaved channel layouts are handled after the
two-lane IQ conversion. Configuring any LVDS lane count other than two raises
`NotImplementedError` during conversion and is rejected by startup preflight.

## Calibration Workflow

`main/run.py` implements three calibration modes: `calibration` for range bias and
12 physical TX/RX channel coefficients, plus `azimuth-calibration` and
`elevation-calibration` for host-side angular offsets. All require four RX
channels and one chirp for each physical TX1, TX2, and TX3. The default source
is `profiles/profile_calibration.cfg`; the operational profile to update is
the normal `--config`, which defaults to `profiles/profile.cfg`.

The launcher never programs the source calibration file directly. It creates a
temporary runtime profile that disables UART GUI output and firmware-side range
measurement, enables raw LVDS streaming, and inserts the requested reflector
distance and search window. The capture process ignores 16 frames by default,
then requires 64 accepted frames within a 90-second timeout.

Range/channel calibration averages the zero-Doppler response across loops,
orders channels by physical TX number, finds the reflector within the requested
range window, and uses three-point interpolation for sub-bin range bias. It
accepts a result only after range, phase, and magnitude stability limits pass,
then emits a TI `compRangeBiasAndRxChanPhase` command. Angular calibration first
imports the operational profile's range/channel compensation into host DSP,
forms a 128-by-128 angle FFT at the reflector bin, and records measured minus
known angle as the selected axis bias.

Hardware is stopped before the result is offered to the operator. Every result
is written to a JSON report. Applying it is an explicit `y/N` confirmation:
range calibration atomically replaces the profile's single
`compRangeBiasAndRxChanPhase` line, while angular calibration updates the SDK-
safe `% hostAngleCalibration ...` comment. Both paths create a timestamped
profile backup first. Normal capture parses these corrections: range bias
shifts the physical range axis, channel coefficients are applied in the ODS
virtual array, and azimuth/elevation offsets correct the reported directions.

## Implemented DSP

The DSP in `dsp.py` is intended for live visualization. It applies configured
range/channel and angular corrections, but it is not a precision metrology or
validated multi-target tracking system.

### Range profile

1. Apply a Hann window across ADC samples.
2. Run an FFT across the sample axis.
3. Average absolute FFT magnitude over chirps and receivers.

If sample rate and frequency slope are present, the range-bin spacing is:

```text
c * sample_rate / (2 * slope * num_adc_samples)
```

### Range-Doppler

The range FFT is reshaped to `[loop, chirp-within-loop, rx, range]`. Doppler FFT
and `fftshift` are applied over loops. The displayed heatmap averages magnitude
over chirp-within-loop and RX, then converts it to dB. Its Y-axis is a Doppler
bin index, not velocity in m/s.

### Point cloud

```text
range FFT [chirp, rx, range]
  -> TDM-aware Doppler FFT [doppler, tx, rx, range]
  -> mean TX/RX power and adaptive clutter normalization
  -> 2D OS-CFAR, Doppler-local peaks, and range/FOV gates
  -> batched 32-by-32 planar-array angle FFT
  -> XYZ points and DBSCAN cluster centers
  -> one constant-velocity, nearest-neighbor target track
```

The point-cloud path:

1. Forms mean range-Doppler power.
2. Applies a power-domain clutter map. During the initial warm-up it
   learns every cell using an exponential moving average and emits no point
   detections. It then divides every cell by its learned background power before
   CFAR and applies a default 3 dB minimum target-to-background ratio. This
   keeps the normalized background near one instead of producing large regions
   of zero-valued CFAR training cells. The map remains fixed after warm-up so
   later targets and scene changes are not absorbed. An FFT-shape change resets
   the map and starts a new warm-up.
3. Runs two-dimensional OS-CFAR with Doppler wrapping and a default requested
   false-alarm probability of `1e-3` per axis.
4. Keeps Doppler-local peaks without suppressing adjacent range detections, then
   removes detections below 0.25 m.
5. Rejects detections beyond 10 m.
6. Maps all candidate cells into virtual antenna grids and estimates their
   directions with one batched 32-by-32 2D FFT. Unlimited point-cloud output
   preserves detection order and skips the unnecessary power sort.
7. Applies ±60-degree azimuth and elevation gates and keeps all in-FOV
   detections.
8. Runs deterministic local DBSCAN for normal-size clouds, with a scikit-learn
   fallback for unusually large clouds, and sends both the original points and
   cluster centers to the display process.
9. Updates one persistent dynamic 3D target track. Initial acquisition uses the
   strongest cluster or point; later updates use gated nearest-neighbor
   association against a constant-velocity prediction.

In parallel, the static-change path sums angle-FFT power from the centered
Doppler bin and its two neighbors. It applies the virtual-array mapping and one
vectorized 32-by-32 angle FFT across every valid range bin on every processed
point-cloud update, producing a `[range, elevation, azimuth]` cube. SciPy's FFT
keeps the complex64 input precision instead of promoting it to complex128.
Power is summed across the three Doppler bins before the smaller float32 power
cube is shifted, reducing allocation and memory-copy cost without changing the
FFT dimensions or update rate.

The first 30 processed detection updates are discarded as warm-up. The next
150 updates build a median reference, a per-range power floor, and a robust
per-cell noise estimate from the median absolute deviation in log power. The
implementation does not determine whether those calibration updates are
target-free, so the operator must keep the target absent until calibration is
complete. Each live change map is corrected by its per-range median to remove
common receiver-gain drift. The corrected instantaneous change is thresholded
without a temporal EMA. A detection must exceed twice the learned cell
variation. Direct `livedatacapture.py` use defaults the additional absolute
floor to 0 dB; the recommended `run.py` launcher passes 3 dB by default. The
reference and noise estimates remain fixed after calibration. Range/FOV gating
occurs before local-maximum testing; when the threshold produces no candidates,
the full 3D maximum scan is skipped. The remaining cells are capped at the 256
strongest before XYZ conversion and DBSCAN. Returned raw candidates are
`[x, y, z, magnitude_db, change_db]`.

Raw static candidates are diagnostic activity, not targets. The static tracker
receives DBSCAN clusters with a default minimum of one point because the 3D
local-maximum pass has already reduced a reflector to one spatial candidate.
A cluster can start a static track only after a confirmed measured dynamic
track has armed handoff; no additional displacement test is required. Handoff
does not open while dynamic measurements remain healthy, and measured dynamic
tracking resets static acquisition. After two consecutive missing dynamic
measurements, clusters within 0.4 m of the last dynamic position become
eligible. The prior dynamic confirmation plus this spatial gate establishes
identity, so the closest qualifying cluster confirms the static track on its
first return. Deployments can restore stricter same-frame density with
`--static-cluster-min-samples 3`. Handoff remains eligible for 60 frames. The
last measured dynamic position remains as an explicit predicted static marker
during this pending interval, including after the dynamic tracker expires; it
is not serialized or displayed as a measured point. The
selected target's range, azimuth, and elevation cells are protected by ±2 bins
while validated. The static tracker uses a zero-velocity model with stronger
position smoothing, and both motion-only protection and the static track
release after 60 consecutive candidate misses. If a nonzero
`--static-background-update-rate` enabled adaptation, released objects can then
be absorbed into the background; the default zero rate keeps the startup
reference fixed. Only the exact DBSCAN members of the validated target are
displayed and saved. All suppressed raw activity is reported separately as
`static_candidate_count`.

For four RX channels and TX masks corresponding to TX1-TX3, the virtual grid
uses the IWR6843ISK-ODS antenna layout and applies a sign inversion to RX2 and
RX3. Other layouts fall back to a generic grid. Returned coordinates use
X=left/right, Y=forward, and Z=elevation. Configured range/channel and fixed
azimuth/elevation corrections are applied for the supported ODS path, but the
result remains a diagnostic angle-FFT estimate.

### Combined point cloud and micro-Doppler

```text
selected 3D track range
  -> nearest range bin plus/minus two bins
  -> [loop, TX slot, RX, gated range]
  -> independent 64-loop Hann STFTs with a 32-loop hop
  -> sum power after the 128-point centered Doppler FFTs
  -> append the frame's three spectra to a 150-spectrum display history
```

The `point-cloud-micro-doppler` display computes one Doppler cube for each
processed update and reuses it for the dynamic point cloud and static angle
cube. The selected track range then gates the original range-FFT cube for the
per-TX STFT. A measured confirmed dynamic track has priority while the target
is moving. A one-update dynamic prediction covers the initial detection miss;
after the handoff opens, a predicted static anchor holds the last measured
position until the first qualified static return takes over. Predicted markers
are shown smaller and translucent. No static-only
clutter or arbitrary range fallback can activate micro-Doppler.

Classification continues extracting a feature step at the selected track's
predicted range during these association-preserving prediction updates. This
keeps the 48-frame input cadence intact across ordinary detection misses;
track-related history resets occur only when the selected target is lost or
the gap-aware micro-Doppler ownership check reports a distant reacquisition.
Classification does not apply a second 3D-distance check during uninterrupted
tracking because angle estimates can jump at dynamic-to-static handoffs while
the selected range and target identity remain continuous.

The micro-Doppler branch reshapes the chronological chirps into explicit loop
and TX-slot axes. It applies independent slow-time FFTs to every TX slot, RX
channel, and gated range bin using 64-loop Hann windows, a 32-loop hop, and a
128-point FFT. TX, RX, and range-bin powers are summed only after the FFT, so
the three TX signals are not coherently merged and require no inter-TX phase
calibration. A 128-loop frame produces three short-time spectra. Each spectrum
is appended to a 150-window history in the frame-processing process. Sending
the complete history through the latest-only display queue prevents GUI queue
replacement from creating holes in the visible spectrogram. The spectrogram
follows the selected target by spatial continuity rather than the dynamic or
static source label. History remains frozen through gaps of up to 30 processed
updates and continues during uninterrupted motion regardless of displacement.
A dynamic-to-static handoff also retains history. After a selection gap, a
reacquired target beyond 0.75 m or a gap longer than 30 updates starts fresh
history. An explicit static gate retains the centered zero-Doppler bin. The
PyQtGraph image uses a visible-spectrum `turbo` color map, running from dark
blue at the fixed 60 dB minimum to red at the fixed 120 dB maximum rather than
rescaling each history update. The point cloud uses the same fixed lookup
table, and the spectrogram has a fixed magnitude colorbar. Status labels report
measured updates per second. A Qt timer drains the latest-only queue and updates
persistent image and OpenGL scatter items without rebuilding axes or layouts.
The combined display renders micro-Doppler and the 3D collections for every
consumed payload.

### Dedicated rotor micro-Doppler

The `micro-doppler` path bypasses Doppler-CFAR, angle estimation, clustering,
and static-scene tracking. It uses a fixed physical range gate with the
known-good `profile.cfg` LVDS waveform by default. Per-TX/RX/range-channel
Hann-weighted complex mean removal nulls the stationary body component before
a 16-loop, 2-loop-hop STFT. With the compatible three-TX profile this gives an
approximately 1.87 ms window and 0.234 ms hop, short enough to separate the
2.80 ms blade passages of a two-blade rotor at 10,700 RPM. Raw window power
and the clutter-rejected result
are both retained; the enhanced view subtracts a robust off-centre noise floor
and uses a fixed 0-to-30 dB relative scale. Per-window noise spread is estimated
as `1.4826 × MAD` from the lower half below the median, preventing positive
blade ridges from biasing their own threshold. The gate is the greater of 3 dB
or three robust standard deviations, capped at 15 dB so deep deterministic
FFT/cancellation nulls cannot blank the whole display. A 3×3 Doppler/time
support filter requires at least three candidate cells. Rejected cells become
zero while retained relative-dB values are not attenuated. Flash scoring and
RPM estimation consume this filtered spectrum.
The default estimator model is two blades with a 500-to-10,700 RPM search band,
matching the current drone. Dedicated rotor processing uses the same verified
`profile.cfg` acquisition waveform as the other live modes.

When classification is enabled, the DSP worker sends only the ordered frame
index, rotor result, and 2-by-64 feature step to a spawned post-processor.
CNN inference and JSON serialization therefore run concurrently with the next
frame's radar DSP without changing STFT or classification cadence. With the
CUDA backend, the post-processor owns the TensorRT execution context and
stream, plus persistent host/device buffers. With the CPU backend, it owns the
PyTorch inference engine instead. The `auto` device selects CUDA on Jetson and
CPU elsewhere. The bounded handoff applies backpressure and never silently
discards a post-processing item.

Window centres combine packet-derived frame time with configured within-frame
chirp timing. RPM estimation retains two seconds of those physical timestamps.
For display only, measured STFT centres are concatenated at their nominal hop,
removing frame blind time, invalid frames, and processing drops. The newest
0.20 seconds of active acquisition is max-pooled onto a bounded 512-column time
grid. Processed output, capture-gap diagnostics, and gap-aware RPM estimation
retain the true sampling intervals. Rotor mode
updates a persistent PyQtGraph `ImageItem` with C-contiguous row-major
`uint8` RGBA data and a fixed 256-entry Turbo lookup table. Quantization and
the 0-to-30 dB clamp are display-only; processed output retains floating-point
relative-dB values without that upper clamp. A Qt timer polls at up to 60 Hz
and drains the one-item queue to the newest payload.
The velocity extent and status layout update only when they change. The clutter
notch is a static overlay, so the uploaded image remains finite. Raw spectra
and noise floors remain in processed output but are omitted from the
cross-process GUI payload.

On Linux/X11, startup requires `libxcb-cursor0`. A workspace-local copy in the
virtual environment is preloaded when present. Dependency validation rejects a
missing library before capture, and the child rejects Qt's `offscreen` or
`minimal` platform fallback so invisible renders cannot inflate redraw
coverage. The visible window is centered, raised, and activated at startup.
The Qt application is created with a sanitized argument list because Qt/X11
also defines `--display`; passing the radar CLI's
`--display micro-doppler` through to `QApplication` would otherwise make Qt
look for an X server named `micro-doppler`.
The display child sends a bounded startup status message to the parent after
the window has been shown and processed at least once. Capture starts only
after this acknowledgement; import failures, non-visible Qt platforms, and
startup timeouts become `CaptureStartupError` messages.
Velocity bins use the configured start frequency and same-TX chirp interval. An
irregular-time Lomb-Scargle periodogram runs at a limited cadence over the
two-second blade-flash history. Search bounds come from the configured RPM
range and blade count; separated peaks become per-rotor RPM estimates.

### Classification backends

`inference.py` defines the feature and result contract shared by both inference
backends. For each update, the selected range bin and its immediate neighbors
form a three-bin target gate. Separated bins on both sides form a background
gate. Target and background powers are averaged over TX, RX, and range, and the
centered Doppler axis is power-averaged to 64 bins. The two feature channels
are `log1p(target_power)` and the log target-to-background difference. A
48-update history produces the model input with shape `[2, 48, 64]`.

The CPU `DroneBirdInference` backend requires PyTorch 2.6 or newer but earlier
than 3.0, joblib, and the following files in the artifact directory:

```text
model_state.pt       CNN architecture and state dictionary
calibration.joblib   clipping, normalization, probability calibration, threshold
manifest.json        model card and compatibility metadata
```

Initialization rejects a mismatched feature version, architecture, input
shape, three-bin gate contract, normalized profile fingerprint, radar
dimensions, or TX schedule. The deployed model specifically requires 64 ADC
samples, four RX channels, 384 chirps, 128 loops, three chirps per loop, and TX
masks `(1, 4, 2)`. Invalid feature steps and inference errors clear the history
and return `unknown`; a complete valid history produces a calibrated drone
probability and thresholded label.

`tensorrt_inference.py` implements the same stateful interface with TensorRT
FP16 and float32 model I/O. In addition to the shared calibration and manifest,
it requires `model.onnx` and `parity.npz`. It imports the Jetson system
TensorRT package when necessary, identifies the CUDA device, and uses pinned
host buffers, device buffers, and one CUDA stream. A missing engine is built
under `<artifact-directory>/generated/`; an existing engine must match its
hashes, TensorRT version, GPU compatibility fields, and parity record. Stale
metadata can be refreshed only after the existing engine passes parity. Parity permits
at most 0.005 absolute probability error and no changed labels. A cached engine
that cannot load or fails parity is rejected with instructions to remove it so
the next run can rebuild it. Initialization also records 20 warm-up and 200
measured inference runs with p50, p95, and maximum latency.

Device `auto` resolves to TensorRT/CUDA on Jetson and PyTorch/CPU elsewhere.
There is no silent CUDA-to-CPU fallback. In dedicated rotor mode, enabled
classification and JSON serialization run in `RotorPostProcessor`; in other
classified modes the frame processor owns the selected backend.

## Optional Live Inference Evaluation

Inference evaluation is independent of the version-5 processed-data writer and
is disabled by default. The integrated launcher prompts `Enable live inference
evaluation log? [y/N]:` only after classification has been enabled. Direct
capture remains non-interactive. `--inference-logging` selects a timestamped
`log/live_inference_*.jsonl`; `--inference-log PATH` selects a path and also
enables the feature. `--evaluation-label` supplies the constant run truth as
`drone`, `not_drone`, or `unlabeled`. Logging without classification, or an
explicitly disabled log combined with a path, is rejected.

`ClassificationEvaluationLogger` is created in the process that owns
inference: `RadarFrameProcessor` for normal/deferred modes or
`RotorPostProcessor` for dedicated rotor mode. It receives results directly,
not through the one-second-throttled terminal relay. Each valid processing
attempt therefore produces one flushed JSON line with frame/capture identity,
probability, predicted confidence, correctness, inference latency, target
context, and the complete `InferenceResult`.

The nested `history` object records append versus reset, empty/warming/ready
state, valid steps before and after the operation, reset reason, discarded
steps, reset sequence, history generation, and attempts since reset. Every
reset request increments its sequence. A generation changes only when history
was actually discarded or target ownership changed, so repeated no-target
resets remain observable without claiming that frames were lost. Capture and
processing timestamps are both retained; class durations use the configured
frame period, and resets, unknown outcomes, and frame-index gaps break temporal
segments.

On orderly worker shutdown the logger appends a run summary and then an
aggregate summary. Ready labeled decisions supply confusion-matrix accuracy,
per-class recall/precision/F1, balanced accuracy, Brier score, and log loss;
coverage and operational correctness separately account for warm-up and
unknown outcomes. Reset totals, reasons, discarded steps, recovery latency,
confidence/latency distributions, class durations, transitions, and session
majority are also recorded. Aggregate input is limited to completed labeled
logs with the same format, artifact hashes, profile hash, threshold, backend,
and precision. AUROC and PR-AUC are emitted only after both truth classes are
available. Missing-support metrics are `null` with an explanation. Because
each line is flushed, an interrupted file retains its complete observations,
but without a run summary it is excluded from later aggregation.

## Processed and Raw Recording

`--processed-output` streams version-5 newline-delimited JSON. Its first record
describes the radar configuration, data axes, and static-reference settings.
Each subsequent update contains the dynamic and static point clouds, their
DBSCAN clusters, static-reference and validation state, suppressed
`static_candidate_count`, target source and track, current micro-Doppler
spectrum, every short-time window generated from the frame, and the selected
range gate. `static_points` and `static_clusters` contain validated target data
only. Existing dynamic `points` and `clusters` fields keep their version-1
layouts. The writer does not duplicate the rolling display history. When it
is enabled, combined point-cloud/micro-Doppler processing runs even with a
different display mode or `--display none`.

In dedicated rotor mode, `rotor_micro_doppler` contains per-frame raw/enhanced
windows, physical axes, per-window noise floors and adaptive gates, flash
scores, RPM estimates, and alias diagnostics.
Spectra are rounded to 0.01 dB for JSON encoding. The legacy point-cloud and
micro-Doppler fields remain present for readers that have not adopted the
structured rotor result, but dedicated mode places only the newest raw window
in `micro_doppler_windows_db`; its complete window set is already stored in
the structured result. A 1 MiB output buffer amortizes filesystem writes.

The clutter-rejected FFT is derived from the raw window FFT by subtracting the
weighted complex mean multiplied by the cached Hann-window spectrum. By FFT
linearity this is equivalent to a second transform of the mean-cancelled
samples, while avoiding that duplicate batched FFT.

`--raw-output` writes only complete valid frames, consecutively and without
DCA1000 headers. At normal shutdown a JSON sidecar records dimensions, sample
format, processing parameters, counts, and paths. An explicit `--raw-metadata`
overrides the sidecar name. Integrated `main/run.py` enables processed output by
default and leaves raw recording disabled unless `--raw-output` is supplied.

Terminal messages are appended to `livedatacapture.log` by default. Raw files
and logs have no rotation or size limit, so long deployments must monitor disk
space externally.

## Automated Test Case Reference

The test suite contains 233 cases in `testcodes/`. Run all of them from the
repository root with `QT_QPA_PLATFORM=offscreen python -m pytest -q`. The
following inventory documents every collected case by file and test class.

### `test_calibrate.py` (15 cases)

`CalibrationProfileTests`:

- `test_discovered_profile_dimensions_and_timing` — verifies calibration discovers the profile's dimensions and frame timing.
- `test_peak_prominence_gate_is_disabled_by_default` — verifies the optional range-peak prominence gate defaults to disabled.
- `test_runtime_profile_changes_only_required_commands` — ensures runtime calibration rewrites only the required profile commands.
- `test_angular_runtime_keeps_firmware_neutral_and_imports_host_compensation` — ensures angular calibration leaves firmware commands neutral while loading host-side corrections.
- `test_duplicate_required_command_is_rejected` — rejects profiles containing duplicate commands that calibration must edit uniquely.
- `test_disabled_runtime_lvds_is_rejected` — rejects calibration profiles without enabled runtime LVDS output.
- `test_missing_antennas_and_invalid_target_window_are_rejected` — rejects incomplete antenna schedules and invalid target search windows.

`CalibrationAlgorithmTests`:

- `test_recovers_bias_equalization_and_physical_tx_order` — recovers synthetic range bias, channel equalization, and physical TX ordering.
- `test_normal_dsp_uses_measured_coefficients_by_physical_tx` — confirms normal DSP applies measured coefficients to the matching physical transmitter.
- `test_azimuth_calibration_recovers_known_offset` — recovers a known synthetic azimuth correction.
- `test_elevation_calibration_uses_positive_up_angle_convention` — verifies positive elevation means upward under the project coordinate convention.

`CalibrationResultSerializationTests`:

- `test_command_uses_ti_fixed_decimal_format_without_scientific_notation` — serializes TI compensation values in fixed-decimal CLI syntax.

`CalibrationApplyTests`:

- `test_apply_creates_backup_and_replaces_one_line` — creates a backup and replaces exactly one target profile line.
- `test_applied_line_drives_normal_range_and_angle_processing` — proves an applied calibration is consumed by normal range and angle processing.
- `test_angular_apply_preserves_other_axis_and_is_parsed_by_normal_dsp` — updates one angular axis without losing the other and confirms normal parsing.

### `test_dsp_kernels.py` (55 cases)

`AdaptiveClutterMapTests`:

- `test_default_minimum_snr_is_three_db` — verifies the default normalized detection floor is 3 dB.
- `test_normalizes_background_after_warmup_and_preserves_new_target` — learns background power and retains a subsequently introduced target.
- `test_detection_protection_prevents_target_absorption` — prevents protected detections from entering the adaptive clutter map.
- `test_map_is_frozen_after_warmup_by_default` — confirms default clutter adaptation stops after warm-up.
- `test_shape_change_resets_warmup` — restarts clutter learning when the input shape changes.

`StaticSceneMapTests`:

- `test_async_reference_finalization_does_not_block_frame_processing` — finalizes the static reference asynchronously without stalling frame processing.
- `test_default_threshold_uses_twice_learned_noise_without_db_floor` — checks the default learned-noise multiplier and absence of an unintended dB floor.
- `test_reference_is_frozen_after_calibration_by_default` — confirms the calibrated static reference is immutable by default.
- `test_frozen_reference_reports_new_static_change_immediately` — detects a new static change against the frozen reference without adaptation delay.
- `test_common_gain_drift_and_weak_reference_cells_are_suppressed` — suppresses common gain changes and unreliable low-power reference cells.
- `test_calibration_variability_raises_per_cell_detection_limit` — raises thresholds for cells that were noisy during calibration.
- `test_default_learned_noise_threshold_uses_two_times_noise` — verifies the default per-cell threshold is twice learned noise variation.
- `test_target_present_during_calibration_is_part_of_reference` — treats objects present during calibration as background.
- `test_warmup_frames_are_discarded_before_calibration` — excludes warm-up samples from reference construction.
- `test_adaptive_background_absorbs_unprotected_drift` — gradually absorbs persistent unprotected background drift.
- `test_protected_background_change_remains_detectable` — keeps a protected changed cell out of background adaptation.
- `test_released_target_is_eventually_absorbed` — allows a formerly protected target to enter the background after release.
- `test_shape_change_restarts_static_warmup_and_calibration` — resets both static warm-up and calibration for a new cube shape.
- `test_static_target_protection_mask_uses_range_and_angle_cells` — protects the configured neighborhood in range, azimuth, and elevation.
- `test_static_angle_power_integrates_centered_doppler_neighbors` — sums power from the centered Doppler bin and its neighbors.
- `test_static_angle_power_matches_full_precision_32_point_fft` — compares optimized static-angle power with a full-precision 32-point reference FFT.
- `test_candidate_local_maxima_matches_full_cube_filter` — matches candidate-only local maxima with full-cube filtering.
- `test_static_points_separate_changes_at_same_range_by_angle` — resolves static changes at one range into distinct angular points.
- `test_point_detection_uses_normalized_power_and_updates_map` — uses clutter-normalized power for detections and updates the map afterward.
- `test_minimum_snr_gate_rejects_normalized_background` — rejects normalized background cells below the minimum SNR.

`BatchedAngleFftTests`:

- `test_batched_virtual_grids_match_single_cell_mapping` — makes batched virtual-array construction match per-cell mapping.
- `test_zero_virtual_arrays_point_forward` — maps an all-zero virtual array to the defined forward direction.

`PointCloudCandidateOrderingTests`:

- `test_unlimited_points_bypass_power_sorting` — preserves detection order when no point cap is requested.
- `test_finite_point_cap_keeps_strongest_first` — retains the strongest detections when a finite cap is applied.

`DopplerPeakMaskTests`:

- `test_preserves_adjacent_range_peaks` — does not suppress peaks merely because they occupy adjacent range bins.
- `test_rejects_weaker_cyclic_doppler_neighbor` — removes a cell weaker than a cyclic neighboring Doppler bin.

`MicroDopplerSpectrogramTests`:

- `test_per_tx_stft_uses_64_loop_window_and_32_loop_hop` — verifies the combined-mode per-TX STFT window and hop contract.

`RotorMicroDopplerTests`:

- `test_single_tx_profile_has_expected_velocity_span` — derives the expected velocity extent for a single-TX profile.
- `test_three_tx_profile_processes_each_tx_at_its_physical_slow_time_rate` — uses the physical same-TX sampling interval in a three-TX schedule.
- `test_weighted_mean_rejects_static_body_and_preserves_tone` — cancels stationary body energy while preserving a moving tone.
- `test_deep_cancellation_nulls_cannot_blank_visible_ridges` — caps the adaptive gate so cancellation nulls cannot erase valid ridges.
- `test_adaptive_gate_blanks_complex_noise` — suppresses complex noise with the adaptive relative-power gate.
- `test_lower_tail_noise_gate_ignores_positive_blade_ridge` — keeps positive blade returns from biasing the noise estimate.
- `test_support_filter_preserves_ridge_peak_and_blanks_isolated_cell` — retains supported ridges and rejects isolated time-frequency cells.
- `test_gap_aware_rpm_estimate_is_within_five_percent` — estimates RPM within five percent despite sampling gaps.
- `test_gap_aware_rpm_estimate_includes_upper_search_boundary` — permits a valid solution at the maximum configured RPM.
- `test_tip_speed_alias_warning_uses_eighty_percent_margin` — raises the alias diagnostic at the 80-percent velocity margin.

`FrameDecodingTests`:

- `test_decodes_two_lane_iq_directly_to_complex64` — decodes two-lane LVDS IQ samples directly into complex64 values.
- `test_channel_interleave_transposes_sample_and_receiver_axes` — reconstructs channel-interleaved data with correct sample/RX axes.

`PointCloudClusteringTests`:

- `test_dbscan_returns_cluster_centers_and_ignores_noise` — returns DBSCAN centers while excluding noise-labelled points.
- `test_zero_radius_disables_clustering` — treats a zero neighborhood radius as clustering disabled.
- `test_small_cloud_dbscan_matches_sklearn` — makes the local small-cloud DBSCAN path agree with scikit-learn.
- `test_cluster_labels_identify_exact_returned_members` — associates returned centers with their exact point members.
- `test_static_sized_cloud_does_not_enter_sklearn_path` — keeps normal static clouds on the optimized local clustering path.

`OsCfarParameterTests`:

- `test_scale_matches_requested_false_alarm_rate` — validates the OS-CFAR scale against its requested false-alarm probability.
- `test_vectorized_windows_exclude_cut_and_guard_cells` — verifies vectorized training windows omit the CUT and guard cells.
- `test_vectorized_thresholds_support_doppler_axis` — applies the threshold kernel correctly along the Doppler axis.

`DspKernelTests`:

- `test_range_and_doppler_cube_layouts` — verifies range and explicit TX/RX Doppler output shapes.
- `test_optimized_ffts_match_numpy_reference` — compares optimized complex64 FFTs with independent NumPy calculations.
- `test_os_cfar_detects_an_isolated_strong_cell` — detects an isolated high-power cell with two-dimensional OS-CFAR.

### `test_inference.py` (11 cases)

`FeatureExtractionTests`:

- `test_reduces_centered_doppler_by_power_averaging` — reduces centered Doppler bins using power rather than complex-amplitude averaging.
- `test_feature_step_matches_notebook_power_formula` — keeps live feature extraction identical to the training notebook formula.
- `test_rejects_edge_gate_and_non_finite_cube` — rejects incomplete range gates and non-finite Doppler data.
- `test_profile_hash_is_independent_of_crlf` — normalizes line endings before computing the profile compatibility hash.

`StatefulInferenceTests`:

- `test_waits_for_48_steps_then_returns_drone` — returns warm-up results until a complete 48-step window can classify a drone.
- `test_probability_below_threshold_is_not_drone` — labels calibrated probabilities below threshold as non-drone.
- `test_precomputed_feature_steps_match_cube_update_history` — makes precomputed feature input agree with live cube updates.
- `test_invalid_shape_resets_accumulated_history` — clears state after receiving an invalid feature shape.

`ArtifactContractTests`:

- `test_accepts_current_profile_and_artifact_contract` — accepts the deployed model bundle with the current radar profile.
- `test_rejects_incompatible_profile_fingerprint` — rejects artifacts trained for a different normalized profile.
- `test_rejects_old_two_bin_target_gate_contract` — rejects obsolete artifacts using the former two-bin target gate.

### `test_classification_evaluation.py` (4 cases)

`MetricTests`:

- `test_binary_metrics_and_durations_are_calculated` — verifies confusion-matrix, per-class accuracy, and nominal class-duration calculations.
- `test_unknown_attempt_affects_coverage_not_ready_accuracy` — keeps ready-decision accuracy separate from unknown-outcome coverage and operational correctness.

`EvaluationLoggerTests`:

- `test_stream_contains_history_reset_metadata_and_summaries` — writes reset sequencing, history generations, confidence, latency, and both shutdown summary records.
- `test_second_compatible_run_aggregates_both_truth_classes` — combines compatible labeled runs and produces both class accuracies and AUROC.

### `test_livedatacapture.py` (91 cases)

`FrameBufferTests`:

- `test_packet_gap_within_one_frame_is_marked_invalid` — marks a partially assembled frame invalid when a packet gap occurs inside it.
- `test_gap_larger_than_one_frame_resynchronizes_without_padding` — resynchronizes after a large gap without fabricating padding bytes.
- `test_memoryview_payload_is_assembled_without_a_slice_copy` — accepts memoryview payloads without introducing a slice copy.

`UdpPacketReceiverTests`:

- `test_receiver_preserves_datagram_order` — enqueues received datagrams in arrival order.
- `test_receiver_counts_bounded_queue_drops` — records drops when the bounded packet queue is full.

`FrameDiagnosticsTests`:

- `test_valid_frame_does_not_emit_routine_diagnostics` — keeps successful frames silent during normal operation.
- `test_error_snapshot_ignores_success_counters` — excludes successful-work counters from the error-change snapshot.
- `test_errors_are_reported_immediately_on_each_counter_change` — emits diagnostics whenever an error counter changes.
- `test_capture_summary_separates_valid_and_queued_frames` — reports valid assembly separately from processor enqueue success.
- `test_graceful_processor_stop_does_not_discard_a_queued_frame` — drains work queued before the shutdown sentinel.
- `test_latest_payload_replacement_is_counted` — counts display payloads replaced by the latest-only queue.

`RotorDisplayPayloadSinkTests`:

- `test_dedicated_mode_accepts_proven_three_tx_profile_and_bypasses_point_cloud` — accepts the verified three-TX rotor profile without invoking point-cloud DSP.
- `test_rotor_frame_worker_initializes_with_three_tx_profile` — initializes the rotor worker using the three-transmitter contract.
- `test_rotor_frame_worker_processes_three_tx_frame` — produces a rotor result from a valid three-TX frame.

`RotorPostprocessorTests`:

- `test_postprocessor_preserves_frame_order_and_classification_alignment` — keeps asynchronous classification, evaluation logging, and serialization aligned with frame order.

`ProcessedOutputWriterTests`:

- `test_writes_metadata_point_cloud_and_micro_doppler_jsonl` — writes the metadata record and combined-mode point/micro-Doppler fields.
- `test_writes_structured_rotor_micro_doppler_result` — serializes the dedicated rotor result schema.

`PointCloudBoundsTests`:

- `test_display_defaults_are_ten_meters_and_sixty_degrees` — verifies default range and angular display bounds.
- `test_point_cloud_axes_match_ten_meter_sixty_degree_fov` — configures 3D axes for the default range and field of view.
- `test_direction_cosine_gate_rejects_points_beyond_sixty_degrees` — rejects points outside the ±60-degree angular gate.
- `test_range_limit_includes_every_range_bin` — includes every bin within the selected radial limit.
- `test_draw_updates_points_and_cluster_centers` — updates dynamic points and cluster-center artists.
- `test_draw_updates_tracked_target_marker` — updates the measured target marker.
- `test_draw_keeps_predicted_target_marker_visible` — retains a translucent marker during prediction-only tracking.
- `test_draw_updates_static_points_and_calibration_indicator` — updates static points and reference-calibration status.
- `test_draw_shows_ready_classification` — renders a completed classification result.
- `test_draw_shows_classification_warmup` — renders classifier history warm-up status.

`SingleTargetTrackerTests`:

- `test_acquires_strongest_then_follows_nearest_candidate` — acquires the strongest target and subsequently associates the nearest candidate.
- `test_nearest_policy_acquires_nearest_candidate` — supports nearest-first initial acquisition policy.
- `test_coasts_through_miss_and_reassociates` — predicts through a miss and reassociates a returning target.
- `test_drops_track_after_missed_update_limit` — removes a track after its configured miss limit.
- `test_cluster_candidates_use_assigned_point_magnitude` — derives cluster candidate strength from assigned member points.

`MotionHandoffQualifierTests`:

- `test_healthy_dynamic_track_does_not_open_handoff` — avoids static handoff while dynamic measurements remain healthy.
- `test_confirmed_dynamic_track_opens_only_after_dynamic_misses` — opens handoff only after a confirmed moving track begins missing.
- `test_dynamic_reacquisition_cancels_active_handoff` — cancels pending static handoff when dynamic tracking resumes.
- `test_unconfirmed_and_predicted_tracks_do_not_arm_handoff` — prevents tentative or prediction-only tracks from authorizing handoff.
- `test_motion_protection_releases_after_configured_misses` — releases the protected motion region after the configured miss count.

`TrackedDisplayPayloadTests`:

- `test_static_tracker_uses_one_hit_handoff_and_two_second_coast` — applies one-hit authorized handoff and the two-second static coast interval.
- `test_combined_display_uses_tracked_range_for_micro_doppler` — centers combined-mode micro-Doppler on the tracked target range.
- `test_processed_writer_runs_without_a_display_queue` — continues processed JSONL output when no display queue exists.
- `test_unqualified_static_clusters_cannot_override_dynamic_track` — prevents static clutter without motion qualification from taking ownership.
- `test_motion_qualified_cluster_hands_off_exact_members_only` — transfers ownership to the qualified cluster and exposes only its members.
- `test_first_qualified_static_maximum_completes_handoff` — permits the first qualified static maximum to complete an authorized handoff.
- `test_stopped_dynamic_target_transitions_without_selection_gap` — moves from dynamic to static ownership without dropping target selection.
- `test_strict_static_cluster_override_rejects_single_maximum` — rejects an isolated static maximum when strict cluster ownership is required.
- `test_static_maximum_outside_handoff_gate_is_rejected` — rejects static candidates beyond the last dynamic-position gate.
- `test_static_handoff_selects_cluster_nearest_dynamic_position` — selects the eligible cluster nearest the last dynamic position.
- `test_disabling_static_detection_skips_static_processing` — bypasses the entire static branch when disabled.
- `test_static_detection_runs_on_every_point_cloud_update` — executes static detection at full point-cloud cadence.
- `test_ready_static_detection_sleeps_without_handoff_or_track` — avoids unnecessary static clustering when no track or handoff exists.
- `test_static_clustering_is_localized_to_handoff` — restricts static clustering to the active handoff neighborhood.
- `test_micro_doppler_history_survives_brief_track_gap` — retains spectrogram history through a short selection gap.
- `test_micro_doppler_history_survives_nearby_static_handoff` — preserves history across a spatially nearby dynamic-to-static handoff.
- `test_micro_doppler_history_survives_continuous_large_motion` — retains history during uninterrupted ownership despite large displacement.
- `test_micro_doppler_history_resets_for_distant_reacquisition` — resets history when reacquisition is spatially distant.
- `test_micro_doppler_history_expires_after_long_gap` — resets history after exceeding the allowed update gap.
- `test_micro_doppler_history_accepts_thirty_update_gap` — treats exactly 30 missing updates as recoverable.

`ClassificationIntegrationTests`:

- `test_fixed_range_is_converted_to_nearest_range_bin` — maps a physical fixed gate to its nearest configured range bin.
- `test_native_feature_is_available_without_a_trained_classifier` — emits native feature steps even when no classifier is loaded.
- `test_combined_mode_overlaps_classification_with_micro_doppler` — overlaps classification work with the next combined-mode DSP update.
- `test_predicted_track_continues_classification_history` — keeps classifier history during association-preserving prediction.
- `test_confirmed_owner_change_resets_classification_history` — clears history when a genuinely different target takes ownership.
- `test_history_reset_is_forwarded_to_evaluation_logger` — forwards explicit target changes and engine-originated failures with frame, target context, prior history length, discarded-step count, and reset reason.

`RangeDisplayBoundsTests`:

- `test_range_profile_uses_ten_meter_default_limit` — limits the default range-profile axis to 10 m.
- `test_range_doppler_uses_ten_meter_default_limit` — limits the default range-Doppler image to 10 m.

`MicroDopplerDisplayTests`:

- `test_shared_magnitude_colorbar_starts_at_sixty_db` — fixes the shared magnitude scale's lower bound at 60 dB.
- `test_combined_point_cloud_keeps_full_rate_rendering` — keeps combined-mode rendering enabled for every consumed update.
- `test_live_history_keeps_150_stft_windows` — bounds the live spectrogram history at 150 windows.
- `test_stft_uses_64_loop_window_and_32_loop_hop` — verifies combined-mode STFT window and hop constants.
- `test_live_range_gate_uses_five_bins` — verifies the live visualization range gate spans five bins.
- `test_draw_sets_centered_doppler_and_history_axes` — updates centered velocity and historical time extents correctly.
- `test_blitted_draw_does_not_mutate_static_axes` — keeps fixed axes unchanged during incremental drawing.
- `test_event_rate_counts_units_after_initial_timestamp` — computes event rates only after establishing the first timestamp.
- `test_rate_indicator_formats_display_rate` — formats a measured display update rate for the status label.
- `test_rate_indicator_shows_measurement_pending` — shows a pending state before enough events exist for a rate.
- `test_dedicated_rotor_defaults_prioritize_flash_timing` — verifies dedicated-mode defaults preserve blade-flash time resolution.
- `test_rotor_time_resolution_separates_max_rpm_blade_passages` — confirms the STFT hop resolves blade passages at maximum RPM.
- `test_rotor_display_frame_includes_notch_overlay_bounds` — includes the clutter-notch limits in the display payload.
- `test_turbo_lookup_table_is_compact_uint8_rgb` — builds a 256-entry uint8 RGB Turbo lookup table.
- `test_rotor_colorization_produces_finite_direct_rgba_image` — converts rotor power directly into finite contiguous RGBA pixels.
- `test_rotor_display_process_dispatches_to_pyqtgraph` — routes dedicated rotor display startup to the PyQtGraph implementation.
- `test_range_display_process_dispatches_to_pyqtgraph` — routes range display startup to the PyQtGraph implementation.
- `test_display_startup_status_reports_ready_backend` — reports the visible GUI backend through the startup-status channel.
- `test_rotor_qt_arguments_exclude_radar_display_option` — prevents the radar `--display` option from reaching Qt's parser.
- `test_rotor_dependency_error_explains_missing_pyqtgraph` — provides an actionable error when PyQtGraph is unavailable.
- `test_rotor_dependency_error_explains_missing_xcb_cursor` — provides the native package remedy for a missing XCB cursor library.
- `test_rotor_raster_is_bounded_and_max_pooling_preserves_flashes` — bounds raster width while retaining short flashes during max pooling.
- `test_rotor_raster_defaults_to_active_acquisition_span` — derives the default display span from active sampling time.
- `test_rotor_display_concatenates_only_active_window_intervals` — removes inactive frame gaps from display-only time coordinates.
- `test_rotor_display_fills_time_gaps_from_nearest_spectrum` — fills display raster gaps using the nearest measured spectrum.
- `test_gap_aware_series_inserts_nan_at_frame_gap` — inserts NaN separators into analysis series across capture gaps.

### `test_run.py` (41 cases)

`ChooseDurationMinutesTests`:

- `test_blank_input_uses_five_minute_default` — uses five minutes when the duration prompt is left blank.
- `test_zero_means_unlimited` — interprets zero duration as no automatic stop.
- `test_cli_value_skips_prompt` — honors a supplied duration without prompting.
- `test_negative_cli_value_is_rejected` — rejects negative durations.
- `test_non_finite_cli_value_is_rejected` — rejects NaN and infinite durations.

`ChooseMicroDopplerRangeTests`:

- `test_blank_input_uses_2_15_meter_default` — uses the 2.15 m default fixed range.
- `test_explicit_positive_range_skips_prompt` — accepts a positive CLI range without prompting.
- `test_invalid_explicit_range_is_rejected` — rejects non-positive or non-finite fixed ranges.

`ChooseDisplayTests`:

- `test_blank_input_uses_combined_display_default` — selects combined point-cloud/micro-Doppler mode by default.
- `test_combined_display_menu_choice` — maps the combined menu selection correctly.
- `test_combined_display_is_a_cli_choice` — exposes combined mode through argument parsing.
- `test_dedicated_rotor_display_is_a_cli_choice` — exposes dedicated rotor mode through argument parsing.
- `test_calibration_is_menu_option_seven` — keeps range calibration assigned to menu option seven.
- `test_angular_calibrations_are_menu_options_eight_and_nine` — maps azimuth and elevation calibration to options eight and nine.
- `test_dedicated_rotor_defaults_match_current_drone` — keeps rotor defaults aligned with the current aircraft configuration.

`ChooseLiveClassificationTests`:

- `test_blank_input_disables_classification_by_default` — defaults the interactive classification choice to disabled.
- `test_yes_enables_classification` — recognizes affirmative interactive input.
- `test_explicit_cli_value_skips_prompt` — honors an explicit classification flag without prompting.

`ChooseInferenceLoggingTests`:

- `test_blank_input_disables_logging_by_default` — keeps optional inference evaluation off when the prompt is left blank.
- `test_yes_enables_logging` — enables evaluation logging after an affirmative response.
- `test_cli_values_skip_prompt` — honors explicit enable/disable flags without prompting.
- `test_custom_path_enables_logging` — treats an explicit evaluation path as opt-in.
- `test_custom_path_conflicts_with_explicit_disable` — rejects a path combined with explicit logging disablement.
- `test_evaluation_label_prompt_normalizes_non_drone` — maps the interactive non-drone spelling to the stored `not_drone` label.
- `test_explicit_evaluation_label_skips_prompt` — honors a supplied ground-truth label without prompting.

`ChooseDatasetOutputDirectoryTests`:

- `test_blank_input_uses_dataset_root` — stores captures under the dataset root by default.
- `test_uav_and_others_choices` — maps labelled UAV and non-UAV directory selections.
- `test_option_three_uses_dataset_root` — maps the third menu option to the unlabelled dataset root.

`CaptureCommandTests`:

- `test_calibration_command_disables_normal_processing` — constructs calibration capture without ordinary detection/classification work.
- `test_processed_output_is_default_and_raw_output_is_opt_in` — enables processed JSONL by default, keeps inference logging off by default, and forwards explicit evaluation options while raw ADC remains opt-in.
- `test_rotor_command_forwards_gate_and_estimator_settings` — forwards dedicated rotor range, geometry, and RPM options.

`SubprocessEnvironmentTests`:

- `test_linux_vnc_environment_defaults_to_display_zero` — supplies `DISPLAY=:0` for the supported Linux VNC environment when unset.
- `test_existing_display_environment_is_preserved` — does not overwrite an explicitly configured display server.

`ClassificationResultChannelTests`:

- `test_relay_extracts_structured_result_and_forwards_other_logs` — parses classification messages while relaying unrelated output.
- `test_relay_reports_capture_readiness` — recognizes and signals the capture-ready marker.
- `test_wait_for_capture_ready_stops_when_capture_exits` — fails readiness waiting when the capture child terminates.
- `test_wait_for_capture_ready_accepts_ready_capture` — succeeds after a live child emits readiness.
- `test_explicit_cuda_allows_first_engine_build` — grants the longer startup window for an explicit CUDA build.
- `test_cpu_classification_keeps_normal_startup_timeout` — retains the normal timeout for CPU classification.
- `test_auto_uses_gpu_timeout_on_jetson` — uses the extended engine-build timeout for automatic Jetson CUDA selection.
- `test_report_drains_ready_classification_without_printing` — consumes ready classification results without duplicate console output.

### `test_startup.py` (5 cases)

`DCA1000PacketDelayTests`:

- `test_supported_packet_delay_boundaries_are_accepted` — accepts both documented DCA1000 packet-delay limits.
- `test_packet_delay_outside_hardware_range_is_rejected` — rejects delays outside the FPGA-supported range.
- `test_fifty_microseconds_encodes_to_6250_fpga_cycles` — converts 50 microseconds into the expected 6,250 cycles.
- `test_repository_setup_uses_fifty_microseconds` — verifies the committed capture setup selects the validated delay.

`SdkProfileCommandTests`:

- `test_repository_profile_does_not_send_host_angle_metadata` — keeps host-only angular compensation out of radar CLI commands.

### `test_tensorrt_inference.py` (11 cases)

`DeviceResolutionTests`:

- `test_auto_requires_cuda_on_jetson` — resolves automatic device selection to CUDA on Jetson.
- `test_auto_uses_cpu_off_jetson` — resolves automatic selection to CPU on other hosts.
- `test_cuda_creation_does_not_fall_back` — makes requested CUDA failures fatal rather than silently selecting CPU.
- `test_cuda_creation_forwards_progress_callback` — forwards engine-build status through the supplied callback.

`EngineCacheTests`:

- `test_fp16_probability_tolerance_accepts_small_calibrated_drift` — accepts FP16 parity drift within the 0.005 calibrated-probability limit.
- `test_cache_requires_matching_engine_hash_and_parity` — requires both engine identity and successful parity metadata.
- `test_cache_ignores_volatile_reported_total_gpu_memory` — excludes changing free/total-memory reporting from compatibility identity.
- `test_existing_engine_is_validated_without_recompiling` — validates and reuses an acceptable existing engine.
- `test_engine_is_compiled_only_when_cache_file_is_absent` — builds TensorRT only when no engine cache exists.
- `test_parity_uses_training_export_without_pytorch` — validates against exported parity data without loading PyTorch.

`StatefulTensorRTTests`:

- `test_feature_history_runs_tensor_rt_at_step_48` — invokes TensorRT when the 48th valid feature step completes the window.

## Current Limitations

- No packet reordering; only duplicate and overlap handling.
- Only complex 16-bit, two-lane LVDS reshape is implemented.
- The dedicated rotor mode derives an approximate velocity axis from profile
  timing. Configured range and channel corrections are available, but there is
  no rotor-specific velocity or blade-geometry calibration.
- Tracking supports one target only. XYZ positions can include configured
  range/channel and fixed angular corrections, but track accuracy and identity
  are not externally validated and association is not Doppler-assisted.
- Range FFT includes the full complex FFT rather than selecting only a
  physically useful half-spectrum.
- The point cloud is a diagnostic visualization, not precision metrology.
- `startup.py` cannot itself run `livedatacapture.py`; use `main/run.py` or two
  terminals.
- Logs and raw captures grow until stopped or the filesystem fills.

## Design Rules for Future Changes

- Never interpret one UDP packet as one chirp or frame.
- Keep receiver work bounded and avoid blocking it on processing or display.
- Keep radar configuration and frame interpretation sourced from the same
  `.cfg`/JSON file.
- Preserve packet-loss and processing-drop counters when adding stages.
- Validate reshape and antenna mapping against known TI capture output before
  treating measurements as calibrated.
