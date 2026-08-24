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
  `-- tensorrt_inference.py

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
