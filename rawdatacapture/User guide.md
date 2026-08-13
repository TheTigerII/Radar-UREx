# Live Raw ADC Capture User Guide

This guide explains `livedatacapture.py` and the integrated `run.py` launcher
for an IWR6843ISK-ODS connected to a DCA1000EVM.

## Prerequisites

- Python 3 with the capture dependencies installed as shown below.
- Git, which pip uses to install the pinned OpenRadar dependency.
- PC Ethernet interface configured as `192.168.33.30/24` by default.
- DCA1000 reachable at `192.168.33.180`.
- Radar running SDK CLI firmware when using `profile.cfg` and direct serial.
- `profile.cfg` must enable ADC LVDS streaming, for example:

```text
lvdsStreamCfg -1 0 1 0
```

Run commands from the repository root. Windows examples use `python` and
backslashes; Linux examples may use `python3` and forward slashes.

Create a project virtual environment and install the pinned capture dependencies
before the first run. On Linux:

```bash
sudo apt-get install libxcb-cursor0
python3 -m venv --system-site-packages .venv
.venv/bin/python -m pip install \
  "numpy<2" scipy pyserial "scikit-learn>=1.4,<2" \
  "pyqtgraph>=0.13.7,<0.15" "PySide6>=6.7,<7" "PyOpenGL>=3.1.7"
.venv/bin/python -m pip install \
  "openradar @ git+https://github.com/PreSenseRadar/OpenRadar.git@65bcd6287af31685acf8b0c32f4505e0f6faab94"
```

On Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install `
  'numpy<2' scipy pyserial 'scikit-learn>=1.4,<2' `
  'pyqtgraph>=0.13.7,<0.15' 'PySide6>=6.7,<7' 'PyOpenGL>=3.1.7'
.venv\Scripts\python -m pip install `
  'openradar @ git+https://github.com/PreSenseRadar/OpenRadar.git@65bcd6287af31685acf8b0c32f4505e0f6faab94'
```

The version bounds and OpenRadar commit are intentional. Use the same commands
when recreating an environment so range/Doppler behavior and saved model
metadata remain reproducible.

`libxcb-cursor0` is required for PySide6 windows on an X11 desktop.
Display startup checks this before capture and reports an installation command
instead of silently running Qt's invisible offscreen backend.
The launcher also isolates the radar `--display micro-doppler` option from
Qt's own X11 command-line parser.
Before starting UDP capture, the display child now reports its backend,
resolved `DISPLAY`, screen name, and geometry to the parent. A missing or
non-visible GUI aborts startup with a direct error instead of continuing with
an empty desktop.

### Classification notebook and CNN dependencies

Live drone/not-drone classification requires PyTorch 2.6 or newer and below
version 3. Combined point-cloud + micro-Doppler mode asks whether to enable it,
with **no** as the default. Install Jupyter as well when using
`classification.ipynb`:

```bash
.venv/bin/python -m pip install jupyter "torch>=2.6,<3"
```

```powershell
.venv\Scripts\python -m pip install jupyter 'torch>=2.6,<3'
```

PyTorch builds depend on the operating system, Python version, and whether CPU
or CUDA acceleration is required. Check the official
[PyTorch installation selector](https://pytorch.org/get-started/locally/) and
use its platform-specific command when it differs from the generic command
above. The CNN cells in `classification.ipynb` are unexecuted by default and
training starts only when the training cell is run manually.
Run `run.py --no-classification` when capture is needed in an environment
without PyTorch.

The notebook uses only native IWR6843 feature captures. Place UAV capture
JSONL files under `dataset/uav/` and tracked non-UAV target captures under
`dataset/others/`. Capture with `--display-update-every 1`; processed JSONL
records save the exact two-channel, two-range-bin `classification_feature`
consumed by live inference. Training creates non-overlapping 48-frame windows
and uses a stratified 70%/15%/15% window-level split. Use
`--no-classification` while collecting the initial training set so obsolete
model artifacts do not block capture startup.

On Jetson, `--classification-device auto` resolves to the required CUDA
backend. Training and ONNX export happen in Colab or on the training
workstation, not on the Jetson. The notebook exports to
`training_output/micro_doppler_cnn/`. Copy `model_state.pt`, `model.onnx`,
`manifest.json`, `calibration.joblib`, and `parity.npz` into the repository's
`model_weights/` directory on the Jetson, then install the Jetson-native
TensorRT runtime once:

```bash
./scripts/setup_jetson_tensorrt.sh
```

The launcher builds and caches a fixed-batch-one TensorRT FP16 engine from the
training-exported `model.onnx` on first use. TensorRT selects optimized
Orin tactics, then the launcher validates the engine against the small
training-exported `parity.npz` reference set. A label mismatch
or calibrated-probability error above `1e-3` aborts startup. Later runs verify
the ONNX, parity data, calibration, model card, radar profile, TensorRT/CUDA,
and GPU fingerprints before reusing the engine. PyTorch and ONNX Python
packages are not required on the Jetson. The launcher allows five minutes for
this one-time GPU build. CUDA/TensorRT failure never silently falls back to
CPU; use `--classification-device cpu` only as an explicit diagnostic override
when PyTorch is installed.

Live range and Doppler processing uses OpenRadar. OS-CFAR uses a vectorized
local implementation for real-time performance. The IWR6843ISK-ODS planar-array
coordinate mapping remains in the local DSP adapter because
OpenRadar's supplied XYZ helper targets the AWR1843 virtual antenna layout.
All live views use PyQtGraph with the PySide6 Qt binding. Range and
range-Doppler views use native 2D plot/image items, 3D point clouds use
PyQtGraph OpenGL, and combined mode places the OpenGL point cloud and
micro-Doppler image in one window.

## Recommended: Integrated Run

```powershell
.venv\Scripts\python run.py
```

`run.py`:

1. Prompts for a display mode unless `--display` is supplied.
2. Prompts for a run duration in minutes. Press Enter for the 5-minute default,
   or enter `0` for an unlimited run.
3. Detects serial ports and asks which command UART to use when needed.
4. Starts `livedatacapture.py` with `profile.cfg` and `setup.json` by default.
5. In combined point-cloud + micro-Doppler mode, asks whether to run live CNN
   classification (default: no), then asks whether to save the timestamped
   JSONL under `dataset/uav`, `dataset/others`, or `dataset` (default: dataset).
   Other display modes retain the timestamped `rawdatacapture\captures` default.
   Raw ADC recording is disabled unless `--raw-output` is explicitly given.
6. When enabled, runs the trained CNN after 48 consecutive valid tracked
   frames, reports `DRONE`, `NOT_DRONE`, or `UNKNOWN`, and saves the calibrated
   probability.
7. Starts `startup.py` with direct serial radar control and direct UDP DCA1000
   control.
8. Stops hardware control before capture when the duration expires or Ctrl+C
   is pressed.

Choose a display without prompting:

```powershell
python run.py --display none
python run.py --display range
python run.py --display range-doppler
python run.py --display point-cloud
python run.py --display point-cloud-micro-doppler
```

Temporarily zoom a live display to the first 0.5 m with:

```powershell
python run.py --display range --max-range-m 0.5
```

Skip the duration prompt with an explicit value:

```powershell
python run.py --duration-minutes 5
python run.py --duration-minutes 0
```

The second command runs without a time limit.

Specify the radar UART if automatic detection is wrong:

```powershell
python run.py --radar-port COM4
```

```bash
python3 run.py --radar-port /dev/ttyUSB0
```

All display modes default to processing every valid frame. Override this with
`--display-update-every N`.

## Capture Receiver Only

This starts UDP capture but does not configure or start the hardware:

```powershell
python rawdatacapture\livedatacapture.py
```

Its defaults are `mmwave.json`, `setup.json`, host `192.168.33.30`, UDP port
4098, no display, an 8,192-datagram receiver queue, a 32-frame processing
queue, and a requested 4 MiB socket receive buffer. The operating system may
grant a different receive-buffer size; the actual value is printed.

The integrated launcher exposes all three buffering controls:

```powershell
python run.py `
  --socket-recv-buffer 4194304 `
  --packet-queue-size 8192 `
  --processing-queue-size 32
```

Use the same config that programmed the radar:

```powershell
python rawdatacapture\livedatacapture.py `
  --config .\rawdatacapture\profile.cfg `
  --setup .\rawdatacapture\setup.json
```

Both SDK CLI `.cfg` and mmWave Studio `.json` radar configs are supported. XML
is not supported.

## Displays

### Range profile

```powershell
python rawdatacapture\livedatacapture.py --display range
```

The plot is average range-FFT magnitude over chirps and receivers. Its X-axis
uses meters when sample rate and frequency slope are available; otherwise it
falls back to bin numbers. The displayed X limit defaults to 10 m:

```powershell
python rawdatacapture\livedatacapture.py --display range --max-range-m 10
python rawdatacapture\livedatacapture.py --display range --max-range-m 0
```

Zero selects the full computed axis.

### Range-Doppler heatmap

```powershell
python rawdatacapture\livedatacapture.py --display range-doppler
```

The Y-axis is centered Doppler-bin index, not calibrated velocity. Processing
uses frame loops as slow time and averages magnitude across TX chirp position
and receivers.

### Point cloud

```powershell
python rawdatacapture\livedatacapture.py --display point-cloud
```

The diagnostic point cloud has parallel moving-target and static-change paths.
The moving path uses an adaptive range-Doppler clutter map, OS-CFAR,
Doppler-only peak filtering, and a batched virtual antenna 2D FFT.

Static-change detection is enabled by default. Keep the radar and monitored
scene fixed and leave the target absent for startup calibration. The first
30 processed detection updates are discarded as warm-up, then the next
150 updates build a median range-azimuth-elevation reference. The plot reports
the warm-up and calibration progress, then `Static reference ready (fixed)`.
The detector learns normal per-angle-cell variation, applies a per-range power
floor, removes common receiver-gain drift, and temporally smooths the change
map. A cell must exceed both the configured 3 dB minimum and four times its
learned noise variation. The reference and learned noise estimates remain
fixed after calibration by default, so later scene changes are not absorbed.

Raw changes do not become displayed static targets by themselves. A confirmed
dynamic track must move at least 0.3 m during the preceding 30 frames. For the
next 60 frames, a static local maximum or cluster can take over only if it is
within 0.75 m of the last dynamic position and remains associated for three
consecutive frames. One point is sufficient in each frame by default because
local-maximum filtering has already reduced a reflector to one candidate.
The validated target is protected by ±2 range, azimuth, and elevation cells
and remains visible after stopping. Protection is released after 30
consecutive misses so removed targets are eventually absorbed. Only the exact
points in the validated cluster are shown as orange squares; its center is
shown as a cyan diamond. Transient and static-only clutter is suppressed.

Static angle processing remains full rate: it runs for every processed
point-cloud update with the complete 32-by-32 angle FFT. Temporal smoothing
can delay a new static point by a few frames. Objects present during
calibration become part of the reference and are not reported as changes.
Calibration counts updates that actually reach point-cloud processing, so
packet/frame loss makes it take longer than 180 physical frames.

Both paths apply a 10 m radial range and a ±60-degree azimuth/elevation field
of view. Spatial DBSCAN defaults to a 0.4 m neighborhood and two-point minimum.
Dynamic points remain magnitude-colored, and red crosses mark their cluster
centers. Coordinates are X left/right, Y forward, and Z elevation. The plot
spans 0 to 10 m forward and approximately -8.66 to +8.66 m across X and Z.
Dynamic point color is fixed from 40 to 120 dB. Angle output is not calibrated.

The software OS-CFAR requested probability of false alarm defaults to
`1e-3` per axis.

Use `--max-range-m` to change the shared range limit for any display, and use
`--point-cloud-fov-deg` to change the point-cloud half-FOV.

Use `--cluster-eps-m` and `--cluster-min-samples` to tune DBSCAN. Set
`--cluster-eps-m 0` to disable cluster-center generation:

```powershell
python run.py --display point-cloud --cluster-eps-m 0.4 --cluster-min-samples 3
```

The clutter-map warm-up update rate defaults to `0.02`, and its minimum
normalized target-to-background ratio defaults to 3 dB. The map is fixed once
warm-up completes. Change the initial learning period and
minimum ratio with `--clutter-map-warmup-frames` and
`--clutter-map-min-snr-db`. Use `--clutter-map-update-rate 0` to disable the
software clutter map. These options affect point detection only; point-cloud
magnitude, the range-Doppler display, and micro-Doppler retain raw power.

Use `--static-warmup-frames`, `--static-reference-frames`,
`--static-background-update-rate`, `--static-cluster-min-samples`, and
`--static-min-change-db` to tune calibration, adaptation, validation, and the
absolute sensitivity floor. Defaults are 30 warm-up frames, 150 reference
frames, a fixed reference (zero adaptation rate), one same-frame cluster member,
and 3 dB. Set a nonzero `--static-background-update-rate` to opt into adaptation.
Set `--static-cluster-min-samples 3` to require three spatially adjacent candidates
in every update in addition to temporal confirmation. The learned
noise threshold can make the effective threshold higher in unstable cells.
Disable the static branch without changing the moving-target path with:

```powershell
python run.py --display point-cloud --no-static-detection
```

### Point cloud with micro-Doppler

```powershell
python run.py --display point-cloud-micro-doppler
```

This mode places the 3D point cloud and a rolling 150-window micro-Doppler
spectrogram side by side. It reuses one Doppler cube for tracking and range-gate
selection. The micro-Doppler path calculates a separate slow-time FFT for each
TX slot using 64-loop Hann windows, a 32-loop hop, and a 128-point FFT. The TX
and RX powers are summed after the FFT; the TX signals are not coherently
merged. A 128-loop frame produces three short-time spectra. The spectrogram
uses a five-range-bin gate centered on a confirmed track. A measured dynamic
track has priority while moving, then a motion-qualified validated static
track takes over after stopping. Static-only clutter cannot activate
micro-Doppler, and no arbitrary range is selected during calibration or track
confirmation. The explicit target gate retains the
centered zero-Doppler bin, so a rigid stationary object appears mainly as a
zero-Doppler line while vibration or internal motion produces sidebands. The
current gate range is shown in the spectrogram title.

The horizontal spectrogram axis is measured in STFT windows, with the newest
column at zero. The vertical axis is centered Doppler bin because velocity is
not yet calibrated. Its visible-spectrum `turbo` scale runs from dark blue at
the fixed 60 dB minimum to red at the fixed 120 dB maximum, matching the 3D
point cloud so colors remain comparable between updates and the two plots.
One shared magnitude colorbar sits between both plots. Spectrogram history
remains frozen through target-selection gaps of up to 30 processed updates and
continues across uninterrupted motion and a nearby dynamic-to-static handoff.
After a selection gap, it starts fresh when the reacquired target is more than
0.75 m from the previous target position or the gap exceeds 30 updates,
preventing different objects from sharing one visible history.

The plot titles or status labels show their measured refresh rate. PyQtGraph
updates persistent curve, image, and OpenGL scatter items rather than rebuilding
the full view for every frame. Both the micro-Doppler and 3D panels render every
available display payload.

### Dedicated rotor micro-Doppler

Use the dedicated single-TX mode when blade-flash visibility or RPM is more
important than simultaneous 3D tracking:

```powershell
python run.py --display micro-doppler --micro-doppler-range-m 2.15 `
  --rotor-blades 2 --rotor-count 1 --rotor-radius-m 0.05 `
  --rotor-rpm-min 1000 --rotor-rpm-max 10700 `
  --raw-output .\rawdatacapture\captures\rotor.bin
```

The default rotor model is two blades with a maximum search speed of
10,700 RPM. These match the current drone and can still be overridden with
`--rotor-blades` and `--rotor-rpm-max`.

Option 6 uses the same `profile.cfg` LVDS waveform as every other live mode.

The compatible profile samples each TX every 117 microseconds, giving an
approximately 8.55 kHz per-TX slow-time rate and ±10.7 m/s unambiguous radial
velocity at 60 GHz. The dedicated mode still applies the three-bin gate,
stationary-return cancellation, relative spectrum, flash score, and RPM
estimator.

The fixed range gate is required because this mode intentionally skips point
cloud detection and tracking. When option 6 is selected interactively,
`run.py` prompts for the gate and defaults to 2.15 m. Its default half width is
one bin, giving a three-bin gate. Each 16-loop Hann window has its weighted
complex mean removed before the FFT, suppressing the stationary body return.
The live plot displays power relative to a robust per-window floor over 0 to
30 dB and masks the central ±2 bins in the enhanced view only. Raw power
remains in processed output. A dedicated adaptive gate estimates noise spread
from `1.4826 × MAD` of the lower half below each window's median; using the
lower tail prevents positive blade ridges from raising their own threshold.
The gate is the greater of 3 dB or three robust standard deviations above the
floor. A cell must also have support from at least three cells in its 3×3
Doppler/time neighborhood. Rejected cells are blanked; retained blade energy
keeps its original relative-dB magnitude.

For responsive plotting, the enhanced view concatenates only measured STFT
windows onto a 0.20-second active-acquisition axis and max-pools them onto 512
time columns. Frame blind time, invalid frames, and processing drops therefore
do not stretch a displayed spectrum. The 16-loop window is about 1.87 ms and
the two-loop hop is about 0.234 ms with the compatible three-TX profile. This
is shorter than the
approximately 2.80 ms blade-passage interval of a two-blade rotor at 10,700
RPM. The RPM estimator separately retains two seconds of physical-time
history. Before 0.20 seconds of active history has accumulated, unused raster
columns are filled from the nearest measured spectrum; normal frame gaps are
removed by concatenation. PyQtGraph updates
a persistent row-major `uint8` RGBA image and flash-score curve with a fixed
Turbo color lookup table; unused raw/noise arrays are not copied to the GUI
process. Display color quantization does not reduce the resolution of the
processed JSONL output.

The horizontal display axis is concatenated active acquisition time. It removes
inactive intervals and missing frames so every adjacent column represents the
same nominal STFT hop. Physical timestamps, gap diagnostics, saved data, and
RPM estimation still use the true irregular sampling. The horizontal gray band
around zero velocity remains because the enhanced view intentionally
masks the central ±2 clutter bins. The vertical axis is radial velocity derived
from the configured start frequency and chirp period. The lower panel shows the
broadband blade-flash score. RPM is estimated from its irregularly sampled
blade-passage periodicity and converted using `--rotor-blades`; multiple
separated peaks can be requested with `--rotor-count`.

`--rotor-radius-m` and the maximum configured RPM are used to compare expected
tip speed with the unambiguous velocity. An alias warning does not disable the
temporal RPM estimate, but velocity values in the spectrogram may be folded.
For the strongest radial modulation, mount the radar rigidly and view roughly
along the rotor plane, not along the rotor axis.

### Display performance

Display rendering is a separate process and receives only the latest result.
The dedicated rotor renderer can consume display updates at up to 60 Hz; its Qt
timer always drains to the newest queued result. The combined display preserves
full 30 Hz DSP by default. Final logs include per-stage p50, p95, and maximum
timings so sustained processing load can be distinguished from short
initialization spikes.

`radar_frame_redraw_coverage` measures GUI cadence, not data retention. Judge
capture integrity using `invalid_frames`, `lost_packets`, and
`processing_drops`.

Use `--display none` for packet-loss testing or headless operation.

## Processed Output

The integrated launcher saves processed data by default. Override its path with:

```powershell
python run.py --processed-output .\rawdatacapture\captures\processed.jsonl
```

The first JSONL record declares format version 5 and contains the radar
configuration, column definitions, calibration settings, adaptive-reference
rate, static-validation policy, and classification model contract. Each
following record contains one
processed update with:

- `points`: `[x_m, y_m, z_m, magnitude_db]` rows;
- `clusters`: `[x_m, y_m, z_m, point_count]` rows;
- `static_points`: validated-target
  `[x_m, y_m, z_m, magnitude_db, change_db]` rows only;
- `static_clusters`: the validated target cluster only;
- `static_candidate_count`: raw static activity before validation/suppression;
- `static_validation`: `warming`, `calibrating`, `background`,
  `handoff_pending`, `validated`, or `disabled`;
- static-reference warm-up, calibration, readiness, and adaptation state;
- the current target-track state;
- the newest complete centered-bin micro-Doppler spectrum in dB;
- conventional-mode short-time micro-Doppler windows, laid out as
  `[window][centered_doppler_bin]`;
- the selected micro-Doppler range gate.
- the `[2, 64]` two-range-bin `classification_feature` used to build native
  CNN training windows, or `null` when no valid target owns the frame;
- `classification`: `label`, calibrated `p_drone`, threshold, status/reason,
  and valid history length.

`not_drone` represents the native examples placed in `dataset/others/`; its
coverage is therefore limited to the negative objects and conditions captured
there. Classification stays `unknown`
until 48 consecutive valid target frames are available and resets when target
quality or identity is lost.

Dedicated rotor records additionally populate `rotor_micro_doppler` with raw
and enhanced per-frame spectra, physical window times, the velocity axis,
noise floor, adaptive `noise_gate_db` for each window, blade-flash score, RPM
estimates, confidence/harmonic fields, and velocity-alias diagnostics. Rotor
spectra are saved to 0.01 dB precision.
To avoid serializing the same high-resolution raw spectra twice, dedicated
mode keeps only the newest window in the legacy
`micro_doppler_windows_db` field; the complete frame is in
`rotor_micro_doppler.raw_spectrogram_db`. Other modes write the structured
field as `null` and retain their existing legacy layouts.

Updates are written incrementally, so a clean shutdown is not required to read
the records already saved. A 1 MiB userspace write buffer reduces per-frame
disk overhead, so an abnormal termination may leave the newest buffered
records unwritten. Processed output is generated even when `--display none`
is used.

## Optional Raw Frames

```powershell
python rawdatacapture\livedatacapture.py `
  --raw-output .\rawdatacapture\captures\test_capture.bin
```

Only valid complete frames are written, without DCA1000 headers. The default
metadata path is `test_capture.bin.json`; override it with `--raw-metadata`.
The sidecar is written during clean shutdown.

Raw recording is opt-in because its files are much larger than processed JSONL
output. Files have no size limit. A `run.py` session is limited to 5 minutes by
default, but `livedatacapture.py` alone and `run.py --duration-minutes 0` run
until stopped.

## Logging and Statistics

Terminal output is appended to `rawdatacapture\livedatacapture.log` by default:

```powershell
python rawdatacapture\livedatacapture.py `
  --log-file .\rawdatacapture\capture_run.log
```

Important counters are:

- `lost_packets`: missing DCA1000 sequence numbers.
- `receiver_queue_drops`: UDP datagrams discarded because the in-process
  receiver queue was full.
- `out_of_order`: late/backward packets or byte counts.
- `duplicates`: repeated sequence/payload data.
- `byte_gaps`: discontinuities inserted into normal frame assembly.
- `stream_resyncs`: byte-count jumps larger than one frame; partial assembly
  was discarded to avoid unbounded memory allocation.
- `invalid_frames`: frames touched by an ordinary byte gap.
- `processing_drops`: valid frames discarded because the bounded processing
  queue was full.
- `postprocessing_drops`: DSP-complete dedicated-rotor frames that did not
  complete ordered inference/output processing.

Healthy capture normally keeps all of these at zero. Frames containing gap
padding are neither processed nor saved.

Successful frames and display latency are not printed continuously. A
statistics line is emitted immediately whenever one or more capture or
processing error counters change. Clean shutdown always prints:

- a capture summary with total, valid, invalid, and queued frames;
- a processing summary with queued and completed frames;
- a post-processing summary with completed frames and queue high-water mark;
- a display summary with physically rendered updates, latest-update
  replacements, and frames not rendered over total assembled frames;
- a static summary with raw candidate, handoff-pending, and validated counts;
- DSP and post-processing timing summaries with p50, p95, and maximum stage
  latency.

The display deliberately keeps only the latest result. A skipped display
update is therefore distinct from packet loss, an invalid frame, or a
processing drop.

If `processing_drops` grows while packet-loss counters remain zero, first
confirm that the 32-frame queue is active and inspect the final per-stage
timings. Increasing the queue further only absorbs short bursts; sustained
total-frame p95 above the 33.33 ms frame period requires profiling the reported
stage rather than lowering the configured radar rate.

## Packet-Loss Troubleshooting

- Connect the PC directly to DCA1000 over wired Ethernet.
- Ensure `packetDelay_us` in `setup.json` matches the DCA1000 configuration.
- The repository default is 50 us. If loss remains after host buffering is
  fixed, test 75 us and ensure the DCA1000 FPGA error LED remains off.
- Avoid routing capture traffic through Wi-Fi, VPNs, or busy switches.
- Increase the OS/network-adapter receive buffers where supported.
- Check the logged requested and actual socket receive-buffer sizes.
- Run with `--display none` to distinguish capture loss from DSP load.

On Linux, if the logged actual receive buffer is below the requested 4 MiB,
raise the host limit before capture:

```bash
sudo sysctl -w net.core.rmem_max=4194304
```

To persist that setting, place `net.core.rmem_max=4194304` in a dedicated file
under `/etc/sysctl.d/` and reload the system settings. The capture program only
diagnoses this condition; it never changes privileged host configuration.

When `packetSequenceEnable` is false, capture still assembles received bytes but
cannot reliably report network packet loss.

## Shutdown

Press Ctrl+C. With `run.py`, the startup process is stopped first, followed by
the capture process. The frame processor ignores the parent SIGINT and stops
only after the queue sentinel, so every frame queued before shutdown is
drained. Clean capture shutdown closes the socket, child processes, queues,
raw file, metadata sidecar, and log file.

Show all receiver options with:

```powershell
python rawdatacapture\livedatacapture.py --help
```
