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
python3 -m venv --system-site-packages .venv
.venv/bin/python -m pip install \
  "numpy<2" scipy matplotlib pyserial "scikit-learn>=1.4,<2"
.venv/bin/python -m pip install \
  "openradar @ git+https://github.com/PreSenseRadar/OpenRadar.git@65bcd6287af31685acf8b0c32f4505e0f6faab94"
```

On Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install `
  'numpy<2' scipy matplotlib pyserial 'scikit-learn>=1.4,<2'
.venv\Scripts\python -m pip install `
  'openradar @ git+https://github.com/PreSenseRadar/OpenRadar.git@65bcd6287af31685acf8b0c32f4505e0f6faab94'
```

The version bounds and OpenRadar commit are intentional. Use the same commands
when recreating an environment so range/Doppler behavior and saved model
metadata remain reproducible.

### Optional classification notebook and CNN dependencies

The live capture programs do not require PyTorch. To use
`classification.ipynb`, install Jupyter in a separate ML environment. The
implemented depthwise-separable CNN additionally requires PyTorch 2.6 or newer
and below version 3:

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

Live range and Doppler processing uses OpenRadar. OS-CFAR uses a vectorized
local implementation for real-time performance. The IWR6843ISK-ODS planar-array
coordinate mapping remains in the local DSP adapter because
OpenRadar's supplied XYZ helper targets the AWR1843 virtual antenna layout.
Range, range-Doppler, and 3D point-cloud displays use Matplotlib. The combined
point-cloud/micro-Doppler mode shows both plots in one window.

## Recommended: Integrated Run

```powershell
.venv\Scripts\python run.py
```

`run.py`:

1. Prompts for a display mode unless `--display` is supplied.
2. Prompts for a run duration in minutes. Press Enter for the 3-minute default,
   or enter `0` for an unlimited run.
3. Detects serial ports and asks which command UART to use when needed.
4. Starts `livedatacapture.py` with `profile.cfg` and `setup.json` by default.
5. Streams processed 3D point clouds and micro-Doppler spectra to a timestamped
   `.jsonl` file under `rawdatacapture\captures`. Raw ADC recording is disabled
   unless `--raw-output` is explicitly given.
6. Starts `startup.py` with direct serial radar control and direct UDP DCA1000
   control.
7. Stops hardware control before capture when the duration expires or Ctrl+C
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
python run.py --duration-minutes 3
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
90 updates build a median range-azimuth-elevation reference. The plot reports
the warm-up and calibration progress, then `Static reference ready (adaptive)`.
The detector learns normal per-angle-cell variation, applies a per-range power
floor, removes common receiver-gain drift, and temporally smooths the change
map. A cell must exceed both the configured 6 dB minimum and four times its
learned noise variation. Unprotected background and noise estimates adapt at
0.01 per processed frame, suppressing slow thermal, gain, and stationary-room
drift.

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
packet/frame loss makes it take longer than 120 physical frames.

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

The clutter-map update rate defaults to `0.02`. A smaller value adapts more
slowly to environmental changes. Change the initial learning period and
minimum target-to-background ratio with `--clutter-map-warmup-frames` and
`--clutter-map-min-snr-db`. Use `--clutter-map-update-rate 0` to disable the
software clutter map. These options affect point detection only; point-cloud
magnitude, the range-Doppler display, and micro-Doppler retain raw power.

Use `--static-warmup-frames`, `--static-reference-frames`,
`--static-background-update-rate`, `--static-cluster-min-samples`, and
`--static-min-change-db` to tune calibration, adaptation, validation, and the
absolute sensitivity floor. Defaults are 30 warm-up frames, 90 reference
frames, a 0.01 adaptation rate, one same-frame cluster member, and 6 dB. Set
`--static-cluster-min-samples 3` to require three spatially adjacent candidates
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

The plot titles show their measured refresh rate. On Matplotlib backends that
support it, the combined view blits only the changing scatter, image, and title
artists rather than redrawing the full 3D axes and colorbar for every frame.
The micro-Doppler panel renders every available display payload, while the
more expensive 3D panel renders every second payload. Its depth shading is
disabled so the fixed magnitude colors remain unchanged and the GUI can keep
up with the radar stream.

### Display performance

Display rendering is a separate process and receives only the latest result.
The combined display preserves full 30 Hz DSP by default. Final logs include
per-stage p50, p95, and maximum timings so sustained processing load can be
distinguished from short initialization spikes.

Use `--display none` for packet-loss testing or headless operation.

## Processed Output

The integrated launcher saves processed data by default. Override its path with:

```powershell
python run.py --processed-output .\rawdatacapture\captures\processed.jsonl
```

The first JSONL record declares format version 3 and contains the radar
configuration, column definitions, calibration settings, adaptive-reference
rate, and static-validation policy. Each following record contains one
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
- all short-time micro-Doppler windows generated from that frame, laid out as
  `[window][centered_doppler_bin]`;
- the selected micro-Doppler range gate.

Updates are written incrementally, so a clean shutdown is not required to read
the records already saved. Processed output is generated even when
`--display none` is used.

## Optional Raw Frames

```powershell
python rawdatacapture\livedatacapture.py `
  --raw-output .\rawdatacapture\captures\test_capture.bin
```

Only valid complete frames are written, without DCA1000 headers. The default
metadata path is `test_capture.bin.json`; override it with `--raw-metadata`.
The sidecar is written during clean shutdown.

Raw recording is opt-in because its files are much larger than processed JSONL
output. Files have no size limit. A `run.py` session is limited to 3 minutes by
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

Healthy capture normally keeps all of these at zero. Frames containing gap
padding are neither processed nor saved.

Successful frames and display latency are not printed continuously. A
statistics line is emitted immediately whenever one or more capture or
processing error counters change. Clean shutdown always prints:

- a capture summary with total, valid, invalid, and queued frames;
- a processing summary with queued and completed frames;
- a display summary with physically rendered updates, latest-update
  replacements, and frames not rendered over total assembled frames;
- a static summary with raw candidate, handoff-pending, and validated counts;
- a processing-timing summary with p50, p95, and maximum stage latency.

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
