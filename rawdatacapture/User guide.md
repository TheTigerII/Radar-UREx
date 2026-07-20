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

Point-cloud display defaults to updating every two valid frames; other modes
default to every frame. Override it with `--display-update-every N`.

## Capture Receiver Only

This starts UDP capture but does not configure or start the hardware:

```powershell
python rawdatacapture\livedatacapture.py
```

Its defaults are `mmwave.json`, `setup.json`, host `192.168.33.30`, UDP port
4098, no display, processing queue size 4, and a requested 4 MiB socket receive
buffer. The operating system may grant a different receive-buffer size; the
actual value is printed.

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

The diagnostic point cloud uses an adaptive range-Doppler clutter map, OS-CFAR,
Doppler-only peak filtering, and a batched virtual antenna 2D FFT. The first 30
display updates learn the background and intentionally produce no point
detections, so start capture with an empty scene when possible. After warm-up,
each cell is divided by its learned background power before CFAR. A detection
must also exceed the background by at least 6 dB. Neighborhoods around current
detections are protected from map updates. The point cloud shows all
detected points within a 10 m radial range and a ±60-degree azimuth/elevation
field of view. Spatial DBSCAN runs after XYZ generation with a default 0.5 m
neighborhood and two-point minimum. Original
points remain visible, while red crosses mark cluster centers and cross size
indicates the number of cluster members. Coordinates are X left/right, Y
forward, and Z elevation. The plot spans 0 to 10 m forward and approximately
-8.66 to +8.66 m across X and Z. Point color is fixed from 40 to 120 dB. Angle
output is not calibrated.

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
uses a five-range-bin gate centered on the strongest point-cloud return. If the
point cloud is empty, it selects the strongest non-zero-Doppler range inside
`--max-range-m`. The current gate range is shown in the spectrogram title.

The horizontal spectrogram axis is measured in STFT windows, with the newest
column at zero. The vertical axis is centered Doppler bin because velocity is
not yet calibrated. Its magnitude color scale uses the same fixed 40 to 120 dB
limits as the 3D point cloud, so colors remain comparable between updates and
the two plots. One shared magnitude colorbar sits between both plots. Automatic
gating can switch between targets when multiple
strong objects are present; it is a visualization aid rather than persistent
target tracking.

The plot titles show their measured refresh rate. On Matplotlib backends that
support it, the combined view blits only the changing scatter, image, and title
artists rather than redrawing the full 3D axes and colorbar for every frame.

### Display performance

Display rendering is a separate process and receives only the latest result.
If processing cannot keep up, reduce display frequency:

```powershell
python rawdatacapture\livedatacapture.py `
  --display range-doppler --display-update-every 3 --display-pause 0.05
```

The combined display is the most computationally expensive mode. Use
`--display-update-every 2` or higher if `processing_drops` increases.

Use `--display none` for packet-loss testing or headless operation.

## Processed Output

The integrated launcher saves processed data by default. Override its path with:

```powershell
python run.py --processed-output .\rawdatacapture\captures\processed.jsonl
```

The first JSONL record contains the radar configuration and column definitions.
Each following record contains one processed update with:

- `points`: `[x_m, y_m, z_m, magnitude_db]` rows;
- `clusters`: `[x_m, y_m, z_m, point_count]` rows;
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

Successful frames and display latency are not printed. A statistics line is
emitted only when one or more capture or processing error counters change.

If `processing_drops` grows while packet-loss counters remain zero, reduce
display updates or use `--display none`. Increasing `--processing-queue-size`
can absorb short bursts but does not fix processing that is consistently too
slow.

## Packet-Loss Troubleshooting

- Connect the PC directly to DCA1000 over wired Ethernet.
- Ensure `packetDelay_us` in `setup.json` matches the DCA1000 configuration.
- Avoid routing capture traffic through Wi-Fi, VPNs, or busy switches.
- Increase the OS/network-adapter receive buffers where supported.
- Check the logged requested and actual socket receive-buffer sizes.
- Run with `--display none` to distinguish capture loss from DSP load.

When `packetSequenceEnable` is false, capture still assembles received bytes but
cannot reliably report network packet loss.

## Shutdown

Press Ctrl+C. With `run.py`, the startup process is stopped first, followed by
the capture process. Clean capture shutdown closes the socket, child processes,
queues, raw file, metadata sidecar, and log file.

Show all receiver options with:

```powershell
python rawdatacapture\livedatacapture.py --help
```
