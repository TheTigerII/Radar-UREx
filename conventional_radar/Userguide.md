# Radar Capture and Hardware Control User Guide

This guide explains the integrated `main/run.py` launcher, the
`livedatacapture.py` data plane, and the `startup.py` hardware control plane for
an IWR6843ISK-ODS connected to a DCA1000EVM.

## Prerequisites

- Python 3 with the capture dependencies installed as shown below.
- PC Ethernet interface configured as `192.168.33.30/24` by default.
- DCA1000 reachable at `192.168.33.180`.
- Radar running SDK CLI firmware when using `profile.cfg` and direct serial.
- Radar booted in the functional mode required by that firmware. For a Rev C
  IWR6843ISK/IWR6843ISK-ODS, the bundled TI user guide lists S1.1 through S1.6
  as Off, Off, On, On, Off, don't-care (`00110X`). Confirm the board revision
  before changing switches; SOP state is sampled at reset.
- Radar command UART known or discoverable, such as `COM4` on Windows or
  `/dev/ttyUSB0` on Linux.
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
  "numpy<2" scipy pyserial "scikit-learn==1.6.1" \
  "pyqtgraph>=0.13.7,<0.15" "PySide6>=6.7,<7" "PyOpenGL>=3.1.7"
```

On Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install `
  'numpy<2' scipy pyserial 'scikit-learn==1.6.1' `
  'pyqtgraph>=0.13.7,<0.15' 'PySide6>=6.7,<7' 'PyOpenGL>=3.1.7'
```

These package bounds are the tested installation baseline. The scikit-learn
pin matches the version that serialized the bundled
`calibration.joblib`; using another version can produce a model-persistence
warning or incompatible behavior, as described in scikit-learn's
[model-persistence guidance](https://scikit-learn.org/stable/model_persistence.html).
The application itself enforces the PyTorch range but does not enforce the
other package versions at runtime. Use the same commands when recreating an
environment to minimize numerical differences.

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

Off-Jetson CPU drone/not-drone classification requires PyTorch 2.6 or newer and
below version 3. Jetson CUDA classification uses TensorRT and does not require
PyTorch. Combined point-cloud + micro-Doppler mode asks whether to enable
classification, with **no** as the default. Install Jupyter as well when using
`machinelearning/classification.ipynb`:

```bash
.venv/bin/python -m pip install jupyter matplotlib "torch>=2.6,<3"
```

```powershell
.venv\Scripts\python -m pip install jupyter matplotlib 'torch>=2.6,<3'
```

PyTorch builds depend on the operating system, Python version, and whether CPU
or CUDA acceleration is required. Check the official
[PyTorch installation selector](https://pytorch.org/get-started/locally/) and
use its platform-specific command when it differs from the generic command
above. The notebook stores outputs from its last Colab training run, but opening
it does not execute any cell. Its first code cell mounts Google Drive and should
be skipped outside Colab. Training starts only when the data and training cells
are run manually. Off Jetson, use `main/run.py --no-classification` when capture is
needed without PyTorch.

The notebook uses only native IWR6843 feature captures. Place UAV capture
JSONL files under `dataset/uav/` and tracked non-UAV target captures under
`dataset/others/`. Capture with `--display-update-every 1`; processed JSONL
records save the exact two-channel, three-range-bin `classification_feature`
consumed by live inference. Training creates non-overlapping 48-frame windows
and uses a stratified 70%/15%/15% window-level split. Use
`--no-classification` while collecting the initial training set so obsolete
model artifacts do not block capture startup.

On Jetson, `--classification-device auto` resolves to the required CUDA
backend. Training and ONNX export happen in Colab or on the training
workstation, not on the Jetson. The notebook exports to
`training_output/micro_doppler_cnn/`. Copy `model_state.pt`, `model.onnx`,
`manifest.json`, `calibration.joblib`, and `parity.npz` into the repository's
`model_weights/` directory on the Jetson. In the `--system-site-packages`
environment described above, install the Jetson-native TensorRT bindings and
CUDA Python runtime. These package names follow NVIDIA's current
[TensorRT Debian installation](https://docs.nvidia.com/deeplearning/tensorrt/latest/installing-tensorrt/install-debian.html)
and [CUDA Python installation](https://nvidia.github.io/cuda-python/cuda-bindings/latest/install.html)
guides:

```bash
sudo apt-get update
sudo apt-get install tensorrt python3-libnvinfer
.venv/bin/python -m pip install cuda-python
```

The launcher builds and caches a fixed-batch-one TensorRT FP16 engine from the
training-exported `model.onnx` on first use. TensorRT selects tactics for the
detected Jetson GPU, then the launcher validates the engine against the small
training-exported `parity.npz` reference set. A label mismatch
or calibrated-probability error above `5e-3` aborts startup. Later runs verify
the ONNX, parity data, calibration, model card, radar profile, TensorRT/CUDA,
and GPU fingerprints before reusing the engine. PyTorch and ONNX Python
packages are not required on the Jetson. The launcher allows five minutes for
this one-time GPU build. CUDA/TensorRT failure never silently falls back to
CPU; use `--classification-device cpu` only as an explicit diagnostic override
when PyTorch is installed.

Live range and Doppler FFTs use local complex64 SciPy kernels checked against
independent NumPy reference calculations. OS-CFAR is a vectorized local
implementation for real-time performance. The IWR6843ISK-ODS planar-array
coordinate mapping is also local and specific to this radar's antenna layout.
All live views use PyQtGraph with the PySide6 Qt binding. Range and
range-Doppler views use native 2D plot/image items, 3D point clouds use
PyQtGraph OpenGL, and combined mode places the OpenGL point cloud and
micro-Doppler image in one window.

## Recommended: Integrated Run

```powershell
.venv\Scripts\python main/run.py
```

`main/run.py`:

1. Prompts for a display mode unless `--display` is supplied.
2. Prompts for a run duration in minutes. Press Enter for the 5-minute default,
   or enter `0` for an unlimited run. The timed duration begins only after the
   initial clutter-map warm-up and static-scene calibration are complete.
3. Detects serial ports and asks which command UART to use when needed.
4. Starts `livedatacapture.py` with `profile.cfg` and `setup.json` by default.
5. In combined point-cloud + micro-Doppler mode, asks whether to run live CNN
   classification (default: no), then asks whether to save the timestamped
   JSONL under `dataset/uav`, `dataset/others`, or `dataset` (default: dataset).
   Other display modes enable classification by default and retain the
   timestamped `calibrationoutput` output default; pass
   `--no-classification` when that is not wanted. Raw ADC recording is disabled
   unless `--raw-output` is explicitly given.
6. When enabled, runs the trained CNN after 48 valid feature steps for one
   continuously owned tracked target, or the fixed range gate in dedicated
   rotor mode. It displays `DRONE`, `NOT_DRONE`, or `UNKNOWN` and saves the
   calibrated probability.
7. Starts `startup.py` with direct serial radar control and direct UDP DCA1000
   control.
8. Stops hardware control before capture when the duration expires or Ctrl+C
   is pressed.

Choose a display without prompting:

```powershell
python main/run.py --display none
python main/run.py --display range
python main/run.py --display range-doppler
python main/run.py --display point-cloud
python main/run.py --display point-cloud-micro-doppler
```

Temporarily zoom a live display to the first 0.5 m with:

```powershell
python main/run.py --display range --max-range-m 0.5
```

Skip the duration prompt with an explicit value:

```powershell
python main/run.py --duration-minutes 5
python main/run.py --duration-minutes 0
```

The second command runs without a time limit.

Specify the radar UART if automatic detection is wrong:

```powershell
python main/run.py --radar-port COM4
```

```bash
python3 main/run.py --radar-port /dev/ttyUSB0
```

All display modes default to processing every valid frame. Override this with
`--display-update-every N`.

## Hardware Control with `startup.py`

`main/startup.py` configures SDK CLI firmware over the radar command UART and
can configure the DCA1000 over UDP. It does not receive, display, or save ADC
data. For normal capture, use `main/run.py`; invoke `startup.py` directly for
preflight, isolated control testing, or a manual two-terminal workflow. `%`/`#`
comments and legacy unprefixed `hostAngleCalibration` metadata are not sent to
SDK CLI firmware.

### Preflight only

Check the direct-serial inputs without sending device commands:

```powershell
python main\startup.py `
  --radar-backend direct-serial `
  --dca-backend dry-run `
  --preflight-only `
  --radar-port COM4 `
  --radar-baud 115200
```

Preflight validates radar dimensions, two-lane LVDS support, DCA1000 settings,
ports, the SDK profile and its required `sensorStart`, and the ability to bind
the capture address. Do not add `--skip-socket-preflight` unless another
receiver already owns that address or the bind probe is deliberately being
bypassed. When direct serial is selected and `--config` remains at its default,
the `--sdk-profile` (normally `profiles/profile.cfg`) is automatically used as
the frame-dimension source.

A successful preflight reports these states, then the `finally` cleanup moves
the state machine through `stopping` and `stopped` without configuring hardware:

```text
startup state: idle -> configs_loaded
startup state: configs_loaded -> preflight_passed
Startup preflight passed.
startup state: preflight_passed -> stopping
startup state: stopping -> stopped
```

### Isolated hardware-control tests

This command really configures and starts the radar but simulates DCA1000
control. It is useful for UART testing, not raw ADC recording:

```powershell
python main\startup.py `
  --radar-backend direct-serial `
  --dca-backend dry-run `
  --radar-port COM4 `
  --radar-baud 115200
```

Press Ctrl+C to send `sensorStop` and close the UART. To test both real control
paths, select direct UDP as well:

```powershell
python main\startup.py `
  --radar-backend direct-serial `
  --dca-backend direct-udp `
  --radar-port COM4 `
  --radar-baud 115200
```

The second command sends DCA1000 configuration and record commands and starts
the radar, but still has no ADC receiver. Use it only when another receiver is
ready or when intentionally testing the control plane.

### Manual two-terminal capture

Start the receiver in Terminal 1 before configuring the hardware. This example
disables classification so model loading cannot prevent the receiver from
becoming ready:

```powershell
python main\livedatacapture.py `
  --config .\profiles\profile.cfg `
  --setup .\profiles\setup.json `
  --host-ip 192.168.33.30 `
  --data-port 4098 `
  --display range `
  --no-classification `
  --raw-output .\calibrationoutput\manual.bin
```

Terminal 2 configures the hardware. The skip flag is required because Terminal
1 already owns UDP port 4098:

```powershell
python main\startup.py `
  --config .\profiles\profile.cfg `
  --sdk-profile .\profiles\profile.cfg `
  --setup .\profiles\setup.json `
  --radar-backend direct-serial `
  --dca-backend direct-udp `
  --skip-socket-preflight `
  --radar-port COM4 `
  --radar-baud 115200 `
  --radar-command-timeout 10
```

Stop Terminal 2 first so it attempts `sensorStop` and `RECORD_STOP`. Then stop
Terminal 1 so its queued frames drain and raw metadata is finalized.

### Startup states and direct-control overrides

A full successful `startup.py` run transitions through:

```text
configs_loaded
preflight_passed
dca1000_ready
radar_ready
receiver_ready
dca1000_armed
radar_streaming
```

Here `receiver_ready` refers only to `startup.py`'s dry-run capture backend; it
does not prove that `livedatacapture.py` is running. The integrated `run.py`
launcher separately waits for the real receiver, frame processor, requested
display, and optional rotor post-processor before it starts the hardware.

Useful options when invoking `startup.py` directly are:

- `--radar-command-timeout 10`: wait longer for each SDK CLI response.
- `--radar-command-delay 0.05`: wait between SDK CLI commands.
- `--radar-line-ending lf`: use LF instead of the default CRLF.
- `--dca-ip ADDRESS --dca-config-port PORT`: override the DCA1000 control
  endpoint, which defaults to `192.168.33.180:4096`.
- `--dca-timeout 3`: acknowledgement timeout for each DCA command attempt.
- `--dca-retries 5`: retries after the initial DCA command attempt.
- `--readiness-delay 0.5`: delay between DCA record arm and `sensorStart`.
- `--sdk-profile PATH --config PATH`: explicitly align the command profile and
  frame-dimension source.
- `--load-firmware`: require configured MSS/BSS firmware paths during
  preflight. There is no flashing backend; dry-run only reports the paths.

`--capture-backend` currently accepts only `dry-run`. It changes control-plane
state and reports the expected data address, but it never starts a real
receiver.

Show the complete control-plane interface with:

```powershell
python main\startup.py --help
```

## Linux Host Setup

The virtual environment should use `--system-site-packages` on Jetson so it can
see NVIDIA's system TensorRT binding. Grant serial access, then log out and back
in so the new group membership takes effect:

```bash
sudo usermod -a -G dialout $USER
```

Configure the DCA1000-facing interface, replacing `eth0` with the correct
wired interface. If the address is already present, do not add it again:

```bash
sudo ip addr add 192.168.33.30/24 dev eth0
sudo ip link set eth0 up
```

Find and test command-UART candidates:

```bash
ls /dev/ttyUSB* /dev/ttyACM*
.venv/bin/python -m serial.tools.miniterm /dev/ttyUSB0 115200
```

The SDK CLI port normally shows `mmwDemo:/>` after Enter. Exit miniterm before
running the application so it releases the port. A headless integrated example
that avoids GUI and model-runtime requirements is:

```bash
.venv/bin/python main/run.py --radar-port /dev/ttyUSB0 \
  --display none --no-classification
```

## Range, Channel, and Angle Calibration

Interactive choices 7, 8, and 9 run range/channel, azimuth, and elevation
calibration. Use a strong corner reflector at a laser-measured distance, keep
the radar and reflector rigid, and remove competing reflectors from the
configured ±0.20 m search window. Run range/channel calibration before either
angular calibration so the latter can import the operational profile's channel
corrections.

```powershell
python main/run.py --display calibration --calibration-distance-m 1.0
python main/run.py --display azimuth-calibration `
  --calibration-distance-m 1.0 --calibration-angle-deg 0
python main/run.py --display elevation-calibration `
  --calibration-distance-m 1.0 --calibration-angle-deg 0
```

For angular calibration, `--calibration-angle-deg` is the known physical angle
from -60 to +60 degrees; omit it to be prompted, with 0 degrees as the default.
The range distance also prompts when omitted and defaults to 1 m. The launcher
uses `profiles/profile_calibration.cfg` as a source, creates a temporary raw-
LVDS runtime profile, ignores 16 warm-up frames, and collects 64 stable frames
within 90 seconds by default. Override those values with
`--calibration-profile`, `--calibration-search-window-m`,
`--calibration-warmup-frames`, `--calibration-frames`, and
`--calibration-timeout-seconds`.

Hardware is stopped before the result is shown. A JSON report is always written
to `--calibration-output`, or to a timestamped file in `--capture-dir` when no
path is supplied. The operational `--config` is changed only after an explicit
`y` confirmation. Applying a range result replaces its one
`compRangeBiasAndRxChanPhase` command; applying an angular result writes an SDK-
safe `% hostAngleCalibration ...` comment. In both cases, the previous profile
is retained as a timestamped `.bak` file.

Applying range/channel calibration changes the normalized profile fingerprint
used by the CNN artifact contract. Disable classification until the model
bundle has been trained or exported against the updated `profile.cfg`;
otherwise inference startup intentionally rejects the profile/artifact
mismatch. Host-only azimuth/elevation bias metadata is excluded from this
fingerprint because it corrects point-cloud coordinates after the CNN's
range-Doppler feature has been formed.

## Capture Receiver Only

This starts UDP capture without model loading, but does not configure or start
the hardware:

```powershell
python main\livedatacapture.py --no-classification
```

Its defaults are `mmwave.json`, `setup.json`, host `192.168.33.30`, UDP port
4098, no display, an 8,192-datagram receiver queue, a 32-frame processing
queue, and a requested 4 MiB socket receive buffer. The operating system may
grant a different receive-buffer size; the actual value is printed.

Unlike combined mode in `main/run.py`, standalone `livedatacapture.py` enables
classification by default. Omitting `--no-classification` therefore requires a
complete compatible model bundle and a `--config` whose normalized fingerprint
matches it. The repository artifacts match `profiles/profile.cfg`, not
the standalone receiver's default `mmwave.json`.

The integrated launcher exposes all three buffering controls:

```powershell
python main/run.py `
  --socket-recv-buffer 4194304 `
  --packet-queue-size 8192 `
  --processing-queue-size 32
```

Use the same config that programmed the radar:

```powershell
python main\livedatacapture.py `
  --config .\profiles\profile.cfg `
  --setup .\profiles\setup.json `
  --no-classification
```

Both SDK CLI `.cfg` and mmWave Studio `.json` radar configs are supported. XML
is not supported.

Less common direct-receiver controls include:

- `--host-compensation-profile PATH` to import range, channel, and host-angle
  corrections from an operational profile without sending it to the radar;
- `--classification --classification-artifacts PATH` to enable classification
  explicitly and select a non-default artifact directory;
- `--inference-logging --evaluation-label LABEL` to opt into a timestamped
  live-inference evaluation log, or `--inference-log PATH` to select its path;
- `--performance-logging` to write a timestamped processing/resource/detection/
  tracking telemetry log, or `--performance-log PATH` to select its path;
- `--micro-doppler-range-half-width-bins N` to change the dedicated rotor
  range-gate width;
- `--static-detection` to explicitly select the default-enabled static branch;
- `--buffer-size` to set the maximum bytes requested from one UDP datagram and
  `--socket-timeout` to set the interruptible socket-poll interval; and
- `--display-pause` to set the GUI event-loop polling interval.

The defaults are usually appropriate. Use `livedatacapture.py --help` for their
types and current default values.

## Displays

### Range profile

```powershell
python main\livedatacapture.py --display range
```

The plot is average range-FFT magnitude over chirps and receivers. Its X-axis
uses meters when sample rate and frequency slope are available; otherwise it
falls back to bin numbers. The displayed X limit defaults to 10 m:

```powershell
python main\livedatacapture.py --display range --max-range-m 10
python main\livedatacapture.py --display range --max-range-m 0
```

Zero selects the full computed axis.

### Range-Doppler heatmap

```powershell
python main\livedatacapture.py --display range-doppler
```

The Y-axis is centered Doppler-bin index, not calibrated velocity. Processing
uses frame loops as slow time and averages magnitude across TX chirp position
and receivers.

### Point cloud

```powershell
python main\livedatacapture.py --display point-cloud
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
floor, and removes common receiver-gain drift. It applies the threshold to each
update's instantaneous change without temporal smoothing. A cell must exceed
twice its learned noise variation. Direct `livedatacapture.py` use defaults the
additional absolute floor to 0 dB, while the recommended `main/run.py` launcher
passes 3 dB by default. The reference and learned noise estimates remain fixed
after calibration by default, so later scene changes are not absorbed.

Raw changes do not become displayed static targets by themselves. A confirmed
measured dynamic track arms rather than starts handoff; no additional minimum
displacement is required. Static acquisition remains disabled and any older
static track is reset while measured dynamic tracking is healthy. After two
consecutive missing dynamic measurements, a 60-frame handoff window opens.
A static local maximum or cluster can take over only if it is within 0.4 m of
the last dynamic position. Because the target identity was already confirmed by
the dynamic tracker, the first qualifying static return completes the handoff;
it does not need three additional temporal hits. When multiple clusters pass
the gate, the one nearest that last position is tried. One point is sufficient
in each frame by default because local-maximum filtering has already reduced a
reflector to one candidate. Until that return arrives, a predicted static marker
is held at the last measured dynamic position. It is continuity state rather
than a new radar point and remains marked as predicted in processed output.
The validated target is protected by ±2 range, azimuth, and elevation cells
and remains visible after stopping. The static tracker uses a zero-velocity
model with stronger position smoothing and coasts through up to 60 consecutive
candidate misses. Protection is released after the same 60-miss interval. A
removed target is absorbed only when a nonzero
`--static-background-update-rate` has enabled adaptation; the default fixed
reference does not absorb it. Only the exact points in the validated cluster
are shown as orange squares; its center is shown as a cyan diamond. Transient
and static-only clutter is suppressed.

Static angle processing remains full rate: it runs for every processed
point-cloud update with the complete 32-by-32 angle FFT. Detection uses the
instantaneous corrected change without a temporal EMA. Objects present during
calibration become part of the reference and are not reported as changes.
Calibration counts updates that actually reach point-cloud processing, so
packet/frame loss makes it take longer than 180 physical frames.

Both paths apply a 10 m radial range and a ±60-degree azimuth/elevation field
of view. Spatial DBSCAN defaults to a 0.4 m neighborhood and two-point minimum.
Dynamic points remain magnitude-colored, and red crosses mark their cluster
centers. Coordinates are X left/right, Y forward, and Z elevation. The plot
spans 0 to 10 m forward and approximately -8.66 to +8.66 m across X and Z.
Dynamic point color is fixed from 60 to 120 dB. The supported ODS path applies
range/channel and azimuth/elevation corrections parsed from the profile, but
the angle-FFT coordinates remain diagnostic rather than metrology-grade.

The software OS-CFAR requested probability of false alarm defaults to
`1e-3` per axis.

Use `--max-range-m` to change the shared range limit for any display.
`--point-cloud-fov-deg` changes the point-cloud half-FOV when invoking
`livedatacapture.py` directly; the integrated `main/run.py` launcher currently uses
the receiver's ±60-degree default.

Use `--cluster-eps-m` and `--cluster-min-samples` to tune DBSCAN. Set
`--cluster-eps-m 0` to disable cluster-center generation:

```powershell
python main/run.py --display point-cloud --cluster-eps-m 0.4 --cluster-min-samples 3
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
and a 3 dB absolute floor when using `main/run.py`. Direct
`livedatacapture.py` defaults that floor to 0 dB. Set a nonzero
`--static-background-update-rate` to opt into adaptation.
Set `--static-cluster-min-samples 3` to require three spatially adjacent
candidates in the qualifying handoff update instead of one local maximum. The
learned noise threshold can make the effective threshold higher in unstable
cells.
Disable the static branch without changing the moving-target path with:

```powershell
python main/run.py --display point-cloud --no-static-detection
```

### Point cloud with micro-Doppler

```powershell
python main/run.py --display point-cloud-micro-doppler
```

This mode places the 3D point cloud and a rolling 150-window micro-Doppler
spectrogram side by side. It reuses one Doppler cube for dynamic and static
tracking, then uses the selected track range to gate the original range-FFT
cube. The micro-Doppler path calculates a separate slow-time FFT for each TX
slot using 64-loop Hann windows, a 32-loop hop, and a 128-point FFT. The TX
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

Use the dedicated fixed-range mode when blade-flash visibility or RPM is more
important than simultaneous 3D tracking. It skips angle processing but retains
the active profile's three-TX TDM schedule and processes each TX independently:

```powershell
python main/run.py --display micro-doppler --micro-doppler-range-m 2.15 `
  --rotor-blades 2 --rotor-count 1 --rotor-radius-m 0.05 `
  --rotor-rpm-min 1000 --rotor-rpm-max 10700 `
  --raw-output .\calibrationoutput\rotor.bin
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
`main/run.py` prompts for the gate and defaults to 2.15 m. Its default half width is
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

The integrated launcher saves processed data by default for normal capture
modes; calibration modes write their separate JSON report instead. Override a
normal capture path with:

```powershell
python main/run.py --processed-output .\calibrationoutput\processed.jsonl
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
- the selected micro-Doppler range gate;
- the `[2, 64]` three-range-bin `classification_feature` used to build native
  CNN training windows, or `null` when no valid target owns the frame;
- `classification`: `label`, calibrated `p_drone`, threshold, status/reason,
  and valid history length.

`not_drone` represents the native examples placed in `dataset/others/`; its
coverage is therefore limited to the negative objects and conditions captured
there. Classification stays `unknown`
until 48 consecutive valid target frames are available and resets when target
quality or identity is lost.

In point-cloud and combined point-cloud + micro-Doppler modes, live
classification is shown in the PyQt point-cloud status panel. It reports the
warm-up history while waiting, then the label, calibrated `p_drone`, and
decision threshold. Per-update classifications are not printed to the
terminal; startup, TensorRT build progress, and errors remain terminal output.

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

## Optional Live Inference Evaluation Log

Inference evaluation logging is off by default. When classification is enabled,
the integrated launcher asks:

```text
Enable live inference evaluation log? [y/N]:
```

Blank input keeps it disabled and no evaluation file or ground-truth prompt is
created. Answer `y` and select `drone`, `not_drone`, or `unlabeled` as the truth
for the complete run. For unattended use:

```powershell
python main/run.py `
  --classification `
  --inference-logging `
  --evaluation-label drone
```

The default file is `log/live_inference_<timestamp>.jsonl`. Select another
location with:

```powershell
python main/run.py `
  --classification `
  --inference-log .\log\outdoor_non_drone.jsonl `
  --evaluation-label not_drone
```

Supplying `--inference-log` enables the feature. Do not combine it with
`--no-inference-logging`. Logging also cannot be enabled with
`--no-classification`. Direct `livedatacapture.py` never prompts and uses the
same explicit flags.

The JSONL begins with metadata describing the run, radar frame period,
classifier/backend, threshold, profile, artifact hashes, and compatibility
fingerprint. Every inference attempt then records:

- frame index, capture timestamp, processing timestamp, and elapsed times;
- `p_drone`, predicted label, predicted-class confidence, threshold, status,
  reason, and correctness when labeled;
- classification latency and selected target range/source; and
- the classifier's 48-step history before and after the attempt.

The history metadata identifies append/reset operations, empty/warming/ready
state, reset reason and timestamp, discarded steps, reset sequence, history
generation, and frames since reset. Target loss/change, invalid feature/range,
invalid probability, and inference-error resets therefore remain distinguishable.
Repeated reset requests while history is already empty are recorded but do not
claim discarded frames.

On orderly shutdown, a run summary provides ready-decision accuracy, drone and
non-drone class accuracy, confusion matrix, precision/recall/F1, balanced
accuracy, coverage, operational correctness, class and unknown durations,
confidence/latency distributions, Brier score, log loss, temporal segments,
label changes, and reset/recovery statistics. Drone class accuracy is drone
recall (`TP / (TP + FN)`); non-drone class accuracy is non-drone recall
(`TN / (TN + FP)`). Unknown/warm-up attempts are excluded from ready-decision
accuracy but included in coverage and operational correctness.

An aggregate summary follows the run summary. It uses only completed labeled
logs in the same directory with matching format, model artifacts, profile,
threshold, backend, and precision. Both class scores, AUROC, and PR-AUC require
compatible runs for both truth classes. Unsupported values are `null` with an
explanation. `unlabeled` runs retain confidence, duration, latency, reset, and
coverage measurements but are excluded from accuracy aggregation.

Each evaluation line is flushed immediately. A crash can omit the final
summaries, but all complete inference lines remain readable; incomplete logs
are not included in later aggregate scores. The 48-frame windows overlap, so
frame results are correlated. Use the session-majority result alongside
frame-weighted metrics when comparing runs of different lengths.

## Optional Performance Telemetry Log

The integrated launcher asks on each capture run:

```text
Enable performance telemetry log? [y/N]:
```

Blank input keeps it disabled. Direct `livedatacapture.py` remains
non-interactive for scripted capture.

Enable machine-readable runtime telemetry with:

```powershell
python main/run.py `
  --performance-logging `
  --performance-sample-interval 1
```

The default path is `log/performance_<timestamp>.jsonl`. Supplying
`--performance-log PATH` also enables logging. Direct `livedatacapture.py`
accepts the same options. Do not combine a path with
`--no-performance-logging`.

The version-1 JSONL contains three record types:

- `frame`: capture-to-completion latency, per-stage processing time, dynamic
  point/cluster counts, static candidates and validation state, selected track
  state/source/position/range/speed/age/hits/misses, and the latest classifier
  result;
- `resource_sample`: frame-processor CPU and resident memory, whole-system CPU
  and memory, plus NVIDIA GPU utilization/memory/temperature/power when
  `nvidia-smi` exposes them; Jetson sysfs supplies GPU utilization, frequency,
  and temperature as a fallback; and
- `run_summary`: p50/p95/max timing and resource distributions, candidate-frame
  rate, measured/predicted/absent track coverage, acquisitions, losses, source
  switches, and longest continuous track.

Resource sampling defaults to once per second to keep overhead low. A `null`
GPU field means that the platform did not expose that measurement; on Jetson,
GPU memory is shared system RAM and may not have a separate value. Detection
candidate rate and track coverage are operational metrics, not accuracy. To
compute precision/recall, localization error, MOTA, or ID metrics, align the
logged frame/timestamps with external ground-truth detections or trajectories.

## Optional Raw Frames

```powershell
python main\livedatacapture.py `
  --raw-output .\calibrationoutput\test_capture.bin
```

Only valid complete frames are written, without DCA1000 headers. The default
metadata path is `test_capture.bin.json`; override it with `--raw-metadata`.
The sidecar is written during clean shutdown.

Raw recording is opt-in because its files are much larger than processed JSONL
output. Files have no size limit. A `main/run.py` session is limited to 5 minutes by
default, but `livedatacapture.py` alone and `main/run.py --duration-minutes 0` run
until stopped.

## Logging and Statistics

Each run writes terminal output to a new
`log\<run-id>_livedatacapture.log` file by default. When performance or
inference logging is enabled, its default filename uses the same run ID, for
example `<run-id>_performance.jsonl` and `<run-id>_live_inference.jsonl`.
An explicit path can still be selected with:

```powershell
python main\livedatacapture.py `
  --log-file .\log\capture_run.log
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
processing error counters change. A normal clean shutdown prints capture,
processing, static, and DSP timing summaries. It additionally prints the
post-processing summary only for classified dedicated-rotor capture and the
display summary only when a visible display was selected:

- a capture summary with total, valid, invalid, and queued frames;
- a processing summary with queued and completed frames;
- when active, a post-processing summary with completed frames and queue
  high-water mark;
- when active, a display summary with physically rendered updates,
  latest-update replacements, and frames not rendered over total assembled
  frames;
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

## Hardware-Control Troubleshooting

### DCA1000 `SYSTEM_CONNECT` timeout

- Confirm the host address is `192.168.33.30/24` and the DCA1000 address is
  `192.168.33.180`.
- Confirm the direct Ethernet connection, board power, and UDP port 4096.
- Close mmWave Studio or other TI DCA tools that may own the control port.
- Increase `--dca-timeout` or `--dca-retries` only after checking the network.

### Socket preflight bind failure

If `livedatacapture.py` already owns `192.168.33.30:4098`, pass
`--skip-socket-preflight` to `startup.py`. Otherwise, the error usually means
the host address is not assigned to the selected interface or another receiver
already owns the port.

### Radar serial timeout

- Verify compatible SDK CLI firmware, functional SOP mode, and a reset after
  changing the mode.
- Select the command UART rather than a logger or high-speed data UART.
- Close miniterm and any other program using the port.
- Confirm the baud rate and try `--radar-command-timeout 10`.
- Try `--radar-line-ending lf` if the prompt does not respond to CRLF.

### Capture window remains empty

- Confirm `livedatacapture.py` reached its listening/ready message before
  hardware startup.
- Confirm `profile.cfg` enables raw LVDS ADC streaming.
- Ensure capture dimensions and SDK commands come from the same profile.
- Check packet-loss, byte-gap, stream-resync, and processing-drop counters.

## Packet-Loss Troubleshooting

- Connect the PC directly to DCA1000 over wired Ethernet.
- Ensure `packetDelay_us` in `setup.json` matches the DCA1000 configuration.
- The repository default is 100 us. If loss remains after host buffering is
  fixed, test a higher delay within the supported range and ensure the DCA1000
  FPGA error LED remains off.
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

Press Ctrl+C. With `main/run.py`, the startup process is stopped first, followed by
the capture process. The frame processor ignores the parent SIGINT and stops
only after the queue sentinel, so every frame queued before shutdown is
drained. Clean capture shutdown closes the socket, child processes, queues,
raw file, metadata sidecar, and log file.

Show all receiver options with:

```powershell
python main\livedatacapture.py --help
```
