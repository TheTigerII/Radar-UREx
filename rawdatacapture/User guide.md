# Mini4 PMM Tracking User Guide

This software operates an IWR6843ISK-ODS with a DCA1000EVM and tracks one PMM
target from 0.3 to 20 m. The live application does not determine whether the
target is a drone, bird, fan, foliage, or another periodic reflector unless a
compatible trained classifier is explicitly enabled. Real-time classification
is off by default.

## Install

On the Jetson Orin Nano:

```bash
python3 -m venv --system-site-packages .venv
source .venv/bin/activate
python -m pip install \
  "numpy<3" "scipy<2" \
  "pyqtgraph>=0.14,<0.15" "PySide6>=6.8,<7" "PyOpenGL>=3.1,<4" \
  pyserial
```

GPU inference requires JetPack's TensorRT installation, the `trtexec` command,
and the `tensorrt` and `cuda.bindings.runtime` Python modules. When
classification starts without an engine, the application runs the equivalent
of this command automatically:

```bash
trtexec --onnx=model_weights/model.onnx \
  --saveEngine=model_weights/generated/model.fp16.engine \
  --fp16 --shapes=doppler_time:1x36x64 --skipInference
```

The full `trtexec` output is printed live so the build is visibly progressing.
After a successful build, the engine is stored as
`model_weights/generated/model.fp16.engine` and loaded for classification.
One existing hardware-named `.engine` file in that directory is also accepted.
Live classification is TensorRT GPU-only and has no PyTorch or ONNX Runtime
fallback. TensorRT engines are device-specific; delete the generated engine to
force a rebuild after changing the Jetson, TensorRT version, or ONNX model.

The radar must run SDK CLI-compatible firmware. Connect the IWR6843 command
UART, connect the DCA1000 Ethernet interface, and configure that interface:

```bash
sudo ip addr add 192.168.33.30/24 dev eth0
sudo ip link set eth0 up
```

Replace `eth0` with the DCA1000-facing interface. The default board address is
`192.168.33.180`; UDP data is received on `192.168.33.30:4098`.

To use `training.ipynb`, install a PyTorch build appropriate for the training
machine, then install the remaining notebook dependencies:

```bash
python -m pip install scikit-learn matplotlib onnx onnxscript jupyter
```

Jetson users should use NVIDIA's supported PyTorch package rather than assuming
the default PyPI wheel supports the installed JetPack release. Training may
also be run on a separate CPU or CUDA workstation.

## Run

From the repository root:

```bash
source .venv/bin/activate
python run.py --display combined
```

The launcher starts capture first, waits for its worker and UDP listener to be
ready, configures the DCA1000, sends the selected SDK CLI profile to the radar,
arms DCA1000 recording, and then sends `sensorStart`. The default profile is
`rawdatacapture/profile-mini4-20m.cfg`; normal PMM capture accepts another
`.cfg` path only when its acquisition values match the Mini4 contract. The
launcher prompts for a duration; Enter selects five minutes and `0` runs until
Ctrl+C.

Without `--radar-port`, the launcher selects the only serial device whose
description contains `CP2105`, `Enhanced`, and `COM Port` (case-insensitive).
If that match is unavailable but exactly one serial port is enumerated, it
uses that port; with multiple ports it prompts for a selection. If no ports are
enumerated, the prompted fallback is `/dev/ttyUSB0` on Linux or `COM4` on
Windows. Use `--radar-port` to override discovery.

For normal capture modes the launcher asks whether to enable real-time UAV
classification. Press Enter to keep it off. When combined display option 5 is
selected and `--processed-output` was not supplied, it then asks where to save
the processed JSONL:

```text
1. dataset/uav
2. dataset/other
3. dataset
```

Press Enter to select `dataset`. The class-specific destinations should be
used only for controlled, independently labelled data collection.

Useful examples:

```bash
python run.py --radar-port /dev/ttyUSB0 --display none \
  --duration-minutes 60

python run.py --radar-port /dev/ttyUSB0 --display range-doppler \
  --raw-output rawdatacapture/captures/session.bin

python run.py --radar-port /dev/ttyUSB0 \
  --pmm-detection-threshold 750 \
  --pmm-background-calibration-seconds 30

python run.py --display combined --classification \
  --model-weights-dir model_weights \
  --dataset-destination dataset
```

The normal display choices are `none`, `range`, `range-doppler`, `point-cloud`,
and `combined`. The launcher also provides the dedicated `calibration`,
`azimuth-calibration`, and `elevation-calibration` modes described below. All
graphical modes use PyQtGraph; the 3D normal modes also use its OpenGL widgets.
Use `none` for unattended or headless operation.

Keep the monitored area target-free during the first 30 seconds. The status is
`calibrating` during that period and no detection is emitted. This period now
calibrates both the range-PMM background and the azimuth/elevation A-PMM
backgrounds used by the paper's projection subtraction.

## Runtime settings

- `--pmm-background-calibration-seconds`: target-free calibration time.
- `--pmm-max-target-speed-m-s`: dynamic-programming motion limit.
- `--pmm-folding-size-min` and `--pmm-folding-size-max`: tested periods;
  defaults are 2 and 32, and the maximum cannot exceed 32 because every
  candidate must retain at least two folding rows.
- `--pmm-detection-threshold`: linear PMM score threshold; default 750.
- `--pmm-history-seconds`: retained range-time history.
- `--pmm-provisional-frames`: observations before tentative tracking.
- `--pmm-confirmation-window-frames` and `--pmm-confirmation-hits`: confirmation
  rule.
- `--pmm-coast-frames`: missing observations allowed after confirmation.
- `--display-update-every`: reduce drawing frequency without reducing DSP rate.
- `--processing-queue-size` and `--packet-queue-size`: bounded queue capacities.
- `--classification` and `--no-classification`: explicitly enable or disable
  real-time classification without an interactive prompt.
- `--model-weights-dir`: directory containing compatible `model.onnx` and
  `manifest.json` artifacts, plus `model.onnx.data` when declared by the
  manifest.
- `--dataset-destination`: select `dataset`, `uav`, or `other` without the
  combined-display save prompt.

Show all options with:

```bash
python run.py --help
python rawdatacapture/livedatacapture.py --help
```

## Hardware calibration

The three calibration display modes use
`profiles/profile_calibration.cfg` to create a temporary raw-LVDS runtime
profile; that source file and the operational profile are not changed during
measurement. Place a strong reflector at a laser-measured distance, keep the
rest of the calibration region clear, and run range/RX-channel calibration
first:

```bash
python run.py --display calibration --calibration-distance-m 1.0
```

The defaults are a 1 m target distance, a ±0.20 m search window, 16 warm-up
frames, 64 accepted frames, and a 90-second timeout. A stable result produces a
JSON report under `rawdatacapture/captures/` unless
`--calibration-output` supplies another path. The launcher then asks before
updating `compRangeBiasAndRxChanPhase` in the operational profile. Applying a
result creates a timestamped `.bak` copy first.

After range/RX calibration has been applied, measure azimuth and elevation at
a known tripod angle:

```bash
python run.py --display azimuth-calibration \
  --calibration-distance-m 1.0 --calibration-angle-deg 0

python run.py --display elevation-calibration \
  --calibration-distance-m 1.0 --calibration-angle-deg 0
```

Angular calibration loads the operational profile's range and RX-channel
corrections while measuring. Applying the result writes a host-only
`% hostAngleCalibration` comment and preserves the other angular bias. These
modes produce a calibration report rather than normal PMM JSONL or raw capture
output. Use the `--calibration-search-window-m`,
`--calibration-warmup-frames`, `--calibration-frames`, and
`--calibration-timeout-seconds` options to override their defaults.

## Output

The generated processed filename is `pmm_capture_<timestamp>.jsonl`. Its
default directory is `rawdatacapture/captures/` for normal modes other than
`combined`; combined mode prompts for `dataset/`, `dataset/uav/`, or
`dataset/other/` and defaults to `dataset/`. Supplying
`--dataset-destination` uses that dataset directory in any normal mode, while
`--processed-output` supplies the exact output path. Each run begins with a
metadata record followed by update records. Look for:

- `pmm_tracking.state`
- `pmm_tracking.label`, always `PMM target`
- `pmm_tracking.range_m`
- `pmm_tracking.radial_velocity_m_s`
- `pmm_tracking.azimuth_deg`
- `pmm_tracking.elevation_deg`
- `pmm_tracking.pmm_score`
- `pmm_tracking.folding_size`
- `doppler_time_db`, the rolling target-gated Doppler-Time history
- `classification`, when enabled, containing status, label, confidence,
  probabilities, PMM gate values, history length, and inference latency
- `diagnostics`

States progress through `calibrating`, `searching`, `tentative`, `confirmed`,
`coasting`, and `lost`. Only `confirmed` represents a confirmed PMM track; it
is not a drone-identification result.

When `--raw-output` is supplied, valid complete ADC frames are also saved to
the given `.bin` file with a JSON metadata sidecar. This is the preferred input
for repeatable threshold tuning.

## Train the offline UAV classifier

The training notebook consumes processed JSONL, not raw `.bin` recordings.
Create the ignored dataset directories and record each controlled session
directly under its known class:

```bash
mkdir -p dataset/uav dataset/other

python run.py --display none \
  --processed-output dataset/uav/uav_session_001.jsonl

python run.py --display none \
  --processed-output dataset/other/bird_session_001.jsonl
```

Use `dataset/uav/` only when a UAV is independently known to be the target.
Use `dataset/other/` for controlled distractors such as birds, kites, balloons,
fans, and moving vegetation. Keep the area target-free during the initial
background calibration for every recording. Do not relabel an uncertain track
based only on its PMM score.

The notebook requires at least seven quality-gated, non-overlapping 3.6-second
segments in each class; this is only a technical minimum, not a sufficient
operational dataset. Collect substantially more sessions across target types,
ranges, motions, sites, and environmental conditions.

Start Jupyter from the repository root and open `training.ipynb`:

```bash
source .venv/bin/activate
jupyter lab training.ipynb
```

For local Jupyter, skip the first code cell: it is a Google Colab bootstrap
that imports `google.colab`, mounts Drive, changes to a Colab-specific path,
and installs export packages. Start with the imports cell after installing the
dependencies above. In Colab, adjust the Drive path in the bootstrap cell and
then run all cells in order. The committed notebook is unexecuted and does not
contain weights or metrics. It validates that all input files share the same
Mini4 profile, feature pipeline, and capture threshold;
extracts non-overlapping 36-by-64 Doppler-Time segments; applies the recorded
PMM threshold; estimates the center-bin DC baseline only from frames whose body
peak is away from DC; aligns the pre-removal body peak; balances the classes;
and creates deterministic stratified 70%, 15%, and 15% training, validation,
and locked-test partitions. Normalization is fitted only on the training
partition.

Feature version `mini4-pmm-tracking-v7` uses the paper's tracking clutter
suppression for both range and angle: non-demeaned Doppler spectra are folded
first, followed by projection subtraction of the calibrated Range-Time-PMM and
Angle-Time-PMM backgrounds. It also searches folding sizes 2 through 32.
Earlier captures and the committed v4 model weights cannot reproduce this
input and must not be relabelled as v7. Record a new v7 dataset and retrain
before enabling classification.

Training uses a two-layer, 128-hidden-unit LSTM for 100 epochs with Adam at
`5e-5` and batch size 10. Validation loss chooses the retained state. The final
test report includes accuracy, precision, recall, F1, a confusion matrix, ROC,
and ROC AUC.

After the export cell runs, the following ignored artifacts are written:

```text
training_output/mmhawkeye_lstm/model_state.pt
training_output/mmhawkeye_lstm/model.onnx
training_output/mmhawkeye_lstm/model.onnx.data  # when external data is emitted
training_output/mmhawkeye_lstm/manifest.json
```

The ONNX model accepts float32 `doppler_time` tensors shaped `[batch, 36, 64]`
and returns two logits ordered as `other`, then `uav`. Exporting the model does
not install it into the live capture path. Install the ONNX model and its
manifest after a successful training and evaluation run:

```bash
mkdir -p model_weights
cp training_output/mmhawkeye_lstm/model.onnx* model_weights/
cp training_output/mmhawkeye_lstm/manifest.json model_weights/manifest.json
```

Then select `yes` at the classification prompt, or use `--classification` for
a non-interactive launch. Use `--no-classification` to explicitly disable it.
The launcher expects `model_weights/model.onnx`,
`model_weights/manifest.json`, and any external tensor-data file declared in
the manifest. It exits before hardware startup if classification is enabled
and an artifact is missing. The worker also rejects artifacts whose profile
fingerprint, feature fingerprint, feature version, input shape, label order,
normalization, architecture, or ONNX interface does not match the running
pipeline.

Once a complete 36-frame Doppler-Time history and 36 associated PMM scores are
available, the point-cloud and combined status text can include `UAV` or
`OTHER` with confidence. The quality gate uses the maximum PMM score in that
window; before the history is complete it reports warm-up, and a window whose
maximum is below threshold remains unknown. Classification metadata and
per-frame results are also written to processed JSONL. This output is a model
prediction, not independent ground truth.

## Replay

Replay a recorded raw ADC file without hardware:

```bash
python rawdatacapture/replay_pmm.py \
  rawdatacapture/captures/session.bin \
  --config rawdatacapture/profile-mini4-20m.cfg \
  --output rawdatacapture/captures/session_replay.jsonl
```

Replay emits one `pmm_replay` JSON object per complete raw frame and rejects an
incomplete trailing frame. Use `python rawdatacapture/replay_pmm.py --help` for
the available output path, background-calibration duration, and threshold
options. The replay CLI does not provide a frame-limit option.

## Hardware preflight

Run the startup configuration preflight without sending hardware commands:

```bash
python rawdatacapture/startup.py \
  --config rawdatacapture/profile-mini4-20m.cfg \
  --sdk-profile rawdatacapture/profile-mini4-20m.cfg \
  --setup rawdatacapture/setup.json \
  --preflight-only --skip-socket-preflight
```

This checks the parsed dimensions, two-lane LVDS support, DCA1000 and serial
settings, and the SDK command file. Because `--skip-socket-preflight` is shown,
it does not test the UDP bind; omit that flag when the host interface is
configured and the port should be checked. The normal capture worker performs
the stricter Mini4 acquisition-contract validation before live reception.

## Troubleshooting

If normal capture rejects a profile, compare its acquisition dimensions,
timing, slope, sample rate, and `(1, 4, 2)` TX order with the repository
`profile-mini4-20m.cfg`. Those acquisition values are intentionally fixed;
range/RX compensation and host angular-bias values may differ after hardware
calibration.

If packet loss increases, verify the dedicated Ethernet address, close other
DCA1000 tools, increase the OS receive buffer if permitted, and avoid slow
storage on the capture path. Queue occupancy and drop counters appear in every
processed record.

If there are frequent false PMM tracks, collect target-free recordings at the
actual site and tune the threshold upward. If a Mini 4 Pro is not confirmed,
collect hover and slow-flight recordings at known distances and inspect the raw
and subtracted scores before lowering the threshold.

If the display is slow but processing remains healthy, increase
`--display-update-every` or select `none`.

If training reports that no usable segments passed the PMM gate, verify that
each session continued long enough after calibration, inspect
`pmm_tracking.pmm_score` against the metadata threshold, and confirm that the
target generated a stable tracked history. Do not lower the gate solely to make
the notebook accept a dataset.

If training rejects mixed fingerprints or thresholds, do not resize or merge
the captures. Re-record them with the same profile and PMM settings, or train
separate models for the incompatible capture contracts.

If live classification fails during startup, confirm that
`model_weights/model.onnx`, its external data when present, and
`model_weights/manifest.json` came from this repository's current notebook and
capture contract. Do not bypass a fingerprint mismatch by editing checkpoint
metadata.
