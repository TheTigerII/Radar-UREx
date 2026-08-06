# Mini4 PMM Tracking User Guide

This software operates an IWR6843ISK-ODS with a DCA1000EVM and tracks one PMM
target from 0.3 to 20 m. The live application does not determine whether the
target is a drone, bird, fan, foliage, or another periodic reflector. An
optional offline notebook can train a UAV-versus-other classifier from labelled
captures, but that model is not used by the live application.

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
python -m pip install scikit-learn matplotlib onnx jupyter
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

The launcher starts capture first, configures and arms the DCA1000, sends the
fixed 20 m profile to the radar, and then sends `sensorStart`. It prompts for a
duration; Enter selects five minutes and `0` runs until Ctrl+C.
By default it selects the serial device whose USB description is
`CP2105 Dual USB to UART Bridge Controller - Enhanced COM Port`. Use
`--radar-port` only to override that selection.

Useful examples:

```bash
python run.py --radar-port /dev/ttyUSB0 --display none \
  --duration-minutes 60

python run.py --radar-port /dev/ttyUSB0 --display range-doppler \
  --raw-output rawdatacapture/captures/session.bin

python run.py --radar-port /dev/ttyUSB0 \
  --pmm-detection-threshold 750 \
  --pmm-background-calibration-seconds 30
```

The display choices are `none`, `range`, `range-doppler`, `point-cloud`, and
`combined`. All graphical modes use PyQtGraph; the 3D modes also use its
OpenGL widgets. Use `none` for unattended or headless operation.

Keep the monitored area target-free during the first 30 seconds. The status is
`calibrating` during that period and no detection is emitted.

## Runtime settings

- `--pmm-background-calibration-seconds`: target-free calibration time.
- `--pmm-max-target-speed-m-s`: dynamic-programming motion limit.
- `--pmm-folding-size-min` and `--pmm-folding-size-max`: tested periods.
- `--pmm-detection-threshold`: linear PMM score threshold; default 750.
- `--pmm-history-seconds`: retained range-time history.
- `--pmm-provisional-frames`: observations before tentative tracking.
- `--pmm-confirmation-window-frames` and `--pmm-confirmation-hits`: confirmation
  rule.
- `--pmm-coast-frames`: missing observations allowed after confirmation.
- `--display-update-every`: reduce drawing frequency without reducing DSP rate.
- `--processing-queue-size` and `--packet-queue-size`: bounded queue capacities.

Show all options with:

```bash
python run.py --help
python rawdatacapture/livedatacapture.py --help
```

## Output

The default processed filename is
`rawdatacapture/captures/pmm_capture_<timestamp>.jsonl`. Each run begins with a
metadata record, followed by frame records. Look for:

- `pmm_tracking.state`
- `pmm_tracking.label`, always `PMM target`
- `pmm_tracking.range_m`
- `pmm_tracking.radial_velocity_m_s`
- `pmm_tracking.azimuth_deg`
- `pmm_tracking.elevation_deg`
- `pmm_tracking.pmm_score`
- `pmm_tracking.folding_size`
- `doppler_time_db`, the rolling target-gated Doppler-Time history
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

Run the cells in order when training is intended. The committed notebook is
unexecuted and does not contain weights or metrics. It validates that all input
files share the same Mini4 profile, feature pipeline, and capture threshold;
extracts non-overlapping 36-by-64 Doppler-Time segments; applies the recorded
PMM threshold; balances the classes; and creates deterministic stratified 70%,
15%, and 15% training, validation, and locked-test partitions. Normalization is
fitted only on the training partition.

Training uses a two-layer, 128-hidden-unit LSTM for 100 epochs with Adam at
`5e-5` and batch size 10. Validation loss chooses the retained state. The final
test report includes accuracy, precision, recall, F1, a confusion matrix, ROC,
and ROC AUC.

After the export cell runs, the following ignored artifacts are written:

```text
training_output/mmhawkeye_lstm/model_state.pt
training_output/mmhawkeye_lstm/model.onnx
training_output/mmhawkeye_lstm/manifest.json
```

The ONNX model accepts float32 `doppler_time` tensors shaped `[batch, 36, 64]`
and returns two logits ordered as `other`, then `uav`. Exporting the model does
not install it into the live capture path.

## Replay

Replay a recorded raw ADC file without hardware:

```bash
python rawdatacapture/replay_pmm.py \
  rawdatacapture/captures/session.bin \
  --config rawdatacapture/profile-mini4-20m.cfg \
  --output rawdatacapture/captures/session_replay.jsonl
```

Use `python rawdatacapture/replay_pmm.py --help` for frame limits and PMM
settings.

## Hardware preflight

Validate the fixed profile and setup without sending hardware commands:

```bash
python rawdatacapture/startup.py \
  --config rawdatacapture/profile-mini4-20m.cfg \
  --sdk-profile rawdatacapture/profile-mini4-20m.cfg \
  --setup rawdatacapture/setup.json \
  --preflight-only --skip-socket-preflight
```

## Troubleshooting

If preflight rejects the profile, use the repository
`profile-mini4-20m.cfg` unchanged. The processing dimensions are intentionally
fixed.

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
