# Mini4 PMM Tracking User Guide

This software operates an IWR6843ISK-ODS with a DCA1000EVM and tracks one PMM
target from 0.3 to 20 m. It does not determine whether the target is a drone,
bird, fan, foliage, or another periodic reflector.

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

## Run

From the repository root:

```bash
source .venv/bin/activate
python run.py --radar-port /dev/ttyUSB0 --display combined
```

The launcher starts capture first, configures and arms the DCA1000, sends the
fixed 20 m profile to the radar, and then sends `sensorStart`. It prompts for a
duration; Enter selects three minutes and `0` runs until Ctrl+C.

Useful examples:

```bash
python run.py --radar-port /dev/ttyUSB0 --display none \
  --duration-minutes 60

python run.py --radar-port /dev/ttyUSB0 --display range-doppler \
  --raw-output rawdatacapture/captures/session.bin

python run.py --radar-port /dev/ttyUSB0 \
  --pmm-detection-threshold 30000 \
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
- `--pmm-detection-threshold`: initial linear PMM score threshold.
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
- `diagnostics`

States progress through `calibrating`, `searching`, `tentative`, `confirmed`,
`coasting`, and `lost`. Only `confirmed` represents a confirmed PMM track; it
is not a drone-identification result.

When `--raw-output` is supplied, valid complete ADC frames are also saved to
the given `.bin` file with a JSON metadata sidecar. This is the preferred input
for repeatable threshold tuning.

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
