# Radar Startup User Guide

This guide covers `startup.py`, which controls an SDK CLI radar and optionally
the DCA1000. It does not receive or save ADC data. For normal operation, use
`python run.py`; use `startup.py` directly for preflight, control testing, or a
manual two-terminal workflow.

## Prerequisites

- Radar flashed with SDK CLI-compatible firmware.
- Radar booted in the functional mode required by that firmware. For the local
  IWR6843 setup this is documented as SOP `00110X`; confirm against the board
  and firmware documentation before changing switches.
- `profile.cfg` contains the radar commands and `sensorStart`.
- `profile.cfg` enables LVDS ADC streaming, normally:

```text
lvdsStreamCfg -1 0 1 0
```

- pyserial installed for direct radar serial control.
- Command UART known, such as `COM4` on Windows or `/dev/ttyUSB0` on Linux.
- For direct DCA1000 control, host Ethernet defaults to `192.168.33.30/24` and
  DCA1000 to `192.168.33.180:4096`.

## Recommended Integrated Command

From the repository root:

```powershell
python run.py --display range
```

This starts capture before hardware control and saves raw data. See
`User guide.md` for display and capture options. It prompts for a run duration;
press Enter for 3 minutes or enter `0` for unlimited. Use
`--duration-minutes N` to provide the value non-interactively.

## Preflight Only

Check direct-serial inputs without sending commands:

```powershell
python rawdatacapture\startup.py `
  --radar-backend direct-serial `
  --dca-backend dry-run `
  --preflight-only `
  --radar-port COM4 `
  --radar-baud 115200
```

Do not add `--skip-socket-preflight` unless another receiver already owns the
capture address or you deliberately want to bypass that check.

Successful output ends with:

```text
startup state: idle -> configs_loaded
startup state: configs_loaded -> preflight_passed
Startup preflight passed.
```

When direct serial is selected without an explicit `--config`, frame dimensions
are read from `profile.cfg` automatically.

## Radar Control Without DCA1000 Control

```powershell
python rawdatacapture\startup.py `
  --radar-backend direct-serial `
  --dca-backend dry-run `
  --radar-port COM4 `
  --radar-baud 115200
```

This really configures and starts the radar, but only simulates DCA1000
configuration. It is useful for serial testing, not raw ADC recording. Press
Ctrl+C to send `sensorStop` and close the UART.

## Radar and DCA1000 Control

```powershell
python rawdatacapture\startup.py `
  --radar-backend direct-serial `
  --dca-backend direct-udp `
  --radar-port COM4 `
  --radar-baud 115200
```

This sends DCA1000 configuration/record commands and starts the radar, but
`startup.py` still does not receive ADC packets. Use it only when another
receiver is ready or when testing the control plane.

## Manual Two-Terminal Capture

Terminal 1 must start first:

```powershell
python rawdatacapture\livedatacapture.py `
  --config .\rawdatacapture\profile.cfg `
  --setup .\rawdatacapture\setup.json `
  --host-ip 192.168.33.30 `
  --data-port 4098 `
  --display range `
  --raw-output .\rawdatacapture\captures\manual.bin
```

Terminal 2 configures hardware. The skip flag is required because Terminal 1
already owns UDP port 4098:

```powershell
python rawdatacapture\startup.py `
  --config .\rawdatacapture\profile.cfg `
  --sdk-profile .\rawdatacapture\profile.cfg `
  --setup .\rawdatacapture\setup.json `
  --radar-backend direct-serial `
  --dca-backend direct-udp `
  --skip-socket-preflight `
  --radar-port COM4 `
  --radar-baud 115200 `
  --radar-command-timeout 10
```

Stop Terminal 2 first so it attempts `sensorStop` and `RECORD_STOP`, then stop
Terminal 1 so raw metadata is finalized.

## Linux / Jetson

Install dependencies:

```bash
python3 -m pip install numpy matplotlib pyserial
```

Grant serial access, then log out and back in:

```bash
sudo usermod -a -G dialout $USER
```

Configure the DCA1000-facing interface, replacing `eth0` as appropriate:

```bash
sudo ip addr add 192.168.33.30/24 dev eth0
sudo ip link set eth0 up
```

Find and test command UART candidates:

```bash
ls /dev/ttyUSB* /dev/ttyACM*
python3 -m serial.tools.miniterm /dev/ttyUSB0 115200
```

The correct SDK CLI port normally shows `mmwDemo:/>` after Enter. Exit
miniterm before running Python so it releases the port.

Integrated Linux example:

```bash
python3 run.py --radar-port /dev/ttyUSB0 --display none
```

Use `none` for headless sessions without a graphical display.

## Useful Overrides

- `--radar-command-timeout 10`: longer wait for each CLI response.
- `--radar-command-delay 0.05`: delay between CLI commands.
- `--radar-line-ending lf`: use LF instead of the default CRLF.
- `--dca-timeout 3`: acknowledgement timeout per DCA command attempt.
- `--dca-retries 5`: retries after the initial DCA attempt.
- `--readiness-delay 0.5`: delay between DCA record arm and `sensorStart`.
- `--sdk-profile PATH --config PATH`: explicitly keep the command profile and
  frame-dimension source aligned.

## Expected Startup States

A full successful run transitions through:

```text
configs_loaded
preflight_passed
dca1000_ready
radar_ready
receiver_ready
dca1000_armed
radar_streaming
```

`receiver_ready` refers only to `startup.py`'s dry-run capture backend. It does
not prove that `livedatacapture.py` is running; `run.py` verifies that process
separately before starting hardware.

## Troubleshooting

### DCA1000 `SYSTEM_CONNECT` timeout

- Confirm host IP `192.168.33.30` and DCA IP `192.168.33.180`.
- Confirm direct Ethernet link, power, and UDP port 4096 availability.
- Close mmWave Studio or TI DCA tools that may own the control port.
- Increase `--dca-timeout` and `--dca-retries` only after checking networking.

### Socket preflight bind failure

If `livedatacapture.py` already owns `192.168.33.30:4098`, use
`--skip-socket-preflight`. Otherwise the error may indicate a wrong host IP or
another receiver already using the port.

### Serial timeout

- Verify SDK CLI firmware and functional SOP mode, then reset/power-cycle.
- Select the command UART rather than the high-speed data UART.
- Close terminal programs using the port.
- Confirm baud rate and try `--radar-command-timeout 10`.
- Try `--radar-line-ending lf` if the prompt never responds to CRLF.

### Capture window remains empty

- Confirm `livedatacapture.py` prints complete frames.
- Confirm `profile.cfg` enables LVDS ADC streaming.
- Ensure capture and startup use the same profile for frame dimensions.
- Check packet-loss, byte-gap, stream-resync, and processing-drop counters.

Show every option with:

```powershell
python rawdatacapture\startup.py --help
```
