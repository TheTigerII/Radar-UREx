# Radar Startup User Guide

This guide covers running `startup.py` to configure the radar with SDK CLI
firmware and optionally configure/arm the DCA1000 over UDP.

`startup.py` runs the startup sequence, starts the radar, and keeps the session
alive until you press Ctrl+C. Ctrl+C sends `sensorStop`, stops DCA1000 if it was
armed, and closes local resources. The real UDP capture and frame processing
loop still lives in `livedatacapture.py`.

Run commands from the repository root:

```powershell
cd C:\Users\eliwe\Desktop\Radar-UREx
```

## Prerequisites

- The radar is flashed with SDK CLI firmware.
- Set the radar SOP/switches to functional mode `00110X` before reset/power-up.
  This lets the flashed out-of-box/SDK CLI firmware boot and respond on the
  command UART. Do not use DCA/mmWave Studio mode `01100X` with this
  `startup.py` direct-serial path; the DCA1000 can still be used while the
  radar is in functional mode.
- `rawdatacapture\profile.cfg` is the SDK CLI profile to send to the radar.
- `profile.cfg` must enable LVDS ADC streaming for DCA1000 capture:

```text
lvdsStreamCfg -1 0 1 0
```

- The radar command UART is known. The examples below use:

```text
COM10
115200 baud
```

- For real DCA1000 UDP mode, the PC Ethernet adapter must use:

```text
192.168.33.30
```

- The DCA1000 must be reachable at:

```text
192.168.33.180:4096
```

## Preflight Only

Use this first. It checks config/profile inputs without sending hardware
commands:

```powershell
python rawdatacapture\startup.py --radar-backend direct-serial --dca-backend dry-run --preflight-only --skip-socket-preflight --radar-port COM10 --radar-baud 115200
```

## Start Radar Only

This sends `profile.cfg` to the radar over SDK CLI serial, but keeps DCA1000 in
dry-run mode. The radar keeps sending until you press Ctrl+C:

```powershell
python rawdatacapture\startup.py --radar-backend direct-serial --dca-backend dry-run --radar-port COM10 --radar-baud 115200
```

`startup.py` sends all commands from `profile.cfg` except `sensorStart` during
configuration. It sends `sensorStart` later, after the DCA1000 arm step in the
startup sequence.

## Start Radar And DCA1000

This configures/arms the DCA1000 over UDP and starts the radar through SDK CLI
serial. The radar keeps sending until you press Ctrl+C:

```powershell
python rawdatacapture\startup.py --radar-backend direct-serial --dca-backend direct-udp --radar-port COM10 --radar-baud 115200
```

Use this only after preflight passes and the DCA1000 network is configured.

## Run With Live Capture

Use two terminals when `livedatacapture.py` is receiving data and `startup.py`
is starting the radar/DCA1000.

Terminal 1 starts the UDP receiver first:

```powershell
python rawdatacapture\livedatacapture.py --config .\rawdatacapture\profile.cfg --setup .\rawdatacapture\setup.json --host-ip 192.168.33.30 --data-port 4098 --display range
```

Use `--display range-doppler` instead of `--display range` for the diagnostic
range-Doppler heatmap. If `--display` is omitted, the default is
`--display none`, so no live window appears.

Terminal 2 starts the hardware. Use `--skip-socket-preflight` because
`livedatacapture.py` already owns `192.168.33.30:4098`:

```powershell
python rawdatacapture\startup.py --radar-backend direct-serial --dca-backend direct-udp --skip-socket-preflight --radar-port COM4 --radar-baud 115200 --radar-command-timeout 10
```

Stop with Ctrl+C in Terminal 2 first so `startup.py` sends `sensorStop` and
stops DCA1000. Then stop Terminal 1.

## Common Overrides

If your radar command UART is not `COM10`, change:

```powershell
--radar-port COM10
```

If your SDK CLI baud rate is different, change:

```powershell
--radar-baud 115200
```

`startup.py` sends SDK CLI commands with CRLF line endings by default. If your
serial console only responds to LF, add:

```powershell
--radar-line-ending lf
```

If you want to use a different SDK CLI profile:

```powershell
--sdk-profile .\rawdatacapture\profile.cfg --config .\rawdatacapture\profile.cfg
```

When `--radar-backend direct-serial` is used and `--config` is not supplied,
`startup.py` automatically uses `rawdatacapture\profile.cfg` for frame
dimensions.

## Expected Output

A successful dry-run preflight should include:

```text
startup state: idle -> configs_loaded
startup state: configs_loaded -> preflight_passed
Startup preflight passed.
```

A successful startup should progress through:

```text
configs_loaded
preflight_passed
dca1000_ready
radar_ready
receiver_ready
dca1000_armed
radar_streaming
```

After `radar_streaming`, the process should print:

```text
Radar is running. Press Ctrl+C to stop.
```

## Troubleshooting

If DCA1000 direct UDP times out at `SYSTEM_CONNECT`, check:

- PC Ethernet adapter IP is `192.168.33.30`.
- DCA1000 is powered and connected directly by Ethernet.
- DCA1000 IP is `192.168.33.180`.
- No other process is using UDP config port `4096`.

If Windows reports `WinError 10048`, another process already owns the same UDP
address and port. The common case is `livedatacapture.py` already listening on
`192.168.33.30:4098`; run `startup.py` with `--skip-socket-preflight` in that
case. If the conflict is on `4096`, close mmWave Studio, the TI DCA1000 CLI, or
any other process controlling the DCA1000.

If the live display window opens but stays empty, check Terminal 1 for
`Complete frame` lines. If no complete frames arrive, confirm that `profile.cfg`
has LVDS ADC streaming enabled:

```text
lvdsStreamCfg -1 0 1 0
```

If SDK CLI serial times out, check:

- The radar is flashed with SDK CLI firmware.
- The radar SOP/switches are in functional mode `00110X`, then reset/power-cycle
  the radar.
- The COM port is the command UART, not the data UART.
- The baud rate is correct.
- No terminal program already has the COM port open.
- Try a longer command timeout, for example `--radar-command-timeout 10`.
- Try the alternate line ending, for example `--radar-line-ending lf`.
