# Radar Startup User Guide

This guide covers running `startup.py` to configure the radar with SDK CLI
firmware and optionally configure/arm the DCA1000 over UDP.

`startup.py` currently validates and runs the startup sequence. The real UDP
capture and frame processing loop still lives in `livedatacapture.py`, so the
standalone startup command cleans up hardware on exit after it reaches
`radar_streaming`.

Run commands from the repository root:

```powershell
cd C:\Users\eliwe\Desktop\Radar-UREx
```

## Prerequisites

- The radar is flashed with SDK CLI firmware.
- `rawdatacapture\profile.cfg` is the SDK CLI profile to send to the radar.
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
dry-run mode:

```powershell
python rawdatacapture\startup.py --radar-backend direct-serial --dca-backend dry-run --radar-port COM10 --radar-baud 115200
```

`startup.py` sends all commands from `profile.cfg` except `sensorStart` during
configuration. It sends `sensorStart` later, after the DCA1000 arm step in the
startup sequence.

## Start Radar And DCA1000

This configures/arms the DCA1000 over UDP and starts the radar through SDK CLI
serial:

```powershell
python rawdatacapture\startup.py --radar-backend direct-serial --dca-backend direct-udp --radar-port COM10 --radar-baud 115200
```

Use this only after preflight passes and the DCA1000 network is configured.

## Common Overrides

If your radar command UART is not `COM10`, change:

```powershell
--radar-port COM10
```

If your SDK CLI baud rate is different, change:

```powershell
--radar-baud 115200
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

## Troubleshooting

If DCA1000 direct UDP times out at `SYSTEM_CONNECT`, check:

- PC Ethernet adapter IP is `192.168.33.30`.
- DCA1000 is powered and connected directly by Ethernet.
- DCA1000 IP is `192.168.33.180`.
- No other process is using UDP config port `4096`.

If SDK CLI serial times out, check:

- The radar is flashed with SDK CLI firmware.
- The COM port is the command UART, not the data UART.
- The baud rate is correct.
- No terminal program already has the COM port open.
