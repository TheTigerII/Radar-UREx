# Radar and DCA1000 Startup Guide

`startup.py` is the hardware control plane. Normal operation should use
`run.py`, which starts the ADC receiver before arming the DCA1000 and starting
the radar.

## Prerequisites

- IWR6843ISK-ODS running SDK CLI-compatible firmware.
- `profiles/profile-mini4-20m.cfg` available unchanged.
- DCA1000 connected directly to a host interface configured as
  `192.168.33.30/24`.
- Radar command UART available, commonly `/dev/ttyUSB0`.
- `pyserial` installed.

## Integrated operation

```bash
python main/run.py --display combined
```

The launcher uses the same profile for hardware commands and frame dimensions,
preventing acquisition/processing mismatches. It automatically selects the
CP2105 `Enhanced COM Port`; an explicit `--radar-port` overrides this default.

## Preflight

```bash
python main/startup.py \
  --config profiles/profile-mini4-20m.cfg \
  --sdk-profile profiles/profile-mini4-20m.cfg \
  --setup profiles/setup.json \
  --preflight-only --skip-socket-preflight
```

Omit `--skip-socket-preflight` when no ADC receiver owns
`192.168.33.30:4098`.

## Manual two-terminal operation

Start capture first:

```bash
python main/livedatacapture.py \
  --config profiles/profile-mini4-20m.cfg \
  --setup profiles/setup.json \
  --host-ip 192.168.33.30 --data-port 4098 \
  --display combined \
  --raw-output dataset/manual.bin
```

Then configure and start the hardware:

```bash
python main/startup.py \
  --config profiles/profile-mini4-20m.cfg \
  --sdk-profile profiles/profile-mini4-20m.cfg \
  --setup profiles/setup.json \
  --radar-backend direct-serial \
  --dca-backend direct-udp \
  --skip-socket-preflight \
  --radar-port /dev/ttyUSB0 \
  --radar-baud 115200
```

Stop the hardware-control terminal first so it sends `sensorStop` and
`RECORD_STOP`, then stop capture.

## Expected states

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

For a DCA1000 timeout, confirm the direct Ethernet link, host address, board
power, and that no TI tool owns UDP port 4096.

For a serial timeout, select the command UART, close terminal programs, confirm
115200 baud and SDK CLI firmware, and try a longer
`--radar-command-timeout`.

For an ADC bind failure, use `--skip-socket-preflight` only when the receiver
already owns port 4098; otherwise correct the host interface address or stop
the conflicting process.
