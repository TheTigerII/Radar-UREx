# Radar Startup Architecture

This document describes the current `startup.py` flow for running the
IWR6843ISK-ODS and DCA1000EVM without using mmWave Studio as the runtime
controller.

mmWave Studio can still be useful as a configuration authoring and reference tool,
but the Python runtime should own startup, validation, capture arming, sensor start,
and shutdown.

## Goal

Validate or start the radar/DCA1000 startup sequence from Python:

```powershell
python rawdatacapture\startup.py
```

The current startup layer can:

- Load radar dimensions from `mmwave.json` or an SDK CLI `.cfg` file.
- Load DCA1000 and board setup from `setup.json`.
- Validate host network, radar control ports, firmware paths, and capture settings.
- Configure and arm the DCA1000 over direct UDP.
- Configure SDK CLI firmware over a serial command UART with `profile.cfg`.
- Defer `sensorStart` until after the DCA1000 is armed.
- Stop the radar and DCA1000 cleanly on exit or error.

## Target Architecture

```text
startup CLI / livedatacapture.py
        |
        v
StartupOrchestrator
        |
        +--> ConfigLoader
        |      - mmwave.json or profile.cfg
        |      - setup.json
        |      - derived frame dimensions
        |
        +--> PreflightValidator
        |      - host Ethernet IP
        |      - DCA1000 control/data endpoints
        |      - radar COM/control channel
        |      - firmware paths
        |      - packet sequence and delay settings
        |
        +--> DCA1000Control
        |      - system connect
        |      - reset/configure FPGA
        |      - apply packet delay
        |      - arm recording/streaming
        |      - stop recording/streaming
        |
        +--> RadarControl
        |      - dry-run or SDK CLI serial
        |      - apply profile.cfg command list
        |      - defer sensorStart until after DCA1000 arm
        |      - sensorStop on shutdown
        |
        +--> CapturePipeline
        |      - dry-run placeholder in startup.py
        |      - real UDP receive remains in livedatacapture.py
        |      - future integration point for frame processing/display
        |
        +--> HealthMonitor
               - startup status
               - packet loss counters
               - byte gaps
               - invalid frames
               - timeout/error handling
```

## Startup Sequence

The order matters because the DCA1000 must be ready before the radar sends LVDS
frame data.

```text
1. Load configuration
   - Read mmwave.json by default.
   - If --radar-backend direct-serial is used without --config, use profile.cfg
     as the radar config source.
   - Read setup.json.
   - Derive bytes_per_frame and radar cube dimensions.

2. Preflight
   - Confirm host Ethernet IP is available.
   - Confirm DCA1000 data port is free for binding.
   - Confirm radar control channel settings exist.
   - Confirm profile.cfg exists and includes sensorStart for direct-serial.
   - Confirm firmware files exist if firmware loading is enabled.

3. Initialize DCA1000
   - Dry-run by default, or use --dca-backend direct-udp.
   - Open local config socket on 0.0.0.0:4096 for direct UDP.
   - Send SYSTEM_CONNECT.
   - Send RESET_FPGA.
   - Apply LVDS raw capture mode.
   - Apply packet delay from setup.json.
   - Send CONFIG_FPGA_GEN and CONFIG_RECORD.

4. Initialize radar
   - Dry-run by default, or use --radar-backend direct-serial.
   - Open radar command UART from --radar-port or setup.json.
   - Use --radar-baud or setup.json baud rate.
   - Send SDK CLI commands from profile.cfg, except sensorStart.

5. Start receive path
   - Dry-run capture receiver placeholder in startup.py.
   - Live UDP receive still lives in livedatacapture.py.

6. Arm capture
   - Send DCA1000 RECORD_START when direct UDP is enabled.
   - Wait for acknowledgement or a short readiness delay.

7. Start radar
   - Send deferred SDK CLI sensorStart when direct serial is enabled.
   - Capture loop begins receiving DCA1000 packets.

8. Run and monitor
   - Report packets, frames, byte gaps, invalid frames, and processing drops.
   - Treat repeated byte gaps or missing packets as capture health warnings.

9. Shutdown
   - Send radar sensor stop.
   - Stop DCA1000 recording/streaming.
   - Drain queues.
   - Close sockets, serial handles, display, and log files.
```

## Runtime State Machine

```text
idle
  -> configs_loaded
  -> preflight_passed
  -> dca1000_ready
  -> radar_ready
  -> receiver_ready
  -> dca1000_armed
  -> radar_streaming
  -> stopping
  -> stopped
```

Any failure after `dca1000_ready` should enter `stopping` so the system attempts
to stop the radar and DCA1000 before releasing local resources.

## Configuration Ownership

### `mmwave.json`

Owns radar signal and LVDS data-path settings for the mmWave Studio JSON flow:

- RF channel enable masks.
- ADC output format.
- Profile configuration.
- Chirp configuration.
- Frame configuration.
- LVDS lane and data format settings.

The capture code can derive these runtime dimensions from this file:

```text
num_adc_samples
num_rx_channels
num_chirps_per_frame
bytes_per_frame
iq_swap
channel_interleave
lvds_lanes
sample_rate_ksps
frequency_slope_mhz_per_us
```

### `setup.json`

Owns board, transport, and capture hardware settings:

- DCA1000 capture mode.
- Packet sequence header enable.
- Packet delay.
- Radar device type.
- Radar control COM port and baud rate.
- MSS/BSS firmware paths.
- Capture output conventions from the TI tooling.

### `profile.cfg`

Owns the SDK CLI radar command flow when `--radar-backend direct-serial` is used:

- `sensorStop`, `flushCfg`, and radar setup commands.
- `profileCfg`, `chirpCfg`, `frameCfg`, `adcbufCfg`, and `lvdsStreamCfg`.
- `sensorStart`, which `startup.py` defers until after DCA1000 is armed.

When `direct-serial` is selected and `--config` is not supplied, `startup.py`
uses `profile.cfg` for both SDK CLI commands and frame-dimension parsing.

## Control Backends

The startup architecture should allow more than one backend while the project
moves away from mmWave Studio.

```text
RadarControl
  DirectSerialRadarControl
    - Python sends an SDK CLI `.cfg` profile over the radar command UART.
    - Implemented in startup.py with `--radar-backend direct-serial`.
    - Configuration commands are sent first; `sensorStart` is deferred until
      after DCA1000 arm.

  CliRadarControl
    - Not implemented.

DCA1000Control
  DirectUdpDca1000Control
    - Python sends DCA1000 control commands over UDP.
    - Implemented in startup.py for system connect, FPGA reset, FPGA config,
      packet config, record start, and record stop.

  CliDca1000Control
    - Not implemented.
```

The public orchestration sequence should stay the same whichever backend is used.
That keeps capture, processing, and display independent from how hardware startup
is implemented.

`DirectUdpDca1000Control` builds default payloads from `setup.json` and
`mmwave.json`/`profile.cfg`-derived data. It sends:

```text
SYSTEM_CONNECT
RESET_FPGA
CONFIG_FPGA_GEN
CONFIG_RECORD
RECORD_START
RECORD_STOP
```

For the current setup, the generated default payloads are:

```text
CONFIG_FPGA_GEN = 01020102031e
CONFIG_RECORD   = be05350c0000
```

If a TI CLI version needs exact byte-for-byte payloads, add hex overrides under:

```json
{
  "directUdpDCA1000": {
    "payloads": {
      "CONFIG_FPGA_GEN": "01020102031e",
      "CONFIG_RECORD": "be05350c0000"
    }
  }
}
```

## Module Split

```text
rawdatacapture/
  livedatacapture.py
    Entry point for live capture and frame processing.

  startup.py
    StartupOrchestrator, direct UDP DCA1000, SDK CLI radar serial,
    dry-run backends, preflight, and state machine.
```

## SDK CLI Startup

When the radar is running SDK CLI firmware, use `profile.cfg` as both the radar
command profile and the source of frame dimensions:

```powershell
python rawdatacapture\startup.py --config .\rawdatacapture\profile.cfg --sdk-profile .\rawdatacapture\profile.cfg --radar-backend direct-serial --radar-port COM10 --radar-baud 115200
```

If `--radar-backend direct-serial` is used and `--config` is not supplied,
`startup.py` automatically uses `rawdatacapture\profile.cfg` for frame
dimensions. This avoids mixing `mmwave.json` dimensions with a different SDK CLI
profile.

The SDK CLI backend sends all commands in the profile except `sensorStart`
during radar configuration. The orchestrator then arms DCA1000 and sends
`sensorStart` only after the receiver path is ready.

For real DCA1000 UDP plus SDK CLI radar startup:

```powershell
python rawdatacapture\startup.py --radar-backend direct-serial --dca-backend direct-udp --radar-port COM10 --radar-baud 115200
```

For SDK CLI radar preflight only:

```powershell
python rawdatacapture\startup.py --radar-backend direct-serial --dca-backend dry-run --preflight-only --skip-socket-preflight --radar-port COM10 --radar-baud 115200
```

## Failure Handling

Startup should fail early with clear messages for configuration and environment
problems:

```text
missing radar config
missing setup.json
host Ethernet IP not assigned
DCA1000 data port already in use
radar COM port not found
missing profile.cfg
firmware path not found
unsupported ADC format
unsupported LVDS lane mode
DCA1000 command timeout
radar command timeout
```

Once hardware has been touched, failures should trigger cleanup:

```text
try:
    startup()
    run_capture()
finally:
    sensor_stop_if_started()
    dca1000_stop_if_armed()
    close_capture_pipeline()
```

## Current Gaps

- `startup.py` has a dry-run capture pipeline placeholder. The real UDP receive,
  frame assembly, processing, and display still live in `livedatacapture.py`.
- `cli` backends are exposed as choices but are not implemented.
- `direct-serial` assumes SDK CLI firmware is already flashed and running.
- `direct-serial` does not flash MSS/BSS firmware.
- `direct-udp` configures and arms DCA1000, but the end-to-end capture handoff to
  `livedatacapture.py` is still a separate integration step.

The intended integrated runtime flow is:

```text
preflight -> configure hardware -> start receiver -> arm DCA1000 -> start radar
```
