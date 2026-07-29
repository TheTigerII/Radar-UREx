# Radar and DCA1000 Startup Design

This document describes the implemented `startup.py` control plane. It
configures radar and DCA1000 hardware but does not receive raw ADC packets.
Use `run.py` for the integrated control-and-capture workflow.

## Responsibilities

`startup.py` can:

- Parse radar dimensions from the fixed SDK CLI `.cfg`.
- Parse board and DCA1000 settings from `setup.json`.
- Validate dimensions, two-lane LVDS support, ports, profile commands, and
  availability of the capture UDP bind address.
- Configure DCA1000 through UDP or simulate it with a dry-run backend.
- Configure SDK CLI firmware over the radar command UART or use dry-run.
- Defer `sensorStart` until DCA1000 recording has been armed.
- Send `sensorStop` and DCA1000 `RECORD_STOP` during shutdown.

It does not flash firmware.

## Backends

### Radar

- `dry-run` (default): prints intended radar operations.
- `direct-serial`: opens the SDK CLI command UART with pyserial, sends commands
  from `--sdk-profile`, defers the profile's `sensorStart`, and sends it after
  capture arm.

The profile must contain `sensorStart`. Blank lines and `%`/`#` comments are
ignored. Command responses are accepted when they contain `Done`, `Error`, the
`mmwDemo:/>` prompt, `Ignored`, or `Skipped`; an error response fails startup
unless cleanup is already in progress.

### DCA1000

- `dry-run` (default): prints intended configuration and arm operations.
- `direct-udp`: binds local UDP port 4096 by default and sends commands to
  `192.168.33.180:4096`.

Direct UDP sends, in order:

```text
SYSTEM_CONNECT
RESET_FPGA
CONFIG_FPGA_GEN
CONFIG_RECORD
RECORD_START
```

Shutdown sends `RECORD_STOP` if record start succeeded. Each command waits for
and validates an acknowledgement, using `--dca-timeout` and `--dca-retries`.
FPGA and record payloads are derived from setup/config data, with optional hex
overrides under `directUdpDCA1000.payloads` in `setup.json`.

### Capture

The only `--capture-backend` is `dry-run`. It changes orchestration state and
prints the expected address/frame size, but does not bind or read UDP data.
This boundary is important: `startup.py` alone starts a streaming sensor but
does not save or display its frames.

## Configuration Selection

The radar-config and SDK-profile defaults are both
`profile-mini4-20m.cfg`; the capture setup default is `setup.json`. When
`--radar-backend direct-serial` is selected and `--config` was not explicitly
changed, that one profile supplies both the capture dimensions and the SDK CLI
commands. This prevents direct-serial startup from interpreting frames using
unrelated JSON dimensions.

`--sdk-profile` chooses commands sent to the radar. Normally it should be the
same `.cfg` passed through `--config`.

## Orchestration Sequence

```text
1. load_configs
   -> CONFIGS_LOADED

2. preflight validation
   -> PREFLIGHT_PASSED

3. configure DCA1000 backend
   -> DCA1000_READY

4. configure radar backend, excluding sensorStart
   -> RADAR_READY

5. start capture backend (currently dry-run only)
   -> RECEIVER_READY

6. arm DCA1000 and wait --readiness-delay (default 0.25 s)
   -> DCA1000_ARMED

7. send deferred sensorStart
   -> RADAR_STREAMING
```

The main function then waits until Ctrl+C. `--preflight-only` stops after step
2 and does not configure hardware.

## Preflight Checks

Preflight validates:

- Positive ADC sample, RX, chirp, and frame-byte dimensions.
- Exactly two LVDS lanes.
- Supported DCA1000 capture hardware and non-negative packet delay.
- Valid UDP data/config ports.
- Radar serial port and baud when direct serial is selected.
- Existing SDK profile containing `sensorStart` for direct serial.
- Ability to bind `host_ip:data_port`, unless `--skip-socket-preflight` is used.

The socket check is only a preflight probe; it closes immediately. In the
two-terminal workflow the real receiver already owns the port, so startup must
be given `--skip-socket-preflight`.

## Shutdown and Failure Handling

Any exit after startup begins enters `STOPPING`. Cleanup is attempted in this
order:

```text
radar sensor stop
DCA1000 record stop and control-socket close
dry-run capture close
```

Cleanup continues if one step fails and reports accumulated errors afterward.
The final state is `STOPPED`.

## Relationship to `run.py`

`run.py` supplies the missing integration externally:

```text
prompt for duration (default 3 minutes; 0 is unlimited)
start livedatacapture.py
wait 1 second and verify it is still running
start startup.py with:
  --radar-backend direct-serial
  --dca-backend direct-udp
  --skip-socket-preflight
stop both processes when the deadline expires or Ctrl+C is pressed
```

It passes the same `.cfg` to capture, startup dimension parsing, and the SDK CLI
command backend. This is the recommended way to keep hardware programming and
frame interpretation consistent.

## Current Limitations

- No firmware flashing backend.
- No real capture backend inside `startup.py`.
- Direct serial assumes compatible SDK CLI firmware is already running.
- DCA1000 control compatibility depends on the firmware accepting the generated
  or overridden command payloads.
- Health monitoring reports state transitions only; packet/frame health comes
  from `livedatacapture.py`.
