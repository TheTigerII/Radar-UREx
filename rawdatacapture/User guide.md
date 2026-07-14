# Live Raw ADC Capture User Guide

This guide explains `livedatacapture.py` and the integrated `run.py` launcher
for an IWR6843ISK-ODS connected to a DCA1000EVM.

## Prerequisites

- Python 3 with the dependencies from the repository `requirements.txt`.
- Git, which pip uses to install the pinned OpenRadar dependency.
- PC Ethernet interface configured as `192.168.33.30/24` by default.
- DCA1000 reachable at `192.168.33.180`.
- Radar running SDK CLI firmware when using `profile.cfg` and direct serial.
- `profile.cfg` must enable ADC LVDS streaming, for example:

```text
lvdsStreamCfg -1 0 1 0
```

Run commands from the repository root. Windows examples use `python` and
backslashes; Linux examples may use `python3` and forward slashes.

Create a project virtual environment and install dependencies before the first
run. On Linux:

```bash
python3 -m venv --system-site-packages .venv
.venv/bin/python -m pip install -r requirements.txt
```

On Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
```

Live range, Doppler, and CFAR processing uses OpenRadar. The IWR6843ISK-ODS
planar-array coordinate mapping remains in the local DSP adapter because
OpenRadar's supplied XYZ helper targets the AWR1843 virtual antenna layout.
Range, range-Doppler, and 3D point-cloud displays use Matplotlib.

## Recommended: Integrated Run

```powershell
.venv\Scripts\python run.py
```

`run.py`:

1. Prompts for a display mode unless `--display` is supplied.
2. Prompts for a run duration in minutes. Press Enter for the 3-minute default,
   or enter `0` for an unlimited run.
3. Detects serial ports and asks which command UART to use when needed.
4. Starts `livedatacapture.py` with `profile.cfg` and `setup.json` by default.
5. Saves valid frames to a timestamped `.bin` under
   `rawdatacapture\captures` unless `--raw-output` is given.
6. Starts `startup.py` with direct serial radar control and direct UDP DCA1000
   control.
7. Stops hardware control before capture when the duration expires or Ctrl+C
   is pressed.

Choose a display without prompting:

```powershell
python run.py --display none
python run.py --display range
python run.py --display range-doppler
python run.py --display point-cloud
```

Skip the duration prompt with an explicit value:

```powershell
python run.py --duration-minutes 3
python run.py --duration-minutes 0
```

The second command runs without a time limit.

Specify the radar UART if automatic detection is wrong:

```powershell
python run.py --radar-port COM4
```

```bash
python3 run.py --radar-port /dev/ttyUSB0
```

Point-cloud display defaults to updating every two valid frames; other modes
default to every frame. Override it with `--display-update-every N`.

## Capture Receiver Only

This starts UDP capture but does not configure or start the hardware:

```powershell
python rawdatacapture\livedatacapture.py
```

Its defaults are `mmwave.json`, `setup.json`, host `192.168.33.30`, UDP port
4098, no display, processing queue size 4, and a requested 4 MiB socket receive
buffer. The operating system may grant a different receive-buffer size; the
actual value is printed.

Use the same config that programmed the radar:

```powershell
python rawdatacapture\livedatacapture.py `
  --config .\rawdatacapture\profile.cfg `
  --setup .\rawdatacapture\setup.json
```

Both SDK CLI `.cfg` and mmWave Studio `.json` radar configs are supported. XML
is not supported.

## Displays

### Range profile

```powershell
python rawdatacapture\livedatacapture.py --display range
```

The plot is average range-FFT magnitude over chirps and receivers. Its X-axis
uses meters when sample rate and frequency slope are available; otherwise it
falls back to bin numbers. The displayed X limit defaults to 20 m:

```powershell
python rawdatacapture\livedatacapture.py --display range --max-range-m 10
python rawdatacapture\livedatacapture.py --display range --max-range-m 0
```

Zero selects the full computed axis.

### Range-Doppler heatmap

```powershell
python rawdatacapture\livedatacapture.py --display range-doppler
```

The Y-axis is centered Doppler-bin index, not calibrated velocity. Processing
uses frame loops as slow time and averages magnitude across TX chirp position
and receivers.

### Point cloud

```powershell
python rawdatacapture\livedatacapture.py --display point-cloud
```

The diagnostic point cloud uses CA-CFAR, local-peak filtering, and a virtual
antenna 2D FFT. It shows at most 50 points. Coordinates are X left/right, Y
forward, and Z elevation. The plot box is fixed at 5 m: X and Z are -2.5 to
2.5 m and Y is 0 to 5 m. Color is fixed from 40 to 120 dB. Angle output is not
calibrated.

### Display performance

Display rendering is a separate process and receives only the latest result.
If processing cannot keep up, reduce display frequency:

```powershell
python rawdatacapture\livedatacapture.py `
  --display range-doppler --display-update-every 3 --display-pause 0.05
```

Use `--display none` for packet-loss testing or headless operation.

## Save Raw Frames

```powershell
python rawdatacapture\livedatacapture.py `
  --raw-output .\rawdatacapture\captures\test_capture.bin
```

Only valid complete frames are written, without DCA1000 headers. The default
metadata path is `test_capture.bin.json`; override it with `--raw-metadata`.
The sidecar is written during clean shutdown.

Raw files have no file-size limit. A `run.py` session is limited to 3 minutes by
default, but `livedatacapture.py` alone and `run.py --duration-minutes 0` run
until stopped. Monitor free disk space during long captures.

## Logging and Statistics

Terminal output is appended to `rawdatacapture\livedatacapture.log` by default:

```powershell
python rawdatacapture\livedatacapture.py `
  --log-file .\rawdatacapture\capture_run.log
```

Important counters are:

- `lost_packets`: missing DCA1000 sequence numbers.
- `out_of_order`: late/backward packets or byte counts.
- `duplicates`: repeated sequence/payload data.
- `byte_gaps`: discontinuities inserted into normal frame assembly.
- `stream_resyncs`: byte-count jumps larger than one frame; partial assembly
  was discarded to avoid unbounded memory allocation.
- `invalid_frames`: frames touched by an ordinary byte gap.
- `processing_drops`: valid frames discarded because the bounded processing
  queue was full.

Healthy capture normally keeps all of these at zero. Frames containing gap
padding are neither processed nor saved.

If `processing_drops` grows while packet-loss counters remain zero, reduce
display updates or use `--display none`. Increasing `--processing-queue-size`
can absorb short bursts but does not fix processing that is consistently too
slow.

## Packet-Loss Troubleshooting

- Connect the PC directly to DCA1000 over wired Ethernet.
- Ensure `packetDelay_us` in `setup.json` matches the DCA1000 configuration.
- Avoid routing capture traffic through Wi-Fi, VPNs, or busy switches.
- Increase the OS/network-adapter receive buffers where supported.
- Check the logged requested and actual socket receive-buffer sizes.
- Run with `--display none` to distinguish capture loss from DSP load.

When `packetSequenceEnable` is false, capture still assembles received bytes but
cannot reliably report network packet loss.

## Shutdown

Press Ctrl+C. With `run.py`, the startup process is stopped first, followed by
the capture process. Clean capture shutdown closes the socket, child processes,
queues, raw file, metadata sidecar, and log file.

Show all receiver options with:

```powershell
python rawdatacapture\livedatacapture.py --help
```
