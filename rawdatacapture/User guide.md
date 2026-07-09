# Live Raw ADC Capture User Guide

This guide is for running `livedatacapture.py` with the IWR6843ISK-ODS and DCA1000EVM.

## Hardware And Tool Setup

- Connect the PC directly to the DCA1000 Ethernet port.
- Configure the PC Ethernet adapter with the host IP used by the script:

```text
192.168.33.30
```

- Configure the DCA1000/radar in mmWave Studio before starting Python.
- Save the mmWave Studio radar config as `rawdatacapture\mmwave.json`.
- Save the mmWave Studio capture setup as `rawdatacapture\setup.json`.
- Keep the DCA1000 packet delay consistent with `setup.json`. The current setup uses:

```text
25 us
```

The current 25 FPS, 16-bit complex, 4-RX setup needs roughly 9,000 DCA1000 payload packets per second. Validate any packet-delay change with:

```text
lost_packets=0
byte_gaps=0/0B
```

## One-Command Startup And Capture

From the repository root, run:

```powershell
python run.py
```

`run.py` asks for the display type, uses `/dev/ttyUSB0` as the default Linux
radar command UART, starts `livedatacapture.py`, starts `startup.py`, and saves
valid raw frames to a timestamped file under:

```text
rawdatacapture\captures\
```

Stop with Ctrl+C. `run.py` stops `startup.py` first so the radar receives
`sensorStop`, then stops `livedatacapture.py` so the raw data file and metadata
sidecar are closed cleanly.

To choose the display mode without the prompt:

```powershell
python run.py --display range
python run.py --display range-doppler
python run.py --display none
```

If the radar command UART is not detected correctly, pass it explicitly:

```powershell
python run.py --radar-port COM4
```

On Linux/Jetson, use the Linux serial device:

```bash
python3 run.py --radar-port /dev/ttyUSB0
```

## Basic Capture

From the repository root, run:

```powershell
python rawdatacapture\livedatacapture.py
```

By default this reads:

```text
rawdatacapture\mmwave.json
rawdatacapture\setup.json
```

To use alternate JSON files:

```powershell
python rawdatacapture\livedatacapture.py --config .\rawdatacapture\mmwave.json --setup .\rawdatacapture\setup.json
```

XML configs are no longer supported by `livedatacapture.py`.

Expected startup output:

```text
Loaded radar config: RadarCaptureConfig(...)
Loaded capture setup: CaptureSetupConfig(...)
Listening for live radar stream on 192.168.33.30:4098; bytes_per_frame=524288
DCA1000 setup: packet_sequence_enable=True, packet_delay_us=25
Trigger frames now. Press Ctrl+C to stop.
```

Expected frame output:

```text
Complete frame cube_shape=(128, 4, 256), peak_range_m=..., peak_range_bin=30, peak_magnitude=...
```

Stop with:

```text
Ctrl+C
```

## Logging

By default, terminal status output is appended to:

```text
rawdatacapture\livedatacapture.log
```

Each log line includes a local timestamp.

To choose a different log file:

```powershell
python rawdatacapture\livedatacapture.py --log-file .\rawdatacapture\capture_run.log
```

## Save Raw Frames

To save complete valid raw ADC frames for later testing:

```powershell
python rawdatacapture\livedatacapture.py --raw-output .\rawdatacapture\captures\test_capture.bin
```

The binary file stores valid frame payloads consecutively, without DCA1000 packet headers. Each frame is `bytes_per_frame` bytes. A JSON sidecar is written beside the binary file by default, for example:

```text
test_capture.bin.json
```

The sidecar records frame count, frame size, ADC format, RX/channel ordering, and radar dimensions so the file can be replayed in later tests.

## Live Range Profile

To show a simple live range profile:

```powershell
python rawdatacapture\livedatacapture.py --display range
```

The live range X-axis shows `0` to `20 m` by default. To change it:

```powershell
python rawdatacapture\livedatacapture.py --display range --max-range-m 10
```

Use `--max-range-m 0` to show the full computed range axis.

Plot axes:

```text
X-axis: range in meters
Y-axis: average FFT magnitude across chirps and RX channels
```

The script derives range in meters from `digOutSampleRate` and `freqSlopeConst` in the radar config. The terminal still prints `peak_range_bin` for debugging, but `peak_range_m` is the physical range estimate.

When a live display is enabled, the script also prints and logs display latency:

```text
display latency: frame_first_byte_to_canvas_draw_ms=...
```

This is measured from the moment the first UDP payload byte belonging to that frame reaches Python until Matplotlib completes the canvas draw for that frame. It is the software-side approximation of first frame byte sent to displayed frame.

## Range-Doppler Heatmap

To show a first-pass range-Doppler heatmap:

```powershell
python rawdatacapture\livedatacapture.py --display range-doppler
```

This is an early diagnostic view, not yet a fully calibrated radar processing product.

## Display Responsiveness

UDP packet receiving is split from FFT/display processing. The receive loop assembles complete frames and hands valid frames to a processing worker. Matplotlib then runs behind that worker in a separate display process with a one-item latest-frame queue.

If plotting still feels sluggish, reduce update rate:

```powershell
python rawdatacapture\livedatacapture.py --display range --display-update-every 2 --display-pause 0.05
```

If packet gaps increase while plotting, try:

```powershell
--display-update-every 3
```

For packet-loss testing, run without display:

```powershell
python rawdatacapture\livedatacapture.py --display none
```

## Packet Loss And Invalid Frames

The script reads `packetSequenceEnable` from `setup.json`. With sequence headers enabled, it tracks DCA1000 sequence numbers and byte counts. If sequence headers are disabled, it can still assemble frames from received bytes, but packet-loss detection is weaker.

Healthy stats look like:

```text
lost_packets=0
byte_gaps=0/0B
invalid_frames=0
processing_drops=0
```

If packets are lost, you may see:

```text
Dropped frame: incomplete payload, gap_bytes=..., bytes_per_frame=524288
```

That is expected behavior. Frames touched by byte gaps are skipped so FFT is not run on zero-filled corrupted data.

If `processing_drops` increases while `lost_packets` stays low, packet receive is keeping up but FFT/display processing is falling behind. Increase `--processing-queue-size`, reduce display updates, or run with `--display none`.

## Common Mitigations For Byte Gaps

- Use the DCA1000 packet delay recorded in `setup.json`; the current setup uses `25 us`.
- Use direct wired Ethernet between PC and DCA1000.
- Avoid Wi-Fi, VPN routing, switches, and busy adapters during capture.
- Increase Ethernet adapter receive buffers in Windows Device Manager if needed.
- Run with `--display none` while validating packet-loss behavior.
- Use `--display-update-every 2` or `3` if plotting causes packet gaps.

## Useful Commands

Show command-line options:

```powershell
python rawdatacapture\livedatacapture.py --help
```

Run with faster Ctrl+C polling:

```powershell
python rawdatacapture\livedatacapture.py --socket-timeout 0.1
```
