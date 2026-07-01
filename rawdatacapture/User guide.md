# Live Raw ADC Capture User Guide

This guide is for running `livedatacapture.py` with the IWR6843ISK-ODS and DCA1000EVM.

## Hardware And Tool Setup

- Connect the PC directly to the DCA1000 Ethernet port.
- Configure the PC Ethernet adapter with the host IP used by the script:

```text
192.168.33.30
```

- Configure the DCA1000/radar in mmWave Studio before starting Python.
- Keep the DCA1000 packet delay at:

```text
200 us
```

Lower packet delays previously caused byte gaps and dropped frames. `200 us` produced clean runs with:

```text
lost_packets=0
byte_gaps=0/0B
```

## Basic Capture

From the repository root, run:

```powershell
python rawdatacapture\livedatacapture.py --config .\rawdatacapture\mmwave_setup.xml
```

Expected startup output:

```text
Loaded radar config: RadarCaptureConfig(...)
Listening for live radar stream on 192.168.33.30:4098; bytes_per_frame=524288
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
python rawdatacapture\livedatacapture.py --config .\rawdatacapture\mmwave_setup.xml --log-file .\rawdatacapture\capture_run.log
```

## Live Range Profile

To show a simple live range profile:

```powershell
python rawdatacapture\livedatacapture.py --config .\rawdatacapture\mmwave_setup.xml --display range
```

The live range X-axis shows `0` to `20 m` by default. To change it:

```powershell
python rawdatacapture\livedatacapture.py --config .\rawdatacapture\mmwave_setup.xml --display range --max-range-m 10
```

Use `--max-range-m 0` to show the full computed range axis.

Plot axes:

```text
X-axis: range in meters
Y-axis: average FFT magnitude across chirps and RX channels
```

The script derives range in meters from `digOutSampleRate` and `freqSlopeConst` in the radar config. The terminal still prints `peak_range_bin` for debugging, but `peak_range_m` is the physical range estimate.

## Range-Doppler Heatmap

To show a first-pass range-Doppler heatmap:

```powershell
python rawdatacapture\livedatacapture.py --config .\rawdatacapture\mmwave_setup.xml --display range-doppler
```

This is an early diagnostic view, not yet a fully calibrated radar processing product.

## Display Responsiveness

Matplotlib runs in a separate display process with a one-item latest-frame queue. If plotting still feels sluggish, reduce update rate:

```powershell
python rawdatacapture\livedatacapture.py --config .\rawdatacapture\mmwave_setup.xml --display range --display-update-every 2 --display-pause 0.05
```

If packet gaps increase while plotting, try:

```powershell
--display-update-every 3
```

For packet-loss testing, run without display:

```powershell
python rawdatacapture\livedatacapture.py --config .\rawdatacapture\mmwave_setup.xml --display none
```

## Packet Loss And Invalid Frames

The script tracks DCA1000 sequence numbers and byte counts.

Healthy stats look like:

```text
lost_packets=0
byte_gaps=0/0B
invalid_frames=0
```

If packets are lost, you may see:

```text
Dropped frame: incomplete payload, gap_bytes=..., bytes_per_frame=524288
```

That is expected behavior. Frames touched by byte gaps are skipped so FFT is not run on zero-filled corrupted data.

## Common Mitigations For Byte Gaps

- Keep DCA1000 packet delay at `200 us`.
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
python rawdatacapture\livedatacapture.py --config .\rawdatacapture\mmwave_setup.xml --socket-timeout 0.1
```
