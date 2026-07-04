# Live Raw ADC Capture Architecture

Target hardware: TI IWR6843ISK-ODS with DCA1000EVM.

Goal: receive raw LVDS ADC samples through DCA1000 Ethernet, reconstruct frames in Python, run custom signal processing, and update a live display without relying on TI post-processing output.

## Current Starting Point

`livedatacapture.py` is now a small frame-level data-plane receiver:

- Binds a UDP socket to host IP `192.168.33.30`, data port `4098`.
- Receives DCA1000 Ethernet packets.
- Uses a short UDP socket timeout so Ctrl+C can stop the receiver cleanly.
- Parses the 10-byte DCA1000 inline packet header when sequence numbering is enabled.
- Tracks packet sequence numbers and reports lost, duplicate, and out-of-order packets.
- Uses the DCA1000 byte-count field to maintain a continuous payload byte stream.
- Reads a radar `.cfg`, mmWave Studio XML, or mmWave Studio JSON to derive frame dimensions.
- Accumulates UDP payload bytes until a full radar frame is available.
- Marks frames touched by byte gaps as invalid and skips FFT on those frames.
- Converts only complete frame bytes to a complex radar cube shaped `[chirp, rx, sample]`.
- Runs a first-pass range FFT across the ADC sample axis and prints the strongest range in meters plus the source FFT bin.
- Optionally shows a live range profile or range-Doppler heatmap.
- Appends terminal status output to a log file for later review.

That keeps packet capture separate from packet-sized processing. A live display still needs a proper UI loop, but FFT code now has a frame-sized input boundary instead of a single UDP packet boundary.

Example:

```powershell
python rawdatacapture\livedatacapture.py --config path\to\radar.cfg
```

or:

```powershell
python rawdatacapture\livedatacapture.py --config .\rawdatacapture\mmwave_setup.xml
```

By default, terminal status output is also appended to:

```text
rawdatacapture\livedatacapture.log
```

Each log line includes a local timestamp, and each run includes an explicit start/end marker.

To choose a different log file:

```powershell
python rawdatacapture\livedatacapture.py --config .\rawdatacapture\mmwave_setup.xml --log-file .\rawdatacapture\capture_run.log
```

To show a simple live range profile:

```powershell
python rawdatacapture\livedatacapture.py --config .\rawdatacapture\mmwave_setup.xml --display range
```

The live range X-axis is limited to `20 m` by default. Use `--max-range-m N` to change it, or `--max-range-m 0` to show the full computed axis.

To show a first-pass range-Doppler heatmap:

```powershell
python rawdatacapture\livedatacapture.py --config .\rawdatacapture\mmwave_setup.xml --display range-doppler
```

The Matplotlib display runs in a separate process with a one-item latest-frame queue. The capture loop never waits for plotting; if the UI falls behind, old display payloads are discarded and only the newest range profile or heatmap is shown. Use `--display none` for packet-loss testing and `--display-update-every N` if plotting still causes packet gaps. If the Matplotlib window looks unresponsive, give the GUI more event-loop time:

```powershell
python rawdatacapture\livedatacapture.py --config .\rawdatacapture\mmwave_setup.xml --display range --display-update-every 2 --display-pause 0.05
```

## Architecture Overview

```text
IWR6843ISK-ODS
  RF frontend + ADC
  LVDS raw data out
        |
        | 60-pin high-speed connector / LVDS lanes
        v
DCA1000EVM
  FPGA packetizer
  Ethernet UDP stream
        |
        | UDP data packets to host: 192.168.33.30:4098
        | UDP config/control to DCA1000: 192.168.33.180:4096
        v
Host PC
  capture service
    - DCA1000 control
    - UDP packet receiver
    - sequence and byte-count validation
    - frame byte buffer
    - raw LVDS reshape
  processing service
    - range FFT
    - Doppler FFT
    - angle estimation / beamforming
    - detection / tracking
  display service
    - range profile
    - range-Doppler heatmap
    - point cloud / occupancy view
```

## Control Plane

The control plane starts and synchronizes the radar sensor and DCA1000.

Recommended first milestone:

1. Use TI mmWave Studio or the Radar Toolbox DCA1000 CLI to configure the DCA1000.
2. Use the IWR6843 command/config UART or mmWave Studio to configure the radar chirps and enable LVDS streaming.
3. Run the Python receiver only as the live data consumer.

Later milestone:

1. Move DCA1000 control into Python, or call the TI CLI from Python.
2. Move radar serial configuration into Python.
3. Start DCA1000 recording before `sensorStart`, because the capture card must already be armed when frame data begins.

Known local references:

- `mmwave_studio_02_01_01_00/mmWaveStudio/PostProc/cf.json`
- `radar_toolbox_4_00_00_05/tools/Adc_Data_Capture_Tool_DCA1000_CLI/`
- `radar_toolbox_4_00_00_05/tools/Adc_Data_Capture_Tool_DCA1000_CLI/gui/rawDataCaptureGUI_DCA1000CLI.m`

The local TI config examples use:

- Host/system IP: `192.168.33.30`
- DCA1000 IP: `192.168.33.180`
- DCA1000 config port: `4096`
- DCA1000 data port: `4098`
- Packet delay: use `100 us` for the current 25 FPS, 16-bit complex, 4-RX frame size
- Raw LVDS capture mode
- Sequence number enabled

The DCA1000 packet delay should be set to `100 us` for the current 25 FPS target. With 256 ADC samples, 4 RX channels, 128 chirps per frame, and 16-bit complex samples, each frame is 524,288 bytes. At 25 FPS this needs about 9,000 DCA1000 payload packets per second, so a 200 us packet delay throttles the stream to roughly half the required rate. After changing to `100 us`, validate the run by checking that `lost_packets`, `byte_gaps`, and `invalid_frames` stay low.

## Data Plane

The data plane is what `livedatacapture.py` starts to implement.

### UDP Packet Receiver

Responsibilities:

- Bind to the host Ethernet interface and DCA1000 data port.
- Receive UDP packets continuously.
- Parse the 10-byte DCA1000 packet header instead of simply discarding it.
- Track sequence numbers to detect lost or out-of-order packets.
- Use byte-count metadata to place payloads into a continuous raw byte stream.
- Push validated payload bytes into a frame buffer.
- Send complete valid frames to a separate processing worker so FFT/display work does not block UDP receive.

Current script status:

```text
implemented:
  socket bind
  packet receive
  Ctrl+C-friendly socket timeout
  DCA1000 header parse
  sequence tracking
  packet loss counters
  byte-count based payload stream
  radar .cfg / mmWave Studio XML / JSON dimension parsing
  frame buffering
  separate frame-processing worker
  invalid-frame tracking for packet gaps
  skip FFT for incomplete frames
  complete-frame int16 conversion
  LVDS/IQ reshape to [chirp, rx, sample]
  first-pass range FFT peak-bin reporting
  optional live range-profile display
  optional live range-Doppler heatmap display

missing:
  packet reordering beyond duplicate/overlap trimming
  full calibrated range/Doppler processing
```

### Frame Builder

The frame builder should only emit complete radar frames.

For xWR68xx/IWR6843 raw capture, the local TI MATLAB reader assumes:

- DCA1000 capture hardware.
- Raw data logging mode.
- ADC-only LVDS packet format.
- 16-bit ADC samples.
- Complex ADC data.
- 2 LVDS lanes for xWR16xx/xWR18xx/xWR68xx.
- RX channel count from the radar config.

The current `mmwave_setup.xml` uses 16-bit complex ADC output:

```text
bitsVal = 2      -> 16-bit ADC samples
formatVal = 1    -> complex ADC samples
IQSwap = 0       -> normal I/Q order
```

That means each complex radar sample is one 16-bit I value plus one 16-bit Q value:

```text
bytes_per_complex_sample = 2 bytes I + 2 bytes Q = 4 bytes
```

Frame size should be computed from the radar profile/chirp/frame config:

```text
num_chirps_per_frame = num_loops * (chirp_end_idx - chirp_start_idx + 1)
bytes_per_complex_sample = 4
bytes_per_chirp = num_adc_samples * num_rx_channels * bytes_per_complex_sample
bytes_per_frame = num_chirps_per_frame * bytes_per_chirp
```

The frame builder API can look like:

```text
add_payload(packet_payload) -> zero or more complete raw frames
```

It should keep leftover bytes between UDP packets because a DCA1000 packet boundary is not the same thing as a chirp or frame boundary.

### LVDS And IQ Reshape

The TI reader reshapes xWR68xx 2-lane LVDS raw data as groups of four `int16` values:

```text
rawData4 = reshape(rawData, [4, len(rawData) / 4])
I = rawData4 rows 1:2 flattened
Q = rawData4 rows 3:4 flattened
complex = I + j*Q, subject to iqSwap setting
```

Then it reshapes the stream into:

```text
[num_chirps_per_frame, num_rx_channels, num_adc_samples]
```

For the current `mmwave_setup.xml`, the capture path assumes non-interleaved RX channels:

```text
channel_interleave = False
```

The first Python implementation keeps this in `livedatacapture.py` as `frame_bytes_to_radar_cube(...)`. As the project grows, this should move into a dedicated module so changes in lane mode, IQ swap, channel interleave, or profile config do not leak into the signal processing code.

## Processing Pipeline

The processing pipeline should consume full frame tensors, not individual UDP packets.

Recommended stages:

```text
raw frame
  -> DC removal / calibration
  -> range window
  -> range FFT
  -> Doppler window
  -> Doppler FFT
  -> static clutter removal
  -> CFAR / thresholding
  -> angle estimation / beamforming
  -> point cloud
  -> tracker or application-specific logic
  -> display model
```

A minimal first live display can stop earlier:

```text
raw frame
  -> reshape
  -> range FFT
  -> magnitude average across chirps/RX
  -> live range profile plot
```

Then expand to range-Doppler, then point cloud.

## Proposed Python Modules

```text
livedatacapture.py
  Main entry point. Wires config, receiver, processor, and display.

capture/dca1000_receiver.py
  UDP socket, packet header parsing, sequence/loss reporting.

capture/frame_builder.py
  Converts packet payload stream into complete frame byte arrays.

radar/config.py
  Parses radar .cfg, mmWave Studio XML, or JSON into derived dimensions.

radar/lvds.py
  Converts frame bytes into complex ADC cube:
  [chirp, rx, sample].

processing/range_fft.py
  First-stage custom signal processing.

processing/doppler_fft.py
  Second-stage custom signal processing.

display/live_view.py
  Matplotlib, PyQtGraph, OpenGL, or web-socket display adapter.
```

For low-latency display, PyQtGraph is a practical first choice on Windows. For browser-based display, keep the capture/processing pipeline in Python and publish processed frames over WebSocket.

## Threading Model

Use bounded queues between stages so display slowness does not block packet capture.

```text
UDP receiver thread
  -> packet queue

packet assembler thread
  -> frame queue

processing thread
  -> display queue

display/UI thread
```

Current implementation note: `livedatacapture.py` still performs UDP receive, frame assembly, and FFT in the main capture process, but Matplotlib rendering is split into a separate `LiveDisplay` process fed by a one-item latest-frame queue. This avoids Qt/Matplotlib event-loop errors on Windows.

Policy:

- Never block the UDP receiver on plotting.
- Drop old display frames if the UI falls behind.
- Do not drop raw packet payloads unless the system is overloaded; log the drop.
- Count packet loss using DCA1000 sequence numbers.

## Configuration Inputs

The capture code needs these values before it can interpret bytes correctly:

```text
network:
  host_ip
  dca_ip
  dca_config_port
  dca_data_port

radar:
  num_adc_samples
  num_rx_channels
  chirp_start_idx
  chirp_end_idx
  num_loops
  num_tx_channels
  adc_bits
  complex_or_real
  iq_swap
  channel_interleave
  lvds_lanes
  sample_rate
  frequency_slope
  start_frequency
  idle_time
  ramp_end_time
```

These should come from the same `.cfg`, mmWave Studio XML, or mmWave Studio JSON used to program the radar, not from hardcoded constants.

## First Implementation Milestones

1. Keep TI tools responsible for radar and DCA1000 setup.
2. Replace per-packet printing in `livedatacapture.py` with DCA1000 header parsing and sequence/loss counters.
3. Add a frame builder using computed `bytes_per_frame`.
4. Port the 2-lane LVDS reshape logic from the local TI MATLAB reader to NumPy.
5. Produce a live range-profile plot from complete frames.
6. Add range-Doppler processing.
7. Add angle estimation and point-cloud display.
8. Move radar/DCA1000 setup into Python once the data path is stable.

## Important Design Notes

- A UDP packet is not a radar frame.
- A complete frame boundary must be derived from radar configuration and accumulated byte counts.
- The DCA1000 packet header is valuable; use it for ordering and loss detection.
- Raw capture over DCA1000 is much heavier than UART TLV output, so keep capture, processing, and display decoupled.
- The IWR6843ISK-ODS antenna geometry matters at the angle-estimation stage. Keep geometry/calibration separate from basic range/Doppler processing.
- Validate the Python reshape by comparing one saved `.bin` capture against TI's MATLAB `rawDataReader.m` output before trusting live visualization.
