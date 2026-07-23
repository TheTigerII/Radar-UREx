# Live Raw ADC Capture Design

This document describes the code currently implemented for live raw ADC capture
from a TI IWR6843ISK-ODS through a DCA1000EVM. It is an implementation guide,
not a description of TI UART TLV output or mmWave Studio post-processing.

Range FFT, TDM separation, and Doppler FFT use the pinned OpenRadar dependency
through `openradar_backend.py`. OS-CFAR uses a local vectorized ordered-window
implementation to keep up with the live stream. Raw DCA1000 decoding and the
IWR6843ISK-ODS-specific planar antenna mapping remain local because OpenRadar's
generic XYZ implementation assumes a different virtual antenna layout.
All live plots, including the 3D point cloud and combined point-cloud/
micro-Doppler view, run with Matplotlib in the display child process.

## Runtime Components

The integrated entry point is `run.py`. It launches two independent programs:

```text
run.py
  +-- livedatacapture.py   UDP receive, frame assembly, DSP, display, raw saving
  +-- startup.py           radar serial control and DCA1000 UDP control
```

`run.py` prompts for a duration before initialization (3 minutes by default;
zero means unlimited), starts the receiver first, waits one second, and then
starts the hardware controller. When the deadline expires or Ctrl+C is pressed,
it stops `startup.py` first so `sensorStop` and DCA1000 `RECORD_STOP` are
attempted before capture is closed.

The programs can also be run manually in two terminals. `startup.py` does not
embed the real capture receiver: its `--capture-backend` currently supports
only `dry-run`.

## Capture Data Flow

```text
DCA1000 UDP packets (host 192.168.33.30, port 4098)
  -> main capture process
       parse optional 10-byte sequence/byte-count header
       track packet order and loss
       assemble a continuous byte stream
       split stream into configured frame sizes
       reject frames containing ordinary byte gaps
       enqueue valid frames in a bounded processing queue
  -> RadarFrameProcessor process
       optionally save raw frame
       reshape int16 LVDS data to [chirp, rx, sample]
       range FFT and selected display DSP
       enqueue latest display result
  -> RadarLiveDisplay process (when enabled)
       update an existing Matplotlib artist
```

Capture, processing, and display use bounded queues. The UDP receiver never
waits for FFT or plotting. A full processing queue increments
`processing_frames_dropped`; the one-item display queue discards stale display
results in favor of the newest one.

### Processes and queue sizes

- Main process: socket receive and `FrameBuffer` assembly.
- `RadarFrameProcessor`: frame conversion, DSP, logging, and optional raw write.
- `RadarLiveDisplay`: Matplotlib UI for any display other than `none`.
- Processing queue: `--processing-queue-size`, default 4 frames.
- Display payload queue: one result.
- Processor log queue: 1,000 messages.

Routine successful frames and display latency are silent. Capture statistics
are emitted only when an error counter changes.

## Configuration

`livedatacapture.py` accepts either an SDK CLI `.cfg` or mmWave Studio `.json`
through `--config`. It derives:

```text
num_adc_samples
num_rx_channels
num_chirps_per_frame
num_loops
num_chirps_per_loop
tx_channel_masks (from .cfg when available)
iq_swap
channel_interleave
lvds_lanes
sample_rate_ksps
frequency_slope_mhz_per_us
bytes_per_frame
```

The frame-size calculation is:

```text
num_chirps_per_frame = num_loops * chirps_per_loop
bytes_per_frame = num_adc_samples * num_rx_channels
                  * num_chirps_per_frame * 4
```

The factor 4 is two bytes for I plus two bytes for Q. The current reshape code
supports complex, 16-bit, two-lane LVDS data.

`setup.json` supplies DCA1000 settings including `packetSequenceEnable` and
`packetDelay_us`. When packet sequence headers are disabled, the receiver uses
synthetic sequence and byte counts; it can assemble received bytes but cannot
detect network loss from DCA1000 metadata.

## Packet and Frame Integrity

With sequence headers enabled, `SequenceTracker` records lost, duplicate, and
out-of-order packet numbers. `FrameBuffer` uses the DCA1000 byte count to detect
gaps and overlaps.

- A small forward gap is zero-filled only to preserve frame alignment. Any
  frame touched by those bytes is marked invalid and is not processed or saved.
- A forward discontinuity larger than one complete frame is treated as an
  implausible stream jump. The partial assembly is discarded and the buffer is
  resynchronized at the current packet. This bounds memory use and increments
  `stream_resyncs`.
- Fully duplicate payloads are ignored.
- Overlapping payload prefixes are trimmed.
- Backward byte counts are treated as out of order and ignored.

The buffer can return more than one frame for one packet, although normal
DCA1000 payloads are much smaller than a frame.

## Raw Data Conversion

`dsp.frame_bytes_to_radar_cube()` reads little-endian `int16` groups of four.
For normal IQ order, the first two values provide I samples and the next two Q
samples; `iq_swap` reverses those roles. The resulting complex stream is
reshaped to:

```text
[num_chirps_per_frame, num_rx_channels, num_adc_samples]
```

Both interleaved and non-interleaved channel layouts are handled after the
two-lane IQ conversion. Configuring any LVDS lane count other than two raises
`NotImplementedError` during conversion and is rejected by startup preflight.

## Implemented DSP

The DSP in `dsp.py` is intended for live visualization, not calibrated target
measurement.

### Range profile

1. Apply a Hann window across ADC samples.
2. Run an FFT across the sample axis.
3. Average absolute FFT magnitude over chirps and receivers.

If sample rate and frequency slope are present, the range-bin spacing is:

```text
c * sample_rate / (2 * slope * num_adc_samples)
```

### Range-Doppler

The range FFT is reshaped to `[loop, chirp-within-loop, rx, range]`. Doppler FFT
and `fftshift` are applied over loops. The displayed heatmap averages magnitude
over chirp-within-loop and RX, then converts it to dB. Its Y-axis is a Doppler
bin index, not velocity in m/s.

### Point cloud

The point-cloud path:

1. Forms mean range-Doppler power.
2. Applies an adaptive power-domain clutter map. During the initial warm-up it
   learns every cell using an exponential moving average and emits no point
   detections. It then divides every cell by its learned background power before
   CFAR and applies a default 6 dB minimum target-to-background ratio. This
   keeps the normalized background near one instead of producing large regions
   of zero-valued CFAR training cells. Range and Doppler guard neighborhoods
   around current detections are frozen during map updates so targets are not
   immediately absorbed. An FFT-shape change resets the map.
3. Runs two-dimensional OS-CFAR with Doppler wrapping and a default requested
   false-alarm probability of `1e-3` per axis.
4. Keeps Doppler-local peaks without suppressing adjacent range detections, then
   removes detections below 0.25 m.
5. Rejects detections beyond 10 m.
6. Maps all candidate cells into virtual antenna grids and estimates their
   directions with one batched 32-by-32 2D FFT. Unlimited point-cloud output
   preserves detection order and skips the unnecessary power sort.
7. Applies ±60-degree azimuth and elevation gates and keeps all in-FOV
   detections.
8. Runs spatial DBSCAN on XYZ points and sends both the original points and
   cluster centers to the display process.
9. Updates one persistent 3D target track. Initial acquisition uses the
   strongest cluster or point; later updates use gated nearest-neighbor
   association against a constant-velocity prediction.

For four RX channels and TX masks corresponding to TX1-TX3, the virtual grid
uses the IWR6843ISK-ODS antenna layout and applies a sign inversion to RX2 and
RX3. Other layouts fall back to a generic grid. Returned coordinates use
X=left/right, Y=forward, and Z=elevation; the estimate is uncalibrated.

### Combined point cloud and micro-Doppler

The `point-cloud-micro-doppler` display computes one Doppler cube for each
display update and reuses it for both outputs. The point-cloud path runs as
described above. The tracked target supplies the center of a five-bin range
gate. During a short detection gap, a constant-velocity prediction maintains
the gate and the green tracked-target marker is shown smaller and translucent.
The track is removed after 10 consecutive missed display updates. When no track
exists, the gate follows the strongest non-zero-Doppler range inside the
configured range limit.

The micro-Doppler branch reshapes the chronological chirps into explicit loop
and TX-slot axes. It applies independent slow-time FFTs to every TX slot, RX
channel, and gated range bin using 64-loop Hann windows, a 32-loop hop, and a
128-point FFT. TX, RX, and range-bin powers are summed only after the FFT, so
the three TX signals are not coherently merged and require no inter-TX phase
calibration. A 128-loop frame produces three short-time spectra. Each spectrum
is appended to a 150-window history in the frame-processing process. Sending
the complete history through the latest-only display queue prevents GUI queue
replacement from creating holes in the visible spectrogram. The spectrogram
follows the single tracked target while its association remains valid. It can
still change targets after track loss and reacquisition. The Matplotlib image
uses a visible-spectrum `turbo` color map, running from dark blue at the fixed
40 dB minimum to red at the fixed 120 dB maximum rather than rescaling each
history update. The combined layout uses one shared magnitude colorbar for both axes.
The colorbar occupies a dedicated narrow grid column between the point cloud
and micro-Doppler axes, with a spacer on its right so it does not crowd the
spectrogram's Doppler-axis label. Both plot titles report measured updates per
second. Supported Matplotlib backends use artist-level blitting for the dynamic
scatters, spectrogram, and titles so the static 3D axes and colorbar do not
require a full redraw every update.

## Processed and Raw Recording

`--processed-output` streams newline-delimited JSON. Its first record describes
the radar configuration and data axes. Each subsequent update contains the
post-CFAR XYZ point cloud, DBSCAN clusters, target-track state, current
micro-Doppler spectrum, every short-time window generated from the frame, and
the selected range gate. The writer does not duplicate the rolling display
history. When it
is enabled, combined point-cloud/micro-Doppler processing runs even with a
different display mode or `--display none`.

`--raw-output` writes only complete valid frames, consecutively and without
DCA1000 headers. At normal shutdown a JSON sidecar records dimensions, sample
format, processing parameters, counts, and paths. An explicit `--raw-metadata`
overrides the sidecar name. Integrated `run.py` enables processed output by
default and leaves raw recording disabled unless `--raw-output` is supplied.

Terminal messages are appended to `livedatacapture.log` by default. Raw files
and logs have no rotation or size limit, so long deployments must monitor disk
space externally.

## Current Limitations

- No packet reordering; only duplicate and overlap handling.
- Only complex 16-bit, two-lane LVDS reshape is implemented.
- No calibrated velocity axis, antenna calibration, or phase calibration.
- Tracking supports one target only, uses uncalibrated XYZ positions, and has
  no Doppler-assisted association or externally validated track identity.
- Range FFT includes the full complex FFT rather than selecting only a
  physically useful half-spectrum.
- The point cloud is a diagnostic visualization, not precision metrology.
- `startup.py` cannot itself run `livedatacapture.py`; use `run.py` or two
  terminals.
- Logs and raw captures grow until stopped or the filesystem fills.

## Design Rules for Future Changes

- Never interpret one UDP packet as one chirp or frame.
- Keep receiver work bounded and avoid blocking it on processing or display.
- Keep radar configuration and frame interpretation sourced from the same
  `.cfg`/JSON file.
- Preserve packet-loss and processing-drop counters when adding stages.
- Validate reshape and antenna mapping against known TI capture output before
  treating measurements as calibrated.
