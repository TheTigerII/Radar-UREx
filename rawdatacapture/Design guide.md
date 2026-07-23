# Live Raw ADC Capture Design

This document describes the code currently implemented for live raw ADC capture
from a TI IWR6843ISK-ODS through a DCA1000EVM. It is an implementation guide,
not a description of TI UART TLV output or mmWave Studio post-processing.

Range FFT, TDM separation, and Doppler FFT use numerically equivalent
complex64 SciPy kernels in `openradar_backend.py`, validated against the pinned
OpenRadar implementation. Hann windows are cached, and the unused OpenRadar
complex128/log-magnitude intermediates are not created. OS-CFAR uses a local
vectorized ordered-window implementation with cached training indices and
scale factors. Raw DCA1000 decoding and the
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
       dedicated receiver thread drains the UDP socket
       bounded in-process packet queue absorbs short host stalls
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

Capture, processing, and display use bounded queues. The UDP receiver thread
never waits for frame assembly, FFT, logging, or plotting. A full packet queue
increments `receiver_queue_drops`; a full processing queue increments
`processing_frames_dropped`; the one-item display queue discards stale display
results in favor of the newest one.

### Processes and queue sizes

- UDP receiver thread: socket receive only.
- Main process: packet dequeue and `FrameBuffer` assembly.
- `RadarFrameProcessor`: frame conversion, DSP, logging, and optional raw write.
- `RadarLiveDisplay`: Matplotlib UI for any display other than `none`.
- Packet queue: `--packet-queue-size`, default 8,192 datagrams.
- Processing queue: `--processing-queue-size`, default 32 frames.
- Display payload queue: one result.
- Processor log queue: 1,000 messages.

Routine successful frames and display latency are silent. Capture statistics
are emitted immediately whenever an error counter changes. Shutdown emits
capture, processing, and display summaries, including the number of updates
actually rendered by the GUI. The processor ignores the parent process's
SIGINT, consumes the queue sentinel, and drains all frames queued before that
sentinel. Its final report includes aggregate p50, p95, and maximum timings for
range FFT, Doppler, dynamic detection/CFAR, static detection, clustering,
micro-Doppler, serialization, and total processing.

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
`packetDelay_us`. The current 50-us delay remains comfortably above the data
rate required by the 33.33-ms radar profile while smoothing Ethernet bursts.
The supported hardware range is validated as 5 to 500 us. When packet sequence
headers are disabled, the receiver uses synthetic sequence and byte counts; it
can assemble received bytes but cannot detect network loss from DCA1000
metadata.

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
8. Runs deterministic local DBSCAN for normal-size clouds, with a scikit-learn
   fallback for unusually large clouds, and sends both the original points and
   cluster centers to the display process.
9. Updates one persistent dynamic 3D target track. Initial acquisition uses the
   strongest cluster or point; later updates use gated nearest-neighbor
   association against a constant-velocity prediction.

In parallel, the static-change path sums angle-FFT power from the centered
Doppler bin and its two neighbors. It applies the virtual-array mapping and one
vectorized 32-by-32 angle FFT across every valid range bin on every processed
point-cloud update, producing a `[range, elevation, azimuth]` cube. SciPy's FFT
keeps the complex64 input precision instead of promoting it to complex128.
Power is summed across the three Doppler bins before the smaller float32 power
cube is shifted, reducing allocation and memory-copy cost without changing the
FFT dimensions or update rate.

The first 30 processed detection updates are discarded as warm-up. The next
90 target-free updates build a median reference, a per-range power floor, and
a robust per-cell noise estimate from the median absolute deviation in log
power. Each live change map is corrected by its per-range median to remove
common receiver-gain drift and updated with a 0.35-rate temporal EMA. A
detection must exceed both the default 6 dB absolute threshold and four times
the learned cell variation. Unprotected reference and noise cells then adapt
at 0.01 per processed frame. Range/FOV gating occurs before local-maximum
testing; when the threshold produces no candidates, the full 3D maximum scan
is skipped. The remaining cells are capped at the 256 strongest before XYZ
conversion and DBSCAN. Returned raw candidates are
`[x, y, z, magnitude_db, change_db]`.

Raw static candidates are diagnostic activity, not targets. The static tracker
receives DBSCAN clusters with a default minimum of one point because the 3D
local-maximum pass has already reduced a reflector to one spatial candidate.
A cluster can start a static track only when a confirmed dynamic track moved
at least 0.3 m within the preceding 30 processed frames, the cluster is within
0.75 m of the last dynamic position, and it remains associated for three
consecutive frames. Deployments can restore stricter same-frame density with
`--static-cluster-min-samples 3`. Handoff remains eligible for 60 frames. The
selected target's range, azimuth, and elevation cells are protected by ±2 bins
while validated; motion-only protection is released after 30 consecutive
misses. The static tracker also releases after 30 misses, allowing removed
objects to be absorbed into the adaptive map. Only the exact DBSCAN members of
the validated target are displayed and saved. All suppressed raw activity is
reported separately as `static_candidate_count`.

For four RX channels and TX masks corresponding to TX1-TX3, the virtual grid
uses the IWR6843ISK-ODS antenna layout and applies a sign inversion to RX2 and
RX3. Other layouts fall back to a generic grid. Returned coordinates use
X=left/right, Y=forward, and Z=elevation; the estimate is uncalibrated.

### Combined point cloud and micro-Doppler

The `point-cloud-micro-doppler` display computes one Doppler cube for each
processed update and reuses it for the dynamic point cloud, static angle cube,
and range-gate selection. A measured confirmed dynamic track has priority
while the target is moving. A validated static track takes over after handoff;
a predicted dynamic track is used only when neither is measured. During a
short dynamic detection gap, a constant-velocity prediction maintains the gate
and the tracked-target marker is shown smaller and translucent. No static-only
clutter or arbitrary range fallback can activate micro-Doppler.

The micro-Doppler branch reshapes the chronological chirps into explicit loop
and TX-slot axes. It applies independent slow-time FFTs to every TX slot, RX
channel, and gated range bin using 64-loop Hann windows, a 32-loop hop, and a
128-point FFT. TX, RX, and range-bin powers are summed only after the FFT, so
the three TX signals are not coherently merged and require no inter-TX phase
calibration. A 128-loop frame produces three short-time spectra. Each spectrum
is appended to a 150-window history in the frame-processing process. Sending
the complete history through the latest-only display queue prevents GUI queue
replacement from creating holes in the visible spectrogram. The spectrogram
follows the selected target by spatial continuity rather than the dynamic or
static source label. History remains frozen through gaps of up to 30 processed
updates and continues across a dynamic-to-static handoff within 0.75 m. A
newly selected target beyond that gate or after a longer gap starts fresh
history. An explicit static gate retains the centered zero-Doppler bin. The
Matplotlib image uses a visible-spectrum `turbo` color map, running from dark
blue at the fixed 60 dB minimum to red at the fixed 120 dB maximum rather than
rescaling each history update. The combined layout uses one shared magnitude
colorbar for both axes.
The colorbar occupies a dedicated narrow grid column between the point cloud
and micro-Doppler axes, with a spacer on its right so it does not crowd the
spectrogram's Doppler-axis label. Both plot titles report measured updates per
second. Supported Matplotlib backends use artist-level blitting for the dynamic
scatters, spectrogram, and titles so the static 3D axes and colorbar do not
require a full redraw every update.

## Processed and Raw Recording

`--processed-output` streams version-3 newline-delimited JSON. Its first record
describes the radar configuration, data axes, and static-reference settings.
Each subsequent update contains the dynamic and static point clouds, their
DBSCAN clusters, static-reference and validation state, suppressed
`static_candidate_count`, target source and track, current micro-Doppler
spectrum, every short-time window generated from the frame, and the selected
range gate. `static_points` and `static_clusters` contain validated target data
only. Existing dynamic `points` and `clusters` fields keep their version-1
layouts. The writer does not duplicate the rolling display history. When it
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
