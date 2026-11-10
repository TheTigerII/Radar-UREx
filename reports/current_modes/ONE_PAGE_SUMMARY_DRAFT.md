# Radar-UREx: portable COTS mmWave radar extension

**UREx one-page summary — draft; final field results intentionally blank**

## Objective

Recover both teammate-developed radar methods and test whether coherent array
processing or multi-frame integration can extend repeatable single-target
tracking of a drone-sized target on the existing IWR6843ISK-ODS, DCA1000EVM
and Jetson Orin Nano platform.

## System and contributions

The first inherited method is a conventional range-Doppler, ordered-statistic
CFAR, point-cloud/tracking and CNN pipeline. The second is the mmHawkeye-style
PMM method using spectrum folding, background subtraction, constrained
tracking, Capon angles and LSTM classification. Both are now selectable in one
touchscreen GUI.

Three experimental choices are also available:

- **Mode 3 — Enhanced MIMO PMM STT:** three sequential transmitters and four
  receivers form 12 virtual channels. Calibrated coherent beams cover +/-60
  degrees before Capon angle refinement.
- **Mode 4 — Two-TX phased STT:** two transmitters emit simultaneously with a
  quantised relative phase and four receivers combine coherently in one
  selected look direction (azimuth limited to +/-30 degrees).
- **Mode 6 — Literature MIMO TBD:** three-TX/four-RX long-range TDM-MIMO with
  constrained 8–12-frame PMM path integration for persistent weak evidence.

The application includes a portable PPI, live calibration/readiness banner,
range/azimuth/elevation and relative-altitude display, target-attached class
label, and processed/inference/performance logging. TensorRT engines were
rebuilt for the Jetson ARM platform.

## Technical basis

Lowering chirp slope extends the sampled unambiguous range to about 93.8 m but
does not improve transmitted power or guarantee detection. Mode 3 changes
effective receive processing gain through coherent combination (ideal upper
bound 10.8 dB over one channel). Mode 4 changes the directional transmit
pattern and coherently combines four receivers, within the sensor's hardware
limits. Mode 6 adds temporal processing gain by integrating a reachable weak
path. Since received power varies as `R^-4`, even array gain produces a much
smaller range factor, and field evidence is essential.

## Verification and classification boundary

The recovered conventional suite passes 252 tests and the enhanced PMM/GUI
suite passes 28 tests. A synthetic Mode 6 target with individual frames below
the ordinary threshold is acquired after eight frames at approximately 20 m,
with range and angles recovered. Jetson synthetic tracker timing is about
46 ms median / 52 ms p95 within a 100 ms frame period; LSTM TensorRT inference
is about 2.9 ms median / 4.4 ms p95. The available PMM model accepts the same
36 x 64 Doppler-time tensor, but its revised/cross-waveform accuracy is
unverified and will be reported only from labelled tests.

## Final matched test — results pending

Each mode will use the same radar pose, target, marked route, weather and
five-minute duration. Tests will progress through 3, 5, 8, 10, 12, 15 and
20 m and farther only after stable nearer operation, with three trials per
point and a 60-second empty-scene false-track interval.

| Mode | Maximum repeatable range | Pd | False tracks/min | Range/angle error | Classification accuracy | p95 processing |
|---|---:|---:|---:|---|---:|---:|
| 1 Original conventional |  |  |  |  |  |  |
| 2 Original PMM |  |  |  |  |  |  |
| 3 Enhanced MIMO STT |  |  |  |  |  |  |
| 4 Two-TX phased STT |  |  |  |  |  |  |
| 6 Literature MIMO TBD |  |  |  |  |  |  |

**Conclusion after final test:**  


