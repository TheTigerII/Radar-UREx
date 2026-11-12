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

Two experimental choices are exposed alongside those controls:

- **Mode 3 — Beam-space MIMO TBD/STT:** the existing three-TX/four-RX TDM
  waveform forms 12 calibrated virtual channels. A +/-60-degree beam/range
  cube is integrated over eight frames before focused 2-degree tracking. This
  lane uses neither PMM folding nor a per-frame CFAR point declaration.
- **Mode 4 — Enhanced conventional:** the inherited Mode 1 waveform, tracker
  and CNN with independently switchable calibrated threshold, averaged clutter
  reference, coherent receive-beam detection and eight-frame integration.

The earlier two-TX phased-look and literature PMM-TBD experiments remain
available by CLI for reference but are intentionally hidden from the final
four-mode GUI.

The application includes a portable PPI, live calibration/readiness banner,
range/azimuth/elevation and relative-altitude display, target-attached class
label, and processed/inference/performance logging. TensorRT engines were
rebuilt for the Jetson ARM platform.

## Technical basis

Lowering Mode 3's chirp slope extends sampled unambiguous range to about
93.8 m but does not improve transmitted power or guarantee detection. Its
coherent 12-channel combination changes effective receive processing gain
(ideal upper bound 10.8 dB over one channel), while its constrained temporal
path lowers the evidence required for a persistent target. Mode 4 changes
detection processing while retaining Mode 1's RF waveform. Since received
power varies as `R^-4`, even array gain gives a much smaller range factor and
field evidence is essential.

## Verification and classification boundary

The recovered conventional suite passes 270 tests and the enhanced PMM/GUI
suite passes 36 tests. Synthetic Mode 3 tests cover TDM phase compensation,
wide-angle acquisition, stale-echo rejection, angle recovery and the complete
36 x 64 classifier segment. On the Jetson, random-cube Mode 3 tracking measured
about 75 ms median / 77 ms p95 within the 100 ms frame period. The existing
TensorRT LSTM accepts and classifies the Mode 3 tensor without a new model or
dataset. Its accuracy is still experimental cross-profile accuracy and will be
reported only from labelled field runs. Mode 3 reuses the checked-in channel
correction; only the ordinary target-free run-start background is required.

## Final matched test — results pending

Each mode will use the same radar pose, target, marked route, weather and
five-minute duration. Tests will progress through 3, 5, 8, 10, 12, 15 and
20 m and farther only after stable nearer operation, with three trials per
point and a 60-second empty-scene false-track interval.

| Mode | Maximum repeatable range | Pd | False tracks/min | Range/angle error | Classification accuracy | p95 processing |
|---|---:|---:|---:|---|---:|---:|
| 1 Original conventional |  |  |  |  |  |  |
| 2 Original PMM |  |  |  |  |  |  |
| 3 Beam-space MIMO TBD/STT |  |  |  |  |  |  |
| 4 Enhanced conventional |  |  |  |  |  |  |

**Conclusion after final test:**  

