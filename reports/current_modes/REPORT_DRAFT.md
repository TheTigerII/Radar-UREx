# Extension of a Portable COTS mmWave Radar

**UREx Project Report — draft for final testing**  
**Student:** [Name / student number]  
**Supervisor:** [Name]  
**Date:** November 2026

## Abstract

This project extends a portable radar demonstrator built around the Texas
Instruments IWR6843ISK-ODS, DCA1000EVM and NVIDIA Jetson Orin Nano. Two
teammate-developed methods were recovered and retained: a conventional
range-Doppler/ordered-statistic CFAR pipeline with three-dimensional tracking
and a CNN classifier, and a periodic-micro-motion (PMM) pipeline based on the
mmHawkeye method with spectrum folding, background subtraction, constrained
tracking, Capon angle estimation and an LSTM classifier. Two experimental
modes are exposed for the final comparison: a distinct 12-channel beam-space
TDM-MIMO track-before-detect lane and an enhanced form of the conventional
pipeline. A unified touchscreen
GUI provides calibration/readiness state, a tactical top-down PPI, attached
classification labels and structured data/performance logging. The final
controlled outdoor comparison has not yet been completed; all result and
discussion fields are intentionally left blank.

## 1. Scope and contribution boundary

The inherited team work comprises the original conventional and PMM
processing pipelines, their data acquisition paths and their trained models.
The present work recovered both paths into a common operator application,
rebuilt Mode 3 and integrated the enhanced conventional Mode 4,
recompiled the TensorRT artifacts for the Jetson, added a touch-oriented PPI
workflow, and prepared a matched validation protocol. Exact authorship should
be confirmed with the team before submission.

No military-grade performance is claimed. The IWR6843ISK-ODS is a small COTS
three-transmitter/four-receiver sensor. Mode 3 is sequential-transmit TDM-MIMO
with host-side digital receive beams, not an AESA with independent high-power
transmit/receive modules or a multi-beam scheduler.

## 2. Hardware and inherited methods

The system consists of an IWR6843ISK-ODS 60 GHz radar, DCA1000EVM raw capture
card, CP2105 command UART and Jetson Orin Nano ARM computer. Radar samples are
received over UDP at `192.168.33.30:4098` and processed locally.

### 2.1 Original conventional pipeline (GUI Mode 1)

The conventional implementation forms range and TDM-aware Doppler FFTs, learns
clutter/static references, applies two-dimensional ordered-statistic CFAR,
estimates angle, produces XYZ detections, clusters them with DBSCAN and
maintains a constant-velocity target track. A target-gated micro-Doppler
sequence feeds a two-class CNN. This recovered lane retains its original
detection and classifier feature contract; only the PPI/operator presentation
was added.

### 2.2 Original PMM/mmHawkeye pipeline (GUI Mode 2)

The PMM implementation folds each Doppler spectrum over candidate periodicity
sizes, subtracts an initial projected background and follows a
continuity-constrained path through range and time. Track lifecycle logic and
particle filtering smooth the single dominant track. Capon/MVDR processing
estimates azimuth and elevation after range acquisition. A 36-frame by
64-Doppler-bin history feeds the inherited LSTM UAV/not-UAV classifier. Mode 2
uses the original approximately 20 m waveform and is the PMM control.

## 3. Added and recovered experimental modes

| GUI mode | RF configuration | Main processing change | Test purpose |
|---|---|---|---|
| 3 — Beam-space MIMO TBD/STT | 3 sequential TX, 4 RX, 96 chirps/frame, 9.994 MHz/us slope | Calibrated 12-channel beam/range/Doppler likelihood, eight-frame constrained integration and focused 2-degree refinement | Test receive-array and temporal processing gain independently of PMM/CFAR |
| 4 — Enhanced conventional | Same RF waveform, calibration, tracker and CNN as Mode 1 | Switchable calibrated threshold, averaged clutter reference, coherent 12-channel detection and eight-frame integration | Attribute any improvement while preserving the inherited conventional lane |

The final touchscreen GUI exposes four modes numbered consecutively from 1 to
4. Earlier phased-look and literature PMM-TBD experiments remain available by
CLI for reference but are excluded from the final matched comparison.

### 3.1 Mode 3: beam-space TDM-MIMO track-before-detect

Mode 3 uses the lower-slope long-range RF profile and the 12 virtual channels
created by three sequential transmitters and four receivers. It compensates
the Doppler-dependent phase offset between TDM transmissions, applies
the checked-in complex channel corrections, and evaluates a coarse bank of
azimuth/elevation steering vectors. The beam bank covers +/-60 degrees using
12-degree azimuth and 30-degree elevation spacing. The strongest non-zero
Doppler energy supplies a likelihood for each beam/range cell. A constrained
eight-frame path is integrated jointly through angle, range and time. This is
deliberately different from Mode 2's PMM folding and from per-frame CFAR. Once
a current persistent path is acquired, a local coherent beam grid refines the
direction at 2-degree spacing.

Ideal coherent combination of 12 equally illuminated channels has an upper
bound of `10 log10(12) = 10.8 dB` relative to one channel. Actual improvement
over the inherited non-coherent average will be smaller because of calibration
error, array geometry, target direction, motion, multipath and beam-grid loss.

Mode 3 does not change the RF profile or stored correction, so selecting it
does not require a new corner-reflector calibration. It supplies the inherited
LSTM with the selected beam's same 36-frame by 64-bin Doppler-time tensor.
This permits use of the existing model and dataset, although its accuracy is
still an experimental cross-profile field result.

### 3.2 Mode 4: enhanced conventional detection

Mode 4 retains Mode 1's waveform, range/channel calibration, target tracker
and CNN. Four operator-switchable detection stages permit an attributable
comparison: a threshold calibrated to measured false-alarm statistics, an
averaged clutter reference, TDM-compensated coherent 12-channel receive-beam
power, and eight-frame range-Doppler integration. It neither changes EIRP nor
claims military AESA operation.

## 4. What changed in the radar equation

The monostatic radar equation may be written as

`Pr = Pt Gt Gr lambda^2 sigma / ((4 pi)^3 R^4 L)`.

| Change | Radar-equation interpretation | Limitation |
|---|---|---|
| Lower chirp slope in Mode 3 | Does not directly change `Pt`, `Gt`, `Gr` or `sigma`; it changes beat-frequency-to-range mapping and extends the sampled unambiguous range to about 93.8 m | Coarser approximately 0.366 m bins and no guarantee that SNR is usable at 93.8 m |
| Mode 3 coherent receive combination | Increases effective receive processing gain `Gr` in a matched direction | Gain is below the 10.8 dB ideal and depends on calibration/beam match |
| Mode 3 multi-frame integration | Reduces the evidence/SNR required by the detector through temporal processing gain | Slower acquisition; persistent clutter can also integrate if calibration or guards are inadequate |
| Mode 4 detection changes | Change the processing loss/detection threshold rather than `Pt` or `Gt`; coherent receive power can increase effective `Gr` in a matched direction | Field gain and false-alarm rate must be measured against Mode 1 |

Because received power falls as `R^-4`, an array-processing gain produces only
the fourth-root range increase in the ideal radar equation. For example, even
an ideal 10.8 dB (factor 12) power gain corresponds to a theoretical range
factor of only `12^(1/4)`, about 1.86, before real losses. Displaying a 90 m
axis or lowering a threshold is not evidence of increased usable range.

## 5. GUI, PPI, calibration and logging

The four-mode GUI removes the need to enter launch commands. It selects the
correct profile, setup, capture directory and classifier artifacts for each
pipeline. Large controls, kinetic touch scrolling and a maximised layout suit
the portable monitor. The tactical PPI shows target trail, slant range,
azimuth, elevation/relative altitude, readiness state and an attached class
label.

Hardware range/channel calibration and run-time background calibration are
different operations. The IFM E23012 corner reflector can be used at a measured
1.5 m boresight position when the board, mounting, enclosure or waveform
calibration changes. The reflector and support must then be removed. At every
PMM run, a target-free 30-second interval learns background clutter; the GUI
shows accepted calibration frames and changes from amber to green only when
the tracker is ready.

Mode 1 and Mode 4 use the recovered CNN model. Mode 2 uses the inherited LSTM.
Mode 3 intentionally supplies that LSTM with the same 36 x 64 input shape and
labels, so no new training dataset is needed to operate it. Beam selection and
the long-range waveform do not exactly match the training fingerprint, so the
software logs cross-profile inference and treats accuracy as unverified until
the labelled final runs.

Each run can write processed JSONL, inference-evaluation JSONL and
runtime-performance JSONL. These contain track state, range/angle, detection
scores and thresholds, classification, timing, deadline misses, packet loss,
queue drops and resource use. Mode 3 additionally records its integrated TBD
score, threshold and window length. Short raw-ADC captures are required for
future offline re-beamforming because processed JSONL does not retain all
complex channel samples.

## 6. Software verification completed before field testing

- 270 recovered conventional-pipeline unit/regression tests pass on the
  Jetson.
- 36 enhanced radar/GUI tests pass, including RF profile/mode validation,
  TDM phase compensation, beam-space persistent-path acquisition, stale-spike
  rejection, angle recovery, PPI/GUI selection and classifier tensor shape.
- Both classifier engines were rebuilt locally for the Jetson. Conventional
  TensorRT parity passed on 128 stored windows with maximum probability error
  0.00259; measured inference was approximately 0.48 ms median and 1.92 ms p95.
- Random-cube Mode 3 tracking measured approximately 75 ms median and 77 ms
  p95 on the Jetson against a 100 ms frame period. Its existing TensorRT LSTM
  completed a 36 x 64 cross-profile inference without rebuilding the model.
- All four GUI selections validate their profile, setup and model files and
  produce prompt-free launch commands.

These checks reduce software risk but cannot replace a controlled radar field
test.

## 7. Final experimental method

The radar position, height, orientation, target, route, weather and five-minute
post-calibration duration will be held constant. Marked distances will begin at
3, 5 and 8 m and extend to 10, 12, 15, 20 m and farther only after stable nearer
operation. Each point will have at least three passes or hovers. A successful
detection will require a confirmed track for at least ten seconds or one
complete pass, rather than a single displayed sample. Each run will finish
with at least 60 seconds of empty scene to measure false tracks.

Mode 4 will be evaluated first with only its two low-cost statistical stages,
then with coherent beams, then with all four stages. Target-free background
calibration will be repeated after every mode change. The detailed procedure
and blank result sheet are in `aesa_radar/TEST_DAY_GUIDE.md`.

## 8. Results — to be completed after final test

| Mode | Maximum repeatable range (m) | Pd | False tracks/min | Median / p95 range error (m) | Az / el error (deg) | Track coverage | Classification accuracy | p95 processing (ms) | Packet or processing loss |
|---|---:|---:|---:|---|---|---:|---:|---:|---|
| 1 Original conventional |  |  |  |  |  |  |  |  |  |
| 2 Original PMM |  |  |  |  |  |  |  |  |  |
| 3 Beam-space MIMO TBD/STT |  |  |  |  |  |  |  |  |  |
| 4 Enhanced conventional |  |  |  |  |  |  |  |  |  |

**Observed conditions:**  


**Log and raw-capture filenames:**  


## 9. Discussion — to be completed after final test




## 10. Limitations

- All deployed modes currently report one dominant target; multi-target
  association is not established.
- Mode 3 processing gain requires the existing channel correction to remain
  valid. Its integrated path can still follow persistent clutter; the
  present-frame guard and empty-scene test reduce but do not eliminate this.
- Mode 4's coherent stage depends on channel calibration and correct beam
  formation; each stage must be compared with the same scene and target.
- Relative altitude is derived from slant range and elevation and is not an
  independently measured absolute altitude.
- The inherited classifier can run Mode 3 without a new dataset, but the old
  dataset does not by itself establish accuracy for beam-selected features.
- The theoretical sampled range is not the useful detection range of a
  drone-sized target.

## 11. Conclusion — update after final test

Both original teammate-developed processing methods have been recovered and
are selectable in one GUI. Rebuilt Mode 3 provides an independent beam-space
MIMO TBD/STT method, while Mode 4 provides switchable extensions to the
conventional control. The application now provides a complete touch/PPI,
classification, calibration and evidence-logging workflow on the ARM Jetson.
Software verification shows that all lanes launch consistently and remain
within the tested computation budget. Whether any experimental mode increases
repeatable drone range over the original PMM control remains an open result for
the final matched field test.

## References

1. W. X. Yang, *Portable COTS mmWave Radar for UAV Detection, Tracking and
   Classification*, UREx report, NUS, 2026.
2. Q. You, *Radar-UREx: Portable COTS mmWave Radar for UAV Detection, Tracking
   and Classification*, report draft V6, NUS, 2026.
3. Y. He et al., "Detection and Identification of Non-cooperative UAV Using a
   COTS mmWave Radar," *ACM Transactions on Sensor Networks*, 20(2), 2024,
   doi: 10.1145/3638767.
4. S. Garcia-Fernandez et al., "4D FMCW MIMO radar based MCMC-EPF
   track-before-detect," 2023.
5. Texas Instruments, *IWR6843 Single-Chip 60- to 64-GHz FMCW Radar Sensor*
   datasheet.
6. Texas Instruments, *IWR6843ISK-ODS Evaluation Module User's Guide*,
   SWRU546E.
7. Texas Instruments, *DCA1000EVM Data Capture Card User's Guide*, SPRUIJ4.
