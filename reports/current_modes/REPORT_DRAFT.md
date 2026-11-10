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
tracking, Capon angle estimation and an LSTM classifier. Three experimental
single-target-track modes were then integrated: a coherent 12-channel
TDM-MIMO receive-beam mode, a two-transmitter coherent phased-look mode, and a
literature-guided multi-frame track-before-detect mode. A unified touchscreen
GUI provides calibration/readiness state, a tactical top-down PPI, attached
classification labels and structured data/performance logging. The final
controlled outdoor comparison has not yet been completed; all result and
discussion fields are intentionally left blank.

## 1. Scope and contribution boundary

The inherited team work comprises the original conventional and PMM
processing pipelines, their data acquisition paths and their trained models.
The present work recovered both paths into a common operator application,
preserved the earlier experimental Modes 3 and 4, implemented Mode 6,
recompiled the TensorRT artifacts for the Jetson, added a touch-oriented PPI
workflow, and prepared a matched validation protocol. Exact authorship should
be confirmed with the team before submission.

No military-grade performance is claimed. The IWR6843ISK-ODS is a small COTS
three-transmitter/four-receiver sensor. The term "phased-array" below describes
a constrained two-transmitter experiment, not an AESA with independent
high-power transmit/receive modules or a multi-beam scheduler.

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
| 3 — Enhanced MIMO PMM STT | 3 sequential TX, 4 RX, 96 chirps/frame, 9.994 MHz/us slope | Coherent 12-channel detection beam bank across +/-60 degrees, then 2-degree Capon refinement | Test receive-array processing gain with wide recovered field of view |
| 4 — Two-TX phased-array STT | TX0 and TX1 simultaneous, 4 RX, 32 chirps/frame, quantised relative TX phase | Steered transmit pair plus coherent four-RX fixed-look combination | Test a directional non-MIMO phased look within +/-30 degrees azimuth |
| 6 — Literature MIMO TBD | 3 sequential TX, 4 RX, 96 chirps/frame, 9.994 MHz/us slope | Normalised constrained PMM path integration over 8–12 frames, then Capon angles | Accumulate persistent weak evidence without reducing every one-frame threshold |

The numbering intentionally retains Mode 6 as its research identifier. The
earlier proposed Mode 5 was replaced by the recovered two-transmitter Mode 4,
so no Mode 5 is exposed in the final GUI.

### 3.1 Mode 3: coherent TDM-MIMO receive processing

Mode 3 uses the lower-slope long-range RF profile and the 12 virtual channels
created by three sequential transmitters and four receivers. It compensates
the Doppler-dependent phase offset between TDM transmissions, applies
calibrated complex channel corrections, and evaluates a coarse bank of
azimuth/elevation steering vectors. The beam bank covers the original +/-60
degree azimuth and elevation limits using 10-degree azimuth and 30-degree
elevation spacing to remain within the 100 ms Jetson frame budget. Once range
is acquired, Capon/MVDR processing refines the angle at 2-degree spacing.

Ideal coherent combination of 12 equally illuminated channels has an upper
bound of `10 log10(12) = 10.8 dB` relative to one channel. Actual improvement
over the inherited non-coherent average will be smaller because of calibration
error, array geometry, target direction, motion, multipath and beam-grid loss.

### 3.2 Mode 4: two-transmitter phased single-target look

Mode 4 transmits the same chirp simultaneously from TX0 and TX1. A runtime
profile quantises the requested relative phase to the IWR6843 six-bit phase
word. Four receiver channels are then phase-corrected and coherently combined
toward the chosen azimuth/elevation. Because the ODS transmitter spacing can
produce grating ambiguity and the implementation is a single fixed look, the
GUI restricts azimuth to +/-30 degrees. Three simultaneous transmitters are
explicitly prohibited.

Unlike TDM-MIMO, the two transmitters are not separated into independent
virtual channels. Mode 4 therefore tests directional transmit/receive
coherence, not MIMO angular diversity. It remains a fixed-panel STT
experiment, not a scanning military AESA.

### 3.3 Mode 6: literature-guided track-before-detect

Mode 6 retains the long-range three-TX TDM-MIMO waveform. Each PMM range-bin
residual is divided by its calibrated adaptive threshold. A dynamic-programming
path is constrained by the maximum target speed and integrated across up to 12
frames. Detection begins only after at least eight frames, when the mean
normalised path score exceeds 0.65 and the present frame contains sufficient
evidence. The present-frame guard is important: without it, a strong old echo
could continue to create a phantom track after the target leaves.

The synthetic regression injects a persistent target at approximately 20 m,
12 degrees azimuth and 5 degrees elevation with each individual PMM score
below the ordinary threshold. It is acquired at frame eight and its range and
angles are recovered. Measured Jetson timing for this synthetic case was about
46 ms median and 52 ms p95 for the tracker, below the 100 ms frame period. The
rebuilt LSTM TensorRT path measured about 2.9 ms median and 4.4 ms p95
end-to-end. These measurements establish implementation feasibility, not
outdoor drone detection range.

## 4. What changed in the radar equation

The monostatic radar equation may be written as

`Pr = Pt Gt Gr lambda^2 sigma / ((4 pi)^3 R^4 L)`.

| Change | Radar-equation interpretation | Limitation |
|---|---|---|
| Lower chirp slope in Modes 3, 4 and 6 | Does not directly change `Pt`, `Gt`, `Gr` or `sigma`; it changes beat-frequency-to-range mapping and extends the sampled unambiguous range to about 93.8 m | Coarser approximately 0.366 m bins and no guarantee that SNR is usable at 93.8 m |
| Mode 3 coherent receive combination | Increases effective receive processing gain `Gr` in a matched direction | Gain is below the 10.8 dB ideal and depends on calibration/beam match |
| Mode 4 coherent two-TX look | Changes the transmit pattern/effective `Gt` in the selected direction and coherently combines four receivers | Total hardware limits remain; one fixed look, +/-30 degree azimuth, unverified field gain |
| Mode 6 multi-frame integration | Reduces the evidence/SNR required by the detector through temporal processing gain | Slower acquisition; persistent clutter can also integrate if calibration or guards are inadequate |

Because received power falls as `R^-4`, an array-processing gain produces only
the fourth-root range increase in the ideal radar equation. For example, even
an ideal 10.8 dB (factor 12) power gain corresponds to a theoretical range
factor of only `12^(1/4)`, about 1.86, before real losses. Displaying a 90 m
axis or lowering a threshold is not evidence of increased usable range.

## 5. GUI, PPI, calibration and logging

The five-mode GUI removes the need to enter launch commands. It selects the
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

Mode 1 uses its recovered CNN model. PMM modes use the available LSTM because
their 36 x 64 input and labels remain compatible. The current PMM tracker and
experimental waveforms do not exactly match the model's training fingerprint,
so the software logs a cross-profile warning and treats accuracy as unverified.
This is preferable to silently claiming model validity; labelled field runs
are required for the final accuracy figure.

Each run can write processed JSONL, inference-evaluation JSONL and
runtime-performance JSONL. These contain track state, range/angle, detection
scores and thresholds, classification, timing, deadline misses, packet loss,
queue drops and resource use. Mode 6 additionally records its integrated TBD
score, threshold and window length. Short raw-ADC captures are required for
future offline re-beamforming because processed JSONL does not retain all
complex channel samples.

## 6. Software verification completed before field testing

- 252 recovered conventional-pipeline unit/regression tests pass on the
  Jetson.
- 28 enhanced PMM/GUI tests pass, including all RF profile/mode validation,
  coherent-angle tests, PPI/GUI selection and persistent sub-threshold Mode 6
  acquisition.
- Both classifier engines were rebuilt locally for the Jetson. Conventional
  TensorRT parity passed on 128 stored windows with maximum probability error
  0.00259; measured inference was approximately 0.48 ms median and 1.92 ms p95.
- All five GUI selections validate their profile, setup and model files and
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

Mode 4 will be steered to the independently measured bearing. Empty-scene
calibration will be repeated after every mode change. The detailed procedure
and blank result sheet are in `aesa_radar/TEST_DAY_GUIDE.md`.

## 8. Results — to be completed after final test

| Mode | Maximum repeatable range (m) | Pd | False tracks/min | Median / p95 range error (m) | Az / el error (deg) | Track coverage | Classification accuracy | p95 processing (ms) | Packet or processing loss |
|---|---:|---:|---:|---|---|---:|---:|---:|---|
| 1 Original conventional |  |  |  |  |  |  |  |  |  |
| 2 Original PMM |  |  |  |  |  |  |  |  |  |
| 3 Enhanced MIMO STT |  |  |  |  |  |  |  |  |  |
| 4 Two-TX phased STT |  |  |  |  |  |  |  |  |  |
| 6 Literature MIMO TBD |  |  |  |  |  |  |  |  |  |

**Observed conditions:**  


**Log and raw-capture filenames:**  


## 9. Discussion — to be completed after final test




## 10. Limitations

- All PMM modes intentionally track one dominant target; multi-target
  association is not established.
- Mode 3 processing gain and Mode 4 transmit/receive coherence require valid
  channel calibration and correct target bearing.
- Mode 6 may integrate persistent environmental micro-motion. Its present-frame
  guard and empty-scene test reduce, but do not eliminate, this risk.
- Relative altitude is derived from slant range and elevation and is not an
  independently measured absolute altitude.
- The inherited classifier dataset does not establish accuracy for revised
  PMM processing or new waveforms.
- The theoretical sampled range is not the useful detection range of a
  drone-sized target.

## 11. Conclusion — update after final test

Both original teammate-developed processing methods have been recovered and
are selectable in one GUI. Modes 3 and 4 retain the recovered experimental
coherent processing, while Mode 6 adds literature-guided persistent weak-target
integration. The application now provides a complete touch/PPI,
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

