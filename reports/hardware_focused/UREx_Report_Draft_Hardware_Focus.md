# Portable COTS mmWave Radar: Hardware and Operator-Interface Development

**UNDERGRADUATE RESEARCH EXPERIENCE (UREx) PROJECT REPORT — DRAFT**  
**Student:** [Your name] ([Student number])  
**Supervisor:** [Supervisor name]  
**Department:** [Department / school]  
**Date:** [Month and year]

> **Draft note:** This report follows the structure and measured, practical emphasis of the supplied V6 report. Replace bracketed text with your own setup, attribution, and results. The repository documents implementation and software-side checks, but does not contain a completed controlled outdoor range comparison for this iteration. Do not state that detection range improved until you have entered and analysed your field measurements. Classification is outside this report's scope.

## Abstract

This project aims to develop a radar system with commercially available off-the-shelf (COTS) components operating in the millimetre-wave radar for detecting and tracking small unmanned aerial vehicles (UAVs). The system uses a Texas Instruments (TI) IWR6843ISK-ODS radar, a TI DCA1000EVM raw-data capture card, and an NVIDIA Jetson Orin Nano mini-computer, all mounted on 3D-printed chassis designed in SolidWorks and fitted to a camera tripod. It is powered by a power bank slotted into the chassis for ease of swapping. The first stage of the project established baseline functionality using range-Doppler and periodic-micro-motion (PMM) processing methods, after which the second staged attempted to improve the user experience and range of the system. A graphical user interface (GUI) with improved symbology was developed to improve the user experience. To investigate potential range improvements, I tested two experimental modes. The first one coherently processes the 12-channel TDM-MIMO virtual array, while the second uses a phase-controlled two-transmitter look. It was found that while absolute range increased to 15-20 metres for a human sized target, detection range 

**Keywords:** FMCW radar; portable radar; IWR6843ISK-ODS; DCA1000EVM; Jetson Orin Nano; operator interface; plan-position indicator; radar calibration; UAV detection.

## Acknowledgements

This report summarises the development of a radar system my groupmates, You Qi Xuan (A0308105R), Weng Xian Yang (A0307751E) and I. Their contribution if mentioned will be acknowledged at the start of the section it appears in. My main contribution was developing the attempts to improve performance after reaching a baseline functionality. My contribution was towards the initial development, data collection, hardware design, and subsequent iterations to increase the detection range. 
I would like to thank Prof Koenraad Mouthaan, our supervisor, for his guidance 

## 1. Introduction

### 1.1 Motivation

In recent years, unmanned aerial verhicles (UAVs) have seen a proliferation in both civilian, government and military use. This creates a corresponding need for affordable sensing systems that can monitor the presence of such systems in a range of environments. However, small UAVs are challenging radar targets because of their small physical size, complex geometry, and range of motion. This presents a target that can have massively  A radar system must also contend with clutter, multipath, antenna alignment, configuration, and operator workload. A portable demonstrator adds practical constraints: the sensor, capture hardware, compute device, power, mounting, cables, and user interface must work together as one deployable system.

### 1.2 Project objective and scope

The overall project investigates a portable COTS radar chain capable of acquiring raw radar data, detecting and tracking a target, and presenting the result locally. This iteration's objectives were:

1. Develop a GUI that makes radar setup, launch, readiness monitoring, and target visualization more accessible to an operator.
2. Investigate hardware and processing changes intended to extend the useful detection range while preserving a repeatable operating procedure.
3. Evaluate the updated system using [DJI Lito exact model / other target] under a documented setup and an objective detection criterion.

This report emphasizes radar hardware, acquisition, profile selection, calibration, the GUI/PPI, and range evaluation. It summarizes the earlier conventional and PMM pipelines only to explain their role in the system. It does not evaluate UAV classification.

### 1.3 Report focus and contribution boundary

The supplied V6 report documents earlier work on radar-profile design, range and angular calibration, DJI Mini 4 Pro measurements, and a temporary aluminium-tape target modification. Those measurements are prior project evidence, not results from the present iteration. The current repository contains the GUI, PPI, radar-mode selection, profile/configuration handling, hardware startup, capture, calibration interfaces, and experimental processing changes.

**My contribution in this iteration was:** [Describe your own code, design, integration, profile, hardware, setup, and test work. Name files or commits where possible. Separate work completed by you from work inherited from groupmates.]

| Work area | Earlier / inherited work | My contribution this iteration (edit) |
|---|---|---|
| Radar and capture hardware | IWR6843ISK-ODS, DCA1000EVM raw capture, Jetson edge processing | [What you physically assembled, configured, or changed] |
| Conventional processing | Range-Doppler, OS-CFAR, angle/point-cloud processing and tracking | [Any changes made, or state inherited unchanged] |
| PMM processing | Spectrum folding, background subtraction, path tracking and angle estimation | [Any changes made, or state inherited unchanged] |
| GUI and PPI | Earlier terminal / plot-based operation [verify baseline] | [Your design and implementation] |
| Range development | Prior profile and target experiments described in V6 | [Your profile / coherent processing / test contribution] |
| Field evaluation | Prior tests used a DJI Mini 4 Pro [per V6] | [Your target, site, roles, and measurements] |

## 2. Background

### 2.1 FMCW ranging and practical range

An FMCW radar transmits a frequency ramp and estimates range from the beat frequency in the received signal. For a given sampled bandwidth, a wider bandwidth provides finer range-bin spacing. The maximum unambiguous range also depends on the sampled beat-frequency interval and waveform slope. These are distinct limits: a displayed range axis or theoretical unambiguous range does not prove that a target return has enough signal-to-noise ratio for reliable detection.

For the repository's baseline Mini4 profile, 256 ADC samples at 6.25 MS/s and a 46.875 MHz/µs slope give approximately 0.078 m range-bin spacing and a nominal 20 m range gate. The experimental long-range profile lowers the slope to 9.994 MHz/µs while retaining the ADC sampling configuration. The profile documentation estimates a sampled unambiguous range of approximately 93.8 m and range-bin spacing of approximately 0.366 m. The coarser bins may make two nearby targets harder to separate. The 93.8 m value is a waveform limit, not a measured detection range.

### 2.2 Array processing and weak-target evidence

The IWR6843ISK-ODS provides three transmit channels and four receive channels. In TDM-MIMO operation, sequential transmitters and the four receivers form a twelve-channel virtual array. Coherently combining calibrated channels can increase sensitivity in a selected direction; ideal combination of twelve equal channels has an upper bound of 10.8 dB relative to one channel. Actual gain depends on channel phase calibration, geometry, target direction and motion, and multipath. The improvement relative to an existing noncoherent twelve-channel average is less than that ideal upper bound and must be measured.

The experimental two-TX mode drives TX0 and TX1 simultaneously with a programmed relative phase and combines four receivers coherently for a selected look direction. It is a constrained single-look experiment on this board, not a general-purpose scanning AESA. A separate PMM track-before-detect experiment integrates a physically reachable path over multiple frames. This may accumulate persistent weak evidence but adds acquisition delay and can also accumulate persistent clutter if scene calibration or rejection logic is inadequate.

### 2.3 Processing approaches retained from earlier work

The conventional method forms range-Doppler information, applies ordered-statistic CFAR, estimates angle, creates and clusters detections, and maintains a target track. The PMM method folds Doppler spectra over candidate periodicities, subtracts a calibrated background, follows a continuity-constrained path, smooths the estimate, and refines target angle. These methods remain different processing approaches; the experimental waveforms and coherent/temporal modes are not evidence that one method is universally superior.

## 3. System overview

### 3.1 Portable hardware architecture

The portable system consists of:

- **Radar front end:** Texas Instruments IWR6843ISK-ODS operating in the 60–64 GHz band.
- **Raw-data capture:** DCA1000EVM, receiving radar ADC data over Ethernet and forwarding it to the host.
- **Edge computer:** NVIDIA Jetson Orin Nano, running acquisition, processing, visualization, and log writing locally.
- **Control path:** radar command UART (the project procedure identifies the CP2105 interface).
- **Mounting and infrastructure:** [tripod/panel mount, cables, Ethernet interface, power supply or battery, enclosure, display, and other components actually used].

In the documented configuration, the host capture interface uses `192.168.33.30`; DCA1000 control is separate from ADC streaming. The host binds the ADC data receiver before the radar starts. The startup sequence configures the radar, connects/configures and arms the DCA1000, and only then issues `sensorStart`. On shutdown, the radar is stopped before recording is stopped. These barriers reduce the chance of starting the radar before the receiver is ready.

**Figure 1.** [Insert a labelled photograph or block diagram showing radar, DCA1000, Jetson, UART, Ethernet/UDP, power, mounting, display, and data flow.]

### 3.2 Data and control flow

The operator selects the mode and profile through the GUI. The launcher starts the capture/processing path, performs configuration checks, and presents the live state. Radar control commands travel over UART; raw ADC packets travel over Ethernet/UDP to the Jetson. The Jetson validates packet and frame integrity, decodes ADC samples, runs the selected processing lane, and forwards current display data to the GUI. Processed and performance logs can be saved for later evaluation.

### 3.3 Operating modes represented in the current GUI source

The current `aesa_radar/main/gui.py` source defines five visible radar choices: the inherited conventional pipeline; the original approximately 20 m PMM baseline; enhanced coherent MIMO PMM single-target tracking; two-TX phased-array single-target tracking; and literature-guided MIMO track-before-detect. Mode numbers retain project identifiers, so the list contains Mode 6 without a Mode 5. The exact installed/deployed build should be confirmed before submission because some repository guide files describe earlier two-mode or three-mode test builds.

| GUI choice (current source) | Hardware / signal-path change | Evaluation role |
|---|---|---|
| Original conventional | Existing profile and conventional range-Doppler path | Historical comparison lane |
| Original PMM | Baseline three-TX TDM-MIMO, four RX, approximately 20 m profile | PMM control |
| Enhanced MIMO PMM STT | Long-range waveform with coherent twelve-channel receive-beam search and Capon/MVDR refinement | Test directional receive-processing gain over a broad field |
| Two-TX phased-array STT | Simultaneous TX0/TX1 with relative phase control and coherent four-RX look; azimuth limited to ±30° | Test a fixed directional look |
| Literature MIMO track-before-detect | Long-range TDM-MIMO plus multi-frame PMM path integration | Test accumulation of persistent weak returns |

The command-line processing library also contains additional waveform/beam experiments. Include them only if they were part of the build you operated and tested: [identify exact release/commit and selected modes].

## 4. Hardware configuration and range development

### 4.1 Baseline waveform and long-range waveform

The baseline profile uses a 46.875 MHz/µs slope and produces approximately 0.078 m bins over the nominal 20 m gate. The experimental lower-slope profile uses 9.994 MHz/µs, 6.25 MS/s sampling, 256 ADC samples, and a 900 µs chirp period. It produces approximately 0.366 m range-bin spacing and an approximately 93.8 m sampled unambiguous-range estimate. This profile extends the sampled range interval while sacrificing range resolution. It does not increase transmit power or by itself establish target detection at the larger range.

The target frame schedule uses 32 three-transmitter loops (96 chirps per frame) for compatible TDM decoding. [Confirm the profile actually used for each result, frame period, chirps per frame, and observed frame rate from the saved metadata.]

### 4.2 Coherent receive processing

In the enhanced MIMO mode, the host compensates for the Doppler-dependent phase offset between TDM transmitters and applies calibrated complex channel corrections before searching a bank of receive steering directions. The current design uses coarse beams to remain within the Jetson's frame budget and then refines the selected angle with Capon/MVDR processing. This approach changes receive processing gain and directionality; it does not increase transmitter output power.

### 4.3 Two-transmitter phased look

The phased single-target mode creates a temporary runtime profile with TX0 and TX1 active simultaneously and a relative phase selected from the requested look angle and calibration. The GUI restricts the azimuth look to ±30° because of the ODS array spacing and grating ambiguity. The standard board's supported simultaneous-transmitter limit is enforced in the profile path; three simultaneous transmitters are not used. The mode is a fixed electronic look and requires target bearing to be known approximately. [Record firmware acceptance, actual phase word, and hardware validation outcome.]

### 4.4 Hardware safety and operating boundary

The experimental profiles do not modify the radar's regulatory transmit-power limit. Mode 3 is host-side receive beamforming, and the two-TX mode changes the directional pattern within the module's limits. The work therefore investigates waveform range mapping and signal-processing/array gain, not a high-power transmitter retrofit. [Add any hardware review, firmware checks, or power measurements completed.]

**Figure 2.** [Insert the baseline and experimental FMCW profiles or a table of slope, bandwidth, ADC rate, samples, chirps, frame period, range-bin spacing, and sampled range.]

## 5. Calibration and operational readiness

### 5.1 Hardware range and receive-channel calibration

The GUI includes a range/RX calibration workflow based on a measured reflector distance and known reference angle. The test-day documentation specifies an IFM E23012 corner reflector at 1.5 m on radar boresight, with its opening aimed toward the panel. Calibration estimates range offset and complex receive-channel corrections. The result is reviewed before applying it to the operational profile, and a timestamped backup is created. [Report whether you performed this calibration, which profile was used, accepted frame count, measured bias, and whether the result was applied.]

### 5.2 Empty-scene background calibration

Hardware calibration and scene-background calibration are separate operations. Before a PMM run, the scene remains empty while the processor learns its background; the GUI shows calibration progress and changes to a ready state when the required frames are collected. The documented default is a 30-second background interval. [State whether the interval was modified, how the field of view was cleared, and how you ensured no target or person entered the measurement region.]

### 5.3 Startup and capture integrity

The application checks the connection configuration and starts the receiver/processing path before allowing radar streaming. The capture code tracks packet sequence and byte continuity and rejects contaminated frames before processing. Performance logs record packet/frame loss, queue drops, processing time, and deadline misses. These measures make hardware status and data integrity observable during field operation; they do not substitute for a measured detection result.

## 6. Graphical interface and tactical PPI

### 6.1 GUI functions

The current source provides a touchscreen-oriented desktop application with Run, Hardware & files, Tracking, and Activity tabs. Depending on the mode and display selected, the operator can configure the radar choice, serial port, network interface, radar/DCA profile, duration, capture output and display; initiate or stop a run; select range, range-Doppler, point-cloud, combined, or top-down PPI display; inspect tracking and calibration settings; and monitor activity messages. The launcher maps each mode to its intended profile and data directories and validates required paths before starting. [Adjust this feature list to match the build and controls you personally implemented.]

### 6.2 Why a top-down tactical-style PPI was used

A range-Doppler plot is useful for inspecting signal structure, but it does not present target position in a direct plan view. A point cloud provides spatial information but can be harder to interpret quickly on a compact field display. The PPI was selected as an operator-view option to show the tracked target's horizontal position and trail relative to the fixed radar, while keeping range, azimuth, elevation, relative-altitude estimate, calibration/readiness state, and track state available in the display. A top-down view can help an operator relate the measurement to a marked route or test area without interpreting raw Doppler bins.

In this implementation, the display converts radar slant range and measured azimuth/elevation into relative coordinates; the vertical component is derived from range and elevation. It is therefore a radar-centered, PPI-like display, not an independently surveyed map or verified absolute-altitude instrument. “Tactical-style” describes its visual organization only and does not imply military-grade capability.

**My design rationale and operating benefit (complete):** [Describe what information you wanted visible at a glance, who operated the system, what tasks the display made easier, and any observation or feedback supporting the claimed benefit. If you did not conduct a usability study, describe the design intent rather than claiming measured improvement.]

**Figure 3.** [Insert a screenshot of the PPI with axes, radar position, a track/trail, track state and readiness state labelled. Remove private or irrelevant information.]

## 7. Methodology

### 7.1 Research question and hypothesis

**Question:** Under a fixed radar position, waveform, target aspect, site, and detection rule, does [name the hardware/profile/processing update] improve repeatable detection or tracking range of [exact DJI Lito model] relative to [baseline mode/version]?

**Hypothesis:** [State the predicted mechanism and a measurable outcome. For example: coherent receive processing will increase the proportion of trials meeting the confirmed-track criterion at one or more farther marked distances, within the selected beam sector.]

### 7.2 Site and hardware setup — fill in

- **Date/time and site:** [ ]
- **Radar board / revision / serial if relevant:** [ ]
- **Capture board and cabling:** [ ]
- **Edge computer and power source:** [ ]
- **Radar mounting:** [tripod/bracket, antenna phase-centre height, pitch, yaw, roll]
- **Host network interface/IP and DCA settings:** [ ]
- **Radar UART and firmware/CLI:** [ ]
- **Software commit/release and GUI build:** [ ]
- **Radar profile / mode / calibration ID:** [ ]
- **Target:** [DJI Lito exact model, rotor state, battery state, payload/material modifications, pilot]
- **Target geometry:** [aspect, height, slant range measurement points, route or hover positions]
- **Site conditions:** [weather, wind, nearby structures/metal, ground surface, clutter]
- **Photographs/diagram:** [file name or figure]

**Setup narrative (write after your session):** [Describe how the sensor and capture hardware were arranged, how the antenna was aimed, how the target distance and bearing were measured, where the operator stood, and how target motion was repeated.]

### 7.3 Calibration and test sequence — fill in

1. Record the software commit, active radar profile, and GUI-selected mode. Confirm that the host network interface and command UART are correct.
2. Complete range/RX calibration using [reflector type, distance, angle, profile, and accepted calibration record], if required by a hardware/profile change.
3. Start the same target-free background calibration for each compared run. Begin the target trial only after the GUI reports **RADAR READY**.
4. Test the baseline and updated mode at identical measured target distances and aspects. [Describe matched ordering, number of repetitions, pass/hover duration, and criteria for repeating an invalid run.]
5. Finish each run with [duration] of empty scene for false-track observation. Save processed output, runtime-performance output, and at least one short raw ADC capture per mode if offline channel analysis may be needed.

### 7.4 Variables and outcome definitions — fill in before final data collection

**Primary outcome:** [e.g. maximum repeatable distance at which a confirmed track is maintained for at least 10 s / one complete marked pass in at least X of Y trials.]

**Secondary outcomes:** [probability of detection, track coverage, time to first confirmed track, range error, angle error, false tracks per empty-scene minute, packet/invalid-frame rate, p95 processing time, frame deadline misses.]

**Controlled factors:** [radar profile/power, position/height/orientation, target model/aspect/rotor state, site, background calibration, threshold, route, duration, weather range, operator, and test order.]

**Ground truth:** [laser range meter / surveyed markers / other independent reference; measurement uncertainty; how time or position was synchronised.]

## 8. Results — complete after field testing

### 8.1 Hardware and GUI operation

| Item | Observation / result |
|---|---|
| GUI build and software commit | [ ] |
| Modes visible in test build | [ ] |
| Radar profile(s) used | [ ] |
| Startup/readiness issues | [ ] |
| Calibration result / range bias | [ ] |
| Packet loss / invalid frames / processing drops | [ ] |
| Operator display issues or feedback | [ ] |

### 8.2 Detection and range results

| Mode / version | Target distance (m) | Trials | Successful confirmed tracks | Detection coverage / Pd | Median range error (m) | False tracks/min | Invalid frames (%) | p95 processing (ms) | Notes |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Baseline: [ ] | | | | | | | | | |
| Updated: [ ] | | | | | | | | | |
| Updated: [ ] | | | | | | | | | |

**Maximum repeatable range (using the predeclared success rule):** baseline [ ] m; updated [ ] m; change [ ] m ([ ]%).  
**Logs/raw captures:** [list paths or filenames].  
**Figure 4.** [Plot trial success or detection probability against independently measured distance. Include all trials, not only the furthest successful observation.]

### 8.3 Prior V6 result for context only

The supplied V6 report describes a DJI Mini 4 Pro aluminium-tape experiment with static detection ranges of 3.30 to 3.67 m for the conventional pipeline and 3.50 to 3.95 m for PMM. It also reports range-calibration biases of 11.9 cm and 10.0 cm for the respective profiles. These are earlier group results on a different target and setup; they are not measurements of this iteration or of the DJI Lito. Compare them only with that limitation stated.

## 9. Discussion — complete after results

### 9.1 Hardware and operating implications

[Discuss whether the updated profile or coherent mode changed observed range, resolution, angular coverage, acquisition time, and Jetson processing load. Separate waveform range limits from usable target detection. Explain any operator benefit observed when using the GUI and PPI.]

### 9.2 Interpreting the PPI and GUI

[Describe whether the display made it easier to judge readiness, target location, and track continuity. State whether feedback was informal or collected systematically. Report limitations such as estimated relative altitude, fixed-panel geometry, clutter, screen visibility, or display latency.]

### 9.3 Range and measurement uncertainty

[Discuss repeatability, distance measurement uncertainty, target aspect and motion, weather, multipath, calibration stability, false tracks, threshold sensitivity, sample size, and whether trials were paired. Do not treat a single farthest detection or a larger GUI range setting as evidence of better radar range.]

## 10. Limitations and future work

The present work uses [one site / one target model / one aspect / limited trials — edit to match]. COTS transmit power remains constrained by the radar module and applicable operating requirements. A lower-slope waveform extends the sampled interval but produces coarser range bins; coherent gain depends on calibration and look direction; and track-before-detect depends on a stable clutter background. The PPI derives its coordinates from the radar's own range and angle estimates and is not independent ground truth.

Next work should [repeat on a second day or site; increase matched trial count; include target aspects; independently time-synchronize a position reference; characterize false tracks in target-absent scenes; measure sustained Jetson temperature/power; or other].

## 11. Conclusion — finalize after results

This iteration [implemented / integrated] a GUI and top-down PPI for operating the IWR6843ISK-ODS/DCA1000EVM/Jetson Orin Nano system and [implemented/evaluated] [specific hardware or processing update]. The interface presents [verified operational functions]. The range comparison found [result from Section 8] under [conditions]. The result [supports / does not support / is inconclusive for] a repeatable range improvement. Further trials are needed to determine whether the result generalizes beyond the tested DJI Lito, profile, aspect, and site.

## References

1. *Radar-UREx_Report_V6_YouQixuan.pdf*, supplied group report, V6 draft, 2026. Use the document's original author, title, and date details in the final bibliography.
2. *UREx_Report (2).pdf*, supplied group report, 2026. Verify author/title/date before final submission.
3. Y. He et al., “Detection and Identification of Non-cooperative UAV Using a COTS mmWave Radar,” *ACM Transactions on Sensor Networks*, vol. 20, no. 2, 2024, doi: 10.1145/3638767.
4. Texas Instruments, *IWR6843 Series User Guide* and *IWR6843ISK-ODS User Guide*, local project copies under `tmp/pdfs/`.
5. Texas Instruments, *DCA1000EVM Data Capture Card User's Guide*, local project copy under `tmp/pdfs/`.
6. Radar-UREx source repository: `aesa_radar/main/gui.py`, `aesa_radar/main/gui_config.py`, `aesa_radar/main/run.py`, `aesa_radar/main/startup.py`, `aesa_radar/main/livedatacapture.py`, `aesa_radar/main/pmm.py`, `aesa_radar/profiles/`, and `aesa_radar/TEST_DAY_GUIDE.md` (accessed [date]).

## Appendix A. Evidence and attribution checklist

- [ ] Replace all bracketed placeholders and verify personal contribution statements with the team.
- [ ] Record exact branch, commit, GUI title/version, and actual selectable modes.
- [ ] Distinguish inherited V6 measurements from this iteration's results.
- [ ] Add hardware and setup photographs with measured geometry.
- [ ] Archive profile, calibration, processed log, runtime log, and raw capture filenames.
- [ ] Complete the results table and plot from trial-level data.
- [ ] Remove any proposed feature or mode that was not present in the tested build.
- [ ] Confirm reference authors, report versions, and access dates.
