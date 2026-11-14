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

This report summarises the development of a radar system my groupmates, You Qi Xuan (A0308105R), Weng Xian Yang (A0307751E) and I. Their contribution if mentioned will be acknowledged at the start of the section it appears in. My contribution includes the initial development, data collection, hardware design, of which was done with Qi Xuan. The second stage of the project with iterations to increase the detection range were done by myself. 
I would like to thank Prof Koenraad Mouthaan, our supervisor, for his guidance and I also acknowledge the support by other collaborators and the laboratory staff that provided administrative assistance and advice.

## 1. Introduction

### 1.1 Motivation

In recent years, unmanned aerial verhicles (UAVs) have seen a proliferation in both civilian, government and military use. This creates a corresponding need for affordable sensing systems that can monitor the presence of such systems in a range of environments. However, small UAVs are challenging radar targets because of their small physical size, complex geometry, and range of motion. This presents a target that has a wide range of motion due to its ability to rapidly change its path of motion and hover. It also has a small but varied radar return depending on attitude. A radar system must also contend with clutter, multipath, antenna alignment, configuration etc. while making tradeoffs in portability, range and cost, among other considerations. As such, this project attempts to develop a working demonstrator of a portable radar system to detect UAVs.

### 1.2 Project objective and scope

The project was developed in two related stages by a three-person team. In the first stage, we designed and assembled the portable radar platform, collected data, and established the system hardware and operating workflow. My initial hardware-design and data-collection work was carried out with Qi Xuan. The original conventional range-Doppler and PMM modes were developed by my project mates before October 2026. Qi Xuan's V6 report documents the first-stage profile selection, calibration, DJI Mini 4 Pro measurements, and aluminium-tape target-return experiment [1]. The present report briefly records that shared foundation and focuses on the second-stage work I completed after October 2026.

The second-stage objectives were to:

1. Develop a graphical user interface (GUI) to simplify radar configuration, launch, readiness monitoring, and target visualization.
2. Add and test Mode 3 and Mode 4 as hardware/processing approaches intended to improve usable detection range or sensitivity.
3. Evaluate those two modes using [DJI Lito model and test setup], with the results reported separately from the earlier DJI Mini 4 Pro measurements.

This report focuses on the portable hardware, the GUI and plan-position-indicator (PPI), the two tested experimental modes, and the second-stage test method and results. It does not repeat the full signal-processing or data-collection account from V6, and it does not assess classification.

### 1.3 Relationship to V6 and contribution boundary

V6 is the first-stage project report and a key reference for this continuation. Its detailed account of the shared prototype and earlier experiments should be read alongside this report. The V6 measurements are cited as historical team results, not as results of my Mode 3/Mode 4 tests.

| Stage / work | Contribution described in this report | Attribution and evidence |
|---|---|---|
| First stage: portable hardware and baseline operation | Radar, DCA1000EVM, Jetson, mounting, acquisition, and original conventional and PMM operation established the shared platform | Hardware design and data collection were shared team work; the original processing modes were developed by project mates before October 2026. See V6 [1]; confirm individual names/roles: [ ]. |
| First stage: profile, calibration, and Mini 4 Pro experiments | Profile testing, range/channel calibration, drone-return modification, and controlled measurements | Summarized from Qi Xuan's V6 report [1]; retain its attribution to the person or people who performed each experiment. |
| Second stage: operator GUI and PPI | GUI workflow, hardware/configuration controls, readiness display, and operator-facing target presentation | My work after October 2026. Add exact implementation details, version, and any operator feedback: [ ]. |
| Second stage: Mode 3 | Coherent receive processing using the 12-channel TDM-MIMO virtual array and the lower-slope profile | My range/sensitivity experiment. Add the test configuration and measured result: [ ]. |
| Second stage: Mode 4 | Phase-controlled simultaneous two-transmitter look using TX0 and TX1 | My range/sensitivity experiment. Add the test configuration and measured result: [ ]. |

The later folders contain copies of earlier software as well as new work. Copying the original conventional or PMM method into a later GUI did not originate those methods; my contribution was the second-stage interface, experimental modes, integration, and tests described here. Exact individual contributions to the shared first stage should be confirmed with the team before submission.

## 2. Background

### 2.1 FMCW range and the practical detection limit

An FMCW radar estimates target range from the beat frequency between transmitted and received chirps. The radar profile determines the sampled range interval and range-bin spacing. These are not the same as practical detection range: a target can lie inside the configured range gate yet remain too weak or cluttered to produce a repeatable track.

The baseline Mini 4 profile in the repository uses a 46.875 MHz/us slope, 256 ADC samples at 6.25 MS/s, and a nominal 20 m processing gate. The experimental long-range profile lowers the slope to 9.994 MHz/us while retaining the ADC sampling settings. Its documented sampled unambiguous range is about 93.8 m, while the configured gate is 90 m; the corresponding range-bin spacing is about 0.366 m, compared with about 0.078 m for the baseline. Thus, the profile increases the sampled range interval at the cost of range resolution. Neither the theoretical limit nor the configured gate is a measured UAV detection range [4].

### 2.2 Shared first-stage processing baselines

The conventional baseline forms range-Doppler data, detects candidate returns with ordered-statistic CFAR, estimates target angle, groups detections, and maintains a track. The PMM baseline folds Doppler spectra over candidate periodicities, subtracts a calibrated background, and follows a constrained path through range and time. These two methods were developed as first-stage team work before October 2026. They provide context for this report; the focus here is not to re-evaluate or claim authorship of their original algorithms.

### 2.3 Mode 3 and Mode 4 rationale

Mode 3 uses three transmitters sequentially with four receivers. The resulting 12-channel TDM-MIMO virtual array is processed coherently after channel and Doppler-dependent phase corrections. A directional receive-beam search is intended to increase sensitivity for returns arriving from a covered direction. The gain depends on calibration, target motion and direction, array geometry, and the environment, so it must be measured.

Mode 4 activates TX0 and TX1 together and applies a programmed relative phase for a selected look direction; the four receive channels are also combined coherently. This is a constrained, single-look experiment for this radar board. It is distinct from Mode 3's sequential-TX TDM-MIMO receive processing. The repository profile limits the selected azimuth to +/-30 degrees. Hardware operation and any range benefit are reported only to the extent confirmed in the test records.

## 3. System overview

### 3.1 Portable hardware architecture

The prototype combines a Texas Instruments IWR6843ISK-ODS radar operating in the 60-64 GHz band, a DCA1000EVM raw-data capture card, and an NVIDIA Jetson Orin Nano for local acquisition, processing, and display. The radar and capture hardware are mounted on [describe the 3D-printed chassis, SolidWorks design, and camera tripod as used in your test]. A swappable power bank is fitted into the chassis [confirm model/capacity if relevant]. The radar command connection uses the CP2105 UART; captured ADC data travels over Ethernet to the Jetson.

**Figure 1.** [Insert a labelled photograph or block diagram of the radar, DCA1000EVM, Jetson, UART, Ethernet, power bank, chassis, tripod, and operator display.]

### 3.2 Acquisition and control path

The GUI selects the operating mode and its associated profile. The host binds and starts the capture receiver before radar streaming is enabled. The startup sequence configures the radar over UART, connects and configures the DCA1000EVM, arms capture, and then issues `sensorStart`. The Jetson receives the raw ADC packets, validates packet/frame continuity, processes valid frames, and forwards the current target state to the display. Processed data and runtime information are saved for evaluation. [Add any setup-specific network address, firmware, and startup deviations from your actual test.]

## 4. First-stage development documented in V6

### 4.1 Shared prototype and profile development

The first stage established the hardware and software baseline needed for later range experiments: a portable radar/capture/compute chain, operating profiles, and conventional and PMM processing paths. V6 describes the team's profile selection and calibration work and explains why a complete working 60 GHz COTS system was prioritized before range optimization [1]. This report treats that work as the shared foundation and does not repeat its detailed profile-development narrative.

The baseline profile uses the 46.875 MHz/us slope and is the reference for the original PMM operation. The later lower-slope profile uses 9.994 MHz/us. Mode 3 and Mode 4 both use this long-range waveform configuration; their receive/transmit processing differs. [Confirm the exact profile filenames and hashes used in your two tests.]

### 4.2 Calibration and earlier range evidence

V6 reports range-calibration biases of 11.9 cm for the conventional profile and 10.0 cm for the PMM profile. It also reports that temporary aluminium foil tape on the DJI Mini 4 Pro body changed the maximum observed static range from 3.30 m to 3.67 m for the conventional method, and from 3.50 m to 3.95 m for PMM [1]. These are first-stage results on the Mini 4 Pro, under the earlier setup. They are included to show the project history, not as a direct comparison with the DJI Lito or as evidence for Mode 3/Mode 4 performance.

For the present tests, hardware range/RX calibration was [performed/not performed] using [reflector, distance, profile, and calibration record]. Scene-background calibration was [procedure and duration]. Keep these calibration activities distinct: reflector calibration corrects range/channel response, while empty-scene calibration estimates background clutter for PMM processing.

## 5. Second-stage changes: GUI, PPI, and range modes

### 5.1 GUI development

The second-stage GUI replaces the earlier [terminal/manual workflow, if accurate] with an operator-facing workflow for selecting an operating mode and profile, configuring hardware connections and output paths, starting or stopping a run, monitoring readiness/activity, and selecting a display. The current source includes Run, Hardware & files, Tracking, and Activity tabs. It also includes hardware-calibration controls and mode-specific profile selection. [Keep only functions present in the version you personally demonstrated; describe any additional interaction or error handling you implemented.]

**My GUI contribution:** [Describe the design and implementation you completed, the files/modules changed, and any operator feedback or observed change in setup time/error rate. Do not claim a usability improvement was measured unless you collected that evidence.]

### 5.2 Reason for using a tactical-style PPI

Range-Doppler displays help inspect signal strength and motion but do not directly show where a tracked target lies relative to the radar. A point-cloud display can show three-dimensional detections, but may be less immediate for an operator monitoring a test route. The top-down PPI was selected to present the radar-relative target position and trail in a plan view while keeping range, azimuth, elevation, track state, and calibration/readiness information visible. This gives the operator a quick view of target direction and movement during setup and testing.

The PPI is radar-centred, not a surveyed map. Its displayed horizontal position is derived from slant range and azimuth; the height/altitude value is derived from slant range and elevation, not independently measured. "Tactical-style" describes the presentation only and does not imply military-grade capability.

**Operator benefit in my test:** [Explain what you needed to see at a glance, how the PPI supported the test workflow, and any observed or collected operator feedback.]

**Figure 2.** [Insert a PPI screenshot with radar position, axes/range marks, target/trail, track state, and readiness state identified.]

### 5.3 Mode 3: coherent TDM-MIMO receive processing

Mode 3 uses the 9.994 MHz/us profile and the three sequential transmitters/four receivers. The host forms the 12-channel virtual array, corrects calibrated channel phase and TDM Doppler phase, searches receive steering directions, and refines the selected angle using Capon/MVDR processing. This is host-side coherent receive processing; it does not increase the radar's transmitter power. The ideal 12-channel array-gain figure is a theoretical upper bound relative to one channel, not the expected gain over the existing baseline or a measured range increase. [Describe the actual beam settings and any Mode 3-specific test observations.]

### 5.4 Mode 4: two-transmitter phased look

Mode 4 uses TX0 and TX1 simultaneously with a runtime phase setting derived from the selected look direction and calibration. The profile uses the long-range slope and samples four receive channels coherently. Its azimuth look is limited to +/-30 degrees; targets outside the selected direction may not be observed. It is a fixed electronic look, not a scanning AESA or a general-purpose multi-beam system. [State whether the radar firmware accepted the profile, what phase was programmed, and how this was verified during your test.]

## 6. Methodology

### 6.1 Research question

Under the test conditions below, how did Mode 3 and Mode 4 compare in repeatable DJI Lito detection/tracking range and operator-visible track quality?

**Expected effect:** [State your prediction for Mode 3 and Mode 4 separately. Keep the claim tied to an observable outcome.]

### 6.2 Site and physical setup — complete

- **Date, time, and site:** [ ]
- **Radar board and capture board:** [ ]
- **Jetson model/software and power source:** [ ]
- **Chassis, tripod, radar height, pitch, and heading:** [ ]
- **Network interface, IP settings, UART, and firmware:** [ ]
- **GUI version/commit and selected display:** [ ]
- **Mode 3 profile/configuration and calibration ID:** [ ]
- **Mode 4 profile/configuration and programmed phase/look direction:** [ ]
- **Target:** [exact DJI Lito model, rotor state, battery state, pilot]
- **Target distances/aspect/height/route or hover positions:** [ ]
- **Distance/bearing reference and uncertainty:** [ ]
- **Wind/weather, surface, nearby structures, and clutter:** [ ]
- **Setup photographs/log filenames:** [ ]

**Setup description:** [Describe the physical layout, how the radar was aimed and fixed, how the target position was measured and repeated, and how the operator used the GUI.]

### 6.3 Calibration and test procedure — complete

1. Record the software version, profile, calibration file, target configuration, and GUI settings for each mode.
2. Perform [range/RX calibration procedure] if required for the hardware/profile and record the result. For each run, collect the same target-free scene background and start trials only after the GUI reports readiness.
3. Use the same radar position, target model, target aspect, site, range marks, trial duration, and processing success rule for Modes 3 and 4. [Describe whether order was alternated/randomized and the number of trials at each distance.]
4. Define a successful trial before scoring: [e.g. confirmed track maintained for at least __ seconds or through one complete marked pass].
5. Record target-absent intervals and the processed/runtime logs. Save a short raw ADC capture for each mode if it is available and relevant to offline analysis.

### 6.4 Outcomes and controls — complete

**Primary outcome:** [maximum repeatable range, with exact trial-success and repetition rule].

**Secondary outcomes:** [detection/track coverage, range error, angle error, time to acquire, false tracks per minute, packet/invalid-frame rate, processing latency, deadline misses].

**Controlled variables:** [profile, transmit settings, radar mounting, target geometry, background-calibration procedure, thresholds, trial duration, weather bounds, and operator procedure].

## 7. Results — complete with your test records

### 7.1 Setup and operation

| Item | Mode 3 | Mode 4 |
|---|---|---|
| Software/GUI version | [ ] | [ ] |
| Active profile and hash | [ ] | [ ] |
| Calibration used | [ ] | [ ] |
| Firmware accepted profile / start confirmed | [ ] | [ ] |
| Capture/invalid-frame issues | [ ] | [ ] |
| PPI/readiness observations | [ ] | [ ] |

### 7.2 Range and track results

| Distance (m) | Mode 3 successful trials / total | Mode 4 successful trials / total | Mode 3 track coverage / range error | Mode 4 track coverage / range error | False tracks / notes |
|---:|---:|---:|---|---|---|
| [ ] | [ ] | [ ] | [ ] | [ ] | [ ] |
| [ ] | [ ] | [ ] | [ ] | [ ] | [ ] |
| [ ] | [ ] | [ ] | [ ] | [ ] | [ ] |

**Maximum repeatable range under the stated criterion:** Mode 3 [ ] m; Mode 4 [ ] m.  
**Other measures:** [false tracks/min, p95 processing time, packet/invalid-frame counts, range/angle errors].  
**Log and raw-data files:** [ ]  
**Figure 3.** [Plot per-trial success or detection/track coverage against measured range for Mode 3 and Mode 4.]

### 7.3 Relation to the first-stage V6 results

The V6 Mini 4 Pro measurements are context for the earlier baseline and aluminium-tape experiment, not a matched reference for this test. The current comparison uses [DJI Lito model] and Modes 3 and 4 under [site/setup]. Differences in target model, profile, aspect, geometry, and test criterion limit direct numerical comparison. [Add a careful comparison only if those factors are sufficiently matched.]

## 8. Discussion — complete after scoring

### 8.1 Hardware and range implications

[Explain how Mode 3 and Mode 4 behaved at each distance. Distinguish the profile's sampled range limit from usable detection. Discuss resolution, calibration, look direction, acquisition delay, Jetson processing load, and any loss or false tracks. State whether the evidence shows a repeatable difference, no difference, or remains inconclusive.]

### 8.2 GUI and PPI operation

[Describe what the interface changed in the test workflow and whether the PPI helped you observe target location, direction, track continuity, and readiness. Separate your design rationale from measured usability results.]

### 8.3 Limitations

[Discuss the number of trials, single target/model, target aspect and motion, site/clutter, distance uncertainty, weather, calibration stability, and any differences between mode configurations.]

## 9. Conclusion — finalize after results

The first project stage, documented in V6, established the shared portable radar platform, baseline processing, and initial profile/calibration/target measurements. In the second stage, I developed [GUI/PPI contribution] and tested Mode 3 and Mode 4 using [target and setup]. Under the stated conditions, the measured maximum repeatable ranges were [ ] m and [ ] m, respectively, with [brief false-track/data-integrity summary]. These results [support/do not support/are inconclusive about] a range improvement for the tested setup. Further matched testing is needed before generalizing to other targets, aspects, or environments.

## References

1. Y. Q. Xuan, *Radar-UREx: Portable COTS mmWave Radar for UAV Detection, Tracking and Classification*, V6 project report, National University of Singapore, 2026. Confirm the title page's exact author name, report title, and date before submission.
2. Texas Instruments, *IWR6843 Series User Guide* and *IWR6843ISK-ODS User Guide*, local project copies under `tmp/pdfs/`.
3. Texas Instruments, *DCA1000EVM Data Capture Card User's Guide*, local project copy under `tmp/pdfs/`.
4. Radar-UREx source repository, especially `aesa_radar/main/gui.py`, `aesa_radar/main/gui_config.py`, `aesa_radar/main/run.py`, `aesa_radar/main/startup.py`, `aesa_radar/main/livedatacapture.py`, `aesa_radar/main/pmm.py`, `aesa_radar/profiles/profile-mini4-20m.cfg`, `aesa_radar/profiles/profile-mini4-long-range.cfg`, `aesa_radar/profiles/profile-mini4-fixed-beam.cfg`, and `aesa_radar/profiles/profile-mini4-phased-stt.cfg` (accessed [date]).

## Appendix A. Completion checklist

- [ ] Confirm the team contribution wording and exact names/roles.
- [ ] Confirm the exact GUI build, profile files, and Mode 3/Mode 4 settings used.
- [ ] Complete the physical setup and calibration fields.
- [ ] Enter trial-level results and apply the predeclared success rule.
- [ ] Add the setup photograph, GUI/PPI screenshot, and range plot.
- [ ] Keep V6 results labelled as first-stage group evidence, not second-stage results.
- [ ] Verify citation details for V6 and hardware documentation.
