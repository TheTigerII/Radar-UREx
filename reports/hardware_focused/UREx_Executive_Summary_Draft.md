# Portable COTS mmWave Radar: Hardware and GUI Development

**UREx Executive Summary — draft**  
**Student:** [Name / student number]  
**Supervisor:** [Name]  
**Date:** [Month and year]

## Project purpose

This project develops a portable small-UAV sensing demonstrator using a Texas Instruments IWR6843ISK-ODS radar, a DCA1000EVM raw-data capture board, and an NVIDIA Jetson Orin Nano. Project mates developed the radar modes already present before October 2026, including the original conventional range-Doppler and PMM methods. My second-stage work, added after October, focuses on an operator GUI and experimental changes intended to improve useful range or directional sensitivity. UAV classification is outside the scope of this summary.

## System and work completed

The radar sends raw ADC samples through the DCA1000EVM to the Jetson, where the project performs capture validation, radar processing, tracking, display, and logging. Radar commands use a serial control connection; ADC samples use Ethernet/UDP. My post-October work added an operator GUI with radar mode and profile selection, connection and capture settings, readiness/activity reporting, and start/stop controls, together with a top-down PPI showing radar-relative target position, track history, range/angle information, and calibration/readiness feedback. The GUI also integrates the inherited conventional and PMM modes; their original processing methods remain the work of project mates. [Confirm exact controls and the build you operated.]

The PPI was chosen to give an operator a direct plan view of the tracked target relative to the fixed radar. It complements signal-level plots such as range-Doppler images, which are valuable for analysis but less direct for understanding where a target is moving around the sensor. The view is radar-relative: displayed height is estimated from range and elevation, not independently surveyed altitude. “Tactical-style” describes the display layout and does not imply military-grade performance.

## Hardware and range investigation

My post-October range attempts added a lower-slope FMCW profile, coherent receive-array processing, a constrained two-transmitter phased look, and multi-frame integration of persistent weak PMM evidence. The baseline waveform uses a 46.875 MHz/µs slope and provides approximately 0.078 m range-bin spacing over a nominal 20 m interval. The experimental long-range waveform uses a 9.994 MHz/µs slope, giving an estimated sampled unambiguous range of approximately 93.8 m but coarser range bins of approximately 0.366 m. This extends the sampled interval; it does not establish useful detection at that distance. The newer modes are attempts to improve useful range or sensitivity and require controlled field testing before any gain can be claimed.

The supplied V6 group report provides historical DJI Mini 4 Pro results: after temporary aluminium-tape modification, maximum static range increased from 3.30 to 3.67 m for the conventional method and from 3.50 to 3.95 m for PMM. Those are prior results using a different target and setup, and must not be presented as this iteration's DJI Lito measurements.

## Evaluation and results

The present iteration's field setup was [site, date, radar position/height/aim, target geometry, power/network arrangement]. The target was [exact DJI Lito model], and distance was measured using [instrument and uncertainty]. Baseline and updated modes were compared at [distances] with [number] repeated trials per point. A successful detection was defined as [predeclared confirmed-track criterion].

| Measure | Baseline | Updated |
|---|---:|---:|
| Maximum repeatable range | [ ] m | [ ] m |
| Successful trials at farthest qualifying range | [ ] / [ ] | [ ] / [ ] |
| Detection probability / track coverage | [ ] | [ ] |
| False tracks per empty-scene minute | [ ] | [ ] |
| Range error / measurement uncertainty | [ ] | [ ] |
| Packet or processing loss | [ ] | [ ] |

**Conclusion:** [State whether the measured evidence supports, does not support, or is inconclusive about a repeatable range improvement. Keep GUI usability findings separate from radar detection performance.]

## Limitations and next steps

The result applies to [target, aspect, waveform, site, and test conditions]. The sampled unambiguous range is not the same as useful target range, and coherent gain depends on calibration, target direction, motion, and the environment. Further work should [repeat trials on another day/site, add aspects or targets, measure false-track rates longer, and/or collect synchronized position ground truth].

**Personal contribution:** After October 2026, I developed the operator GUI and PPI and added experimental modes intended to improve radar range or directional sensitivity, including [confirm which of the lower-slope profile, coherent receive mode, two-TX phased look, and Mode 6 you personally implemented]. The original conventional and PMM modes were developed by my project mates before October; I [integrated/restored] them into the later test interface. I evaluated the new modes using [target, setup, and test method]; measured results are [complete after testing].
