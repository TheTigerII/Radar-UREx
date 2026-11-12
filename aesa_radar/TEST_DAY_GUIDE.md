# Radar-UREx final test-day guide

This guide is the short, repeatable procedure for comparing the four choices
in the deployed GUI, numbered 1 to 4.
The earlier two-TX phased-array STT and literature MIMO TBD modes were removed
from the GUI on 2026-11-12. Their code and profiles remain and can still be run
through `aesa_radar/main/run.py --operation-mode phased-array-stt` or
`literature-tbd-mimo`. The current Mode 4 is a different mode: Mode 1 with
enhanced detection (`conventional_radar/README_MODE4.md`).

## Modes available

| GUI choice | Pipeline and purpose | Important boundary |
|---|---|---|
| 1 — Original conventional | Range-Doppler, OS-CFAR, XYZ/DBSCAN, single-target tracking and CNN/TensorRT | Inherited teammate pipeline; 10 m GUI range initially |
| 2 — Original PMM | Original 20 m waveform, PMM folding/tracking, Capon angles and LSTM/TensorRT | Inherited teammate pipeline and control |
| 3 — Beam-space MIMO TBD/STT | Existing long-range three-TX TDM waveform, coherent 12-channel virtual array, joint beam-range-Doppler path integration over +/-60 degrees and focused 2-degree refinement | Distinct from PMM and CFAR; reuses the checked-in channel correction and 36 x 64 LSTM input |
| 4 — Enhanced conventional | Mode 1's waveform, calibration, tracker and CNN with four switchable detection stages: calibrated CFAR threshold, averaged clutter reference, coherent 12-channel beam detection, 8-frame integration | New on 2026-11-13; simulated only, not yet run on the radar; 12 m GUI range |

Every choice supports the tactical PPI, structured performance logging and a
target-attached classification label. Modes 1 and 4 use the recovered CNN. Modes 2
and 3 use the available LSTM. Mode 3 supplies the same 36 x 64 Doppler-time
shape from its selected coherent beam, so no replacement training dataset is
needed to run it. Its feature distribution is still cross-profile, so its
predictions are marked experimental and must be checked against labelled runs
before reporting accuracy.

## Before the first run

1. Run the Jetson from stable mains power. Connect the radar UART, DCA1000 USB
   and Ethernet before opening the application.
2. Open **Radar-UREx Five-Mode Test Center** from the taskbar.
3. In **Connection**, select the Ethernet interface connected to the DCA1000
   and use **Configure 192.168.33.30/24**. The host must actually own
   `192.168.33.30`; entering the address in the GUI is not enough.
4. Select the enhanced CP2105 command UART, normally `/dev/ttyUSB0`.
5. Confirm no other capture is running. Only one process may bind UDP port
   4098. If the GUI reports "address already in use" or "receiver not ready",
   press **Stop all radar processes**. This cleans orphaned launchers,
   receivers, hardware controllers and PPI workers and verifies that UDP 4096
   and 4098 are free before the next run.
6. Fix the panel position, height, pitch and heading. Mark the target route and
   measure slant range/bearing independently. Do not move the radar between
   modes.

## Hardware calibration

Do not overwrite a known-good hardware calibration immediately before the
comparison unless the radar board, mounting, enclosure, cables or waveform
calibration has changed.

Mode 3 deliberately reuses the checked-in range bias and 12-channel complex
correction. Selecting or testing Mode 3 is not a reason to recalibrate with a
corner reflector. The ordinary target-free background interval at the start
of every run is still required; it measures the current scene, not the RF
hardware.

When calibration is required, place the IFM E23012 corner reflector 1.5 m from
the antenna phase centre, at the same height and on boresight. Aim its opening
directly at the panel. An aluminium tripod can create an extra return; keep the
reflector at least 10 cm above its metal parts, put the legs behind it, and
remove all other metal/people from the beam. A stone or wooden table is usually
less severe but must still remain outside the reflector gate where possible.

Use **Range / RX calibration**, 1.5 m, and the profile/window selected by the
GUI. Apply a result only when the reflector peak is stable. The launcher makes
a timestamped profile backup. Remove the reflector and support before any
target run. This hardware calibration is separate from the empty-scene
background calibration at the beginning of every run.

## Matched range test

Use the same target, course, weather, radar position and run length for every
mode. A five-minute run gives enough post-calibration data for the final
comparison.

For each mode:

1. Select **Top-down tactical PPI**, **5.0 min**, enable classification and
   performance logging, and select the true ground-truth label (`Drone / UAV`
   for flight runs, `Not a drone` for human controls).
2. Leave **Exact processed output** blank so a timestamped JSONL name is
   generated. For one short run per mode, choose a raw ADC output file as well.
3. Start the radar. Keep the full field of view empty while the amber
   calibration banner advances. Introduce no target until it becomes green and
   says **RADAR READY**.
4. Use measured points at 3, 5 and 8 m first. If tracking is repeatable,
   continue at 10, 12, 15 and 20 m, then 25, 30 m and farther only while the
   site remains safe and the return is stable.
5. At each distance, make at least three passes or hovers. A defensible
   detection is a confirmed track held for at least 10 seconds or through one
   complete marked pass—not a single plotted point.
6. Finish with at least 60 seconds of empty scene. This provides a false-track
   sample and reveals phantom reacquisition.

Mode-specific controls:

- Mode 3 scans +/-60 degrees automatically; no manual beam centre or reflector
  capture is needed. Keep the scene empty during its amber background phase,
  then introduce the target only after **RADAR READY**. Its detector rejects
  the centre +/-3 Doppler bins based on the 2026-11-13 empty-scene false-lock
  run; stationary reflectors should no longer create a track solely through
  adjacent-bin leakage.
- Mode 4 has four "Mode 4 stages" checkboxes, all on by default. Before the
  range test, run it for a few minutes on an empty scene and check the
  runtime-performance log for period overruns and the PPI for false tracks.
  If overruns rise, untick the coherent beam stage first. To attribute any
  range gain, run Mode 1, then Mode 4 with only the two "no extra processing"
  stages, then with coherent beams added, then with all four.

## What counts as improvement

Do not use the largest instantaneous range in a log. Report the furthest marked
distance that meets the same confirmation rule in repeated trials. Compare:

- detection opportunities and confirmed detections (`Pd`);
- maximum repeatable range;
- median and p95 range error;
- azimuth/elevation error at measured points;
- track continuity and time to reacquire;
- false tracks per empty-scene minute;
- UAV/not-UAV classification accuracy on labelled runs;
- packet loss, invalid frames, queue/processing drops and resynchronisations;
- p50/p95/max processing time and deadline-miss rate.

## Result sheet — leave blank until testing

| Mode | Maximum repeatable range (m) | Confirmed / opportunities | Pd | False tracks/min | Median range error (m) | Azimuth error (deg) | Elevation error (deg) | Classification correct / total | p95 processing (ms) | Deadline misses | Notes |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1 Original conventional |  |  |  |  |  |  |  |  |  |  |  |
| 2 Original PMM |  |  |  |  |  |  |  |  |  |  |  |
| 3 Beam-space MIMO TBD/STT |  |  |  |  |  |  |  |  |  |  |  |
| 4 Enhanced conventional: threshold + reference only |  |  |  |  |  |  |  |  |  |  |  |
| 4 Enhanced conventional: + coherent beams |  |  |  |  |  |  |  |  |  |  |  |
| 4 Enhanced conventional: all four stages |  |  |  |  |  |  |  |  |  |  |  |

Record the processed, inference-evaluation, runtime-performance and raw file
names beside the paper result sheet before changing mode.
