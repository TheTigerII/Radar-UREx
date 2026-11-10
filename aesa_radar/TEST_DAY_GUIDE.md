# Radar-UREx final test-day guide

This guide is the short, repeatable procedure for comparing the five choices
in the deployed GUI. The numbered choices deliberately retain **Mode 6** as the
research label; there is no visible Mode 5 because the earlier proposal was
replaced by the recovered two-transmitter Mode 4.

## Modes available

| GUI choice | Pipeline and purpose | Important boundary |
|---|---|---|
| 1 — Original conventional | Range-Doppler, OS-CFAR, XYZ/DBSCAN, single-target tracking and CNN/TensorRT | Inherited teammate pipeline; 10 m GUI range initially |
| 2 — Original PMM | Original 20 m waveform, PMM folding/tracking, Capon angles and LSTM/TensorRT | Inherited teammate pipeline and control |
| 3 — Enhanced MIMO PMM STT | Long-range three-TX TDM-MIMO, coherent 12-channel beam bank over +/-60 degrees and Capon refinement | Recovered Mode 3; strongest receive-processing experiment |
| 4 — Two-TX phased-array STT | Two simultaneous transmitters with quantised relative phase and coherent four-RX fixed-look processing | Recovered Mode 4; azimuth setting limited to +/-30 degrees |
| 6 — Literature MIMO TBD | Long-range three-TX TDM-MIMO PMM plus 8–12-frame weak-path integration and Capon angles | New experimental Mode 6; field range not yet established |

Every choice supports the tactical PPI, structured performance logging and a
target-attached classification label. Mode 1 uses its recovered CNN. Modes 2,
3, 4 and 6 use the available LSTM. PMM inference keeps the correct 36 x 64
input contract, but revised/cross-waveform predictions are marked experimental
and must be evaluated from labelled runs before reporting accuracy.

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
   use its stop control, close old radar windows, and retry once the activity
   log reports the process has exited.
6. Fix the panel position, height, pitch and heading. Mark the target route and
   measure slant range/bearing independently. Do not move the radar between
   modes.

## Hardware calibration

Do not overwrite a known-good hardware calibration immediately before the
comparison unless the radar board, mounting, enclosure, cables or waveform
calibration has changed.

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

- Mode 3 scans the recovered +/-60 degree field automatically. Leave its beam
  centre at zero for the initial matched comparison.
- Mode 4 is a fixed electronically steered look. Enter the measured target
  bearing before starting and keep it within +/-30 degrees. A target outside
  this beam can legitimately disappear.
- Mode 6 needs 0.8 seconds of persistent evidence before initial acquisition;
  a slight acquisition delay is expected.

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
| 3 Enhanced MIMO STT |  |  |  |  |  |  |  |  |  |  |  |
| 4 Two-TX phased STT |  |  |  |  |  |  |  |  |  |  |  |
| 6 Literature MIMO TBD |  |  |  |  |  |  |  |  |  |  |  |

Record the processed, inference-evaluation, runtime-performance and raw file
names beside the paper result sheet before changing mode.
