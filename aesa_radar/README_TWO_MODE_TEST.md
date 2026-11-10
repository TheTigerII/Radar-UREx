# Radar-UREx two-mode range test

The deployed **Radar-UREx Two-Mode Range Test** application contains exactly
two visible operating modes:

| GUI selection | RF and host processing | Purpose |
|---|---|---|
| **1 — Enhanced MIMO PMM STT** (fixed Mode 3) | Three sequential TDM transmitters, four receivers and 12 calibrated virtual channels; coherent sector detection followed by Capon/MVDR azimuth and elevation | Long-range MIMO tracking with measured 3-D direction |
| **2 — Phased-array STT** (proposed Mode 5) | TX0 and TX1 transmit simultaneously with calibrated 6-bit relative phase; four receivers combine coherently at the selected look direction | Non-MIMO transmit/receive beamforming comparison |

Mode 2 uses transmitter mask `3`; mask `7` is explicitly prohibited. The
standard IWR6843 supply supports two simultaneous transmitters, while three
simultaneous transmitters require a different hardware power configuration.
The ODS horizontal TX spacing is one wavelength, so the GUI limits the
transmit look to +/-30 degrees. Transmit steering is in azimuth; elevation is
receive-side only.

This is a small fixed-panel phased-array experiment, not a military AESA. It
has one steered STT look, no independent transmit/receive modules and no
multi-beam scheduler. Range improvement must be measured rather than assumed.

Mode 1 reports measured target angles. Its coherent detector searches the
original 20 m baseline field of view: +/-60 degrees in both azimuth and
elevation. Detection beams use 10-degree azimuth and 30-degree elevation
spacing to stay within the ARM frame budget; the calibrated MIMO virtual array
then refines the direction at 2-degree spacing. The
former noncoherent fallback remains removed so the PPI reports a measured
coherent direction instead of substituting boresight.

## Calibration

Use the IFM E23012 corner reflector before comparing the modes:

1. Mount the radar and reflector rigidly at the same height and on boresight.
   Put the reflector 1.5 m from the radar phase centre, with its opening aimed
   directly at the panel. Remove people and nearby metal objects.
2. In the GUI select **2 — Phased-array STT**, then select **Range / RX
   calibration**. Keep the provided 1.5 m distance and +/-0.75 m window.
3. Start calibration and apply the result when it reports a stable result.
   The application measures the three-TX Mode 3 waveform and copies the 12
   coefficients to the Mode 2 profile. Mode 2 derives its TX0/TX1 relative
   phase correction from those coefficients at startup.
4. Remove the reflector. Return to the PPI display, start the selected mode and
   keep the complete field of view empty during the 30-second background
   calibration. Test only after the display says **RADAR READY**.

Repeat reflector calibration after changing the board, profile, enclosure,
cables or radar mounting. The 30-second empty-scene calibration is separate
and is required at the start of every run.

## First Mode 2 hardware check

Start at 0 degrees azimuth and elevation. In the Activity log verify all of
the following before doing a range test:

- `mask=3 (TX0+TX1)` and a `TX1 code` are printed;
- the SDK CLI reports that the radar was configured;
- `sensor start acknowledged` appears; and
- calibration frames advance without packet or processing failures.

If the firmware rejects the nonzero `profileCfg` phase word or simultaneous-TX
chirp, stop and retain Mode 1 for the submission. Do not substitute mask `7`.

## Short comparative range test

1. Fix the radar position, height and orientation. Use the same target route,
   weather and five-minute duration for both modes.
2. Calibrate and run Mode 1 with the full +/-60-degree field of view. Test the
   target at 3, 5 and 8 m, then 10, 12, 15 and 20 m only after the nearer point
   is stable.
3. Run Mode 2 at the same points with its STT beam aimed at the target bearing.
   Repeat each distance three times. For an off-axis test, use a measured
   bearing within +/-30 degrees and set the GUI azimuth to that value.
4. Record confirmed detections versus opportunities, range error, false locks,
   track losses, packet loss, processing drops and deadline-miss rate. The
   defensible maximum range is the furthest point with repeatable confirmed
   tracks—not a single transient detection.

Processed PMM JSONL and runtime-performance JSONL are written automatically.
Enable **Raw ADC output** for at least one short run per mode if later offline
beam analysis is required. The processed metadata records the operation mode,
TX mask and Mode 2 runtime phase word.
