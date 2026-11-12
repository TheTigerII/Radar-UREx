# Mode 3 — beam-space MIMO TBD/STT

Mode 3 is the independent long-range experiment in the four-mode GUI. It is
designed to run with the hardware state and classifier artifacts already in
this repository.

## What it does

1. The radar keeps the existing three-transmitter TDM schedule. TX0, TX1 and
   TX2 transmit sequentially; they are not driven simultaneously.
2. The host applies the stored 12-channel complex correction and compensates
   the motion-dependent phase offset between the three TDM transmissions.
3. All three transmitters and four receivers form a 12-channel virtual array.
   Coherent receive beams cover +/-60 degrees azimuth and elevation. The ARM
   coarse grid uses 12-degree azimuth and 30-degree elevation spacing.
4. The detector scores the strongest non-zero-Doppler energy in every
   beam/range cell. It does not use PMM folding and does not first create a
   per-frame CFAR point detection.
5. An eight-frame dynamic-programming path integrates evidence jointly through
   beam, range and time. Angular movement is limited to neighbouring beams and
   range movement is limited by the configured target speed.
6. Once a current, persistent path passes the threshold, a local 2-degree beam
   grid within the coarse cell refines azimuth and elevation. The normal
   range/angle particle filters
   then provide the PPI track.

This is digital TDM-MIMO receive beamforming. It uses every available TX in
the supported sequential schedule, but it is not a simultaneous-transmit AESA
and does not increase the board's EIRP.

## Calibration and classification compatibility

No new corner-reflector capture is required. Mode 3 uses
`profile-mini4-fixed-beam.cfg` without changing its waveform or its checked-in
`compRangeBiasAndRxChanPhase` values. A new hardware calibration should be
performed only after a genuine RF-path change such as replacing or remounting
the radar, enclosure or cables.

The amber target-free background phase at the beginning of every run is still
required. It learns the current scene's beam/range median and robust noise
scale; it is not a new range/channel calibration and needs no reflector.

The inherited LSTM remains usable without collecting a new training dataset.
Mode 3 records the selected beam as the same 36 frames by 64 Doppler bins
expected by the model, uses the same labels and runs through the existing
TensorRT path. Because coherent beam selection changes the statistical feature
distribution, the software intentionally logs it as experimental cross-profile
inference. Compatibility means it runs with the existing model; classification
accuracy must still be measured from the final labelled field test.

## Operator procedure

1. Open the GUI and choose **3 — Beam-space MIMO TBD/STT (3-TX)**.
2. Choose **Top-down tactical PPI** and leave classification enabled.
3. Remove the reflector, tripod and people from the field of view.
4. Start the radar and keep the scene unchanged until the amber banner becomes
   green and says **RADAR READY**.
5. Introduce the human or drone target and follow the matched-distance test in
   `TEST_DAY_GUIDE.md`.

Processed JSONL records the beam-space score, integrated threshold, track,
angles and classifier result. Runtime-performance JSONL records processing
timing and deadline misses. These outputs are sufficient for the planned
comparison, although raw ADC capture is still needed for later offline
algorithm changes.

## Validation boundary

Automated tests cover waveform/profile validation, TDM phase compensation,
wide-angle beam formation, persistent-path acquisition, rejection of a stale
single-frame spike, angle refinement, metadata, and classifier tensor shape.
They establish software correctness and ARM feasibility, not a guaranteed
outdoor range. Report only the furthest repeatable measured target distance,
not the largest instantaneous plotted range.
