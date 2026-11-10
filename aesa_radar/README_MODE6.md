# Mode 6 — literature-guided MIMO track-before-detect

Mode 6 is an experimental long-range single-target-track mode for the
IWR6843ISK-ODS and DCA1000. It is available in the five-mode touchscreen GUI as
**6 — Literature MIMO track-before-detect**.

## What it does

- Uses three sequential transmitters and four receivers (12 TDM-MIMO virtual
  channels).
- Uses the 9.994 MHz/us, 6.25 MS/s long-range profile. Its sampled range axis is
  about 93.8 m with approximately 0.366 m bins.
- Runs the existing PMM spectrum-folding detector on every frame.
- Divides each range-bin residual by its calibrated adaptive threshold and
  integrates a physically reachable path over up to 12 frames.
- Requires at least 8 frames of persistent evidence and evidence in the
  current frame. This allows repeatable sub-threshold returns to accumulate
  while rejecting an old isolated spike.
- After range acquisition, estimates azimuth and elevation using the recovered
  Capon/MVDR angle path and displays the track on the PPI.
- Supplies the same 36-frame by 64-Doppler-bin tensor expected by the available
  LSTM classifier.

This is host-side temporal integration and receive-angle processing. It does
not raise transmitter power and it is not an AESA transmit mode.

## Current validation boundary

The automated synthetic regression injects a persistent target near 20 m at
12 degrees azimuth and 5 degrees elevation. Each individual PMM score is below
the ordinary one-frame threshold; Mode 6 acquires it at the eighth frame and
recovers range and direction. On the Jetson, the measured synthetic tracker
timing was about 46 ms median and 52 ms p95, below the 100 ms frame period. The
rebuilt TensorRT LSTM measured about 2.9 ms median and 4.4 ms p95 end-to-end.

These tests demonstrate the algorithm and ARM timing, not outdoor drone range.
No range improvement should be claimed until the controlled test in
[TEST_DAY_GUIDE.md](TEST_DAY_GUIDE.md) has been completed.

The available classifier model was trained on earlier baseline data. Its input
shape and labels are compatible, but Mode 6 predictions are explicitly logged
as cross-profile and their accuracy is not established. Choose the correct
ground-truth label in the GUI so the evaluation log can be used later.

## Calibration and operation

1. If the radar board, mounting, enclosure or RF profile calibration has
   changed, use the IFM E23012 reflector at the measured 1.5 m boresight
   distance with **Range / RX calibration**. Keep metal tripod parts behind the
   reflector and out of the beam as far as practical.
2. Remove the reflector, tripod and people from the field of view.
3. Select Mode 6, **Top-down tactical PPI**, the correct UART, and the intended
   ground-truth label.
4. Start the radar and keep the complete scene clear throughout the 30-second
   PMM background calibration. Do not introduce a target before the banner
   says **RADAR READY**.
5. Test progressively from known near distances. Do not begin at the maximum
   display range.

Processed JSONL includes `tbd_integrated_score`, `tbd_threshold` and
`tbd_window_frames` in addition to the normal range, angle, tracker and
classification fields. Runtime JSONL records timing, deadline misses and
resource use. Record one short raw-ADC run if later offline reprocessing may be
needed.

## Literature basis

- Y. He et al., "Detection and Identification of Non-cooperative UAV Using a
  COTS mmWave Radar," ACM TOSN 20(2), 2024. A local copy is in
  `../tmp/pdfs/Detection and Identification of Non-cooperative UAV Using a COTS
  mmWave Radar.pdf`.
- S. Garcia-Fernandez et al., "4D FMCW MIMO radar based MCMC-EPF track-before-
  detect," 2023. The implementation adopts the multi-frame TBD principle, not
  that paper's complete MCMC-EPF algorithm.
- Texas Instruments, IWR6843 and IWR6843ISK-ODS documentation. Mode 6 remains
  within the existing three-transmitter TDM capture architecture.
