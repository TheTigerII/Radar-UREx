# Radar-UREx range and beamforming experiments

> The deployed submission GUI has since been reduced to the two modes in
> [README_TWO_MODE_TEST.md](README_TWO_MODE_TEST.md). The historical baseline,
> long-range and receive-scan CLI paths documented below remain available for
> recovery and controlled comparisons.

This directory is an isolated copy. The original `../main/`, `../profiles/`,
launcher, data, and logs are not modified. Experimental output is written under
this directory unless an absolute output path is chosen.

## What the literature supports

The repository's Tsinghua mmHawkeye paper uses the same IWR6843ISK-ODS and
DCA1000EVM. It reports a 60.25 GHz start frequency, 1.92 GHz sweep, 900 us
chirp, 9.994 MHz/us slope, 6.25 MS/s ADC, three alternating TX antennas, and
100 chirps per frame. Its theoretical sampled unambiguous range is:

`Rmax = c * Fs / (2 * slope) = 93.8 m`.

It reports detection to 80 m for its tested 8 kg, six-wing UAV with 25 cm
blades. That result is not a guarantee for smaller drones, other terrain, or
this ARM computer. Its long-range result also depends on PMM spectrum folding,
background subtraction, constrained tracking, and particle filtering already
represented in this codebase. The paper applies Capon beamforming after finding
the target range to estimate angle; it does not claim transmit beam-steering
gain.

This implementation keeps 32 complete three-TX loops (96 chirps/frame), rather
than the paper's stated 100 chirps, because the existing decoder requires equal
TDM groups. The full RF ramp spans about 1.92 GHz, but 256 ADC samples at
6.25 MS/s cover 40.96 us of it; therefore the processed range-bin spacing is
about 0.366 m, not the `c/(2*1.92 GHz)` value.

The other local UAS/bird paper uses a custom 8-TX/8-RX FDM-MIMO radar, not this
board. Its larger array and FDM waveform cannot be copied directly to the
3-TX/4-RX ODS hardware.

TI documents about 10 dBm maximum device output and about 5 dBi antenna gain
for the ODS board. The local SDK guide says only `txPhaseShifter=0` was tested
for this demo path. Consequently, mode 3 is calibrated, host-side receive
beamforming—not military transmit phased-array steering and not higher EIRP.

Local sources:

- [Tsinghua mmHawkeye paper](../tmp/pdfs/Detection%20and%20Identification%20of%20Non-cooperative%20UAV%20Using%20a%20COTS%20mmWave%20Radar.pdf)
- [60 GHz UAS/bird paper](../tmp/pdfs/Millimeter_Wave_Radar_Measurements_Distinguishing_UAS_and_Birds_Based_on_60_GHz_micro-Doppler_Signatures.pdf)
- [TI IWR6843/ODS guide](../tmp/pdfs/IWR6843Series_UserGuide.pdf)
- [TI profile configuration guide](../tmp/pdfs/profile_cfg_guide%20OCR.pdf)

## Three test modes

| GUI mode | RF/DSP behavior | Nominal range bins | Intended comparison |
|---|---|---:|---|
| 1 — Baseline 20 m TDM-MIMO | Original waveform and noncoherent 12-channel PMM input | 0.078 m | Control; unchanged behavior |
| 2 — Long-range TDM-MIMO | 9.994 MHz/us slope, 1.92 GHz total ramp, 900 us chirp period; noncoherent PMM input | 0.366 m | Wider unambiguous range, coarser range bins |
| 3 — Sector receive beam | Mode 2 RF waveform plus a calibrated 15-beam delay-and-sum bank across 12 virtual channels | 0.366 m | Coherent sensitivity across a small manually centred sector |

Mode 3 is not a non-MIMO transmitter mode. It still uses TDM-MIMO samples
because the three transmitters fire in sequence. The code phase-aligns and
combines the resulting 12 virtual channels and compensates the
Doppler-dependent phase between TDM transmitters. The revised Mode 3 scans a
15-beam bank covering +/-20 degrees azimuth and +/-10 degrees elevation around
the entered sector centre. It selects the strongest coherent response for each
range/Doppler cell and retains the noncoherent channel average as a fallback if
imperfect RX phase calibration would otherwise cause destructive summation.
The PMM tracker still intentionally follows one dominant target.

Ideal coherent combining has an upper-bound array gain of roughly
`10*log10(12) = 10.8 dB` over one channel. Real gain will be lower and the
improvement over the existing noncoherent 12-channel average must be measured;
calibration error, multipath, target motion, and beam mismatch all reduce it.

The bundled classifier is disabled in modes 2 and 3 because its model was
trained against the baseline waveform and feature fingerprint.

## Run

From a terminal, only to install the launcher initially:

```bash
chmod +x experimental_radar/launch_gui.sh
cp experimental_radar/Radar-UREx-Experiments.desktop ~/Desktop/
chmod +x ~/Desktop/Radar-UREx-Experiments.desktop
```

Then double-click **Radar-UREx Experiments**. Select a radar operation; the GUI
chooses the matching profile and range limit. In fixed-beam mode, enter the
look azimuth and elevation from -60 to +60 degrees. Keep the scene empty during
the initial PMM background calibration.

CLI remains available for recovery and repeatable tests:

```bash
./.venv/bin/python -u experimental_radar/main/run.py \
  --operation-mode long-range-mimo --display combined \
  --radar-port /dev/ttyUSB0 --host-ip 192.168.33.30 \
  --max-range-m 90 --duration-minutes 5 --no-classification

./.venv/bin/python -u experimental_radar/main/run.py \
  --operation-mode fixed-beam --beam-azimuth-deg 0 --beam-elevation-deg 0 \
  --display combined --radar-port /dev/ttyUSB0 --host-ip 192.168.33.30 \
  --max-range-m 90 --duration-minutes 5 --no-classification
```

When the config is omitted, the experimental CLI automatically chooses the
correct profile for the selected operation mode.

## Concise test procedure

1. Use the same tripod position, antenna orientation, target, route, weather,
   and five-minute capture for all three modes. Confirm the Ethernet interface
   owns `192.168.33.30/24` and that only one capture process uses UDP 4098.
2. Run baseline first at known distances inside 20 m. Record packet drops,
   deadline misses, PMM detections, range error, and CPU/memory from the runtime
   JSONL.
3. Run long-range MIMO at 20, 30, 40 m and increase distance only after stable
   results. Do not interpret the 90 m gate as a guaranteed detection distance.
4. Run sector-beam mode on the same route with the sector centred at 0°, then
   repeat with centres such as -30°, 0°, and +30°. Each centre covers +/-20°
   azimuth and +/-10° elevation. Recalibrate the empty background after
   changing mode or physically aiming the radar.
5. Compare probability of detection, false alarms, range error, track loss,
   packet loss, and ARM processing deadlines. Prefer mode 2 if the target moves
   across a wide field; prefer mode 3 only when its bearing is approximately
   known.

The profiles retain the existing RX compensation values as an initial value.
For trustworthy beam comparisons, perform the project's range/RX and angular
calibration on the actual board, then copy the resulting compensation line and
host angle biases into both experimental long-range profiles so they remain
identical.

## Data-driven Mode 3 revision (2026-11-07)

The first fixed-beam capture showed useful coherent returns at 4-6 m but only
23 threshold crossings from 6-10 m, while a stable unrelated return near
31.4 m was reacquired. The single 0-degree beam was also vulnerable to target
motion and installation/phase error. The revised Mode 3 therefore:

- scans a small sector instead of relying on one exact look direction;
- keeps a noncoherent fallback so phase error cannot null the target entirely;
- starts with an adaptive-threshold minimum of 350 rather than the baseline
  value of 700;
- starts with a conservative 20 m acquisition/display gate, which can be
  increased only after the nearer test is stable; and
- hides below-threshold search candidates from the displayed target range.

Processed JSONL files contain scores and tracking results but not the complex
per-channel samples needed to optimize beam weights offline. Enable **Raw ADC
output** for at least one short, controlled flight if further calibration or
offline re-beamforming is required.
