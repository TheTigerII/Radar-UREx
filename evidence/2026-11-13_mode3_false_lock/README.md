# Mode 3 false-lock evidence — 2026-11-13

This directory preserves the complete run used to diagnose and correct the
first field false lock from rebuilt Mode 3 (`fixed-beam`). The run used the
three-TX/four-RX long-range profile, PPI display and existing TensorRT LSTM.
The inference ground-truth label was `unlabeled`; its classifications must not
be counted as accuracy measurements.

## Run summary

- Start: `2026-11-13 02:18:17 SGT`
- Duration: 331.97 s
- Frames: 3,279 at 9.88 processed frames/s
- Packets: 885,816
- Invalid/lost packets: 0
- Receiver/processing queue drops: 0
- Full-pipeline timing: 80.82 ms p50, 89.85 ms p95
- Deadline-miss rate: 1.77%

After a transient path around 15 m, the empty-scene tracker held a return near
6.1 m and -46 degrees for 2,293 consecutive hits. The selected spectrum's
centre +/-3 Doppler-bin cluster was 26.5 dB stronger than every bin outside
that band. This showed that adjacent-to-DC static-clutter leakage—not packet
loss or a receiver fault—was driving the persistent false track.

The correction is commit `14355be7e`: Mode 3 now rejects the centre +/-3
Doppler bins, applies a 10% per-cell multiplicative background floor, requires
the complete eight-frame path, and uses 5.5 integrated / 3.0 present-frame
evidence gates.

## Files

- `pmm_capture_2026_11_13T02_18_17.jsonl.gz` — complete processed updates and
  36-by-64 Doppler histories.
- `20261113_021817_runtime_performance.jsonl.gz` — per-frame timing, transport,
  resource and final summary records.
- `20261113_021818_live_inference.jsonl.gz` — complete inference-evaluation
  records with `unlabeled` ground truth.
- `20261113_021818_livedatacapture.log` — terminal/startup and capture summary.

The JSONL files are compressed losslessly to stay below GitHub's 100 MB
single-file limit. Extract one with, for example:

```bash
gzip -dk pmm_capture_2026_11_13T02_18_17.jsonl.gz
```

## SHA-256 of uncompressed originals

```text
f563909cc032da9cddf6a77c69eeae016b6e44618d816d685cfda5b8a556e5d7  pmm_capture_2026_11_13T02_18_17.jsonl
16aa6327c6ea96af9476673bdff68e57356b3a1d989ffe4d2bfbb73ad982d9c4  20261113_021817_runtime_performance.jsonl
8de19453e558cfca9137fce25e36d2f369a2b0fdb642f4c64330a60f6b41cb5e  20261113_021818_live_inference.jsonl
e273347883f35e8a10dd5a141163e4fcc522d3b26bd36a696d30726826fd427f  20261113_021818_livedatacapture.log
```
