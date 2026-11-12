# Radar UREx Codex Handoff

Last updated: 2026-11-13 02:14 SGT (UTC+8)

This file is the continuation point for another Codex account working on the Radar-UREx project on the Jetson ARM computer. The immediate objective is a reliable, touchscreen-friendly GUI that can run the two inherited radar pipelines and two final experimental modes, show a tactical PPI with classification and angle information, recover cleanly from failed runs, and produce evidence for the final report. The implementations are present; the main unfinished work is controlled field testing, evidence-based tuning, and filling the blank result sections in the reports.

## Read this first

- Work in `/home/rgmtc/Desktop/Radar-UREx`.
- Preserve all user captures, logs, calibration files, model artifacts, and unrelated worktree changes.
- Do not use destructive Git commands such as `git reset --hard` or overwrite the current branch from another branch.
- Do not push, rewrite history, or make system-level network changes unless the user explicitly asks.
- Before starting a radar process, inspect the current processes, interfaces, UARTs, and ports. Machine state can change after this handoff.
- Do not claim increased range or reliable drone classification until field measurements support it. Rebuilt Mode 3 and enhanced conventional Mode 4 have automated/synthetic validation only; neither has an established field range.

## Git state at handoff

- Active branch: `aesa-recovery-2026-11-11`
- Base before the 2026-11-13 Mode 3 rebuild: `b6212b233 Add enhanced conventional Mode 4, renumber and hide experimental modes`
- Handoff commit: `448427ac4 Add Codex project handoff`
- Receiver-recovery commit: `c81ff8649 Add emergency radar cleanup and receiver recovery`
- Important earlier commits:
  - `41aa5a42b Restore original modes and complete five-mode test center`
  - `34ca65238 tue night changes post recovery`
  - `89be01f40 Add literature-guided Mode 6 track-before-detect` (now Mode 5)
  - `1760a8a77 Recover AESA radar modes and touchscreen GUI`
- The earlier divergence was resolved without conflicts in merge commit `18a3b5b89`. It combines the local handoff and receiver-recovery commits with `dff49c975 report draft update` from the remote.
- `dff49c975` adds report drafts and `rawdatacapture/profile-mini4-20m.cfg`; these files are now preserved in the combined history.
- At 20:00 SGT, the combined branch was committed locally but its push was waiting for GitHub authentication on this computer.

Recommended first Git checks:

```bash
cd /home/rgmtc/Desktop/Radar-UREx
git status --short --branch
git log -8 --oneline --decorate
git log --oneline --left-right HEAD...origin/aesa-recovery-2026-11-11
```

## Current machine and hardware state

At handoff time, no Radar-UREx GUI or radar worker process was running and UDP ports 4096 and 4098 were free. The following state was observed, but it must be rechecked before use:

- Platform: NVIDIA Jetson/ARM Linux.
- DCA Ethernet interface: `enP8p1s0`.
- Host capture address: `192.168.33.30/24`.
- DCA1000 address: `192.168.33.180`.
- DCA1000 data port: UDP 4098; configuration traffic uses UDP 4096.
- Radar command UART: `/dev/ttyUSB0`, the CP2105 enhanced port.
- `/dev/ttyUSB1` is the CP2105 standard port and is not the normal command selection.
- The user belongs to the required serial-device group.
- A complete hardware startup diagnostic succeeded after the Ethernet link recovered: DCA commands were acknowledged, radar CLI configuration was accepted, `sensorStart` was acknowledged, and the session stopped cleanly.
- A subsequent Mode 3 run received hundreds of frames with no packet loss or queue drops.

Useful read-only checks:

```bash
ip -brief address show enP8p1s0
cat /sys/class/net/enP8p1s0/carrier
ip neigh show 192.168.33.180
ls -l /dev/ttyUSB0 /dev/ttyUSB1
ss -lunp | rg ':(4096|4098)\b'
.venv/bin/python -u aesa_radar/main/stop_all.py --list
```

## Launching and stopping the application

Launch the deployed GUI with:

```bash
cd /home/rgmtc/Desktop/Radar-UREx
./aesa_radar/launch_gui.sh
```

The installed desktop entry is:

```text
/home/rgmtc/.local/share/applications/radar-urex-simo-scan.desktop
```

Its desktop ID is `radar-urex-simo-scan.desktop`, and it launches `aesa_radar/launch_gui.sh`. The source desktop file is `aesa_radar/radar-urex-simo-scan.desktop`.

The GUI now has an orange **Stop all radar processes** button. It deliberately leaves the GUI alive while stopping repository `run.py`, `startup.py`, and `livedatacapture.py` workers and their child processes. It then checks that UDP 4098 and 4096 are available. The GUI also offers automatic cleanup and retry when startup detects an occupied receiver port.

The same recovery is available from the terminal:

```bash
cd /home/rgmtc/Desktop/Radar-UREx
.venv/bin/python -u aesa_radar/main/stop_all.py
```

Do not replace this with a broad `pkill python` command. Other Python applications may be running.

## Why receiver not ready occurred and what was fixed

The concrete failure was an orphaned `aesa_radar/main/livedatacapture.py` process. Its launcher had exited, but the capture worker remained alive and held UDP 4098. A later GUI run therefore could not make the receiver ready.

Commit `c81ff8649` adds:

- `aesa_radar/main/stop_all.py`, with repository-scoped process matching and staged interrupt/terminate/kill cleanup.
- The orange GUI cleanup button.
- Automatic cleanup/retry for address-in-use startup failures.
- A carrier check before launch.
- A 30-second normal-stop allowance so the launcher can clean up its children.
- Fallback to the scoped cleanup helper if normal shutdown still times out.
- Tests for process matching and GUI integration.

Common startup errors:

| Symptom | Likely cause | Action |
| --- | --- | --- |
| `[Errno 99] Cannot assign requested address` | `192.168.33.30` is not assigned to `enP8p1s0`, or the link has dropped | Check the interface address and carrier. Restore `192.168.33.30/24` through the existing network configuration. |
| `[Errno 98] Address already in use` | A previous capture worker still owns UDP 4098 | Press **Stop all radar processes**, or run `stop_all.py`, then verify the ports. |
| `receiver not ready` with free port | Ethernet link/DCA has not recovered, radar/DCA startup failed, or startup was interrupted | Check carrier, neighbor entry, UART choice, and the latest terminal log. Allow hardware a short recovery interval before retrying. |
| Zero packets after successful bind | DCA stream did not start or Ethernet flapped | Inspect link state and DCA/radar acknowledgements in the latest log; do not tune DSP parameters first. |

The Linux receive buffer has sometimes been granted as only 425984 bytes when the application requests 4194304 bytes. Successful Mode 3 operation nevertheless showed no packet drops. Treat a `net.core.rmem_max` change as a system modification and ask the user before making it.

## Deployed modes

The touchscreen GUI exposes four choices, numbered 1 to 4. On 2026-11-12 the literature track-before-detect mode was first renumbered from Mode 6 to Mode 5, and then the phased-array STT and literature TBD modes (rows marked hidden below) were both removed from the GUI. On 2026-11-13 a new GUI Mode 4, "Enhanced conventional", was added: Mode 1 plus opt-in detection stages in `conventional_radar/main/enhanced_detection.py` (see `conventional_radar/README_MODE4.md`). It is validated on synthetic data only and has not yet run on the radar or been timed on the Jetson. Their code and profiles remain and are still runnable through `run.py --operation-mode phased-array-stt` / `literature-tbd-mimo`; the rows below are kept for reference. Commit messages and the frozen snapshot folders still say Mode 6.

| GUI mode | Implementation | Signal processing and purpose | Status |
| --- | --- | --- | --- |
| 1 Original conventional range-Doppler/CNN | `conventional_radar/` | Inherited range-Doppler, ordered-statistic CFAR, 3D/single-target tracking, and its own CNN/TensorRT classifier | Recovered and integrated; needs final comparative field run |
| 2 Original PMM/mmHawkeye LSTM 20 m | Original PMM path through `aesa_radar/` in `baseline` mode | Inherited spectrum folding, background subtraction, constrained path tracking, particle filtering, Capon angles, and LSTM classifier | Available as the control/baseline |
| 3 Beam-space MIMO TBD/STT | `aesa_radar/`, `fixed-beam` operation | Existing sequential 3-TX/4-RX waveform and stored 12-channel correction; coherent +/-60-degree beam cube, robust beam/range background and eight-frame constrained path integration, then focused 2-degree digital-beam refinement | Rebuilt on 2026-11-13; automated and ARM timing validation completed; field range and classification measurements pending |
| 4 Enhanced conventional | `conventional_radar/` with `--cfar-threshold calibrated --clutter-reference mean --detection-power coherent-beam --tbd-window-frames 8` | Mode 1 waveform, calibration, tracker and CNN; calibrated CFAR threshold, averaged clutter reference, TDM-compensated coherent 12-channel beam detection and 8-frame range-Doppler integration, each switchable in the GUI | Unit-tested and simulated only; Jetson timing and field range pending |
| Former Mode 4: two-TX phased-array STT (hidden from GUI) | `aesa_radar/`, `phased-array-stt` | TX0 and TX1 transmit simultaneously with quantised relative phase; four receivers combine coherently in the selected look direction; azimuth control limited to +/-30 degrees | Recovered replacement for an earlier phased-transmit proposal; firmware/hardware-sensitive and field validation pending |
| Former Mode 5: literature MIMO track-before-detect (hidden from GUI) | `aesa_radar/`, literature TBD operation | Long-range 3TX/4RX TDM-MIMO, normalised constrained weak-path integration over 8-12 frames, present-frame guard, then Capon angles | Synthetic weak-target acquisition works; actual drone range is unverified |

Important terminology:

- Mode 3 is TDM-MIMO receive processing, not a military active electronically scanned array. It deliberately reuses the checked-in channel calibration and the existing 36 x 64 LSTM input, so it needs neither a new corner-reflector calibration nor a new training dataset. Its normal target-free run-start background phase is still required, and cross-profile classification accuracy remains a field-test result rather than an assumption.
- The former Mode 4 (phased-array STT, hidden) is the closest available hardware-safe phased transmit experiment, but the standard board is restricted to two simultaneous transmitters. Do not describe it as equivalent to a military AESA.
- The PPI altitude is an estimate derived from measured range and elevation. It is not independently instrumented altitude.

The authoritative operator summary is `aesa_radar/TEST_DAY_GUIDE.md`. The hidden literature TBD mode (former Mode 5) is described in `aesa_radar/README_MODE5.md`; the current Mode 4 in `conventional_radar/README_MODE4.md`.

## Profiles, setup, and model artifacts

The previously missing files have been restored. Required profiles include:

```text
aesa_radar/profiles/profile-mini4-20m.cfg
aesa_radar/profiles/profile-mini4-long-range.cfg
aesa_radar/profiles/profile-mini4-fixed-beam.cfg
aesa_radar/profiles/profile-mini4-phased-stt.cfg
aesa_radar/profiles/profile-mini4-literature-tbd.cfg
aesa_radar/profiles/profile-mini4-simo-scan.cfg
aesa_radar/profiles/setup.json
```

The PMM/LSTM artifacts are in `aesa_radar/model_weights/`, including:

```text
model.onnx
manifest.json
generated/model.fp16.engine
```

The conventional CNN artifacts are in `conventional_radar/model_weights/`, including:

```text
model.onnx
manifest.json
generated/model_sm87_fp16.engine
```

Both TensorRT engines were rebuilt locally for this Jetson. The conventional engine was checked against ONNX on 128 windows with maximum probability error about 0.00258 and benchmarked around 0.48 ms median inference. The PMM model was trained under an earlier feature/profile version; the runtime deliberately permits cross-profile inference and logs contract mismatches. That compatibility does not prove drone-classification accuracy in Modes 3, 4, or 5, so classification must be scored during field testing.

## Calibration and test setup

There are two distinct activities that must not be confused:

1. PMM background calibration: start each run with the field of view empty and still for the full on-screen calibration period, normally 30 seconds. People, the drone, tripod, and reflector should not move through the scene during this interval.
2. Corner-reflector characterization: use the IFM E23012 at a measured range and boresight to assess response or derive controlled corrections. This is a separate test, not a substitute for the empty-scene PMM calibration.

The available reflector may be mounted on an aluminium tripod. Keep the reflector separated from metal parts as much as practical, record the mounting geometry, and compare against an empty-tripod control if using the measurements quantitatively. Nearby bushes can create clutter or micro-motion; use a clear background for calibration and range comparisons where possible. The prior setup used about 1.5 m direct distance and roughly 10 cm clearance from tripod metal, but that run alone should not be treated as a complete range calibration.

For a range comparison:

- Use the same radar position, power source, profile, target path, weather, and background for every mode.
- Mark measured target distances rather than estimating them visually.
- Start with an empty-scene calibration.
- Move a human-sized target or drone through repeated inbound/outbound passes at fixed range marks and several azimuths.
- Record acquisition range, longest sustained range, false tracks after the target leaves, angle error, classification output, packet/frame loss, and processing deadline misses.
- Run at least three comparable passes per mode. Do not tune a mode between passes that are being treated as one comparison set.
- For drone work, use a safe and legally permitted test area and keep an independent ground-truth record, such as marked distance plus synchronized video.

Use the detailed checklist and blank table in `aesa_radar/TEST_DAY_GUIDE.md`.

## Data and logs

- AESA/PMM captures: `aesa_radar/dataset/`
- AESA/PMM runtime logs: `aesa_radar/log/`
- Original/base captures and logs may also appear in root `dataset/` and `log/`.
- Experimental historical material is in `experimental_radar/`.
- Nested capture and log outputs are intentionally ignored by Git. They are still user data; do not delete them merely because they are untracked or ignored.
- The GUI Activity tab and terminal log path printed at startup are the first places to inspect after a failed run.
- Literature TBD (former Mode 5) output includes `tbd_integrated_score`, `tbd_threshold`, and `tbd_window_frames` fields for evidence analysis.

Earlier problematic observations included tracks snapping to phantom ranges after a target left, limited off-boresight tracking, and drone loss beyond a few metres. These symptoms can arise from background contamination, sidelobes/clutter, stale path integration, incorrect angle gating, low target radar cross-section, or hardware/stream faults. Do not infer the cause from the displayed range alone; correlate processed detections with raw/processed logs, calibration state, angle, score, and ground truth.

## Validation already completed

The latest code validation before handoff included:

```bash
cd /home/rgmtc/Desktop/Radar-UREx/aesa_radar
QT_QPA_PLATFORM=offscreen ../.venv/bin/python -m pytest -q tests
```

Current result: 36 tests pass after the Mode 3 beam-space rebuild.

The conventional suite now passes 270 tests. Python compile checks and an
off-screen launch through the primary desktop script pass. Re-run the relevant
suite after any merge or code change rather than relying only on these results.

Useful targeted checks:

```bash
cd /home/rgmtc/Desktop/Radar-UREx
.venv/bin/python -m compileall -q aesa_radar/main conventional_radar/main
QT_QPA_PLATFORM=offscreen .venv/bin/python -m pytest -q aesa_radar/tests
cd conventional_radar && QT_QPA_PLATFORM=offscreen ../.venv/bin/python -m pytest -q testcodes
```

The rebuilt Mode 3 random-cube benchmark on this Jetson was about 75 ms median
and 77 ms p95 for the complete tracker, within its 100 ms frame period. A
200-trial Gaussian path-noise stress check produced zero declarations at the
configured threshold. The installed TensorRT LSTM accepted a complete 36 x 64
Mode 3 segment and returned a classification without rebuilding the engine or
dataset; the logged fingerprint mismatches correctly keep that accuracy marked
experimental.

The literature TBD (former Mode 5) synthetic regression acquired a persistent weak target at about 20 m on frame 8 when individual-frame evidence was below the normal declaration threshold. Synthetic tracker timing was roughly 46 ms median and 52 ms p95, and TensorRT LSTM inference was roughly 2.9 ms median and 4.4 ms p95. These are engineering checks, not field range results.

An observed successful Mode 3 run had hundreds of valid frames without packet loss or queue drops. Some early 100 ms deadline misses may still appear during calibration/startup; evaluate the sustained rate and queue recovery in the performance log.

## Reports and literature

Current editable drafts:

- `reports/current_modes/REPORT_DRAFT.md`
- `reports/current_modes/ONE_PAGE_SUMMARY_DRAFT.md`

Both intentionally leave final field results and discussion entries open for the user to fill after testing. Do not invent values. Older HTML or historical READMEs may describe superseded mode sets; use the current Markdown report drafts and `TEST_DAY_GUIDE.md` as the current account of the four deployed choices.

User-provided reference reports remain outside the repository at:

```text
/home/rgmtc/Downloads/Radar-UREx_Report_V6_YouQixuan.pdf
/home/rgmtc/Downloads/UREx_Report.pdf
/home/rgmtc/Downloads/UREx_Executivesummary.pdf
```

Additional papers collected during development are under `tmp/pdfs/`. Preserve attribution and distinguish literature-derived design ideas from measured project results.

## Recommended continuation sequence

1. Confirm that the remote branch contains merge commit `18a3b5b89` and the later handoff-state update. If not, authenticate this computer and push the existing local branch without rewriting it.
2. Re-run automated tests after reconciliation.
3. Verify hardware readiness with the read-only checks above, then launch the GUI.
4. Press **Stop all radar processes** before the first run if the previous shutdown state is uncertain.
5. Run the four GUI modes in a controlled order using the same marked course and empty-scene calibration.
6. Save the exact log/capture path and ground-truth notes for every run.
7. Analyze detection range, continuity, false tracks, angle accuracy, classification, packet loss, and processing timing before changing parameters.
8. Tune only the mode with evidence showing a specific failure mechanism; keep the original Modes 1 and 2 as controls.
9. Enter measured results into the blank tables and complete the report discussion without overstating military equivalence, field range, or classifier validity.

If a run fails, first determine whether the failure is hardware transport, stale process ownership, calibration, DSP tracking, or classification. A receiver or Ethernet failure should be fixed before changing beamforming or PMM thresholds.
