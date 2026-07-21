# Phase 1 Machine Learning Plan for Radar-UREx

## 1. Purpose and current objective

This document defines the first achievable machine-learning milestone for the
live IWR6843ISK-ODS and DCA1000EVM pipeline implemented in this repository. It
does not replace the fuller production architecture in `AI.md`.

The Phase 1 objective is a **drone-versus-bird micro-Doppler classifier**:

> Given a consistently selected aerial target and approximately 1-2 seconds of
> recent radar returns, estimate `P(drone)` versus `P(bird)`.

This is intentionally narrower than a general drone detector. The available
archive contains drone and bionic-bird recordings but almost no representative
empty-scene or environmental background data. It therefore cannot establish
general non-drone rejection or operational false alarms per hour.

The phrase "consistently selected target" is important. The live code now has a
lightweight single-target 3D tracker, but its association is not yet calibrated
or validated against labelled multi-object recordings.

## 2. What the project currently implements

The implemented acquisition and DSP path is:

```text
IWR6843ISK-ODS
    -> DCA1000EVM raw LVDS capture
    -> UDP packets received by livedatacapture.py
    -> packet-loss checks and valid-frame assembly
    -> complex radar cube [chirp, RX, ADC sample]
    -> range FFT
    -> Doppler FFT [Doppler, TX, RX, range]
    -> OS-CFAR point detections and angle estimation
    -> single-target 3D association and predicted range gate
    -> one micro-Doppler spectrum per display update
    -> rolling 60-update spectrogram (approximately 2 seconds) for visualization
```

`run.py` launches the live receiver and hardware controller. Capture,
processing, and display run independently through bounded queues so slow DSP or
plotting does not block UDP reception. Invalid frames affected by detected byte
gaps are not processed or saved.

The active `profile.cfg` gives the main Phase 1 feature dimensions:

| Property | Current value |
|---|---:|
| Frame period | 33.33 ms, approximately 30 Hz |
| ADC/range FFT bins | 128 |
| Doppler bins | 64 |
| RX channels | 4 |
| TDM transmitters | 3 |
| Range-bin spacing | approximately 0.08365 m |

The repository also contains `mmwave.json`, which defines a different profile.
Data from the two profiles must not be silently resized or mixed. Every
generated feature and model artifact must record a radar-configuration
fingerprint.

### Current limitations affecting ML

- The tracker follows only one target and does not publish a durable session
  track ID outside the live processing path.
- Association uses uncalibrated XYZ continuity without Doppler, covariance, or
  target-shape information.
- After 10 missed updates the track is dropped; later strongest-candidate
  reacquisition can select a different object.
- Doppler is currently expressed as a centred bin index, not calibrated radial
  velocity in metres per second.
- Point-cloud position and angles are uncalibrated diagnostic estimates.
- The current micro-Doppler feature is log power only; it does not yet include
  local-background contrast, validity masks, or ML normalization.
- Raw captures and logs do not contain human labels or persistent target truth.
- PyTorch, ONNX Runtime, and TensorRT are not current project dependencies.

The live pipeline is therefore ready to support single-target feature recording
experiments, but the tracker still requires labelled replay validation before
reliable target-conditioned online classification.

## 3. Dataset assessment

The external dataset assessment identified 102 usable complex recordings:

| Class | Recordings |
|---|---:|
| DJI Mavic | 45 |
| Bionic bird | 38 |
| Phantom 3 Pro | 18 |
| Pole | 1 |

Most recordings were reported as complex64 tensors with shape
`1000 x 128 x 168`. Two 3-kHz recordings use `2000 x 128 x 112` and
`1000 x 128 x 112`. The archive itself is not stored in this repository, so a
versioned dataset manifest must verify these counts, shapes, labels, recording
sessions, and source hashes before training.

The archive is useful for Phase 1 pretraining, but it is not equivalent to live
IWR6843 data:

- It was recorded with a different 60-GHz acquisition system.
- Its tensors do not expose the same ADC/chirp/RX layout used by this project.
- Many examples are controlled measurements at fixed angles and poses.
- The negative class is primarily a bionic bird, not diverse real birds.
- One pole sample is not meaningful coverage of stationary clutter.
- It lacks substantial empty scenes, insects, people, vehicles, foliage, rain,
  rotating machinery, weak tracks, and other operational false alarms.

Archive data must not be fed directly into the deployed model merely because a
tensor can be resized. Archive and IWR6843 recordings must first be converted
to a documented common micro-Doppler representation, and transfer performance
must be measured explicitly.

## 4. Phase 1 feature contract

Start with one model input rather than the three-branch production design in
`AI.md`:

| Tensor | Proposed shape | Description |
|---|---|---|
| `micro_doppler` | `[2, 48, 64]` | Approximately 1.6 seconds of target-gated Doppler history |
| `time_mask` | `[48]` | Valid positions in the temporal window |

The two micro-Doppler channels should be:

1. target-gate `log1p` power;
2. target-to-local-background contrast computed from range bins beside the
   target gate.

The current five-bin target gate is a suitable initial central gate, but the
background bins must not overlap it. Preserve all 64 Doppler bins, including
zero Doppler. Apply clipping and standardization parameters calculated only
from the training partition. Padding is zero after standardization and is
identified by `time_mask`.

Each saved example must include:

- class label: `drone`, `bird`, or `unknown`;
- session and recording ID;
- start and end timestamps;
- source-data hash;
- radar-configuration fingerprint;
- feature-pipeline version;
- target-selection or track ID when tracking becomes available;
- tensor axis order, units, and masks.

`unknown` data is retained for robustness evaluation but excluded from the
binary classification loss.

## 5. Phase 1 model and training

Use a small depthwise-separable 2D CNN as the first neural baseline:

```text
Input [2, 48, 64]
    -> small convolutional stem
    -> 2-3 depthwise-separable residual blocks
    -> global pooling
    -> linear layer
    -> one drone logit
```

Target fewer than one million trainable parameters. This is a validation target,
not a reason to add complexity before a simpler model is measured.

Training rules:

1. Merge Mavic and Phantom recordings into the `drone` class.
2. Keep bionic-bird recordings as the initial `bird` class.
3. Split complete sessions and recordings, never individual frames or heavily
   overlapping windows.
4. Keep related angles, poses, or repeated captures from one experiment in the
   same partition wherever metadata permits.
5. Fit clipping, normalization, class weighting, and probability calibration
   using only training or validation data as appropriate.
6. Begin with weighted binary cross-entropy. Compare focal loss only if class
   imbalance or hard examples justify it.
7. Use radar-consistent augmentation: small amplitude changes, measured noise
   injection, limited time/range shifts, and masked frame dropout.
8. Do not use arbitrary image rotation, aggressive warping, or augmentation
   that breaks the relationship between target power and Doppler.
9. Report a micro-Doppler logistic-regression or shallow-CNN baseline before
   accepting the deeper model.
10. Pretrain on the archive only if it improves held-out IWR6843 results.

The primary development format should be PyTorch. Export a frozen candidate to
ONNX and verify numerical parity. ONNX Runtime is the portable initial inference
backend. TensorRT FP16 should be added only when Jetson deployment is confirmed
and measured on the actual target device; INT8 should follow only if calibration
data are representative and accuracy remains acceptable.

### Google Colab training workflow

Place the extracted archive recordings in Google Drive at:

```text
/content/drive/MyDrive/UREX/Data
```

The directory may contain the `.pkl` files directly or inside nested folders.
`classification.ipynb` discovers recordings recursively and ignores macOS
resource forks. Keep the Radar-UREx repository in one of these locations, or
set the `RADAR_UREX_REPO` environment variable before running the first cell:

```text
/content/Radar-UREx
/content/drive/MyDrive/UREX/Radar-UREx
```

When Colab is detected, the first notebook cell mounts Google Drive and selects
`/content/drive/MyDrive/UREX/Data` automatically. The generated feature cache and
future model artifacts persist across Colab sessions under:

```text
/content/drive/MyDrive/UREX/Radar-UREx-output
```

Before training:

1. Select a GPU runtime in Colab.
2. Run `classification.ipynb` in order through dataset auditing, feature
   generation, grouped splitting, and training-only normalization.
3. Run the CNN dependency, architecture, and data-pipeline cells.
4. Run the CNN training cell manually.
5. After training completes, run the calibration, locked-test evaluation, and
   export cell.

Opening the notebook does not train the CNN. The CNN training and export cells
are intentionally unexecuted in the repository version. Do not bypass the
grouped split, fit preprocessing on validation/test data, or inspect locked-test
metrics during model selection.

## 6. Evaluation

Evaluate on complete held-out recordings and sessions. Report:

- drone recall;
- bird false-positive rate;
- precision, F1, PR-AUC, and ROC-AUC;
- confusion matrix at the selected validation threshold;
- probability calibration error;
- performance by range, angle, signal strength, target motion, and drone model;
- results on drone models and IWR6843 sessions absent from training;
- archive-only, IWR6843-only, and archive-pretrained/IWR6843-fine-tuned results;
- feature-generation, model-inference, and end-to-end latency;
- rejection rate for missing, incompatible, or low-quality inputs.

Do not claim system-level false alarms per hour from the archive. That metric
becomes valid only after collecting long, labelled, drone-free IWR6843 sessions
covering the intended operating environment.

## 7. What needs to be done next

The next work should be completed in dependency order.

### Priority 1: make captured data reproducible and labelable

1. Define a session manifest containing radar/setup hashes, timestamps,
   location, weather, target class, target model, distance, aspect, motion, and
   operator notes.
2. Add a feature-recording output that saves range-Doppler power and the range
   information needed to regenerate micro-Doppler windows. Do not rely on plot
   images as training data.
3. Store valid-frame and processing-drop information beside each session so
   missing history is not interpreted as a physical signal.
4. Verify the external archive with a source hash and generated manifest.

**Exit condition:** one raw session can deterministically regenerate identical
versioned features and labels.

### Priority 2: validate and harden persistent target tracking

1. Build labelled replay cases for crossing targets, missed detections, clutter,
   and track loss.
2. Add Doppler-bin continuity and timestamps to the current XYZ association.
3. Publish a stable track ID and explicit confirmed, predicted, and lost state
   for feature recording.
4. Measure identity switches and tune the association gate, confirmation count,
   and missed-update limit on validation sessions.

**Exit condition:** labelled replay tests show that a micro-Doppler window stays
on the intended physical target through ordinary missed detections.

### Priority 3: build one shared offline/online feature pipeline

1. Implement the `[2, 48, 64]` feature contract and masks.
2. Use the same feature code for saved-data generation and live inference.
3. Add deterministic shape, axis-order, boundary-mask, and configuration-
   mismatch tests.
4. Compare the feature numerically with the existing live spectrogram.

**Exit condition:** offline replay and live processing produce equivalent
features from the same valid frames within a defined tolerance.

### Priority 4: establish the dataset and baseline

1. Convert the external archive to the common representation where physically
   defensible.
2. Collect labelled IWR6843 drone and bird sessions across multiple ranges,
   aspects, motions, days, and sites.
3. Create grouped train, validation, and locked-test partitions.
4. Train and report simple baselines before the CNN.

**Exit condition:** a leakage-checked dataset manifest and reproducible baseline
report exist.

### Priority 5: train, calibrate, and integrate Phase 1

1. Train the compact CNN and compare it with the baselines.
2. Fit probability calibration and the decision threshold on validation
   sessions only.
3. Export to ONNX and test PyTorch/ONNX parity, invalid inputs, early windows,
   batching, and latency.
4. Add inference after feature generation in the processing process or in a
   separate bounded worker. Never block the UDP receiver on ML inference.
5. Smooth per-track probabilities and expose an `unknown` result for inadequate
   history, configuration mismatch, or failed track quality.

**Exit condition:** the frozen model passes the locked IWR6843 test and runs
without increasing UDP capture loss or unbounded queue growth.

### Priority 6: expand toward the production objective

Collect long drone-free sessions and difficult non-drone examples: real birds,
insects, people, vehicles, foliage, rain, fans, weak targets, and crossing or
merged tracks. Only then evaluate drone-versus-general-non-drone classification,
false classifications per hour, the additional range-Doppler and track-feature
branches in `AI.md`, and deployment-specific acceleration.

## 8. Decision

The immediate project goal should be a **tracked drone-versus-bird
micro-Doppler baseline**, built from the repository's DCA1000 raw-ADC processing
path. The current code supplies the necessary range and Doppler computations,
but reliable ML development first requires reproducible labelled sessions,
persistent tracking, and a shared feature pipeline.

The external archive may accelerate pretraining, but success is determined by
held-out IWR6843 data. A general drone detector and a multi-input fused model are
later milestones that require substantially broader data and stronger upstream
tracking than the project currently has.
