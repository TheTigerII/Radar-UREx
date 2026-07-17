# Machine Learning Objective for the IWR6843ISK-ODS Dataset

## Recommended objective

Build a **target-conditioned drone-versus-bird micro-Doppler classifier**:

> Given a tracked aerial target and the latest 1-2 seconds of radar returns, estimate `P(drone)` versus `P(bird)` in real time.

This is more defensible than calling it a general drone detector because the dataset contains almost no empty-scene or background data.

## Dataset contents

`Data.zip` contains 102 usable complex radar recordings:

| Class | Recordings |
|---|---:|
| DJI Mavic | 45 |
| Bionic bird | 38 |
| Phantom 3 Pro | 18 |
| Pole | 1 |

Most recordings are complex64 tensors with shape `1000 x 128 x 168`. Two 3-kHz recordings use shapes `2000 x 128 x 112` and `1000 x 128 x 112`.

The dataset covers:

- Azimuths from 0 to 180 degrees.
- Tripod azimuth and elevation combinations.
- Stationary and rotating propellers.
- Bird wing speeds of 5%, 50%, and 100%.
- Mavic hovering at 3-4 ft and moving toward the radar.
- Different absorber configurations.

The dataset was created for distinguishing drones from birds using 60-GHz micro-Doppler, amplitude, and phase information. The associated research explored amplitude CNNs and phase-based convolutional LSTMs.

## Recommended IWR6843ISK-ODS pipeline

```text
IWR6843ISK-ODS
    -> range-Doppler heatmap
    -> CFAR detection and target tracking
    -> target-centred micro-Doppler window
    -> lightweight CNN
    -> drone probability with temporal smoothing
```

Use the IWR6843 range-Doppler heatmap rather than expecting the archive's raw tensors to feed directly into the deployed model. TI's out-of-box ODS firmware can output the range-Doppler matrix as UART TLV type 5. Its size is:

```text
range FFT bins x Doppler FFT bins x 2 bytes
```

Relevant local documentation:

- `radar_toolbox_4_00_00_05/software_docs/Understanding_UART_Data_Output_Format.html`
- `radar_toolbox_4_00_00_05/source/ti/examples/Out_Of_Box_Demo/src/xwr6843ODS/main.c`

## Recommended model

- Input: `64 x 64` or `64 x 128` log-magnitude, target-centred micro-Doppler spectrogram.
- Architecture: small depthwise-separable 2D CNN or 1D temporal CNN.
- Model size target: fewer than 1-2 million parameters.
- Output: binary sigmoid or softmax for `bird` versus `drone`.
- Deployment: export through ONNX to TensorRT FP16 initially, followed by INT8 calibration if accuracy remains acceptable.
- Update rate: approximately 5-10 predictions per second using a sliding temporal window.
- Inference target: less than 10 ms, to be measured on the actual Jetson.

The Jetson Orin Nano Super has sufficient compute for a model of this size. The historical radar window will probably contribute more latency than the neural-network inference.

## Sensor-transfer limitation

The archive is useful for pretraining, but it is not sufficient for final IWR6843 deployment:

- It was recorded using a different 60-GHz acquisition system.
- The tensor does not contain an explicit antenna dimension.
- Many examples are controlled tripod measurements at fixed angles.
- The bird is a bionic bird rather than a diverse collection of real birds.
- There is only one pole recording and no substantial collection of empty scenes, ground clutter, people, trees, vehicles, rain, or other false-alarm sources.
- The normal IWR6843 USB interface provides processed outputs. Raw ADC collection requires a DCA1000EVM.

Therefore, convert the archive and the IWR6843 recordings into the same target-centred range-Doppler or micro-Doppler representation. Pretrain on the archive, then fine-tune and validate using data collected with the IWR6843ISK-ODS.

## Training procedure

1. Merge Mavic and Phantom 3 Pro into a single `drone` class.
2. Convert each complex recording into target-centred, log-scaled micro-Doppler windows.
3. Split the data by complete recording, angle, pose, and measurement session.
4. Do not randomly split individual frames from the same recording across training and testing; this would produce serious data leakage.
5. Train a lightweight CNN using binary cross-entropy or focal loss.
6. Use angle, signal level, artificial noise, clutter, and small frequency/time shifts as augmentations.
7. Collect IWR6843 examples at several distances, angles, locations, days, and target speeds.
8. Fine-tune the model on IWR6843 recordings.
9. Test on locations, sessions, and drone models that were excluded from training.
10. Export the final model to TensorRT and measure end-to-end latency on the Jetson.

Mavic-versus-Phantom classification can be retained as an auxiliary research benchmark, but it should not be the primary deployment objective because it is unlikely to generalize to unseen drone models.

## Evaluation metrics

Measure:

- Drone recall or probability of detection.
- Bird false-positive rate.
- Precision, F1 score, and AUROC.
- Performance by distance, angle, and signal-to-noise ratio.
- Performance on unseen drone models and measurement locations.
- TensorRT inference latency and memory consumption.
- System-level false alarms per hour after background data have been collected.

The current archive cannot measure false alarms per hour because it does not contain enough no-target and environmental background data.

## Verdict

This dataset can support a deployable **drone-versus-bird classifier that operates after conventional radar detection and tracking**. It cannot, by itself, support a reliable general-purpose drone-presence detector. The strongest development strategy is:

```text
archive pretraining
    -> common micro-Doppler representation
    -> IWR6843 fine-tuning
    -> TensorRT deployment on Jetson Orin Nano
```

## References

- [IWR6843ISK-ODS evaluation board](https://www.ti.com/tool/IWR6843ISK-ODS)
- [Jetson Orin Nano Developer Kit guide](https://docs.nvidia.com/jetson/orin-nano-devkit/user-guide/index.html)
- [Millimeter Wave Radar Measurements: Distinguishing UAS and Birds Based on 60 GHz Micro-Doppler Signatures](https://cris.vtt.fi/en/publications/millimeter-wave-radar-measurements-distinguishing-uas-and-birds-b/)
- [Classification of Flying Drones Using Millimeter-Wave Radar](https://www.mdpi.com/1424-8220/25/3/721)
