import json
import tempfile
import unittest
from collections import deque
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import ANY, patch

import numpy as np

from main.inference import INPUT_SHAPE_CHW, WINDOW_STEPS
from main.tensorrt_inference import (
    PARITY_PROBABILITY_TOLERANCE,
    TensorRTDroneBirdInference,
    TensorRTInferenceError,
    create_inference_engine,
    resolve_classification_device,
)


class _Calibrator:
    def predict_proba(self, values):
        probability = float(np.asarray(values)[0, 0])
        return np.asarray(((1.0 - probability, probability),))


class _Runtime:
    allocation_bytes = 24_580

    def __init__(self, logit=0.8):
        self.logit = logit
        self.inputs = []

    def infer_logit(self, normalized):
        self.inputs.append(np.asarray(normalized).copy())
        return self.logit

    def close(self):
        pass


class DeviceResolutionTests(unittest.TestCase):
    def test_auto_requires_cuda_on_jetson(self):
        with patch("main.tensorrt_inference.is_jetson", return_value=True):
            self.assertEqual(resolve_classification_device("auto"), "cuda")

    def test_auto_uses_cpu_off_jetson(self):
        with patch("main.tensorrt_inference.is_jetson", return_value=False):
            self.assertEqual(resolve_classification_device("auto"), "cpu")

    def test_cuda_creation_does_not_fall_back(self):
        with patch(
            "main.tensorrt_inference.TensorRTDroneBirdInference",
            side_effect=TensorRTInferenceError("unavailable"),
        ):
            with self.assertRaisesRegex(TensorRTInferenceError, "unavailable"):
                create_inference_engine(
                    Path("artifacts"),
                    SimpleNamespace(),
                    Path("profile.cfg"),
                    device="cuda",
                )

    def test_cuda_creation_forwards_progress_callback(self):
        progress_messages = []
        with patch(
            "main.tensorrt_inference.TensorRTDroneBirdInference"
        ) as inference_class:
            create_inference_engine(
                Path("model_weights"),
                SimpleNamespace(),
                Path("profile.cfg"),
                device="cuda",
                progress_callback=progress_messages.append,
            )
        inference_class.assert_called_once_with(
            Path("model_weights"),
            ANY,
            Path("profile.cfg"),
            parity_data_path=None,
            progress_callback=progress_messages.append,
        )


class EngineCacheTests(unittest.TestCase):
    def test_fp16_probability_tolerance_accepts_small_calibrated_drift(self):
        self.assertGreaterEqual(PARITY_PROBABILITY_TOLERANCE, 0.00217362)
        self.assertLess(PARITY_PROBABILITY_TOLERANCE, 0.01)

    def test_cache_requires_matching_engine_hash_and_parity(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            engine = root / "model.engine"
            metadata = root / "model.engine.json"
            engine.write_bytes(b"engine")
            instance = TensorRTDroneBirdInference.__new__(
                TensorRTDroneBirdInference
            )
            instance.engine_path = engine
            instance.metadata_path = metadata
            expected = {"format_version": 1, "precision": "fp16"}

            import hashlib

            metadata.write_text(
                json.dumps(
                    {
                        **expected,
                        "engine_sha256": hashlib.sha256(b"engine").hexdigest(),
                        "parity": {
                            "label_mismatches": 0,
                            "max_probability_error": (
                                PARITY_PROBABILITY_TOLERANCE
                            ),
                        },
                    }
                ),
                encoding="utf-8",
            )
            self.assertTrue(instance._cache_is_valid(expected))

            engine.write_bytes(b"changed")
            self.assertFalse(instance._cache_is_valid(expected))

    def test_cache_ignores_volatile_reported_total_gpu_memory(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            engine = root / "model.engine"
            metadata = root / "model.engine.json"
            engine.write_bytes(b"engine")
            instance = TensorRTDroneBirdInference.__new__(
                TensorRTDroneBirdInference
            )
            instance.engine_path = engine
            instance.metadata_path = metadata
            expected_gpu = {
                "device_index": 0,
                "name": "Orin",
                "compute_capability": [8, 7],
                "total_memory_bytes": 7_849_046_016,
                "cuda_runtime_version": 13030,
                "cuda_driver_version": 13020,
            }

            import hashlib

            metadata.write_text(
                json.dumps(
                    {
                        "format_version": 2,
                        "gpu": {
                            **expected_gpu,
                            "total_memory_bytes": 7_849_054_208,
                        },
                        "engine_sha256": hashlib.sha256(b"engine").hexdigest(),
                        "parity": {
                            "label_mismatches": 0,
                            "max_probability_error": 0.0,
                        },
                    }
                ),
                encoding="utf-8",
            )

            self.assertTrue(
                instance._cache_is_valid(
                    {"format_version": 2, "gpu": expected_gpu}
                )
            )

    def test_existing_engine_is_validated_without_recompiling(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            instance = TensorRTDroneBirdInference.__new__(
                TensorRTDroneBirdInference
            )
            instance.engine_path = Path(temporary_directory) / "model.engine"
            instance.engine_path.write_bytes(b"engine")
            instance._trt = object()
            instance._cudart = object()
            messages = []
            instance._report_progress = messages.append
            runtime = _Runtime()
            parity = {
                "windows": 1,
                "label_mismatches": 0,
                "max_probability_error": 0.0,
            }

            with (
                patch(
                    "main.tensorrt_inference._TensorRTRuntime",
                    return_value=runtime,
                ),
                patch.object(instance, "_cache_is_valid", return_value=False),
                patch.object(instance, "_validate_parity", return_value=parity),
                patch.object(instance, "_write_cache_metadata") as write_metadata,
                patch.object(instance, "_build_engine") as build_engine,
            ):
                loaded = instance._load_or_build_runtime({"format_version": 2})

            self.assertIs(loaded, runtime)
            build_engine.assert_not_called()
            write_metadata.assert_called_once_with(
                {"format_version": 2}, parity
            )
            self.assertTrue(
                any("without recompiling" in message for message in messages)
            )

    def test_engine_is_compiled_only_when_cache_file_is_absent(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            instance = TensorRTDroneBirdInference.__new__(
                TensorRTDroneBirdInference
            )
            instance.engine_path = Path(temporary_directory) / "model.engine"
            instance.onnx_path = Path(temporary_directory) / "model.onnx"
            instance._trt = object()
            instance._cudart = object()
            instance._report_progress = lambda _message: None
            runtime = _Runtime()

            def create_engine(_expected):
                instance.engine_path.write_bytes(b"engine")

            with (
                patch(
                    "main.tensorrt_inference._TensorRTRuntime",
                    return_value=runtime,
                ),
                patch.object(
                    instance,
                    "_build_engine",
                    side_effect=create_engine,
                ) as build_engine,
            ):
                loaded = instance._load_or_build_runtime({"format_version": 2})

            self.assertIs(loaded, runtime)
            build_engine.assert_called_once_with({"format_version": 2})

    def test_parity_uses_training_export_without_pytorch(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            instance = TensorRTDroneBirdInference.__new__(
                TensorRTDroneBirdInference
            )
            instance.parity_data_path = (
                Path(temporary_directory) / "model_parity.npz"
            )
            instance.calibrator = _Calibrator()
            instance.threshold = 0.75
            np.savez_compressed(
                instance.parity_data_path,
                normalized_windows=np.zeros(
                    (1, *INPUT_SHAPE_CHW), dtype=np.float32
                ),
                probabilities=np.asarray([0.8], dtype=np.float64),
            )

            parity = instance._validate_parity(_Runtime(0.8))

            self.assertEqual(parity["windows"], 1)
            self.assertEqual(parity["label_mismatches"], 0)
            self.assertEqual(parity["max_probability_error"], 0.0)


class StatefulTensorRTTests(unittest.TestCase):
    def test_feature_history_runs_tensor_rt_at_step_48(self):
        instance = TensorRTDroneBirdInference.__new__(
            TensorRTDroneBirdInference
        )
        instance._history = deque(maxlen=WINDOW_STEPS)
        instance.threshold = 0.75
        instance.clip_low = np.asarray((-100.0, -100.0), dtype=np.float32)
        instance.clip_high = np.asarray((100.0, 100.0), dtype=np.float32)
        instance.channel_mean = np.zeros(2, dtype=np.float32)
        instance.channel_std = np.ones(2, dtype=np.float32)
        instance.calibrator = _Calibrator()
        instance.runtime = _Runtime(0.8)
        step = np.ones((2, 64), dtype=np.float32)

        for _ in range(WINDOW_STEPS):
            result = instance.update_feature_step(step)

        self.assertEqual(result.label, "drone")
        self.assertEqual(result.valid_steps, WINDOW_STEPS)
        self.assertEqual(len(instance.runtime.inputs), 1)


if __name__ == "__main__":
    unittest.main()
