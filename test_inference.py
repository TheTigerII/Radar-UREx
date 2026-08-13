import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from inference import (
    DOPPLER_BINS,
    LABEL_TO_INDEX,
    SEGMENT_FRAMES,
    RealtimeUavClassifier,
    align_doppler_time,
    resolve_model_artifact_paths,
)


def _runtime_contract() -> dict:
    return {
        "profile_fingerprint": "profile-test",
        "feature_fingerprint": "feature-test",
        "feature_version": "mini4-pmm-tracking-v4",
    }


def _write_artifacts(directory: Path, **dataset_overrides) -> tuple[Path, Path]:
    dataset = {**_runtime_contract(), **dataset_overrides}
    manifest = {
        "artifacts": {"pytorch": "model_state.pt", "onnx": "model.onnx"},
        "architecture": {
            "input_size": DOPPLER_BINS,
            "hidden_size": 128,
            "num_layers": 2,
            "num_classes": 2,
        },
        "input_contract": {
            "dtype": "float32",
            "shape": ["batch", SEGMENT_FRAMES, DOPPLER_BINS],
        },
        "label_to_index": dict(LABEL_TO_INDEX),
        "normalization": {"mean": 0.0, "std": 1.0},
        "dataset": dataset,
    }
    directory.mkdir(parents=True, exist_ok=True)
    model_path = directory / "model.onnx"
    manifest_path = directory / "manifest.json"
    model_path.write_bytes(b"test ONNX model")
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return model_path, manifest_path


class _FakeNodeArg:
    def __init__(self, name: str, shape: list) -> None:
        self.name = name
        self.shape = shape
        self.type = "tensor(float)"


class _FakeSession:
    def __init__(self, model_path: str, *, providers: list[str]) -> None:
        self.model_path = model_path
        self.providers = providers

    def get_inputs(self) -> list[_FakeNodeArg]:
        return [_FakeNodeArg("doppler_time", ["batch", SEGMENT_FRAMES, DOPPLER_BINS])]

    def get_outputs(self) -> list[_FakeNodeArg]:
        return [_FakeNodeArg("logits", ["batch", len(LABEL_TO_INDEX)])]

    def get_providers(self) -> list[str]:
        return list(self.providers)

    def run(
        self,
        output_names: list[str],
        inputs: dict[str, np.ndarray],
    ) -> list[np.ndarray]:
        batch_size = inputs["doppler_time"].shape[0]
        logits = np.array([[0.25, 0.75]], dtype=np.float32)
        return [np.tile(logits, (batch_size, 1))]


class _FakeOrt:
    InferenceSession = _FakeSession

    @staticmethod
    def get_available_providers() -> list[str]:
        return ["CPUExecutionProvider"]


def _classifier(weights: Path) -> RealtimeUavClassifier:
    with patch("inference._load_onnxruntime", return_value=_FakeOrt):
        return RealtimeUavClassifier(weights, _runtime_contract(), device="cpu")


def _history(frame_count: int, peak_bin: int = 20) -> np.ndarray:
    values = np.full((DOPPLER_BINS, frame_count), -40.0, dtype=np.float32)
    values[peak_bin] = 20.0
    return values


class PreprocessingTests(unittest.TestCase):
    def test_alignment_centers_each_strongest_bin(self) -> None:
        history = _history(SEGMENT_FRAMES, peak_bin=17).T
        aligned = align_doppler_time(history)

        self.assertEqual(aligned.shape, (SEGMENT_FRAMES, DOPPLER_BINS))
        np.testing.assert_array_equal(
            np.argmax(np.expm1(aligned), axis=1),
            np.full(SEGMENT_FRAMES, DOPPLER_BINS // 2),
        )

    def test_wrong_history_shape_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "Expected"):
            align_doppler_time(np.zeros((35, DOPPLER_BINS), dtype=np.float32))


class RealtimeClassifierTests(unittest.TestCase):
    def test_missing_onnx_artifacts_are_reported(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(FileNotFoundError, "model.onnx"):
                resolve_model_artifact_paths(Path(directory))

    def test_contract_mismatch_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            weights = Path(directory) / "model_weights"
            _write_artifacts(weights, feature_fingerprint="different")

            with self.assertRaisesRegex(ValueError, "feature_fingerprint mismatch"):
                RealtimeUavClassifier(weights, _runtime_contract(), device="cpu")

    def test_declared_external_data_must_exist(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            weights = Path(directory) / "model_weights"
            _, manifest_path = _write_artifacts(weights)
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["artifacts"]["onnx_data"] = "model.onnx.data"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

            with self.assertRaisesRegex(FileNotFoundError, "external data"):
                RealtimeUavClassifier(weights, _runtime_contract(), device="cpu")

    def test_warms_up_then_classifies_a_full_quality_gated_history(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            weights = Path(directory) / "model_weights"
            _write_artifacts(weights)
            classifier = _classifier(weights)

            for frame_count in range(1, SEGMENT_FRAMES):
                result = classifier.classify(
                    _history(frame_count),
                    pmm_score=100.0,
                    threshold=50.0,
                )
                self.assertEqual(result.status, "warming_up")
                self.assertEqual(result.label, "unknown")

            result = classifier.classify(
                _history(SEGMENT_FRAMES),
                pmm_score=100.0,
                threshold=50.0,
            )

            self.assertEqual(result.status, "classified")
            self.assertIn(result.label, LABEL_TO_INDEX)
            self.assertIsNotNone(result.probabilities)
            self.assertAlmostEqual(sum(result.probabilities.values()), 1.0, places=6)
            self.assertIsNotNone(result.inference_ms)

    def test_full_low_score_history_stays_unknown(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            weights = Path(directory) / "model_weights"
            _write_artifacts(weights)
            classifier = _classifier(weights)
            result = None
            for frame_count in range(1, SEGMENT_FRAMES + 1):
                result = classifier.classify(
                    _history(frame_count),
                    pmm_score=10.0,
                    threshold=50.0,
                )

            self.assertIsNotNone(result)
            self.assertEqual(result.status, "below_pmm_threshold")
            self.assertEqual(result.label, "unknown")
            self.assertIsNone(result.probabilities)


if __name__ == "__main__":
    unittest.main()
