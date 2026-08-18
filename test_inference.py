import json
import subprocess
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
    ensure_tensorrt_engine,
    resolve_model_artifact_paths,
)


def _runtime_contract() -> dict:
    return {
        "profile_fingerprint": "profile-test",
        "feature_fingerprint": "feature-test",
        "feature_version": "mini4-pmm-tracking-v7",
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


class _FakeTensorRtSession:
    def run(
        self,
        output_names: list[str],
        inputs: dict[str, np.ndarray],
    ) -> list[np.ndarray]:
        batch_size = inputs["doppler_time"].shape[0]
        logits = np.array([[0.25, 0.75]], dtype=np.float32)
        return [np.tile(logits, (batch_size, 1))]


def _classifier(weights: Path) -> RealtimeUavClassifier:
    generated = weights / "generated"
    generated.mkdir()
    (generated / "model.fp16.engine").write_bytes(b"test TensorRT engine")
    with patch("inference._TensorRtSession", return_value=_FakeTensorRtSession()):
        return RealtimeUavClassifier(weights, _runtime_contract())


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

    def test_dc_noise_uses_only_frames_with_body_peak_away_from_center(self) -> None:
        amplitude = np.ones((SEGMENT_FRAMES, DOPPLER_BINS), dtype=np.float32)
        amplitude[:, DOPPLER_BINS // 2] = 10.0
        amplitude[: SEGMENT_FRAMES // 2, 12] = 100.0
        amplitude[SEGMENT_FRAMES // 2 :, DOPPLER_BINS // 2] = 50.0
        history_db = 20.0 * np.log10(amplitude)

        processed = np.expm1(align_doppler_time(history_db))

        np.testing.assert_allclose(
            processed[: SEGMENT_FRAMES // 2, DOPPLER_BINS // 2],
            100.0,
            atol=1e-4,
        )
        np.testing.assert_allclose(
            processed[SEGMENT_FRAMES // 2 :, DOPPLER_BINS // 2],
            40.0,
            atol=1e-4,
        )

    def test_all_hovering_segment_preserves_center_peak(self) -> None:
        history = _history(SEGMENT_FRAMES, peak_bin=DOPPLER_BINS // 2).T

        processed = np.expm1(align_doppler_time(history))

        np.testing.assert_array_equal(
            np.argmax(processed, axis=1),
            np.full(SEGMENT_FRAMES, DOPPLER_BINS // 2),
        )
        np.testing.assert_allclose(
            processed[:, DOPPLER_BINS // 2],
            10.0,
            rtol=1e-5,
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
                RealtimeUavClassifier(weights, _runtime_contract())

    def test_legacy_preprocessing_version_is_rejected_before_model_loading(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            weights = Path(directory) / "model_weights"
            _write_artifacts(weights, feature_version="mini4-pmm-tracking-v5")

            with self.assertRaisesRegex(ValueError, "feature_version mismatch"):
                RealtimeUavClassifier(weights, _runtime_contract())

    def test_declared_external_data_must_exist(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            weights = Path(directory) / "model_weights"
            _, manifest_path = _write_artifacts(weights)
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["artifacts"]["onnx_data"] = "model.onnx.data"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

            with self.assertRaisesRegex(FileNotFoundError, "external data"):
                RealtimeUavClassifier(weights, _runtime_contract())

    def test_tensorrt_engine_is_preferred_for_default_device(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            weights = Path(directory) / "model_weights"
            _write_artifacts(weights)
            engine_path = weights / "generated" / "model.fp16.engine"
            engine_path.parent.mkdir()
            engine_path.write_bytes(b"test TensorRT engine")
            fake_session = object()

            with patch("inference._TensorRtSession", return_value=fake_session):
                classifier = RealtimeUavClassifier(weights, _runtime_contract())

            self.assertIs(classifier.session, fake_session)
            self.assertEqual(classifier.metadata["runtime"], "tensorrt")
            self.assertEqual(classifier.metadata["device"], "cuda")
            self.assertEqual(classifier.metadata["model_path"], str(engine_path))

    def test_missing_engine_is_compiled_into_generated_directory(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            weights = Path(directory) / "model_weights"
            model_path, _ = _write_artifacts(weights)

            def fake_run(command: list[str], *, check: bool):
                self.assertFalse(check)
                destination = next(
                    value.removeprefix("--saveEngine=")
                    for value in command
                    if value.startswith("--saveEngine=")
                )
                Path(destination).write_bytes(b"compiled TensorRT engine")
                return subprocess.CompletedProcess(command, 0)

            with (
                patch("inference.shutil.which", return_value="/usr/bin/trtexec"),
                patch("inference.subprocess.run", side_effect=fake_run),
                patch("builtins.print") as output,
            ):
                engine_path = ensure_tensorrt_engine(model_path)

            self.assertEqual(
                engine_path,
                weights / "generated" / "model.fp16.engine",
            )
            self.assertEqual(engine_path.read_bytes(), b"compiled TensorRT engine")
            self.assertTrue(
                any("compiling it now" in str(call) for call in output.call_args_list)
            )

    def test_single_hardware_named_generated_engine_is_reused(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            weights = Path(directory) / "model_weights"
            model_path, _ = _write_artifacts(weights)
            generated = weights / "generated"
            generated.mkdir()
            engine_path = generated / "model_sm87_fp16.engine"
            engine_path.write_bytes(b"existing TensorRT engine")

            with patch("inference.subprocess.run") as compiler:
                resolved = ensure_tensorrt_engine(model_path)

            self.assertEqual(resolved, engine_path)
            compiler.assert_not_called()

    def test_cpu_device_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            weights = Path(directory) / "model_weights"
            _write_artifacts(weights)

            with self.assertRaisesRegex(ValueError, "TensorRT GPU-only"):
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

    def test_quality_gate_compares_each_score_with_its_adaptive_threshold(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            weights = Path(directory) / "model_weights"
            _write_artifacts(weights)
            classifier = _classifier(weights)

            for frame_count in range(1, SEGMENT_FRAMES + 1):
                result = classifier.classify(
                    _history(frame_count),
                    pmm_score=100.0,
                    threshold=50.0 if frame_count == 1 else 200.0,
                )

            self.assertEqual(result.status, "classified")


if __name__ == "__main__":
    unittest.main()
