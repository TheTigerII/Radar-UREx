import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from inference import (
    DOPPLER_BINS,
    LABEL_TO_INDEX,
    SEGMENT_FRAMES,
    MmHawkeyeLSTM,
    RealtimeUavClassifier,
    align_doppler_time,
    resolve_model_state_path,
)


def _runtime_contract() -> dict:
    return {
        "profile_fingerprint": "profile-test",
        "feature_fingerprint": "feature-test",
        "feature_version": "mini4-pmm-tracking-v4",
    }


def _write_checkpoint(directory: Path, **overrides) -> Path:
    model = MmHawkeyeLSTM()
    checkpoint = {
        "state_dict": model.state_dict(),
        "architecture": {
            "input_size": DOPPLER_BINS,
            "hidden_size": 128,
            "num_layers": 2,
            "num_classes": 2,
        },
        "input_shape": [SEGMENT_FRAMES, DOPPLER_BINS],
        "label_to_index": dict(LABEL_TO_INDEX),
        "normalization": {"mean": 0.0, "std": 1.0},
        **_runtime_contract(),
    }
    checkpoint.update(overrides)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "model_state.pt"
    torch.save(checkpoint, path)
    return path


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
    def test_missing_model_state_is_reported(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(FileNotFoundError, "model_state.pt"):
                resolve_model_state_path(Path(directory))

    def test_contract_mismatch_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            weights = Path(directory) / "model_weights"
            _write_checkpoint(weights, feature_fingerprint="different")

            with self.assertRaisesRegex(ValueError, "feature_fingerprint mismatch"):
                RealtimeUavClassifier(weights, _runtime_contract(), device="cpu")

    def test_warms_up_then_classifies_a_full_quality_gated_history(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            weights = Path(directory) / "model_weights"
            _write_checkpoint(weights)
            classifier = RealtimeUavClassifier(
                weights,
                _runtime_contract(),
                device="cpu",
            )

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
            _write_checkpoint(weights)
            classifier = RealtimeUavClassifier(
                weights,
                _runtime_contract(),
                device="cpu",
            )
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
