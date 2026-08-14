import hashlib
import tempfile
import unittest
from collections import deque
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from inference import (
    CNN_CONFIG,
    DOPPLER_BINS,
    DroneBirdInference,
    FEATURE_VERSION,
    TARGET_GATE_BINS,
    WINDOW_STEPS,
    doppler_cube_to_feature_step,
    normalized_profile_sha256,
    reduce_centered_doppler,
)


class FeatureExtractionTests(unittest.TestCase):
    def test_reduces_centered_doppler_by_power_averaging(self) -> None:
        values = np.arange(128, dtype=np.float32)
        reduced = reduce_centered_doppler(values)

        self.assertEqual(reduced.shape, (DOPPLER_BINS,))
        self.assertEqual(reduced[0], 0.5)
        self.assertEqual(reduced[-1], 126.5)

    def test_feature_step_matches_notebook_power_formula(self) -> None:
        cube = np.ones((128, 3, 4, 64), dtype=np.complex64)
        cube[..., 15] *= 2.0
        cube[..., 16] *= 3.0
        cube[..., 17] *= 4.0

        step = doppler_cube_to_feature_step(cube, 16)

        self.assertEqual(step.shape, (2, 64))
        expected_target_power = (4.0 + 9.0 + 16.0) / 3.0
        np.testing.assert_allclose(
            step[0], np.log1p(expected_target_power), rtol=1e-6
        )
        np.testing.assert_allclose(
            step[1],
            np.log1p(expected_target_power) - np.log1p(1.0),
            rtol=1e-6,
        )

    def test_rejects_edge_gate_and_non_finite_cube(self) -> None:
        cube = np.ones((128, 3, 4, 64), dtype=np.complex64)
        with self.assertRaisesRegex(ValueError, "Insufficient separated"):
            doppler_cube_to_feature_step(cube, 0)

        cube[0, 0, 0, 16] = np.nan
        with self.assertRaisesRegex(ValueError, "non-finite"):
            doppler_cube_to_feature_step(cube, 16)

    def test_profile_hash_is_independent_of_crlf(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            lf_path = Path(temp_dir) / "lf.cfg"
            crlf_path = Path(temp_dir) / "crlf.cfg"
            lf_path.write_bytes(b"profileCfg 1\nframeCfg 2\n")
            crlf_path.write_bytes(b"profileCfg 1\r\nframeCfg 2\r\n")

            self.assertEqual(
                normalized_profile_sha256(lf_path),
                normalized_profile_sha256(crlf_path),
            )
            self.assertEqual(
                normalized_profile_sha256(lf_path),
                hashlib.sha256(lf_path.read_bytes()).hexdigest(),
            )


class _FakeTensorResult:
    def __init__(self, value: float) -> None:
        self.value = value

    def cpu(self):
        return self

    def numpy(self) -> np.ndarray:
        return np.asarray((self.value,), dtype=np.float32)


class _FakeModel:
    def __init__(self, logit: float) -> None:
        self.logit = logit

    def __call__(self, _tensor):
        return _FakeTensorResult(self.logit)


class _FakeCalibrator:
    def __init__(self, probability: float) -> None:
        self.probability = probability

    def predict_proba(self, values: np.ndarray) -> np.ndarray:
        self.last_values = values
        return np.asarray(
            ((1.0 - self.probability, self.probability),),
            dtype=np.float64,
        )


class _FakeTorch:
    @staticmethod
    def inference_mode():
        return nullcontext()

    @staticmethod
    def from_numpy(value: np.ndarray) -> np.ndarray:
        return value


def _inference_without_artifacts(probability: float) -> DroneBirdInference:
    engine = DroneBirdInference.__new__(DroneBirdInference)
    engine.config = SimpleNamespace(
        num_adc_samples=64,
        num_rx_channels=4,
        num_chirps_per_frame=384,
        num_loops=128,
        num_chirps_per_loop=3,
    )
    engine._history = deque(maxlen=WINDOW_STEPS)
    engine.threshold = 0.75
    engine.clip_low = np.asarray((-100.0, -100.0), dtype=np.float32)
    engine.clip_high = np.asarray((100.0, 100.0), dtype=np.float32)
    engine.channel_mean = np.zeros(2, dtype=np.float32)
    engine.channel_std = np.ones(2, dtype=np.float32)
    engine._torch = _FakeTorch()
    engine.model = _FakeModel(0.25)
    engine.calibrator = _FakeCalibrator(probability)
    return engine


class StatefulInferenceTests(unittest.TestCase):
    def test_waits_for_48_steps_then_returns_drone(self) -> None:
        engine = _inference_without_artifacts(0.75)
        cube = np.ones((128, 3, 4, 64), dtype=np.complex64)

        for expected_steps in range(1, WINDOW_STEPS):
            result = engine.update(cube, 16)
            self.assertEqual(result.label, "unknown")
            self.assertEqual(result.valid_steps, expected_steps)

        result = engine.update(cube, 16)
        self.assertEqual(result.label, "drone")
        self.assertEqual(result.status, "ready")
        self.assertEqual(result.p_drone, 0.75)

    def test_probability_below_threshold_is_not_drone(self) -> None:
        engine = _inference_without_artifacts(0.749)
        cube = np.ones((128, 3, 4, 64), dtype=np.complex64)

        for _ in range(WINDOW_STEPS):
            result = engine.update(cube, 16)

        self.assertEqual(result.label, "not_drone")

    def test_precomputed_feature_steps_match_cube_update_history(self) -> None:
        cube_engine = _inference_without_artifacts(0.75)
        feature_engine = _inference_without_artifacts(0.75)
        cube = np.ones((128, 3, 4, 64), dtype=np.complex64)
        step = doppler_cube_to_feature_step(cube, 16)

        for _ in range(WINDOW_STEPS):
            cube_result = cube_engine.update(cube, 16)
            feature_result = feature_engine.update_feature_step(step)

        self.assertEqual(feature_result, cube_result)
        self.assertEqual(feature_engine.valid_steps, WINDOW_STEPS)

    def test_invalid_shape_resets_accumulated_history(self) -> None:
        engine = _inference_without_artifacts(0.9)
        cube = np.ones((128, 3, 4, 64), dtype=np.complex64)
        engine.update(cube, 16)

        result = engine.update(cube[:64], 16)

        self.assertEqual(result.label, "unknown")
        self.assertEqual(result.valid_steps, 0)
        self.assertIn("invalid_doppler_shape", result.reason)


class ArtifactContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.engine = DroneBirdInference.__new__(DroneBirdInference)
        self.engine.profile_path = Path("rawdatacapture/profile.cfg")
        self.engine.config = SimpleNamespace(
            num_adc_samples=64,
            num_rx_channels=4,
            num_chirps_per_frame=384,
            num_loops=128,
            num_chirps_per_loop=3,
            tx_channel_masks=(1, 4, 2),
        )
        self.profile_hash = normalized_profile_sha256(
            self.engine.profile_path
        )
        self.checkpoint = {
            "architecture": "MicroDopplerCNN",
            "input_shape_chw": (2, 48, 64),
            "feature_version": FEATURE_VERSION,
            "cnn_config": dict(CNN_CONFIG),
            "target_gate_range_bins": TARGET_GATE_BINS,
            "state_dict": {},
        }
        self.calibration = {
            "calibrator": object(),
            "threshold": 0.9,
            "clip_low": np.zeros(2),
            "clip_high": np.ones(2),
            "channel_mean": np.zeros(2),
            "channel_std": np.ones(2),
            "compatible_profile_sha256": self.profile_hash,
        }
        self.card = {
            "input_shape_chw": [2, 48, 64],
            "feature_version": FEATURE_VERSION,
            "target_gate_range_bins": TARGET_GATE_BINS,
            "compatible_profile_sha256": self.profile_hash,
        }

    def test_accepts_current_profile_and_artifact_contract(self) -> None:
        self.engine._validate_artifacts(
            self.checkpoint,
            self.calibration,
            self.card,
        )

    def test_rejects_incompatible_profile_fingerprint(self) -> None:
        self.calibration["compatible_profile_sha256"] = "0" * 64
        self.card["compatible_profile_sha256"] = "0" * 64

        with self.assertRaisesRegex(ValueError, "configuration mismatch"):
            self.engine._validate_artifacts(
                self.checkpoint,
                self.calibration,
                self.card,
            )

    def test_rejects_old_two_bin_target_gate_contract(self) -> None:
        self.checkpoint["target_gate_range_bins"] = 2

        with self.assertRaisesRegex(ValueError, "target range-bin contract"):
            self.engine._validate_artifacts(
                self.checkpoint,
                self.calibration,
                self.card,
            )


if __name__ == "__main__":
    unittest.main()
