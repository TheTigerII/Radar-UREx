import json
import tempfile
import unittest
from datetime import datetime
from pathlib import Path

from main.classification_evaluation import (
    ClassificationEvaluationLogger,
    FORMAT_VERSION,
    calculate_metrics,
    default_inference_log_path,
)
from main.inference import ClassificationResult


def result(
    label: str,
    probability: float | None,
    *,
    status: str = "classified",
    history_frames: int = 36,
) -> ClassificationResult:
    probabilities = (
        {"other": 1.0 - probability, "uav": probability}
        if probability is not None
        else None
    )
    confidence = (
        max(probabilities.values()) if probabilities is not None else None
    )
    return ClassificationResult(
        status=status,
        label=label,
        history_frames=history_frames,
        maximum_pmm_score=1000.0,
        threshold=700.0,
        probabilities=probabilities,
        confidence=confidence,
        inference_ms=0.5 if probabilities is not None else None,
    )


def observation(
    truth: str,
    classification: ClassificationResult,
    index: int,
) -> dict:
    return {
        "frame_index": index,
        "capture_elapsed_s": index * 0.1,
        "ground_truth": truth,
        "result": classification.to_dict(),
        "predicted_confidence": classification.confidence,
        "classification_latency_ms": 1.0,
        "history": {"reset_requested": False},
    }


class MetricTests(unittest.TestCase):
    def test_native_labels_are_mapped_for_binary_metrics(self) -> None:
        observations = [
            observation("drone", result("uav", 0.9), 1),
            observation("drone", result("other", 0.2), 2),
            observation("not_drone", result("uav", 0.8), 3),
            observation("not_drone", result("other", 0.1), 4),
        ]

        metrics = calculate_metrics(observations, frame_period_s=0.1)

        self.assertEqual(
            metrics["confusion_matrix"],
            {"tn": 1, "fp": 1, "fn": 1, "tp": 1},
        )
        self.assertEqual(metrics["overall_accuracy"], 0.5)
        self.assertEqual(metrics["drone"]["class_accuracy"], 0.5)
        self.assertEqual(metrics["not_drone"]["class_accuracy"], 0.5)
        self.assertAlmostEqual(metrics["predicted_duration_s"]["drone"], 0.2)
        self.assertAlmostEqual(
            metrics["predicted_duration_s"]["not_drone"], 0.2
        )

    def test_unknown_attempt_affects_coverage_not_ready_accuracy(self) -> None:
        observations = [
            observation(
                "drone",
                result(
                    "unknown",
                    None,
                    status="warming_up",
                    history_frames=1,
                ),
                1,
            ),
            observation("drone", result("uav", 0.9), 2),
        ]

        metrics = calculate_metrics(observations, frame_period_s=0.1)

        self.assertEqual(metrics["overall_accuracy"], 1.0)
        self.assertEqual(metrics["operational_correctness"], 0.5)
        self.assertEqual(metrics["readiness_coverage"], 0.5)
        self.assertEqual(metrics["unavailable_reasons"], {"warming_up": 1})


class EvaluationLoggerTests(unittest.TestCase):
    def _logger(
        self,
        path: Path,
        truth: str,
    ) -> ClassificationEvaluationLogger:
        profile = path.parent / "profile.cfg"
        profile.write_text("profileCfg test\n", encoding="utf-8")
        return ClassificationEvaluationLogger(
            path,
            ground_truth=truth,
            frame_period_s=0.1,
            classifier_metadata={
                "runtime": "tensorrt",
                "device": "cuda",
                "input_shape": [36, 64],
                "label_to_index": {"other": 0, "uav": 1},
                "profile_fingerprint": "profile",
                "feature_fingerprint": "feature",
                "feature_version": "test-v1",
            },
            pmm_metadata={"config": {"history_seconds": 3.6}},
            profile_path=profile,
        )

    def test_default_path_is_timestamped_under_log_directory(self) -> None:
        now = datetime(2026, 8, 28, 13, 45, 12, 123456)
        path = default_inference_log_path(now)

        self.assertEqual(path.parent.name, "log")
        self.assertEqual(path.name, "20260828_134512_live_inference.jsonl")

    def test_stream_flushes_native_result_and_history_reset_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "live_inference.jsonl"
            logger = self._logger(path, "drone")
            logger.record(
                result("unknown", None, status="warming_up", history_frames=1),
                frame_index=12,
                captured_at_s=None,
                history_frames_before=35,
                reset_requested=True,
                reset_reason="tracking_history_restarted",
                steps_cleared=35,
                target_range_m=2.5,
                target_state="tentative",
            )

            flushed = [
                json.loads(line)
                for line in path.read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual(len(flushed), 2)
            self.assertEqual(flushed[0]["version"], FORMAT_VERSION)
            self.assertEqual(flushed[0]["label_mapping"]["uav"], "drone")
            self.assertEqual(flushed[1]["result"]["label"], "unknown")
            self.assertEqual(flushed[1]["history"]["steps_cleared"], 35)
            self.assertEqual(flushed[1]["history"]["history_generation"], 1)

            logger.record(
                result("uav", 0.85),
                frame_index=48,
                captured_at_s=None,
                history_frames_before=35,
                classification_latency_ms=2.0,
                target_range_m=2.4,
                target_state="confirmed",
            )
            logger.close()
            records = [
                json.loads(line)
                for line in path.read_text(encoding="utf-8").splitlines()
            ]

        ready = records[2]
        self.assertEqual(ready["result"]["label"], "uav")
        self.assertAlmostEqual(ready["predicted_confidence"], 0.85)
        self.assertTrue(ready["correct"])
        self.assertEqual(ready["classification_latency_ms"], 2.0)
        self.assertEqual(records[-2]["scope"], "run")
        self.assertEqual(records[-1]["scope"], "aggregate")

    def test_compatible_runs_aggregate_and_incomplete_log_is_excluded(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            directory = Path(temporary_directory)
            drone_logger = self._logger(directory / "drone.jsonl", "drone")
            drone_logger.record(
                result("uav", 0.9),
                frame_index=36,
                captured_at_s=None,
                history_frames_before=35,
            )
            drone_logger.close()

            other_logger = self._logger(
                directory / "not_drone.jsonl", "not_drone"
            )
            other_logger.record(
                result("other", 0.1),
                frame_index=36,
                captured_at_s=None,
                history_frames_before=35,
            )
            (directory / "interrupted.jsonl").write_text(
                '{"record_type":"metadata"}\n{"record_type":',
                encoding="utf-8",
            )
            _, aggregate = other_logger.close()

        self.assertEqual(aggregate["included_runs"], 2)
        self.assertEqual(aggregate["metrics"]["overall_accuracy"], 1.0)
        self.assertEqual(aggregate["metrics"]["roc_auc"], 1.0)
        self.assertEqual(
            aggregate["excluded_logs"]["malformed_or_incomplete"], 1
        )

    def test_failed_log_is_not_included_in_later_aggregation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            directory = Path(temporary_directory)
            failed = self._logger(directory / "failed.jsonl", "drone")
            failed.record(
                result("uav", 0.9),
                frame_index=36,
                captured_at_s=None,
                history_frames_before=35,
            )
            failed.close("failed")

            completed = self._logger(directory / "completed.jsonl", "drone")
            completed.record(
                result("uav", 0.9),
                frame_index=36,
                captured_at_s=None,
                history_frames_before=35,
            )
            _, aggregate = completed.close()

        self.assertEqual(aggregate["included_runs"], 1)
        self.assertEqual(aggregate["excluded_logs"]["failed"], 1)


if __name__ == "__main__":
    unittest.main()
