import json
import tempfile
import unittest
from pathlib import Path

from main.classification_evaluation import (
    ClassificationEvaluationLogger,
    calculate_metrics,
)
from main.inference import InferenceResult


def result(
    label: str,
    probability: float | None,
    *,
    status: str = "ready",
    reason: str | None = None,
    valid_steps: int = 48,
) -> InferenceResult:
    return InferenceResult(
        label=label,
        p_drone=probability,
        threshold=0.5,
        status=status,
        reason=reason,
        valid_steps=valid_steps,
    )


class MetricTests(unittest.TestCase):
    def test_binary_metrics_and_durations_are_calculated(self) -> None:
        observations = []
        cases = (
            ("drone", result("drone", 0.9)),
            ("drone", result("not_drone", 0.2)),
            ("not_drone", result("drone", 0.8)),
            ("not_drone", result("not_drone", 0.1)),
        )
        for index, (truth, inference_result) in enumerate(cases, start=1):
            observations.append(
                {
                    "frame_index": index,
                    "capture_elapsed_s": index * 0.1,
                    "ground_truth": truth,
                    "result": inference_result.to_dict(),
                    "predicted_confidence": 0.9,
                    "classification_latency_ms": 1.0,
                    "history": {"action": "append", "reset_requested": False},
                }
            )

        metrics = calculate_metrics(observations, frame_period_s=0.05)

        self.assertEqual(
            metrics["confusion_matrix"],
            {"tn": 1, "fp": 1, "fn": 1, "tp": 1},
        )
        self.assertEqual(metrics["overall_accuracy"], 0.5)
        self.assertEqual(metrics["drone"]["class_accuracy"], 0.5)
        self.assertEqual(metrics["not_drone"]["class_accuracy"], 0.5)
        self.assertAlmostEqual(metrics["predicted_duration_s"]["drone"], 0.1)
        self.assertAlmostEqual(
            metrics["predicted_duration_s"]["not_drone"], 0.1
        )

    def test_unknown_attempt_affects_coverage_not_ready_accuracy(self) -> None:
        observations = [
            {
                "frame_index": 1,
                "ground_truth": "drone",
                "result": result(
                    "unknown",
                    None,
                    status="waiting",
                    reason="insufficient_history",
                    valid_steps=1,
                ).to_dict(),
                "history": {"action": "append", "reset_requested": False},
            },
            {
                "frame_index": 2,
                "ground_truth": "drone",
                "result": result("drone", 0.9).to_dict(),
                "predicted_confidence": 0.9,
                "history": {"action": "append", "reset_requested": False},
            },
        ]

        metrics = calculate_metrics(observations, frame_period_s=0.05)

        self.assertEqual(metrics["overall_accuracy"], 1.0)
        self.assertEqual(metrics["operational_correctness"], 0.5)
        self.assertEqual(metrics["readiness_coverage"], 0.5)


class EvaluationLoggerTests(unittest.TestCase):
    def _logger(
        self,
        path: Path,
        truth: str,
    ) -> ClassificationEvaluationLogger:
        return ClassificationEvaluationLogger(
            path,
            ground_truth=truth,
            frame_period_s=0.05,
            classifier_metadata={
                "model": "test_cnn",
                "backend": "pytorch",
                "feature_version": "test-v1",
                "threshold": 0.5,
            },
            artifact_dir=path.parent / "artifacts",
            profile_path=path.parent / "profile.cfg",
        )

    def test_stream_contains_history_reset_metadata_and_summaries(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "live_inference_one.jsonl"
            logger = self._logger(path, "drone")
            logger.record(
                result(
                    "unknown",
                    None,
                    status="waiting",
                    reason="target_changed",
                    valid_steps=0,
                ),
                frame_index=12,
                captured_at_s=None,
                valid_steps_before=31,
                reset_requested=True,
                reset_reason="target_changed",
                steps_cleared=31,
            )
            logger.record(
                result(
                    "unknown",
                    None,
                    status="waiting",
                    reason="no_confirmed_target",
                    valid_steps=0,
                ),
                frame_index=13,
                captured_at_s=None,
                valid_steps_before=0,
                reset_requested=True,
                reset_reason="no_confirmed_target",
                steps_cleared=0,
            )
            logger.record(
                result("drone", 0.85),
                frame_index=61,
                captured_at_s=None,
                valid_steps_before=47,
                classification_latency_s=0.002,
            )
            logger.close()

            records = [
                json.loads(line)
                for line in path.read_text(encoding="utf-8").splitlines()
            ]

        self.assertEqual(records[0]["record_type"], "metadata")
        first_reset = records[1]["history"]
        repeated_empty_reset = records[2]["history"]
        ready_record = records[3]
        self.assertEqual(first_reset["steps_cleared"], 31)
        self.assertEqual(first_reset["history_generation"], 1)
        self.assertEqual(repeated_empty_reset["reset_sequence"], 2)
        self.assertEqual(repeated_empty_reset["history_generation"], 1)
        self.assertAlmostEqual(ready_record["predicted_confidence"], 0.85)
        self.assertEqual(ready_record["classification_latency_ms"], 2.0)
        self.assertEqual(records[-2]["scope"], "run")
        self.assertEqual(records[-1]["scope"], "aggregate")

    def test_second_compatible_run_aggregates_both_truth_classes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            directory = Path(temporary_directory)
            drone_logger = self._logger(
                directory / "live_inference_drone.jsonl",
                "drone",
            )
            drone_logger.record(
                result("drone", 0.9),
                frame_index=48,
                captured_at_s=None,
                valid_steps_before=47,
            )
            drone_logger.close()

            non_drone_logger = self._logger(
                directory / "live_inference_non_drone.jsonl",
                "not_drone",
            )
            non_drone_logger.record(
                result("not_drone", 0.1),
                frame_index=48,
                captured_at_s=None,
                valid_steps_before=47,
            )
            (directory / "interrupted.jsonl").write_text(
                '{"record_type":"metadata"}\n{"record_type":',
                encoding="utf-8",
            )
            _, aggregate = non_drone_logger.close()

        self.assertEqual(aggregate["included_runs"], 2)
        self.assertEqual(aggregate["metrics"]["drone"]["class_accuracy"], 1.0)
        self.assertEqual(
            aggregate["metrics"]["not_drone"]["class_accuracy"], 1.0
        )
        self.assertEqual(aggregate["metrics"]["roc_auc"], 1.0)
        self.assertEqual(
            aggregate["excluded_logs"]["malformed_or_incomplete"],
            1,
        )


if __name__ == "__main__":
    unittest.main()
