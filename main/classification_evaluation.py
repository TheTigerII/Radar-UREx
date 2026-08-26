from __future__ import annotations

import hashlib
import json
import math
import time
import uuid
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Optional, TextIO

from .inference import ClassificationResult, SEGMENT_FRAMES


FORMAT_NAME = "radar-live-inference-jsonl"
FORMAT_VERSION = 2
GROUND_TRUTH_LABELS = ("drone", "not_drone", "unlabeled")
NATIVE_TO_EVALUATION_LABEL = {
    "uav": "drone",
    "other": "not_drone",
    "unknown": "unknown",
}
DEFAULT_LOG_DIRECTORY = Path(__file__).resolve().parent.parent / "log"


def default_inference_log_path(now: Optional[datetime] = None) -> Path:
    timestamp = (now or datetime.now().astimezone()).strftime("%Y%m%d_%H%M%S_%f")
    return DEFAULT_LOG_DIRECTORY / f"live_inference_{timestamp}.jsonl"


def _iso_now() -> str:
    return datetime.now().astimezone().isoformat(timespec="milliseconds")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def artifact_hashes(classifier_metadata: Mapping[str, Any]) -> dict[str, str]:
    paths: dict[str, Path] = {}
    for field in (
        "model_path",
        "onnx_model_path",
        "manifest_path",
        "external_data_path",
    ):
        value = classifier_metadata.get(field)
        if not isinstance(value, str) or not value:
            continue
        path = Path(value)
        if path.is_file():
            paths[field] = path
    return {field: _sha256(path) for field, path in sorted(paths.items())}


def compatibility_key(
    classifier_metadata: Mapping[str, Any],
    pmm_metadata: Mapping[str, Any],
    artifacts: Mapping[str, str],
    profile_sha256: Optional[str],
) -> str:
    identity = {
        "format_version": FORMAT_VERSION,
        "runtime": classifier_metadata.get("runtime"),
        "device": classifier_metadata.get("device"),
        "input_shape": classifier_metadata.get("input_shape"),
        "label_to_index": classifier_metadata.get("label_to_index"),
        "profile_fingerprint": classifier_metadata.get("profile_fingerprint"),
        "feature_fingerprint": classifier_metadata.get("feature_fingerprint"),
        "feature_version": classifier_metadata.get("feature_version"),
        "pmm_config": pmm_metadata.get("config"),
        "profile_sha256": profile_sha256,
        "artifacts": dict(sorted(artifacts.items())),
    }
    payload = json.dumps(identity, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _ratio(numerator: int | float, denominator: int | float) -> Optional[float]:
    return float(numerator / denominator) if denominator else None


def _percentile(values: Iterable[float], percentile: float) -> Optional[float]:
    ordered = sorted(float(value) for value in values if math.isfinite(float(value)))
    if not ordered:
        return None
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * percentile
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _distribution(values: Iterable[float]) -> dict[str, Optional[float] | int]:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    return {
        "count": len(finite),
        "min": min(finite) if finite else None,
        "mean": sum(finite) / len(finite) if finite else None,
        "p50": _percentile(finite, 0.50),
        "p95": _percentile(finite, 0.95),
        "max": max(finite) if finite else None,
    }


def _roc_auc(labels: list[int], scores: list[float]) -> Optional[float]:
    positives = sum(labels)
    negatives = len(labels) - positives
    if not positives or not negatives:
        return None
    ordered = sorted(zip(scores, labels), key=lambda item: item[0])
    rank_sum = 0.0
    index = 0
    while index < len(ordered):
        end = index + 1
        while end < len(ordered) and ordered[end][0] == ordered[index][0]:
            end += 1
        average_rank = ((index + 1) + end) / 2.0
        rank_sum += average_rank * sum(label for _, label in ordered[index:end])
        index = end
    return (rank_sum - positives * (positives + 1) / 2.0) / (
        positives * negatives
    )


def _pr_auc(labels: list[int], scores: list[float]) -> Optional[float]:
    positives = sum(labels)
    if not positives or positives == len(labels):
        return None
    ordered = sorted(zip(scores, labels), key=lambda item: item[0], reverse=True)
    true_positives = false_positives = 0
    previous_recall = area = 0.0
    index = 0
    while index < len(ordered):
        end = index + 1
        while end < len(ordered) and ordered[end][0] == ordered[index][0]:
            end += 1
        group = [label for _, label in ordered[index:end]]
        true_positives += sum(group)
        false_positives += len(group) - sum(group)
        recall = true_positives / positives
        precision = true_positives / (true_positives + false_positives)
        area += (recall - previous_recall) * precision
        previous_recall = recall
        index = end
    return area


def _predicted_label(observation: Mapping[str, Any]) -> str:
    native = observation.get("result", {}).get("label", "unknown")
    return NATIVE_TO_EVALUATION_LABEL.get(str(native), "unknown")


def calculate_metrics(
    observations: list[dict[str, Any]],
    *,
    frame_period_s: Optional[float],
    run_duration_s: Optional[float] = None,
) -> dict[str, Any]:
    attempts = len(observations)
    classified = [
        item
        for item in observations
        if item.get("result", {}).get("status") == "classified"
        and _predicted_label(item) in {"drone", "not_drone"}
    ]
    labeled = [
        item
        for item in classified
        if item.get("ground_truth") in {"drone", "not_drone"}
    ]
    tp = tn = fp = fn = correct = 0
    probabilities: list[float] = []
    labels: list[int] = []
    for item in labeled:
        truth = item["ground_truth"]
        predicted = _predicted_label(item)
        if truth == "drone" and predicted == "drone":
            tp += 1
        elif truth == "drone":
            fn += 1
        elif predicted == "drone":
            fp += 1
        else:
            tn += 1
        correct += int(truth == predicted)
        probability = item.get("result", {}).get("probabilities", {}).get("uav")
        if probability is not None and math.isfinite(float(probability)):
            probabilities.append(float(probability))
            labels.append(1 if truth == "drone" else 0)

    drone_precision = _ratio(tp, tp + fp)
    drone_recall = _ratio(tp, tp + fn)
    other_precision = _ratio(tn, tn + fn)
    other_recall = _ratio(tn, tn + fp)

    def f1(precision: Optional[float], recall: Optional[float]) -> Optional[float]:
        if precision is None or recall is None or precision + recall == 0.0:
            return None
        return 2.0 * precision * recall / (precision + recall)

    brier_score = None
    log_loss = None
    if probabilities:
        brier_score = sum(
            (score - label) ** 2 for score, label in zip(probabilities, labels)
        ) / len(probabilities)
        epsilon = 1e-15
        log_loss = -sum(
            label * math.log(min(max(score, epsilon), 1.0 - epsilon))
            + (1 - label)
            * math.log(min(max(1.0 - score, epsilon), 1.0 - epsilon))
            for score, label in zip(probabilities, labels)
        ) / len(probabilities)

    predicted_counts = Counter(_predicted_label(item) for item in observations)
    predicted_duration_s = {
        label: (
            predicted_counts.get(label, 0) * frame_period_s
            if frame_period_s is not None and frame_period_s > 0.0
            else None
        )
        for label in ("drone", "not_drone", "unknown")
    }
    unavailable_reasons = Counter(
        item.get("result", {}).get("status", "unknown")
        for item in observations
        if item not in classified
    )
    confidences = [
        float(item["predicted_confidence"])
        for item in classified
        if item.get("predicted_confidence") is not None
    ]
    latencies = [
        float(item["classification_latency_ms"])
        for item in observations
        if item.get("classification_latency_ms") is not None
    ]

    transitions = 0
    previous_label: Optional[str] = None
    segments: list[dict[str, Any]] = []
    for item in observations:
        label = _predicted_label(item)
        if previous_label is not None and label != previous_label:
            transitions += 1
        previous_label = label
        if not segments or segments[-1]["label"] != label:
            segments.append({"label": label, "attempts": 1})
        else:
            segments[-1]["attempts"] += 1
    segment_summary = {
        label: _distribution(
            segment["attempts"] * frame_period_s
            for segment in segments
            if segment["label"] == label and frame_period_s is not None
        )
        for label in ("drone", "not_drone", "unknown")
    }
    first_ready_elapsed = next(
        (
            item.get("capture_elapsed_s")
            for item in classified
            if item.get("capture_elapsed_s") is not None
        ),
        None,
    )
    classified_counts = Counter(_predicted_label(item) for item in classified)
    majority = None
    if classified_counts:
        top = classified_counts.most_common()
        if len(top) == 1 or top[0][1] != top[1][1]:
            majority = top[0][0]

    resets = [
        item for item in observations if item.get("history", {}).get("reset_requested")
    ]
    effective_resets = [
        item for item in resets if int(item.get("history", {}).get("steps_cleared", 0)) > 0
    ]
    reset_reasons = Counter(
        item.get("history", {}).get("reset_reason") or "unspecified" for item in resets
    )
    recovery_times: list[float] = []
    unrecovered = 0
    for reset in effective_resets:
        reset_sequence = reset.get("history", {}).get("reset_sequence")
        reset_elapsed = reset.get("capture_elapsed_s")
        recovered = next(
            (
                item
                for item in observations
                if item.get("history", {}).get("reset_sequence") == reset_sequence
                and item.get("result", {}).get("status") == "classified"
            ),
            None,
        )
        if (
            recovered is not None
            and reset_elapsed is not None
            and recovered.get("capture_elapsed_s") is not None
        ):
            recovery_times.append(recovered["capture_elapsed_s"] - reset_elapsed)
        else:
            unrecovered += 1

    return {
        "attempts": attempts,
        "classified_attempts": len(classified),
        "unknown_attempts": attempts - len(classified),
        "readiness_coverage": _ratio(len(classified), attempts),
        "overall_accuracy": _ratio(correct, len(labeled)),
        "operational_correctness": _ratio(correct, attempts),
        "confusion_matrix": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
        "drone": {
            "class_accuracy": drone_recall,
            "precision": drone_precision,
            "recall": drone_recall,
            "f1": f1(drone_precision, drone_recall),
        },
        "not_drone": {
            "class_accuracy": other_recall,
            "precision": other_precision,
            "recall": other_recall,
            "f1": f1(other_precision, other_recall),
        },
        "balanced_accuracy": (
            (drone_recall + other_recall) / 2.0
            if drone_recall is not None and other_recall is not None
            else None
        ),
        "brier_score": brier_score,
        "log_loss": log_loss,
        "roc_auc": _roc_auc(labels, probabilities),
        "pr_auc": _pr_auc(labels, probabilities),
        "predicted_duration_s": predicted_duration_s,
        "confidence": _distribution(confidences),
        "classification_latency_ms": _distribution(latencies),
        "label_transitions": transitions,
        "segments": segment_summary,
        "time_to_first_classified_s": first_ready_elapsed,
        "session_majority_prediction": majority,
        "history_resets": {
            "count": len(resets),
            "effective_count": len(effective_resets),
            "reasons": dict(sorted(reset_reasons.items())),
            "steps_cleared": sum(
                int(item.get("history", {}).get("steps_cleared", 0))
                for item in resets
            ),
            "per_hour": (
                len(resets) * 3600.0 / run_duration_s
                if run_duration_s is not None and run_duration_s > 0.0
                else None
            ),
            "recovery_time_s": _distribution(recovery_times),
            "unrecovered_effective_resets": unrecovered,
        },
        "unavailable_reasons": dict(sorted(unavailable_reasons.items())),
    }


def _read_completed_log(
    path: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    metadata: Optional[dict[str, Any]] = None
    observations: list[dict[str, Any]] = []
    summary: Optional[dict[str, Any]] = None
    with path.open("r", encoding="utf-8") as source:
        for line in source:
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get("record_type") == "metadata":
                metadata = record
            elif record.get("record_type") == "inference":
                observations.append(record)
            elif record.get("record_type") == "summary" and record.get("scope") == "run":
                summary = record
    if metadata is None or summary is None:
        raise ValueError("log is incomplete")
    return metadata, observations, summary


def aggregate_compatible_logs(
    directory: Path,
    expected_compatibility_key: str,
) -> dict[str, Any]:
    observations: list[dict[str, Any]] = []
    included_runs = 0
    frame_periods: set[float] = set()
    exclusions: Counter[str] = Counter()
    majority_correct = labeled_majorities = 0
    for path in sorted(Path(directory).glob("*.jsonl")):
        try:
            metadata, records, summary = _read_completed_log(path)
        except (OSError, ValueError, json.JSONDecodeError, AttributeError):
            exclusions["malformed_or_incomplete"] += 1
            continue
        if metadata.get("format") != FORMAT_NAME or metadata.get("version") != FORMAT_VERSION:
            exclusions["different_format"] += 1
            continue
        if metadata.get("compatibility_key") != expected_compatibility_key:
            exclusions["incompatible"] += 1
            continue
        if summary.get("termination_status") != "completed":
            exclusions["failed"] += 1
            continue
        truth = metadata.get("ground_truth")
        if truth not in {"drone", "not_drone"}:
            exclusions["unlabeled"] += 1
            continue
        included_runs += 1
        period = metadata.get("radar", {}).get("frame_period_s")
        if isinstance(period, (int, float)) and period > 0.0:
            frame_periods.add(float(period))
        for record in records:
            record = dict(record)
            record["ground_truth"] = truth
            observations.append(record)
        majority = summary.get("metrics", {}).get("session_majority_prediction")
        if majority in {"drone", "not_drone"}:
            labeled_majorities += 1
            majority_correct += int(majority == truth)
    frame_period_s = next(iter(frame_periods)) if len(frame_periods) == 1 else None
    metrics = calculate_metrics(observations, frame_period_s=frame_period_s)
    metrics["session_majority_accuracy"] = _ratio(
        majority_correct, labeled_majorities
    )
    return {
        "included_runs": included_runs,
        "excluded_logs": dict(sorted(exclusions.items())),
        "metrics": metrics,
    }


class ClassificationEvaluationLogger:
    def __init__(
        self,
        output_path: Path,
        *,
        ground_truth: str,
        frame_period_s: Optional[float],
        classifier_metadata: Mapping[str, Any],
        pmm_metadata: Mapping[str, Any],
        profile_path: Path,
        emit_func: Optional[Callable[[str], None]] = None,
    ) -> None:
        if ground_truth not in GROUND_TRUTH_LABELS:
            raise ValueError(f"Unsupported evaluation label: {ground_truth}")
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.file: TextIO = self.output_path.open("w", encoding="utf-8", buffering=1)
        self.ground_truth = ground_truth
        self.frame_period_s = (
            float(frame_period_s)
            if frame_period_s is not None and float(frame_period_s) > 0.0
            else None
        )
        self.classifier_metadata = dict(classifier_metadata)
        self.pmm_metadata = dict(pmm_metadata)
        self.artifacts = artifact_hashes(self.classifier_metadata)
        self.profile_sha256 = _sha256(profile_path) if Path(profile_path).is_file() else None
        self.compatibility_key = compatibility_key(
            self.classifier_metadata,
            self.pmm_metadata,
            self.artifacts,
            self.profile_sha256,
        )
        self.emit = emit_func or (lambda _message: None)
        self.started_monotonic_s = time.perf_counter()
        self.wall_clock_offset_s = time.time() - self.started_monotonic_s
        self.observations: list[dict[str, Any]] = []
        self.reset_sequence = 0
        self.history_generation = 0
        self.frames_since_reset = 0
        self.closed = False
        self._write(
            {
                "record_type": "metadata",
                "format": FORMAT_NAME,
                "version": FORMAT_VERSION,
                "run_id": str(uuid.uuid4()),
                "created_at": _iso_now(),
                "ground_truth": ground_truth,
                "required_history_frames": SEGMENT_FRAMES,
                "label_mapping": dict(NATIVE_TO_EVALUATION_LABEL),
                "radar": {"frame_period_s": self.frame_period_s},
                "classifier": self.classifier_metadata,
                "pmm_tracking": self.pmm_metadata,
                "artifacts": self.artifacts,
                "profile_path": str(Path(profile_path)),
                "profile_sha256": self.profile_sha256,
                "compatibility_key": self.compatibility_key,
            }
        )

    def _write(self, record: Mapping[str, Any]) -> None:
        self.file.write(json.dumps(record, separators=(",", ":"), allow_nan=False) + "\n")
        self.file.flush()

    def record(
        self,
        result: ClassificationResult,
        *,
        frame_index: int,
        captured_at_s: Optional[float],
        history_frames_before: int,
        reset_requested: bool = False,
        reset_reason: Optional[str] = None,
        steps_cleared: Optional[int] = None,
        classification_latency_ms: Optional[float] = None,
        target_range_m: Optional[float] = None,
        target_state: Optional[str] = None,
    ) -> None:
        if self.closed:
            return
        history_after = max(int(result.history_frames), 0)
        cleared = (
            max(int(steps_cleared), 0)
            if steps_cleared is not None
            else (max(int(history_frames_before), 0) if reset_requested else 0)
        )
        if reset_requested:
            self.reset_sequence += 1
            if cleared > 0:
                self.history_generation += 1
            self.frames_since_reset = 0
        else:
            self.frames_since_reset += 1
        probability = (
            result.probabilities.get("uav")
            if result.probabilities is not None
            else None
        )
        predicted = NATIVE_TO_EVALUATION_LABEL.get(result.label, "unknown")
        correct = (
            predicted == self.ground_truth
            if self.ground_truth in {"drone", "not_drone"}
            and result.status == "classified"
            else None
        )
        now_monotonic_s = time.perf_counter()
        captured_at = None
        capture_elapsed_s = None
        if captured_at_s is not None and math.isfinite(float(captured_at_s)):
            captured_value = float(captured_at_s)
            captured_at = datetime.fromtimestamp(
                captured_value + self.wall_clock_offset_s
            ).astimezone().isoformat(timespec="milliseconds")
            capture_elapsed_s = captured_value - self.started_monotonic_s
        record = {
            "record_type": "inference",
            "sequence": len(self.observations) + 1,
            "frame_index": int(frame_index),
            "captured_at": captured_at,
            "processed_at": _iso_now(),
            "capture_elapsed_s": capture_elapsed_s,
            "processing_elapsed_s": now_monotonic_s - self.started_monotonic_s,
            "ground_truth": self.ground_truth,
            "result": result.to_dict(),
            "predicted_confidence": (
                float(result.confidence)
                if result.status == "classified" and result.confidence is not None
                else None
            ),
            "correct": correct,
            "classification_latency_ms": (
                float(classification_latency_ms)
                if classification_latency_ms is not None
                else None
            ),
            "target_range_m": (
                float(target_range_m) if target_range_m is not None else None
            ),
            "target_state": target_state,
            "history": {
                "action": "reset" if reset_requested else "append",
                "state_after": (
                    "ready"
                    if result.status == "classified"
                    else "warming_up"
                    if history_after > 0
                    else "empty"
                ),
                "frames_before": max(int(history_frames_before), 0),
                "frames_after": history_after,
                "required_frames": SEGMENT_FRAMES,
                "reset_requested": bool(reset_requested),
                "reset_reason": reset_reason if reset_requested else None,
                "steps_cleared": cleared,
                "reset_sequence": self.reset_sequence,
                "history_generation": self.history_generation,
                "frames_since_reset": self.frames_since_reset,
            },
        }
        self.observations.append(record)
        self._write(record)

    def close(
        self, termination_status: str = "completed"
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        if self.closed:
            return {}, {}
        run_duration_s = time.perf_counter() - self.started_monotonic_s
        run_summary = {
            "record_type": "summary",
            "scope": "run",
            "created_at": _iso_now(),
            "termination_status": termination_status,
            "run_duration_s": run_duration_s,
            "metrics": calculate_metrics(
                self.observations,
                frame_period_s=self.frame_period_s,
                run_duration_s=run_duration_s,
            ),
        }
        self._write(run_summary)
        aggregate = aggregate_compatible_logs(
            self.output_path.parent, self.compatibility_key
        )
        aggregate_summary = {
            "record_type": "summary",
            "scope": "aggregate",
            "created_at": _iso_now(),
            **aggregate,
        }
        self._write(aggregate_summary)
        self.file.close()
        self.closed = True
        return run_summary, aggregate_summary
