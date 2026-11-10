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

from .inference import InferenceResult, WINDOW_STEPS
from .log_paths import (
    DEFAULT_LOG_DIRECTORY,
    default_run_log_path,
    new_run_log_id,
)


FORMAT_NAME = "radar-live-inference-jsonl"
FORMAT_VERSION = 1
GROUND_TRUTH_LABELS = ("drone", "not_drone", "unlabeled")
ARTIFACT_NAMES = (
    "manifest.json",
    "calibration.joblib",
    "model_state.pt",
    "model.onnx",
)


def default_inference_log_path(
    now: Optional[datetime] = None,
    *,
    run_id: Optional[str] = None,
) -> Path:
    return default_run_log_path(
        "live_inference",
        ".jsonl",
        run_id=run_id or new_run_log_id(now),
    )


def _iso_now() -> str:
    return datetime.now().astimezone().isoformat(timespec="milliseconds")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def artifact_hashes(artifact_dir: Path) -> dict[str, str]:
    artifact_dir = Path(artifact_dir)
    return {
        name: _sha256(artifact_dir / name)
        for name in ARTIFACT_NAMES
        if (artifact_dir / name).is_file()
    }


def compatibility_key(
    classifier_metadata: Mapping[str, Any],
    artifacts: Mapping[str, str],
    profile_sha256: Optional[str] = None,
) -> str:
    identity = {
        "format_version": FORMAT_VERSION,
        "model": classifier_metadata.get("model"),
        "backend": classifier_metadata.get("backend"),
        "precision": classifier_metadata.get("precision", "float32"),
        "feature_version": classifier_metadata.get("feature_version"),
        "compatible_profile_sha256": classifier_metadata.get(
            "compatible_profile_sha256"
        ),
        "profile_sha256": profile_sha256,
        "threshold": classifier_metadata.get("threshold"),
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
    true_positives = 0
    false_positives = 0
    previous_recall = 0.0
    area = 0.0
    index = 0
    while index < len(ordered):
        end = index + 1
        while end < len(ordered) and ordered[end][0] == ordered[index][0]:
            end += 1
        group_labels = [label for _, label in ordered[index:end]]
        true_positives += sum(group_labels)
        false_positives += len(group_labels) - sum(group_labels)
        recall = true_positives / positives
        precision = true_positives / (true_positives + false_positives)
        area += (recall - previous_recall) * precision
        previous_recall = recall
        index = end
    return area


def calculate_metrics(
    observations: list[dict[str, Any]],
    *,
    frame_period_s: Optional[float],
    run_duration_s: Optional[float] = None,
) -> dict[str, Any]:
    attempts = len(observations)
    ready = [
        item
        for item in observations
        if item.get("result", {}).get("status") == "ready"
        and item.get("result", {}).get("label") in {"drone", "not_drone"}
    ]
    unknown = attempts - len(ready)
    labeled_ready = [
        item
        for item in ready
        if item.get("ground_truth") in {"drone", "not_drone"}
    ]

    tp = tn = fp = fn = 0
    probabilities: list[float] = []
    labels: list[int] = []
    correct = 0
    for item in labeled_ready:
        truth = item["ground_truth"]
        predicted = item["result"]["label"]
        if truth == "drone" and predicted == "drone":
            tp += 1
        elif truth == "drone":
            fn += 1
        elif predicted == "drone":
            fp += 1
        else:
            tn += 1
        correct += int(truth == predicted)
        probability = item["result"].get("p_drone")
        if probability is not None and math.isfinite(float(probability)):
            probabilities.append(float(probability))
            labels.append(1 if truth == "drone" else 0)

    drone_precision = _ratio(tp, tp + fp)
    drone_recall = _ratio(tp, tp + fn)
    non_drone_precision = _ratio(tn, tn + fn)
    non_drone_recall = _ratio(tn, tn + fp)

    def f1(precision: Optional[float], recall: Optional[float]) -> Optional[float]:
        if precision is None or recall is None or precision + recall == 0.0:
            return None
        return 2.0 * precision * recall / (precision + recall)

    brier = None
    log_loss = None
    if probabilities:
        brier = sum(
            (score - label) ** 2
            for score, label in zip(probabilities, labels)
        ) / len(probabilities)
        epsilon = 1e-15
        log_loss = -sum(
            label * math.log(min(max(score, epsilon), 1.0 - epsilon))
            + (1 - label) * math.log(min(max(1.0 - score, epsilon), 1.0 - epsilon))
            for score, label in zip(probabilities, labels)
        ) / len(probabilities)

    duration_counts = Counter(
        item.get("result", {}).get("label", "unknown") for item in observations
    )
    durations = {
        label: (
            duration_counts.get(label, 0) * frame_period_s
            if frame_period_s is not None and frame_period_s > 0.0
            else None
        )
        for label in ("drone", "not_drone", "unknown")
    }

    reason_counts = Counter(
        item.get("result", {}).get("reason") or "ready" for item in observations
    )
    confidences = [
        float(item["predicted_confidence"])
        for item in ready
        if item.get("predicted_confidence") is not None
    ]
    latencies = [
        float(item["classification_latency_ms"])
        for item in observations
        if item.get("classification_latency_ms") is not None
    ]

    transitions = 0
    segments: list[dict[str, Any]] = []
    active: Optional[dict[str, Any]] = None
    for item in observations:
        result = item.get("result", {})
        label = result.get("label") if result.get("status") == "ready" else None
        frame_index = item.get("frame_index")
        contiguous = bool(
            active is not None
            and isinstance(frame_index, int)
            and frame_index == active["last_frame_index"] + 1
            and item.get("history", {}).get("action") != "reset"
        )
        if label is None:
            if active is not None:
                segments.append(active)
                active = None
            continue
        if active is None or not contiguous or active["label"] != label:
            if active is not None:
                if contiguous and active["label"] != label:
                    transitions += 1
                segments.append(active)
            active = {
                "label": label,
                "frames": 1,
                "first_frame_index": frame_index,
                "last_frame_index": frame_index,
            }
        else:
            active["frames"] += 1
            active["last_frame_index"] = frame_index
    if active is not None:
        segments.append(active)
    segment_summary: dict[str, Any] = {}
    for label in ("drone", "not_drone"):
        label_segments = [segment for segment in segments if segment["label"] == label]
        segment_summary[label] = {
            "count": len(label_segments),
            "longest_frames": max((segment["frames"] for segment in label_segments), default=0),
            "longest_duration_s": (
                max((segment["frames"] for segment in label_segments), default=0) * frame_period_s
                if frame_period_s is not None and frame_period_s > 0.0
                else None
            ),
        }

    resets = [item for item in observations if item.get("history", {}).get("reset_requested")]
    effective_resets = [
        item for item in resets if int(item.get("history", {}).get("steps_cleared", 0)) > 0
    ]
    reset_reasons = Counter(
        item.get("history", {}).get("reset_reason") or "unspecified" for item in resets
    )
    recovery_times: list[float] = []
    unrecovered = 0
    effective_reset_indices = [
        index
        for index, item in enumerate(observations)
        if item.get("history", {}).get("reset_requested")
        and int(item.get("history", {}).get("steps_cleared", 0)) > 0
    ]
    for reset_index in effective_reset_indices:
        reset = observations[reset_index]
        reset_elapsed = reset.get("capture_elapsed_s")
        recovered = next(
            (
                item
                for item in observations[reset_index + 1 :]
                if item.get("result", {}).get("status") == "ready"
            ),
            None,
        )
        if recovered is None:
            unrecovered += 1
        elif reset_elapsed is not None and recovered.get("capture_elapsed_s") is not None:
            recovery_times.append(float(recovered["capture_elapsed_s"]) - float(reset_elapsed))

    prediction_counts = Counter(item["result"]["label"] for item in ready)
    majority = "unknown"
    if prediction_counts["drone"] > prediction_counts["not_drone"]:
        majority = "drone"
    elif prediction_counts["not_drone"] > prediction_counts["drone"]:
        majority = "not_drone"

    first_ready_elapsed = next(
        (
            item.get("capture_elapsed_s")
            for item in observations
            if item.get("result", {}).get("status") == "ready"
            and item.get("result", {}).get("label")
            in {"drone", "not_drone"}
        ),
        None,
    )
    labeled_attempts = sum(
        item.get("ground_truth") in {"drone", "not_drone"} for item in observations
    )
    unavailable: dict[str, str] = {}
    if drone_recall is None:
        unavailable["drone_class_accuracy"] = "no labeled drone ready decisions"
    if non_drone_recall is None:
        unavailable["not_drone_class_accuracy"] = "no labeled non-drone ready decisions"
    if _roc_auc(labels, probabilities) is None:
        unavailable["roc_auc"] = "both labeled truth classes are required"
        unavailable["pr_auc"] = "both labeled truth classes are required"

    return {
        "attempts": attempts,
        "ready_decisions": len(ready),
        "unknown_decisions": unknown,
        "readiness_coverage": _ratio(len(ready), attempts),
        "outcome_counts": dict(sorted(reason_counts.items())),
        "predicted_counts": {
            label: duration_counts.get(label, 0)
            for label in ("drone", "not_drone", "unknown")
        },
        "predicted_duration_s": durations,
        "confusion_matrix": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
        "overall_accuracy": _ratio(correct, len(labeled_ready)),
        "operational_correctness": _ratio(correct, labeled_attempts),
        "drone": {
            "support": tp + fn,
            "class_accuracy": drone_recall,
            "precision": drone_precision,
            "recall": drone_recall,
            "f1": f1(drone_precision, drone_recall),
        },
        "not_drone": {
            "support": tn + fp,
            "class_accuracy": non_drone_recall,
            "precision": non_drone_precision,
            "recall": non_drone_recall,
            "f1": f1(non_drone_precision, non_drone_recall),
        },
        "balanced_accuracy": (
            (drone_recall + non_drone_recall) / 2.0
            if drone_recall is not None and non_drone_recall is not None
            else None
        ),
        "brier_score": brier,
        "log_loss": log_loss,
        "roc_auc": _roc_auc(labels, probabilities),
        "pr_auc": _pr_auc(labels, probabilities),
        "confidence": _distribution(confidences),
        "classification_latency_ms": _distribution(latencies),
        "label_transitions": transitions,
        "segments": segment_summary,
        "time_to_first_ready_s": first_ready_elapsed,
        "session_majority_prediction": majority,
        "history_resets": {
            "count": len(resets),
            "effective_count": len(effective_resets),
            "reasons": dict(sorted(reset_reasons.items())),
            "steps_cleared": sum(int(item["history"].get("steps_cleared", 0)) for item in resets),
            "per_hour": (
                len(resets) * 3600.0 / run_duration_s
                if run_duration_s is not None and run_duration_s > 0.0
                else None
            ),
            "recovery_time_s": _distribution(recovery_times),
            "unrecovered_effective_resets": unrecovered,
        },
        "unavailable_reasons": unavailable,
    }


def _read_completed_log(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
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
    run_majority_correct = 0
    labeled_run_majorities = 0
    for path in sorted(Path(directory).glob("*.jsonl")):
        try:
            metadata, records, summary = _read_completed_log(path)
        except (OSError, ValueError, json.JSONDecodeError):
            exclusions["malformed_or_incomplete"] += 1
            continue
        if metadata.get("format") != FORMAT_NAME:
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
            labeled_run_majorities += 1
            run_majority_correct += int(majority == truth)
    frame_period_s = next(iter(frame_periods)) if len(frame_periods) == 1 else None
    metrics = calculate_metrics(observations, frame_period_s=frame_period_s)
    metrics["session_majority_accuracy"] = _ratio(
        run_majority_correct,
        labeled_run_majorities,
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
        artifact_dir: Path,
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
        self.artifacts = artifact_hashes(artifact_dir)
        self.profile_sha256 = (
            _sha256(profile_path) if Path(profile_path).is_file() else None
        )
        self.compatibility_key = compatibility_key(
            self.classifier_metadata,
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
        metadata = {
            "record_type": "metadata",
            "format": FORMAT_NAME,
            "version": FORMAT_VERSION,
            "run_id": str(uuid.uuid4()),
            "created_at": _iso_now(),
            "ground_truth": ground_truth,
            "required_history_steps": WINDOW_STEPS,
            "radar": {"frame_period_s": self.frame_period_s},
            "classifier": self.classifier_metadata,
            "artifacts": self.artifacts,
            "profile_path": str(Path(profile_path)),
            "profile_sha256": self.profile_sha256,
            "compatibility_key": self.compatibility_key,
        }
        self._write(metadata)

    def _write(self, record: Mapping[str, Any]) -> None:
        self.file.write(json.dumps(record, separators=(",", ":"), allow_nan=False) + "\n")
        self.file.flush()

    def record(
        self,
        result: InferenceResult,
        *,
        frame_index: int,
        captured_at_s: Optional[float],
        valid_steps_before: int,
        reset_requested: bool = False,
        reset_reason: Optional[str] = None,
        steps_cleared: Optional[int] = None,
        classification_latency_s: Optional[float] = None,
        target_range_m: Optional[float] = None,
        target_source: Optional[str] = None,
    ) -> None:
        if self.closed:
            return
        valid_after = int(result.valid_steps)
        cleared = (
            max(int(steps_cleared), 0)
            if steps_cleared is not None
            else max(int(valid_steps_before) - valid_after, 0)
        )
        if reset_requested:
            self.reset_sequence += 1
            if cleared > 0 or reset_reason == "target_changed":
                self.history_generation += 1
            self.frames_since_reset = 0
        else:
            self.frames_since_reset += 1
        state_after = (
            "ready"
            if result.status == "ready"
            else "warming_up"
            if valid_after > 0
            else "empty"
        )
        probability = result.p_drone
        predicted_confidence = None
        if result.status == "ready" and probability is not None:
            predicted_confidence = (
                float(probability)
                if result.label == "drone"
                else 1.0 - float(probability)
            )
        correct = (
            result.label == self.ground_truth
            if self.ground_truth in {"drone", "not_drone"}
            and result.status == "ready"
            else None
        )
        now_monotonic_s = time.perf_counter()
        captured_at = None
        capture_elapsed_s = None
        if captured_at_s is not None and math.isfinite(float(captured_at_s)):
            captured_at_value = float(captured_at_s)
            captured_at = datetime.fromtimestamp(
                captured_at_value + self.wall_clock_offset_s
            ).astimezone().isoformat(timespec="milliseconds")
            capture_elapsed_s = captured_at_value - self.started_monotonic_s
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
            "predicted_confidence": predicted_confidence,
            "correct": correct,
            "classification_latency_ms": (
                float(classification_latency_s) * 1000.0
                if classification_latency_s is not None
                else None
            ),
            "target_range_m": (
                float(target_range_m) if target_range_m is not None else None
            ),
            "target_source": target_source,
            "history": {
                "action": "reset" if reset_requested else "append",
                "state_after": state_after,
                "valid_steps_before": int(valid_steps_before),
                "valid_steps_after": valid_after,
                "required_steps": WINDOW_STEPS,
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

    def close(self, termination_status: str = "completed") -> tuple[dict[str, Any], dict[str, Any]]:
        if self.closed:
            return {}, {}
        run_duration_s = time.perf_counter() - self.started_monotonic_s
        metrics = calculate_metrics(
            self.observations,
            frame_period_s=self.frame_period_s,
            run_duration_s=run_duration_s,
        )
        run_summary = {
            "record_type": "summary",
            "scope": "run",
            "created_at": _iso_now(),
            "termination_status": termination_status,
            "run_duration_s": run_duration_s,
            "metrics": metrics,
        }
        self._write(run_summary)
        aggregate = aggregate_compatible_logs(
            self.output_path.parent,
            self.compatibility_key,
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
