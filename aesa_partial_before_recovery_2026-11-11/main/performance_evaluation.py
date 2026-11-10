"""Low-overhead runtime performance logging for live radar capture."""

from __future__ import annotations

import json
import math
import os
import time
import uuid
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, TextIO


FORMAT_NAME = "radar-runtime-performance-jsonl"
FORMAT_VERSION = 1
DEFAULT_LOG_DIRECTORY = Path(__file__).resolve().parent.parent / "log"


def default_performance_log_path(now: Optional[datetime] = None) -> Path:
    timestamp = (now or datetime.now().astimezone()).strftime("%Y%m%d_%H%M%S")
    return DEFAULT_LOG_DIRECTORY / f"{timestamp}_runtime_performance.jsonl"


def _iso_now() -> str:
    return datetime.now().astimezone().isoformat(timespec="milliseconds")


def _finite(value: Any) -> Optional[float]:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _percentile(values: list[float], percentile: float) -> Optional[float]:
    ordered = sorted(value for value in values if math.isfinite(value))
    if not ordered:
        return None
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * percentile
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _distribution(values: list[float]) -> dict[str, Optional[float] | int]:
    finite = [value for value in values if math.isfinite(value)]
    return {
        "count": len(finite),
        "min": min(finite) if finite else None,
        "mean": sum(finite) / len(finite) if finite else None,
        "p50": _percentile(finite, 0.50),
        "p95": _percentile(finite, 0.95),
        "max": max(finite) if finite else None,
    }


def _ratio(numerator: int | float, denominator: int | float) -> Optional[float]:
    return float(numerator / denominator) if denominator else None


class SystemResourceSampler:
    """Sample Linux process/system load without adding third-party dependencies."""

    GPU_LOAD_PATHS = (
        Path("/sys/devices/platform/bus@0/17000000.gpu/load"),
        Path("/sys/devices/platform/17000000.gpu/load"),
        Path("/sys/devices/gpu.0/load"),
        Path("/sys/class/devfreq/17000000.gpu/load"),
        Path("/sys/class/devfreq/17000000.ga10b/load"),
        Path("/sys/class/devfreq/57000000.gpu/load"),
    )
    GPU_DISCOVERY_PATTERNS = (
        (Path("/sys/class/devfreq"), "*gpu*/load"),
        (Path("/sys/devices/platform"), "**/*.gpu/load"),
        (Path("/sys/devices/platform"), "**/gpu.0/load"),
    )

    def __init__(
        self,
        interval_s: float = 1.0,
        *,
        cuda_memory_provider: Optional[Callable[[], Mapping[str, Any]]] = None,
    ) -> None:
        interval_value = float(interval_s)
        if not math.isfinite(interval_value):
            raise ValueError("Resource sample interval must be finite")
        self.interval_s = max(interval_value, 0.1)
        self.cuda_memory_provider = cuda_memory_provider
        self.last_wall_s = time.perf_counter()
        self.last_sample_s: Optional[float] = self.last_wall_s
        self.last_process_cpu_s = time.process_time()
        self.last_cpu_totals = self._read_cpu_totals()
        self.gpu_load_path = self._find_gpu_load_path()

    @staticmethod
    def _read_cpu_totals() -> Optional[tuple[int, int]]:
        try:
            fields = Path("/proc/stat").read_text(encoding="utf-8").splitlines()[0]
            values = [int(value) for value in fields.split()[1:]]
        except (OSError, ValueError, IndexError):
            return None
        if len(values) < 4:
            return None
        idle = values[3] + (values[4] if len(values) > 4 else 0)
        return sum(values), idle

    @staticmethod
    def _read_memory() -> dict[str, Optional[float | int]]:
        values: dict[str, int] = {}
        try:
            for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
                name, raw = line.split(":", 1)
                token = raw.strip().split()[0]
                values[name] = int(token) * 1024
        except (OSError, ValueError, IndexError):
            values = {}
        total = values.get("MemTotal")
        available = values.get("MemAvailable")
        used_percent = (
            (total - available) * 100.0 / total
            if total and available is not None
            else None
        )
        rss_bytes: Optional[int] = None
        try:
            fields = Path("/proc/self/statm").read_text(encoding="utf-8").split()
            rss_bytes = int(fields[1]) * int(os.sysconf("SC_PAGE_SIZE"))
        except (OSError, ValueError, IndexError):
            pass
        return {
            "system_total_bytes": total,
            "system_available_bytes": available,
            "system_used_percent": used_percent,
            "process_rss_bytes": rss_bytes,
        }

    @classmethod
    def _gpu_load_candidates(cls) -> list[Path]:
        candidates = list(cls.GPU_LOAD_PATHS)
        for root, pattern in cls.GPU_DISCOVERY_PATTERNS:
            try:
                candidates.extend(root.glob(pattern))
            except OSError:
                continue
        return list(dict.fromkeys(candidates))

    @staticmethod
    def _read_gpu_load_path(path: Path) -> Optional[float]:
        try:
            raw = float(path.read_text(encoding="utf-8").strip())
        except (OSError, ValueError):
            return None
        if not math.isfinite(raw):
            return None
        # Jetson's devfreq ``load`` ABI reports permille (0..1000).
        return min(max(raw / 10.0, 0.0), 100.0)

    @classmethod
    def _find_gpu_load_path(cls) -> Optional[Path]:
        for path in cls._gpu_load_candidates():
            if cls._read_gpu_load_path(path) is not None:
                return path
        return None

    def _read_gpu_load(self) -> tuple[Optional[float], Optional[str]]:
        if self.gpu_load_path is not None:
            percent = self._read_gpu_load_path(self.gpu_load_path)
            if percent is not None:
                return percent, str(self.gpu_load_path)
        self.gpu_load_path = self._find_gpu_load_path()
        if self.gpu_load_path is not None:
            percent = self._read_gpu_load_path(self.gpu_load_path)
            if percent is not None:
                return percent, str(self.gpu_load_path)
        return None, None

    def sample(self, *, force: bool = False) -> Optional[dict[str, Any]]:
        now_s = time.perf_counter()
        if (
            not force
            and self.last_sample_s is not None
            and now_s - self.last_sample_s < self.interval_s
        ):
            return None
        wall_delta = now_s - self.last_wall_s
        process_cpu_s = time.process_time()
        process_delta = process_cpu_s - self.last_process_cpu_s
        process_cpu_percent = (
            process_delta * 100.0 / wall_delta if wall_delta > 0.0 else None
        )
        cpu_totals = self._read_cpu_totals()
        system_cpu_percent = None
        if cpu_totals is not None and self.last_cpu_totals is not None:
            total_delta = cpu_totals[0] - self.last_cpu_totals[0]
            idle_delta = cpu_totals[1] - self.last_cpu_totals[1]
            if total_delta > 0:
                system_cpu_percent = (total_delta - idle_delta) * 100.0 / total_delta
        gpu_percent, gpu_source = self._read_gpu_load()
        gpu_memory: dict[str, Any] = {
            "used_bytes": None,
            "free_bytes": None,
            "total_bytes": None,
        }
        if self.cuda_memory_provider is not None:
            try:
                provided = self.cuda_memory_provider()
                gpu_memory.update(
                    {
                        key: provided.get(key)
                        for key in gpu_memory
                        if provided.get(key) is not None
                    }
                )
            except Exception:
                # Performance logging must never interrupt radar processing.
                pass
        self.last_sample_s = now_s
        self.last_wall_s = now_s
        self.last_process_cpu_s = process_cpu_s
        self.last_cpu_totals = cpu_totals
        return {
            "sampled_at": _iso_now(),
            "monotonic_s": now_s,
            "sample_window_s": wall_delta,
            "cpu": {
                "system_percent": system_cpu_percent,
                "process_percent_one_core": process_cpu_percent,
                "process_cpu_cores": (
                    process_cpu_percent / 100.0
                    if process_cpu_percent is not None
                    else None
                ),
                "logical_cpu_count": os.cpu_count(),
            },
            "memory": self._read_memory(),
            "gpu": {
                "utilization_percent": gpu_percent,
                "utilization_source": gpu_source,
                "memory": gpu_memory,
            },
        }


class PerformanceEvaluationLogger:
    """Write frame-aligned runtime metrics and a compact run summary."""

    def __init__(
        self,
        output_path: Path,
        *,
        frame_period_s: Optional[float],
        radar_metadata: Mapping[str, Any],
        pmm_metadata: Mapping[str, Any],
        resource_sample_interval_s: float = 1.0,
        cuda_memory_provider: Optional[Callable[[], Mapping[str, Any]]] = None,
    ) -> None:
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.file: TextIO = self.output_path.open("w", encoding="utf-8", buffering=1)
        self.frame_period_s = (
            float(frame_period_s)
            if frame_period_s is not None and float(frame_period_s) > 0.0
            else None
        )
        self.started_s = time.perf_counter()
        self.sampler = SystemResourceSampler(
            resource_sample_interval_s,
            cuda_memory_provider=cuda_memory_provider,
        )
        self.frame_count = 0
        self.operational_frames = 0
        self.previous_state: Optional[str] = None
        self.state_counts: Counter[str] = Counter()
        self.detection_evaluated = 0
        self.detections = 0
        self.confirmed_frames = 0
        self.tracked_frames = 0
        self.predicted_frames = 0
        self.track_acquisitions = 0
        self.track_losses = 0
        self.deadline_misses = 0
        self.latencies: dict[str, list[float]] = {}
        self.resources: dict[str, list[float]] = {}
        self.resource_samples = 0
        self.closed = False
        self._write(
            {
                "record_type": "metadata",
                "format": FORMAT_NAME,
                "version": FORMAT_VERSION,
                "run_id": str(uuid.uuid4()),
                "created_at": _iso_now(),
                "frame_period_s": self.frame_period_s,
                "radar": dict(radar_metadata),
                "pmm_tracking": dict(pmm_metadata),
                "semantics": {
                    "detected": "PMM score is at or above its threshold",
                    "tracked": (
                        "state is tentative, confirmed, or coasting and has a range"
                    ),
                    "confirmed": "state is confirmed or coasting",
                    "accuracy_note": (
                        "Rates measure continuity and load. Detection accuracy and "
                        "position error require external ground-truth labels/reference "
                        "positions."
                    ),
                },
            }
        )

    def _write(self, record: Mapping[str, Any]) -> None:
        payload = json.dumps(record, separators=(",", ":"), allow_nan=False)
        self.file.write(payload + "\n")

    def _add_latency(self, name: str, value: Any) -> None:
        finite = _finite(value)
        if finite is not None:
            self.latencies.setdefault(name, []).append(finite)

    def _record_resource(self, snapshot: Mapping[str, Any]) -> None:
        paths = {
            "system_cpu_percent": snapshot.get("cpu", {}).get("system_percent"),
            "process_cpu_percent_one_core": snapshot.get("cpu", {}).get(
                "process_percent_one_core"
            ),
            "process_rss_bytes": snapshot.get("memory", {}).get("process_rss_bytes"),
            "system_memory_used_percent": snapshot.get("memory", {}).get(
                "system_used_percent"
            ),
            "gpu_utilization_percent": snapshot.get("gpu", {}).get(
                "utilization_percent"
            ),
            "gpu_memory_used_bytes": snapshot.get("gpu", {})
            .get("memory", {})
            .get("used_bytes"),
        }
        for name, value in paths.items():
            finite = _finite(value)
            if finite is not None:
                self.resources.setdefault(name, []).append(finite)

    def record_frame(
        self,
        *,
        frame_index: int,
        captured_at_s: Optional[float],
        completed_at_s: Optional[float],
        latency_ms: Mapping[str, Any],
        pmm_result: Any,
        capture_diagnostics: Mapping[str, Any],
        classification: Optional[Mapping[str, Any]] = None,
    ) -> None:
        if self.closed:
            return
        self.frame_count += 1
        state = str(pmm_result.state)
        score = _finite(pmm_result.pmm_score)
        threshold = _finite(pmm_result.threshold)
        detected = score is not None and threshold is not None and score >= threshold
        detection_evaluated = state != "calibrating" and score is not None
        has_track = bool(pmm_result.has_track)
        confirmed = state in {"confirmed", "coasting"} and has_track
        previous_state = self.previous_state
        previous_active = previous_state in {"tentative", "confirmed", "coasting"}
        active = state in {"tentative", "confirmed", "coasting"}
        acquired = active and not previous_active
        lost = state == "lost" and previous_active
        self.state_counts[state] += 1
        self.operational_frames += int(state != "calibrating")
        self.detection_evaluated += int(detection_evaluated)
        self.detections += int(detection_evaluated and detected)
        self.tracked_frames += int(has_track)
        self.confirmed_frames += int(confirmed)
        self.predicted_frames += int(bool(pmm_result.predicted))
        self.track_acquisitions += int(acquired)
        self.track_losses += int(lost)
        clean_latency: dict[str, Optional[float]] = {}
        for name, value in latency_ms.items():
            finite = _finite(value)
            clean_latency[str(name)] = finite
            if finite is not None:
                self._add_latency(str(name), finite)
        pipeline_ms = clean_latency.get("pipeline_total")
        deadline_missed = bool(
            pipeline_ms is not None
            and self.frame_period_s is not None
            and pipeline_ms > self.frame_period_s * 1_000.0
        )
        self.deadline_misses += int(deadline_missed)
        capture_to_complete_ms = None
        if captured_at_s is not None and completed_at_s is not None:
            capture_to_complete_ms = max(
                (float(completed_at_s) - float(captured_at_s)) * 1_000.0,
                0.0,
            )
            self._add_latency("capture_to_complete", capture_to_complete_ms)
        resource_snapshot = self.sampler.sample()
        resource_sequence = self.resource_samples or None
        if resource_snapshot is not None:
            self.resource_samples += 1
            resource_sequence = self.resource_samples
            self._record_resource(resource_snapshot)
            self._write(
                {
                    "record_type": "resource",
                    "sequence": resource_sequence,
                    "frame_index": int(frame_index),
                    **resource_snapshot,
                }
            )
        record = {
            "record_type": "frame",
            "sequence": self.frame_count,
            "frame_index": int(frame_index),
            "processed_at": _iso_now(),
            "timing_ms": {
                **clean_latency,
                "capture_to_complete": capture_to_complete_ms,
                "deadline_missed": deadline_missed,
            },
            "detection": {
                "evaluated": detection_evaluated,
                "detected": detected if detection_evaluated else None,
                "score": score,
                "threshold": threshold,
                "margin": (
                    score - threshold
                    if score is not None and threshold is not None
                    else None
                ),
            },
            "tracking": {
                "state": state,
                "previous_state": previous_state,
                "has_track": has_track,
                "confirmed": confirmed,
                "predicted": bool(pmm_result.predicted),
                "acquired": acquired,
                "lost": lost,
                "range_m": _finite(pmm_result.range_m),
                "radial_velocity_m_s": _finite(pmm_result.radial_velocity_m_s),
                "azimuth_deg": _finite(pmm_result.azimuth_deg),
                "elevation_deg": _finite(pmm_result.elevation_deg),
                "age_frames": int(pmm_result.age_frames),
                "hits": int(pmm_result.hits),
                "misses": int(pmm_result.misses),
            },
            "classification": (
                dict(classification) if classification is not None else None
            ),
            "capture": dict(capture_diagnostics),
            "resource_sample_sequence": resource_sequence,
        }
        self.previous_state = state
        self._write(record)

    def close(self, termination_status: str = "completed") -> dict[str, Any]:
        if self.closed:
            return {}
        last_sample_s = _finite(getattr(self.sampler, "last_sample_s", None))
        time_since_resource_s = (
            time.perf_counter() - last_sample_s
            if last_sample_s is not None
            else math.inf
        )
        final_resource = (
            self.sampler.sample(force=True)
            if self.resource_samples == 0 or time_since_resource_s >= 0.1
            else None
        )
        if final_resource is not None:
            self.resource_samples += 1
            self._record_resource(final_resource)
            self._write(
                {
                    "record_type": "resource",
                    "sequence": self.resource_samples,
                    "frame_index": self.frame_count,
                    **final_resource,
                }
            )
        run_duration_s = time.perf_counter() - self.started_s
        summary = {
            "record_type": "summary",
            "created_at": _iso_now(),
            "termination_status": termination_status,
            "run_duration_s": run_duration_s,
            "frames": self.frame_count,
            "processed_fps": _ratio(self.frame_count, run_duration_s),
            "processing_latency_ms": {
                name: _distribution(values)
                for name, values in sorted(self.latencies.items())
            },
            "deadline": {
                "frame_period_s": self.frame_period_s,
                "misses": self.deadline_misses,
                "miss_rate": _ratio(self.deadline_misses, self.frame_count),
            },
            "resources": {
                name: _distribution(values)
                for name, values in sorted(self.resources.items())
            },
            "detection": {
                "evaluated_frames": self.detection_evaluated,
                "positive_frames": self.detections,
                "positive_rate": _ratio(self.detections, self.detection_evaluated),
            },
            "tracking": {
                "state_counts": dict(sorted(self.state_counts.items())),
                "operational_frames": self.operational_frames,
                "tracked_frames": self.tracked_frames,
                "tracked_frame_rate": _ratio(
                    self.tracked_frames, self.operational_frames
                ),
                "tracked_frame_rate_all_frames": _ratio(
                    self.tracked_frames, self.frame_count
                ),
                "confirmed_frames": self.confirmed_frames,
                "confirmed_frame_rate": _ratio(
                    self.confirmed_frames, self.operational_frames
                ),
                "confirmed_frame_rate_all_frames": _ratio(
                    self.confirmed_frames, self.frame_count
                ),
                "predicted_frames": self.predicted_frames,
                "acquisitions": self.track_acquisitions,
                "losses": self.track_losses,
            },
        }
        self._write(summary)
        self.file.close()
        self.closed = True
        return summary
