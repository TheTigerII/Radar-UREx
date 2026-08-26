"""Machine-readable runtime, resource, detection, and tracking telemetry."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, TextIO


PERFORMANCE_LOG_FORMAT = "radar-performance-jsonl"
PERFORMANCE_LOG_VERSION = 1
DEFAULT_RESOURCE_SAMPLE_INTERVAL_S = 1.0


def default_performance_log_path(log_dir: Optional[Path] = None) -> Path:
    directory = log_dir or Path(__file__).resolve().parent.parent / "log"
    timestamp = datetime.now().astimezone().strftime("%Y_%m_%dT%H_%M_%S")
    return directory / f"performance_{timestamp}.jsonl"


def _distribution(values: list[float]) -> dict[str, Optional[float] | int]:
    if not values:
        return {"count": 0, "mean": None, "p50": None, "p95": None, "max": None}
    ordered = sorted(float(value) for value in values)

    def percentile(percent: float) -> float:
        position = (len(ordered) - 1) * percent / 100.0
        lower = int(position)
        upper = min(lower + 1, len(ordered) - 1)
        fraction = position - lower
        return ordered[lower] + fraction * (ordered[upper] - ordered[lower])

    return {
        "count": len(ordered),
        "mean": sum(ordered) / len(ordered),
        "p50": percentile(50.0),
        "p95": percentile(95.0),
        "max": ordered[-1],
    }


def _read_text(path: Path) -> Optional[str]:
    try:
        return path.read_text(encoding="utf-8").strip()
    except (OSError, UnicodeError):
        return None


class ResourceSampler:
    """Sample process/system resources with optional NVIDIA GPU data."""

    _JETSON_GPU_LOAD_PATHS = (
        Path("/sys/devices/platform/bus@0/17000000.gpu/load"),
        Path("/sys/devices/gpu.0/load"),
        Path("/sys/devices/17000000.gpu/load"),
    )
    _JETSON_GPU_FREQ_PATHS = (
        Path(
            "/sys/devices/platform/bus@0/17000000.gpu/"
            "devfreq/17000000.gpu/cur_freq"
        ),
        Path("/sys/devices/gpu.0/devfreq/17000000.gpu/cur_freq"),
        Path("/sys/class/devfreq/17000000.gpu/cur_freq"),
    )
    _JETSON_GPU_THERMAL_ZONE_TYPE = "gpu-thermal"
    _THERMAL_ROOT = Path("/sys/class/thermal")

    def __init__(self) -> None:
        self.cpu_count = max(os.cpu_count() or 1, 1)
        try:
            self.page_size = int(os.sysconf("SC_PAGE_SIZE"))
        except (AttributeError, OSError, ValueError):
            self.page_size = 4096
        self._last_wall_s = time.perf_counter()
        self._last_process_cpu_s = time.process_time()
        self._last_system_cpu = self._read_system_cpu_times()
        self._nvidia_smi = shutil.which("nvidia-smi")
        self._nvidia_smi_failed = False

    @staticmethod
    def _read_system_cpu_times() -> Optional[tuple[int, int]]:
        if os.name == "nt":
            try:
                import ctypes

                idle = ctypes.c_ulonglong()
                kernel = ctypes.c_ulonglong()
                user = ctypes.c_ulonglong()
                success = ctypes.windll.kernel32.GetSystemTimes(
                    ctypes.byref(idle),
                    ctypes.byref(kernel),
                    ctypes.byref(user),
                )
                if success:
                    return int(kernel.value + user.value), int(idle.value)
            except (AttributeError, OSError, TypeError):
                return None
        text = _read_text(Path("/proc/stat"))
        if not text:
            return None
        fields = text.splitlines()[0].split()
        if not fields or fields[0] != "cpu" or len(fields) < 5:
            return None
        values = [int(value) for value in fields[1:]]
        idle = values[3] + (values[4] if len(values) > 4 else 0)
        return sum(values), idle

    @staticmethod
    def _memory_sample() -> dict[str, Optional[float | int]]:
        if os.name == "nt":
            try:
                import ctypes

                class MemoryStatus(ctypes.Structure):
                    _fields_ = [
                        ("length", ctypes.c_ulong),
                        ("memory_load", ctypes.c_ulong),
                        ("total_physical", ctypes.c_ulonglong),
                        ("available_physical", ctypes.c_ulonglong),
                        ("total_page_file", ctypes.c_ulonglong),
                        ("available_page_file", ctypes.c_ulonglong),
                        ("total_virtual", ctypes.c_ulonglong),
                        ("available_virtual", ctypes.c_ulonglong),
                        ("available_extended_virtual", ctypes.c_ulonglong),
                    ]

                status = MemoryStatus()
                status.length = ctypes.sizeof(status)
                if ctypes.windll.kernel32.GlobalMemoryStatusEx(
                    ctypes.byref(status)
                ):
                    return {
                        "total_bytes": int(status.total_physical),
                        "available_bytes": int(status.available_physical),
                        "used_percent": float(status.memory_load),
                    }
            except (AttributeError, OSError, TypeError):
                pass
        text = _read_text(Path("/proc/meminfo"))
        values: dict[str, int] = {}
        if text:
            for line in text.splitlines():
                key, separator, remainder = line.partition(":")
                if not separator:
                    continue
                token = remainder.strip().split()[0]
                try:
                    values[key] = int(token) * 1024
                except (ValueError, IndexError):
                    continue
        total = values.get("MemTotal")
        available = values.get("MemAvailable")
        used_percent = (
            100.0 * (total - available) / total
            if total and available is not None
            else None
        )
        return {
            "total_bytes": total,
            "available_bytes": available,
            "used_percent": used_percent,
        }

    def _process_rss_bytes(self) -> Optional[int]:
        if os.name == "nt":
            try:
                import ctypes

                class ProcessMemoryCounters(ctypes.Structure):
                    _fields_ = [
                        ("cb", ctypes.c_ulong),
                        ("page_fault_count", ctypes.c_ulong),
                        ("peak_working_set_size", ctypes.c_size_t),
                        ("working_set_size", ctypes.c_size_t),
                        ("quota_peak_paged_pool_usage", ctypes.c_size_t),
                        ("quota_paged_pool_usage", ctypes.c_size_t),
                        ("quota_peak_non_paged_pool_usage", ctypes.c_size_t),
                        ("quota_non_paged_pool_usage", ctypes.c_size_t),
                        ("pagefile_usage", ctypes.c_size_t),
                        ("peak_pagefile_usage", ctypes.c_size_t),
                    ]

                counters = ProcessMemoryCounters()
                counters.cb = ctypes.sizeof(counters)
                process = ctypes.windll.kernel32.GetCurrentProcess()
                if ctypes.windll.psapi.GetProcessMemoryInfo(
                    process,
                    ctypes.byref(counters),
                    counters.cb,
                ):
                    return int(counters.working_set_size)
            except (AttributeError, OSError, TypeError):
                return None
        text = _read_text(Path("/proc/self/statm"))
        if not text:
            return None
        try:
            return int(text.split()[1]) * self.page_size
        except (ValueError, IndexError):
            return None

    @staticmethod
    def _scaled_gpu_load(raw_value: str) -> Optional[float]:
        try:
            value = float(raw_value)
        except ValueError:
            return None
        # Jetson's load node is normally permille, while some devfreq drivers
        # expose a percentage. Normalize both representations to percent.
        return min(max(value / 10.0 if value > 100.0 else value, 0.0), 100.0)

    def _jetson_gpu_sample(self) -> Optional[dict[str, Any]]:
        utilization = None
        for path in self._JETSON_GPU_LOAD_PATHS:
            text = _read_text(path)
            if text is not None:
                utilization = self._scaled_gpu_load(text)
                if utilization is not None:
                    break
        frequency_hz = None
        for path in self._JETSON_GPU_FREQ_PATHS:
            text = _read_text(path)
            if text is not None:
                try:
                    frequency_hz = int(text)
                except ValueError:
                    pass
                break
        temperature_c = None
        try:
            thermal_zones = tuple(self._THERMAL_ROOT.glob("thermal_zone*"))
        except OSError:
            thermal_zones = ()
        for thermal_zone in thermal_zones:
            if _read_text(thermal_zone / "type") != self._JETSON_GPU_THERMAL_ZONE_TYPE:
                continue
            text = _read_text(thermal_zone / "temp")
            if text is not None:
                try:
                    temperature_c = float(text)
                    if temperature_c > 1000.0:
                        temperature_c /= 1000.0
                except ValueError:
                    pass
            break
        # A thermal zone alone is not sufficient to identify a Jetson GPU;
        # discrete systems may expose similarly named thermal sensors.
        if utilization is None and frequency_hz is None:
            return None
        return {
            "provider": "jetson_sysfs",
            "utilization_percent": utilization,
            "memory_used_bytes": None,
            "memory_total_bytes": None,
            "temperature_c": temperature_c,
            "power_w": None,
            "frequency_hz": frequency_hz,
            "shared_system_memory": True,
        }

    def _nvidia_smi_sample(self) -> Optional[dict[str, Any]]:
        if self._nvidia_smi is None or self._nvidia_smi_failed:
            return None
        try:
            completed = subprocess.run(
                [
                    self._nvidia_smi,
                    "--query-gpu=utilization.gpu,memory.used,memory.total,"
                    "temperature.gpu,power.draw",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                check=True,
                text=True,
                timeout=0.5,
            )
            first_gpu = completed.stdout.splitlines()[0]
            fields = [field.strip() for field in first_gpu.split(",")]
            if len(fields) != 5:
                raise ValueError("unexpected nvidia-smi output")

            def number(value: str) -> Optional[float]:
                try:
                    return float(value)
                except ValueError:
                    return None

            utilization, memory_used, memory_total, temperature, power = (
                number(value) for value in fields
            )
            if all(
                value is None
                for value in (
                    utilization,
                    memory_used,
                    memory_total,
                    temperature,
                    power,
                )
            ):
                raise ValueError("nvidia-smi returned no usable GPU metrics")
            return {
                "provider": "nvidia_smi",
                "utilization_percent": utilization,
                "memory_used_bytes": (
                    int(memory_used * 1024 * 1024)
                    if memory_used is not None
                    else None
                ),
                "memory_total_bytes": (
                    int(memory_total * 1024 * 1024)
                    if memory_total is not None
                    else None
                ),
                "temperature_c": temperature,
                "power_w": power,
                "frequency_hz": None,
                "shared_system_memory": False,
            }
        except (OSError, subprocess.SubprocessError, ValueError, IndexError):
            self._nvidia_smi_failed = True
            return None

    def sample(self) -> dict[str, Any]:
        now_s = time.perf_counter()
        process_cpu_s = time.process_time()
        wall_delta_s = max(now_s - self._last_wall_s, 1e-9)
        process_cpu_delta_s = max(process_cpu_s - self._last_process_cpu_s, 0.0)
        process_cpu_one_core = 100.0 * process_cpu_delta_s / wall_delta_s
        self._last_wall_s = now_s
        self._last_process_cpu_s = process_cpu_s

        system_cpu = self._read_system_cpu_times()
        system_cpu_percent = None
        if system_cpu is not None and self._last_system_cpu is not None:
            total_delta = system_cpu[0] - self._last_system_cpu[0]
            idle_delta = system_cpu[1] - self._last_system_cpu[1]
            if total_delta > 0:
                system_cpu_percent = 100.0 * (total_delta - idle_delta) / total_delta
        self._last_system_cpu = system_cpu

        # Jetson's sysfs counters are cheap and reliable on integrated GPUs;
        # prefer them over nvidia-smi, which can return a successful all-N/A
        # row on Jetson and would otherwise suppress the working fallback.
        gpu = self._jetson_gpu_sample()
        if gpu is None:
            gpu = self._nvidia_smi_sample()

        return {
            "process": {
                "pid": os.getpid(),
                "cpu_percent_one_core": process_cpu_one_core,
                "cpu_percent_normalized": process_cpu_one_core / self.cpu_count,
                "rss_bytes": self._process_rss_bytes(),
            },
            "system": {
                "cpu_percent": system_cpu_percent,
                "cpu_count": self.cpu_count,
                "memory": self._memory_sample(),
            },
            "gpu": gpu,
        }


class PerformanceMetricsLogger:
    """Write frame telemetry, resource samples, and an orderly run summary."""

    def __init__(
        self,
        output_path: Path,
        *,
        run_metadata: Optional[Mapping[str, Any]] = None,
        resource_sample_interval_s: float = DEFAULT_RESOURCE_SAMPLE_INTERVAL_S,
        emit_func: Optional[Callable[[str], None]] = None,
        resource_sampler: Optional[ResourceSampler] = None,
    ) -> None:
        if resource_sample_interval_s <= 0.0:
            raise ValueError("Resource sample interval must be positive")
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.file: Optional[TextIO] = self.output_path.open(
            "w", encoding="utf-8", buffering=1
        )
        self.emit = emit_func or (lambda _message: None)
        self.resource_sample_interval_s = float(resource_sample_interval_s)
        self.resource_sampler = resource_sampler or ResourceSampler()
        self.started_monotonic_s = time.perf_counter()
        self._next_resource_sample_s = self.started_monotonic_s
        self._frame_count = 0
        self._timings: dict[str, list[float]] = {}
        self._resource_values: dict[str, list[float]] = {
            "process_cpu_percent_one_core": [],
            "process_rss_bytes": [],
            "system_cpu_percent": [],
            "system_memory_used_percent": [],
            "gpu_utilization_percent": [],
            "gpu_memory_used_bytes": [],
            "gpu_temperature_c": [],
            "gpu_power_w": [],
            "gpu_frequency_hz": [],
        }
        self._detection_frames = 0
        self._dynamic_point_counts: list[float] = []
        self._dynamic_cluster_counts: list[float] = []
        self._static_candidate_counts: list[float] = []
        self._tracking_states = {
            "unavailable": 0,
            "absent": 0,
            "tentative": 0,
            "measured": 0,
            "predicted": 0,
        }
        self._track_acquisitions = 0
        self._track_losses = 0
        self._track_source_switches = 0
        self._previous_track_active = False
        self._previous_track_source: Optional[str] = None
        self._continuous_track_frames = 0
        self._longest_continuous_track_frames = 0

        self._write(
            {
                "record_type": "metadata",
                "format": PERFORMANCE_LOG_FORMAT,
                "version": PERFORMANCE_LOG_VERSION,
                "created_at": datetime.now().astimezone().isoformat(
                    timespec="milliseconds"
                ),
                "resource_sample_interval_s": self.resource_sample_interval_s,
                "run": dict(run_metadata or {}),
                "semantics": {
                    "processing_timings": "wall-clock time spent in each stage for this frame",
                    "capture_to_complete_ms": (
                        "monotonic latency from the frame's first received byte to "
                        "completion of host processing"
                    ),
                    "dynamic_points": "post-CFAR/FOV point detections, not ground truth",
                    "track_measured": "confirmed track associated with a current measurement",
                    "track_predicted": "confirmed track retained through a detection miss",
                    "accuracy_limitation": (
                        "Operational detection and tracking coverage are logged. "
                        "Precision, recall, localization error, ID metrics, and MOTA "
                        "require synchronized external ground truth."
                    ),
                },
            }
        )
        self.emit(f"Performance telemetry logging enabled: path={self.output_path}")

    def _write(self, record: Mapping[str, Any]) -> None:
        if self.file is None:
            return
        self.file.write(json.dumps(record, separators=(",", ":")) + "\n")

    def _record_resource_sample(self, now_s: float) -> None:
        resources = self.resource_sampler.sample()
        self._write(
            {
                "record_type": "resource_sample",
                "sampled_at": datetime.now().astimezone().isoformat(
                    timespec="milliseconds"
                ),
                "elapsed_s": now_s - self.started_monotonic_s,
                **resources,
            }
        )
        paths = {
            "process_cpu_percent_one_core": ("process", "cpu_percent_one_core"),
            "process_rss_bytes": ("process", "rss_bytes"),
            "system_cpu_percent": ("system", "cpu_percent"),
            "system_memory_used_percent": ("system", "memory", "used_percent"),
            "gpu_utilization_percent": ("gpu", "utilization_percent"),
            "gpu_memory_used_bytes": ("gpu", "memory_used_bytes"),
            "gpu_temperature_c": ("gpu", "temperature_c"),
            "gpu_power_w": ("gpu", "power_w"),
            "gpu_frequency_hz": ("gpu", "frequency_hz"),
        }
        for name, path in paths.items():
            value: Any = resources
            for key in path:
                if not isinstance(value, Mapping):
                    value = None
                    break
                value = value.get(key)
            if value is not None:
                self._resource_values[name].append(float(value))

    def record_frame(
        self,
        *,
        frame_index: int,
        captured_at_s: Optional[float],
        processing_timings_ms: Mapping[str, float],
        detection: Mapping[str, Any],
        tracking: Mapping[str, Any],
        classification: Optional[Mapping[str, Any]] = None,
    ) -> None:
        now_s = time.perf_counter()
        while now_s >= self._next_resource_sample_s:
            self._record_resource_sample(now_s)
            self._next_resource_sample_s = now_s + self.resource_sample_interval_s

        timings = {
            str(stage): max(float(value), 0.0)
            for stage, value in processing_timings_ms.items()
        }
        for stage, value in timings.items():
            self._timings.setdefault(stage, []).append(value)

        available = bool(detection.get("available", False))
        dynamic_points = int(detection.get("dynamic_point_count", 0))
        dynamic_clusters = int(detection.get("dynamic_cluster_count", 0))
        static_candidates = int(detection.get("static_candidate_count", 0))
        if available:
            self._dynamic_point_counts.append(float(dynamic_points))
            self._dynamic_cluster_counts.append(float(dynamic_clusters))
            self._static_candidate_counts.append(float(static_candidates))
            if dynamic_points > 0 or static_candidates > 0:
                self._detection_frames += 1

        state = str(tracking.get("state", "unavailable"))
        if state not in self._tracking_states:
            state = "unavailable"
        self._tracking_states[state] += 1
        track_active = state in {"measured", "predicted"}
        source = tracking.get("source") if track_active else None
        if track_active and not self._previous_track_active:
            self._track_acquisitions += 1
        elif not track_active and self._previous_track_active:
            self._track_losses += 1
        if (
            track_active
            and self._previous_track_active
            and source is not None
            and self._previous_track_source is not None
            and source != self._previous_track_source
        ):
            self._track_source_switches += 1
        if track_active:
            self._continuous_track_frames += 1
            self._longest_continuous_track_frames = max(
                self._longest_continuous_track_frames,
                self._continuous_track_frames,
            )
        else:
            self._continuous_track_frames = 0
        self._previous_track_active = track_active
        self._previous_track_source = str(source) if source is not None else None
        self._frame_count += 1

        self._write(
            {
                "record_type": "frame",
                "frame_index": int(frame_index),
                "recorded_at": datetime.now().astimezone().isoformat(
                    timespec="milliseconds"
                ),
                "captured_at_monotonic_s": (
                    float(captured_at_s) if captured_at_s is not None else None
                ),
                "capture_to_complete_ms": (
                    max((now_s - float(captured_at_s)) * 1000.0, 0.0)
                    if captured_at_s is not None
                    else None
                ),
                "processing_ms": timings,
                "detection": dict(detection),
                "tracking": dict(tracking),
                "classification": dict(classification) if classification else None,
            }
        )

    def close(self, termination_status: str = "completed") -> dict[str, Any]:
        if self.file is None:
            return {}
        available_detection_frames = len(self._dynamic_point_counts)
        track_observed_frames = (
            self._tracking_states["measured"] + self._tracking_states["predicted"]
        )
        track_evaluable_frames = self._frame_count - self._tracking_states["unavailable"]
        summary = {
            "record_type": "run_summary",
            "termination_status": str(termination_status),
            "ended_at": datetime.now().astimezone().isoformat(
                timespec="milliseconds"
            ),
            "duration_s": time.perf_counter() - self.started_monotonic_s,
            "frames": self._frame_count,
            "processing_ms": {
                stage: _distribution(values)
                for stage, values in self._timings.items()
            },
            "resources": {
                name: _distribution(values)
                for name, values in self._resource_values.items()
            },
            "detection": {
                "evaluable_frames": available_detection_frames,
                "frames_with_candidates": self._detection_frames,
                "candidate_frame_rate": (
                    self._detection_frames / available_detection_frames
                    if available_detection_frames
                    else None
                ),
                "dynamic_point_count": _distribution(self._dynamic_point_counts),
                "dynamic_cluster_count": _distribution(self._dynamic_cluster_counts),
                "static_candidate_count": _distribution(self._static_candidate_counts),
            },
            "tracking": {
                "states": dict(self._tracking_states),
                "evaluable_frames": track_evaluable_frames,
                "continuity_coverage": (
                    track_observed_frames / track_evaluable_frames
                    if track_evaluable_frames
                    else None
                ),
                "measurement_coverage": (
                    self._tracking_states["measured"] / track_evaluable_frames
                    if track_evaluable_frames
                    else None
                ),
                "prediction_fraction_while_tracked": (
                    self._tracking_states["predicted"] / track_observed_frames
                    if track_observed_frames
                    else None
                ),
                "acquisitions": self._track_acquisitions,
                "losses": self._track_losses,
                "source_switches": self._track_source_switches,
                "longest_continuous_track_frames": self._longest_continuous_track_frames,
            },
        }
        self._write(summary)
        self.file.close()
        self.file = None
        self.emit(
            "Performance telemetry saved: "
            f"frames={self._frame_count}, path={self.output_path}"
        )
        return summary
