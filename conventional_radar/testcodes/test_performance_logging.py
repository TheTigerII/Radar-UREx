import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from main.livedatacapture import ProcessingTimingStats
from main.performance_logging import PerformanceMetricsLogger, ResourceSampler


class FakeResourceSampler:
    def sample(self):
        return {
            "process": {
                "pid": 123,
                "cpu_percent_one_core": 50.0,
                "cpu_percent_normalized": 12.5,
                "rss_bytes": 1000,
            },
            "system": {
                "cpu_percent": 25.0,
                "cpu_count": 4,
                "memory": {
                    "total_bytes": 10_000,
                    "available_bytes": 6_000,
                    "used_percent": 40.0,
                },
            },
            "gpu": {
                "provider": "test",
                "utilization_percent": 75.0,
                "memory_used_bytes": 2000,
                "memory_total_bytes": 8000,
                "temperature_c": 60.0,
                "power_w": 8.0,
                "frequency_hz": None,
            },
        }


def test_processing_timing_stats_isolates_one_frame_samples():
    timings = ProcessingTimingStats()
    timings.add("range_fft", 0.001)
    cursor = timings.snapshot_counts()

    timings.add("range_fft", 0.002)
    timings.add("classification", 0.003)

    assert timings.samples_since(cursor) == {
        "range_fft": pytest.approx(2.0),
        "classification": pytest.approx(3.0),
    }


def test_performance_log_records_frames_resources_and_summary(tmp_path: Path):
    output = tmp_path / "performance.jsonl"
    logger = PerformanceMetricsLogger(
        output,
        run_metadata={"display_mode": "point-cloud"},
        resource_sample_interval_s=60.0,
        resource_sampler=FakeResourceSampler(),
    )
    frames = [
        ("absent", None, 0, 0),
        ("measured", "dynamic", 2, 1),
        ("predicted", "dynamic", 0, 0),
        ("measured", "static", 0, 1),
        ("absent", None, 0, 0),
    ]
    for frame_index, (state, source, points, static_candidates) in enumerate(
        frames, start=1
    ):
        logger.record_frame(
            frame_index=frame_index,
            captured_at_s=None,
            processing_timings_ms={"total": float(frame_index)},
            detection={
                "available": True,
                "dynamic_point_count": points,
                "dynamic_cluster_count": int(points > 0),
                "static_candidate_count": static_candidates,
            },
            tracking={"state": state, "source": source},
        )
    summary = logger.close("completed")

    records = [json.loads(line) for line in output.read_text().splitlines()]
    assert records[0]["record_type"] == "metadata"
    assert records[0]["format"] == "radar-performance-jsonl"
    assert records[1]["record_type"] == "resource_sample"
    assert len([item for item in records if item["record_type"] == "frame"]) == 5
    assert records[-1] == summary
    assert summary["processing_ms"]["total"]["p50"] == pytest.approx(3.0)
    assert summary["resources"]["gpu_utilization_percent"]["mean"] == 75.0
    assert summary["resources"]["gpu_temperature_c"]["mean"] == 60.0
    assert summary["resources"]["gpu_power_w"]["mean"] == 8.0
    assert summary["detection"]["frames_with_candidates"] == 2
    assert summary["tracking"]["states"] == {
        "unavailable": 0,
        "absent": 2,
        "tentative": 0,
        "measured": 2,
        "predicted": 1,
    }
    assert summary["tracking"]["acquisitions"] == 1
    assert summary["tracking"]["losses"] == 1
    assert summary["tracking"]["source_switches"] == 1
    assert summary["tracking"]["longest_continuous_track_frames"] == 3


def test_performance_log_rejects_non_positive_resource_interval(tmp_path: Path):
    with pytest.raises(ValueError, match="must be positive"):
        PerformanceMetricsLogger(
            tmp_path / "performance.jsonl",
            resource_sample_interval_s=0.0,
            resource_sampler=FakeResourceSampler(),
        )


def test_jetson_sysfs_gpu_sample_uses_orin_paths_and_thermal_zone(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    load = tmp_path / "gpu" / "load"
    frequency = tmp_path / "gpu" / "devfreq" / "cur_freq"
    thermal_zone = tmp_path / "thermal" / "thermal_zone1"
    load.parent.mkdir(parents=True)
    frequency.parent.mkdir(parents=True)
    thermal_zone.mkdir(parents=True)
    load.write_text("354\n", encoding="utf-8")
    frequency.write_text("624750000\n", encoding="utf-8")
    (thermal_zone / "type").write_text("gpu-thermal\n", encoding="utf-8")
    (thermal_zone / "temp").write_text("51843\n", encoding="utf-8")
    monkeypatch.setattr(ResourceSampler, "_JETSON_GPU_LOAD_PATHS", (load,))
    monkeypatch.setattr(ResourceSampler, "_JETSON_GPU_FREQ_PATHS", (frequency,))
    monkeypatch.setattr(
        ResourceSampler,
        "_THERMAL_ROOT",
        thermal_zone.parent,
    )

    gpu = ResourceSampler()._jetson_gpu_sample()

    assert gpu is not None
    assert gpu["provider"] == "jetson_sysfs"
    assert gpu["utilization_percent"] == pytest.approx(35.4)
    assert gpu["frequency_hz"] == 624_750_000
    assert gpu["temperature_c"] == pytest.approx(51.843)
    assert gpu["shared_system_memory"] is True


def test_nvidia_smi_all_na_result_triggers_fallback(
    monkeypatch: pytest.MonkeyPatch,
):
    sampler = ResourceSampler()
    sampler._nvidia_smi = "/usr/bin/nvidia-smi"
    monkeypatch.setattr(
        "main.performance_logging.subprocess.run",
        lambda *args, **kwargs: SimpleNamespace(
            stdout="N/A, N/A, N/A, N/A, N/A\n"
        ),
    )

    assert sampler._nvidia_smi_sample() is None
    assert sampler._nvidia_smi_failed is True


def test_resource_sampler_prefers_jetson_sysfs_over_nvidia_smi(
    monkeypatch: pytest.MonkeyPatch,
):
    sampler = ResourceSampler()
    expected = {
        "provider": "jetson_sysfs",
        "utilization_percent": 12.5,
    }
    monkeypatch.setattr(sampler, "_jetson_gpu_sample", lambda: expected)

    def unexpected_nvidia_smi():
        raise AssertionError("nvidia-smi should not run when Jetson sysfs works")

    monkeypatch.setattr(sampler, "_nvidia_smi_sample", unexpected_nvidia_smi)

    assert sampler.sample()["gpu"] == expected
