import json
import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from unittest.mock import Mock, patch

from main.performance_evaluation import (
    PerformanceEvaluationLogger,
    SystemResourceSampler,
    default_performance_log_path,
)
from main.pmm import PmmTrackResult


def _result(state: str, *, score: float = 1000.0) -> PmmTrackResult:
    has_position = state in {"tentative", "confirmed", "coasting"}
    return PmmTrackResult(
        state=state,
        label="PMM target",
        calibration_frames_seen=10,
        calibration_frames_required=10,
        history_frames=10,
        range_bin=4 if has_position else None,
        range_m=2.5 if has_position else None,
        radial_velocity_m_s=0.1 if has_position else None,
        azimuth_deg=1.0 if has_position else None,
        elevation_deg=2.0 if has_position else None,
        raw_pmm_score=score,
        pmm_score=score,
        folding_size=4,
        background_projection_gain=1.0,
        azimuth_background_projection_gain=1.0,
        elevation_background_projection_gain=1.0,
        threshold=700.0,
        age_frames=10,
        hits=8,
        misses=0,
        predicted=state == "coasting",
        dp_transition_bins=1,
        dp_path_score=score,
        particle_count=100,
    )


def _resource() -> dict:
    return {
        "sampled_at": "2026-01-01T00:00:00.000+00:00",
        "monotonic_s": 1.0,
        "cpu": {
            "system_percent": 25.0,
            "process_percent_one_core": 50.0,
            "process_cpu_cores": 0.5,
            "logical_cpu_count": 8,
        },
        "memory": {
            "system_total_bytes": 1000,
            "system_available_bytes": 400,
            "system_used_percent": 60.0,
            "process_rss_bytes": 100,
        },
        "gpu": {
            "utilization_percent": 75.0,
            "utilization_source": "/sys/test/load",
            "memory": {
                "used_bytes": 200,
                "free_bytes": 300,
                "total_bytes": 500,
            },
        },
    }


class PerformanceEvaluationLoggerTests(unittest.TestCase):
    def test_default_path_is_timestamped_under_log_directory(self) -> None:
        now = datetime(2026, 8, 28, 13, 45, 12, 123456)
        path = default_performance_log_path(now)

        self.assertEqual(path.parent.name, "log")
        self.assertEqual(path.name, "20260828_134512_runtime_performance.jsonl")

    def test_jetson_gpu_load_is_discovered_and_converted_from_permille(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            load_path = root / "bus@0" / "17000000.gpu" / "load"
            load_path.parent.mkdir(parents=True)
            load_path.write_text("419\n", encoding="utf-8")
            with (
                patch.object(SystemResourceSampler, "GPU_LOAD_PATHS", ()),
                patch.object(
                    SystemResourceSampler,
                    "GPU_DISCOVERY_PATTERNS",
                    ((root, "**/*.gpu/load"),),
                ),
            ):
                sampler = SystemResourceSampler()
                utilization, source = sampler._read_gpu_load()

        self.assertEqual(utilization, 41.9)
        self.assertEqual(source, str(load_path))

    def test_writes_frame_resources_and_tracking_summary(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "performance.jsonl"
            logger = PerformanceEvaluationLogger(
                path,
                frame_period_s=0.1,
                radar_metadata={"profile": "test"},
                pmm_metadata={"version": "test"},
            )
            logger.sampler = Mock()
            logger.sampler.sample.side_effect = (_resource(), None, _resource())
            logger.record_frame(
                frame_index=1,
                captured_at_s=1.0,
                completed_at_s=1.05,
                latency_ms={"pipeline_total": 50.0, "pmm_tracking": 2.0},
                pmm_result=_result("confirmed"),
                capture_diagnostics={"lost_packets": 0},
            )
            logger.record_frame(
                frame_index=2,
                captured_at_s=1.1,
                completed_at_s=1.25,
                latency_ms={"pipeline_total": 150.0, "pmm_tracking": 3.0},
                pmm_result=_result("lost", score=100.0),
                capture_diagnostics={"lost_packets": 1},
            )
            summary = logger.close()
            records = [
                json.loads(line)
                for line in path.read_text(encoding="utf-8").splitlines()
            ]

        self.assertEqual(records[0]["format"], "radar-runtime-performance-jsonl")
        frames = [record for record in records if record["record_type"] == "frame"]
        self.assertTrue(frames[0]["tracking"]["acquired"])
        self.assertTrue(frames[1]["tracking"]["lost"])
        self.assertEqual(frames[1]["tracking"]["previous_state"], "confirmed")
        self.assertEqual(summary["deadline"]["misses"], 1)
        self.assertEqual(summary["detection"]["positive_frames"], 1)
        self.assertEqual(summary["tracking"]["acquisitions"], 1)
        self.assertEqual(summary["tracking"]["losses"], 1)
        self.assertEqual(
            summary["resources"]["gpu_utilization_percent"]["mean"],
            75.0,
        )


if __name__ == "__main__":
    unittest.main()
