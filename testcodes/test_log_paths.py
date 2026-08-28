from datetime import datetime, timezone

from main.classification_evaluation import default_inference_log_path
from main.log_paths import default_terminal_log_path, new_run_log_id
from main.performance_logging import default_performance_log_path


def test_run_log_id_uses_date_and_second_only(tmp_path) -> None:
    now = datetime(2026, 8, 28, 15, 30, 45, 123456, tzinfo=timezone.utc)

    run_id = new_run_log_id(now, log_dir=tmp_path)

    assert run_id == "20260828_153045"


def test_run_log_id_adds_counter_when_second_already_exists(tmp_path) -> None:
    now = datetime(2026, 8, 28, 15, 30, 45, tzinfo=timezone.utc)
    (tmp_path / "livedatacapture_20260828_153045.log").touch()

    run_id = new_run_log_id(now, log_dir=tmp_path)

    assert run_id == "20260828_153045_2"


def test_default_log_filenames_share_the_capture_run_id() -> None:
    run_id = "20260828_153045"

    terminal = default_terminal_log_path(run_id=run_id)
    performance = default_performance_log_path(run_id=run_id)
    inference = default_inference_log_path(run_id=run_id)

    assert terminal.name == f"livedatacapture_{run_id}.log"
    assert performance.name == f"performance_{run_id}.jsonl"
    assert inference.name == f"live_inference_{run_id}.jsonl"
