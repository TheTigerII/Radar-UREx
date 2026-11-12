from pathlib import Path

from main.stop_all import is_radar_process_command


def test_radar_process_matcher_includes_all_deployed_workers() -> None:
    root = Path("/opt/Radar-UREx")
    for script in (
        root / "main" / "run.py",
        root / "aesa_radar" / "main" / "startup.py",
        root / "conventional_radar" / "main" / "livedatacapture.py",
        root / "experimental_radar" / "main" / "run.py",
    ):
        assert is_radar_process_command(
            ("/usr/bin/python", str(script)),
            project_root=root,
        )


def test_radar_process_matcher_never_selects_gui_or_unrelated_python() -> None:
    root = Path("/opt/Radar-UREx")
    assert not is_radar_process_command(
        ("/usr/bin/python", str(root / "aesa_radar" / "main" / "gui.py")),
        project_root=root,
    )
    assert not is_radar_process_command(
        ("/usr/bin/python", "/other/project/main/run.py"),
        project_root=root,
    )
    assert not is_radar_process_command(
        ("/usr/bin/python", "-m", "pytest"),
        project_root=root,
    )
