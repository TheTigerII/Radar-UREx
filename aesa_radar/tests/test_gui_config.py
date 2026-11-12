from pathlib import Path

from main.gui_config import (
    CONVENTIONAL_CONFIG_PATH,
    CONVENTIONAL_MAIN,
    CONVENTIONAL_MODEL_WEIGHTS_DIR,
    CONVENTIONAL_OPERATION_MODE,
    CONVENTIONAL_SETUP_PATH,
    GuiSettings,
)


def test_conventional_mode_builds_prompt_free_ppi_gpu_command(tmp_path: Path) -> None:
    settings = GuiSettings(
        display="ppi",
        operation_mode=CONVENTIONAL_OPERATION_MODE,
        radar_port="/dev/ttyUSB0",
        config=CONVENTIONAL_CONFIG_PATH,
        setup=CONVENTIONAL_SETUP_PATH,
        capture_dir=tmp_path,
        classification=True,
        model_weights_dir=CONVENTIONAL_MODEL_WEIGHTS_DIR,
    )

    assert settings.validation_errors() == []
    command = settings.command("/test/python")
    assert command[:3] == ["/test/python", "-u", str(CONVENTIONAL_MAIN)]
    assert command[command.index("--display") + 1] == "ppi"
    assert command[command.index("--classification-device") + 1] == "auto"
    assert "--classification" in command
    assert "--performance-logging" in command


def _conventional_settings(tmp_path: Path, **overrides) -> GuiSettings:
    values = dict(
        display="ppi",
        operation_mode=CONVENTIONAL_OPERATION_MODE,
        radar_port="/dev/ttyUSB0",
        config=CONVENTIONAL_CONFIG_PATH,
        setup=CONVENTIONAL_SETUP_PATH,
        capture_dir=tmp_path,
        processed_output=tmp_path / "capture.jsonl",
        classification=True,
        model_weights_dir=CONVENTIONAL_MODEL_WEIGHTS_DIR,
    )
    values.update(overrides)
    return GuiSettings(**values)


MODE4_FLAGS = (
    "--detection-power",
    "--clutter-reference",
    "--cfar-threshold",
    "--tbd-window-frames",
)


def test_mode_1_command_never_contains_mode_4_flags(tmp_path: Path) -> None:
    command = _conventional_settings(tmp_path).command("/test/python")
    assert not set(MODE4_FLAGS) & set(command)


def test_mode_4_is_mode_1_plus_enhanced_detection_flags(tmp_path: Path) -> None:
    from main.gui_config import CONVENTIONAL_ENHANCED_OPERATION_MODE

    mode1 = _conventional_settings(tmp_path).command("/test/python")
    settings = _conventional_settings(
        tmp_path, operation_mode=CONVENTIONAL_ENHANCED_OPERATION_MODE
    )
    assert settings.validation_errors() == []
    mode4 = settings.command("/test/python")
    assert mode4[: len(mode1)] == mode1
    assert mode4[len(mode1) :] == [
        "--detection-power", "coherent-beam",
        "--clutter-reference", "mean",
        "--cfar-threshold", "calibrated",
        "--tbd-window-frames", "8",
    ]


def test_mode_4_stages_can_be_disabled_individually(tmp_path: Path) -> None:
    from main.gui_config import CONVENTIONAL_ENHANCED_OPERATION_MODE

    command = _conventional_settings(
        tmp_path,
        operation_mode=CONVENTIONAL_ENHANCED_OPERATION_MODE,
        enhanced_coherent_beams=False,
        enhanced_track_before_detect=False,
    ).command("/test/python")
    assert "--detection-power" not in command
    assert "--tbd-window-frames" not in command
    assert command[command.index("--cfar-threshold") + 1] == "calibrated"
    assert command[command.index("--clutter-reference") + 1] == "mean"
