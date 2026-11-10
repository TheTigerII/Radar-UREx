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
