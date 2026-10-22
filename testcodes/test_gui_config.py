import tempfile
import unittest
from pathlib import Path

from main import run
from main.gui_config import GuiSettings


class GuiSettingsTests(unittest.TestCase):
    def settings(self) -> GuiSettings:
        return GuiSettings(
            radar_port="/dev/ttyUSB1",
            config=run.DEFAULT_CONFIG_PATH,
            setup=run.DEFAULT_SETUP_PATH,
        )

    def test_normal_command_supplies_every_interactive_choice(self) -> None:
        settings = self.settings()
        command = settings.command("python-test")

        self.assertEqual(command[:2], ["python-test", "-u"])
        self.assertIn("--display", command)
        self.assertIn("--radar-port", command)
        self.assertIn("--duration-minutes", command)
        self.assertIn("--dataset-destination", command)
        self.assertIn("--no-classification", command)

    def test_classification_and_logging_options_are_forwarded(self) -> None:
        settings = self.settings()
        settings.classification = True
        settings.inference_logging = True
        settings.evaluation_label = "drone"
        settings.inference_log = Path("evaluation.jsonl")

        command = settings.command()

        self.assertIn("--classification", command)
        self.assertIn("--inference-logging", command)
        self.assertEqual(command[command.index("--evaluation-label") + 1], "drone")
        self.assertEqual(
            command[command.index("--inference-log") + 1], "evaluation.jsonl"
        )

    def test_calibration_command_uses_calibration_values(self) -> None:
        settings = self.settings()
        settings.display = "azimuth-calibration"
        settings.calibration_distance_m = 1.25
        settings.calibration_angle_deg = -4.5

        command = settings.command()

        self.assertIn("--calibration-distance-m", command)
        self.assertIn("--calibration-angle-deg", command)
        self.assertNotIn("--duration-minutes", command)
        self.assertNotIn("--classification", command)

    def test_validation_reports_missing_required_files_and_port(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            missing = Path(directory) / "missing.cfg"
            settings = GuiSettings(radar_port="", config=missing, setup=missing)

            errors = settings.validation_errors()

        self.assertTrue(any("UART" in error for error in errors))
        self.assertTrue(any("Radar profile" in error for error in errors))
        self.assertTrue(any("DCA1000 setup" in error for error in errors))

    def test_fixed_threshold_is_optional(self) -> None:
        settings = self.settings()
        self.assertNotIn("--pmm-detection-threshold", settings.command())
        settings.pmm_detection_threshold = 1234.5
        self.assertIn("--pmm-detection-threshold", settings.command())


if __name__ == "__main__":
    unittest.main()
