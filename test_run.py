import unittest
from unittest.mock import patch

import run


class ChooseDurationMinutesTests(unittest.TestCase):
    def test_blank_input_uses_three_minute_default(self) -> None:
        with patch("builtins.input", return_value=""):
            self.assertEqual(
                run.choose_duration_minutes(None),
                run.DEFAULT_DURATION_MINUTES,
            )

    def test_zero_means_unlimited(self) -> None:
        with patch("builtins.input", return_value="0"):
            self.assertEqual(run.choose_duration_minutes(None), 0.0)

    def test_cli_value_skips_prompt(self) -> None:
        with patch("builtins.input") as prompt:
            self.assertEqual(run.choose_duration_minutes(1.5), 1.5)
            prompt.assert_not_called()

    def test_negative_cli_value_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "finite, non-negative"):
            run.choose_duration_minutes(-1.0)

    def test_non_finite_cli_value_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "finite, non-negative"):
            run.choose_duration_minutes(float("nan"))


class ChooseDisplayTests(unittest.TestCase):
    def test_combined_display_menu_choice(self) -> None:
        with patch("builtins.input", return_value="5"):
            self.assertEqual(
                run.choose_display(None),
                "point-cloud-micro-doppler",
            )

    def test_combined_display_is_a_cli_choice(self) -> None:
        self.assertIn("point-cloud-micro-doppler", run.DISPLAY_CHOICES)


if __name__ == "__main__":
    unittest.main()
