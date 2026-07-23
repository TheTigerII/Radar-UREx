import unittest
from pathlib import Path
from types import SimpleNamespace
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
    def test_blank_input_uses_combined_display_default(self) -> None:
        with patch("builtins.input", return_value=""):
            self.assertEqual(
                run.choose_display(None),
                "point-cloud-micro-doppler",
            )

    def test_combined_display_menu_choice(self) -> None:
        with patch("builtins.input", return_value="5"):
            self.assertEqual(
                run.choose_display(None),
                "point-cloud-micro-doppler",
            )

    def test_combined_display_is_a_cli_choice(self) -> None:
        self.assertIn("point-cloud-micro-doppler", run.DISPLAY_CHOICES)


class CaptureCommandTests(unittest.TestCase):
    def test_processed_output_is_default_and_raw_output_is_opt_in(self) -> None:
        args = SimpleNamespace(
            display_update_every=1,
            config=Path("profile.cfg"),
            setup=Path("setup.json"),
            host_ip="192.168.33.30",
            data_port=4098,
            socket_recv_buffer=4 * 1024 * 1024,
            packet_queue_size=8192,
            processing_queue_size=32,
            max_range_m=10.0,
            cluster_eps_m=0.5,
            cluster_min_samples=2,
            clutter_map_update_rate=0.02,
            clutter_map_warmup_frames=30,
            clutter_map_min_snr_db=6.0,
        )

        command = run.build_capture_command(
            args,
            "point-cloud-micro-doppler",
            Path("processed.jsonl"),
        )

        self.assertIn("--processed-output", command)
        self.assertIn("processed.jsonl", command)
        self.assertIn("--socket-recv-buffer", command)
        self.assertIn("--packet-queue-size", command)
        self.assertIn("--processing-queue-size", command)
        self.assertNotIn("--raw-output", command)


if __name__ == "__main__":
    unittest.main()
