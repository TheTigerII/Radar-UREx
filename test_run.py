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


class ChooseMicroDopplerRangeTests(unittest.TestCase):
    def test_blank_input_uses_2_15_meter_default(self) -> None:
        with patch("builtins.input", return_value=""):
            self.assertEqual(
                run.choose_micro_doppler_range_m(None),
                2.15,
            )

    def test_explicit_positive_range_skips_prompt(self) -> None:
        with patch("builtins.input") as prompt:
            self.assertEqual(
                run.choose_micro_doppler_range_m(3.25),
                3.25,
            )
            prompt.assert_not_called()

    def test_invalid_explicit_range_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "finite positive"):
            run.choose_micro_doppler_range_m(0.0)


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

    def test_dedicated_rotor_display_is_a_cli_choice(self) -> None:
        self.assertIn("micro-doppler", run.DISPLAY_CHOICES)
        with patch("builtins.input", return_value="6"):
            self.assertEqual(run.choose_display(None), "micro-doppler")

    def test_dedicated_rotor_defaults_to_proven_lvds_profile(self) -> None:
        self.assertEqual(run.DEFAULT_ROTOR_CONFIG_PATH, run.DEFAULT_CONFIG_PATH)

    def test_dedicated_rotor_defaults_match_current_drone(self) -> None:
        with patch("sys.argv", ("run.py",)):
            args = run.parse_args()

        self.assertEqual(args.rotor_blades, 2)
        self.assertEqual(args.rotor_rpm_max, 10_700.0)


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
            static_detection=True,
            static_warmup_frames=30,
            static_reference_frames=90,
            static_min_change_db=6.0,
            static_background_update_rate=0.01,
            static_cluster_min_samples=3,
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
        self.assertIn("--static-detection", command)
        self.assertIn("--static-warmup-frames", command)
        self.assertIn("--static-reference-frames", command)
        self.assertIn("--static-min-change-db", command)
        self.assertIn("--static-background-update-rate", command)
        self.assertIn("--static-cluster-min-samples", command)
        self.assertNotIn("--raw-output", command)

        args.static_detection = False
        disabled_command = run.build_capture_command(
            args,
            "point-cloud",
            Path("processed.jsonl"),
        )
        self.assertIn("--no-static-detection", disabled_command)
        self.assertNotIn("--static-detection", disabled_command)

    def test_rotor_command_forwards_gate_and_estimator_settings(self) -> None:
        args = SimpleNamespace(
            display_update_every=1,
            config=Path("rotor_profile.cfg"),
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
            static_detection=False,
            static_warmup_frames=30,
            static_reference_frames=90,
            static_min_change_db=6.0,
            static_background_update_rate=0.01,
            static_cluster_min_samples=1,
            micro_doppler_range_m=2.15,
            micro_doppler_range_half_width_bins=1,
            rotor_blades=2,
            rotor_count=2,
            rotor_radius_m=0.05,
            rotor_rpm_min=2_000.0,
            rotor_rpm_max=12_000.0,
        )

        command = run.build_capture_command(
            args,
            "micro-doppler",
            Path("processed.jsonl"),
        )

        self.assertIn("--micro-doppler-range-m", command)
        self.assertIn("2.15", command)
        self.assertIn("--rotor-radius-m", command)
        self.assertIn("--rotor-rpm-max", command)


class SubprocessEnvironmentTests(unittest.TestCase):
    def test_linux_vnc_environment_defaults_to_display_zero(self) -> None:
        with (
            patch.dict(
                "os.environ",
                {"PATH": "/usr/bin"},
                clear=True,
            ),
            patch("run.os.name", "posix"),
            patch("run.os.getuid", return_value=1000),
            patch("run.Path.exists", return_value=True),
            patch("run.Path.is_file", return_value=True),
        ):
            environment = run.subprocess_environment()

        self.assertEqual(environment["DISPLAY"], ":0")
        self.assertEqual(
            environment["XAUTHORITY"],
            "/run/user/1000/gdm/Xauthority",
        )

    def test_existing_display_environment_is_preserved(self) -> None:
        with (
            patch.dict(
                "os.environ",
                {
                    "DISPLAY": ":7",
                    "XAUTHORITY": "/tmp/custom-xauthority",
                },
                clear=True,
            ),
            patch("run.os.name", "posix"),
        ):
            environment = run.subprocess_environment()

        self.assertEqual(environment["DISPLAY"], ":7")
        self.assertEqual(
            environment["XAUTHORITY"],
            "/tmp/custom-xauthority",
        )


if __name__ == "__main__":
    unittest.main()
