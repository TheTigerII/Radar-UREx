import argparse
import queue
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import run


def _args() -> argparse.Namespace:
    return argparse.Namespace(
        config=run.DEFAULT_CONFIG_PATH,
        setup=run.DEFAULT_SETUP_PATH,
        host_ip=run.DEFAULT_HOST_IP,
        data_port=run.DEFAULT_DATA_PORT,
        socket_recv_buffer=run.DEFAULT_SOCKET_RECV_BUFFER_BYTES,
        packet_queue_size=run.DEFAULT_PACKET_QUEUE_SIZE,
        processing_queue_size=run.DEFAULT_PROCESSING_QUEUE_SIZE,
        display_update_every=1,
        max_range_m=20.0,
        pmm_background_calibration_seconds=30.0,
        pmm_max_target_speed_m_s=4.0,
        pmm_folding_size_min=2,
        pmm_folding_size_max=32,
        pmm_detection_threshold=30_000.0,
        pmm_adaptive_threshold_sigma=6.0,
        pmm_adaptive_threshold_minimum=700.0,
        pmm_history_seconds=3.6,
        pmm_provisional_frames=5,
        pmm_confirmation_window_frames=10,
        pmm_confirmation_hits=7,
        pmm_coast_frames=10,
        radar_baud=115200,
        radar_command_timeout=10.0,
        dca_timeout=3.0,
        dca_retries=5,
        calibration_distance_m=1.0,
        calibration_search_window_m=0.2,
        calibration_warmup_frames=16,
        calibration_frames=64,
        calibration_timeout_seconds=90.0,
        calibration_angle_deg=0.0,
    )


class CaptureCommandTests(unittest.TestCase):
    def test_forwards_only_pmm_capture_options(self) -> None:
        command = run.build_capture_command(
            _args(),
            "combined",
            Path("/tmp/pmm.jsonl"),
            Path("/tmp/raw.bin"),
        )

        self.assertIn("--pmm-detection-threshold", command)
        self.assertIn("--pmm-adaptive-threshold-sigma", command)
        self.assertIn("--pmm-adaptive-threshold-minimum", command)
        self.assertIn("--pmm-background-calibration-seconds", command)
        self.assertIn("--raw-output", command)

    def test_forwards_model_weights_only_when_classification_is_enabled(self) -> None:
        disabled = run.build_capture_command(
            _args(),
            "combined",
            Path("/tmp/pmm.jsonl"),
        )
        enabled = run.build_capture_command(
            _args(),
            "combined",
            Path("/tmp/pmm.jsonl"),
            model_weights_dir=Path("/tmp/model_weights"),
        )

        self.assertNotIn("--model-weights-dir", disabled)
        self.assertIn("--model-weights-dir", enabled)
        self.assertIn("/tmp/model_weights", enabled)

    def test_startup_uses_same_mini4_profile(self) -> None:
        command = run.build_startup_command(_args(), "/dev/ttyUSB0")

        config_values = [
            command[index + 1]
            for index, value in enumerate(command)
            if value in {"--config", "--sdk-profile"}
        ]
        self.assertEqual(
            config_values,
            [str(run.DEFAULT_CONFIG_PATH), str(run.DEFAULT_CONFIG_PATH)],
        )

    def test_calibration_command_uses_runtime_profile_without_pmm_output(self) -> None:
        command = run.build_capture_command(
            _args(),
            "calibration",
            Path("pmm.jsonl"),
            config_path=Path("runtime.cfg"),
        )
        self.assertIn("runtime.cfg", command)
        self.assertNotIn("--processed-output", command)
        self.assertIn("--calibration-frames", command)

        angular = run.build_capture_command(
            _args(),
            "azimuth-calibration",
            None,
            config_path=Path("runtime.cfg"),
            host_compensation_profile=Path("profile-mini4-20m.cfg"),
        )
        self.assertIn("--host-compensation-profile", angular)
        with self.assertRaisesRegex(ValueError, "host compensation"):
            run.build_capture_command(
                _args(),
                "elevation-calibration",
                None,
                config_path=Path("runtime.cfg"),
            )


class PromptTests(unittest.TestCase):
    def test_cp2105_enhanced_port_is_the_default(self) -> None:
        ports = [
            run.SerialPortInfo(
                "/dev/ttyUSB0",
                "CP2105 Dual USB to UART Bridge Controller - Standard COM Port",
            ),
            run.SerialPortInfo(
                "/dev/ttyUSB1",
                "CP2105 Dual USB to UART Bridge Controller - Enhanced COM Port",
            ),
        ]
        with (
            patch("run.list_serial_ports", return_value=ports),
            patch("builtins.print"),
        ):
            selected = run.resolve_radar_port(None)

        self.assertEqual(selected, "/dev/ttyUSB1")

    def test_explicit_radar_port_overrides_cp2105_default(self) -> None:
        with patch("run.list_serial_ports") as list_ports:
            selected = run.resolve_radar_port("/dev/serial/radar")

        self.assertEqual(selected, "/dev/serial/radar")
        list_ports.assert_not_called()

    def test_blank_display_uses_combined_mode(self) -> None:
        with patch("builtins.input", return_value=""):
            self.assertEqual(run.choose_display(None), "combined")

    def test_blank_duration_uses_five_minutes(self) -> None:
        with patch("builtins.input", return_value=""):
            self.assertEqual(run.choose_duration_minutes(None), 5.0)

    def test_realtime_classification_defaults_off(self) -> None:
        with patch("builtins.input", return_value=""):
            self.assertFalse(run.choose_realtime_classification(None))

    def test_realtime_classification_accepts_yes(self) -> None:
        with patch("builtins.input", return_value="yes"):
            self.assertTrue(run.choose_realtime_classification(None))

    def test_dataset_destination_defaults_to_dataset_root(self) -> None:
        with (
            patch("builtins.input", return_value=""),
            patch("builtins.print"),
        ):
            self.assertEqual(
                run.choose_dataset_destination(None),
                run.DEFAULT_DATASET_DIR,
            )

    def test_dataset_destination_accepts_uav_and_other(self) -> None:
        self.assertEqual(
            run.choose_dataset_destination("uav"),
            run.DEFAULT_DATASET_DIR / "uav",
        )
        self.assertEqual(
            run.choose_dataset_destination("other"),
            run.DEFAULT_DATASET_DIR / "other",
        )

    def test_calibration_modes_are_menu_options_six_through_eight(self) -> None:
        expected = (
            "calibration",
            "azimuth-calibration",
            "elevation-calibration",
        )
        for menu_value, display in zip(("6", "7", "8"), expected):
            with self.subTest(menu_value=menu_value):
                with patch("builtins.input", return_value=menu_value):
                    self.assertEqual(run.choose_display(None), display)

    def test_menu_displays_friendly_calibration_names(self) -> None:
        with (
            patch("builtins.input", return_value="6"),
            patch("builtins.print") as print_mock,
        ):
            self.assertEqual(run.choose_display(None), "calibration")

        rendered_lines = [
            str(call.args[0]) for call in print_mock.call_args_list if call.args
        ]
        self.assertIn("  6. range calibration", rendered_lines)
        self.assertIn("  7. azimuth calibration", rendered_lines)
        self.assertIn("  8. elevation calibration", rendered_lines)

    def test_friendly_range_calibration_name_is_accepted(self) -> None:
        with patch("builtins.input", return_value="range calibration"):
            self.assertEqual(run.choose_display(None), "calibration")

    def test_negative_duration_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            run.choose_duration_minutes(-1.0)


class CaptureReadinessTests(unittest.TestCase):
    def test_relay_sets_ready_on_listener_message(self) -> None:
        process = Mock()
        process.stdout = iter(
            (
                "Radar frame processor ready.\n",
                f"{run.CAPTURE_READY_PREFIX}on 192.168.33.30:4098\n",
            )
        )
        ready = threading.Event()

        with patch("builtins.print"):
            run.relay_capture_output(process, ready)

        self.assertTrue(ready.is_set())

    def test_waits_for_calibration_result_relayed_during_shutdown(self) -> None:
        result_queue: queue.SimpleQueue = queue.SimpleQueue()

        def relay_delayed_result() -> None:
            time.sleep(0.02)
            result_queue.put({"calibration_type": "range"})

        relay_thread = threading.Thread(target=relay_delayed_result)
        relay_thread.start()

        payload = run._wait_for_calibration_payload(
            result_queue,
            relay_thread,
            timeout_seconds=1.0,
        )

        relay_thread.join(timeout=1.0)
        self.assertEqual(payload, {"calibration_type": "range"})


if __name__ == "__main__":
    unittest.main()
