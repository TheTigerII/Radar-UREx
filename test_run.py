import argparse
import threading
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
        pmm_folding_size_max=20,
        pmm_detection_threshold=30_000.0,
        pmm_history_seconds=3.6,
        pmm_provisional_frames=5,
        pmm_confirmation_window_frames=10,
        pmm_confirmation_hits=7,
        pmm_coast_frames=10,
        radar_baud=115200,
        radar_command_timeout=10.0,
        dca_timeout=3.0,
        dca_retries=5,
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
        self.assertIn("--pmm-background-calibration-seconds", command)
        self.assertIn("--raw-output", command)

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


if __name__ == "__main__":
    unittest.main()
