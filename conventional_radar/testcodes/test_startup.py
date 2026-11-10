import json
import unittest
from pathlib import Path
from types import SimpleNamespace

from main.startup import (
    DCA1000Setup,
    PreflightValidator,
    RuntimeOptions,
    _build_dca1000_packet_payload,
    _iter_sdk_cli_profile_commands,
)


class DCA1000PacketDelayTests(unittest.TestCase):
    def setUp(self) -> None:
        options = RuntimeOptions(
            config_path=Path("profile.cfg"),
            setup_path=Path("setup.json"),
        )
        self.validator = PreflightValidator(options)

    @staticmethod
    def _config(packet_delay_us: int):
        dca1000 = DCA1000Setup(
            capture_hardware="DCA1000",
            data_logging_mode="raw",
            data_transfer_mode="LVDSCapture",
            data_capture_mode="ethernetStream",
            packet_sequence_enable=True,
            packet_delay_us=packet_delay_us,
        )
        return SimpleNamespace(dca1000=dca1000, setup_json={})

    def test_supported_packet_delay_boundaries_are_accepted(self) -> None:
        for delay_us in (5, 50, 500):
            errors = []
            self.validator._validate_dca1000_settings(
                self._config(delay_us),
                errors,
            )
            self.assertEqual(errors, [])

    def test_packet_delay_outside_hardware_range_is_rejected(self) -> None:
        for delay_us in (4, 501):
            errors = []
            self.validator._validate_dca1000_settings(
                self._config(delay_us),
                errors,
            )
            self.assertTrue(
                any("between 5 and 500" in error for error in errors),
                errors,
            )

    def test_fifty_microseconds_encodes_to_6250_fpga_cycles(self) -> None:
        payload = _build_dca1000_packet_payload(self._config(50))

        self.assertEqual(int.from_bytes(payload[0:2], "little"), 1470)
        self.assertEqual(int.from_bytes(payload[2:4], "little"), 6250)

    def test_repository_setup_uses_one_hundred_microseconds(self) -> None:
        setup_path = (
            Path(__file__).resolve().parent.parent / "profiles" / "setup.json"
        )
        setup = json.loads(setup_path.read_text(encoding="utf-8"))

        self.assertEqual(setup["DCA1000Config"]["packetDelay_us"], 100)


class SdkProfileCommandTests(unittest.TestCase):
    def test_repository_profile_does_not_send_host_angle_metadata(self) -> None:
        profile_path = (
            Path(__file__).resolve().parent.parent / "profiles" / "profile.cfg"
        )

        commands = list(_iter_sdk_cli_profile_commands(profile_path))

        self.assertIn("sensorStart", commands)
        self.assertFalse(
            any(command.startswith("hostAngleCalibration") for command in commands)
        )


if __name__ == "__main__":
    unittest.main()
