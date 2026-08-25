import json
import unittest
from pathlib import Path
from types import SimpleNamespace

from main.startup import (
    DCA1000Setup,
    PreflightValidator,
    RuntimeOptions,
    StartupError,
    DCA1000Command,
    _build_dca1000_packet_payload,
    _dca1000_payload_override,
)


class DCA1000PacketDelayTests(unittest.TestCase):
    def setUp(self) -> None:
        options = RuntimeOptions(
            config_path=Path("profile-mini4-20m.cfg"),
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

    def test_repository_setup_uses_fifty_microseconds(self) -> None:
        setup_path = Path(__file__).resolve().parent.parent / "profiles" / "setup.json"
        setup = json.loads(setup_path.read_text(encoding="utf-8"))

        self.assertEqual(setup["DCA1000Config"]["packetDelay_us"], 50)


class StartupValidationTests(unittest.TestCase):
    def test_dry_run_does_not_require_serial_settings(self) -> None:
        validator = PreflightValidator(
            RuntimeOptions(Path("profile.cfg"), Path("setup.json"))
        )
        errors = []

        validator._validate_radar_control_settings(
            SimpleNamespace(
                radar_device=SimpleNamespace(control_port=None, baud_rate=None)
            ),
            errors,
        )

        self.assertEqual(errors, [])

    def test_direct_serial_requires_port_and_baud(self) -> None:
        validator = PreflightValidator(
            RuntimeOptions(
                Path("profile.cfg"),
                Path("setup.json"),
                radar_backend="direct-serial",
            )
        )
        errors = []

        validator._validate_radar_control_settings(
            SimpleNamespace(
                radar_device=SimpleNamespace(control_port=None, baud_rate=None)
            ),
            errors,
        )

        self.assertEqual(len(errors), 2)

    def test_invalid_hex_payload_override_has_startup_error(self) -> None:
        config = SimpleNamespace(
            setup_json={
                "directUdpDCA1000": {
                    "payloads": {"CONFIG_RECORD": "not-hex"}
                }
            }
        )

        with self.assertRaisesRegex(StartupError, "not valid hex"):
            _dca1000_payload_override(config, DCA1000Command.CONFIG_RECORD)


if __name__ == "__main__":
    unittest.main()
