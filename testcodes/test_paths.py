import unittest
from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory

from main import livedatacapture, replay_pmm, run, startup


REPOSITORY_ROOT = Path(__file__).resolve().parent.parent


class RepositoryPathTests(unittest.TestCase):
    def test_application_defaults_follow_reorganized_directories(self) -> None:
        profile = REPOSITORY_ROOT / "profiles" / "profile-mini4-20m.cfg"
        setup = REPOSITORY_ROOT / "profiles" / "setup.json"

        self.assertEqual(run.DEFAULT_CONFIG_PATH, profile)
        self.assertEqual(run.DEFAULT_SETUP_PATH, setup)
        self.assertEqual(livedatacapture.DEFAULT_CONFIG_PATH, profile)
        self.assertEqual(livedatacapture.DEFAULT_SETUP_PATH, setup)
        self.assertEqual(replay_pmm.DEFAULT_CONFIG_PATH, profile)
        self.assertEqual(startup.DEFAULT_SDK_PROFILE_PATH, profile)

    def test_generated_files_default_to_data_and_log_directories(self) -> None:
        self.assertEqual(run.DEFAULT_CAPTURE_DIR, REPOSITORY_ROOT / "dataset")
        now = datetime(2026, 8, 28, 13, 45, 12, 123456, tzinfo=timezone.utc)
        first = livedatacapture.default_terminal_log_path(now)
        second = livedatacapture.default_terminal_log_path(now)

        self.assertEqual(first.parent, REPOSITORY_ROOT / "log")
        self.assertRegex(
            first.name,
            r"^livedatacapture_20260828_134512_123456_[0-9a-f]{8}\.log$",
        )
        self.assertNotEqual(first, second)

    def test_terminal_log_is_created_exclusively(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "capture.log"
            try:
                resolved = livedatacapture.setup_terminal_log(path)
                livedatacapture.emit("one run")
            finally:
                livedatacapture.close_terminal_log()

            self.assertEqual(resolved, path)
            self.assertEqual(path.read_text(encoding="utf-8"), "one run\n")
            with self.assertRaises(FileExistsError):
                livedatacapture.setup_terminal_log(path)

    def test_relocated_profile_names_resolve_from_any_working_directory(self) -> None:
        expected = REPOSITORY_ROOT / "profiles" / "profile-mini4-20m.cfg"

        self.assertEqual(
            livedatacapture._resolve_config_path(Path("profile-mini4-20m.cfg")),
            expected,
        )
        self.assertEqual(
            startup._resolve_existing_path(Path("profile-mini4-20m.cfg")),
            expected,
        )


if __name__ == "__main__":
    unittest.main()
