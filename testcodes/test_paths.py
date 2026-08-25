import unittest
from pathlib import Path

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
        self.assertEqual(
            livedatacapture.DEFAULT_LOG_PATH,
            REPOSITORY_ROOT / "log" / "livedatacapture.log",
        )

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
