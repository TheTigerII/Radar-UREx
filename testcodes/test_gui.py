import os
import unittest


os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
try:
    from PySide6 import QtWidgets

    from main.gui import RadarControlCenter

    GUI_AVAILABLE = True
except ImportError:
    GUI_AVAILABLE = False


@unittest.skipUnless(GUI_AVAILABLE, "PySide6 is not installed")
class TouchGuiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def setUp(self) -> None:
        self.window = RadarControlCenter()
        self.window.show()
        self.app.processEvents()

    def tearDown(self) -> None:
        self.window.close()
        self.window.deleteLater()
        self.app.processEvents()

    def test_primary_touch_targets_are_at_least_44_pixels_high(self) -> None:
        controls = (
            self.window.start_button,
            self.window.stop_button,
            self.window.mode,
            self.window.radar_port,
            self.window.duration,
        )
        self.assertTrue(all(control.height() >= 44 for control in controls))

    def test_forms_stack_on_a_narrow_touchscreen(self) -> None:
        self.window.resize(800, 480)
        self.app.processEvents()
        self.assertTrue(
            all(
                form.rowWrapPolicy()
                == QtWidgets.QFormLayout.RowWrapPolicy.WrapAllRows
                for form in self.window._forms
            )
        )
        self.assertEqual(self.window.open_output_button.text(), "Data")

    def test_forms_use_two_columns_on_a_wide_touchscreen(self) -> None:
        self.window.resize(1024, 600)
        self.app.processEvents()
        self.assertTrue(
            all(
                form.rowWrapPolicy()
                == QtWidgets.QFormLayout.RowWrapPolicy.DontWrapRows
                for form in self.window._forms
            )
        )
        self.assertEqual(self.window.open_output_button.text(), "Open data folder")


if __name__ == "__main__":
    unittest.main()
