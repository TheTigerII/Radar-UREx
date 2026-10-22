import os
import queue
import threading
import unittest

import numpy as np

from main.livedatacapture import (
    CombinedDisplayPayload,
    PointCloudDisplayPayload,
    TargetTrack,
    _fixed_width_number,
    _image_levels,
)


os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
try:
    import pyqtgraph as pg
    import pyqtgraph.opengl as gl
    from PySide6 import QtCore, QtWidgets

    GUI_AVAILABLE = True
except ImportError:
    GUI_AVAILABLE = False


class _Counter:
    def __init__(self) -> None:
        self.value = 0
        self.lock = threading.Lock()

    def get_lock(self) -> threading.Lock:
        return self.lock


class ImageLevelTests(unittest.TestCase):
    def test_constant_and_non_finite_images_get_valid_levels(self) -> None:
        self.assertEqual(_image_levels(np.ones((2, 2))), (1.0, 2.0))
        self.assertEqual(_image_levels(np.asarray((np.nan, np.inf))), (0.0, 1.0))

    def test_status_numbers_keep_the_same_width(self) -> None:
        fields = (
            _fixed_width_number(None),
            _fixed_width_number(9.0),
            _fixed_width_number(12_345_678.0),
        )
        self.assertEqual({len(field) for field in fields}, {15})


@unittest.skipUnless(GUI_AVAILABLE, "PyQtGraph/PySide6 is not installed")
class PyQtGraphDisplayTests(unittest.TestCase):
    def _display(self, mode: str):
        from main.livedatacapture import _PyQtGraphDisplay

        return _PyQtGraphDisplay(
            mode=mode,
            max_range_m=20.0,
            fov_deg=60.0,
            payload_queue=queue.Queue(),
            stop_event=threading.Event(),
            rendered_updates=_Counter(),
            pg=pg,
            gl=gl,
            QtCore=QtCore,
            QtWidgets=QtWidgets,
        )

    def _close(self, display) -> None:
        display.timer.stop()
        display.window.close()
        display.window.deleteLater()
        display.app.processEvents()

    def test_range_display_updates_curve(self) -> None:
        display = self._display("range")
        try:
            axis = np.linspace(0.3, 20.0, 16, dtype=np.float32)
            display._render((axis, np.arange(16, dtype=np.float32)))
            self.assertEqual(display.range_curve.xData.size, 16)
            fixed_range = display.range_plot.viewRange()
            display._render((axis, np.arange(16, dtype=np.float32) * 100.0))
            self.assertEqual(display.range_plot.viewRange(), fixed_range)
            self.assertFalse(display.range_plot.getViewBox().mouseEnabled()[0])
            self.assertFalse(display.range_plot.getViewBox().mouseEnabled()[1])
        finally:
            self._close(display)

    def test_range_doppler_display_updates_image(self) -> None:
        display = self._display("range-doppler")
        try:
            axis = np.linspace(0.3, 20.0, 16, dtype=np.float32)
            display._render((axis, np.ones((64, 16), dtype=np.float32)))
            self.assertEqual(display.range_doppler_image.image.shape, (64, 16))
            lookup_table = display.range_doppler_image.lut
            self.assertEqual(lookup_table.shape[0], 256)
            self.assertFalse(np.array_equal(lookup_table[0], lookup_table[-1]))
        finally:
            self._close(display)

    def test_point_cloud_display_updates_target(self) -> None:
        display = self._display("point-cloud")
        try:
            payload = PointCloudDisplayPayload(
                TargetTrack((1.0, 5.0, 2.0)),
                "PMM target — confirmed",
            )
            display._render(payload)
            self.assertEqual(display.target_scatter.pos.shape, (1, 3))
            self.assertIn("confirmed", display.point_status.text())
            self.assertEqual(
                display.point_status.sizePolicy().horizontalPolicy(),
                QtWidgets.QSizePolicy.Policy.Ignored,
            )
            self.assertTrue(
                display.point_view.testAttribute(
                    QtCore.Qt.WidgetAttribute.WA_AcceptTouchEvents
                )
            )
        finally:
            self._close(display)

    def test_point_cloud_camera_supports_touch_navigation_and_reset(self) -> None:
        display = self._display("point-cloud")
        try:
            view = display.point_view
            initial_azimuth = float(view.opts["azimuth"])
            initial_elevation = float(view.opts["elevation"])
            initial_distance = float(view.opts["distance"])
            initial_center = (
                float(view.opts["center"].x()),
                float(view.opts["center"].y()),
                float(view.opts["center"].z()),
            )

            view._apply_orbit_delta(20.0, -10.0)
            self.assertNotEqual(float(view.opts["azimuth"]), initial_azimuth)
            self.assertNotEqual(float(view.opts["elevation"]), initial_elevation)

            view._apply_pan_delta(25.0, 15.0)
            moved_center = (
                float(view.opts["center"].x()),
                float(view.opts["center"].y()),
                float(view.opts["center"].z()),
            )
            self.assertNotEqual(moved_center, initial_center)

            view._apply_zoom_factor(1.25)
            self.assertLess(float(view.opts["distance"]), initial_distance)

            display.reset_view_button.click()
            self.assertAlmostEqual(float(view.opts["azimuth"]), initial_azimuth)
            self.assertAlmostEqual(float(view.opts["elevation"]), initial_elevation)
            self.assertAlmostEqual(float(view.opts["distance"]), initial_distance)
            reset_center = (
                float(view.opts["center"].x()),
                float(view.opts["center"].y()),
                float(view.opts["center"].z()),
            )
            self.assertEqual(reset_center, initial_center)
        finally:
            self._close(display)

    def test_combined_display_updates_target_and_spectrum(self) -> None:
        display = self._display("combined")
        try:
            point = PointCloudDisplayPayload(
                TargetTrack((1.0, 5.0, 2.0)),
                "PMM target — confirmed",
            )
            display._render(
                CombinedDisplayPayload(
                    point,
                    np.ones((64, 36), dtype=np.float32),
                )
            )
            self.assertEqual(display.target_scatter.pos.shape, (1, 3))
            self.assertEqual(display.doppler_time_image.image.shape, (64, 36))
            lookup_table = display.doppler_time_image.lut
            self.assertEqual(lookup_table.shape[0], 256)
            self.assertFalse(np.array_equal(lookup_table[0], lookup_table[-1]))
        finally:
            self._close(display)


if __name__ == "__main__":
    unittest.main()
