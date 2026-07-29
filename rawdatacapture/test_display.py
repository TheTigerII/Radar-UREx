import os
import queue
import threading
import unittest

import numpy as np

from rawdatacapture.livedatacapture import (
    CombinedDisplayPayload,
    PointCloudDisplayPayload,
    TargetTrack,
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


@unittest.skipUnless(GUI_AVAILABLE, "PyQtGraph/PySide6 is not installed")
class PyQtGraphDisplayTests(unittest.TestCase):
    def _display(self, mode: str):
        from rawdatacapture.livedatacapture import _PyQtGraphDisplay

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
        finally:
            self._close(display)

    def test_range_doppler_display_updates_image(self) -> None:
        display = self._display("range-doppler")
        try:
            axis = np.linspace(0.3, 20.0, 16, dtype=np.float32)
            display._render((axis, np.ones((64, 16), dtype=np.float32)))
            self.assertEqual(display.range_doppler_image.image.shape, (64, 16))
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
        finally:
            self._close(display)


if __name__ == "__main__":
    unittest.main()
