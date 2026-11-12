import os
import queue
import threading

import pytest


os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


class _Counter:
    value = 0

    def get_lock(self) -> threading.Lock:
        return threading.Lock()


def test_ppi_renders_target_and_altitude_status() -> None:
    pg = pytest.importorskip("pyqtgraph")
    gl = pytest.importorskip("pyqtgraph.opengl")
    QtCore = pytest.importorskip("PySide6.QtCore")
    QtWidgets = pytest.importorskip("PySide6.QtWidgets")
    from main.livedatacapture import (
        PointCloudDisplayPayload,
        TargetTrack,
        _PyQtGraphDisplay,
    )

    display = _PyQtGraphDisplay(
        mode="ppi",
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
    try:
        display._render(
            PointCloudDisplayPayload(
                target_track=None,
                tracking_status="PMM target — calibrating",
                tracking_state="calibrating",
                calibration_frames_seen=125,
                calibration_frames_required=300,
            )
        )
        assert display.ppi_readiness.text() == "CALIBRATING — 42%"
        assert "KEEP THE RADAR VIEW CLEAR" in (
            display.ppi_calibration_instruction.text()
        )
        assert display.ppi_calibration_progress.value() == 125
        display._render(
            PointCloudDisplayPayload(
                target_track=TargetTrack((3.0, 12.0, 2.5)),
                tracking_status=(
                    "SIMO target — confirmed\nrange 12.62 m\n"
                    "az +14.0° | el +11.4° | altitude +2.50 m"
                ),
                tracking_state="confirmed",
                calibration_frames_seen=300,
                calibration_frames_required=300,
                classification_annotation="UAV / DRONE  91%",
                classification_colour="#fb7185",
            )
        )
        assert display.ppi_trail == [(3.0, 12.0)]
        assert "altitude +2.50 m" in display.ppi_status.text()
        assert display.ppi_readiness.text() == "RADAR READY — TARGET TRACKED"
        assert display.ppi_calibration_progress.format() == (
            "Calibration complete"
        )
        assert display.ppi_target_label.toPlainText() == "UAV / DRONE  91%"
        assert display.ppi_target_label.isVisible()
    finally:
        display.timer.stop()
        display.window.close()
        display.window.deleteLater()
        display.app.processEvents()


def test_gui_exposes_four_range_test_modes() -> None:
    QtWidgets = pytest.importorskip("PySide6.QtWidgets")
    from main import run
    from main.gui import RadarControlCenter

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    window = RadarControlCenter()
    try:
        assert window.operation.count() == 4
        assert [
            window.operation.itemData(index)
            for index in range(window.operation.count())
        ] == [
            "conventional",
            "baseline",
            "fixed-beam",
            "conventional-enhanced",
        ]
        # Hidden modes must remain runnable from the command line.
        assert "phased-array-stt" in run.EXPERIMENT_OPERATION_MODES
        assert run.LITERATURE_TBD_OPERATION_MODE in run.EXPERIMENT_OPERATION_MODES
        assert window.operation.currentData() == "fixed-beam"
        assert window.config.required_path() == run.FIXED_BEAM_CONFIG_PATH
        assert window.mode.currentData() == "ppi"
        assert window.stop_all_button.text() == "Stop all radar processes"
        assert window.stop_all_button.isEnabled()
        assert not window.enhanced_stages.isEnabled()

        from main.gui_config import CONVENTIONAL_CONFIG_PATH, CONVENTIONAL_SETUP_PATH

        window.operation.setCurrentIndex(3)
        assert window.operation.currentData() == "conventional-enhanced"
        assert window.enhanced_stages.isEnabled()
        assert window.config.required_path() == CONVENTIONAL_CONFIG_PATH
        assert window.setup.required_path() == CONVENTIONAL_SETUP_PATH
        assert window.max_range.value() == 12.0
        settings = window._settings()
        assert settings.operation_mode == "conventional-enhanced"
        assert settings.enhanced_detection_arguments() == [
            "--detection-power", "coherent-beam",
            "--clutter-reference", "mean",
            "--cfar-threshold", "calibrated",
            "--tbd-window-frames", "8",
        ]
        window.enhanced_coherent_beams.setChecked(False)
        assert "--detection-power" not in (
            window._settings().enhanced_detection_arguments()
        )

        window.operation.setCurrentIndex(0)
        assert not window.enhanced_stages.isEnabled()
        assert window._settings().enhanced_detection_arguments() == []
    finally:
        window.close()
        window.deleteLater()
        app.processEvents()
