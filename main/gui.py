"""Native desktop control center for Radar-UREx."""

from __future__ import annotations

import os
from pathlib import Path
import shlex
import shutil
import signal
import socket
import sys

if __package__ in {None, ""}:
    repository_root = str(Path(__file__).resolve().parent.parent)
    if repository_root not in sys.path:
        sys.path.insert(0, repository_root)

from PySide6 import QtCore, QtGui, QtWidgets

from main import calibrate, run
from main.gui_config import GuiSettings


ROOT = Path(__file__).resolve().parent.parent
TOUCH_BREAKPOINT_PX = 900


class PathPicker(QtWidgets.QWidget):
    """A line edit with a native file or directory picker."""

    def __init__(
        self,
        path: Path | None,
        *,
        directory: bool = False,
        save: bool = False,
        file_filter: str = "All files (*)",
    ) -> None:
        super().__init__()
        self.directory = directory
        self.save = save
        self.file_filter = file_filter
        self.edit = QtWidgets.QLineEdit("" if path is None else str(path))
        self.button = QtWidgets.QToolButton()
        self.button.setText("Browse…")
        self.button.clicked.connect(self._browse)
        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.edit, 1)
        layout.addWidget(self.button)

    def optional_path(self) -> Path | None:
        text = self.edit.text().strip()
        return Path(text).expanduser() if text else None

    def required_path(self) -> Path:
        return self.optional_path() or Path("")

    def _browse(self) -> None:
        current = self.edit.text().strip() or str(ROOT)
        if self.directory:
            selected = QtWidgets.QFileDialog.getExistingDirectory(
                self, "Choose directory", current
            )
        elif self.save:
            selected, _ = QtWidgets.QFileDialog.getSaveFileName(
                self, "Choose output file", current, self.file_filter
            )
        else:
            selected, _ = QtWidgets.QFileDialog.getOpenFileName(
                self, "Choose file", current, self.file_filter
            )
        if selected:
            self.edit.setText(selected)


def _spin(value: int, minimum: int, maximum: int) -> QtWidgets.QSpinBox:
    widget = QtWidgets.QSpinBox()
    widget.setRange(minimum, maximum)
    widget.setValue(value)
    return widget


def _double_spin(
    value: float,
    minimum: float,
    maximum: float,
    *,
    decimals: int = 2,
    suffix: str = "",
) -> QtWidgets.QDoubleSpinBox:
    widget = QtWidgets.QDoubleSpinBox()
    widget.setRange(minimum, maximum)
    widget.setDecimals(decimals)
    widget.setValue(value)
    widget.setSuffix(suffix)
    widget.setSingleStep(0.1)
    return widget


class RadarControlCenter(QtWidgets.QMainWindow):
    """Configure and supervise the existing integrated launcher."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Radar-UREx Control Center")
        self.setMinimumSize(720, 480)
        self.resize(1100, 760)
        self.process: QtCore.QProcess | None = None
        self.network_process: QtCore.QProcess | None = None
        self._network_steps: list[list[str]] = []
        self._forms: list[QtWidgets.QFormLayout] = []
        self._output_tail = ""
        self._calibration_question_answered = False
        self._close_when_finished = False

        self._build_ui()
        self._refresh_ports()
        self._refresh_interfaces()
        self._mode_changed()
        self._classification_changed()
        self._set_running(False)
        self._apply_responsive_layout(self.width())

        fullscreen_shortcut = QtGui.QShortcut(QtGui.QKeySequence("F11"), self)
        fullscreen_shortcut.activated.connect(self._toggle_fullscreen)

    def _build_ui(self) -> None:
        central = QtWidgets.QWidget()
        outer = QtWidgets.QVBoxLayout(central)

        title_row = QtWidgets.QHBoxLayout()
        title = QtWidgets.QLabel("Radar-UREx")
        title.setObjectName("title")
        subtitle = QtWidgets.QLabel("IWR6843 + DCA1000 PMM control center")
        subtitle.setObjectName("subtitle")
        title_column = QtWidgets.QVBoxLayout()
        title_column.setSpacing(1)
        title_column.addWidget(title)
        title_column.addWidget(subtitle)
        self.status = QtWidgets.QLabel("Ready")
        self.status.setObjectName("status")
        self.status.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.status.setMinimumWidth(150)
        self.fullscreen_button = QtWidgets.QPushButton("Full screen")
        self.fullscreen_button.setToolTip("Toggle full screen (F11)")
        self.fullscreen_button.clicked.connect(self._toggle_fullscreen)
        title_row.addLayout(title_column)
        title_row.addStretch()
        title_row.addWidget(self.status)
        title_row.addWidget(self.fullscreen_button)
        outer.addLayout(title_row)

        self.tabs = QtWidgets.QTabWidget()
        self.tabs.addTab(self._build_run_tab(), "Run")
        self.tabs.addTab(self._build_connection_tab(), "Hardware && files")
        self.tabs.addTab(self._build_tracking_tab(), "Tracking")
        self.tabs.addTab(self._build_activity_tab(), "Activity")
        outer.addWidget(self.tabs, 1)

        controls = QtWidgets.QHBoxLayout()
        self.open_output_button = QtWidgets.QPushButton("Open data folder")
        self.open_output_button.clicked.connect(self._open_output_folder)
        self.start_button = QtWidgets.QPushButton("Start radar")
        self.start_button.setObjectName("startButton")
        self.start_button.clicked.connect(self._start)
        self.stop_button = QtWidgets.QPushButton("Stop")
        self.stop_button.setObjectName("stopButton")
        self.stop_button.clicked.connect(self._stop)
        self.start_button.setMinimumWidth(170)
        self.stop_button.setMinimumWidth(130)
        controls.addWidget(self.open_output_button)
        controls.addStretch()
        controls.addWidget(self.stop_button)
        controls.addWidget(self.start_button)
        outer.addLayout(controls)
        self.setCentralWidget(central)

        self.setStyleSheet(
            """
            QMainWindow { background: #111827; }
            QWidget { color: #e5e7eb; font-size: 15px; }
            QLabel#title { font-size: 28px; font-weight: 700; color: #f9fafb; }
            QLabel#subtitle { color: #9ca3af; }
            QLabel#status { min-height: 34px; padding: 5px 14px; border-radius: 12px;
                            background: #1f2937; color: #93c5fd; font-weight: 600; }
            QTabWidget::pane, QGroupBox { border: 1px solid #374151;
                                         border-radius: 7px; background: #172033; }
            QGroupBox { margin-top: 14px; padding: 12px 8px 8px 8px; font-weight: 600; }
            QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 5px; }
            QTabBar::tab { background: #1f2937; min-height: 48px; min-width: 100px;
                           padding: 3px 14px; }
            QTabBar::tab:selected { background: #2563eb; }
            QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox {
                min-height: 46px;
                background: #0f172a; border: 1px solid #475569; border-radius: 4px;
                padding: 0 10px; selection-background-color: #2563eb;
            }
            QPlainTextEdit { background: #0f172a; border: 1px solid #475569;
                             border-radius: 4px; padding: 7px;
                             selection-background-color: #2563eb; }
            QComboBox::drop-down { width: 42px; border-left: 1px solid #475569; }
            QComboBox QAbstractItemView { background: #0f172a; selection-background-color: #2563eb;
                                          padding: 4px; outline: 0; }
            QComboBox QAbstractItemView::item { min-height: 46px; padding: 5px 10px; }
            QSpinBox::up-button, QDoubleSpinBox::up-button,
            QSpinBox::down-button, QDoubleSpinBox::down-button {
                width: 42px; background: #334155; border-left: 1px solid #475569;
            }
            QSpinBox::up-button, QDoubleSpinBox::up-button {
                subcontrol-origin: border; subcontrol-position: top right;
                border-bottom: 1px solid #475569;
            }
            QSpinBox::down-button, QDoubleSpinBox::down-button {
                subcontrol-origin: border; subcontrol-position: bottom right;
            }
            QCheckBox { min-height: 44px; spacing: 12px; }
            QCheckBox::indicator { width: 28px; height: 28px; }
            QPushButton, QToolButton { min-height: 46px; background: #334155; border: 0;
                                      border-radius: 6px; padding: 2px 16px; }
            QToolButton { min-width: 92px; }
            QPushButton:hover, QToolButton:hover { background: #475569; }
            QPushButton:pressed, QToolButton:pressed { background: #64748b; }
            QPushButton#startButton { min-height: 54px; background: #2563eb;
                                      font-weight: 700; font-size: 17px; }
            QPushButton#startButton:hover { background: #1d4ed8; }
            QPushButton#stopButton { min-height: 54px; background: #b91c1c;
                                     font-weight: 700; font-size: 17px; }
            QPushButton#stopButton:disabled { color: #6b7280; background: #1f2937; }
            QPushButton:disabled { color: #6b7280; background: #1f2937; }
            QScrollBar:vertical { background: #111827; width: 22px; margin: 0; }
            QScrollBar::handle:vertical { background: #64748b; min-height: 48px;
                                          border-radius: 8px; margin: 2px; }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }
            QScrollBar:horizontal { background: #111827; height: 22px; margin: 0; }
            QScrollBar::handle:horizontal { background: #64748b; min-width: 48px;
                                            border-radius: 8px; margin: 2px; }
            QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal { width: 0; }
            """
        )

    def _form(self, parent: QtWidgets.QWidget) -> QtWidgets.QFormLayout:
        form = QtWidgets.QFormLayout(parent)
        form.setFieldGrowthPolicy(
            QtWidgets.QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow
        )
        form.setLabelAlignment(
            QtCore.Qt.AlignmentFlag.AlignLeft | QtCore.Qt.AlignmentFlag.AlignVCenter
        )
        form.setHorizontalSpacing(18)
        form.setVerticalSpacing(10)
        self._forms.append(form)
        return form

    def _scroll_form(self, content: QtWidgets.QWidget) -> QtWidgets.QScrollArea:
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(
            QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        scroll.setWidget(content)
        scroll.verticalScrollBar().setSingleStep(24)
        QtWidgets.QScroller.grabGesture(
            scroll.viewport(),
            QtWidgets.QScroller.ScrollerGestureType.TouchGesture,
        )
        return scroll

    def _build_run_tab(self) -> QtWidgets.QWidget:
        content = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(content)

        primary = QtWidgets.QGroupBox("Capture")
        form = self._form(primary)
        self.mode = QtWidgets.QComboBox()
        mode_labels = {
            "none": "Headless (no plot)",
            "range": "Range plot",
            "range-doppler": "Range-Doppler plot",
            "point-cloud": "3D point cloud",
            "combined": "Combined dashboard",
            "calibration": "Range / RX calibration",
            "azimuth-calibration": "Azimuth calibration",
            "elevation-calibration": "Elevation calibration",
        }
        for value in run.DISPLAY_CHOICES:
            self.mode.addItem(mode_labels[value], value)
        self.mode.setCurrentIndex(self.mode.findData("combined"))
        self.mode.currentIndexChanged.connect(self._mode_changed)
        form.addRow("Mode", self.mode)

        port_row = QtWidgets.QWidget()
        port_layout = QtWidgets.QHBoxLayout(port_row)
        port_layout.setContentsMargins(0, 0, 0, 0)
        self.radar_port = QtWidgets.QComboBox()
        self.radar_port.setEditable(True)
        self.refresh_ports_button = QtWidgets.QPushButton("Refresh")
        self.refresh_ports_button.clicked.connect(self._refresh_ports)
        port_layout.addWidget(self.radar_port, 1)
        port_layout.addWidget(self.refresh_ports_button)
        form.addRow("Radar command UART", port_row)

        self.duration = _double_spin(5.0, 0.0, 10080.0, suffix=" min")
        self.duration.setSpecialValueText("Unlimited")
        form.addRow("Run time after calibration", self.duration)
        self.dataset_destination = QtWidgets.QComboBox()
        self.dataset_destination.addItem("Unlabelled — dataset/", "dataset")
        self.dataset_destination.addItem("Known UAV — dataset/uav/", "uav")
        self.dataset_destination.addItem("Known other — dataset/other/", "other")
        form.addRow("Dataset destination", self.dataset_destination)
        layout.addWidget(primary)

        self.classification_group = QtWidgets.QGroupBox("Real-time classification")
        class_form = self._form(self.classification_group)
        self.classification = QtWidgets.QCheckBox("Enable TensorRT UAV classification")
        self.classification.toggled.connect(self._classification_changed)
        class_form.addRow(self.classification)
        self.model_weights = PathPicker(run.DEFAULT_MODEL_WEIGHTS_DIR, directory=True)
        class_form.addRow("Model directory", self.model_weights)
        self.inference_logging = QtWidgets.QCheckBox("Write evaluation log")
        self.inference_logging.toggled.connect(self._classification_changed)
        class_form.addRow(self.inference_logging)
        self.evaluation_label = QtWidgets.QComboBox()
        self.evaluation_label.addItem("Unlabelled", "unlabeled")
        self.evaluation_label.addItem("Drone / UAV", "drone")
        self.evaluation_label.addItem("Not a drone", "not_drone")
        class_form.addRow("Ground truth", self.evaluation_label)
        self.inference_log = PathPicker(
            None, save=True, file_filter="JSON Lines (*.jsonl);;All files (*)"
        )
        class_form.addRow("Evaluation log (optional)", self.inference_log)
        layout.addWidget(self.classification_group)

        self.calibration_group = QtWidgets.QGroupBox("Hardware calibration")
        calibration_form = self._form(self.calibration_group)
        self.calibration_distance = _double_spin(1.0, 0.01, 100.0, suffix=" m")
        self.calibration_angle = _double_spin(0.0, -60.0, 60.0, suffix="°")
        self.calibration_window = _double_spin(0.2, 0.01, 10.0, suffix=" m")
        self.calibration_warmup = _spin(16, 0, 10000)
        self.calibration_frames = _spin(64, 1, 100000)
        self.calibration_timeout = _double_spin(90.0, 1.0, 3600.0, suffix=" s")
        self.calibration_output = PathPicker(
            None, save=True, file_filter="JSON (*.json);;All files (*)"
        )
        calibration_form.addRow("Measured reflector distance", self.calibration_distance)
        calibration_form.addRow("Known tripod angle", self.calibration_angle)
        calibration_form.addRow("Range search window", self.calibration_window)
        calibration_form.addRow("Warm-up frames", self.calibration_warmup)
        calibration_form.addRow("Accepted frames", self.calibration_frames)
        calibration_form.addRow("Timeout", self.calibration_timeout)
        calibration_form.addRow("Report path (optional)", self.calibration_output)
        layout.addWidget(self.calibration_group)
        layout.addStretch()
        return self._scroll_form(content)

    def _build_connection_tab(self) -> QtWidgets.QWidget:
        content = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(content)

        paths = QtWidgets.QGroupBox("Profiles and output")
        path_form = self._form(paths)
        self.config = PathPicker(run.DEFAULT_CONFIG_PATH, file_filter="Radar profile (*.cfg)")
        self.setup = PathPicker(run.DEFAULT_SETUP_PATH, file_filter="JSON (*.json)")
        self.calibration_profile = PathPicker(
            run.DEFAULT_CALIBRATION_PROFILE_PATH, file_filter="Radar profile (*.cfg)"
        )
        self.capture_dir = PathPicker(run.DEFAULT_CAPTURE_DIR, directory=True)
        self.processed_output = PathPicker(
            None, save=True, file_filter="JSON Lines (*.jsonl);;All files (*)"
        )
        self.raw_output = PathPicker(
            None, save=True, file_filter="Raw radar capture (*.bin);;All files (*)"
        )
        path_form.addRow("Operational radar profile", self.config)
        path_form.addRow("DCA1000 setup", self.setup)
        path_form.addRow("Calibration profile", self.calibration_profile)
        path_form.addRow("Capture directory", self.capture_dir)
        path_form.addRow("Exact processed output (optional)", self.processed_output)
        path_form.addRow("Raw ADC output (optional)", self.raw_output)
        layout.addWidget(paths)

        network = QtWidgets.QGroupBox("Connection and buffering")
        form = self._form(network)
        interface_row = QtWidgets.QWidget()
        interface_layout = QtWidgets.QHBoxLayout(interface_row)
        interface_layout.setContentsMargins(0, 0, 0, 0)
        self.network_interface = QtWidgets.QComboBox()
        self.network_interface.setEditable(True)
        self.refresh_interfaces_button = QtWidgets.QPushButton("Refresh")
        self.refresh_interfaces_button.clicked.connect(self._refresh_interfaces)
        interface_layout.addWidget(self.network_interface, 1)
        interface_layout.addWidget(self.refresh_interfaces_button)
        self.host_ip = QtWidgets.QLineEdit(run.DEFAULT_HOST_IP)
        self.configure_network_button = QtWidgets.QPushButton(
            "Configure selected interface for DCA1000"
        )
        self.configure_network_button.clicked.connect(self._configure_dca_network)
        self.data_port = _spin(run.DEFAULT_DATA_PORT, 1, 65535)
        self.radar_baud = _spin(run.DEFAULT_RADAR_BAUD, 1, 4_000_000)
        self.radar_timeout = _double_spin(10.0, 0.1, 300.0, suffix=" s")
        self.dca_timeout = _double_spin(3.0, 0.1, 300.0, suffix=" s")
        self.dca_retries = _spin(5, 0, 100)
        self.socket_buffer = _spin(run.DEFAULT_SOCKET_RECV_BUFFER_BYTES, 0, 2_147_483_647)
        self.packet_queue = _spin(run.DEFAULT_PACKET_QUEUE_SIZE, 1, 10_000_000)
        self.processing_queue = _spin(run.DEFAULT_PROCESSING_QUEUE_SIZE, 1, 1_000_000)
        form.addRow("DCA1000 Ethernet interface", interface_row)
        form.addRow("Host / receiver IP", self.host_ip)
        form.addRow("", self.configure_network_button)
        form.addRow("UDP data port", self.data_port)
        form.addRow("Radar UART baud", self.radar_baud)
        form.addRow("Radar command timeout", self.radar_timeout)
        form.addRow("DCA1000 timeout", self.dca_timeout)
        form.addRow("DCA1000 retries", self.dca_retries)
        form.addRow("Socket receive buffer (bytes)", self.socket_buffer)
        form.addRow("Packet queue size", self.packet_queue)
        form.addRow("Processing queue size", self.processing_queue)
        layout.addWidget(network)
        layout.addStretch()
        return self._scroll_form(content)

    def _build_tracking_tab(self) -> QtWidgets.QWidget:
        content = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(content)
        tracking = QtWidgets.QGroupBox("PMM detection and tracking")
        form = self._form(tracking)
        self.background_seconds = _double_spin(30.0, 0.0, 3600.0, suffix=" s")
        self.max_speed = _double_spin(4.0, 0.0, 100.0, suffix=" m/s")
        self.folding_min = _spin(2, 2, 32)
        self.folding_max = _spin(32, 2, 32)
        self.fixed_threshold_enabled = QtWidgets.QCheckBox("Use fixed threshold")
        self.fixed_threshold = _double_spin(30000.0, 0.0, 1e12, decimals=1)
        self.fixed_threshold.setEnabled(False)
        self.fixed_threshold_enabled.toggled.connect(self.fixed_threshold.setEnabled)
        threshold_row = QtWidgets.QWidget()
        threshold_layout = QtWidgets.QHBoxLayout(threshold_row)
        threshold_layout.setContentsMargins(0, 0, 0, 0)
        threshold_layout.addWidget(self.fixed_threshold_enabled)
        threshold_layout.addWidget(self.fixed_threshold, 1)
        self.threshold_sigma = _double_spin(
            run.MINI4_DEFAULT_ADAPTIVE_THRESHOLD_SIGMA, 0.0, 1000.0
        )
        self.threshold_minimum = _double_spin(
            run.MINI4_DEFAULT_ADAPTIVE_THRESHOLD_MINIMUM, 0.0, 1e12, decimals=1
        )
        self.history_seconds = _double_spin(3.6, 0.1, 3600.0, suffix=" s")
        self.provisional_frames = _spin(5, 1, 100000)
        self.confirmation_window = _spin(10, 1, 100000)
        self.confirmation_hits = _spin(7, 1, 100000)
        self.coast_frames = _spin(10, 0, 100000)
        self.max_range = _double_spin(20.0, 0.3, 20.0, suffix=" m")
        self.display_update = _spin(1, 1, 100000)
        form.addRow("Background calibration", self.background_seconds)
        form.addRow("Maximum target speed", self.max_speed)
        form.addRow("Folding size minimum", self.folding_min)
        form.addRow("Folding size maximum", self.folding_max)
        form.addRow("Detection threshold", threshold_row)
        form.addRow("Adaptive threshold sigma", self.threshold_sigma)
        form.addRow("Adaptive threshold minimum", self.threshold_minimum)
        form.addRow("History", self.history_seconds)
        form.addRow("Provisional frames", self.provisional_frames)
        form.addRow("Confirmation window", self.confirmation_window)
        form.addRow("Confirmation hits", self.confirmation_hits)
        form.addRow("Coast frames", self.coast_frames)
        form.addRow("Maximum displayed range", self.max_range)
        form.addRow("Draw every N frames", self.display_update)
        layout.addWidget(tracking)

        logs = QtWidgets.QGroupBox("Performance logging")
        log_form = self._form(logs)
        self.performance_logging = QtWidgets.QCheckBox("Write runtime performance log")
        self.performance_logging.setChecked(True)
        self.performance_log = PathPicker(
            None, save=True, file_filter="JSON Lines (*.jsonl);;All files (*)"
        )
        self.resource_interval = _double_spin(1.0, 0.1, 3600.0, suffix=" s")
        log_form.addRow(self.performance_logging)
        log_form.addRow("Exact log path (optional)", self.performance_log)
        log_form.addRow("Resource sample interval", self.resource_interval)
        layout.addWidget(logs)
        layout.addStretch()
        return self._scroll_form(content)

    def _build_activity_tab(self) -> QtWidgets.QWidget:
        tab = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(tab)
        hint = QtWidgets.QLabel(
            "Launcher, hardware, TensorRT, capture, and shutdown messages appear here."
        )
        hint.setWordWrap(True)
        self.console = QtWidgets.QPlainTextEdit()
        self.console.setReadOnly(True)
        self.console.setMaximumBlockCount(10000)
        QtWidgets.QScroller.grabGesture(
            self.console.viewport(),
            QtWidgets.QScroller.ScrollerGestureType.TouchGesture,
        )
        self.clear_console_button = QtWidgets.QPushButton("Clear")
        self.clear_console_button.clicked.connect(self.console.clear)
        layout.addWidget(hint)
        layout.addWidget(self.console, 1)
        layout.addWidget(self.clear_console_button, alignment=QtCore.Qt.AlignmentFlag.AlignRight)
        return tab

    def _refresh_ports(self) -> None:
        current = self.radar_port.currentText().strip() if hasattr(self, "radar_port") else ""
        ports = run.list_serial_ports()
        self.radar_port.clear()
        preferred_index = -1
        for index, port in enumerate(ports):
            label = f"{port.device} — {port.description}" if port.description else port.device
            self.radar_port.addItem(label, port.device)
            description = port.description.casefold()
            if all(word in description for word in ("cp2105", "enhanced", "com port")):
                preferred_index = index
        if ports:
            self.radar_port.setCurrentIndex(preferred_index if preferred_index >= 0 else 0)
        else:
            self.radar_port.addItem(
                current or (run.WINDOWS_DEFAULT_RADAR_PORT if os.name == "nt" else run.LINUX_DEFAULT_RADAR_PORT)
            )
        if current and current not in [port.device for port in ports]:
            self.radar_port.setEditText(current)

    def _selected_port(self) -> str:
        data = self.radar_port.currentData()
        text = self.radar_port.currentText().strip()
        return str(data) if data and " — " in text else text

    def _refresh_interfaces(self) -> None:
        current = (
            self.network_interface.currentText().strip()
            if hasattr(self, "network_interface")
            else ""
        )
        self.network_interface.clear()
        try:
            names = [name for _, name in socket.if_nameindex() if name != "lo"]
        except OSError:
            names = []
        self.network_interface.addItems(names)
        if current and current not in names:
            self.network_interface.setEditText(current)

    def _configure_dca_network(self) -> None:
        interface = self.network_interface.currentText().strip()
        host_ip = self.host_ip.text().strip()
        if not interface or not host_ip:
            QtWidgets.QMessageBox.warning(
                self, "Network settings", "Choose an interface and enter the host IP."
            )
            return
        if os.name == "nt":
            QtWidgets.QMessageBox.information(
                self,
                "Network settings",
                "Automatic adapter configuration is currently available on Linux only.",
            )
            return
        pkexec = shutil.which("pkexec")
        ip_command = shutil.which("ip")
        if not pkexec or not ip_command:
            QtWidgets.QMessageBox.warning(
                self,
                "Network settings",
                "The graphical privilege helper (pkexec) or the ip utility is unavailable.",
            )
            return
        self._network_steps = [
            [pkexec, ip_command, "address", "replace", f"{host_ip}/24", "dev", interface],
            [pkexec, ip_command, "link", "set", interface, "up"],
        ]
        self.configure_network_button.setEnabled(False)
        self.tabs.setCurrentIndex(3)
        self.console.appendPlainText(
            f"\nConfiguring {interface} as {host_ip}/24 for the DCA1000…"
        )
        self._run_next_network_step()

    def _run_next_network_step(self) -> None:
        if not self._network_steps:
            self.configure_network_button.setEnabled(True)
            self.console.appendPlainText("DCA1000 Ethernet interface is ready.")
            self.status.setText("Network ready")
            QtWidgets.QMessageBox.information(
                self, "Network ready", "The selected DCA1000 Ethernet interface is configured."
            )
            return
        command = self._network_steps.pop(0)
        self.network_process = QtCore.QProcess(self)
        self.network_process.setProcessChannelMode(
            QtCore.QProcess.ProcessChannelMode.MergedChannels
        )
        self.network_process.readyReadStandardOutput.connect(self._read_network_output)
        self.network_process.finished.connect(self._network_step_finished)
        self.network_process.start(command[0], command[1:])

    def _read_network_output(self) -> None:
        if self.network_process is None:
            return
        text = bytes(self.network_process.readAllStandardOutput()).decode(errors="replace")
        if text:
            self.console.appendPlainText(text.rstrip())

    def _network_step_finished(
        self, exit_code: int, exit_status: QtCore.QProcess.ExitStatus
    ) -> None:
        del exit_status
        if self.network_process is not None:
            self.network_process.deleteLater()
        self.network_process = None
        if exit_code != 0:
            self._network_steps.clear()
            self.configure_network_button.setEnabled(True)
            self.status.setText("Network setup failed")
            QtWidgets.QMessageBox.warning(
                self,
                "Network setup failed",
                "The interface was not configured. Check the Activity tab for details.",
            )
            return
        self._run_next_network_step()

    def _mode_changed(self) -> None:
        display = self.mode.currentData()
        is_calibration = display in calibrate.CALIBRATION_DISPLAY_MODES
        self.calibration_group.setVisible(is_calibration)
        self.classification_group.setVisible(not is_calibration)
        self.duration.setEnabled(not is_calibration)
        self.dataset_destination.setEnabled(not is_calibration)
        self.calibration_angle.setEnabled(
            display in {
                calibrate.AZIMUTH_CALIBRATION_DISPLAY_MODE,
                calibrate.ELEVATION_CALIBRATION_DISPLAY_MODE,
            }
        )

    def _classification_changed(self) -> None:
        enabled = self.classification.isChecked()
        self.model_weights.setEnabled(enabled)
        self.inference_logging.setEnabled(enabled)
        logging_enabled = enabled and self.inference_logging.isChecked()
        self.evaluation_label.setEnabled(logging_enabled)
        self.inference_log.setEnabled(logging_enabled)

    def _settings(self) -> GuiSettings:
        return GuiSettings(
            display=str(self.mode.currentData()),
            radar_port=self._selected_port(),
            duration_minutes=self.duration.value(),
            dataset_destination=str(self.dataset_destination.currentData()),
            config=self.config.required_path(),
            setup=self.setup.required_path(),
            host_ip=self.host_ip.text(),
            data_port=self.data_port.value(),
            radar_baud=self.radar_baud.value(),
            radar_command_timeout=self.radar_timeout.value(),
            dca_timeout=self.dca_timeout.value(),
            dca_retries=self.dca_retries.value(),
            socket_recv_buffer=self.socket_buffer.value(),
            packet_queue_size=self.packet_queue.value(),
            processing_queue_size=self.processing_queue.value(),
            display_update_every=self.display_update.value(),
            max_range_m=self.max_range.value(),
            capture_dir=self.capture_dir.required_path(),
            processed_output=self.processed_output.optional_path(),
            raw_output=self.raw_output.optional_path(),
            classification=self.classification.isChecked(),
            model_weights_dir=self.model_weights.required_path(),
            inference_logging=self.inference_logging.isChecked(),
            inference_log=self.inference_log.optional_path(),
            evaluation_label=str(self.evaluation_label.currentData()),
            performance_logging=self.performance_logging.isChecked(),
            performance_log=self.performance_log.optional_path(),
            resource_sample_interval_seconds=self.resource_interval.value(),
            pmm_background_calibration_seconds=self.background_seconds.value(),
            pmm_max_target_speed_m_s=self.max_speed.value(),
            pmm_folding_size_min=self.folding_min.value(),
            pmm_folding_size_max=self.folding_max.value(),
            pmm_detection_threshold=(
                self.fixed_threshold.value()
                if self.fixed_threshold_enabled.isChecked()
                else None
            ),
            pmm_adaptive_threshold_sigma=self.threshold_sigma.value(),
            pmm_adaptive_threshold_minimum=self.threshold_minimum.value(),
            pmm_history_seconds=self.history_seconds.value(),
            pmm_provisional_frames=self.provisional_frames.value(),
            pmm_confirmation_window_frames=self.confirmation_window.value(),
            pmm_confirmation_hits=self.confirmation_hits.value(),
            pmm_coast_frames=self.coast_frames.value(),
            calibration_profile=self.calibration_profile.required_path(),
            calibration_distance_m=self.calibration_distance.value(),
            calibration_search_window_m=self.calibration_window.value(),
            calibration_warmup_frames=self.calibration_warmup.value(),
            calibration_frames=self.calibration_frames.value(),
            calibration_timeout_seconds=self.calibration_timeout.value(),
            calibration_angle_deg=self.calibration_angle.value(),
            calibration_output=self.calibration_output.optional_path(),
        )

    def _start(self) -> None:
        settings = self._settings()
        errors = settings.validation_errors()
        if errors:
            QtWidgets.QMessageBox.warning(
                self, "Check settings", "Please correct the following:\n\n• " + "\n• ".join(errors)
            )
            return
        command = settings.command()
        self.tabs.setCurrentIndex(3)
        self.console.appendPlainText("\n$ " + shlex.join(command) + "\n")
        self._output_tail = ""
        self._calibration_question_answered = False
        self.process = QtCore.QProcess(self)
        self.process.setWorkingDirectory(str(ROOT))
        self.process.setProcessChannelMode(QtCore.QProcess.ProcessChannelMode.MergedChannels)
        self.process.readyReadStandardOutput.connect(self._read_output)
        self.process.started.connect(self._process_started)
        self.process.errorOccurred.connect(self._process_error)
        self.process.finished.connect(self._process_finished)
        self._set_running(True, "Starting…")
        self.process.start(command[0], command[1:])

    def _read_output(self) -> None:
        if self.process is None:
            return
        text = bytes(self.process.readAllStandardOutput()).decode(errors="replace")
        if not text:
            return
        self.console.moveCursor(QtGui.QTextCursor.MoveOperation.End)
        self.console.insertPlainText(text)
        self.console.moveCursor(QtGui.QTextCursor.MoveOperation.End)
        self._output_tail = (self._output_tail + text)[-4096:]
        if (
            "Apply this calibration to" in self._output_tail
            and not self._calibration_question_answered
        ):
            self._calibration_question_answered = True
            answer = QtWidgets.QMessageBox.question(
                self,
                "Apply calibration?",
                "The calibration passed. Apply it to the operational radar profile?\n\n"
                "A timestamped backup will be created first.",
                QtWidgets.QMessageBox.StandardButton.Yes
                | QtWidgets.QMessageBox.StandardButton.No,
                QtWidgets.QMessageBox.StandardButton.No,
            )
            response = b"y\n" if answer == QtWidgets.QMessageBox.StandardButton.Yes else b"n\n"
            self.process.write(response)

    def _process_started(self) -> None:
        self._set_running(True, "Running")

    def _process_error(self, error: QtCore.QProcess.ProcessError) -> None:
        if self.process is not None:
            self.console.appendPlainText(f"\nLauncher error: {self.process.errorString()}")
        if error == QtCore.QProcess.ProcessError.FailedToStart:
            self._set_running(False, "Could not start")

    def _process_finished(
        self, exit_code: int, exit_status: QtCore.QProcess.ExitStatus
    ) -> None:
        del exit_status
        self.console.appendPlainText(f"\nProcess finished with exit code {exit_code}.")
        label = "Completed" if exit_code == 0 else f"Stopped (exit {exit_code})"
        self._set_running(False, label)
        if self.process is not None:
            self.process.deleteLater()
        self.process = None
        if self._close_when_finished:
            QtCore.QTimer.singleShot(0, self.close)

    def _set_running(self, running: bool, status: str | None = None) -> None:
        self.start_button.setEnabled(not running)
        self.stop_button.setEnabled(running)
        self.refresh_ports_button.setEnabled(not running)
        self.refresh_interfaces_button.setEnabled(not running)
        self.configure_network_button.setEnabled(not running)
        self.tabs.setTabEnabled(0, not running)
        self.tabs.setTabEnabled(1, not running)
        self.tabs.setTabEnabled(2, not running)
        self.status.setText(status or ("Running" if running else "Ready"))

    def _toggle_fullscreen(self) -> None:
        if self.isFullScreen():
            self.showMaximized()
            self.fullscreen_button.setText("Full screen")
        else:
            self.showFullScreen()
            self.fullscreen_button.setText("Exit full screen")

    def _apply_responsive_layout(self, width: int) -> None:
        wrap = width < TOUCH_BREAKPOINT_PX
        policy = (
            QtWidgets.QFormLayout.RowWrapPolicy.WrapAllRows
            if wrap
            else QtWidgets.QFormLayout.RowWrapPolicy.DontWrapRows
        )
        for form in self._forms:
            form.setRowWrapPolicy(policy)
        self.tabs.tabBar().setExpanding(wrap)
        self.open_output_button.setText("Data" if wrap else "Open data folder")
        self.fullscreen_button.setText(
            ("Exit" if wrap else "Exit full screen")
            if self.isFullScreen()
            else ("Full" if wrap else "Full screen")
        )

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:  # noqa: N802
        super().resizeEvent(event)
        self._apply_responsive_layout(event.size().width())

    def _stop(self) -> None:
        if self.process is None or self.process.state() == QtCore.QProcess.ProcessState.NotRunning:
            return
        self.status.setText("Stopping…")
        pid = self.process.processId()
        if os.name != "nt" and pid:
            try:
                os.kill(pid, signal.SIGINT)
            except ProcessLookupError:
                pass
        else:
            self.process.terminate()
        QtCore.QTimer.singleShot(10000, self._force_stop_if_needed)

    def _force_stop_if_needed(self) -> None:
        if self.process is not None and self.process.state() != QtCore.QProcess.ProcessState.NotRunning:
            self.console.appendPlainText("\nShutdown timed out; forcing the launcher to stop.")
            self.process.kill()

    def _open_output_folder(self) -> None:
        exact_output = self.processed_output.optional_path()
        path = (
            exact_output.parent
            if exact_output is not None
            else self.capture_dir.optional_path() or run.DEFAULT_CAPTURE_DIR
        )
        path.mkdir(parents=True, exist_ok=True)
        QtGui.QDesktopServices.openUrl(QtCore.QUrl.fromLocalFile(str(path.resolve())))

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:  # noqa: N802
        if self.process is not None and self.process.state() != QtCore.QProcess.ProcessState.NotRunning:
            choice = QtWidgets.QMessageBox.question(
                self,
                "Stop radar?",
                "A radar session is still running. Stop it and close the application?",
                QtWidgets.QMessageBox.StandardButton.Yes
                | QtWidgets.QMessageBox.StandardButton.Cancel,
                QtWidgets.QMessageBox.StandardButton.Cancel,
            )
            if choice != QtWidgets.QMessageBox.StandardButton.Yes:
                event.ignore()
                return
            self._close_when_finished = True
            self._stop()
            event.ignore()
            return
        event.accept()


def main() -> int:
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    app.setApplicationName("Radar-UREx Control Center")
    app.setOrganizationName("Radar-UREx")
    window = RadarControlCenter()
    window.showMaximized()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
