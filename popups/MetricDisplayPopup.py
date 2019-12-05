from PyQt5 import QtWidgets, QtGui, QtCore, uic
from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt4agg import NavigationToolbar2QT as NavigationToolbar
import matplotlib
from matplotlib.figure import Figure
import os
import pandas as pd
from scipy.signal import savgol_filter

from typing import Optional, List, Tuple, Dict

class ToolBar(NavigationToolbar):
    selecting_frame: bool = False

    mouse_move_signal = QtCore.pyqtSignal(object)
    frame_select_signal = QtCore.pyqtSignal(bool)

    def __init__(self, canvas, parent, coordinates=True):
        super(ToolBar, self).__init__(canvas, parent, coordinates=coordinates)

    def _init_toolbar(self):
        self.basedir = os.path.join(matplotlib.rcParams['datapath'], 'images')
        self.toolitems = list(self.toolitems)
        del self.toolitems[7]
        self.toolitems.insert(6, ("Frame", "Go to frame", "move", "frame"))
        self.toolitems.insert(7, tuple([None for i in range(4)]))

        for text, tooltip_text, image_file, callback in self.toolitems:
            if text is None:
                self.addSeparator()
            else:
                a = self.addAction(self._icon(image_file + '.png'),
                                   text, getattr(self, callback))
                self._actions[callback] = a
                if callback in ['zoom', 'pan', 'frame']:
                    a.setCheckable(True)
                if tooltip_text is not None:
                    a.setToolTip(tooltip_text)
                if text == 'Subplots':
                    a = self.addAction(self._icon("qt4_editor_options.png"),
                                       'Customize', self.edit_parameters)
                    a.setToolTip('Edit axis, curve and image parameters')

        # Add the x,y location widget at the right side of the toolbar
        # The stretch factor is 1 which means any resizing of the toolbar
        # will resize this label instead of the buttons.
        if self.coordinates:
            self.locLabel = QtWidgets.QLabel("", self)
            self.locLabel.setAlignment(
                QtCore.Qt.AlignRight | QtCore.Qt.AlignTop)
            self.locLabel.setSizePolicy(
                QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding,
                                      QtWidgets.QSizePolicy.Ignored))
            labelAction = self.addWidget(self.locLabel)
            labelAction.setVisible(True)

        # Esthetic adjustments - we need to set these explicitly in PyQt5
        # otherwise the layout looks different - but we don't want to set it if
        # not using HiDPI icons otherwise they look worse than before.
        if self.canvas._dpi_ratio > 1:
            self.setIconSize(QtCore.QSize(24, 24))
            self.layout().setSpacing(12)

    def _update_buttons_checked(self):
        # sync button checkstates to match active mode
        self._actions['pan'].setChecked(self._active == 'PAN')
        self._actions['zoom'].setChecked(self._active == 'ZOOM')
        self._actions['frame'].setChecked(self._active == 'FRAME')
        self.selecting_frame = self._active == "FRAME"

    def frame(self):
        if self._active == 'FRAME':
            self._active = None
        else:
            self._active = 'FRAME'
            self.frame_select_signal.emit(False)
        self._update_buttons_checked()

    def set_message(self, s):
        try:
            split = s.split("=")
            frame = int(round(float(split[-2][:-1])))
            val = float(split[-1])
            pos = (frame, val)
            super().set_message(f"Frame: {frame}")
            self.mouse_move_signal.emit(pos)
        except Exception:
            super().set_message(s)


class Canvas(FigureCanvas):
    mouse_clicked_signal = QtCore.pyqtSignal(object)

    def __init__(self, figure):
        super(Canvas, self).__init__(figure)

    def mousePressEvent(self, event):
        button = self.buttond.get(event.button())
        self.mouse_clicked_signal.emit(button)
        super().mousePressEvent(event)


class MetricDisplayWindow(QtWidgets.QMainWindow):
    main_container: QtWidgets.QVBoxLayout
    metric_container: QtWidgets.QHBoxLayout
    normalize_checkbox: QtWidgets.QCheckBox
    subtract_checkbox: QtWidgets.QCheckBox

    _subtract: bool = False
    _normalize: bool = False

    _metrics: pd.DataFrame
    _metric_checkboxes: Dict[str, QtWidgets.QCheckBox] = None
    _figure: Figure
    _canvas: FigureCanvas
    _toolbar: NavigationToolbar
    _mouse_pos: Tuple[float, float] = (-1, -1)
    row: List[str] = ""

    curr_display: int = 0
    num_metrics: int = 0

    colors = ["b", "g", "r", "c", "m", "y", "k", "w"]

    ax = None

    select_frame_signal = QtCore.pyqtSignal(int)

    def __init__(self, parent=None, metrics: Optional[pd.DataFrame]=None):
        super(MetricDisplayWindow, self).__init__(parent=parent)
        uic.loadUi("uis/MetricViewerPopup.ui", self)

        self._metrics = metrics
        self._figure = Figure()
        self._canvas = Canvas(self._figure)
        self._toolbar = ToolBar(self._canvas, self)

        self._toolbar.mouse_move_signal.connect(self.on_mouse_move)
        self._canvas.mouse_clicked_signal.connect(self.on_click)

        self.normalize_checkbox.stateChanged.connect(self.set_normalize)
        self.subtract_checkbox.stateChanged.connect(self.set_subtract)

        layout = self.main_container
        layout.addWidget(self._canvas)
        self.addToolBar(self._toolbar)

        self._metric_checkboxes = {}
        for i, metric in enumerate(self._metrics.columns[1:]):
            metric_checkbox = QtWidgets.QCheckBox(metric)
            metric_checkbox.stateChanged.connect(self.set_metrics_by_checkbox)
            if i == len(self._metrics.columns)-2:
                metric_checkbox.toggle()
            self._metric_checkboxes[metric] = metric_checkbox
            self.metric_container.insertWidget(0, metric_checkbox)
        self.set_metrics_by_checkbox()


    @QtCore.pyqtSlot(object)
    def on_mouse_move(self, pos: Tuple[float, float]):
        self._mouse_pos = pos

    @QtCore.pyqtSlot(object)
    def on_click(self, button):
        if self._toolbar.selecting_frame and self._mouse_pos is not None:
            self.select_frame_signal.emit(self._mouse_pos[0])

    @QtCore.pyqtSlot(int)
    def set_normalize(self, checked: int):
        self._normalize = self.normalize_checkbox.isChecked()
        self.set_metrics_by_checkbox()

    @QtCore.pyqtSlot(int)
    def set_subtract(self, checked: int):
        self._subtract = self.subtract_checkbox.isChecked()
        self.set_metrics_by_checkbox()

    @QtCore.pyqtSlot(int)
    def set_metrics_by_checkbox(self, i: int=None):
        active_metrics = []
        for metric_name in self._metric_checkboxes:
            checkbox = self._metric_checkboxes[metric_name]
            if checkbox.isChecked():
                active_metrics.append(metric_name)
        self.plot(rows=active_metrics, subtract=self._subtract, normalize=self._normalize)

    def plot(self, frame: int = -1, rows: List[str]=None, subtract=False, show_orig=None, normalize=True) -> bool:
        if show_orig is None:
            show_orig = len(rows) == 1
        if rows is not None:
            self.rows = rows
        frames = self._metrics["Frame_number"].to_numpy()
        smoothed_data = []
        raw_data = []
        title_rows = []
        for row in self.rows:
            title_rows.append(row)
            data = self._metrics[row].to_numpy()
            raw_datum = data
            smoothed_datum = savgol_filter(data, 51, 3)
            if normalize:
                raw_datum = raw_datum / smoothed_datum.max()
                smoothed_datum = smoothed_datum / smoothed_datum.max()
            raw_data.append(raw_datum)
            smoothed_data.append(smoothed_datum)

        if subtract:
            if len(title_rows) != 2:
                self.ax = self._figure.add_subplot(111)
                self.ax.clear()
                self._canvas.draw()
                return False
            sub_smoothed = smoothed_data[1] - smoothed_data[0]
            sub_raw = raw_data[1] - raw_data[0]
            smoothed_data = [sub_smoothed]
            raw_data = [sub_raw]
            title_rows = [f"{title_rows[0]} - {title_rows[1]}"]

        title = ", ".join(title_rows) + " Metrics"
        self.setWindowTitle(title)

        self.ax = self._figure.add_subplot(111)

        self.ax.clear()

        handles = []
        for i in range(len(title_rows)):
            raw = raw_data[i]
            smoothed = smoothed_data[i]
            color = self.colors[i%len(self.colors)]
            handle = self.ax.plot(frames, smoothed, '-', color=color, label=title_rows[i])[0]
            handles.append(handle)
            if show_orig:
                self.ax.plot(frames, raw, "--", color=color, markersize=1, alpha=0.4, label="Original Data")
        if len(handles) > 0:
            self.ax.legend(handles=handles)

        if frame > -1:
            self.ax.axvline(x=frame)

        # refresh canvas
        self._canvas.draw()
        return True

