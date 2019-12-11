from PyQt5 import QtWidgets, QtGui, QtCore
from uis.MetricViewerPopup import Ui_Form
from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt4agg import NavigationToolbar2QT as NavigationToolbar
import matplotlib
from matplotlib.figure import Figure
import os
import pandas as pd
import numpy as np
from scipy.signal import savgol_filter

from typing import Optional, List, Tuple, Dict

class ToolBar(NavigationToolbar):
    selecting_frame: bool = False

    mouse_move_signal = QtCore.pyqtSignal(object)
    frame_select_signal = QtCore.pyqtSignal(bool)

    def __init__(self, canvas, parent=None, coordinates=True):
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


class RenderCanvas(QtCore.QRunnable):
    canvas: Canvas

    def __init__(self, canvas: Canvas):
        super(RenderCanvas, self).__init__()
        self.canvas = canvas

    def run(self):
        self.canvas.draw()


class MetricDisplayWindow(QtWidgets.QMainWindow):
    ui: Ui_Form

    _thread_pool: QtCore.QThreadPool

    _normalize: bool = False

    _metrics: pd.DataFrame
    _metric_checkboxes: Dict[str, QtWidgets.QCheckBox] = None
    _figure: Figure
    _canvas: FigureCanvas
    _toolbar: NavigationToolbar
    _mouse_pos: Tuple[float, float] = (-1, -1)
    row: List[str] = ""

    _raw_plot_lines: Dict[str, matplotlib.lines.Line2D]
    _smoothed_plot_lines: Dict[str, matplotlib.lines.Line2D]

    _frames = np.ndarray
    _raw_data: Dict[str, np.ndarray]

    curr_display: int = 0
    num_metrics: int = 0

    colors = ["b", "g", "r", "c", "m", "y", "k", "w"]

    ax = None

    select_frame_signal = QtCore.pyqtSignal(int)

    def __init__(self, parent=None, metrics: Optional[pd.DataFrame]=None):
        super(MetricDisplayWindow, self).__init__(parent=parent)
        self.ui = Ui_Form()

        self._raw_data = {}

        self._figure = Figure()
        self._canvas = Canvas(self._figure)
        self._toolbar = ToolBar(self._canvas)
        self.addToolBar(QtCore.Qt.BottomToolBarArea, self._toolbar)

        self.ui.setupUi(self)

        self._thread_pool = QtCore.QThreadPool()

        self._toolbar.mouse_move_signal.connect(self.on_mouse_move)
        self._canvas.mouse_clicked_signal.connect(self.on_click)

        self.ui.normalize_checkbox.stateChanged.connect(self.set_normalize)

        layout = self.ui.main_container
        layout.addWidget(self._canvas)

        self.set_metrics(metrics)
        self.create_plot_basic()
        self.set_metrics_by_checkbox()

    def set_metrics(self, metrics: Optional[pd.DataFrame]) -> bool:
        self._metrics = metrics
        self._metric_checkboxes = {}
        if self._metrics is None or len(self._metrics.columns) < 2:
            return False
        for i, metric in enumerate(self._metrics.columns[1:]):
            metric_checkbox = QtWidgets.QCheckBox(metric)
            metric_checkbox.stateChanged.connect(self.set_metrics_by_checkbox)
            if i == len(self._metrics.columns) - 2:
                metric_checkbox.toggle()
            self._metric_checkboxes[metric] = metric_checkbox
            self.ui.metric_container.insertWidget(0, metric_checkbox)
        self.create_plot_basic(self._normalize)
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
        self._normalize = self.ui.normalize_checkbox.isChecked()
        self.create_plot_basic(self._normalize)
        self.set_metrics_by_checkbox()

    @QtCore.pyqtSlot(int)
    def set_metrics_by_checkbox(self, i: int=None):
        active_metrics = []
        for metric_name in self._metric_checkboxes:
            checkbox = self._metric_checkboxes[metric_name]
            if checkbox.isChecked():
                active_metrics.append(metric_name)
        self.draw_plot(cols=active_metrics)

    def create_plot_basic(self, normalize=False) -> bool:
        # TODO: Plot length and area seperatly and normalize based on the inter-eye distance
        # TODO: Allow users to delete and save
        self.ax = self._figure.add_subplot(111)
        self.ax.clear()
        self._raw_plot_lines = {}
        self._smoothed_plot_lines = {}
        if self._metrics is None or self._metrics.empty:
            return False
        self._frames = self._metrics["Frame_number"].to_numpy()
        for i, column in enumerate(self._metrics.columns[1:]):
            data = self._metrics[column].to_numpy()
            self._raw_data[column] = data
            window = min(len(data), 51)
            if window % 2 == 0:
                window -= 1
            if window < 3:
                smoothed = data
            else:
                smoothed = savgol_filter(data, window, 3)
            if normalize:
                shift = smoothed.min()
                factor = smoothed.max()-shift
                data = (data-shift)/factor
                smoothed = (smoothed-shift)/factor
            color = self.colors[i % len(self.colors)]
            self._smoothed_plot_lines[column] = self.ax.plot(self._frames, smoothed, 'None', color=color, label=column)[0]
            self._raw_plot_lines[column] = self.ax.plot(self._frames, data, "None", color=color, markersize=1, alpha=0.4, label=column)[0]
        self.ax.set_xlabel("Frame")
        return True

    def create_plot_advanced(self, data: List[Tuple[str, np.ndarray]]):
        self.ax = self._figure.add_subplot(111)
        self.ax.clear()
        self._smoothed_plot_lines = {}
        self._raw_plot_lines = {}
        if self._metrics is None or self._metrics.empty:
            return False
        self._frames = self._metrics["Frame_number"].to_numpy()
        for i, info in enumerate(data):
            name, raw_data = info
            self._raw_data[name] = raw_data
            color = self.colors[i % len(self.colors)]
            self._smoothed_plot_lines[name] = self.ax.plot(self._frames, raw_data, "-", color=color, markersize=1, alpha=0.4, label=name)[0]
            self._raw_plot_lines[name] = self._smoothed_plot_lines[name]
        self.ax.set_xlabel("Frame")
        self.draw_plot()
        return True

    def draw_plot(self, cols: List[str]=None, show_orig: bool = None):
        if show_orig is None:
            show_orig = False  # len(cols) == 1
        for col in self._raw_plot_lines:
            self._raw_plot_lines[col].set_linestyle("None")
            self._smoothed_plot_lines[col].set_linestyle("None")
        max_val = None
        min_val = None
        if cols is None:
            cols = self._smoothed_plot_lines.keys()
        for col in cols:
            line = self._smoothed_plot_lines[col]
            line.set_linestyle("-")
            orig_data = line.get_ydata()
            line_max = orig_data.max()
            line_min = orig_data.min()
            if max_val is None or line_max > max_val:
                max_val = line_max
            if min_val is None or line_min < min_val:
                min_val = line_min
            if show_orig:
                self._raw_plot_lines[col].set_linestyle("--")
        if min_val is not None:
            padding = max(min_val, max_val)*0.03
            self.ax.set_ylim(min_val-padding, max_val+padding)
        self.ax.legend(handles=[self._smoothed_plot_lines[col] for col in cols])
        self.render_plot()

    def render_plot(self):
        self._thread_pool.start(RenderCanvas(self._canvas))
