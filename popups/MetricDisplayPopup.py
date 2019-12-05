from PyQt5 import QtWidgets, QtGui, QtCore
from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt4agg import NavigationToolbar2QT as NavigationToolbar
import matplotlib
from matplotlib.figure import Figure
import os
import pandas as pd
from scipy.signal import savgol_filter

from typing import Optional, List, Tuple

class ToolBar(NavigationToolbar):
    selecting_frame: bool = False

    mouse_move_signal = QtCore.pyqtSignal(object)
    frame_select_signal = QtCore.pyqtSignal(bool)

    def __init__(self, canvas, parent, coordinates=True):
        super(ToolBar, self).__init__(canvas, parent, coordinates=coordinates)

    def _init_toolbar(self):
        self.basedir = os.path.join(matplotlib.rcParams['datapath'], 'images')
        self.toolitems = list(self.toolitems)
        self.toolitems.insert(6, ("Frame", "Go to frame", "move", "frame"))

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
    _metrics: pd.DataFrame
    _main_widget = QtWidgets.QWidget
    _figure: Figure
    _canvas: FigureCanvas
    _toolbar: NavigationToolbar
    _mouse_pos: Tuple[float, float] = (-1, -1)
    row: List[str] = ""

    curr_display: int = 0
    num_metrics: int = 0

    ax = None

    select_frame_signal = QtCore.pyqtSignal(int)

    def __init__(self, parent=None, metrics: Optional[pd.DataFrame]=None):
        super(MetricDisplayWindow, self).__init__(parent=parent)
        self._main_widget = QtWidgets.QWidget()
        self.setCentralWidget(self._main_widget)
        self._figure = Figure()
        self._canvas = Canvas(self._figure)
        self._toolbar = ToolBar(self._canvas, self)
        self._metrics = metrics

        self._toolbar.mouse_move_signal.connect(self.on_mouse_move)
        self._canvas.mouse_clicked_signal.connect(self.on_click)

        layout = QtWidgets.QVBoxLayout(self._main_widget)
        layout.addWidget(self._canvas)
        self.addToolBar(self._toolbar)

        self.num_metrics = len(self._metrics.columns)-1
        if self._metrics is not None:
            self.set_metric_index(index=0)

    @QtCore.pyqtSlot(object)
    def on_mouse_move(self, pos: Tuple[float, float]):
        self._mouse_pos = pos

    @QtCore.pyqtSlot(object)
    def on_click(self, button):
        if self._toolbar.selecting_frame and self._mouse_pos is not None:
            self.select_frame_signal.emit(self._mouse_pos[0])

    def keyPressEvent(self, event: QtCore.QEvent):
        key = event.key()
        if key == QtCore.Qt.Key_Left:
            self.curr_display += 1
            if self.curr_display >= self.num_metrics:
                self.curr_display = 0
            self.set_metric_index(index=self.curr_display)
        if key == QtCore.Qt.Key_Right:
            self.curr_display -= 1
            if self.curr_display < 0:
                self.curr_display = self.num_metrics-1
            self.set_metric_index(index=self.curr_display)

    def set_metric_index(self, frame: int = -1, index: int = 0):
        self.plot(frame, rows=[self._metrics.columns[index+1]])

    def plot(self, frame: int = -1, rows: List[str] = None):
        ''' plot some random stuff '''
        # random data
        if rows is not None:
            self.rows = rows
        frames = self._metrics["Frame_number"].to_numpy()
        cols = []
        title_rows = []
        for row in self.rows:
            title_rows.append(row)
            data = self._metrics[row].to_numpy()
            cols.append(savgol_filter(data, 51, 3))
        title = ", ".join(title_rows) + " Metrics"
        self.setWindowTitle(title)

        # create an axis
        self.ax = self._figure.add_subplot(111)

        # discards the old graph
        self.ax.clear()

        # plot data

        for data in cols:
            self.ax.plot(frames, data, '-', label="Smoothed Data")
        # self.ax.plot(frames, data, "o", color="black", markersize=1, alpha=0.4, label="Original Data")

        if frame > -1:
            self.ax.axvline(x=frame)

        # refresh canvas
        self._canvas.draw()

    def set_plot(self, rows: List[str]):
        self.rows = rows

