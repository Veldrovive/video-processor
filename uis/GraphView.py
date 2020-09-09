from PyQt5.QtCore import QObject, QUrl, pyqtSlot, pyqtSignal, pyqtProperty, QThread
from PyQt5 import QtWidgets
from PyQt5.QtGui import QGuiApplication
from PyQt5.QtQml import QQmlApplicationEngine, QQmlContext, QQmlComponent, qmlRegisterType
from utils.qmlBase import WindowHandler
from utils.DataHolders import MetricType
import os
from utils.Globals import Globals
from utils import DataHolders

import numpy as np
import pandas as pd
import sys
import matplotlib
import matplotlib.gridspec as gridspec
from scipy.signal import savgol_filter
from matplotlib.backend_bases import MouseButton
from matplotlib_backend_qtquick.backend_qtquickagg import FigureCanvasQtQuickAgg
from matplotlib_backend_qtquick.backend_qtquick import NavigationToolbar2QtQuick

from typing import Optional, List, Dict, Union


class DisplayBridge(QObject):
    """ A bridge class to interact with the plot in python
    """
    _zooming: bool

    _colors = ["b", "g", "r", "c", "m", "y", "k"]

    coordinatesChanged = pyqtSignal(int)
    frameChange = pyqtSignal(int)
    gotNormalization = pyqtSignal(float)

    def __init__(self, parent=None):
        self._curr_lines = {}
        self._current_active = []
        self._metric_colors = {}
        self.frames = []
        self.data = {}
        self.data = None
        self._glo = Globals.get()
        super().__init__(parent)

        # The figure and toolbar
        self.figure = None
        self.toolbar = None
        self.canvas = None

        self._zooming = False
        self._frame = 0
        self.coordinates = (0, 0)

    def updateWithCanvas(self, canvas):
        """ initialize with the canvas for the figure
        """
        self.canvas = canvas
        self.figure = canvas.figure

        self.toolbar = NavigationToolbar2QtQuick(canvas=canvas)

        self._gridspec_full = gridspec.GridSpec(1, 1)
        self._gridspec_duo = gridspec.GridSpec(2, 1)

        self._length_axis, self._area_axis = self.figure.subplots(nrows=2,
                                                                   ncols=1,
                                                                   sharex="all")
        self.canvas.draw_idle()

        # connect for displaying the coordinates
        self.figure.canvas.mpl_connect('motion_notify_event', self.on_motion)
        self.figure.canvas.mpl_connect('button_press_event', self.on_click)

    # define the coordinates property
    # (I have had problems using the @QtCore.Property directy in the past)
    def getFrame(self):
        return f"Frame: {int(round(self._coordinates[0])) + 1}"

    def setCoordinates(self, coordinates):
        self._coordinates = coordinates
        self.coordinatesChanged.emit(self._frame)

    def set_data(self, data: pd.DataFrame):
        self.data = data

    def setup_graph(self, smooth: bool, normalize_on: Optional[str] = None, normalize_factor: Optional[float] = None):
        self.frames, raw_data = self.get_raw_data()
        self.calc_data, normalized = self.calculate_data(raw_data, smooth=smooth,
                                               normalize_on=normalize_on,
                                               normalize_factor=normalize_factor)
        self.create_plot(self.frames, self.calc_data, normalized)

    def render_metrics(self, active_metrics: List[str]):
        """Renders the data to the graph"""
        self._current_active = active_metrics
        self.draw_plot(active_metrics)

    def get_raw_data(self):
        """Turns the metric dataframe into a dict"""
        if self.data is None:
            return [], {}
        frames = self.data["Frame_number"].to_numpy()
        raw_data = {}
        for i, col in enumerate(self.data.columns):
            if col == "Frame_number":
                continue
            self._metric_colors[col] = self._colors[i % len(self._colors)]
            raw_data[col] = self.data[col].to_numpy()
        return frames, raw_data

    def calculate_data(self, raw_data, smooth: bool = True, normalize_on: Optional[str] = None, normalize_factor: Optional[float] = None):
        """Runs smoothing and normalization"""
        if normalize_on is not None:
            normalize_factor = raw_data[normalize_on].mean()
            self.gotNormalization.emit(normalize_factor)
        if normalize_factor is None:
            normalize_factor = 1.0
        curr_data = {}
        for metric_name, raw_col in raw_data.items():
            data = raw_col.copy()
            if smooth:
                window = min(len(data), 51)
                if window % 2 == 0:
                    window -= 1
                if window >= 3:
                    try:
                        data = savgol_filter(data, window, 3)
                    except ValueError:
                        pass
            data = data / normalize_factor
            curr_data[metric_name] = data
        return curr_data, normalize_factor != 1.0

    def create_plot(self, frames: np.ndarray, data: Dict, normalized: bool):
        self._curr_lines = {}
        self._length_axis.clear()
        self._area_axis.clear()
        self._area_axis.set_xlabel("Frame")
        if normalized:
            self._area_axis.set_ylabel("Area Normalized")
            self._length_axis.set_ylabel("Length Normalized")
        else:
            self._area_axis.set_ylabel("Area (px^2)")
            self._length_axis.set_ylabel("Length (px)")
        for metric_name, data in data.items():
            metric = self._glo.metrics.get(metric_name)
            if metric is None or data is None:
                continue
            metric_type = metric.type
            if metric_type == DataHolders.MetricType.AREA:
                ax = self._area_axis
            else:
                ax = self._length_axis
            self._curr_lines[metric_name] = ax.plot(frames, data, 'None',
                                                    color=self._metric_colors[metric_name],
                                                    label=metric_name)[0]
        self.draw_plot(self._current_active)

    def draw_plot(self, active_metrics) -> bool:
        if self._curr_lines is None:
            return False
        active_plots = set()
        length_cols = list()
        area_cols = list()
        for metric_name, line in self._curr_lines.items():
            line.set_linestyle("None")
        for metric_name in active_metrics:
            # TODO: Fix this patch for deleting metrics
            if metric_name not in self._curr_lines:
                continue
            line = self._curr_lines[metric_name]
            line.set_linestyle("-")
            metric = self._glo.metrics.get(metric_name)
            if metric.type == DataHolders.MetricType.LENGTH:
                length_cols.append(metric.name)
            else:
                area_cols.append(metric.name)
            active_plots.add(metric.type)
        active_plots = list(active_plots)
        self._length_axis.set_visible(False)
        self._area_axis.set_visible(False)
        if len(active_plots) == 2:
            self._length_axis.set_visible(True)
            self._area_axis.set_visible(True)
            self._length_axis.set_position(self._gridspec_duo[0].get_position(self.figure))
            self._area_axis.set_position(self._gridspec_duo[1].get_position(self.figure))
        elif len(active_plots) == 1:
            axis = self._length_axis if active_plots[0] == DataHolders.MetricType.LENGTH else self._area_axis
            axis.set_visible(True)
            axis.set_position(self._gridspec_full[0].get_position(self.figure))
        self._length_axis.legend(handles=[self._curr_lines[col] for col in length_cols])
        self._area_axis.legend(handles=[self._curr_lines[col] for col in area_cols])
        self.canvas.draw_idle()
        return True

    coordinates = pyqtProperty(str, getFrame, setCoordinates,
                               notify=coordinatesChanged)

    # The toolbar commands
    @pyqtSlot(name="pan")
    def pan(self, *args):
        """Activate the pan tool."""
        self.toolbar.pan(*args)

    @pyqtSlot(name="zoom")
    def zoom(self, *args):
        """activate zoom tool."""
        self.toolbar.zoom(*args)

    @pyqtSlot()
    def home(self, *args):
        self.toolbar.home(*args)

    @pyqtSlot()
    def back(self, *args):
        self.toolbar.back(*args)

    @pyqtSlot()
    def forward(self, *args):
        self.toolbar.forward(*args)

    def on_motion(self, event):
        """
        Update the coordinates on the display
        """
        # if event.inaxes == self.axes:
        if event.xdata is None or event.ydata is None:
            return
        self.coordinates = (event.xdata, event.ydata)

    def on_click(self, event):
        """
        Alerts that frame should be changed
        """
        if event.button == MouseButton.LEFT:
            self.frameChange.emit(int(round(self._coordinates[0])))

class MetricCalc(QThread):
    """
        Asynchronously calculates the values of all metrics at each frame
        """
    _glo: Globals

    _distances = Dict[str, List[float]]  # Metric name: values
    _areas = Dict[str, List[float]]  # Metric name: Values

    frameDoneSignal = pyqtSignal(int, float)  # Frame index, percent complete
    metricsCompleteSignal = pyqtSignal(pd.DataFrame)  # The results of the metrics

    def __init__(self):
        self._glo = Globals.get()
        super().__init__()

    @staticmethod
    def poly_area(x, y):
        return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))

    def calc_area(self, frame: int,
                  landmark_defs: List[Union[int, List[int]]]) -> float:
        """
        Calculates an area of a metric given the frame number and a list of
        points or centroids
        :param frame: The frame number
        :param landmark_defs: A list of landmark indexes or centroids
        :return: The area of the metric on this frame
        """
        positions = []
        for landmark_def in landmark_defs:
            if isinstance(landmark_def, int):
                positions.append(
                    self._glo.curr_landmarks.get_landmark_locations(frame,
                                                                    landmark_def)[
                        0])
            else:
                positions.append(
                    self._glo.curr_landmarks.get_centroid(frame, landmark_def))
        poly_x = [loc[0] for loc in positions]
        poly_y = [loc[1] for loc in positions]
        return self.poly_area(poly_x, poly_y)

    def calc_distance(self, frame: int,
                      landmark_defs: List[Union[int, List[int]]]) -> float:
        """
        Calculates a distance metric given the frame number and a list of points
        or centroids
        :param frame: The frame number
        :param landmark_defs: A list of landmark indexes or centroids
        :return: The summed distance between all metrics
        """
        positions = []
        for landmark_def in landmark_defs:
            if isinstance(landmark_def, int):
                positions.append(
                    self._glo.curr_landmarks.get_landmark_locations(frame,
                                                                    landmark_def)[
                        0])
            else:
                positions.append(
                    self._glo.curr_landmarks.get_centroid(frame, landmark_def))
        return sum(
            [np.linalg.norm(np.array(positions[i]) - np.array(positions[i + 1]))
             for i in range(len(positions) - 1)])

    def run(self):
        frames = self._glo.curr_landmarks.get_frames()  # All frames with calculated landmarks
        metrics = self._glo.metrics.metrics  # A list of metric objects
        metrics_df = pd.DataFrame(None, columns=["Frame_number"].extend(
            [metric.name for metric in metrics]))
        metrics_df["Frame_number"] = frames
        total = len(frames) * len(
            metrics)  # The total number of calculations to make
        done_count = 0  # How many frames have been processed
        self.frameDoneSignal.emit(done_count,
                                  done_count / total)  # Initialize the done count
        for metric in metrics:
            metric_type, landmarks, name = metric.type, metric.landmarks, metric.name
            measures = []
            for frame in frames:
                if metric_type == DataHolders.MetricType.LENGTH:
                    measures.append(self.calc_distance(frame, landmarks))
                if metric_type == DataHolders.MetricType.AREA:
                    measures.append(self.calc_area(frame, landmarks))
                done_count += 1
                self.frameDoneSignal.emit(done_count, done_count / total)
            metrics_df[name] = measures
        self.metricsCompleteSignal.emit(metrics_df)
        return

class GraphViewHandlerV2(WindowHandler):
    filesUpdated = pyqtSignal()
    @pyqtProperty(list, notify=filesUpdated)
    def files_list(self):
        if self._glo.project is None:
            return []
        return [{"name": os.path.basename(file), "path": file} for file in self._glo.project.files]

    metricsUpdated = pyqtSignal()
    @pyqtProperty(list, notify=metricsUpdated)
    def metrics_list(self):
        if self._glo.metrics is None:
            return []
        return [{"name": metric.name, "type": metric.type.name} for metric in self._glo.metrics.get_all()]

    @pyqtProperty(list, notify=metricsUpdated)
    def metrics_names(self):
        return [metric["name"] for metric in self.metrics_list]

    normalization_value: float
    normalize_on: Optional[str]

    normalizationValueChanged = pyqtSignal(float, arguments=["value"])
    fileChange = pyqtSignal(str, arguments=["fileName"])
    progressChanged = pyqtSignal(float, arguments=["progress"])

    calculator: MetricCalc

    def __init__(self, engine):
        self._display_bridge = DisplayBridge()
        self._display_bridge.gotNormalization.connect(lambda value: self.got_normalization_value(value))
        self.normalization_value = None
        self.metric_shown_map = {}
        super().__init__(engine, "GraphView.qml",
                         "View Metrics")
        figure = self._window.findChild(QObject, "figure")
        self._display_bridge.updateWithCanvas(figure)
        self._glo.onProjectChange.connect(self.on_project_changed)
        self._glo.onMetricsChange.connect(self.on_metrics_changed)

    def show(self):
        """Runs the metrics whenever the page is opened"""
        self.on_metrics_changed()
        super().show()

    def on_project_changed(self):
        """Handles view changes when the project is changed"""
        self.metric_shown_map = {metric: False for metric in self.metrics_names}
        self.on_metrics_changed()
        self.filesUpdated.emit()
        self._display_bridge.frameChange.connect(self._glo.video_config.seek_to)

    def on_metrics_changed(self):
        """Handles view changes when the metrics change"""
        for metric in self.metrics_names:
            if metric not in self.metric_shown_map:
                self.metric_shown_map[metric] = False
        keys = list(self.metric_shown_map.keys())
        for metric in keys:
            if metric not in self.metrics_names:
                del self.metric_shown_map[metric]
        self.run_metric_calc()
        self.metricsUpdated.emit()

    @pyqtSlot(str, result=bool)
    def check_shown(self, metric_name: str):
        """Returns whether a metric is shown"""
        return self.metric_shown_map[metric_name]

    def run_metric_calc(self):
        landmarks = self._glo.curr_landmarks
        if landmarks is not None:
            self.progressChanged.emit(0)
            self.calculator = MetricCalc()
            self.calculator.frameDoneSignal.connect(lambda _, progress: self.progressChanged.emit(progress))
            self.calculator.metricsCompleteSignal.connect(self.set_data)
            self.calculator.start()

    def set_data(self, metrics: pd.DataFrame):
        print("Finished getting metrics")
        self._display_bridge.set_data(metrics)
        self._display_bridge.setup_graph(True)
        self.render_graph()

    def render_graph(self):
        active_metrics = [metric for metric, active in self.metric_shown_map.items() if active]
        self._display_bridge.render_metrics(active_metrics)

    def setup_contexts(self):
        """Overrides context setup to add display view"""
        self.add_context("displayBridge", self._display_bridge)

    @pyqtSlot(int, str)
    def on_normalization_combo_set(self, index: int, value: str):
        print("Need to implement setting normalization")
        # self.normalizationValueChanged.emit(self.normalization_value)
        self.normalize_on = value
        if value == "None":
            self.normalization_value = 1.0
            self.normalizationValueChanged.emit(self.normalization_value)
            self._display_bridge.setup_graph(True, None, self.normalization_value)
        else:
            self._display_bridge.setup_graph(True, self.normalize_on, self.normalization_value)

    @pyqtSlot(float)
    def on_normalization_value_set(self, value: float):
        self.normalization_value = value
        self.normalize_on = None
        self._display_bridge.setup_graph(True, None, self.normalization_value)
        self.normalizationValueChanged.emit(value)

    @pyqtSlot(float)
    def got_normalization_value(self, value: float):
        self.normalization_value = value
        self.normalize_on = None
        self.normalizationValueChanged.emit(value)

    @pyqtSlot(int, str, bool)
    def set_metric_shown(self, metric_index: int, metric_name: str, shown: bool):
        """Sets whether a metric should be rendered"""
        self.metric_shown_map[metric_name] = shown
        self.render_graph()

    @pyqtSlot(int, str, str)
    def change_metric_name(self, metric_index: int, old_name: str, new_name: str):
        self._glo.metrics.rename(old_name, new_name)
        self.metricsUpdated.emit()

    @pyqtSlot(int, str)
    def delete_metric(self, metric_index: int, metric_name: str):
        self._glo.metrics.remove(metric_name)
        self.metricsUpdated.emit()
        self._glo.video_config.refresh_frames()

    @pyqtSlot(int, str, str)
    def change_metric_type(self, metric_index: int, metric_name: str, new_type: str):
        if new_type == "Distance":
            m_type = MetricType.LENGTH
        else:
            m_type = MetricType.AREA
        self._glo.metrics.set_metric_type(metric_name, m_type)
        self.metricsUpdated.emit()
        self._glo.video_config.refresh_frames()

    @pyqtSlot(str)
    def change_file(self, page_id: str):
        self.fileChange.emit(page_id)
        self._glo.select_file(page_id)
        self.run_metric_calc()

    def save_file_dialog(self, title: str, allowed_types: Union[List[str], str], default: str=None) -> Optional[str]:
        """
        Prompt the user to choose a file to save
        :param title: The title window
        :param type: The file type that will be saved
        """
        if isinstance(allowed_types, list):
            allowed_types = f"Types ({' '.join([f'*.{file_type}' for file_type in allowed_types])})"
        file = QtWidgets.QFileDialog.getSaveFileName(caption=title, filter=allowed_types, directory=default)[0]
        return file if len(file) > 0 else None

    @pyqtSlot()
    def save_metrics(self):
        # print("Saving Metrics to", save_loc)
        frames = self._display_bridge.frames
        data = self._display_bridge.calc_data
        metric_frame = pd.DataFrame()
        metric_frame["Frame_number"] = frames
        active_metrics = [metric for metric, active in self.metric_shown_map.items() if active]
        for metric_name, data in data.items():
            if metric_name in active_metrics:
                metric_frame[metric_name] = data
        save_path = self.save_file_dialog("Save Metrics", ["csv"], self._glo.project.metrics_dir+f"/output_{os.path.splitext(os.path.basename(self._glo.curr_file))[0]}.csv")
        if save_path is not None:
            metric_frame.to_csv(save_path)


