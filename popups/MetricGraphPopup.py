from PyQt5 import QtWidgets, QtCore, uic
from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt4agg import NavigationToolbar2QT as NavigationToolbar
import matplotlib
import matplotlib.gridspec as gridspec
from matplotlib.figure import Figure
import os
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from popups.Confirmation import Confirmation
import DataHolders
import persistentConfig
from typing import List, Tuple, Dict, Optional, Union, Any

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
        try:
            self.canvas.draw()
        except IndexError as e:
            print("Failed to draw:", e)


class MetricCalc(QtCore.QThread):
    _metrics: List[DataHolders.Metric] = None
    _landmarks: DataHolders.Landmarks = None

    _distances = Dict[str, List[float]]
    _areas = Dict[str, List[float]]

    frame_done_signal = QtCore.pyqtSignal(int, float)  # Emits the index of the frame as well as the percent complete
    metrics_complete_signal = QtCore.pyqtSignal(pd.DataFrame)  # Emits the results of the metrics as a dataframe

    def __init__(self, metrics: List[DataHolders.Metric], landmarks: DataHolders.Landmarks):
        super(MetricCalc, self).__init__()
        self._metrics = metrics
        self._landmarks = landmarks

    @staticmethod
    def poly_area(x, y):
        return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))

    def calc_area(self, frame: int, landmark_defs: List[Union[int, List[int]]]):
        positions = []
        for landmark_def in landmark_defs:
            if isinstance(landmark_def, int):
                positions.append(self._landmarks.get_landmark_locations(frame, landmark_def)[0])
            else:
                positions.append(self._landmarks.get_centroid(frame, landmark_def))
        poly_x = [loc[0] for loc in positions]
        poly_y = [loc[1] for loc in positions]
        return self.poly_area(poly_x, poly_y)

    def calc_distance(self, frame: int, landmark_defs: List[Union[int, List[int]]]):
        positions = []
        for landmark_def in landmark_defs:
            if isinstance(landmark_def, int):
                positions.append(self._landmarks.get_landmark_locations(frame, landmark_def))
            else:
                positions.append(self._landmarks.get_centroid(frame, landmark_def))
        return sum([np.linalg.norm(np.array(positions[i])-np.array(positions[i+1])) for i in range(len(positions)-1)])

    def run(self):
        frames = self._landmarks.get_frames()
        metrics_df = pd.DataFrame(None, columns=["Frame_number"].extend([metric.name for metric in self._metrics]))
        metrics_df["Frame_number"] = frames
        total = len(frames)*len(self._metrics)
        done_count = 0
        self.frame_done_signal.emit(done_count, done_count / total)
        for metric in self._metrics:
            metric_type, landmarks, name = metric.type, metric.landmarks, metric.name
            measures = []
            for frame in frames:
                if metric_type == DataHolders.MetricType.LENGTH:
                    measures.append(self.calc_distance(frame, landmarks))
                if metric_type == DataHolders.MetricType.AREA:
                    measures.append(self.calc_area(frame, landmarks))
                done_count += 1
                self.frame_done_signal.emit(done_count, done_count / total)
            metrics_df[name] = measures
        self.metrics_complete_signal.emit(metrics_df)
        return


class RenamableLabel(QtWidgets.QLineEdit):
    on_name_changed_signal = QtCore.pyqtSignal(str, str) # Old name, New name
    on_editing_signal = QtCore.pyqtSignal()
    stop_editing_signal = QtCore.pyqtSignal()
    clicked = QtCore.pyqtSignal()

    _old_name: str

    def __init__(self, label: str, parent=None):
        super(RenamableLabel, self).__init__(label, parent)
        self.setStyleSheet("QLineEdit:read-only {"
                      "    border: none;"
                      "    background: transparent; }"
                      "QLineEdit {"
                      "    background: white; }")
        self.setReadOnly(True)
        self.editingFinished.connect(self.on_edit_finished)
        self._old_name = label

    @QtCore.pyqtSlot()
    def on_edit_finished(self):
        new_value = self.displayText()
        self.unsetCursor()
        self.setSelection(0, 0)
        self.setReadOnly(True)
        self.stop_editing_signal.emit()
        if new_value != self._old_name:
            self.on_name_changed_signal.emit(self._old_name, new_value)
            self._old_name = new_value

    def mousePressEvent(self, event):
        if self.isReadOnly():
            self.clicked.emit()
        super().mousePressEvent(event)

    def mouseDoubleClickEvent(self, event):
        self.setReadOnly(False)
        self.on_editing_signal.emit()
        self.clicked.emit()
        self.selectAll()
        super().mouseDoubleClickEvent(event)


class EditableCheckbox(QtWidgets.QWidget):
    metric: DataHolders.Metric

    stateChanged: QtCore.pyqtSignal = QtCore.pyqtSignal()
    nameChanged: QtCore.pyqtSignal = QtCore.pyqtSignal(str, str)
    typeChanged: QtCore.pyqtSignal = QtCore.pyqtSignal(DataHolders.Metric, DataHolders.MetricType)

    checkbox: QtWidgets.QCheckBox
    label: RenamableLabel
    type_combo: QtWidgets.QComboBox

    def __init__(self, metric: DataHolders.Metric, parent=None):
        super(EditableCheckbox, self).__init__(parent=parent)
        self.metric = metric
        self.layout = QtWidgets.QHBoxLayout()

        self.setSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)

        self.checkbox = QtWidgets.QCheckBox()
        self.label = RenamableLabel(metric.name)
        self.type_combo = QtWidgets.QComboBox()
        self.type_combo.addItem("Length", DataHolders.MetricType.LENGTH)
        self.type_combo.addItem("Area", DataHolders.MetricType.AREA)
        self.type_combo.setCurrentIndex(int(metric.type == DataHolders.MetricType.AREA))
        self.layout.addWidget(self.checkbox)
        self.layout.addWidget(self.label)
        self.layout.addWidget(self.type_combo)
        self.type_combo.hide()

        self.setLayout(self.layout)

        self.label.clicked.connect(self.checkbox.toggle)

        self.checkbox.stateChanged.connect(self.stateChanged.emit)
        self.label.on_editing_signal.connect(self.on_editing)
        self.label.stop_editing_signal.connect(self.on_stop_editing)
        self.label.on_name_changed_signal.connect(self.nameChanged.emit)
        self.type_combo.currentIndexChanged.connect(self.on_type_changed)

    @QtCore.pyqtSlot()
    def on_editing(self):
        self.type_combo.show()

    @QtCore.pyqtSlot()
    def on_stop_editing(self):
        self.type_combo.hide()

    @QtCore.pyqtSlot(int)
    def on_type_changed(self, index: int):
        new_type = self.type_combo.itemData(index)
        self.typeChanged.emit(self.metric, new_type)

    def isChecked(self) -> bool:
        return self.checkbox.isChecked()

class MetricGraphWindow(QtWidgets.QMainWindow):
    # Threading
    _thread_pool: QtCore.QThreadPool
    _metric_calculator: MetricCalc

    # Ui Work
    normalize_checkbox: QtWidgets.QCheckBox
    normalize_on_chooser: QtWidgets.QComboBox
    recalc_metrics_button: QtWidgets.QPushButton
    save_metrics_button: QtWidgets.QPushButton
    delete_metrics_button: QtWidgets.QPushButton

    metric_calc_progress: QtWidgets.QProgressBar
    metric_scroll_area: QtWidgets.QScrollArea
    metrics_picker: QtWidgets.QVBoxLayout
    graph_container: QtWidgets.QHBoxLayout

    metric_checkboxes: Dict[str, EditableCheckbox] = None

    # Plotting Widgets
    _figure: Figure
    _canvas: Canvas
    _toolbar: ToolBar

    _length_axis: matplotlib.figure.Axes
    _area_axis: matplotlib.figure.Axes

    _gridspec_full: matplotlib.figure.GridSpec
    _gridspec_duo: matplotlib.figure.GridSpec

    # Styling Data
    _colors = ["b", "g", "r", "c", "m", "y", "k"]
    _metric_colors: Dict[str, str]
    _active_metrics: List[str]

    # Landmark Data
    _landmarks: DataHolders.Landmarks
    _frames: np.ndarray

    # Metric Data
    _metrics: List[DataHolders.Metric]
    _raw_data: Dict[str, Optional[np.ndarray]] = None
    _curr_data: Dict[str, Optional[np.ndarray]]
    _mouse_pos: Tuple[float, float] = (-1, -1)
    _config: persistentConfig.Config

    # Plot Data
    _curr_lines: Dict[str, Any] = None

    # Data Options
    _normalize: bool = True
    _normalize_on: Optional[DataHolders.Metric] = None
    _smooth: bool = True

    save_name: str = None
    select_frame_signal = QtCore.pyqtSignal(int)

    def __init__(self, parent=None):
        super(MetricGraphWindow, self).__init__(parent=parent)
        uic.loadUi("./uis/MetricGraph.ui", self)
        self.setWindowTitle("Metric Analysis")
        self._thread_pool = QtCore.QThreadPool()
        self._active_metrics = []

        self.metric_calc_progress.hide()
        self.normalize_checkbox.stateChanged.connect(self.on_normalize_change)
        self.recalc_metrics_button.clicked.connect(self.calc_metrics)
        self.save_metrics_button.clicked.connect(self.save_metrics)
        self.delete_metrics_button.clicked.connect(self.delete_selected_metrics)
        self.normalize_checkbox.toggle()

        self._gridspec_full = gridspec.GridSpec(1, 1)
        self._gridspec_duo = gridspec.GridSpec(2, 1)
        self.setup_graph()

    def setup_graph(self):
        self._figure = Figure()
        self._canvas = Canvas(self._figure)
        self._toolbar = ToolBar(self._canvas)
        self.addToolBar(QtCore.Qt.TopToolBarArea, self._toolbar)
        self.graph_container.addWidget(self._canvas)
        self._toolbar.mouse_move_signal.connect(self.on_mouse_move)
        self._canvas.mouse_clicked_signal.connect(self.on_click)
        self._length_axis, self._area_axis = self._figure.subplots(nrows=2, ncols=1, sharex="all")

    def set_config(self, config: persistentConfig.Config, landmarks: DataHolders.Landmarks):
        if self.metric_checkboxes is not None:
            for name, checkbox in self.metric_checkboxes.items():
                self.metrics_picker.removeWidget(checkbox)
                checkbox.setParent(None)
                checkbox.deleteLater()
        self._metric_colors = {}
        self.metric_checkboxes = {}
        self._raw_data = {}
        self._curr_data = {}
        self._config = config
        self._metrics = self._config.metrics.get_all()
        self._landmarks = landmarks
        for i, metric in enumerate(self._metrics):
            self._metric_colors[metric.name] = self._colors[i % len(self._colors)]
            self._raw_data[metric.name] = None
            self._curr_data[metric.name] = None
        self.calc_metrics()
        self.populate_normalize_combo()
        self.populate_metric_picker()

    def populate_metric_picker(self):
        for metric in self._metrics:
            metric_checkbox = EditableCheckbox(metric, self)
            metric_checkbox.stateChanged.connect(self.set_metrics_by_checkbox)
            metric_checkbox.nameChanged.connect(self.on_metric_name_changed)
            metric_checkbox.typeChanged.connect(self.on_metric_type_changed)
            metric_checkbox.resize(500, 50)
            self.metric_checkboxes[metric.name] = metric_checkbox
            self.metrics_picker.insertWidget(0, metric_checkbox)

    def set_metrics_by_checkbox(self):
        self._active_metrics = []
        for metric_name in self.metric_checkboxes:
            checkbox = self.metric_checkboxes[metric_name]
            if checkbox.isChecked():
                self._active_metrics.append(metric_name)
        self.draw_plot()

    @staticmethod
    def edit_dict_key(old, new, d: Dict):
        list_rep = []
        for key, val in d.items():
            if key != old:
                list_rep.append((key, val))
            else:
                list_rep.append((new, val))
        d.clear()
        for key, val in list_rep:
            d[key] = val

    @QtCore.pyqtSlot(str, str)
    def on_metric_name_changed(self, old: str, new: str):
        print("Changing name of",old,"to",new)
        self.edit_dict_key(old, new, self._raw_data)
        self.edit_dict_key(old, new, self._curr_data)
        self.edit_dict_key(old, new, self.metric_checkboxes)
        self.edit_dict_key(old, new, self._metric_colors)
        for metric in self._metrics:
            if metric.name == old:
                metric.name = new
        if old in self._active_metrics:
            self._active_metrics[self._active_metrics.index(old)] = new
        self.populate_normalize_combo()
        self.create_plot()
        self.draw_plot()
        self._config.save()

    @QtCore.pyqtSlot(DataHolders.Metric, DataHolders.MetricType)
    def on_metric_type_changed(self, metric: DataHolders.Metric, metric_type: DataHolders.MetricType):
        metric.type = metric_type
        self.calc_metrics()

    def populate_normalize_combo(self):
        norm_index = 0
        self.normalize_on_chooser.clear()
        for i, metric in enumerate(self._metrics):
            self.normalize_on_chooser.addItem(metric.name, metric)
            if metric.landmarks == [39, 42]:
                self.normalize_on_chooser.setCurrentIndex(i)
                norm_index = i
        self._normalize_on = self.normalize_on_chooser.itemData(norm_index)
        self.normalize_chooser_changed(norm_index)
        self.normalize_on_chooser.currentIndexChanged.connect(self.normalize_chooser_changed)

    @QtCore.pyqtSlot()
    def on_normalize_change(self):
        self._normalize = self.normalize_checkbox.isChecked()
        self.calculate_curr()

    @QtCore.pyqtSlot(int)
    def normalize_chooser_changed(self, index: int):
        self._normalize_on = self.normalize_on_chooser.itemData(index)
        if self._normalize_on is not None and self._raw_data[self._normalize_on.name] is not None:
            self.calculate_curr()

    def calc_metrics(self) -> bool:
        if self._landmarks is None:
            return False
        self._metric_calculator = MetricCalc(self._metrics, self._landmarks)
        self.metric_calc_progress.setValue(0)
        self.metric_calc_progress.show()
        self._metric_calculator.frame_done_signal.connect(self.on_frame_calc)
        self._metric_calculator.metrics_complete_signal.connect(self.on_calc_finished)
        self._metric_calculator.start()

    @QtCore.pyqtSlot(int, float)
    def on_frame_calc(self, frame: int, percent: float):
        self.metric_calc_progress.setValue(int(round(percent*1000)))

    @QtCore.pyqtSlot(pd.DataFrame)
    def on_calc_finished(self, metrics: pd.DataFrame):
        self._frames = metrics["Frame_number"].to_numpy()
        for col in metrics.columns:
            if col == "Frame_number":
                continue
            self._raw_data[col] = metrics[col].to_numpy()
        self.metric_calc_progress.hide()
        self.calculate_curr()

    def calculate_curr(self) -> bool:
        normalize_factor = 1
        if self._raw_data is None:
            return False
        if self._normalize and self._normalize_on is not None:
            normalize_name = self._normalize_on.name
            normalize_factor = self._raw_data[normalize_name].mean()
        for metric_name, raw_data in self._raw_data.items():
            data = raw_data.copy()
            if self._smooth:
                window = min(len(data), 51)
                if window % 2 == 0:
                    window -= 1
                if window >= 3:
                    data = savgol_filter(data, window, 3)
            data = data / normalize_factor
            self._curr_data[metric_name] = data
        self.create_plot()
        self.draw_plot()
        return True

    def create_plot(self):
        self._curr_lines = {}
        self._length_axis.clear()
        self._area_axis.clear()
        self._area_axis.set_xlabel("Frame")
        if self._normalize:
            self._area_axis.set_ylabel("Area Normalized")
            self._length_axis.set_ylabel("Length Normalized")
        else:
            self._area_axis.set_ylabel("Area (px^2)")
            self._length_axis.set_ylabel("Length (px)")
        for metric_name, data in self._curr_data.items():
            metric = self.get_metric_by_name(metric_name)
            if metric is None or data is None:
                continue
            metric_type = metric.type
            if metric_type == DataHolders.MetricType.AREA:
                ax = self._area_axis
            else:
                ax = self._length_axis
            self._curr_lines[metric_name] = ax.plot(self._frames, data, 'None', color=self._metric_colors[metric_name], label=metric_name)[0]

    def draw_plot(self) -> bool:
        if self._curr_lines is None:
            return False
        active_plots = set()
        length_cols = list()
        area_cols = list()
        for metric_name, line in self._curr_lines.items():
            line.set_linestyle("None")
        for metric_name in self._active_metrics:
            line = self._curr_lines[metric_name]
            line.set_linestyle("-")
            metric = self.get_metric_by_name(metric_name)
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
            self._length_axis.set_position(self._gridspec_duo[0].get_position(self._figure))
            self._area_axis.set_position(self._gridspec_duo[1].get_position(self._figure))
        elif len(active_plots) == 1:
            axis = self._length_axis if active_plots[0] == DataHolders.MetricType.LENGTH else self._area_axis
            axis.set_visible(True)
            axis.set_position(self._gridspec_full[0].get_position(self._figure))
        self._length_axis.legend(handles=[self._curr_lines[col] for col in length_cols])
        self._area_axis.legend(handles=[self._curr_lines[col] for col in area_cols])
        self.render_plot()
        return True

    def render_plot(self):
        self._thread_pool.start(RenderCanvas(self._canvas))

    @QtCore.pyqtSlot(object)
    def on_mouse_move(self, pos: Tuple[float, float]):
        self._mouse_pos = pos

    @QtCore.pyqtSlot(object)
    def on_click(self, button):
        if self._toolbar.selecting_frame and self._mouse_pos is not None:
            self.select_frame_signal.emit(self._mouse_pos[0])

    def get_metric_by_name(self, name: str) -> Optional[DataHolders.Metric]:
        for metric in self._metrics:
            if metric.name == name:
                return metric
        return None

    def save_file_dialog(self, title: str, allowed_types: Union[List[str], str], default: str=None) -> Optional[str]:
        """
        Prompt the user to choose a file to save
        :param title: The title window
        :param type: The file type that will be saved
        """
        if isinstance(allowed_types, list):
            allowed_types = f"Types ({' '.join([f'*.{file_type}' for file_type in allowed_types])})"
        file = QtWidgets.QFileDialog.getSaveFileName(self, title, filter=allowed_types, directory=default)[0]
        return file if len(file) > 0 else None

    def set_save_name(self, video_path: str):
        if video_path is not None:
            self.save_name = os.path.splitext(video_path)[0]+"_metrics.csv"

    def save_metrics(self):
        metric_frame = pd.DataFrame()
        metric_frame["Frame_number"] = self._frames
        for metric_name, data in self._curr_data.items():
            if metric_name in self._active_metrics:
                metric_frame[metric_name] = data
        save_path = self.save_file_dialog("Save Metrics", ["csv"], default=self.save_name)
        if save_path is not None:
            metric_frame.to_csv(save_path)

    def delete_selected_metrics(self):
        """
        Removes the selected metrics from the metrics list
        """
        # TODO: Maybe make this not reload the metrics
        message = "Delete Metrics: \n{}".format(',\n'.join(self._active_metrics))
        self.conf_box = Confirmation("Deleting metrics", message, parent=self, on_conf=self.on_delete_conf)

    def on_delete_conf(self):
        """
        Runs once user has confirmed they want to delete selected metrics
        """
        for metric_name in self._active_metrics:
            self._config.metrics.remove(metric_name)
        self.set_config(self._config, self._landmarks)
        self._config.save()
