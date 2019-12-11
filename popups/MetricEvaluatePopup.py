from PyQt5 import QtWidgets, QtGui, QtCore
from uis.MetricEvaluatePopup import Ui_Form
from typing import Dict, List, Tuple, Union
import pandas as pd
import numpy as np
import DataHolders

class MetricCalc(QtCore.QThread):
    _metrics: List[DataHolders.Metric] = []
    _landmarks: DataHolders.Landmarks = None

    _distances = Dict[str, List[float]]
    _areas = Dict[str, List[float]]

    frame_done_signal = QtCore.pyqtSignal(int, float)  # Emits the index of the frame as well as the percent complete
    metrics_complete_signal = QtCore.pyqtSignal(pd.DataFrame)  # Emits the results of the metrics as a dataframe

    def __init__(self, metrics, landmarks: DataHolders.Landmarks):
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


class MetricEvaluateWindow(QtWidgets.QMainWindow):
    ui: Ui_Form

    go_button: QtWidgets.QPushButton
    metric_container: QtWidgets.QVBoxLayout
    progress: QtWidgets.QProgressBar
    _metrics: List[DataHolders.Metric] = []
    _landmarks: DataHolders.Landmarks = None
    _metric_checkboxes: Dict[str, QtWidgets.QCheckBox] = {}
    _metrics_done: bool = False
    _save_path: str = None
    _final_metrics: pd.DataFrame = None

    metric_calc: MetricCalc

    metric_done_signal = QtCore.pyqtSignal(pd.DataFrame) # Emitted when metrics are done calculating

    def __init__(self, parent=None, metrics=None, landmarks: DataHolders.Landmarks=None):
        super(MetricEvaluateWindow, self).__init__(parent)
        self.ui = Ui_Form()
        self.ui.setupUi(self)
        self.set_metrics(metrics, landmarks)
        self.move(0, 0)
        self.ui.go_button.clicked.connect(self.run_metrics)

    def set_metrics(self, metrics: List[DataHolders.Metric], landmarks: DataHolders.Landmarks):
        self._metrics = metrics
        self._landmarks = landmarks
        if self._metrics is None or self._landmarks is None:
            self.ui.go_button.hide()
            return
        for checkbox_name in self._metric_checkboxes:
            checkbox = self._metric_checkboxes[checkbox_name]
            self.ui.metric_container.removeWidget(checkbox)
        self._metric_checkboxes = {}
        for metric in self._metrics:
            name = metric.name
            metric_checkbox = QtWidgets.QCheckBox(name)
            metric_checkbox.toggle()
            self._metric_checkboxes[name] = metric_checkbox
            self.ui.metric_container.insertWidget(0, metric_checkbox)
        self.ui.go_button.show()

    def run_metrics(self) -> bool:
        running_metrics = []
        for metric_name in self._metric_checkboxes:
            checkbox = self._metric_checkboxes[metric_name]
            if checkbox.isChecked():
                try:
                    running_metrics.extend([metric for metric in self._metrics if metric.name == metric_name])
                except TypeError:
                    # Then somehow the metric was deleted before use
                    pass
        if len(running_metrics) < 1:
            return False
        if not self._landmarks.has_landmarks():
            return False
        # TODO: Make this a ThreadPool instead of a single thread
        self.metric_calc = MetricCalc(running_metrics, self._landmarks)
        self.metric_calc.started.connect(self.on_metrics_start)
        self.metric_calc.frame_done_signal.connect(self.on_frame_done)
        self.metric_calc.metrics_complete_signal.connect(self.on_metrics_done)
        self.metric_calc.start()
        return True

    def on_metrics_start(self):
        self.ui.progress = QtWidgets.QProgressBar(self)
        self.ui.progress.setRange(0, 1000)
        self.ui.metric_container.addWidget(self.ui.progress)

    def on_frame_done(self, frame_count: int, progress: float):
        self.ui.progress.setValue(int(progress*1000))

    def on_metrics_done(self, metrics: pd.DataFrame):
        self.metric_done_signal.emit(metrics)
        self._final_metrics = metrics
        self.close()

