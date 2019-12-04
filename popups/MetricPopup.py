from PyQt5 import QtWidgets, QtGui, QtCore, uic
from typing import Dict, List, Tuple, Union
import pandas as pd
import numpy as np
import utils

class MetricCalc(QtCore.QThread):
    _metrics: List[utils.Metric] = []
    _landmarks: utils.Landmarks = None

    _distances = Dict[str, List[float]]
    _areas = Dict[str, List[float]]

    frame_done_signal = QtCore.pyqtSignal(int, float)  # Emits the index of the frame as well as the percent complete
    metrics_complete_signal = QtCore.pyqtSignal(object)  # Emits the results of the metrics as a dataframe

    def __init__(self, metrics, landmarks: utils.Landmarks):
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
        num_frames = self._landmarks.get_num_frames()
        metrics_df = pd.DataFrame(None, columns=["Frame_number"].extend([metric.name for metric in self._metrics]))
        metrics_df["Frame_number"] = list(range(num_frames))
        total = num_frames*len(self._metrics)
        done_count = 0
        self.frame_done_signal.emit(done_count, done_count / total)
        for metric in self._metrics:
            metric_type, landmarks, name = metric.type, metric.landmarks, metric.name
            measures = []
            for frame in range(num_frames):
                if metric_type == utils.MetricType.LENGTH:
                    measures.append(self.calc_distance(frame, landmarks))
                if metric_type == utils.MetricType.AREA:
                    measures.append(self.calc_area(frame, landmarks))
                done_count += 1
                self.frame_done_signal.emit(done_count, done_count / total)
            metrics_df[name] = measures
        self.metrics_complete_signal.emit(metrics_df)
        return


class MetricWindow(QtWidgets.QMainWindow):
    go_button: QtWidgets.QPushButton
    metric_container: QtWidgets.QVBoxLayout
    progress: QtWidgets.QProgressBar
    _metrics: List[utils.Metric] = []
    _landmarks: utils.Landmarks = None
    _metric_checkboxes: Dict[str, QtWidgets.QCheckBox] = {}
    _metrics_done: bool = False
    _save_path: str = None
    _final_metrics: pd.DataFrame = None

    metric_done_signal = QtCore.pyqtSignal(object) # Emitted when metrics are done calculating

    def __init__(self, parent=None, metrics=None, landmarks: utils.Landmarks=None):
        super(MetricWindow, self).__init__(parent)
        uic.loadUi("uis/MetricPopup.ui", self)
        self._metrics = metrics
        self._landmarks = landmarks
        self.move(0, 0)
        for metric in self._metrics:
            name = metric.name
            metric_checkbox = QtWidgets.QCheckBox(name)
            metric_checkbox.toggle()
            self._metric_checkboxes[name] = metric_checkbox
            self.metric_container.insertWidget(0, metric_checkbox)
        self.go_button.clicked.connect(self.run_metrics)

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
        metric_calc = MetricCalc(running_metrics, self._landmarks)
        metric_calc.started.connect(self.on_metrics_start)
        metric_calc.frame_done_signal.connect(self.on_frame_done)
        metric_calc.metrics_complete_signal.connect(self.on_metrics_done)
        metric_calc.start()
        return True

    def on_metrics_start(self):
        self.progress = QtWidgets.QProgressBar(self)
        self.progress.setRange(0, 1000)
        self.metric_container.addWidget(self.progress)
        self.file_save()

    def on_frame_done(self, frame_count: int, progress: float):
        self.progress.setValue(int(progress*1000))

    def on_metrics_done(self, metrics: pd.DataFrame):
        self.metric_done_signal.emit(metrics)
        self._final_metrics = metrics
        self.save_metrics()

    def file_save(self):
        save_path = QtWidgets.QFileDialog.getSaveFileName(self, 'Save File')[0]
        if save_path[-4:] != ".csv":
            save_path += ".csv"
        self._save_path = save_path
        self.save_metrics()

    def save_metrics(self):
        if self._save_path is None:
            return False
        if self._final_metrics is None:
            return False
        self._final_metrics.to_csv(self._save_path)
        self.close()

