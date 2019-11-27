from PyQt5 import QtWidgets, QtGui, QtCore, uic
from typing import Dict, List, Tuple
import pandas as pd
import numpy as np
import utils

class MetricCalc(QtCore.QThread):
    _metrics: List[Tuple[utils.Metric, List[utils.Landmark], str]] = []
    _landmarks: pd.DataFrame = None

    _distances = Dict[str, List[float]]
    _areas = Dict[str, List[float]]

    # TODO: Implement the frame done signal so that a progress bar can be made
    frame_done_signal = QtCore.pyqtSignal(int, float) # Emits the index of the frame as well as the percent complete
    metrics_complete_signal = QtCore.pyqtSignal(object) # Emits the results of the metrics as a dataframe

    def __init__(self, metrics=None, landmarks: pd.DataFrame=None):
        super(MetricCalc, self).__init__()
        self._metrics = metrics
        self._landmarks = landmarks

    @staticmethod
    def poly_area(x, y):
        return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))

    def calc_area(self, locations: List[Tuple[int, int]]):
        poly_x = [loc[0] for loc in locations]
        poly_y = [loc[1] for loc in locations]
        return self.poly_area(poly_x, poly_y)

    def calc_distance(self, locations: List[Tuple[int, int]]):
        return sum([np.linalg.norm(np.array(locations[i])-np.array(locations[i+1])) for i in range(len(locations)-1)])

    def run(self):
        metrics_df = pd.DataFrame(None, columns=["Frame_number"].extend([metric[2] for metric in self._metrics]))
        metrics_df["Frame_number"] = self._landmarks["Frame_number"]
        for metric in self._metrics:
            type, landmarks, name = metric
            landmark_cols = []
            for landmark in landmarks:
                # TODO: Make this into the correct landmark for the file
                landmark_cols.extend([f"landmark_{landmark.index-1}_x", f"landmark_{landmark.index-1}_y"])
            positions = list(zip(*[self._landmarks[col] for col in landmark_cols]))
            coords = [[(int(frame_pos[2*i]), int(frame_pos[2*i+1])) for i in range(int(len(frame_pos)/2))] for frame_pos in positions]

            measures = []
            for frame_positions in coords:
                if type == utils.Metric.LENGTH:
                    measures.append(self.calc_distance(frame_positions))
                if type == utils.Metric.AREA:
                    measures.append(self.calc_area(frame_positions))
            metrics_df[name] = measures
        self.metrics_complete_signal.emit(metrics_df)
        return




class MetricWindow(QtWidgets.QMainWindow):
    go_button: QtWidgets.QPushButton
    metric_container: QtWidgets.QVBoxLayout
    _metrics: List[Tuple[utils.Metric, List[utils.Landmark], str]] = []
    _landmarks: pd.DataFrame = None
    _metric_checkboxes: Dict[str, QtWidgets.QCheckBox] = {}

    metric_done_signal = QtCore.pyqtSignal(object) # Emitted when metrics are done calculating

    def __init__(self, parent=None, metrics=None, landmarks: pd.DataFrame=None):
        super(MetricWindow, self).__init__(parent)
        uic.loadUi("uis/MetricPopup.ui", self)
        self._metrics = metrics
        self._landmarks = landmarks
        self.move(0, 0)
        for metric in self._metrics:
            name = metric[2]
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
                    running_metrics.append(*[metric for metric in self._metrics if metric[2] == metric_name])
                except TypeError:
                    # Then somehow the metric was deleted before use
                    pass
        metric_calc = MetricCalc(running_metrics, self._landmarks)
        metric_calc.start()
        metric_calc.metrics_complete_signal.connect(self.metric_done_signal.emit)
        return True

