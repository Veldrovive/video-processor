from PyQt5 import QtWidgets, QtGui, QtCore, uic
import utils
from typing import Dict, List

class MetricCreationWindow(QtWidgets.QMainWindow):
    name_input: QtWidgets.QLineEdit
    type_selector: QtWidgets.QComboBox
    finished_button: QtWidgets.QPushButton
    _metric: utils.Metric = None
    _metrics: List[utils.Metric] = []
    _curr_name: str = None

    _index_map: Dict[int, utils.MetricType] = {
        0: utils.MetricType.LENGTH,
        1: utils.MetricType.AREA
    }

    def __init__(self, parent=None, metric: utils.Metric=None, all_metrics: List[utils.Metric]=[]):
        super(MetricCreationWindow, self).__init__(parent)
        self._metric = metric
        self._metrics = all_metrics
        uic.loadUi("uis/MetricCreationPopup.ui", self)
        self.update_title(self._metric.name)

        for index in self._index_map:
            if self._index_map[index] == self._metric.type:
                self.type_selector.setCurrentIndex(index)

        self.name_input.textChanged.connect(self.on_name_change)
        self.name_input.editingFinished.connect(self.on_name_finished)

        self.type_selector.currentIndexChanged.connect(self.on_type_change)

        self.finished_button.clicked.connect(self.on_finished)

    def update_title(self, name: str) -> bool:
        self.setWindowTitle(name)
        return True

    @QtCore.pyqtSlot(str)
    def on_name_change(self, input: str) -> bool:
        if len(input) == 0:
            return False
        self._curr_name = input
        self.update_title(f"Editing: {self._curr_name}")
        return True

    @QtCore.pyqtSlot()
    def on_name_finished(self):
        metric_names = [metric.name for metric in self._metrics if metric.name != self._metric.name]

        def create_name(count: int):
            return self._curr_name if count == 0 else f"{self._curr_name}-{count}"
        name_count = 0
        while create_name(name_count) in metric_names:
            name_count += 1
        self._metric.name = create_name(name_count)
        self.update_title(self._metric.name)

    @QtCore.pyqtSlot()
    def on_finished(self):
        self.on_name_finished()
        self.close()

    @QtCore.pyqtSlot(int)
    def on_type_change(self, index: int):
        self._metric.type = self._index_map[index]
        return True


