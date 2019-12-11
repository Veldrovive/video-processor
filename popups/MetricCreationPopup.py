from PyQt5 import QtWidgets, QtGui, QtCore
from uis.MetricCreationPopup import Ui_Form
import DataHolders
from typing import Dict, List

class MetricCreationWindow(QtWidgets.QMainWindow):
    ui: Ui_Form

    _metric: DataHolders.Metric = None
    _metrics: List[DataHolders.Metric] = []
    _curr_name: str = None

    _index_map: Dict[int, DataHolders.MetricType] = {
        0: DataHolders.MetricType.LENGTH,
        1: DataHolders.MetricType.AREA
    }

    def __init__(self, parent=None, metric: DataHolders.Metric=None, all_metrics: List[DataHolders.Metric]=[]):
        super(MetricCreationWindow, self).__init__(parent)
        self.ui = Ui_Form()
        self.ui.setupUi(self)

        self._metric = metric
        self._metrics = all_metrics
        self.update_title(self._metric.name)

        for index in self._index_map:
            if self._index_map[index] == self._metric.type:
                self.ui.type_selector.setCurrentIndex(index)

        self.ui.name_input.textChanged.connect(self.on_name_change)
        self.ui.name_input.editingFinished.connect(self.on_name_finished)

        self.ui.type_selector.currentIndexChanged.connect(self.on_type_change)

        self.ui.finished_button.clicked.connect(self.on_finished)

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


