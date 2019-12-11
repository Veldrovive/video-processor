from PyQt5 import QtWidgets, QtGui, QtCore
from uis.LandmarkEditor import Ui_Form
import vidViewer
import DataHolders

class MoveLandmarkWindow(QtWidgets.QMainWindow):
    _landmark: DataHolders.Landmark
    _view: vidViewer.ImageViewer

    def __init__(self, parent=None, view: vidViewer.ImageViewer=None, landmark: DataHolders.Landmark=None):
        super(MoveLandmarkWindow, self).__init__(parent)
        self.ui = Ui_Form()
        self.ui.setupUi(self)

        self.move(0, 0)
        self._landmark = landmark
        self._view = view
        if self._landmark is not None:
            self.setWindowTitle(f"Editing: {self._landmark.index}")
            self.position.setText(str(self._landmark.location))
            if self._view is not None:
                for index, group in enumerate(self._view.get_groups()):
                    self.group_dropdown.addItem(group)
                    if group == self._landmark.group:
                        self.group_dropdown.setCurrentIndex(index)
                self.group_dropdown.activated[str].connect(self.set_group)
        else:
            self.setWindowTitle("Editing: Nothing")

    @QtCore.pyqtSlot(str)
    def set_group(self, group: str):
        print("Moving:", self._landmark, "to", group)
        self._view.add_to_group(group, [self._landmark.index])


