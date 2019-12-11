from PyQt5 import QtWidgets, QtGui
from uis.ConfigPopup import Ui_Form
import vidViewer

class ConfigWindow(QtWidgets.QMainWindow):
    ui: Ui_Form

    _view: vidViewer.ImageViewer
    _conf_path: str  # Holds a path to a config json file

    def __init__(self, parent=None, view: vidViewer.ImageViewer=None, conf_path: str=None):
        super(ConfigWindow, self).__init__(parent)
        self.ui = Ui_Form()
        self.ui.setupUi(self)
        self._view = view
        self._conf_path = conf_path
        # self.setWindowFlags(QtCore.Qt.FramelessWindowHint) # This has weird behavior
        self.move(0, 0)
        self.ui.playback_speed_input.valueChanged.connect(self.on_speed_change)
        if self._view is not None:
            for group in self._view.get_groups():
                self.ui.group_dropdown.addItem(group)
        self.ui.set_color_button.clicked.connect(self.set_color)

    def set_color(self):
        color: QtGui.QColor = QtWidgets.QColorDialog.getColor()
        print("Setting color to:", color.getRgb()[:-1])

    def on_speed_change(self, speed: float):
        if self._view is not None:
            self._view.set_playback_speed(speed)


