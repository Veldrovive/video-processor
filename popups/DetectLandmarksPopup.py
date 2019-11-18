from PyQt5 import QtWidgets, QtGui, QtCore, uic
import utils

class DetectLandmarksWindow(QtWidgets.QMainWindow):
    go_button: QtWidgets.QPushButton
    this_frame_radio: QtWidgets.QRadioButton
    all_frames_radio: QtWidgets.QRadioButton
    some_frames_radio: QtWidgets.QRadioButton
    some_frames_input: QtWidgets.QLineEdit

    def __init__(self, parent=None):
        super(DetectLandmarksWindow, self).__init__(parent)
        uic.loadUi("uis/DetectLandmarksPopup.ui", self)
        self.move(0, 0)
        self.go_button.clicked.connect(self.run_detection)

    def run_detection(self):
        if self.this_frame_radio.isChecked():
            print("Running detect for this frame")
        elif self.all_frames_radio.isChecked():
            print("Running detect for all frames")
        elif self.some_frames_radio.isChecked():
            frames = self.some_frames_input.text()
            print("Running detect for frames:",frames)

