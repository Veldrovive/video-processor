from PyQt5 import QtWidgets, QtGui, QtCore, uic
import cv2
import pandas as pd
import utils
import time

from landmark_detection.Detector import LandmarkDetector


class DetectLandmarksWindow(QtWidgets.QMainWindow):
    go_button: QtWidgets.QPushButton
    this_frame_radio: QtWidgets.QRadioButton
    all_frames_radio: QtWidgets.QRadioButton
    some_frames_radio: QtWidgets.QRadioButton
    some_frames_input: QtWidgets.QLineEdit
    progress: QtWidgets.QProgressBar
    time_estimate: QtWidgets.QLabel

    _start_time: float

    _video: cv2.VideoCapture
    _save_path: str
    _landmarks: pd.DataFrame

    _detector: LandmarkDetector

    got_landmarks_signal = QtCore.pyqtSignal(str, pd.DataFrame)

    def __init__(self, video: cv2.VideoCapture, save_path: str, parent=None):
        super(DetectLandmarksWindow, self).__init__(parent)
        uic.loadUi("uis/DetectLandmarksPopup.ui", self)

        self.progress.hide()
        self.time_estimate.hide()

        self._video = video
        self._save_path = save_path
        self._detector = LandmarkDetector(num_frames=1)
        self._detector.frame_done_signal.connect(self.on_new_frame)
        self._detector.landmarks_complete_signal.connect(self.on_finished)
        self._detector.new_video_started_signal.connect(self.on_start)
        self._detector.start()
        self.move(0, 0)
        self.go_button.clicked.connect(self.run_detection)

    def run_detection(self):
        if self.this_frame_radio.isChecked():
            print("Running detect for this frame")
        elif self.all_frames_radio.isChecked():
            print("Running detect for all frames")
            self._detector.add_video(self._save_path, self._video)
        elif self.some_frames_radio.isChecked():
            frames = self.some_frames_input.text()
            print("Running detect for frames:", frames)

    @QtCore.pyqtSlot(str)
    def on_start(self, name: str):
        self._start_time = time.time()
        self.progress.setValue(0)
        self.progress.setRange(0, 1000)
        self.progress.show()
        self.time_estimate.show()

    @QtCore.pyqtSlot(str, pd.DataFrame)
    def on_finished(self, name: str, landmarks: pd.DataFrame):
        print("Got Landmarks")
        self.got_landmarks_signal.emit(name, landmarks)
        self._detector.stop()
        self.close()

    @QtCore.pyqtSlot(int, float)
    def on_new_frame(self, frame: int, percent: float):
        print(f"{frame} frames detected at {round(percent * 100, 2)}% done")
        total_time = time.time()-self._start_time
        if percent == 0:
            self.time_estimate.setText(f"Estimated Time Left: Infinite")
        else:
            time_left = total_time*((1-percent)/percent)
            minutes, seconds = divmod(time_left, 60)
            hours, minutes = divmod(minutes, 60)
            periods = [('h', int(hours)), ('m', int(minutes)), ('s', int(seconds))]
            time_string = ', '.join('{} {}'.format(value, name)
                                    for name, value in periods
                                    if value)
            self.time_estimate.setText(f"Estimated Time Left: {time_string}")

        self.progress.setValue(int(percent*1000))

    def stop(self):
        self._detector.stop()
