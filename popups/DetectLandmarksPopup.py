from PyQt5 import QtWidgets, QtGui, QtCore, uic
import cv2
import pandas as pd
import utils
import time

from typing import List, Set

from landmark_detection.Detector import LandmarkDetector


class DetectLandmarksWindow(QtWidgets.QMainWindow):
    go_button: QtWidgets.QPushButton
    all_frames_radio: QtWidgets.QRadioButton
    some_frames_radio: QtWidgets.QRadioButton
    some_frames_input: QtWidgets.QLineEdit
    progress: QtWidgets.QProgressBar
    time_estimate: QtWidgets.QLabel

    _start_time: float
    _times: List[float]

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

        self._times = []

        self._video = video
        self._save_path = save_path
        self.move(0, 0)
        self.go_button.clicked.connect(self.run_detection)

    @staticmethod
    def frame_desc_to_list(desc: str) -> Set[int]:
        frames = set()
        sections = [sec.strip() for sec in desc.split(",")]
        for section in sections:
            parts = [part.strip() for part in section.split("-")]
            if len(parts) == 1:
                try:
                    frames.add(int(parts[0])-1)
                except ValueError:
                    continue
            elif len(parts) > 1:
                try:
                    frames.update(range(int(parts[0])-1, int(parts[-1])))
                except ValueError:
                    continue
        return frames

    def run_detection(self):
        self._detector = LandmarkDetector(num_frames=1)
        self._detector.frame_done_signal.connect(self.on_new_frame)
        self._detector.landmarks_complete_signal.connect(self.on_finished)
        self._detector.new_video_started_signal.connect(self.on_start)
        self._detector.start()
        if self.all_frames_radio.isChecked():
            print("Running detect for all frames")
            self._detector.add_video(self._save_path, self._video)
        elif self.some_frames_radio.isChecked():
            frame_desc = self.some_frames_input.text()
            frames = self.frame_desc_to_list(frame_desc)
            self._detector.set_frames(frames)
            self._detector.add_video(self._save_path, self._video)
            print("Running detect for frames:", frames)

    @QtCore.pyqtSlot(str)
    def on_start(self, name: str):
        self._start_time = time.time()
        self.progress.setValue(0)
        self.progress.setRange(0, 1000)
        self.progress.show()
        self.time_estimate.show()
        self._times.append(time.time())

    @QtCore.pyqtSlot(str, pd.DataFrame)
    def on_finished(self, name: str, landmarks: pd.DataFrame):
        print("Got Landmarks")
        self.got_landmarks_signal.emit(name, landmarks)
        self._detector.stop()
        self.close()

    @QtCore.pyqtSlot(int, float)
    def on_new_frame(self, frame: int, percent: float):
        print(f"{frame} frames detected at {round(percent * 100, 2)}% done")
        self._times.append(time.time())
        if percent == 0:
            self.time_estimate.setText(f"Estimated Time Left: Infinite")
        else:
            frame_delta = min(30, len(self._times))
            time_diff = self._times[-1]-self._times[-1*frame_delta]
            time_left = frame*((1/percent)-1)*(time_diff/frame_delta)
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
