import sys
import pandas as pd
import cv2

from PyQt5 import QtWidgets, QtGui, QtCore

from popups.DetectLandmarksPopup import DetectLandmarksWindow
from popups.MetricGraphPopup import MetricGraphWindow
from popups.ConfigPopup import ConfigWindow
from popups.MetricFlow import MetricFlow

from uis.VideoViewerPopup import VideoViewerWindow

class MainWindow(VideoViewerWindow):
    _calculated_metrics: pd.DataFrame

    def __init__(self):
        super(MainWindow, self).__init__()
        self._calculated_metrics = pd.DataFrame()
        self.add_menu("&Metrics")
        self.setup_metric_flow_window()
        self.add_menu("&Landmarks")
        self.setup_landmark_detector_window()
        self.setup_graph_window()

    def setup_graph_window(self):
        window: MetricGraphWindow = self.add_window("graph_metrics", MetricGraphWindow)
        self.add_action("&Metrics", "Analyze Metrics",
                        shortcut="Ctrl+Shift+S",
                        status_tip="Show Metrics",
                        callback=lambda: self.populate_graph())
        window.select_frame_signal.connect(self.viewer.seek_frame)

    def populate_graph(self):
        graph_window: MetricGraphWindow = self.get_window("graph_metrics")
        landmarks = self.viewer.get_landmarks ()
        graph_window.set_config(self.config, landmarks)
        graph_window.set_save_name(self.video_file)
        graph_window.show()

    def setup_metric_flow_window(self):
        self.add_window("metrics_flow", MetricFlow)
        self.add_action("&Metrics", "Advanced Metric Analysis",
                        shortcut="Ctrl+Shift+A",
                        status_tip="Does statistics based on a user defined flow diagram",
                        callback=self.display_flow)

    def display_flow(self):
        flow_window: MetricFlow = self.get_window("metrics_flow")
        flow_window.set_metric_data(self._calculated_metrics)
        flow_window.show()
        # grapher_window: MetricDisplayWindow = self.get_window("display_metrics")
        # flow_window.set_graph_callback(grapher_window.create_plot_advanced)

    def setup_landmark_detector_window(self):
        self.add_window("landmark_detector", DetectLandmarksWindow)
        self.add_action("&Landmarks", "Process Frames",
                        shortcut="Ctrl+A",
                        status_tip="Find facial landmarks for video",
                        callback=self.display_landmark_detector)

    def display_landmark_detector(self):
        detector_window: DetectLandmarksWindow = self.get_window("landmark_detector")
        if self.cap is None:
            return False
        detector_window.set_detector(cv2.VideoCapture(self.video_file), self.landmark_file)
        detector_window.show()
        detector_window.got_landmarks_signal.connect(self.on_got_landmarks)

    @QtCore.pyqtSlot(str, pd.DataFrame)
    def on_got_landmarks(self, save_path: str, data: pd.DataFrame):
        # TODO: Maybe make this concat with the old landmarks
        if not isinstance(self.landmarks_frame, pd.DataFrame):
            self.landmarks_frame = pd.DataFrame()
        try:
            self.landmarks_frame.set_index("Frame_number")
            data.set_index("Frame_number")
            new_frame: pd.DataFrame = pd.concat([self.landmarks_frame, data]).drop_duplicates(["Frame_number"], keep='last').sort_values(by=['Frame_number'], ascending=False)
            new_frame.to_csv(self.landmark_file)
            self.landmarks_frame = new_frame
            self.viewer.reset()
            self.viewer.set_reader(self.cap)
            self.viewer.set_landmarks(self.landmarks_frame)
        except Exception as e:
            print("Failed to save landmarks:", save_path)






if __name__ == '__main__':
    __spec__ = "ModuleSpec(name='builtins', loader=<class '_frozen_importlib.BuiltinImporter'>)"
    # freeze_support()

    if not QtWidgets.QApplication.instance():
        app = QtWidgets.QApplication(sys.argv)
    else:
        app = QtWidgets.QApplication.instance()

    app.setStyle(QtWidgets.QStyleFactory.create('Cleanlooks'))

    GUI = MainWindow()
    # GUI.show()
    GUI.showMaximized()
    app.exec_()
