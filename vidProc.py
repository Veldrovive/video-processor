import os
import sys
import cv2
import pandas as pd
from typing import Tuple, Optional, Union

import vidViewer
import utils

from PyQt5 import QtWidgets, QtGui, QtCore

import numpy as np

from popups.MoveLandmarkPopup import MoveLandmarkWindow
from popups.DetectLandmarksPopup import DetectLandmarksWindow
from popups.MetricEvaluatePopup import MetricEvaluateWindow
from popups.MetricDisplayPopup import MetricDisplayWindow
from popups.MetricGraphPopup import MetricGraphWindow
from popups.ConfigPopup import ConfigWindow
from popups.MetricFlow import MetricFlow
import persistentConfig

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
        self.add_action("&Metrics", "Display Metrics V2",
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
        grapher_window: MetricDisplayWindow = self.get_window("display_metrics")
        flow_window.set_graph_callback(grapher_window.create_plot_advanced)

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
        detector_window.set_detector(self.cap, self.landmark_file)
        detector_window.show()
        detector_window.got_landmarks_signal.connect(self.on_got_landmarks)

    @QtCore.pyqtSlot(str, pd.DataFrame)
    def on_got_landmarks(self, save_path: str, data: pd.DataFrame):
        # TODO: Maybe make this concat with the old landmarks
        try:
            data.to_csv(save_path)
            self.landmarks_frame = data
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
