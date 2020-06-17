import sys
import pandas as pd
import cv2
import time

import pkg_resources.py2_warn

from PyQt5 import QtWidgets, QtGui, QtCore, QtQuick
from PyQt5.QtQml import qmlRegisterType
from matplotlib_backend_qtquick.backend_qtquickagg import FigureCanvasQtQuickAgg
from uis.projectView import ProjectHandler
from uis.GraphView import GraphViewHandlerV2 as GraphViewHandler
from uis.LandmarkDetectionView import LandmarkDetectionHandler
from utils.Globals import Globals
from utils import DataHolders

from uis.VideoViewerPopup import VideoViewerWindowV2 as VideoViewerWindow

class MainWindow(VideoViewerWindow):
    _calculated_metrics: pd.DataFrame
    glo: Globals  # Comes from the superclass

    def __init__(self):
        super(MainWindow, self).__init__()
        self._calculated_metrics = pd.DataFrame()
        self.add_menu("&Metrics")
        self.add_menu("&Landmarks")
        self.setup_graph_window()
        self.setup_project_window()
        self.setup_detection_view()

    def setup_graph_window(self):
        # window: MetricGraphWindow = self.add_window("graph_metrics", MetricGraphWindow)
        # self.add_action("&Metrics", "Analyze Metrics",
        #                 shortcut="Ctrl+Shift+S",
        #                 status_tip="Show Metrics",
        #                 callback=lambda: self.populate_graph())
        # window.select_frame_signal.connect(self.viewer.seek_frame)
        window: GraphViewHandler = self.add_window("graph_handler", GraphViewHandler)
        self.add_action("&Metrics", "Analyze Metrics",
                        shortcut="Ctrl+Shift+S",
                        status_tip="Show Metrics",
                        callback=lambda: self.populate_graph())
        # window.select_frame_signal.connect(self.viewer.seek_frame)

    def populate_graph(self):
        graph_window: GraphViewHandler = self.get_window("graph_handler")
        graph_window.show()

    # Functions that handle initializing the project editor view
    def setup_project_window(self):
        window: ProjectHandler = self.add_window("project_handler", ProjectHandler)
        self.add_action("&File", "Edit Project",
                        callback=lambda: self.edit_project())
        self.add_action("&File", "New Project",
                        callback=lambda: self.new_project())

    def edit_project(self):
        window: ProjectHandler = self.get_window("project_handler")
        window.show()

    def new_project(self):
        window: ProjectHandler = self.get_window("project_handler")
        window.init_project(DataHolders.Project(""))
        window.show()

    def setup_detection_view(self):
        self.add_window("detection_handler", LandmarkDetectionHandler)
        self.add_action("&Landmarks", "Detect Landmarks",
                        callback=lambda: self.detect_landmarks())

    def detect_landmarks(self):
        window: LandmarkDetectionHandler = self.get_window("detection_handler")
        window.show()

    @QtCore.pyqtSlot(str, pd.DataFrame)
    def on_got_landmarks(self, save_path: str, data: pd.DataFrame):
        if not isinstance(self.landmarks_frame, pd.DataFrame):
            self.landmarks_frame = pd.DataFrame()
        if self.landmarks_frame.empty:
            self.landmarks_frame["Frame_number"] = []
        try:
            self.landmarks_frame.set_index("Frame_number")
            data.set_index("Frame_number")
            new_frame: pd.DataFrame = pd.concat([self.landmarks_frame, data]).drop_duplicates(["Frame_number"], keep='last').sort_values(by=['Frame_number'], ascending=False)
            new_frame.to_csv(self.landmark_file)
            self.landmarks_frame = new_frame
            self.viewer.reset()
            time.sleep(0.01)
            self.viewer.set_reader(self.cap, self.video_file)
            self.viewer.set_landmarks(self.landmarks_frame)
        except Exception as e:
            print(e)
            print("Failed to save landmarks:", save_path)


if __name__ == '__main__':
    print("App started")
    __spec__ = "ModuleSpec(name='builtins', loader=<class '_frozen_importlib.BuiltinImporter'>)"
    # freeze_support()

    if not QtWidgets.QApplication.instance():
        app = QtWidgets.QApplication(sys.argv)
    else:
        app = QtWidgets.QApplication.instance()

    print("Constructed app")
    app.setStyle(QtWidgets.QStyleFactory.create('Cleanlooks'))
    qmlRegisterType(FigureCanvasQtQuickAgg, "Backend", 1, 0, "FigureCanvas")

    GUI = MainWindow()
    # GUI.show()
    GUI.showMaximized()
    app.exec_()
