import os
import sys
import cv2
import pandas as pd
from typing import Tuple

import vidViewer
import utils

from PyQt5 import QtWidgets, QtGui, QtCore

from popups.MoveLandmarkPopup import MoveLandmarkWindow
from popups.DetectLandmarksPopup import DetectLandmarksWindow
from popups.ConfigPopup import ConfigWindow


class MainWindow(QtWidgets.QMainWindow):
    _file_path: str = None

    move_landmark_popup: MoveLandmarkWindow = None
    detect_landmarks_window: DetectLandmarksWindow = None
    config_window: ConfigWindow = None

    def __init__(self):
        super(MainWindow, self).__init__()
        self.setWindowTitle('Video Processing')

        # Set up a main view
        self.main_Widget = QtWidgets.QWidget(self)
        self.setCentralWidget(self.main_Widget)

        # Initialize Widgets
        self.viewer = vidViewer.ImageViewer()
        self.viewer.frame_change_signal.connect(self.slider_on_frame_change)
        self.viewer.metadata_update_signal.connect(self.on_vid_metadata_change)
        self.viewer.point_moved_signal.connect(self.on_point_moving)

        self.menuBar = QtWidgets.QMenuBar(self)
        self.setStyleSheet("""
                                           QMenuBar {
                                           font-size:18px;
                                           background : transparent;
                                           }
                                           """)

        # Tool bar Top - Functions to analyze the current image
        self.toolBar_Top = QtWidgets.QToolBar(self)

        # Tool bar Bottom  - Play/pause buttons
        self.toolBar_Bottom = QtWidgets.QToolBar(self)

        # Frame Slider Bottom  - Easily move between frames
        self.slider_bottom = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.slider_bottom.setMinimum(1)
        self.slider_bottom.setMaximum(100)
        self.slider_bottom.setValue(1)
        self.slider_bottom.setTickInterval(1)
        self.slider_bottom.setEnabled(False)
        self.slider_bottom.valueChanged.connect(self.slider_value_change)
        self.slider_bottom.sliderPressed.connect(self.slider_pressed)
        self.slider_bottom.sliderReleased.connect(self.slider_value_final)

        # Status Bar Bottom - Show the current frame number
        self.frameLabel = QtWidgets.QLabel('')
        self.frameLabel.setFont(QtGui.QFont("Times", 10))
        self.statusBar_Bottom = QtWidgets.QStatusBar()
        self.statusBar_Bottom.setFont(QtGui.QFont("Times", 10))
        self.statusBar_Bottom.addPermanentWidget(self.frameLabel)

        # initialize the User Interface
        self.initUI()
        self.show()

    def initUI(self):
        # Populate the different menus
        # File Menu
        file_menu = self.menuBar.addMenu("&File")

        load_video = file_menu.addAction("Load Video File")
        load_video.setShortcut("Ctrl+F")
        load_video.setStatusTip(
            'Load video file, accepted formats : .mp4, .avi, .mov')
        load_video.triggered.connect(self.select_video_file)

        save_csv_edits = file_menu.addAction("Save Landmark Edits")
        save_csv_edits.setShortcut("Ctrl+S")
        save_csv_edits.setStatusTip('Save your edits to the csv file')
        # TODO: Make this robust to saving before opening
        save_csv_edits.triggered.connect(lambda: self.viewer.save_edits(self._file_path[:-3]+"csv"))

        load_video = file_menu.addAction("Open Config")
        load_video.setShortcut("Ctrl+O")
        load_video.setStatusTip(
            'Open the configuration window')
        load_video.triggered.connect(self.open_config)

        quit_program = file_menu.addAction("Quit")
        quit_program.setShortcut("Ctrl+Q")

        #Video Menu
        video_menu = self.menuBar.addMenu("&Video")

        play_video = video_menu.addAction("Play Video")
        play_video.setShortcut("Ctrl+P")
        play_video.setStatusTip('Play video at given playback speed')
        play_video.triggered.connect(lambda: self.viewer.play())

        stop_video = video_menu.addAction("Stop Video")
        stop_video.setShortcut("Ctrl+L")
        stop_video.setStatusTip('Stop video playback')
        stop_video.triggered.connect(lambda: self.viewer.pause())

        jump_to_frame = video_menu.addAction("Jump to Frame")
        jump_to_frame.setShortcut("Ctrl+J")
        jump_to_frame.setStatusTip('Jump to certain frame')
        # jump_to_frame.triggered.connect()

        playback_settings = video_menu.addAction("Playback Settings")
        playback_settings.setShortcut("Ctrl+P")
        playback_settings.setStatusTip('Define video playback settings')
        # playback_settings.triggered.connect()

        # Landmarks Menu
        landmarks_menu = self.menuBar.addMenu("&Landmarks")

        process_current_frame = landmarks_menu.addAction("Process Current Frame")
        process_current_frame.setShortcut("Ctrl+C")
        process_current_frame.setStatusTip('Determine facial landmarks for current frame')
        # process_current_frame.triggered.connect(self.load_file)

        process_some_frame = landmarks_menu.addAction("Process Frames")
        process_some_frame.setShortcut("Ctrl+A")
        process_some_frame.setStatusTip(
            'Determine facial landmarks for some frames in the video')
        # process_some_frame.triggered.connect(self.process_frames)

        # Top toolbar population
        toggle_landmark = QtWidgets.QAction('Show/Hide facial landmarks', self)
        toggle_landmark.setIcon(QtGui.QIcon('./icons/facial-analysis.png'))
        toggle_landmark.triggered.connect(self.viewer.toggle_landmarks)

        show_metrics = QtWidgets.QAction(
            'Display facial metrics in current frame', self)
        show_metrics.setIcon(QtGui.QIcon('./icons/facial-metrics.png'))

        snapshot = QtWidgets.QAction('Save snapshot of current view', self)
        snapshot.setIcon(QtGui.QIcon('./icons/profile.png'))
        snapshot.triggered.connect(self.save_snapshot)

        self.toolBar_Top.addActions((toggle_landmark, show_metrics, snapshot))
        self.toolBar_Top.setIconSize(QtCore.QSize(50, 50))
        for action in self.toolBar_Top.actions():
            widget = self.toolBar_Top.widgetForAction(action)
            widget.setFixedSize(50, 50)
        self.toolBar_Top.setMinimumSize(self.toolBar_Top.sizeHint())
        self.toolBar_Top.setStyleSheet('QToolBar{spacing:8px;}')

        # Bottom toolbar population
        play_action = QtWidgets.QAction('Play', self)
        play_action.setShortcut('Shift+S')
        play_action.setIcon(QtGui.QIcon('./icons/play-arrow.png'))
        play_action.triggered.connect(lambda: self.viewer.play())

        stop_action = QtWidgets.QAction('Stop', self)
        stop_action.setShortcut('Shift+Z')
        stop_action.setIcon(QtGui.QIcon('./icons/pause.png'))
        stop_action.triggered.connect(lambda: self.viewer.pause())

        fastforward_action = QtWidgets.QAction('Jump Forward', self)
        fastforward_action.setShortcut('Shift+D')
        fastforward_action.setIcon(QtGui.QIcon('./icons/fast-forward.png'))
        fastforward_action.triggered.connect(lambda: self.viewer.jump_frames(1))

        rewind_action = QtWidgets.QAction('Jump Back', self)
        rewind_action.setShortcut('Shift+A')
        rewind_action.setIcon(QtGui.QIcon('./icons/rewind.png'))
        rewind_action.triggered.connect(lambda: self.viewer.jump_frames(-1))

        # spacer widget for left
        left_spacer = QtWidgets.QWidget(self)
        left_spacer.setSizePolicy(QtWidgets.QSizePolicy.Expanding,
                                  QtWidgets.QSizePolicy.Expanding)
        # spacer widget for right
        right_spacer = QtWidgets.QWidget(self)
        right_spacer.setSizePolicy(QtWidgets.QSizePolicy.Expanding,
                                   QtWidgets.QSizePolicy.Expanding)

        # fill the bottom toolbar
        self.toolBar_Bottom.addWidget(left_spacer)
        self.toolBar_Bottom.addActions(
            (rewind_action, play_action, stop_action, fastforward_action))
        self.toolBar_Bottom.addWidget(right_spacer)
        self.toolBar_Bottom.setIconSize(QtCore.QSize(35, 35))

        self.toolBar_Bottom.setMinimumSize(self.toolBar_Bottom.sizeHint())
        self.toolBar_Bottom.setStyleSheet('QToolBar{spacing:8px;}')

        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.menuBar)
        layout.addWidget(self.toolBar_Top)
        layout.addWidget(self.viewer)
        layout.addWidget(self.toolBar_Bottom)
        layout.addWidget(self.slider_bottom)
        self.setStatusBar(self.statusBar_Bottom)

        self.main_Widget.setLayout(layout)
        self.setGeometry(600, 100, self.sizeHint().width(),
                         self.sizeHint().height())

    # Video Interactions
    def select_video_file(self):
        filename, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, 'Load Video File',
            '', "Video files (*.mp4 *.avi *.mov *.MP4 *.AVI *.MOV)")
        if filename:
            filename = os.path.normpath(filename)
            self._file_path = filename
            landmark_file = filename[:-3] + 'csv'
            self.viewer.reset()
            self.open_video_file(filename)
            if os.path.exists(landmark_file):
                self.open_landmark_file(landmark_file)

    def open_video_file(self, file: str) -> bool:
        try:
            vid_cap = cv2.VideoCapture(file)
        except cv2.error as e:
            return False
        self.viewer.set_reader(vid_cap)
        return True

    def open_landmark_file(self, file: str) -> bool:
        try:
            landmarks = pd.read_csv(file)
        except Exception as e:
            # TODO: make this exception handling more specific
            print(e)
            return False
        self.viewer.set_landmarks(landmarks)
        return True

    def process_frames(self):
        self.detect_landmarks_window = DetectLandmarksWindow(self)
        self.detect_landmarks_window.show()

    def open_config(self):
        # Open Config Window
        self.config_window = ConfigWindow(self, self.viewer)
        self.config_window.show()
        # TODO: Make it so that you cannot open more than one config window

    def save_snapshot(self) -> bool:
        if self.viewer is None:
            return False
        snapshot = self.viewer.get_current_view()
        if snapshot is None:
            return False
        save_path = QtWidgets.QFileDialog.getSaveFileName(self, 'Save Snapshot', filter="Images (*.png *.jpg)")[0]
        bgr_img = cv2.cvtColor(snapshot, cv2.COLOR_RGB2BGR)
        cv2.imwrite(save_path, bgr_img)

    # Events:
    def resizeEvent(self, event):
        self.viewer.fitInView()
    @QtCore.pyqtSlot(int, float, object)
    def on_vid_metadata_change(self, length: int, fps: int, resolution: Tuple[int, int]):
        self.slider_bottom.setMinimum(1)
        self.slider_bottom.setMaximum(length)
        self.slider_bottom.setValue(1)
        self.slider_bottom.setTickInterval(1)
        self.slider_bottom.setEnabled(True)

    # Slider Interactions
    @QtCore.pyqtSlot()
    def slider_pressed(self):
        self._video_playing = self.viewer.is_playing()
        self.viewer.pause()

    @QtCore.pyqtSlot()
    def slider_value_final(self):
        frame = self.slider_bottom.value()
        self.viewer.seek_frame(frame-1)
        if self._video_playing:
            self.viewer.play()
            self._video_playing = False

    def slider_value_change(self):
        frame = self.slider_bottom.value()
        self.frameLabel.setText(
            'Frame : ' + str(int(frame)) + '/' + str(
                self.viewer._video_length))

    @QtCore.pyqtSlot(int)
    def slider_on_frame_change(self, frame: int):
        self.slider_bottom.setValue(frame+1)
        self.frameLabel.setText(
            'Frame : ' + str(int(frame) + 1) + '/' + str(
                self.viewer._video_length))

    # Editing interactions
    # @QtCore.pyqtSlot(int, utils.Landmark) # Errors for some reason
    def on_point_moving(self, moving: bool, landmark: utils.Landmark):
        if moving:
            # Then a point has just been picked up
            # TODO: Reuse the same popup to reduce the time it takes to show
            # self.move_landmark_popup = MoveLandmarkWindow(self, view=self.viewer, landmark=landmark)
            # self.move_landmark_popup.show()
            # self.raise_()
            # self.activateWindow()
            pass
        else:
            # Then a point has just been placed
            try:
                self.move_landmark_popup.close()
            except AttributeError as e:
                pass


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
