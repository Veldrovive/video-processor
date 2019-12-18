from PyQt5 import QtWidgets, QtCore, QtGui
from vidViewer import ImageViewer
import cv2
import pandas as pd
import persistentConfig
import os
import time

from typing import Tuple, Dict, List, Union, Optional

class VideoViewerWindow(QtWidgets.QMainWindow):
    menu_bar: QtWidgets.QMenuBar
    viewer: ImageViewer

    config: persistentConfig.Config

    # Slider
    slider_bottom: QtWidgets.QSlider
    video_length: int = -1

    # Frame bottom
    frame_label: QtWidgets.QLabel
    statusbar_bottom: QtWidgets.QStatusBar

    # Toolbars
    toolBar_Top: QtWidgets.QToolBar
    toolBar_Bottom: QtWidgets.QToolBar

    # All Actions:
    menus: Dict[Optional[str], Optional[QtWidgets.QMenu]]
    actions: Dict[str, QtWidgets.QAction]

    # Windows:
    windows: Dict[str, QtWidgets.QMainWindow]

    # Data
    cap: cv2.VideoCapture = None
    video_file: str = None
    landmarks_frame: pd.DataFrame = None
    landmark_file: str = None

    def __init__(self):
        super(VideoViewerWindow, self).__init__()
        self.setWindowTitle("Video Viewer")
        self.config = persistentConfig.Config()

        self.main_Widget = QtWidgets.QWidget(self)
        self.setCentralWidget(self.main_Widget)
        self.layout = QtWidgets.QVBoxLayout()
        self.main_Widget.setLayout(self.layout)

        self.menus = {None: None}
        self.actions = {}
        self.windows = {}

        self.setup_viewer()
        self.setup_menu()
        self.setup_bottom_bar()
        self.setup_toolbars()
        self.setup_slider()

        self.populate_layout()

        self.show()

    def setup_viewer(self):
        """
        Creates the viewer object used to display the video
        :return:
        """
        self.viewer = ImageViewer(config=self.config)
        self.viewer.frame_change_signal.connect(self.on_frame_change)
        self.viewer.metadata_update_signal.connect(self.on_vid_metadata_change)
        self.viewer.point_moved_signal.connect(self.on_point_moving)
        self.viewer.playback_change_signal.connect(self.on_vid_playback_change)

    def add_menu(self, name: str):
        """
        Creates a menu for the top bar
        :param name: The name of the menu and a way for the menu to be accessed
        :return:
        """
        self.menus[name] = self.menu_bar.addMenu(name)

    def add_action(self, menu: Optional[str], name: str, visible: bool = True, shortcut: str=None, status_tip: str=None, icon: str=None, callback=None):
        """
        Shorthand for action creation. Stores actions in the self.action dict
        :param menu: The menu the action should be added to
        :param name: A name by which the action can be accessed later
        :param visible: Whether the action should actually appear in the menu
        :param shortcut: The hotkeys used to activate the action
        :param status_tip: What to show when the action is hovered over
        :param icon: A visual icon to diplay with the action
        :param callback: A callback for when the action is activated
        :return:
        """
        if menu not in self.menus:
            self.add_menu(menu)
        if menu is None:
            action = QtWidgets.QAction(name)
        else:
            action = self.menus[menu].addAction(name)
        if shortcut is not None:
            action.setShortcut(shortcut)
        if status_tip is not None:
            action.setStatusTip(status_tip)
        if callback is not None:
            action.triggered.connect(callback)
        if icon is not None:
            action.setIcon(QtGui.QIcon(icon))
        action.setVisible(visible)
        self.actions[name] = action

    def setup_menu(self):
        """
        Initializes actions and bind them to their menus
        """
        self.menu_bar = QtWidgets.QMenuBar(self)
        self.setStyleSheet("""
                           QMenuBar {
                           font-size:18px;
                           background : transparent;
                           }
                           """)

        self.add_menu("&File")
        self.add_action("&File", "Load Video File",
                        shortcut="Ctrl+F",
                        status_tip="Load video file, accepted formats : .mp4, .avi, .mov",
                        callback=self.select_video_file)
        self.add_action("&File", "Save Landmark Edits",
                        shortcut="Ctrl+S",
                        status_tip="Save your edits to the csv file",
                        callback=self.save_landmark_edits)
        self.add_action("&File", "Open Config",
                        shortcut="Ctrl+O",
                        status_tip="Open the configuration window",
                        callback=self.open_config)
        self.add_action("&File", "Quit",
                        shortcut="Ctrl+Q",
                        status_tip="Quit the program",
                        callback=self.close)

        self.add_menu("&Video")
        self.add_action("&Video", "Play Video",
                        visible=True,
                        status_tip="Play the Video",
                        icon="./icons/play-arrow.png",
                        callback=self.viewer.play)
        self.add_action("&Video", "Pause Video",
                        visible=False,
                        status_tip="Pause video playback",
                        icon="./icons/pause.png",
                        callback=self.viewer.pause)
        self.add_action("&Video", "Toggle Playback",
                        visible=True,
                        shortcut="Space",
                        status_tip="Toggle video playback",
                        callback=self.viewer.toggle_play)
        self.add_action(None, "Seek Forward",
                        visible=True,
                        shortcut="right",
                        status_tip="Pause video playback",
                        icon="./icons/fast-forward.png",
                        callback=lambda: self.viewer.jump_frames(1))
        self.add_action(None, "Seek Back",
                        visible=True,
                        shortcut="left",
                        status_tip="Pause video playback",
                        icon="./icons/rewind.png",
                        callback=lambda: self.viewer.jump_frames(-1))

        self.add_menu("&Image")
        self.add_action("&Image", "Take Snapshot",
                        shortcut="Ctrl+P",
                        status_tip="Take a snapshot of the current display",
                        icon="./icons/profile.png",
                        callback=self.save_snapshot)
        self.add_action("&Image", "Toggle Landmarks",
                        shortcut="Ctrl+L",
                        status_tip="Show or Hide the landmarks",
                        callback=self.viewer.toggle_landmarks)
        self.add_action("&Image", "Toggle Metrics",
                        shortcut="Ctrl+M",
                        status_tip="Show or Hide the metrics",
                        callback=self.viewer.toggle_metrics)
        self.add_action("&Image", "Toggle Bounding Box",
                        shortcut="Ctrl+B",
                        status_tip="Show or Hide the Bounding Box",
                        callback=self.viewer.toggle_bounding_box)

    def setup_bottom_bar(self):
        """
        Initializes the frame's bottom bar
        """
        self.frame_label = QtWidgets.QLabel('')
        self.frame_label.setFont(QtGui.QFont("Times", 10))
        self.statusbar_bottom = QtWidgets.QStatusBar()
        self.statusbar_bottom.setFont(QtGui.QFont("Times", 10))
        self.statusbar_bottom.addPermanentWidget(self.frame_label)

    def setup_toolbars(self):
        """
        Links actions to the toolbars
        """
        # Tool bar Top - Functions to analyze the current image
        self.toolBar_Top = QtWidgets.QToolBar(self)
        # Tool bar Bottom  - Play/pause buttons
        self.toolBar_Bottom = QtWidgets.QToolBar(self)

        self.toolBar_Top.addActions(([self.actions["Take Snapshot"]]))
        self.toolBar_Top.setIconSize(QtCore.QSize(50, 50))
        for action in self.toolBar_Top.actions():
            widget = self.toolBar_Top.widgetForAction(action)
            widget.setFixedSize(50, 50)
        self.toolBar_Top.setMinimumSize(self.toolBar_Top.sizeHint())
        self.toolBar_Top.setStyleSheet('QToolBar{spacing:8px;}')

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
        self.toolBar_Bottom.addActions((self.actions["Seek Back"], self.actions["Play Video"], self.actions["Pause Video"], self.actions["Seek Forward"]))
        self.toolBar_Bottom.addWidget(right_spacer)
        self.toolBar_Bottom.setIconSize(QtCore.QSize(35, 35))

        self.toolBar_Bottom.setMinimumSize(self.toolBar_Bottom.sizeHint())
        self.toolBar_Bottom.setStyleSheet('QToolBar{spacing:8px;}')

    def setup_slider(self):
        """
        Creates the slider widget and binds its signals
        """
        self.slider_bottom = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.slider_bottom.setMinimum(1)
        self.slider_bottom.setMaximum(100)
        self.slider_bottom.setValue(1)
        self.slider_bottom.setTickInterval(1)
        self.slider_bottom.setEnabled(False)
        self.slider_bottom.valueChanged.connect(self.slider_value_change)
        self.slider_bottom.sliderPressed.connect(self.slider_pressed)
        self.slider_bottom.sliderReleased.connect(self.slider_value_final)

    def populate_layout(self):
        """
        Put all elements into the GUI
        """
        self.layout.addWidget(self.menu_bar)
        self.layout.addWidget(self.toolBar_Top)
        self.layout.addWidget(self.viewer)
        self.layout.addWidget(self.toolBar_Bottom)
        self.layout.addWidget(self.slider_bottom)
        self.setStatusBar(self.statusbar_bottom)

        self.setGeometry(600, 100, self.sizeHint().width(),
                         self.sizeHint().height())

    def add_window(self, name: str, window_object, *window_args):
        """
        Creates a new window that can be manipulated by the main thread
        :param name: A name used to access the window later
        :param window_object: The class that constructs the window
        :param window_args: Any arguments to be passed to the window on initialization
        :return:
        """
        self.windows[name] = window_object(self, *window_args)
        self.windows[name].hide()
        return self.get_window(name)

    def get_window(self, name: str) -> Optional:
        if name in self.windows:
            return self.windows[name]
        return None

    def open_file_dialog(self, title: str, allowed_types: Union[List[str], str], multi=False) -> Optional[Union[List[str], str]]:
        """
        Prompt the user to choose a file that will be opened
        :param title: The title of the window
        :param allowed_types: File types that may be selected
        :return: The file or files select or None
        """
        if isinstance(allowed_types, list):
            allowed_types = f"Types ({' '.join([f'*.{file_type}' for file_type in allowed_types])})"
        if multi:
            files = QtWidgets.QFileDialog.getOpenFileNames(self, title, filter=allowed_types)[0]
        else:
            files = QtWidgets.QFileDialog.getOpenFileName(self, title, filter=allowed_types)[0]
        return files if len(files) > 0 else None

    def save_file_dialog(self, title: str, allowed_types: Union[List[str], str]) -> Optional[str]:
        """
        Prompt the user to choose a file to save
        :param title: The title window
        :param type: The file type that will be saved. E.g. ['mp4', 'mov']
        :return: The path to the chosen file name or None
        """
        if isinstance(allowed_types, list):
            allowed_types = f"Types ({' '.join([f'*.{file_type}' for file_type in allowed_types])})"
        file = QtWidgets.QFileDialog.getSaveFileName(self, title, filter=allowed_types)[0]
        return file if len(file) > 0 else None

    @QtCore.pyqtSlot(int)
    def on_frame_change(self, frame: int):
        """
        Handles the actions that should be taken when the viewer moves to a new
        frame
        :param frame: The frame number the video has moved to
        :return:
        """
        # Human readable frame number of off by one of the machine one
        self.slider_bottom.setValue(frame + 1)
        self.frame_label.setText(
            'Frame : ' + str(int(frame) + 1) + '/' + str(
                self.video_length))

    @QtCore.pyqtSlot(int, float, object)
    def on_vid_metadata_change(self, length: int, frame_rate: float, resolution: Tuple[int, int]):
        """
        Handles actions that should be taken when the video changes
        :param length: The length of the video
        :param frame_rate: The frame rate of the video
        :param resolution: The resolution of the video
        :return:
        """
        # We dont need access to the frame rate or resolution, but we set the slider based on the length
        self.video_length = length
        self.slider_bottom.setMinimum(1)
        self.slider_bottom.setMaximum(length)
        self.slider_bottom.setValue(1)
        self.slider_bottom.setTickInterval(1)
        self.slider_bottom.setEnabled(True)

    @QtCore.pyqtSlot(bool)
    def on_vid_playback_change(self, playing: bool):
        """
        Handles actions that should be taken when the video is played or paused
        """
        # We want the user to only see the action that would change the video state
        self.actions["Play Video"].setVisible(not playing)
        self.actions["Pause Video"].setVisible(playing)

    @QtCore.pyqtSlot(bool, int)
    def on_point_moving(self, lifted: bool, landmark: int):
        """
        Handles actions that should be taken when a point is lifted or placed
        :param lifted: Whether a point has been lifted or placed
        :param landmark: The index of the landmark
        """
        pass

    @QtCore.pyqtSlot(int)
    def slider_value_change(self):
        """
        Handles actions that should be taken when the slider is moved
        """
        frame = self.slider_bottom.value()
        # Immediately change the bottom label, but don't update the viewer as that would be expensive
        self.frame_label.setText('Frame : ' + str(int(frame)) + '/' + str(self.video_length))

    @QtCore.pyqtSlot()
    def slider_pressed(self):
        """
        Handles actions that should be taken when the slider is pressed
        """
        self.viewer.pause()

    @QtCore.pyqtSlot()
    def slider_value_final(self):
        """
        Handles actions that should be taken when the user releases the slider
        """
        frame = self.slider_bottom.value()
        # Frame is in human readable so we convert back to start=0
        self.viewer.seek_frame(frame - 1)
        # Ugly work around to an issue with the thread updating the current frame too late
        # TODO: Make the thread immediately update its current frame in this case
        time.sleep(1)
        self.viewer.play(cond_was_playing=True)

    @QtCore.pyqtSlot()
    def save_landmark_edits(self):
        """
        Handles actions that should be taken when the user wants to save their edits
        """
        frame = self.viewer.get_edits()
        if self.landmark_file is None:
            return False
        if os.path.exists(self.landmark_file):
            name, ext = os.path.splitext(self.landmark_file)
            os.rename(self.landmark_file, name+"_orig"+ext)
        frame.to_csv(self.landmark_file, index=False)

    @QtCore.pyqtSlot()
    def save_snapshot(self):
        """
        Handles actions that should be taken when the user takes a snapshot
        """
        if self.viewer is None:
            return False
        snapshot = self.viewer.get_current_view()
        if snapshot is None:
            return False
        save_path = self.save_file_dialog("Save Snapshot", ["png", "jpg"])
        # imwrite saves in BGR so we have to convert back
        bgr_img = cv2.cvtColor(snapshot, cv2.COLOR_RGB2BGR)
        if save_path is not None:
            cv2.imwrite(save_path, bgr_img)

    @QtCore.pyqtSlot()
    def open_config(self):
        """
        Handles actions that should be taken when the user opens the config menu
        """
        print("Open Config event should be implemented")
        pass

    @QtCore.pyqtSlot()
    def select_video_file(self):
        """
        Handles actions that should be taken when the user selects a video
        """
        self.video_file = self.open_file_dialog("Select Video File", ["mp4", "avi", "mov", "MP4", "AVI", "MOV"], multi=False)
        if self.video_file is None:
            return False
        self.landmark_file = os.path.splitext(self.video_file)[0] + ".csv"
        self.cap = self.open_video_file(self.video_file)
        if self.cap is None:
            return False
        self.viewer.reset()
        self.viewer.set_reader(self.cap)
        self.landmarks_frame = self.open_landmark_file(self.landmark_file)
        self.viewer.set_landmarks(self.landmarks_frame)

    def resizeEvent(self, event):
        # Resizes the video when the user changes video size
        self.viewer.fitInView()
        return super(VideoViewerWindow, self).resizeEvent(event)

    def event(self, event: QtCore.QEvent):
        # Catches all events since windowActivate event did not seem to work
        if event.type() == 25 or event.type() == 24:
            # These events catch focus and defocus of the window
            # This solves a problem where shift would be stored
            self.viewer.reset_held_keys()
        return super(VideoViewerWindow, self).event(event)

    def open_video_file(self, file: str) -> Optional[cv2.VideoCapture]:
        """
        Tries to open the video stream from a file
        :param file: The Video file path
        :return: The video capture object or None
        """
        try:
            cap = cv2.VideoCapture(file)
        except cv2.error as e:
            return None
        if cap is None or not cap.isOpened():
            return None
        return cap

    def open_landmark_file(self, file: str) -> pd.DataFrame:
        """
        Tries to open a landmark file and returns an empty DataFrame if
        one is not found
        :param file: The landmark file path
        :return: A DataFrame containing the landmarks if the file exists or an empty DataFrame
        """
        try:
            landmarks = pd.read_csv(file)
        except Exception as e:
            # TODO: make this exception handling more specific
            landmarks = pd.DataFrame()
        return landmarks
