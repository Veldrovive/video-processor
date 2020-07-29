from PyQt5 import QtWidgets, QtCore, QtGui
from vidViewer import ImageViewerV2 as ImageViewer
from utils.qmlBase import WindowHandler
from PyQt5.QtQml import qmlRegisterType, QQmlComponent, QQmlEngine, QQmlContext
from utils import DataHolders
from utils.Globals import Globals
import cv2
import pandas as pd
import os
import time

from typing import Tuple, Dict, List, Union, Optional

class MarkedSlider(QtWidgets.QSlider):
    _glo: Globals
    tick_height_percent: float = 0.7
    snap_radius: int = 10

    def __init__(self, parent=None, glo: Globals = None):
        self._glo = glo
        super(MarkedSlider, self).__init__(parent)

    def mouseReleaseEvent(self, ev: QtGui.QMouseEvent) -> None:
        """
        Catches when a new position on the slider is selected. This allows us
        to snap to the nearest tickmark.
        """
        if self._glo.video_config is not None:
            val_list = self._glo.video_config.get_keypoints()
            if len(val_list) > 0:
                curr_pos = self.value() - 1
                dists = [abs(pos-curr_pos) for pos in val_list]
                if min(dists) <= self.snap_radius:
                    snap_index = dists.index(min(dists))
                    snap_frame = val_list[snap_index] + 1
                    self.setValue(snap_frame)
        self.repaint()
        super().mouseReleaseEvent(ev)

    def frame_to_pos(self, frame: int):
        """
        Gets the x position on the bar that corresponds to the input frame
        """
        contents = self.contentsRect()
        # The bounding box isn't quite correct so some reason so we shift the min and max x values
        x_min = contents.x() + 7
        x_max = (contents.x() + contents.width()) - 13
        scaled = int(round((frame + self.minimum()) * ((x_max - x_min) / (self.maximum() - self.minimum()))))
        return x_min + scaled

    def radius_to_y_diff(self, radius: int):
        contents = self.contentsRect()
        return int(round(contents.y() + contents.height() / 2 + radius))

    def paintEvent(self, event):
        """
        We catch the paint event in order to introduce the new feature of
        tickmarks. These act to designate the position of key points in the
        video that the user wants to return to multiple times.
        """
        curr_pos = self.value() - 1
        qp = QtGui.QPainter(self)
        pen = QtGui.QPen()
        pen.setColor(QtCore.Qt.gray)
        qp.setPen(pen)
        contents = self.contentsRect()
        if self._glo.curr_landmarks is not None:
            # Then we draw an orange outline around the frames that have landmarks
            pen.setColor(QtGui.QColor("#f39c12"))
            pen.setWidth(6)
            qp.setPen(pen)
            ranges = self._glo.curr_landmarks.calculate_landmark_ranges()
            for range in ranges:
                if range[1] - range[0] == 0:
                    qp.drawLine(self.frame_to_pos(range[0]) - 1, self.radius_to_y_diff(0), self.frame_to_pos(range[1]) + 1, self.radius_to_y_diff(0), )
                else:
                    qp.drawLine(self.frame_to_pos(range[0]), self.radius_to_y_diff(0), self.frame_to_pos(range[1]), self.radius_to_y_diff(0))
        if self._glo.video_config is not None:
            # Then we draw tick marks at the keypoints
            pen.setWidth(2)
            val_list = self._glo.video_config.get_keypoints()
            pad_amount = int(round((contents.height() * (1-self.tick_height_percent))/2))
            for val in val_list:
                if val == curr_pos:
                    pen.setColor(QtGui.QColor("#e74c3c"))
                else:
                    pen.setColor(QtGui.QColor("#95a5a6"))
                qp.setPen(pen)
                x_pos = self.frame_to_pos(val)
                qp.drawLine(x_pos, contents.y() + pad_amount, x_pos, contents.y()+contents.height()-pad_amount)
        super(MarkedSlider, self).paintEvent(event)

class VideoViewerWindowV2(QtWidgets.QMainWindow):
    menu_bar: QtWidgets.QMenuBar
    viewer: ImageViewer

    glo: Globals
    engine: QQmlEngine

    # Slider
    slider_bottom: MarkedSlider
    video_length: int = -1

    # Frame bottom
    frame_label: QtWidgets.QLabel
    video_index_label: QtWidgets.QLabel
    video_name_label: QtWidgets.QLabel
    statusbar_bottom: QtWidgets.QStatusBar

    # Toolbar
    toolBar_Bottom: QtWidgets.QToolBar

    # All Actions:
    menus: Dict[Optional[str], Optional[QtWidgets.QMenu]]
    actions: Dict[str, QtWidgets.QAction]
    shortcuts: Dict[str, QtWidgets.QShortcut]

    # Windows:
    windows: Dict[str, QtWidgets.QMainWindow]

    # Data
    cap: cv2.VideoCapture = None
    video_file: str = None
    landmarks_frame: pd.DataFrame = None
    landmark_file: str = None

    def __init__(self):
        super(VideoViewerWindowV2, self).__init__()
        self.setWindowTitle("Video Viewer")
        self.engine = QQmlEngine()

        self.setup_globals()

        self.edited_frames_file = "./edited_frames.csv"
        self.edited_frames = pd.read_csv(self.edited_frames_file) if os.path.isfile(self.edited_frames_file) else pd.DataFrame(columns=["vidFile", "frameNumber"])

        self.main_Widget = QtWidgets.QWidget(self)
        self.setCentralWidget(self.main_Widget)
        self.layout = QtWidgets.QVBoxLayout()
        self.main_Widget.setLayout(self.layout)

        self.menus = {None: None}
        self.actions = {}
        self.shortcuts = {}
        self.windows = {}

        self.setup_viewer()
        self.setup_menu()
        self.setup_bottom_bar()
        self.setup_toolbars()
        self.setup_slider()

        self.populate_layout()

        self.show()
        self.glo.onConfigChange.connect(lambda: self.slider_bottom.update())
        self.glo.onFileChange.connect(self.on_video_change)

    def setup_globals(self):
        """
        Create the globals object and fill it with default values
        :return: Void
        """
        self.glo = Globals()
        # self.glo.onProjectChange.connect()

    def setup_viewer(self):
        """
        Creates the viewer object used to display the video
        :return:
        """
        self.viewer = ImageViewer()
        self.viewer.frame_change_signal.connect(self.on_frame_change)
        self.viewer.playback_change_signal.connect(self.on_vid_playback_change)
        self.glo.onFileChange.connect(self.on_vid_metadata_change)

    def add_menu(self, name: str):
        """
        Creates a menu for the top bar
        :param name: The name of the menu and a way for the menu to be accessed
        :return:
        """
        self.menus[name] = self.menu_bar.addMenu(name)

    def add_menu_seperator(self, name: str) -> bool:
        """
        Adds a visual seperator into a menu
        :param name: The name of the menu
        """
        if name in self.menus:
            self.menus[name].addSeparator()
            return True
        return False

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
            action = QtWidgets.QAction(name, self)
        else:
            action = self.menus[menu].addAction(name)
        if shortcut is not None:
            if menu is None:
                shortcut = QtWidgets.QShortcut(QtGui.QKeySequence(shortcut), self)
                shortcut.activated.connect(callback)
                self.shortcuts[name] = shortcut
            else:
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
        self.add_action("&File", "Load Project",
                        shortcut="Ctrl+F",
                        status_tip="Load project directory. Accepted formats : .mp4, .avi, .mov",
                        callback=self.open_project_dir)
        self.add_action(None, "Next Video",
                        shortcut="Ctrl+Shift+right",
                        status_tip="Move to the next video in the project",
                        icon="./icons/forward.svg",
                        callback=lambda: self.glo.select_file(self.glo.curr_file_index + 1))
        self.add_action(None, "Previous Video",
                        shortcut="Ctrl+Shift+left",
                        status_tip="Move to the previous video in the project",
                        icon="./icons/back.svg",
                        callback=lambda: self.glo.select_file(self.glo.curr_file_index - 1))
        self.add_action("&File", "Quit",
                        shortcut="Ctrl+Q",
                        status_tip="Quit the program",
                        callback=self.close)

        self.add_menu("&View")
        self.add_action(None, "Play",
                        visible=True,
                        status_tip="Play the Video",
                        icon="./icons/play-arrow.png",
                        callback=self.viewer.play)
        self.add_action(None, "Pause",
                        visible=False,
                        status_tip="Pause video playback",
                        icon="./icons/pause.png",
                        callback=self.viewer.pause)
        self.add_action(None, "Toggle Playback",
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

        def jump_forward():
            curr_frame = self.viewer.get_curr_frame()
            next_keypoint = self.glo.video_config.get_next_keypoint(curr_frame)
            self.viewer.pause()
            self.viewer.seek_frame(next_keypoint)
        self.add_action(None, "Jump Forward",
                        visible=True,
                        shortcut="Shift+right",
                        status_tip="Jump to the next keypoint",
                        icon="./icons/fast-forward.png",
                        callback=jump_forward)
        self.add_action(None, "Seek Back",
                        visible=True,
                        shortcut="left",
                        status_tip="Pause video playback",
                        icon="./icons/rewind.png",
                        callback=lambda: self.viewer.jump_frames(-1))

        def jump_backwards():
            curr_frame = self.viewer.get_curr_frame()
            last_keypoint = self.glo.video_config.get_previous_keypoint(curr_frame)
            self.viewer.pause()
            self.viewer.seek_frame(last_keypoint)
        self.add_action(None, "Jump Backward",
                        visible=True,
                        shortcut="Shift+left",
                        status_tip="Jump to the previous keypoint",
                        icon="./icons/rewind.png",
                        callback=jump_backwards)
        self.add_action("&View", "Add Keypoint",
                        visible=True,
                        shortcut="Ctrl+k",
                        callback=lambda: self.glo.video_config.toggle_keypoint(self.viewer.get_curr_frame()))

        self.add_action("&View", "Take Snapshot",
                        shortcut="Ctrl+P",
                        status_tip="Take a snapshot of the current display",
                        callback=self.save_snapshot)
        self.add_menu_seperator("&View")
        self.add_action("&View", "Toggle Bounding Box",
                        shortcut="Ctrl+B",
                        status_tip="Show or Hide the Bounding Box",
                        callback=lambda: self.glo.visual_config.toggle_bounding())
        self.add_action("&View", "Toggle Landmarks",
                        shortcut="Ctrl+L",
                        status_tip="Show or Hide the landmarks",
                        callback=lambda: self.glo.visual_config.toggle_landmarks())
        self.add_action("&View", "Toggle Metrics",
                        shortcut="Ctrl+M",
                        status_tip="Show or Hide the metrics",
                        callback=lambda: self.glo.visual_config.toggle_metrics())
        self.add_menu_seperator("&View")

        self.add_menu("&Analysis")

    def setup_bottom_bar(self):
        """
        Initializes the frame's bottom bar
        """
        self.frame_label = QtWidgets.QLabel('')
        self.frame_label.setFont(QtGui.QFont("Times", 14))
        self.video_index_label = QtWidgets.QLabel('')
        self.video_index_label.setFont(QtGui.QFont("Times", 14))
        self.video_name_label = QtWidgets.QLabel('')
        self.statusbar_bottom = QtWidgets.QStatusBar()
        self.statusbar_bottom.setFont(QtGui.QFont("Times", 14))
        self.statusbar_bottom.addPermanentWidget(self.video_index_label)
        self.statusbar_bottom.addPermanentWidget(self.frame_label)
        self.statusbar_bottom.addWidget(self.video_name_label)

    def setup_toolbars(self):
        """
        Links actions to the toolbars
        """
        # Tool bar Bottom  - Play/pause buttons
        self.toolBar_Bottom = QtWidgets.QToolBar(self)

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
        self.toolBar_Bottom.addActions((self.actions["Previous Video"], self.actions["Seek Back"], self.actions["Play"], self.actions["Pause"], self.actions["Seek Forward"], self.actions["Next Video"]))
        self.toolBar_Bottom.addWidget(right_spacer)
        self.toolBar_Bottom.setIconSize(QtCore.QSize(35, 35))

        self.toolBar_Bottom.setMinimumSize(self.toolBar_Bottom.sizeHint())
        self.toolBar_Bottom.setStyleSheet('QToolBar{spacing:8px;}')

    def setup_slider(self):
        """
        Creates the slider widget and binds its signals
        """
        # self.slider_bottom = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.slider_bottom = MarkedSlider(QtCore.Qt.Horizontal, self.glo)
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
        if issubclass(window_object, WindowHandler):
            # Then this is a qml window and needs the engine instance
            self.windows[name] = window_object(self.engine, *window_args)
        else:
            # Then this is a classic qml window
            self.windows[name] = window_object(self, *window_args)
        self.windows[name].hide()
        return self.get_window(name)

    def add_window_qml(self):
        """
        Creates a new window managed by qtQuick

        :return:
        """

    def get_window(self, name: str) -> Optional:
        """
        Gets the window object by its name
        :param name: The name assigned by the user when the window was added
        :return: The window object with the given name
        """
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

    def open_project_dialog(self, title: str) -> str:
        """
        Prompt the user to choose a file that will be opened
        :param title: The title of the window
        :return: The project directory
        """
        p_dir = str(QtWidgets.QFileDialog.getExistingDirectory(self, title))
        return p_dir

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
        self.frame_label.setText(f"Frame: {frame + 1}/{int(self.video_length)}")

    @QtCore.pyqtSlot()
    def on_video_change(self):
        self.video_index_label.setText(f"Video: {self.glo.curr_file_index + 1}/{len(self.glo.project.files_map)}")
        self.video_name_label.setText(f"Video: {os.path.basename(self.glo.curr_file)}")
        self.setWindowTitle(f"Current Video: {os.path.basename(self.glo.curr_file)}")
        # self.statusbar_bottom.showMessage(f"New Video: {os.path.basename(self.glo.curr_file)}", 5000)

    @QtCore.pyqtSlot()
    def on_vid_metadata_change(self):
        """
        Handles actions that should be taken when the video changes
        :param length: The length of the video
        :param frame_rate: The frame rate of the video
        :param resolution: The resolution of the video
        :return:
        """
        # We dont need access to the frame rate or resolution, but we set the slider based on the length
        length = self.glo.visual_config.video_length
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
        self.actions["Play"].setVisible(not playing)
        self.actions["Pause"].setVisible(playing)

    @QtCore.pyqtSlot(bool, int, int, DataHolders.Landmarks)
    def on_point_moving(self, lifted: bool, landmark: int, frame: int, landmarks: DataHolders.Landmarks):
        """
        Handles actions that should be taken when a point is lifted or placed
        :param lifted: Whether a point has been lifted or placed
        :param landmark: The index of the landmark
        :param frame: The frame where the landmark was moved
        """
        # TODO: Implement this with groups in mind
        if not lifted:
            new_locations = landmarks.get_landmarks(frame)
            new_row = {
                "vidFile": self.video_file,
                "frameNumber": frame,
            }
            for pos, group, index in new_locations:
                new_row[f"landmark_{index}_x"] = pos[0]
                new_row[f"landmark_{index}_y"] = pos[1]
                if f"landmark_{index}_x" not in self.edited_frames.columns:
                    self.edited_frames[f"landmark_{index}_x"] = pd.Series()
                if f"landmark_{index}_y" not in self.edited_frames.columns:
                    self.edited_frames[f"landmark_{index}_y"] = pd.Series()
            if ((self.edited_frames["frameNumber"] == frame) & (self.edited_frames["vidFile"] == self.video_file)).any():
                print("Frame already in edited landmarks")
                # self.edited_frames.loc[(self.edited_frames["frameNumber"] == frame) & (self.edited_frames["vidFile"] == self.video_file)] = pd.DataFrame(new_row, index=["frameNumber"])
                self.edited_frames.loc[(self.edited_frames.frameNumber == frame) & (self.edited_frames.vidFile == self.video_file), self.edited_frames.columns] = list(new_row.values())
            else:
                print("Appending frame as new edit")
                self.edited_frames = self.edited_frames.append(new_row, ignore_index=True)
            # self.edited_frames.to_csv(self.edited_frames_file, index=None)
            print("Editing")

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
        self.viewer.set_reader(self.cap, self.video_file)
        self.landmarks_frame = self.open_landmark_file(self.landmark_file)
        self.viewer.set_landmarks(self.landmarks_frame)

    @QtCore.pyqtSlot()
    def open_project_dir(self) -> bool:
        """
        Opens a project stored on disk through its directory
        :param p_dir: The path to the project directory
        :return: Whether the opening was successful
        """
        p_dir = self.open_project_dialog("Open Project")
        print(p_dir)
        if self.glo.select_project(p_dir):
            # Then the project was opened correctly
            return True
        else:
            # Then the project failed to open
            return False

    def resizeEvent(self, event):
        # Resizes the video when the user changes video size
        self.viewer.fitInView()
        return super(VideoViewerWindowV2, self).resizeEvent(event)

    def event(self, event: QtCore.QEvent):
        # Catches all events since windowActivate event did not seem to work
        if event.type() == 25 or event.type() == 24:
            # These events catch focus and defocus of the window
            # This solves a problem where shift would be stored
            self.viewer.reset_held_keys()
        return super(VideoViewerWindowV2, self).event(event)

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
            landmarks = pd.read_csv(file, index_col=0)
        except Exception as e:
            # TODO: make this exception handling more specific
            landmarks = pd.DataFrame()
        return landmarks
