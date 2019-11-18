from PyQt5 import QtWidgets, QtGui, QtCore
import os
import numpy as np
import cv2
import pandas as pd
import utils
from scipy.spatial.distance import cdist
from typing import Dict, List, Tuple, Union, Optional
from enum import Enum

# My motivation for this class is currently that it will hold references to all
# information required for displaying such as the landmarks and video array.
# It will not actually load any of this from memory. Instead that will happen
# in a main file. This will deal with the displaying and callbacks that the main
# file needs

# It may also be nice to store all actions taken so that a ctl+z action could
# be implemented, but that could be done later as long as I keep this general

class Mode(Enum):
    EDIT = 1

class ImageViewer(QtWidgets.QGraphicsView):
    _scale_factor: float = 0.1  # Defines the zoom speed
    _zoom: int = 0
    _scene: Optional[QtWidgets.QGraphicsScene] = None
    _display: Optional[QtWidgets.QGraphicsPixmapItem] = None
    _landmarks: Optional[pd.DataFrame] = None  # Holds all landmark points
    _landmarks_editing: Optional[pd.DataFrame] = None # A copy of landmarks for editing
    _groups: Dict[str, List[int]] = {}  # Used to clarify which points belong to which features
    _video_reader: Optional[cv2.VideoCapture] = None
    _video_length: int = -1
    _video_fps: int = -1
    _playback_speed = 1
    _resolution: Tuple[int, int] = (-1, -1) # Stored as (width, height)

    _playback_timer: QtCore.QTimer

    _curr_frame: Optional[np.ndarray] = None
    _curr_landmarks: Optional[utils.FaceLandmarks] = None # Holds landmarks for the current frame
    _show_landmarks: bool = True
    _lifted_point: Optional[utils.Landmark] = None
    _eyes_lifted: Dict[str, bool] = {"left": False, "right": False}

    _mode: Mode = Mode.EDIT

    frame_change_signal = QtCore.pyqtSignal(int) # Emitted when frame number changes
    playback_change_signal = QtCore.pyqtSignal(bool) # Emitted when video is paused or played
    metadata_update_signal = QtCore.pyqtSignal(int, float, object) # Emits length, frame rate, resolution
    point_moved_signal = QtCore.pyqtSignal(bool, utils.Landmark) # Emitted when a landmark is moved

    def __init__(self, reader: cv2.VideoCapture=None, landmarks: pd.DataFrame=None, max_size: Tuple[int, int]=(-1, -1)):
        super(ImageViewer, self).__init__()
        self._video_reader = reader
        self._landmarks = landmarks
        self._playback_timer = QtCore.QTimer()
        self._scale_factor = 0.1
        self.max_size = max_size
        self._playback_timer.timeout.connect(self.display_next_frame)
        self.setup_view_window()
        self.setup_default_groups()
        self.load_metadata()

        self.display_next_frame()
        self.auto_resize_view()

        self.frame_change_signal.connect(self.on_frame_change)

    def setup_view_window(self):
        self._scene = QtWidgets.QGraphicsScene(self)
        self._display = QtWidgets.QGraphicsPixmapItem()
        self._scene.addItem(self._display)
        self.setScene(self._scene)
        self.setTransformationAnchor(QtWidgets.QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QtWidgets.QGraphicsView.AnchorUnderMouse)
        self.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.setBackgroundBrush(QtGui.QBrush(QtGui.QColor(100,100,100)))
        self.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.setDragMode(QtWidgets.QGraphicsView.RubberBandDrag)
        self.setMouseTracking(True)

    def add_to_group(self, name: str, indices: List[int]):
        indices.sort()
        indices = set(indices)
        if name not in self._groups:
            self._groups[name] = []
        for index in indices:
            for group in self._groups:
                try:
                    self._groups[group].remove(index)
                except ValueError as e:
                    pass
            self._groups[name].append(index)

    def setup_default_groups(self):
        self._groups["face"] = list(range(1, 69))
        self.add_to_group("right_eye", list(range(37, 42 + 1)))
        self.add_to_group("left_eye", list(range(43, 48 + 1)))
        self.add_to_group("nose", list(range(28, 36 + 1)))
        self.add_to_group("inner_mouth", list(range(61, 68 + 1)))
        self.add_to_group("outer_mouth", list(range(49, 60 + 1)))
        self.add_to_group("right_eyebrow", list(range(18, 22 + 1)))
        self.add_to_group("left_eyebrow", list(range(23, 27 + 1)))
        self.add_to_group("chin_outline", list(range(1, 17 + 1)))

    def set_vid_capture(self, reader: cv2.VideoCapture):
        self._video_reader = reader
        self.load_metadata()
        self.seek_frame(0)
        self.auto_resize_view()
        self.fitInView()

    def set_landmarks(self, landmarks: pd.DataFrame=None):
        self._landmarks = landmarks
        self._landmarks_editing = landmarks.copy()

    def toggle_landmarks(self):
        self._show_landmarks = not self._show_landmarks
        self.update_curr_frame()

    def set_max_size(self, max_size: Tuple[int, int]):
        self.max_size = max_size
        self.auto_resize_view()

    def get_groups(self):
        return self._groups.copy()


    # These functions deal with displaying our frames
    def set_display(self, image=None):
        height, width, channels = image.shape
        bytesPerLine = 3 * width
        qt_img = QtGui.QImage(image.data, width, height, bytesPerLine, QtGui.QImage.Format_RGB888)
        pixmap = QtGui.QPixmap.fromImage(qt_img)

        # Updates the view with the new pixmap
        if pixmap and not pixmap.isNull():
            self.setDragMode(QtWidgets.QGraphicsView.RubberBandDrag)
            self._display.setPixmap(pixmap)
        else:
            self.setDragMode(QtWidgets.QGraphicsView.NoDrag)
            self._display.setPixmap(QtGui.QPixmap())

    def get_frame_landmarks(self, frame_num: int) -> Optional[pd.DataFrame]:
        if not self._show_landmarks:
            return None
        try:
            landmarks = self._landmarks_editing.loc[self._landmarks_editing["Frame_number"] == frame_num]
        except (ValueError, KeyError, AttributeError) as e:
            landmarks = None
        return landmarks

    def display_next_frame(self, excluded_landmarks=None) -> bool:
        frame_num = self.get_frame_num()
        self.frame_change_signal.emit(frame_num)
        if not self.is_valid_frame(frame_num) and self._playback_timer.isActive():
            self._playback_timer.stop()
            return False
        if not self.cap_is_good():
            if self._playback_timer.isActive():
                self._playback_timer.stop()
            return False
        self._curr_frame = self.read_frame()
        landmarks = self.get_frame_landmarks(frame_num)

        if landmarks is not None and not landmarks.empty:
            self._curr_landmarks = utils.landmark_frame_to_shapes(landmarks, self._groups)
            test_highlight = [list(range(10)), [(250, 234, 62) for i in range(10)]]
            frame = utils.markup_image(self._curr_frame, self._curr_landmarks, color_override=test_highlight, excluded_landmarks=excluded_landmarks)
        else:
            frame = self._curr_frame

        self.set_display(frame)
        return True

    def update_curr_frame(self, excluded_landmarks=None) -> bool:
        landmarks = self.get_frame_landmarks(self.get_frame_num())
        if landmarks is not None and not landmarks.empty:
            self._curr_landmarks = utils.landmark_frame_to_shapes(landmarks, self._groups)
            test_highlight = [list(range(10)), [(250, 234, 62) for i in range(10)]]
            frame = utils.markup_image(self._curr_frame, self._curr_landmarks, color_override=test_highlight, excluded_landmarks=excluded_landmarks)
        else:
            frame = self._curr_frame
        self.set_display(frame)
        return True


    # These functions deal with video control
    def play(self, frame_num=None) -> bool:
        if not self.is_valid_frame(self.get_frame_num()):
            return False
        if frame_num is not None:
            if self.is_valid_frame(frame_num):
                self.seek_frame(frame_num)
            else:
                return False
        if self.cap_is_good():
            self._playback_timer.start(1000.0/(self._video_fps*self._playback_speed))
            self.playback_change_signal.emit(True)
        return True

    def pause(self) -> bool:
        if self._playback_timer.isActive():
            self._playback_timer.stop()
            self.playback_change_signal.emit(False)
            return True
        return False

    def is_playing(self):
        return self._playback_timer.isActive()

    def set_playback_speed(self, speed: float) -> bool:
        curr_status = self._playback_timer.isActive()
        if curr_status:
            self._playback_timer.stop()
        self._playback_speed = speed
        if curr_status:
            self.play()
        return True

    def jump_frames(self, frame_jump=1):
        self.pause()
        next_frame = self.get_frame_num()+frame_jump
        seek_res = self.seek_frame(next_frame)
        if not seek_res:
            return False
        return True


    # These functions deal with opencv interaction
    def cap_is_good(self) -> bool:
        return self._video_reader is not None and self._video_reader.isOpened()

    def read_frame(self) -> Union[np.ndarray, bool]:
        if self.cap_is_good():
            grabbed, frame = self._video_reader.read()
            return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) if grabbed else False
        return False

    def seek_frame(self, frame_num: int) -> bool:
        if self.is_valid_frame(frame_num):
            # For some reason, video_reader set takes a long time
            self._video_reader.set(1, frame_num)  # CAP_PROP_POS_FRAMES = 1
            self.frame_change_signal.emit(frame_num)
            self.display_next_frame()
            self._video_reader.set(1, frame_num)
            return True
        return False

    def is_valid_frame(self, frame_num: int) -> bool:
        before_end = frame_num < self._video_length
        after_beginning = frame_num >= 0
        return before_end and after_beginning

    def get_frame_num(self) -> int:
        if self.cap_is_good():
            return self._video_reader.get(1)  # CAP_PROP_POS_FRAMES = 1
        return -1

    def load_metadata(self) -> bool:
        if self.cap_is_good():
            self._video_length = int(self._video_reader.get(7))  # CAP_PROP_FRAME_COUNT = 7
            self._video_fps = int(self._video_reader.get(5))  # CAP_PROP_FPS = 5
            self._resolution = (int(self._video_reader.get(3)), int(self._video_reader.get(4)))
            self.metadata_update_signal.emit(self._video_length, self._video_fps,
                                        self._resolution)
            return True
        return False


    # These functions handle the sizing of the video
    # Capitalization convention broken for pyQT integration
    def fitInView(self):
        # this function takes care of accommodating the view so that it can fit
        # in the scene, it resets the zoom to 0 (i think is a overkill, i took
        # it from somewhere else)
        rect = QtCore.QRectF(self._display.pixmap().rect())
        if not rect.isNull():
            unity = self.transform().mapRect(QtCore.QRectF(0, 0, 1, 1))
            self.scale(1 / unity.width(), 1 / unity.height())
            viewrect = self.viewport().rect()
            scenerect = self.transform().mapRect(rect)
            factor = min(viewrect.width() / scenerect.width(), viewrect.height() / scenerect.height())
            self.scale(factor, factor)
            self.centerOn(rect.center())
            self._zoom = 0

    def auto_resize_view(self):
        # TODO: This function should take into account how much horizontal space it has and resize to fill it
        if not self.cap_is_good():
            size_factor = 0
        else:
            size_factor = min(self.max_size[0]/self._resolution[0],
                              self.max_size[1]/self._resolution[1],
                              1)
        self.setFixedHeight(self._resolution[1]*size_factor)
        self.setFixedWidth(self._resolution[0]*size_factor)

    # Capitalization convention broken for pyQT integration
    def wheelEvent(self, event):
        # this take care of the zoom, it modifies the zoom factor if the mouse
        # wheel is moved forward or backward
        if not self._display.pixmap().isNull():
            move = (event.angleDelta().y() / 120)
            if move > 0:
                factor = 1 + self._scale_factor
                self._zoom += 1
            elif move < 0:
                factor = 1 - self._scale_factor
                self._zoom -= 1
            else:
                factor = 1

            if self._zoom > 0 or (self._zoom == 0 and factor > 1):
                self.scale(factor, factor)
            if self._zoom < 0:
                self._zoom = 0
                self.fitInView()

    # These functions deal with user interaction
    def mousePressEvent(self, event):
        scene_pos = self.mapToScene(event.pos())
        mouse_pos = (scene_pos.toPoint().x(), scene_pos.toPoint().y())
        if self._mode == Mode.EDIT:
            self.edit_locations(mouse_pos, event.button())
        self.setDragMode(QtWidgets.QGraphicsView.ScrollHandDrag)
        QtWidgets.QGraphicsView.mousePressEvent(self, event)

    def edit_locations(self, mouse_pos: Tuple[int, int], button: object) -> bool:
        if button == QtCore.Qt.RightButton:
            if self._lifted_point is None:
                self._lifted_point = self.get_closest_point(mouse_pos)
                if self._lifted_point is not None:
                    # self._curr_landmarks.landmarks[self._lifted_point.group].remove(self._lifted_point)
                    self.point_moved_signal.emit(True, self._lifted_point)
                    self.update_curr_frame(excluded_landmarks=[self._lifted_point.index])
                else:
                    return False
        if button == QtCore.Qt.LeftButton:
            if self._lifted_point is not None:
                self._lifted_point.location = mouse_pos
                if mouse_pos[0] < 0 or mouse_pos[0] > self._resolution[0] or mouse_pos[1] < 0 or mouse_pos[1] > self._resolution[1]:
                    return False
                # self._curr_landmarks.landmarks[self._lifted_point.group].append(self._lifted_point)
                self.point_moved_signal.emit(False, self._lifted_point)
                frame_num = self.get_frame_num()
                self._landmarks_editing.at[self._landmarks_editing["Frame_number"] == frame_num, f"landmark_{self._lifted_point.index - 1}_x"] = mouse_pos[0]
                self._landmarks_editing.at[self._landmarks_editing["Frame_number"] == frame_num, f"landmark_{self._lifted_point.index - 1}_y"] = mouse_pos[1]
                self.update_curr_frame()
                self._lifted_point = None
        return True

    def save_edits(self, save_path: str) -> bool:
        # TODO: Make this more robust to users being stupid
        if save_path.split(".")[-1] != "csv":
            return False
        old_path = ".".join(save_path.split(".")[:-1])+"_orig.csv"
        if os.path.exists(save_path) and not os.path.exists(old_path):
            os.rename(save_path, old_path)
        self._landmarks_editing.to_csv(save_path, index=False)

    def get_closest_point(self, mouse_pos, distance_thresh = 6) -> Optional[utils.Landmark]:
        if self._curr_landmarks is None:
            return None
        all_landmarks = []
        for group in self._curr_landmarks.landmarks:
            all_landmarks.extend(self._curr_landmarks.landmarks[group])
        landmark_locs = [landmark.location for landmark in all_landmarks]
        distance = cdist(np.array(landmark_locs), np.array([mouse_pos]))[:, 0]
        min_index = np.argmin(distance)
        min_dist = distance[min_index]
        if min_dist < distance_thresh:
            return all_landmarks[min_index]
        else:
            return None

    def mouseReleaseEvent(self, event):
        self.setDragMode(QtWidgets.QGraphicsView.NoDrag)
        QtWidgets.QGraphicsView.mouseReleaseEvent(self, event)

    # Event Connections
    @QtCore.pyqtSlot(int)
    def on_frame_change(self, frame: int):
        if self._lifted_point is not None:
            self.point_moved_signal.emit(False, self._lifted_point)
            self._lifted_point = None
        self._eyes_lifted = {"left": False, "right": False}


