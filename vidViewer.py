from typing import Dict, List, Tuple, Optional, Union
from PyQt5 import QtWidgets, QtGui, QtCore
import os
import numpy as np
import cv2
import pandas as pd
from scipy.spatial.distance import cdist
import utils

from popups.MetricPopup import MetricWindow
from popups.MetricCreationPopup import MetricCreationWindow

class FrameBuffer(QtCore.QThread):
    _buffer: list
    _buffer_radius: int
    _buffer_length: int

    _reader: cv2.VideoCapture = None
    _target_frame: int = 0
    _curr_frame: int = 0
    _max_frame: int = -100
    _resolution: Tuple[int, int] = (-1, -1)
    _scale_factor: float = 1

    _landmarks: Optional[pd.DataFrame] = None
    _landmark_features: Optional[utils.LandmarkFeatures] = None

    _playback_timer: QtCore.QTimer
    _was_playing = False
    _force_update = False
    frame_changed_signal = QtCore.pyqtSignal(object)

    _running = True

    def __init__(self, reader: cv2.VideoCapture = None,
                 landmarks: Optional[pd.DataFrame] = None,
                 landmark_features: Optional[utils.LandmarkFeatures] = None,
                 buffer_radius=15):
        super(FrameBuffer, self).__init__()
        self._buffer_radius = buffer_radius
        self._buffer_length = 1 + 2 * buffer_radius
        self._buffer = [None for i in range(self._buffer_length)]

        if reader is not None:
            self.set_reader(reader)
        if landmark_features is not None:
            self.set_landmark_features(landmark_features)
        if landmarks is not None:
            self.set_landmarks(landmarks)

        self._playback_timer = QtCore.QTimer()
        self._playback_timer.timeout.connect(
            lambda: self.set_frame(delta=1))

    def set_reader(self, reader: Optional[cv2.VideoCapture], scale_factor: float = 1) -> bool:
        self._reader = reader
        if reader is None:
            return False
        self._max_frame = int(self._reader.get(7)) - 1
        self._resolution = (int(self._reader.get(3)), int(self._reader.get(4)))
        self._scale_factor = scale_factor
        return True

    def set_landmarks(self, landmarks: Optional[pd.DataFrame]) -> bool:
        self._landmarks = landmarks
        if landmarks is None:
            return False
        return True

    def set_landmark_features(self, features: utils.LandmarkFeatures) -> bool:
        self._landmark_features = features
        return True

    def init_buffer(self) -> bool:
        if self._reader is not None and self._landmarks is not None:
            self.set_frame(0)
            self._force_update = True
        return True

    def set_frame(self, index: int=None, delta: int=None) -> bool:
        # If index is used, then it specifies the exact frame. Delta specifies a
        # number of frames to jump
        if index is None:
            index = self._curr_frame
        if delta is not None:
            index += delta
        if index > self._max_frame:
            self.pause()
        self._target_frame = max(0, min(index, self._max_frame))
        return True

    def update_curr_frame(self, remark=True) -> bool:
        # Remarkup the current frame and then emit a frame change
        if remark:
            self._force_update = True
        else:
            self.frame_changed_signal.emit(self._buffer[self._buffer_radius])
        return True

    def get_frame_num(self) -> int:
        return self._curr_frame

    def play(self, frame_rate: float=30,  cond_was_playing: bool=False) -> bool:
        if cond_was_playing and not self._was_playing:
            return False
        self.pause()
        self._playback_timer.start(1000.0/frame_rate)
        return True

    def pause(self) -> bool:
        self._was_playing = self._playback_timer.isActive()
        if self._playback_timer.isActive():
            self._playback_timer.stop()
            return True
        return False

    def is_playing(self) -> bool:
        return self._playback_timer.isActive()

    def get_frame_landmarks(self, frame_num: int) -> Optional[pd.DataFrame]:
        try:
            landmarks = self._landmarks.loc[self._landmarks["Frame_number"] == frame_num]
        except (ValueError, KeyError, AttributeError) as e:
            landmarks = None
        return landmarks

    def get_marked_frame(self, frame: np.ndarray, frame_num: int) -> Tuple[Optional[np.ndarray], Optional[utils.FaceLandmarks]]:
        marked_frame = None
        landmarks = self.get_frame_landmarks(frame_num)
        curr_landmarks = None
        if landmarks is not None and not landmarks.empty:
            curr_landmarks = utils.landmark_frame_to_shapes(landmarks, self._landmark_features)
            marked_frame = utils.markup_image(frame, curr_landmarks, self._landmark_features, resolution=self._resolution, scale_factor=self._scale_factor)
        return marked_frame, curr_landmarks

    def read(self) -> Optional[np.ndarray]:
        ret, frame = self._reader.read()
        if not ret:
            return None
        if self._scale_factor != 1:
            frame = cv2.resize(frame, None, fx=self._scale_factor, fy=self._scale_factor, interpolation = cv2.INTER_CUBIC)
        # frame = cv2.blur(frame, (3, 3))
        frame = cv2.GaussianBlur(frame, (3, 3), 0)
        # frame = cv2.medianBlur(frame, 3)
        return frame

    def read_frames(self, start: int, num_frames: int, insert_pos: utils.Position) -> bool:
        def insert(frame_obj):
            if insert_pos == utils.Position.BEG:
                self._buffer.pop(0)
                self._buffer.append(frame_obj)
            else:
                self._buffer.pop()
                self._buffer.insert(0, frame_obj)
        reader_loc = self._reader.get(1)
        if reader_loc != start:
            self._reader.set(1, max(0, start))
        for i in range(num_frames):
            true_frame = start+i
            if true_frame < 0:
                insert(None)
            elif true_frame > self._max_frame:
                insert(None)
            else:
                # ret, frame = self._reader.read()
                frame = self.read()
                # if ret:
                if frame is not None:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    marked_frame = self.get_marked_frame(frame, true_frame)
                    insert((true_frame, frame, *marked_frame))
                else:
                    insert(None)
        return True

    def update(self) -> bool:
        if self._force_update:
            updated_current = False
            if self._buffer[self._buffer_radius] is not None:
                index, frame, marked_frame, landmarks = self._buffer[self._buffer_radius]
                marked_up, new_landmarks = self.get_marked_frame(frame, index)
                self.frame_changed_signal.emit((index, frame, marked_up, new_landmarks))
                updated_current = True
            self.read_frames(self._curr_frame-self._buffer_radius, self._buffer_length, utils.Position.BEG)
            if not updated_current:
                self.frame_changed_signal.emit(self._buffer[self._buffer_radius])
            self._force_update = False

        if self._target_frame == self._curr_frame:
            return False

        delta = self._target_frame - self._curr_frame

        frame_emit = False
        if abs(delta) <= self._buffer_radius:
            self.frame_changed_signal.emit(self._buffer[delta+self._buffer_radius])
            frame_emit = True

        start = 0
        read_num = 0
        reader_pos = utils.Position.BEG
        if abs(delta) >= self._buffer_length:
            start = self._target_frame-self._buffer_radius
            read_num = self._buffer_length
            reader_pos = utils.Position.BEG
        elif delta > 0:
            start = self._curr_frame+self._buffer_radius+1
            read_num = delta
            reader_pos = utils.Position.BEG
        elif delta < 0:
            start = self._target_frame-self._buffer_radius
            read_num = abs(delta)
            reader_pos = utils.Position.END

        self.read_frames(start, read_num, reader_pos)

        if not frame_emit:
            self.frame_changed_signal.emit(self._buffer[self._buffer_radius])
            pass
        self._curr_frame = self._target_frame

    def stop(self) -> bool:
        self._running = False
        return True

    def run(self):
        while self._running:
            self.update()

class ImageViewer(QtWidgets.QGraphicsView):
    _zoom_factor: float = 0.1  # Defines the zoom speed
    _zoom: int = 0
    _scene: Optional[QtWidgets.QGraphicsScene] = None
    _display: Optional[QtWidgets.QGraphicsPixmapItem] = None

    _landmarks: Optional[pd.DataFrame] = None  # Holds all landmark points
    _landmarks_editing: Optional[pd.DataFrame] = None # A copy of landmarks for editing
    _landmark_features: Optional[utils.LandmarkFeatures] = utils.LandmarkFeatures() # Holds information about how to display landmarks
    _current_landmarks: utils.FaceLandmarks

    _reader: Optional[cv2.VideoCapture] = None
    _video_length: int = -1
    _video_fps: int = -1
    _playback_speed: float = 1
    _resolution: Tuple[int, int] = (-1, -1)  # Stored as (width, height)
    _min_dim = 1280
    _scale_factor = 1
    _resize_on_get_frame: bool = False

    _frame_buffer: FrameBuffer = None

    _lifted_point: Optional[utils.Landmark] = None
    _eyes_lifted: Dict[str, bool] = {"left": False, "right": False}
    _mouse_positions: List[Tuple[int, int]] = [(-1, -1), (-1, -1), (-1, -1), (-1, -1)]
    _selected_landmarks: List[Union[utils.Landmark, utils.LandmarkGroup]] = []
    _held_keys: List[object] = []

    _metrics: List[utils.Metric] = []

    _mode: utils.InteractionMode = utils.InteractionMode.POINT

    frame_change_signal = QtCore.pyqtSignal(int) # Emitted when frame number changes
    playback_change_signal = QtCore.pyqtSignal(bool) # Emitted when video is paused or played
    metadata_update_signal = QtCore.pyqtSignal(int, float, object) # Emits length, frame rate, resolution
    point_moved_signal = QtCore.pyqtSignal(bool, utils.Landmark) # Emitted when a landmark is moved

    _metric_creation_window: MetricCreationWindow = None
    _metric_popup: MetricWindow = None

    def __init__(self, reader: cv2.VideoCapture=None, landmarks: pd.DataFrame=None):
        super(ImageViewer, self).__init__()
        self.setup_view_window()
        self.setup_default_groups()
        if reader is not None:
            self.set_reader(reader)
        if landmarks is not None:
            self.set_landmarks(landmarks)
        self.fitInView()

    def reader_is_good(self) -> bool:
        return self._reader is not None and self._reader.isOpened()

    def reset_buffer(self) -> bool:
        if self._frame_buffer is not None:
            self._frame_buffer.stop()
            self._frame_buffer = None
            self._landmarks = None
            self._landmarks_editing = None
            self._reader = None
            self._metrics = []
        return True

    def set_reader(self, reader: cv2.VideoCapture) -> bool:
        self._reader = reader
        if self.reader_is_good():
            self._video_length = int(self._reader.get(7))
            self._video_fps = int(self._reader.get(5))
            self._resolution = (int(self._reader.get(3)), int(self._reader.get(4)))
            self._scale_factor = max(self._min_dim / max(self._resolution), 1)
            self.metadata_update_signal.emit(
                self._video_length,
                self._video_fps,
                self._resolution
            )
            self.update_frame_buffer()
            self.fitInView()
            return True
        return False

    def set_landmarks(self, landmarks: pd.DataFrame) -> bool:
        self._landmarks = landmarks
        self._landmarks_editing = landmarks.copy(deep=True)
        self.update_frame_buffer()
        return True

    def update_frame_buffer(self) -> bool:
        if self._frame_buffer is None:
            self._frame_buffer = FrameBuffer()
            self._frame_buffer.frame_changed_signal.connect(self.on_new_frame)
        if self._landmarks_editing is not None:
            self._frame_buffer.set_landmarks(self._landmarks_editing)
        if self._landmark_features is not None:
            self._frame_buffer.set_landmark_features(self._landmark_features)
        if self._reader is not None:
            self._frame_buffer.set_reader(self._reader, self._scale_factor)
        try:
            self._resize_on_get_frame = True
            self._frame_buffer.init_buffer()
        except AttributeError as e:
            pass
        self._frame_buffer.start()
        return True

    def setup_view_window(self) -> bool:
        self._scene = QtWidgets.QGraphicsScene(self)
        self._display = QtWidgets.QGraphicsPixmapItem()
        self._scene.addItem(self._display)
        self.setScene(self._scene)
        self.setTransformationAnchor(QtWidgets.QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QtWidgets.QGraphicsView.AnchorUnderMouse)
        self.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.setBackgroundBrush(QtGui.QBrush(QtGui.QColor(100, 100, 100)))
        self.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.setDragMode(QtWidgets.QGraphicsView.RubberBandDrag)
        self.setMouseTracking(True)
        return True

    def add_to_group(self, name: str, indices: List[int]) -> bool:
        groups = self._landmark_features.groups
        indices.sort()
        indices = set(indices)
        if name not in groups:
            groups[name] = []
        for index in indices:
            for group in groups:
                try:
                    groups[group].remove(index)
                except ValueError as e:
                    pass
            groups[name].append(index)
        return True

    def setup_default_groups(self) -> bool:
        self._landmark_features = utils.LandmarkFeatures()
        groups = self._landmark_features.groups
        groups["face"] = list(range(1, 69))
        # self.add_to_group("right_eye", list(range(37, 42 + 1)))
        # self.add_to_group("left_eye", list(range(43, 48 + 1)))
        # self.add_to_group("nose", list(range(28, 36 + 1)))
        # self.add_to_group("inner_mouth", list(range(61, 68 + 1)))
        # self.add_to_group("outer_mouth", list(range(49, 60 + 1)))
        # self.add_to_group("right_eyebrow", list(range(18, 22 + 1)))
        # self.add_to_group("left_eyebrow", list(range(23, 27 + 1)))
        # self.add_to_group("chin_outline", list(range(1, 17 + 1)))
        self.add_to_group("lower_eye", [41, 42, 48, 47])
        self.add_to_group("upper_mouth", [62, 63, 64])
        self.add_to_group("lower_mouth", [66, 67, 68])
        return True

    def toggle_landmarks(self) -> bool:
        self._landmark_features.show.landmarks = not self._landmark_features.show.landmarks
        if not self._frame_buffer.is_playing():
            self._frame_buffer.update_curr_frame(remark=True)
        return self._landmark_features.show.landmarks

    def toggle_bounding_box(self) -> bool:
        self._landmark_features.show.bounding_box = not self._landmark_features.show.bounding_box
        if not self._frame_buffer.is_playing():
            self._frame_buffer.update_curr_frame(remark=True)
        return self._landmark_features.show.bounding_box

    def toggle_metrics(self) -> bool:
        self._landmark_features.show.metrics = not self._landmark_features.show.metrics
        if not self._frame_buffer.is_playing():
            self._frame_buffer.update_curr_frame(remark=True)
        return self._landmark_features.show.metrics

    def get_groups(self) -> Dict[str, List[int]]:
        return self._landmark_features.groups.copy()

    def set_display(self, image=None) -> bool:
        height, width, channels = image.shape
        bytesPerLine = 3 * width
        qt_img = QtGui.QImage(image.data, width, height, bytesPerLine, QtGui.QImage.Format_RGB888)
        pixmap = QtGui.QPixmap.fromImage(qt_img)

        # Updates the view with the new pixmap
        if pixmap and not pixmap.isNull():
            self.setDragMode(QtWidgets.QGraphicsView.RubberBandDrag)
            self._display.setPixmap(pixmap)
            return True
        else:
            self.setDragMode(QtWidgets.QGraphicsView.NoDrag)
            self._display.setPixmap(QtGui.QPixmap())
            return False

    def on_new_frame(self, frame_info: Optional[Tuple[int, np.ndarray, Optional[np.ndarray]]]) -> bool:
        if frame_info is None:
            return False
        frame_num, frame, marked_frame, self._current_landmarks = frame_info
        # frame = marked_frame if marked_frame is not None and self._landmark_features.show.landmarks else frame
        frame = marked_frame if marked_frame is not None else frame
        self.frame_change_signal.emit(frame_num)
        self.set_display(frame)
        if self._resize_on_get_frame:
            self.fitInView()
            self._resize_on_get_frame = False
        return True

    def play(self) -> bool:
        return self._frame_buffer.play(frame_rate=self._video_fps*self._playback_speed)

    def toggle_play(self) -> bool:
        if self._frame_buffer.is_playing():
            self._frame_buffer.pause()
            self.playback_change_signal.emit(True)
            return True
        else:
            self._frame_buffer.play()
            self.playback_change_signal.emit(False)
            return True

    def pause(self) -> bool:
        return self._frame_buffer.pause()

    def is_playing(self) -> bool:
        if self._frame_buffer is not None:
            return self._frame_buffer.is_playing()
        else:
            return False

    def set_playback_speed(self, speed: float = 1) -> bool:
        self._playback_speed = speed
        return self._frame_buffer.play(frame_rate = self._video_fps*self._playback_speed, cond_was_playing=True)

    def jump_frames(self, delta=1) -> bool:
        return self._frame_buffer.set_frame(delta=delta)

    def seek_frame(self, frame=0) -> bool:
        return self._frame_buffer.set_frame(index=frame)

    def get_curr_frame(self) -> int:
        return self._frame_buffer.get_frame_num()

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

    def wheelEvent(self, event):
        # this take care of the zoom, it modifies the zoom factor if the mouse
        # wheel is moved forward or backward
        if not self._display.pixmap().isNull():
            move = (event.angleDelta().y() / 120)
            if move > 0:
                factor = 1 + self._zoom_factor
                self._zoom += 1
            elif move < 0:
                factor = 1 - self._zoom_factor
                self._zoom -= 1
            else:
                factor = 1

            if self._zoom > 0 or (self._zoom == 0 and factor > 1):
                self.scale(factor, factor)
            if self._zoom < 0:
                self._zoom = 0
                self.fitInView()

    def keyPressEvent(self, event: QtCore.QEvent):
        key = event.key()
        if key == QtCore.Qt.Key_Space:
            self.toggle_play()
        if key == QtCore.Qt.Key_Left:
            self.jump_frames(-1)
        if key == QtCore.Qt.Key_Right:
            self.jump_frames(1)
        if key == QtCore.Qt.Key_Shift:
            self._mode = utils.InteractionMode.AREA
        if key == QtCore.Qt.Key_1:
            self.create_metric(utils.MetricType.LENGTH)
        if key == QtCore.Qt.Key_2:
            self.create_metric(utils.MetricType.AREA)
        if key == QtCore.Qt.Key_3:
            self.analyze_metrics()
        if key == QtCore.Qt.Key_0:
            self.remove_metric()

        if key == QtCore.Qt.Key_B:
            self.toggle_bounding_box()
        if key == QtCore.Qt.Key_M:
            self.toggle_metrics()
        if key == QtCore.Qt.Key_L:
            self.toggle_landmarks()

        if key not in self._held_keys:
            self._held_keys.append(key)

    def keyReleaseEvent(self, event):
        key = event.key()
        if key == QtCore.Qt.Key_Shift:
            self.setDragMode(QtWidgets.QGraphicsView.NoDrag)
            self._mode = utils.InteractionMode.POINT
        if key in self._held_keys:
            self._held_keys.remove(key)

    def mousePressEvent(self, event):
        scene_pos = self.mapToScene(event.pos())
        # TODO: Don't round values
        mouse_pos = (scene_pos.toPoint().x()/self._scale_factor, scene_pos.toPoint().y()/self._scale_factor)
        button = event.button()
        self._mouse_positions[0] = mouse_pos
        self._mouse_positions[2] = event.pos()
        if self._mode == utils.InteractionMode.POINT:
            self.setDragMode(QtWidgets.QGraphicsView.ScrollHandDrag)
        if self._mode == utils.InteractionMode.AREA:
            self.setDragMode(QtWidgets.QGraphicsView.RubberBandDrag)
        QtWidgets.QGraphicsView.mousePressEvent(self, event)

    def mouseReleaseEvent(self, event):
        self.setDragMode(QtWidgets.QGraphicsView.NoDrag)
        scene_pos = self.mapToScene(event.pos())
        mouse_pos = (scene_pos.toPoint().x() / self._scale_factor,
                     scene_pos.toPoint().y() / self._scale_factor)
        button = event.button()
        self._mouse_positions[1] = mouse_pos
        self._mouse_positions[3] = event.pos()
        if self._mode == utils.InteractionMode.POINT:
            if self._mouse_positions[2] == self._mouse_positions[3]:
                if button == QtCore.Qt.RightButton:
                    self.edit_locations(mouse_pos)
                else:
                    self.select_landmarks(mouse_pos)
        if self._mode == utils.InteractionMode.AREA:
            self.select_landmarks()
        QtWidgets.QGraphicsView.mouseReleaseEvent(self, event)

    def get_closest_point(self, mouse_pos, distance_thresh = 6) -> Optional[utils.Landmark]:
        if self._current_landmarks is None:
            return None
        all_landmarks = []
        for group in self._current_landmarks.landmarks:
            all_landmarks.extend(self._current_landmarks.landmarks[group])
        landmark_locs = [landmark.location for landmark in all_landmarks]
        distance = cdist(np.array(landmark_locs), np.array([mouse_pos]))[:, 0]
        min_index = np.argmin(distance)
        min_dist = distance[min_index]
        if min_dist < distance_thresh:
            return all_landmarks[min_index]
        else:
            return None

    def edit_locations(self, mouse_pos: Tuple[int, int]) -> bool:
        if self._lifted_point is None:
            self._lifted_point = self.get_closest_point(mouse_pos)
            if self._lifted_point is not None:
                self._landmark_features.excluded.append(self._lifted_point.index)
                self._frame_buffer.update_curr_frame()
                self.point_moved_signal.emit(True, self._lifted_point)
            else:
                return False
        else:
            self._lifted_point.location = mouse_pos
            if mouse_pos[0] < 0 or mouse_pos[0] > self._resolution[0] or mouse_pos[1] < 0 or mouse_pos[1] > self._resolution[1]:
                return False
            self.point_moved_signal.emit(False, self._lifted_point)
            self._landmark_features.excluded.remove(self._lifted_point.index)
            frame_num = self._frame_buffer.get_frame_num()
            self._landmarks_editing.at[self._landmarks_editing["Frame_number"] == frame_num, f"landmark_{self._lifted_point.index - 1}_x"] = mouse_pos[0]
            self._landmarks_editing.at[self._landmarks_editing["Frame_number"] == frame_num, f"landmark_{self._lifted_point.index - 1}_y"] = mouse_pos[1]
            self._frame_buffer.update_curr_frame()
            self._lifted_point = None
        return True

    def select_landmarks(self, mouse_pos: Optional[Tuple[float, float]]=None) -> int:
        if self._mode == utils.InteractionMode.POINT:
            selected_landmark = self.get_closest_point(mouse_pos)
            if selected_landmark is not None:
                self._selected_landmarks.append(selected_landmark)
            else:
                self._selected_landmarks = []
        elif self._mode == utils.InteractionMode.AREA:
            if self._current_landmarks is None:
                return 0
            all_landmarks = []
            for group in self._current_landmarks.landmarks:
                all_landmarks.extend(self._current_landmarks.landmarks[group])
            selected_landmarks = []
            x_max = max(self._mouse_positions[0][0], self._mouse_positions[1][0])
            x_min = min(self._mouse_positions[0][0], self._mouse_positions[1][0])
            y_max = max(self._mouse_positions[0][1], self._mouse_positions[1][1])
            y_min = min(self._mouse_positions[0][1], self._mouse_positions[1][1])
            for landmark in all_landmarks:
                in_x = x_max >= landmark.location[0] >= x_min
                in_y = y_max >= landmark.location[1] >= y_min
                if in_x and in_y:
                    selected_landmarks.append(landmark)
            if len(selected_landmarks) < 1:
                self._selected_landmarks = []
            else:
                self._selected_landmarks.append(utils.LandmarkGroup(selected_landmarks))
        self._landmark_features.selected = []
        for landmark_def in self._selected_landmarks:
            if isinstance(landmark_def, utils.Landmark):
                self._landmark_features.selected.append(landmark_def.index)
            else:
                self._landmark_features.selected.extend([landmark.index for landmark in landmark_def.landmarks])
        self._frame_buffer.update_curr_frame(remark=True)
        return len(self._selected_landmarks)

    def draw_metrics(self) -> bool:
        # TODO: Make this able to draw centroids of landmark groups
        self._landmark_features.lines = []
        for metric in self._metrics:
            type = metric.type
            landmarks = []
            for landmark_def in metric.landmarks:
                if isinstance(landmark_def, utils.Landmark):
                    landmarks.append(landmark_def.index)
                else:
                    landmarks.extend([landmark.index for landmark in landmark_def.landmarks])
            if type == utils.MetricType.AREA:
                landmarks += [landmarks[0]]
            self._landmark_features.lines.append(landmarks)
        self._frame_buffer.update_curr_frame()
        return True

    def create_metric(self, type: utils.MetricType, name: str=None) -> bool:
        if len(self._selected_landmarks) < 2:
            return False
        if name is None:
            name = f"{type.name}:{','.join([str(landmark.index) for landmark in self._selected_landmarks if isinstance(landmark, utils.Landmark)])}"
        metric = utils.Metric(name, type, self._selected_landmarks[:])
        self._metrics.append(metric)
        if self._metric_creation_window is not None:
            self._metric_creation_window.close()
        self._metric_creation_window = MetricCreationWindow(self, metric, self._metrics)
        self._metric_creation_window.show()
        self.draw_metrics()
        return True

    def remove_metric(self) -> bool:
        # Removes the metric represented by the selected landmarks
        # TODO: Make this work with centroids
        for i, metric in enumerate(self._metrics):
            landmarks = [landmark.index for landmark in metric.landmarks if isinstance(landmark, utils.Landmark)]
            all_included = True
            for landmark in self._selected_landmarks:
                all_included = all_included and landmark.index in landmarks
            if all_included:
                self._metrics.remove(metric)
        self.draw_metrics()
        return True

    def remove_metric_by_index(self, index: int) -> bool:
        try:
            del self._metrics[index]
            self.draw_metrics()
            return True
        except IndexError:
            return False

    def analyze_metrics(self) -> bool:
        # Open a window that lists the metrics
        self._metric_popup = MetricWindow(self, self._metrics, self._landmarks_editing)
        self._metric_popup.show()
        self._metric_popup.metric_done_signal.connect(self.on_metrics_done)
        pass

    def on_metrics_done(self, metrics: pd.DataFrame):
        # metrics.to_csv("./movement_analysis/metrics.csv")
        return

    def save_edits(self, save_path: str) -> bool:
        # TODO: Make this more robust to users inputting unexpected values
        if save_path.split(".")[-1] != "csv":
            return False
        old_path = ".".join(save_path.split(".")[:-1])+"_orig.csv"
        if os.path.exists(save_path) and not os.path.exists(old_path):
            os.rename(save_path, old_path)
        self._landmarks_editing.to_csv(save_path, index=False)

    # Event Connections
    @QtCore.pyqtSlot(int)
    def on_frame_change(self, frame: int):
        if self._lifted_point is not None:
            self.point_moved_signal.emit(False, self._lifted_point)
            self._lifted_point = None
        self._eyes_lifted = {"left": False, "right": False}

