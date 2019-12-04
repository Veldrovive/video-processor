from typing import Dict, List, Tuple, Optional, Union
from PyQt5 import QtWidgets, QtGui, QtCore
import os
import numpy as np
import cv2
import pandas as pd

import config
import utils

from popups.MetricPopup import MetricWindow
from popups.MetricCreationPopup import MetricCreationWindow

class FrameBuffer(QtCore.QThread):
    _reader: Optional[cv2.VideoCapture] = None
    _marker: Optional[utils.ImageMarker] = None

    _buffer: list
    _buffer_radius: int
    _buffer_length: int

    _scale_factor: float = 1
    _target_frame: int = 0
    _curr_frame: int = 0
    _max_frame: int = -100

    new_frame_signal = QtCore.pyqtSignal(object)

    _force_update = False
    _was_playing = False
    _running = True

    def __init__(self, reader: Optional[cv2.VideoCapture] = None,
                 marker: Optional[utils.ImageMarker] = None,
                 scale_factor: float = 1,
                 buffer_radius=15):
        super(FrameBuffer, self).__init__()
        self.set_reader(reader)
        self._marker = marker

        self._buffer_radius = buffer_radius
        self._buffer_length = 1 + 2 * buffer_radius
        self._buffer = [None for i in range(self._buffer_length)]

        self._scale_factor = scale_factor

        self._playback_timer = QtCore.QTimer()
        self._playback_timer.timeout.connect(
            lambda: self.set_frame(delta=1))

    def set_reader(self, reader: Optional[cv2.VideoCapture], scale_factor: float=1) -> bool:
        if reader is None:
            return False
        self._reader = reader
        self._max_frame = int(self._reader.get(7)) - 1
        self._scale_factor = scale_factor
        return True

    def set_img_marker(self, img_marker: utils.ImageMarker) -> bool:
        if img_marker is None:
            return False
        self._marker = img_marker
        return True

    def init_buffer(self) -> bool:
        if self._reader is not None and self._marker is not None:
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
            self.new_frame_signal.emit(self._buffer[self._buffer_radius])
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

    def get_marked_frame(self, frame: np.ndarray, frame_num: int) -> Optional[np.ndarray]:
        if frame_num > self._max_frame:
            return None
        else:
            return self._marker.markup_image(frame, frame_num)

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
                    insert((true_frame, frame, marked_frame))
                else:
                    insert(None)
        return True

    def update(self) -> bool:
        if self._force_update:
            updated_current = False
            if self._buffer[self._buffer_radius] is not None:
                index, frame, marked_frame = self._buffer[self._buffer_radius]
                marked_up = self.get_marked_frame(frame, index)
                self.new_frame_signal.emit((index, frame, marked_up))
                updated_current = True
            self.read_frames(self._curr_frame-self._buffer_radius, self._buffer_length, utils.Position.BEG)
            if not updated_current:
                self.new_frame_signal.emit(self._buffer[self._buffer_radius])
            self._force_update = False

        if self._target_frame == self._curr_frame:
            return False

        delta = self._target_frame - self._curr_frame

        frame_emit = False
        if abs(delta) <= self._buffer_radius:
            self.new_frame_signal.emit(self._buffer[delta+self._buffer_radius])
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
            self.new_frame_signal.emit(self._buffer[self._buffer_radius])
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

    _reader: Optional[cv2.VideoCapture] = None
    _min_dim: int = 1920
    _video_length: int = -1
    _playback_speed: float = 1
    _video_fps: float = -1
    _resolution: Tuple[int, int] = (-1, -1)  # Width, Height
    _scale_factor: float = 1
    _frame_buffer: Optional[FrameBuffer] = None
    _img_marker: Optional[utils.ImageMarker] = None

    _resize_on_get_frame: bool = True

    _landmarks: Optional[utils.Landmarks] = None
    _landmarks_frame:  Optional[pd.DataFrame] = None

    _metrics: List[utils.Metric] = []
    _metric_selector: List[int] = []

    color_config: config.Config = config.Config()

    frame_change_signal = QtCore.pyqtSignal(int)  # Emitted when frame number changes
    playback_change_signal = QtCore.pyqtSignal(bool)  # Emitted when video is paused or played
    metadata_update_signal = QtCore.pyqtSignal(int, float, object)  # Emits length, frame rate, resolution
    point_moved_signal = QtCore.pyqtSignal(bool, int)  # Emitted when a landmark is moved

    _held_keys: List = []
    _mouse_positions: List[Tuple[int, int]] = [(-1, -1) for i in range(4)]
    _mode: utils.InteractionMode = utils.InteractionMode.POINT

    _lifted_point: Optional[int] = None

    _metric_creation_window: MetricCreationWindow = None
    _metric_popup: MetricWindow = None

    def __init__(self, reader: cv2.VideoCapture=None, landmarks: pd.DataFrame=None):
        super(ImageViewer, self).__init__()
        self.setup_view_window()
        if reader is not None:
            self.set_reader(reader)
        if landmarks is not None:
            self.set_landmarks(landmarks)
        self.fitInView()

    def reset(self) -> bool:
        if self._frame_buffer is not None:
            self._frame_buffer.stop()
        self._frame_buffer = None
        self._landmarks = None
        self._landmarks_frame = None
        self._reader = None
        self._metrics = []
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

    def set_reader(self, reader: cv2.VideoCapture) -> bool:
        if self.reader_is_good(reader):
            self._reader = reader
            self._video_length = int(self._reader.get(7))
            self._video_fps = int(self._reader.get(5))
            self._resolution = (int(self._reader.get(3)), int(self._reader.get(4)))
            self._scale_factor = max(self._min_dim / max(self._resolution), 1)
            if self._img_marker is not None:
                self._img_marker.set_scale_factor(self._scale_factor)
            self.metadata_update_signal.emit(
                self._video_length,
                self._video_fps,
                self._resolution
            )
            self.update_frame_buffer()
            self._resize_on_get_frame = True
            return True
        return False

    def reader_is_good(self, reader: Optional[cv2.VideoCapture]=None) -> bool:
        if reader is None:
            reader = self._reader
        return reader is not None and reader.isOpened()

    def set_landmarks(self, landmarks: pd.DataFrame) -> bool:
        self._landmarks = utils.Landmarks(landmarks, n_landmarks=68)
        self.setup_default_groups()
        self.setup_default_metrics()
        if self._img_marker is not None:
            self._img_marker.set_landmarks(self._landmarks)

        self.update_frame_buffer()
        return True

    def update_frame_buffer(self) -> bool:
        if self._landmarks is None or not self.reader_is_good():
            return False
        if self._frame_buffer is None:
            self._img_marker = utils.ImageMarker(self._landmarks, self._scale_factor, self.color_config)
            self._img_marker.set_metrics(self._metrics)
            self._frame_buffer = FrameBuffer(self._reader, self._img_marker, buffer_radius=15, scale_factor=self._scale_factor)
            self._frame_buffer.new_frame_signal.connect(self.on_new_frame)
        else:
            self._frame_buffer.set_landmarks(self._landmarks)
            self._frame_buffer.set_reader(self._reader, self._scale_factor)
        try:
            self._resize_on_get_frame = True
            self._frame_buffer.init_buffer()
            self._frame_buffer.start()
            return True
        except AttributeError:
            return False

    def set_display(self, image: np.ndarray) -> bool:
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

    @QtCore.pyqtSlot(object, name="DisplayNewFrame")
    def on_new_frame(self, frame_info: Optional[Tuple[int, np.ndarray, Optional[np.ndarray]]]) -> bool:
        if frame_info is None:
            return False
        frame_num, frame, marked_frame = frame_info
        frame = marked_frame if marked_frame is not None else frame
        self.frame_change_signal.emit(frame_num)
        self.set_display(frame)
        if self._resize_on_get_frame:
            self.fitInView()
            self._resize_on_get_frame = False
        return True

    def add_to_group(self, name: str, indices: List[int]) -> bool:
        group_colors = self.color_config.group.colors
        if name not in group_colors:
            group_colors[name] = (255, 255, 255)
        self._landmarks.set_group(indices, name)
        return True

    def setup_default_groups(self) -> bool:
        # TODO: Define the default groups
        self.add_to_group("lower_eye", [40, 41, 46, 47])
        # self.add_to_group("upper_mouth", [61, 62, 63, 64, 65])
        return True

    def setup_default_metrics(self) -> bool:
        self._metrics.append(utils.Metric(
            "Inter-Eye Distance",
            utils.MetricType.LENGTH,
            [39, 42]
        ))
        self._metrics.append(utils.Metric(
            "Left Mouth Area",
            utils.MetricType.AREA,
            list(range(51, 57+1))
        ))
        self._metrics.append(utils.Metric(
            "Right Mouth Area",
            utils.MetricType.AREA,
            [57, 58, 59, 48, 49, 50, 51]
        ))

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

    def edit_locations(self, mouse_pos: Tuple[int, int]) -> bool:
        frame_num = self._frame_buffer.get_frame_num()
        if self._lifted_point is None:
            self._lifted_point = self._landmarks.get_nearest_point(frame_num, mouse_pos)
            if self._lifted_point is not None:
                self._img_marker.exclude([self._lifted_point])
                self._frame_buffer.update_curr_frame()
                self.point_moved_signal.emit(True, self._lifted_point)
            else:
                return False
        else:
            if mouse_pos[0] < 0 or mouse_pos[0] > self._resolution[0] or mouse_pos[1] < 0 or mouse_pos[1] > self._resolution[1]:
                return False
            self._landmarks.set_location(frame_num, self._lifted_point, mouse_pos)
            self.point_moved_signal.emit(False, self._lifted_point)
            self._img_marker.include([self._lifted_point])
            self._landmarks.set_location(frame_num, self._lifted_point, mouse_pos)
            self._frame_buffer.update_curr_frame()
            self._lifted_point = None
        return True

    def select_landmarks(self, mouse_pos: Optional[Tuple[float, float]]=None) -> bool:
        frame_num = self._frame_buffer.get_frame_num()
        if self._mode == utils.InteractionMode.POINT:
            selected_landmark = self._landmarks.get_nearest_point(frame_num, mouse_pos)
            if selected_landmark is None:
                self._img_marker.deselect()
                self._metric_selector = []
            else:
                self._img_marker.select([selected_landmark])
                self._metric_selector.append(selected_landmark)
        elif self._mode == utils.InteractionMode.AREA:
            x_max = max(self._mouse_positions[0][0], self._mouse_positions[1][0])
            x_min = min(self._mouse_positions[0][0], self._mouse_positions[1][0])
            y_max = max(self._mouse_positions[0][1], self._mouse_positions[1][1])
            y_min = min(self._mouse_positions[0][1], self._mouse_positions[1][1])
            selected_landmarks = self._landmarks.get_point_area(frame_num, x_max, x_min, y_max, y_min)
            if len(selected_landmarks) < 1:
                self._img_marker.deselect()
                self._metric_selector = []
            else:
                self._img_marker.select(selected_landmarks)
                self._metric_selector.append(selected_landmarks)
        self._frame_buffer.update_curr_frame(remark=True)
        self.check_metric_completion()
        return True

    def check_metric_completion(self) -> bool:
        if len(self._metric_selector) > 1 and self._metric_selector[-1] == self._metric_selector[0]:
            self._metrics.append(utils.Metric(
                name=f"PLACEHOLDER:{','.join([str(landmark) for landmark in self._metric_selector[:-1]])}",
                type=utils.MetricType.LENGTH,
                landmarks=self._metric_selector[:-1]
            ))
            return True
        return False

    def analyze_metrics(self) -> bool:
        # Open a window that lists the metrics
        self._metric_popup = MetricWindow(self, self._metrics, self._landmarks)
        self._metric_popup.show()
        return True

    def save_edits(self, save_path: str) -> bool:
        # TODO: Make this more robust to users inputting unexpected values
        if save_path.split(".")[-1] != "csv":
            return False
        old_path = ".".join(save_path.split(".")[:-1])+"_orig.csv"
        if os.path.exists(save_path) and not os.path.exists(old_path):
            os.rename(save_path, old_path)
        frame = self._landmarks.get_dataframe()
        frame.to_csv(save_path, index=False)


