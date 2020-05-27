from typing import List, Tuple, Optional
from PyQt5 import QtWidgets, QtGui, QtCore
import numpy as np
import cv2
import pandas as pd
import time

from queue import Queue

from utils import ImageMarker, DataHolders
from utils.Globals import Globals


class FrameMarker(QtCore.QRunnable):
    def __init__(self, target=None, args=()):
        super(FrameMarker, self).__init__()
        self.target = target
        self.args = args

    def run(self):
        self.target(*self.args)

class FrameBufferV2(QtCore.QThread):
    """
    This whole thing is pretty ugly so I'll give a rundown of why things are
    done this way. Basically, we need to be dynamically displaying content on
    top of a visual being shown in real time. On modern computer, this works
    fine without any threading. On an older computer, however, it runs at a much
    lower frame rate. At first, I tried to just run the marker in one separate
    thread, but I found that that still lagged on some systems. Therefore, I
    created a thread pool to handle it instead. It seems well optimized so it
    works well most of the time. This does require some more complicated logic
    however and means some frames may load later than expected.
    Also, we need to buffer frames in the future and past which adds some
    amount of complexity.
    """
    _glo: Globals

    _thead_pool: QtCore.QThreadPool

    _reader: Optional[cv2.VideoCapture] = None
    _marker: Optional[ImageMarker.ImageMarkerV2] = None

    _buffer: list = None
    _new_frames: Queue = Queue()
    _buffer_radius: int
    _buffer_length: int

    _target_frame: int = 0
    _curr_frame: int = 0

    new_frame_signal = QtCore.pyqtSignal(object)
    refresh_image_signal = QtCore.pyqtSignal()

    _force_update = False
    _was_playing = False
    _running = True

    def __init__(self, buffer_radius=15):
        """
        :param buffer_radius:
        """
        super(FrameBufferV2, self).__init__()
        self._glo = Globals.get()
        self._marker = ImageMarker.ImageMarkerV2()
        self._glo.onFileChange.connect(self.handle_file_change)
        self._thead_pool = QtCore.QThreadPool()

        self._buffer_radius = buffer_radius
        self._buffer_length = 1 + 2 * buffer_radius

        self._playback_timer = QtCore.QTimer()
        self._playback_timer.timeout.connect(lambda: self.set_frame(delta=1))

        self._glo.onConfigChange.connect(self.cond_reset_buffer)

    def handle_file_change(self):
        """
        Takes actions to change the currently playing video
        Actions to take (So I can remember):
        1. Stop threads and empty the new frame buffer
        2. Update the reader with the new video
        3. Initialize the buffer
        """
        self.pause()
        self._new_frames = Queue()
        self._reader = self._glo.get_video()
        self.init_buffer(self._glo.video_config.get_position())
        self.refresh_image_signal.emit()

    def set_frame(self, index: int = None, delta: int = None) -> bool:
        """
        Sets the buffer position to the specified frame
        :param index: If this is set, the buffer moves to this position
        :param delta: If this is set, the buffer shifts by this number
        """
        if index is None:
            index = self._curr_frame
        if delta is not None:
            index += delta
        if index > self._glo.visual_config.video_length:
            self.pause()
        self._target_frame = max(0, min(index, self._glo.visual_config.video_length))
        return True

    def init_buffer(self, start_frame: int = 0) -> bool:
        """
        Forces the buffer to reset and fills it with the first frames
        """
        self._buffer = [None for i in range(self._buffer_length)]
        if self._reader is not None and self._marker is not None:
            self.set_frame(start_frame)
            self._force_update = True
        return True

    def cond_reset_buffer(self, change_type: str):
        """
        Resets the buffer only if the visual config has changed
        If we don't do this, the buffer has to refresh every time the user seeks
        to a new frame which slows down the program significantly
        """
        if change_type == "visual_config":
            self.reset_buffer()

    def reset_buffer(self):
        # self._buffer = [None for i in range(self._buffer_length)]
        if self._buffer is not None:
            self._force_update = True

    def update_curr_frame(self, remark=True) -> Optional[np.ndarray]:
        """
        Emits the current frame again
        :param remark: If True, then the frame will be run through the frame
            marker again. If False, the buffered frame will be re-emited
        :return: None if remarking, the cv2 frame if not remarking
        """
        if remark:
            self._force_update = True
            return None
        else:
            self.new_frame_signal.emit(self._buffer[self._buffer_radius])
            return self._buffer[self._buffer_radius]

    def get_frame_num(self) -> int:
        """
        :return: The current frame number
        """
        return self._curr_frame

    def pause(self) -> bool:
        """
        Stops the playback timer
        :return: Whether the timer was stopped
        """
        self._was_playing = self._playback_timer.isActive()
        if self._playback_timer.isActive():
            self._playback_timer.stop()
            return True
        return False

    def play(self, frame_rate: float = 30, cond_was_playing: bool = False) -> bool:
        """
        Starts the frame timer
        :param frame_rate: Sets the interval between frame updates
        :param cond_was_playing: If this is True, the timer will only start if
            the video was paused in the past
        :return: Whether the timer started or not
        """
        if cond_was_playing and not self._was_playing:
            return False
        self.pause()
        self._playback_timer.start(int(round(1000.0/frame_rate)))
        return True

    def is_playing(self) -> bool:
        """
        Gets the current playback status of the buffer.
        :return:
        """
        return self._playback_timer.isActive()

    def read_frames(self, start: int, num_frames: int):
        """
        Uses the video capture object to read the necessary number of frames
        :param start: The frame to begin reading from
        :param num_frames: The number of frames to read
        """
        reader_loc = self._reader.get(1)
        if reader_loc != start:
            self._reader.set(1, max(0, start))
        for i in range(num_frames):
            true_frame = start+i
            if 0 <= true_frame <= self._glo.visual_config.video_length:
                ret, frame = self._reader.read()
                if not ret:
                    # Then we probably hit the end of the video and we should
                    # update the max frame based on this
                    self._glo.visual_config.video_length = min(self._glo.visual_config.video_length, true_frame)
                    continue
                if ret:
                    # Then we put this frame into the frame marking pool
                    frame_marker = FrameMarker(target=self.process_frame,
                                               args=(true_frame, frame,))
                    self._thead_pool.start(frame_marker)

    def process_frame(self, frame_num, frame):
        """
        This handles all the actions necessary to take a raw frame into the
        frame that the user will see
        :param frame_num: The video frame number
        :param frame: A numpy array containing the current frame
        """
        # cv2 loads frames in BGR, but qt displays in RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # If the video is too large to display quickly or to small to show
        # detail when zoomed it, we resample to to a better resolution
        scale_factor = self._glo.visual_config.scale_factor
        if scale_factor != 1:
            frame = cv2.resize(frame, None, fx=scale_factor,
                               fy=scale_factor,
                               interpolation=cv2.INTER_CUBIC)
        # When zoomed in, a blur makes it easier to find details
        frame = cv2.GaussianBlur(frame, (3, 3), 0)
        marked_frame = self._marker.markup_image(frame, frame_num)
        self._new_frames.put((frame_num, frame, marked_frame))

    def update(self) -> bool:
        """
        Main loop. Handles all actions that should happen at the given fps
        :return: Whether an update actually occurred
        """
        if self._force_update:
            # Then we want to completely refresh the buffer
            if self._buffer[self._buffer_radius] is not None:
                # For user speed purposes, if the current frame is already in
                # the buffer, we remark and emit it
                index, frame, marked_frame = self._buffer[self._buffer_radius]
                marked_up = self._marker.markup_image(frame, index)
                self.new_frame_signal.emit((index, frame, marked_up))
            self.read_frames(self._curr_frame-self._buffer_radius, self._buffer_length)
            self._force_update = False

        if self._target_frame == self._curr_frame:
            return False
        # The frame delta shows us how many frames we need to refresh so we
        # don't do an unnecessary amount of computation
        delta = self._target_frame - self._curr_frame

        if abs(delta) <= self._buffer_radius:
            # Then the new frame is in the buffer so we should immediately emit
            # it in order to have a stable frame rate
            self.new_frame_signal.emit(self._buffer[delta+self._buffer_radius])
            self._curr_frame = self._target_frame

        start = 0
        read_num = 0
        if abs(delta) >= self._buffer_length:
            # Then we need to refresh the entire buffer
            start = self._target_frame-self._buffer_radius
            read_num = self._buffer_length
        elif delta > 0:
            # Then the frame to jump to is in the future
            start = self._curr_frame+self._buffer_radius
            read_num = delta
        elif delta < 0:
            # Then the frame to jump to is in the past
            start = self._target_frame-self._buffer_radius
            read_num = abs(delta)

        self.read_frames(start, read_num)
        self.update_buffer()
        return True

    def update_buffer(self):
        """
        Takes frames from the new frames queue and puts them in the frame buffer
        """
        new_buffer = [None for i in range(self._buffer_length)]
        for frame in self._buffer:
            # This loop places old frames into the correct positions in the new
            # buffer
            if frame is None:
                continue
            buffer_pos = frame[0] - self._target_frame + self._buffer_radius
            if 0 <= buffer_pos < self._buffer_length:
                try:
                    new_buffer[buffer_pos] = frame
                except TypeError:
                    continue
        while not self._new_frames.empty():
            # Then some threads have finished processing frames and the new
            # frames should be inserted into the buffer
            new_frame = self._new_frames.get()
            if new_frame is None:
                # Then the frame was outside of the video
                continue
            if new_frame[0] == self._target_frame:
                # Then the current frame has been processed and should be displayed
                self.new_frame_signal.emit(new_frame)
                self._curr_frame = self._target_frame
            buffer_pos = new_frame[0] - self._target_frame + self._buffer_radius
            if 0 <= buffer_pos < self._buffer_length:
                try:
                    new_buffer[int(buffer_pos)] = new_frame
                except TypeError as err:
                    print("Errored when creating the buffer: ", err)
        self._buffer = new_buffer

    def stop(self) -> bool:
        """Send the signal to end the buffer loop"""
        self._running = False
        return True

    def run(self):
        """Updates the buffer every millisecond"""
        while self._running:
            self.update()
            if not self._new_frames.empty():
                self.update_buffer()
            time.sleep(1/1000)

class ImageViewerV2(QtWidgets.QGraphicsView):
    """
    This widget displays the information coming from the frame buffer
    """
    _glo: Globals

    _zoom_factor: float = 0.1  # Defines the zoom speed
    _zoom: int = 0

    _scene: Optional[QtWidgets.QGraphicsScene] = None
    _display: Optional[QtWidgets.QGraphicsPixmapItem] = None

    _reader: Optional[cv2.VideoCapture] = None

    _frame_buffer: Optional[FrameBufferV2] = None

    _resize_on_get_frame: bool = True

    _creating_metric: bool = True

    frame_change_signal = QtCore.pyqtSignal(int)  # Emitted when frame number changes
    playback_change_signal = QtCore.pyqtSignal(bool)  # Emitted when video is paused or played

    _held_keys: List = []
    _mouse_positions: List[Tuple[float, float]] = [(-1, -1) for i in range(4)]
    _mode: DataHolders.InteractionMode
    onMode = DataHolders.InteractionMode.POINT

    _lifted_point: Optional[int] = None

    def __init__(self):
        super(ImageViewerV2, self).__init__()
        self._glo = Globals.get()
        self.setup_view_window()
        self.fitInView()
        self._frame_buffer = FrameBufferV2()
        self._frame_buffer.new_frame_signal.connect(self.on_new_frame)
        self._frame_buffer.start()
        self._frame_buffer.refresh_image_signal.connect(self.start_refresh)
        self._glo.onProjectChange.connect(self.setup_video_config_connections)

    def setup_video_config_connections(self):
        """Allows the video config to change the view"""
        self._glo.video_config.setPosition.connect(self.seek_frame)
        self._glo.video_config.remarkFrames.connect(lambda: self._frame_buffer.update_curr_frame(remark=True))

    def start_refresh(self):
        self._resize_on_get_frame = True

    def setup_view_window(self) -> bool:
        """
        Creates the graphics renderer that will show the frames
        """
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

    def set_display(self, image: np.ndarray) -> bool:
        """
        Displays the given numpy image on the screen
        :param image: An ndarray containing the image to be displayed
        :return: The success value
        """
        height, width, channels = image.shape
        bytesPerLine = 3 * width
        qt_img = QtGui.QImage(image.data, width, height, bytesPerLine, QtGui.QImage.Format_RGB888)
        pixmap = QtGui.QPixmap.fromImage(qt_img)

        # Updates the view with the new pixmap
        # TODO: Update this for QML: https://stackoverflow.com/questions/42080565/how-to-paint-sequential-image-in-efficient-way-in-qquickpainteditem
        if pixmap and not pixmap.isNull():
            self.setDragMode(QtWidgets.QGraphicsView.RubberBandDrag)
            self._display.setPixmap(pixmap)
            return True
        else:
            self.setDragMode(QtWidgets.QGraphicsView.NoDrag)
            self._display.setPixmap(QtGui.QPixmap())
            return False

    def get_current_view(self) -> Optional[np.ndarray]:
        """
        Used to print out a picture of the current screen
        :return: The current frame
        """
        if not self._glo.ready():
            return None
        return self._frame_buffer.get_marked_frame()

    @QtCore.pyqtSlot(object, name="DisplayNewFrame")
    def on_new_frame(self, frame_info: Optional[Tuple[int, np.ndarray, Optional[np.ndarray]]]) -> bool:
        """Handler for the new frame signal from the buffer"""
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

    def play(self, cond_was_playing=False) -> bool:
        """
        Starts the video buffer if everything is ready
        :param cond_was_playing:
        :return: Whether the video is now playing
        """
        if not self._glo.ready():
            return False

        is_playing = self._frame_buffer.play(frame_rate=self._glo.visual_config.fps*self._glo.visual_config.playback_speed, cond_was_playing=cond_was_playing)
        self.playback_change_signal.emit(is_playing)
        return is_playing

    def toggle_play(self) -> bool:
        """
        Toggles the playback state of the frame buffer
        :return: Whether the video is now playing
        """
        if not self._glo.ready():
            return False
        if self._frame_buffer.is_playing():
            self.pause()
            return True
        else:
            self.play()
            return True

    def pause(self) -> bool:
        """Pauses the frame buffer"""
        if not self._glo.ready():
            return False
        self.playback_change_signal.emit(False)
        self._glo.video_config.set_position(self.get_curr_frame())
        return self._frame_buffer.pause()

    def jump_frames(self, delta=1) -> bool:
        """Jumps the frame buffer by delta frames"""
        if not self._glo.ready():
            return False
        self._glo.video_config.set_position(self.get_curr_frame() + delta)
        return self._frame_buffer.set_frame(delta=delta)

    def seek_frame(self, frame=0) -> bool:
        """
        Sets the buffer to the given frame
        :param frame: The frame to jump to
        :return:
        """
        if not self._glo.ready():
            return False
        self._glo.video_config.set_position(frame)
        return self._frame_buffer.set_frame(index=frame)

    def get_curr_frame(self) -> int:
        """
        :return: The frame buffer's last frame
        """
        if not self._glo.ready():
            return -1
        return self._frame_buffer.get_frame_num()

    def wheelEvent(self, event):
        """
        Main zoom loop. The zoom factor is displaced by the angleDelta of the
        mouse wheel
        """
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
        """
        Resets the zoom factor to 1
        """
        rect = QtCore.QRectF(self._display.pixmap().rect())
        if not rect.isNull():
            unity = self.transform().mapRect(QtCore.QRectF(0, 0, 1, 1))
            self.scale(1 / unity.width(), 1 / unity.height())
            viewrect = self.viewport().rect()
            scenerect = self.transform().mapRect(rect)
            factor = min(viewrect.width() / scenerect.width(), viewrect.height() / scenerect.height())
            self.scale(factor, factor)
            self.setSceneRect(rect)
            self.centerOn(rect.center())
            self._zoom = 0

    def reset_held_keys(self):
        """
        Sets the held keys to none
        """
        self._held_keys = []
        self.setDragMode(QtWidgets.QGraphicsView.NoDrag)
        self._mode = DataHolders.InteractionMode.POINT

    def keyPressEvent(self, event: QtCore.QEvent):
        """Handles key press actions"""
        key = event.key()
        if key == QtCore.Qt.Key_Shift:
            # Then the user wants to select an area
            self.setDragMode(QtWidgets.QGraphicsView.RubberBandDrag)
            self._mode = DataHolders.InteractionMode.AREA
        if key == QtCore.Qt.Key_Escape:
            # Then the user wants to deselect all leandmarks
            self._glo.deselect_landmarks()
            self._frame_buffer.update_curr_frame(remark=True)

        if key not in self._held_keys:
            self._held_keys.append(key)

    def keyReleaseEvent(self, event):
        """Handles key release actions"""
        key = event.key()
        if key == QtCore.Qt.Key_Shift:
            # Then we want to go back into standard point mode
            self.setDragMode(QtWidgets.QGraphicsView.NoDrag)
            self._mode = DataHolders.InteractionMode.POINT
        if key in self._held_keys:
            self._held_keys.remove(key)

    def mousePressEvent(self, event):
        """Allows the user to interact with the scene"""
        if self._glo.ready():
            scale_factor = self._glo.visual_config.scale_factor
            scene_pos = self.mapToScene(event.pos())
            mouse_pos = (scene_pos.toPoint().x()/scale_factor, scene_pos.toPoint().y()/scale_factor)
            button = event.button()
            self._mouse_positions[0] = mouse_pos
            self._mouse_positions[2] = event.pos()
            if self._mode == DataHolders.InteractionMode.POINT:
                self.setDragMode(QtWidgets.QGraphicsView.ScrollHandDrag)
            if self._mode == DataHolders.InteractionMode.AREA:
                self.setDragMode(QtWidgets.QGraphicsView.RubberBandDrag)
        QtWidgets.QGraphicsView.mousePressEvent(self, event)

    def mouseReleaseEvent(self, event):
        """Handles actions taken after the user releases the mouse pointer"""
        if self._glo.ready():
            self.setDragMode(QtWidgets.QGraphicsView.NoDrag)
            scale_factor = self._glo.visual_config.scale_factor
            scene_pos = self.mapToScene(event.pos())
            mouse_pos = (scene_pos.toPoint().x() / scale_factor,
                         scene_pos.toPoint().y() / scale_factor)
            button = event.button()
            self._mouse_positions[1] = mouse_pos
            self._mouse_positions[3] = event.pos()
            if self._mode == DataHolders.InteractionMode.POINT:
                if self._mouse_positions[2] == self._mouse_positions[3]:
                    # Then the user did not drag
                    if button == QtCore.Qt.RightButton:
                        # Then the user wants to pick up the landmark
                        self.edit_locations(mouse_pos)
                    else:
                        # Then the user is defining a metric
                        self.select_landmarks(mouse_pos)
            if self._mode == DataHolders.InteractionMode.AREA:
                # Then the user is defining a centroid
                self.select_landmarks()
        QtWidgets.QGraphicsView.mouseReleaseEvent(self, event)

    def edit_locations(self, mouse_pos: Tuple[float, float]) -> bool:
        """
        Called on clicks. Handles actions related to the editing of landmark
        locations
        :param mouse_pos: The position of the click cast to the original scale
        :return: The success value
        """
        frame_num = self._frame_buffer.get_frame_num()
        if self._lifted_point is None:
            self._lifted_point = self._glo.curr_landmarks.get_nearest_point(frame_num, mouse_pos)
            if self._lifted_point is not None:
                # Then the user clicked on a point to move it
                self._glo.visual_config.exclude([self._lifted_point])
                self._frame_buffer.update_curr_frame()
            else:
                return False
        else:
            resolution = self._glo.visual_config.resolution
            if mouse_pos[0] < 0 or mouse_pos[0] > resolution[0] or mouse_pos[1] < 0 or mouse_pos[1] > resolution[1]:
                # Then the user tried to place the landmark outside of the video area
                return False
            self._glo.move_landmark(frame_num, self._lifted_point, mouse_pos)
            self._glo.visual_config.include([self._lifted_point])
            self._frame_buffer.update_curr_frame()
            self._lifted_point = None
        return True

    def select_landmarks(self, mouse_pos: Optional[Tuple[float, float]] = None) -> bool:
        """
        Called on clicks. Handles actions taken when a user is creating a metric
        :param mouse_pos: The position of the click
        """
        frame_num = self._frame_buffer.get_frame_num()
        if self._mode == DataHolders.InteractionMode.POINT:
            selected_landmark = self._glo.curr_landmarks.get_nearest_point(frame_num, mouse_pos)
            if selected_landmark is None:
                self._glo.deselect_landmarks()
            else:
                self._glo.select_landmark(selected_landmark)
        elif self._mode == DataHolders.InteractionMode.AREA:
            x_max = max(self._mouse_positions[0][0], self._mouse_positions[1][0])
            x_min = min(self._mouse_positions[0][0], self._mouse_positions[1][0])
            y_max = max(self._mouse_positions[0][1], self._mouse_positions[1][1])
            y_min = min(self._mouse_positions[0][1], self._mouse_positions[1][1])
            selected_landmarks = self._glo.curr_landmarks.get_point_area(frame_num, x_max, x_min, y_max, y_min)
            if len(selected_landmarks) < 1:
                self._glo.deselect_landmarks()
            else:
                self._glo.select_landmark(selected_landmarks)
        self._frame_buffer.update_curr_frame(remark=True)
        return True

# class FrameBuffer(QtCore.QThread):
#     """
#     This whole thing is pretty ugly so I'll give a rundown of why things are
#     done this way. Basically, we need to be dynamically displaying content on
#     top of a visual being shown in real time. On modern computer, this works
#     fine without any threading. On an older computer, however, it runs at a much
#     lower frame rate. At first, I tried to just run the marker in one separate
#     thread, but I found that that still lagged on some systems. Therefore, I
#     created a thread pool to handle it instead. It seems well optimized so it
#     works well most of the time. This does require some more complicated logic
#     however and means some frames may load later than expected.
#     """
#     _thead_pool: QtCore.QThreadPool
#
#     _reader: Optional[cv2.VideoCapture] = None
#     _video_file: str = None
#     _marker: Optional[ImageMarker.ImageMarkerV2] = None
#
#     _buffer: list
#     _new_frames: Queue = Queue()
#     _buffer_radius: int
#     _buffer_length: int
#
#     _scale_factor: float = 1
#     _target_frame: int = 0
#     _curr_frame: int = 0
#     _max_frame: int = -100
#
#     new_frame_signal = QtCore.pyqtSignal(object)
#
#     _force_update = False
#     _was_playing = False
#     _running = True
#
#     def __init__(self, reader: Optional[cv2.VideoCapture] = None,
#                  video_file: str = None,
#                  marker: Optional[ImageMarker.ImageMarker] = None,
#                  scale_factor: float = 1,
#                  buffer_radius=15):
#         super(FrameBuffer, self).__init__()
#         self._thead_pool = QtCore.QThreadPool()
#
#         self.set_reader(reader)
#         self._video_file = video_file
#         self._marker = marker
#
#         self._buffer_radius = buffer_radius
#         self._buffer_length = 1 + 2 * buffer_radius
#         self._buffer = [None for i in range(self._buffer_length)]
#
#         self._scale_factor = scale_factor
#
#         self._playback_timer = QtCore.QTimer()
#         self._playback_timer.timeout.connect(
#             lambda: self.set_frame(delta=1))
#
#     def calc_frames(self, vid_file: str):
#         if vid_file is None:
#             return None
#         reader = cv2.VideoCapture(vid_file)
#         max_frame = -2
#         reader.set(0, 0)
#         ret = True
#         while ret:
#             ret, frame = reader.read()
#             max_frame += 1
#         self._max_frame = max_frame
#
#     def set_reader(self, reader: Optional[cv2.VideoCapture], scale_factor: float = 1) -> bool:
#         if reader is None:
#             return False
#         self._reader = reader
#         self._max_frame = int(reader.get(cv2.CAP_PROP_FRAME_COUNT)) - 1
#
#         # self._max_frame = -2
#         # self._reader.set(0, 0)
#         # ret = True
#         # while ret:
#         #     ret, frame = self._reader.read()
#         #     self._max_frame += 1
#         # self._reader.set(0, 0)
#         self._scale_factor = scale_factor
#         return True
#
#     def set_img_marker(self, img_marker: ImageMarker.ImageMarkerV2) -> bool:
#         if img_marker is None:
#             return False
#         self._marker = img_marker
#         return True
#
#     def init_buffer(self) -> bool:
#         if self._reader is not None and self._marker is not None:
#             self.set_frame(0)
#             self._force_update = True
#         return True
#
#     def set_frame(self, index: int = None, delta: int = None) -> bool:
#         # If index is used, then it specifies the exact frame. Delta specifies a
#         # number of frames to jump
#         if index is None:
#             index = self._curr_frame
#         if delta is not None:
#             index += delta
#         if index > self._max_frame:
#             self.pause()
#         self._target_frame = max(0, min(index, self._max_frame))
#         return True
#
#     def update_curr_frame(self, remark=True) -> bool:
#         # Remarkup the current frame and then emit a frame change
#         if remark:
#             self._force_update = True
#         else:
#             self.new_frame_signal.emit(self._buffer[self._buffer_radius])
#         return True
#
#     def get_frame_num(self) -> int:
#         return self._curr_frame
#
#     def get_marked_frame(self) -> Optional[np.ndarray]:
#         frame = self._buffer[self._buffer_radius]
#         if frame is not None:
#             return frame[2]
#         return None
#
#     def play(self, frame_rate: float=30, cond_was_playing: bool=False) -> bool:
#         if cond_was_playing and not self._was_playing:
#             return False
#         self.pause()
#         self._playback_timer.start(1000.0/frame_rate)
#         return True
#
#     def pause(self) -> bool:
#         self._was_playing = self._playback_timer.isActive()
#         if self._playback_timer.isActive():
#             self._playback_timer.stop()
#             return True
#         return False
#
#     def is_playing(self) -> bool:
#         return self._playback_timer.isActive()
#
#     def read_frames(self, start: int, num_frames: int):
#         reader_loc = self._reader.get(1)
#         if reader_loc != start:
#             self._reader.set(1, max(0, start))
#         for i in range(num_frames):
#             true_frame = start+i
#             if 0 <= true_frame <= self._max_frame:
#                 ret, frame = self._reader.read()
#                 if not ret:
#                     self._max_frame = min(self._max_frame, true_frame)
#                     continue
#                 if ret:
#                     frame_marker = FrameMarker(target=self.process_frame,
#                                                args=(true_frame, frame,))
#                     self._thead_pool.start(frame_marker)
#
#     def process_frame(self, frame_num, frame):
#         frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         if self._scale_factor != 1:
#             frame = cv2.resize(frame, None, fx=self._scale_factor,
#                                fy=self._scale_factor,
#                                interpolation=cv2.INTER_CUBIC)
#         frame = cv2.GaussianBlur(frame, (3, 3), 0)
#         marked_frame = self._marker.markup_image(frame, frame_num)
#         self._new_frames.put((frame_num, frame, marked_frame))
#
#     def update(self) -> bool:
#         if self._force_update:
#             updated_current = False
#             # Then we want to completely refresh the buffer
#             if self._buffer[self._buffer_radius] is not None:
#                 # For user speed purposes, if the current frame is already in
#                 # the buffer, we remark and emit it
#                 index, frame, marked_frame = self._buffer[self._buffer_radius]
#                 marked_up = self._marker.markup_image(frame, index)
#                 self.new_frame_signal.emit((index, frame, marked_up))
#                 updated_current = True
#             self.read_frames(self._curr_frame-self._buffer_radius, self._buffer_length)
#             self._force_update = False
#
#         if self._target_frame == self._curr_frame:
#             return False
#
#         delta = self._target_frame - self._curr_frame
#
#         if abs(delta) <= self._buffer_radius:
#             self.new_frame_signal.emit(self._buffer[delta+self._buffer_radius])
#             self._curr_frame = self._target_frame
#
#         start = 0
#         read_num = 0
#         if abs(delta) >= self._buffer_length:
#             start = self._target_frame-self._buffer_radius
#             read_num = self._buffer_length
#         elif delta > 0:
#             start = self._curr_frame+self._buffer_radius
#             read_num = delta
#         elif delta < 0:
#             start = self._target_frame-self._buffer_radius
#             read_num = abs(delta)
#
#         self.read_frames(start, read_num)
#         self.update_buffer()
#
#     def update_buffer(self) -> bool:
#         new_buffer = [None for i in range(self._buffer_length)]
#         for frame in self._buffer:
#             if frame is None:
#                 continue
#             buffer_pos = frame[0] - self._target_frame + self._buffer_radius
#             if 0 <= buffer_pos < self._buffer_length:
#                 new_buffer[buffer_pos] = frame
#         while not self._new_frames.empty():
#             new_frame = self._new_frames.get()
#             if new_frame is None:
#                 continue
#             if new_frame[0] == self._target_frame:
#                 self.new_frame_signal.emit(new_frame)
#                 self._curr_frame = self._target_frame
#             buffer_pos = new_frame[0] - self._target_frame + self._buffer_radius
#             if 0 <= buffer_pos < self._buffer_length:
#                 new_buffer[buffer_pos] = new_frame
#         self._buffer = new_buffer
#         return True
#
#     def stop(self) -> bool:
#         self._running = False
#         return True
#
#     def run(self):
#         while self._running:
#             self.update()
#             if not self._new_frames.empty():
#                 self.update_buffer()
#             time.sleep(1/1000)
#
# class ImageViewer(QtWidgets.QGraphicsView):
#     _zoom_factor: float = 0.1  # Defines the zoom speed
#     _zoom: int = 0
#     _scene: Optional[QtWidgets.QGraphicsScene] = None
#     _display: Optional[QtWidgets.QGraphicsPixmapItem] = None
#
#     _reader: Optional[cv2.VideoCapture] = None
#     _vid_file: Optional[str] = None
#     _min_dim: int = 1920
#     _video_length: int = -1
#     _playback_speed: float = 1
#     _video_fps: float = -1
#     _resolution: Tuple[int, int] = (-1, -1)  # Width, Height
#     _scale_factor: float = 1
#     _frame_buffer: Optional[FrameBuffer] = None
#     _img_marker: Optional[ImageMarker.ImageMarker] = None
#
#     _resize_on_get_frame: bool = True
#
#     _landmarks: Optional[DataHolders.Landmarks] = None
#     _landmarks_frame:  Optional[pd.DataFrame] = None
#
#     _working_metrics: List[DataHolders.Metric] = []
#     _metric_selector: List[int] = []
#     _creating_metric: bool = True
#
#     glo: Globals = None
#
#     frame_change_signal = QtCore.pyqtSignal(int)  # Emitted when frame number changes
#     playback_change_signal = QtCore.pyqtSignal(bool)  # Emitted when video is paused or played
#     metadata_update_signal = QtCore.pyqtSignal(int, float, object)  # Emits length, frame rate, resolution
#     point_moved_signal = QtCore.pyqtSignal(bool, int, int, DataHolders.Landmarks)  # Emitted when a landmark is moved. Lifted state, landmark number, frame
#
#     _held_keys: List = []
#     _mouse_positions: List[Tuple[int, int]] = [(-1, -1) for i in range(4)]
#     _mode: DataHolders.InteractionMode
#     onMode = DataHolders.InteractionMode.POINT
#
#     _lifted_point: Optional[int] = None
#
#     def __init__(self, glo: Globals, vid_file: str = None, reader: cv2.VideoCapture=None, landmarks: pd.DataFrame=None):
#         super(ImageViewer, self).__init__()
#         self.glo = glo
#         self.setup_view_window()
#         if vid_file is not None:
#             self._vid_file = vid_file
#         if reader is not None:
#             self.set_reader(reader)
#         if landmarks is not None:
#             self.set_landmarks(landmarks)
#         self.fitInView()
#
#     def reset(self) -> bool:
#         """
#         Ends all running processes and clears variables in order to render and new video
#         """
#         if self._frame_buffer is not None:
#             self._frame_buffer.stop()
#         self._frame_buffer = None
#         self._landmarks = None
#         self._landmarks_frame = None
#         self._reader = None
#         return True
#
#     def ready(self) -> bool:
#         """
#         Returns whether the viewer is ready to render frames
#         """
#         buffer_ready = self._frame_buffer is not None
#         return buffer_ready
#
#     def setup_view_window(self) -> bool:
#         """
#         Creates the graphics renderer that will show the frames
#         """
#         self._scene = QtWidgets.QGraphicsScene(self)
#         self._display = QtWidgets.QGraphicsPixmapItem()
#         self._scene.addItem(self._display)
#         self.setScene(self._scene)
#         self.setTransformationAnchor(QtWidgets.QGraphicsView.AnchorUnderMouse)
#         self.setResizeAnchor(QtWidgets.QGraphicsView.AnchorUnderMouse)
#         self.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
#         self.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
#         self.setBackgroundBrush(QtGui.QBrush(QtGui.QColor(100, 100, 100)))
#         self.setFrameShape(QtWidgets.QFrame.NoFrame)
#         self.setDragMode(QtWidgets.QGraphicsView.RubberBandDrag)
#         self.setMouseTracking(True)
#         return True
#
#     def set_reader(self, reader: cv2.VideoCapture, vid_file: Optional[str] = None) -> bool:
#         """
#
#         """
#         if vid_file is not None:
#             self._vid_file = vid_file
#         if self.reader_is_good(reader):
#             self._reader = reader
#             # self._video_length = -1
#             # ret = True
#             # while ret:
#             #     ret, frame = self._reader.read()
#             #     self._video_length += 1
#             # self._reader.set(0, 0)
#             self._video_length = int(reader.get(cv2.CAP_PROP_FRAME_COUNT))
#             self._video_fps = int(self._reader.get(5))
#             self._resolution = (int(self._reader.get(3)), int(self._reader.get(4)))
#             self._scale_factor = max(self._min_dim / max(self._resolution), 1)
#             if self._img_marker is not None:
#                 self._img_marker.set_scale_factor(self._scale_factor)
#             self.metadata_update_signal.emit(
#                 self._video_length,
#                 self._video_fps,
#                 self._resolution
#             )
#             self.update_frame_buffer()
#             self._resize_on_get_frame = True
#             return True
#         return False
#
#     def reader_is_good(self, reader: Optional[cv2.VideoCapture]=None) -> bool:
#         if reader is None:
#             reader = self._reader
#         return reader is not None and reader.isOpened()
#
#     def set_landmarks(self, landmarks: pd.DataFrame) -> bool:
#         self._landmarks = DataHolders.Landmarks(landmarks, n_landmarks=68)
#         self.setup_default_groups()
#         self._img_marker = ImageMarker.ImageMarker(self._landmarks, self._scale_factor, self.glo.config)
#         self._img_marker.set_metrics(self.glo.config.metrics.get_all(), self._working_metrics)
#
#         self.update_frame_buffer()
#         return True
#
#     def update_frame_buffer(self) -> bool:
#         if self._landmarks is None or not self.reader_is_good():
#             return False
#         if self._frame_buffer is None:
#             self._img_marker = ImageMarker.ImageMarker(self._landmarks, self._scale_factor, self.glo.config)
#             self._img_marker.set_metrics(self.glo.config.metrics.get_all(), self._working_metrics)
#             self._frame_buffer = FrameBuffer(self._reader, self._vid_file, self._img_marker, buffer_radius=15, scale_factor=self._scale_factor)
#             self._frame_buffer.new_frame_signal.connect(self.on_new_frame)
#         else:
#             self._frame_buffer.set_img_marker(self._img_marker)
#             self._frame_buffer.set_reader(self._reader, self._scale_factor)
#         try:
#             self._resize_on_get_frame = True
#             self._frame_buffer.init_buffer()
#             self._frame_buffer.start()
#             return True
#         except AttributeError:
#             return False
#
#     def set_display(self, image: np.ndarray) -> bool:
#         height, width, channels = image.shape
#         bytesPerLine = 3 * width
#         qt_img = QtGui.QImage(image.data, width, height, bytesPerLine, QtGui.QImage.Format_RGB888)
#         pixmap = QtGui.QPixmap.fromImage(qt_img)
#
#         # Updates the view with the new pixmap
#         if pixmap and not pixmap.isNull():
#             self.setDragMode(QtWidgets.QGraphicsView.RubberBandDrag)
#             self._display.setPixmap(pixmap)
#             return True
#         else:
#             self.setDragMode(QtWidgets.QGraphicsView.NoDrag)
#             self._display.setPixmap(QtGui.QPixmap())
#             return False
#
#     def get_current_view(self) -> Optional[np.ndarray]:
#         if not self.ready():
#             return None
#         return self._frame_buffer.get_marked_frame()
#
#     @QtCore.pyqtSlot(object, name="DisplayNewFrame")
#     def on_new_frame(self, frame_info: Optional[Tuple[int, np.ndarray, Optional[np.ndarray]]]) -> bool:
#         if frame_info is None:
#             return False
#         frame_num, frame, marked_frame = frame_info
#         frame = marked_frame if marked_frame is not None else frame
#         self.frame_change_signal.emit(frame_num)
#         self.set_display(frame)
#         if self._resize_on_get_frame:
#             self.fitInView()
#             self._resize_on_get_frame = False
#         return True
#
#     def add_to_group(self, name: str, indices: List[int]) -> bool:
#         group_colors = self.glo.config.group.colors
#         if name not in group_colors:
#             group_colors[name] = (255, 255, 255)
#         self._landmarks.set_group(indices, name)
#         return True
#
#     def setup_default_groups(self) -> bool:
#         # TODO: Define the default groups
#         self.add_to_group("lower_eye", [40, 41, 46, 47])
#         # self.add_to_group("upper_mouth", [61, 62, 63, 64, 65])
#         return True
#
#     def toggle_landmarks(self) -> bool:
#         if self._img_marker is None:
#             return False
#         val = self._img_marker.toggle_landmarks()
#         if not self._frame_buffer.is_playing():
#             self._frame_buffer.update_curr_frame()
#         return val
#
#     def toggle_bounding_box(self) -> bool:
#         if self._img_marker is None:
#             return False
#         val = self._img_marker.toggle_bounding_box()
#         if not self._frame_buffer.is_playing():
#             self._frame_buffer.update_curr_frame()
#         return val
#
#     def toggle_metrics(self) -> bool:
#         if self._img_marker is None:
#             return False
#         val = self._img_marker.toggle_metrics()
#         if not self._frame_buffer.is_playing():
#             self._frame_buffer.update_curr_frame()
#         return val
#
#     def play(self, cond_was_playing=False) -> bool:
#         if not self.ready():
#             return False
#         is_playing = self._frame_buffer.play(frame_rate=self._video_fps*self._playback_speed, cond_was_playing=cond_was_playing)
#         self.playback_change_signal.emit(is_playing)
#         return is_playing
#
#     def toggle_play(self) -> bool:
#         if not self.ready():
#             return False
#         if self._frame_buffer.is_playing():
#             self.pause()
#             return True
#         else:
#             self.play()
#             return True
#
#     def pause(self) -> bool:
#         if not self.ready():
#             return False
#         self.playback_change_signal.emit(False)
#         return self._frame_buffer.pause()
#
#     def is_playing(self) -> bool:
#         if self.ready():
#             return self._frame_buffer.is_playing()
#         else:
#             return False
#
#     def set_playback_speed(self, speed: float = 1) -> bool:
#         self._playback_speed = speed
#         if not self.ready():
#             return False
#         return self._frame_buffer.play(frame_rate = self._video_fps*self._playback_speed, cond_was_playing=True)
#
#     def jump_frames(self, delta=1) -> bool:
#         if not self.ready():
#             return False
#         return self._frame_buffer.set_frame(delta=delta)
#
#     def seek_frame(self, frame=0) -> bool:
#         if not self.ready():
#             return False
#         return self._frame_buffer.set_frame(index=frame)
#
#     def get_curr_frame(self) -> int:
#         if not self._ready():
#             return -1
#         return self._frame_buffer.get_frame_num()
#
#     def wheelEvent(self, event):
#         # this take care of the zoom, it modifies the zoom factor if the mouse
#         # wheel is moved forward or backward
#         if not self._display.pixmap().isNull():
#             move = (event.angleDelta().y() / 120)
#             if move > 0:
#                 factor = 1 + self._zoom_factor
#                 self._zoom += 1
#             elif move < 0:
#                 factor = 1 - self._zoom_factor
#                 self._zoom -= 1
#             else:
#                 factor = 1
#
#             if self._zoom > 0 or (self._zoom == 0 and factor > 1):
#                 self.scale(factor, factor)
#             if self._zoom < 0:
#                 self._zoom = 0
#                 self.fitInView()
#
#     def fitInView(self):
#         # this function takes care of accommodating the view so that it can fit
#         # in the scene, it resets the zoom to 0 (i think is a overkill, i took
#         # it from somewhere else)
#         rect = QtCore.QRectF(self._display.pixmap().rect())
#         if not rect.isNull():
#             unity = self.transform().mapRect(QtCore.QRectF(0, 0, 1, 1))
#             self.scale(1 / unity.width(), 1 / unity.height())
#             viewrect = self.viewport().rect()
#             scenerect = self.transform().mapRect(rect)
#             factor = min(viewrect.width() / scenerect.width(), viewrect.height() / scenerect.height())
#             self.scale(factor, factor)
#             self.setSceneRect(rect)
#             self.centerOn(rect.center())
#             self._zoom = 0
#
#     def reset_held_keys(self):
#         self._held_keys = []
#         self.setDragMode(QtWidgets.QGraphicsView.NoDrag)
#         self._mode = DataHolders.InteractionMode.POINT
#
#     def keyPressEvent(self, event: QtCore.QEvent):
#         key = event.key()
#         if key == QtCore.Qt.Key_Shift:
#             self.setDragMode(QtWidgets.QGraphicsView.RubberBandDrag)
#             self._mode = DataHolders.InteractionMode.AREA
#         if key == QtCore.Qt.Key_Escape:
#             self.deselect_landmarks()
#
#         if key not in self._held_keys:
#             self._held_keys.append(key)
#
#     def keyReleaseEvent(self, event):
#         key = event.key()
#         if key == QtCore.Qt.Key_Shift:
#             self.setDragMode(QtWidgets.QGraphicsView.NoDrag)
#             self._mode = DataHolders.InteractionMode.POINT
#         if key in self._held_keys:
#             self._held_keys.remove(key)
#
#     def mousePressEvent(self, event):
#         if self.ready():
#             scene_pos = self.mapToScene(event.pos())
#             mouse_pos = (scene_pos.toPoint().x()/self._scale_factor, scene_pos.toPoint().y()/self._scale_factor)
#             button = event.button()
#             self._mouse_positions[0] = mouse_pos
#             self._mouse_positions[2] = event.pos()
#             if self._mode == DataHolders.InteractionMode.POINT:
#                 self.setDragMode(QtWidgets.QGraphicsView.ScrollHandDrag)
#             if self._mode == DataHolders.InteractionMode.AREA:
#                 self.setDragMode(QtWidgets.QGraphicsView.RubberBandDrag)
#         QtWidgets.QGraphicsView.mousePressEvent(self, event)
#
#     def mouseReleaseEvent(self, event):
#         if self.ready():
#             self.setDragMode(QtWidgets.QGraphicsView.NoDrag)
#             scene_pos = self.mapToScene(event.pos())
#             mouse_pos = (scene_pos.toPoint().x() / self._scale_factor,
#                          scene_pos.toPoint().y() / self._scale_factor)
#             button = event.button()
#             self._mouse_positions[1] = mouse_pos
#             self._mouse_positions[3] = event.pos()
#             if self._mode == DataHolders.InteractionMode.POINT:
#                 if self._mouse_positions[2] == self._mouse_positions[3]:
#                     if button == QtCore.Qt.RightButton:
#                         self.edit_locations(mouse_pos)
#                     else:
#                         self.select_landmarks(mouse_pos)
#             if self._mode == DataHolders.InteractionMode.AREA:
#                 self.select_landmarks()
#         QtWidgets.QGraphicsView.mouseReleaseEvent(self, event)
#
#     def edit_locations(self, mouse_pos: Tuple[int, int]) -> bool:
#         frame_num = self._frame_buffer.get_frame_num()
#         if self._lifted_point is None:
#             self._lifted_point = self._landmarks.get_nearest_point(frame_num, mouse_pos)
#             if self._lifted_point is not None:
#                 self._img_marker.exclude([self._lifted_point])
#                 self._frame_buffer.update_curr_frame()
#                 self.point_moved_signal.emit(True, self._lifted_point, frame_num, self._landmarks)
#             else:
#                 return False
#         else:
#             if mouse_pos[0] < 0 or mouse_pos[0] > self._resolution[0] or mouse_pos[1] < 0 or mouse_pos[1] > self._resolution[1]:
#                 return False
#             self._landmarks.set_location(frame_num, self._lifted_point, mouse_pos)
#             self.point_moved_signal.emit(False, self._lifted_point, frame_num, self._landmarks)
#             self._img_marker.include([self._lifted_point])
#             self._landmarks.set_location(frame_num, self._lifted_point, mouse_pos)
#             self._frame_buffer.update_curr_frame()
#             self._lifted_point = None
#         return True
#
#     def select_landmarks(self, mouse_pos: Optional[Tuple[float, float]]=None) -> bool:
#         # TODO: Pressing escape should remove selection
#         frame_num = self._frame_buffer.get_frame_num()
#         if self._mode == DataHolders.InteractionMode.POINT:
#             selected_landmark = self._landmarks.get_nearest_point(frame_num, mouse_pos)
#             if selected_landmark is None:
#                 self._img_marker.deselect()
#                 self._metric_selector = []
#             else:
#                 self._img_marker.select([selected_landmark])
#                 self._metric_selector.append(selected_landmark)
#         elif self._mode == DataHolders.InteractionMode.AREA:
#             x_max = max(self._mouse_positions[0][0], self._mouse_positions[1][0])
#             x_min = min(self._mouse_positions[0][0], self._mouse_positions[1][0])
#             y_max = max(self._mouse_positions[0][1], self._mouse_positions[1][1])
#             y_min = min(self._mouse_positions[0][1], self._mouse_positions[1][1])
#             selected_landmarks = self._landmarks.get_point_area(frame_num, x_max, x_min, y_max, y_min)
#             if len(selected_landmarks) < 1:
#                 self._img_marker.deselect()
#                 self._metric_selector = []
#             else:
#                 self._img_marker.select(selected_landmarks)
#                 self._metric_selector.append(selected_landmarks)
#         self.check_metric_completion()
#         self._frame_buffer.update_curr_frame(remark=True)
#         return True
#
#     def deselect_landmarks(self) -> bool:
#         if self.ready():
#             self._img_marker.deselect()
#             self._metric_selector = []
#             self.check_metric_completion()
#             self._frame_buffer.update_curr_frame(remark=True)
#             return True
#         return False
#
#     def check_metric_completion(self) -> bool:
#         if len(self._metric_selector) > 1 and self._metric_selector[-1] == self._metric_selector[0]:
#             self.glo.config.metrics.add(DataHolders.Metric(
#                 name=f"PLACEHOLDER:{','.join([str(landmark) for landmark in self._metric_selector[:-1]])}",
#                 type=DataHolders.MetricType.LENGTH,
#                 landmarks=self._metric_selector[:-1]
#             ))
#             self.glo.config.save()
#             self._metric_selector = []
#             self._img_marker.deselect()
#             return True
#         else:
#             self._working_metrics.clear()
#             if self._creating_metric:
#                 self._working_metrics.append(DataHolders.Metric(
#                     name="Working",
#                     type=DataHolders.MetricType.LENGTH,
#                     landmarks=self._metric_selector.copy()
#                 ))
#         return False
#
#     def remove_metric(self) -> bool:
#         # Removes the metric represented by the selected landmarks
#         # TODO: Make this work with centroids
#         for i, metric in enumerate(self._metrics):
#             all_included = True
#             for landmark in self._metric_selector:
#                 all_included = all_included and landmark in metric.landmarks
#             if all_included:
#                 self.glo.config.metrics.remove(metric)
#         self._frame_buffer.update_curr_frame()
#         return True
#
#     def get_landmarks(self) -> DataHolders.Landmarks:
#         return self._landmarks
#
#     def get_edits(self) -> pd.DataFrame:
#         return self._landmarks.get_dataframe()


