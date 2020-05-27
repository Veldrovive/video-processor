from utils.qmlBase import WindowHandler
from PyQt5 import QtQuick, QtCore
import cv2
from utils.Globals import Globals
from typing import List, Dict, Set, Tuple
import os

import pandas as pd
from landmark_detection.Detector import LandmarkDetectorV2 as LandmarkDetector


class FileListModel(QtCore.QAbstractListModel):
    _glo: Globals

    FileNameRole = QtCore.Qt.UserRole + 1

    @property
    def files(self):
        try:
            return self._glo.project.file_names
        except AttributeError:
            return []

    _curr_files: List

    def __init__(self):
        self._curr_files = []
        self._glo = Globals.get()
        super().__init__()

    def data(self, index, role=None):
        row = index.row()
        if role == FileListModel.FileNameRole:
            return self._curr_files[row]

    def rowCount(self, parent=None, *args, **kwargs):
        return len(self._curr_files)

    def roleNames(self):
        return {
            FileListModel.FileNameRole: b'fileName'
        }

    def reset(self):
        self.beginRemoveColumns(QtCore.QModelIndex(), 0, self.rowCount())
        self._curr_files = []
        self.endRemoveRows()

        self.beginInsertRows(QtCore.QModelIndex(), 0, len(self.files) - 1)
        self._curr_files = self.files
        self.endInsertRows()


class KeypointListModel(QtCore.QAbstractListModel):
    KeypointRole = QtCore.Qt.UserRole + 2
    CheckedRole = QtCore.Qt.UserRole + 3

    curr_keypoints: List = []

    checked_keypoint_indexes: List = []

    @QtCore.pyqtSlot(int, result=bool, name="isChecked")
    def is_checked(self, index):
        return index in self.checked_keypoint_indexes

    def update_checked_indexes(self, indexes):
        self.checked_keypoint_indexes = [index for index, keypoint in enumerate(self.curr_keypoints) if keypoint in indexes]

    def data(self, index, role=None):
        row = index.row()
        if role == KeypointListModel.KeypointRole:
            return self.curr_keypoints[row] + 1

    def rowCount(self, parent=None, *args, **kwargs):
        return len(self.curr_keypoints)

    def roleNames(self):
        return {
            KeypointListModel.KeypointRole: b'keypoint',
            KeypointListModel.CheckedRole: b'checked'
        }

    def reset(self, keypoints: List[int]):
        self.beginRemoveColumns(QtCore.QModelIndex(), 0, self.rowCount())
        self.curr_keypoints = []
        self.endRemoveRows()

        self.beginInsertRows(QtCore.QModelIndex(), 0, len(keypoints) - 1)
        self.curr_keypoints = keypoints
        self.endInsertRows()


class LandmarkDetectionHandler(WindowHandler):
    file_list_model: FileListModel
    keypoint_list_model: KeypointListModel

    file: str = None
    frames_map: Dict = {}
    full_video_map: Dict = {}
    text_range_map: Dict = {}

    @QtCore.pyqtProperty(str)
    def project_name(self):
        return self._glo.project.name

    @QtCore.pyqtProperty(bool)
    def curr_full_video_state(self):
        return self.frames_map[self.file]['full_video']

    @QtCore.pyqtProperty(str)
    def curr_video_range_state(self):
        return self.frames_map[self.file]['text_range']

    @QtCore.pyqtProperty(int)
    def total_frames(self):
        all_frames = self.get_frames_map()
        total = 0
        for frames in all_frames.values():
            total += len(frames)
        return total

    projectChanged = QtCore.pyqtSignal()
    videoChanged = QtCore.pyqtSignal()
    numFramesChanged = QtCore.pyqtSignal()
    progressChanged = QtCore.pyqtSignal(float, arguments=["progress"])

    def __init__(self, engine: QtQuick.QQuickView):
        self.file_list_model = FileListModel()
        self.keypoint_list_model = KeypointListModel()
        super().__init__(engine, "uis/LandmarkDetectionView.qml",
                         "Detect Landmarks")
        self._list_view = self._window.findChild(QtCore.QObject, "listView")
        self._progress_bar = self._window.findChild(QtCore.QObject, "progressBar")
        self._glo.onProjectChange.connect(self.on_project_changed)

    def on_project_changed(self):
        """Handles actions taken when the project changes"""
        self._list_view.currentIndex = -1
        self.projectChanged.emit()

    def setup_contexts(self):
        """Overrides contexts to insert file list model"""
        self.add_context("fileListModel", self.file_list_model)
        self.add_context("keypointListModel", self.keypoint_list_model)

    def show(self):
        """Overrides the show function so that the list can be updated"""
        self.file_list_model.reset()
        self.setup_maps()
        self.get_video_data(0)
        super().show()

    def setup_maps(self):
        """Creates a map to the frame to find the landmarks of"""
        if self._glo.project is None:
            self.frames_map = {}
        else:
            curr_keys = list(self.frames_map.keys())
            for key in curr_keys:
                if key not in self._glo.project.files:
                    del self.frames_map[key]
            for file in self._glo.project.files:
                if file not in self.frames_map:
                    self.frames_map[file] = {
                        'frames': set(),
                        'full_video': False,
                        'text_range': None
                    }

    @QtCore.pyqtSlot(int, name="getVideoData")
    def get_video_data(self, index: int):
        try:
            self.file = self._glo.project.files[index]
            keypoints = self._glo.video_config.key_points[self.file]
            keypoints.sort()
            self.keypoint_list_model.reset(keypoints[1:-1])
            self.keypoint_list_model.update_checked_indexes(self.frames_map[self.file]['frames'])
            self._list_view.currentIndex = index
            self.videoChanged.emit()
            self._glo.select_file(self.file)
        except (IndexError, AttributeError):
            self.keypoint_list_model.reset([])

    @QtCore.pyqtSlot(int, bool, name="setFileKeypoint")
    def set_file_keypoint(self, keypoint_index: int, state: bool):
        key_frame = self.keypoint_list_model.curr_keypoints[keypoint_index]
        if state:
            self.frames_map[self.file]['frames'].add(key_frame)
        else:
            try:
                self.frames_map[self.file]['frames'].remove(key_frame)
            except KeyError:
                pass
        self.keypoint_list_model.update_checked_indexes(self.frames_map[self.file]['frames'])
        self.numFramesChanged.emit()

    @QtCore.pyqtSlot(bool, name="setFullVideo")
    def set_full_video(self, state: bool):
        if self.file is not None:
            self.frames_map[self.file]['full_video'] = state
            self.numFramesChanged.emit()

    @QtCore.pyqtSlot(str, name="setFrameRange")
    def set_frame_range(self, state: str):
        if self.file is not None:
            self.frames_map[self.file]['text_range'] = state
            self.numFramesChanged.emit()

    @staticmethod
    def frame_desc_to_list(desc: str) -> Tuple[Set[int], bool]:
        """
        Takes a description of numbers and turns it into a set containing all
        number included. I.E. (1-5, 8, 10) -> [1, 2, 3, 4, 5, 8, 10]
        :param desc: A string defining the numbers
        :return: A sorted set of the numbers
        """
        has_err = False
        frames = set()
        sections = [sec.strip() for sec in desc.split(",")]
        for section in sections:
            parts = [part.strip() for part in section.split("-")]
            if len(parts) == 1:
                try:
                    frames.add(int(parts[0]) - 1)
                except ValueError:
                    has_err = True
                    continue
            elif len(parts) > 1:
                try:
                    frames.update(range(int(parts[0]) - 1, int(parts[-1])))
                except ValueError:
                    has_err = True
                    continue
        return frames, has_err

    def get_frames_map(self) -> Dict:
        """Calculates the actual sets of frames that will be detected"""
        frame_map = {}
        for file, config in self.frames_map.items():
            full_frames = set()
            if config['full_video']:
                cap = cv2.VideoCapture(file)
                if cap is not None and cap.isOpened():
                    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    full_frames.update(range(frame_count))
            if config['text_range'] is not None and len(config['text_range']) > 0:
                frames, err = self.frame_desc_to_list(config['text_range'])
                full_frames.update(frames)
                # if err:
                #     self.send_message("Your range has a syntax error.\nAn example of the correct format is 1-10, 12, 14, 30-40")
            full_frames.update(config['frames'])
            frame_map[file] = set(sorted(full_frames))
        return frame_map

    def on_frame_done(self, frame_number: int, prop_complete: float):
        """Updates the visuals for when a frame is done"""
        self.progressChanged.emit(prop_complete)

    def on_video_done(self, video: str, landmarks: pd.DataFrame):
        """Updates landmarks with the new ones"""
        add_landmarks = False
        landmarks.set_index("Frame_number")
        landmark_file = self._glo.project.files_map[video]
        if landmark_file is None:
            add_landmarks = True
            old_landmarks = pd.DataFrame()
            landmark_file = os.path.splitext(video)[0]+".csv"
        else:
            old_landmarks = pd.read_csv(landmark_file)
        if old_landmarks.empty:
            old_landmarks["Frame_number"] = []
        old_landmarks.set_index("Frame_number")
        new_frame: pd.DataFrame = pd.concat([landmarks, old_landmarks]).drop_duplicates(["Frame_number"], keep='last').sort_values(by=['Frame_number'], ascending=False)
        new_frame.to_csv(landmark_file)
        if add_landmarks:
            self._glo.project.set_landmarks(video, landmark_file)
        self._glo.select_file(video)

    def on_inference_finished(self):
        """Updates the visuals for when the inference is totally done"""
        self.progressChanged.emit(1)
        self.send_message("Inference Finished")

    @QtCore.pyqtSlot(name="detect")
    def detect(self):
        detector = LandmarkDetector(self.get_frames_map())
        detector.frame_done_signal.connect(self.on_frame_done)
        detector.landmarks_complete_signal.connect(self.on_video_done)
        detector.inference_done_signal.connect(self.on_inference_finished)
        detector.start()


