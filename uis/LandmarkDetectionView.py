from utils.qmlBase import WindowHandler
from PyQt5 import QtQuick, QtCore
import cv2
from utils.Globals import Globals
from typing import List, Dict, Set, Tuple
import os

import pandas as pd
from landmark_detection.Detector import LightningFANDetector as LandmarkDetector

class LandmarkDetectionHandler(WindowHandler):
    _glo: Globals

    progressChanged = QtCore.pyqtSignal(float)  # Changes the progress bar position. 0=Empty, 1=Full

    _detecting: bool = False
    detectingUpdated = QtCore.pyqtSignal()
    @QtCore.pyqtProperty(bool, notify=detectingUpdated)
    def detecting(self):
        return self._detecting
    @detecting.setter
    def detecting(self, new_state: bool):
        self._detecting = new_state
        self.detectingUpdated.emit()

    # detecting: bool = False

    all_frames_map: Dict[str, bool]
    allFramesMapUpdated = QtCore.pyqtSignal()
    @QtCore.pyqtProperty(bool, notify=allFramesMapUpdated)
    def curr_all_frames(self):
        try:
            return self.all_frames_map[self.current_video]
        except KeyError:
            return False
    @curr_all_frames.setter
    def curr_all_frames(self, new_state: bool):
        self.all_frames_map[self.current_video] = new_state
        self.allFramesMapUpdated.emit()
        self.totalFramesUpdated.emit()

    some_frames_map: Dict[str, str]
    someFramesMapUpdated = QtCore.pyqtSignal()
    @QtCore.pyqtProperty(str, notify=someFramesMapUpdated)
    def curr_some_frames(self):
        try:
            return self.some_frames_map[self.current_video]
        except KeyError:
            return ""
    @curr_some_frames.setter
    def curr_some_frames(self, new_state: str):
        if len(new_state) == 0:
            self.some_frames_map[self.current_video] = new_state
            self.someFramesMapUpdated.emit()
            self.totalFramesUpdated.emit()
            return
        frames, has_err = self.frame_desc_to_list(new_state)
        if has_err:
            self.send_message("Invalid Frame Range")
        else:
            self.some_frames_map[self.current_video] = new_state
            self.totalFramesUpdated.emit()
        self.someFramesMapUpdated.emit()

    selected_keypoints_map: Dict[str, Set[int]]
    selectedKeypointsUpdated = QtCore.pyqtSignal()
    @QtCore.pyqtProperty(list, notify=selectedKeypointsUpdated)
    def curr_active_keypoints(self):
        try:
            return self.selected_keypoints_map[self.current_video]
        except KeyError:
            return []
    @QtCore.pyqtSlot(int, bool)
    def set_keypoint_active(self, keypoint: int, active: bool):
        keypoints = self.selected_keypoints_map[self.current_video]
        if active:
            keypoints.add(keypoint)
        else:
            keypoints.discard(keypoint)
        self.selectedKeypointsUpdated.emit()
        self.keypointsUpdated.emit()
        self.totalFramesUpdated.emit()
    @QtCore.pyqtSlot()
    def toggle_all_keypoints(self):
        keypoints = self.selected_keypoints_map[self.current_video]
        if len(keypoints) > 0:
            keypoints.clear()
        else:
            keypoints.update(self._glo.video_config.key_points[self.current_video_path])
        self.selectedKeypointsUpdated.emit()
        self.keypointsUpdated.emit()
        self.totalFramesUpdated.emit()

    _curr_video_index: int = -1
    videoIndexUpdated = QtCore.pyqtSignal()
    @QtCore.pyqtProperty(int, notify=videoIndexUpdated)
    def curr_video_index(self):
        return self._curr_video_index
    @curr_video_index.setter
    def curr_video_index(self, new_val):
        self._curr_video_index = new_val
        self._glo.select_file(self.current_video_path)
        self.videoIndexUpdated.emit()
        self.keypointsUpdated.emit()
        self.allFramesMapUpdated.emit()
        self.someFramesMapUpdated.emit()
    @QtCore.pyqtProperty(str)
    def current_video(self):
        try:
            return self.videos_list[self._curr_video_index]["name"]
        except IndexError:
            return ""

    @QtCore.pyqtProperty(str)
    def current_video_path(self):
        try:
            return self.videos_list[self._curr_video_index]["path"]
        except KeyError:
            return ""

    videosUpdated = QtCore.pyqtSignal()
    @QtCore.pyqtProperty(list, notify=videosUpdated)
    def videos_list(self):
        if self._glo.project is None:
            return []
        return [{"name": os.path.basename(file), "path": file} for file in self._glo.project.files]

    keypointsUpdated = QtCore.pyqtSignal()
    @QtCore.pyqtProperty(list, notify=keypointsUpdated)
    def curr_keypoints(self):
        if self._curr_video_index < 0:
            return []
        keypoints = self._glo.video_config.key_points[self.current_video_path]
        active_keypoints = self.curr_active_keypoints
        return sorted([{'frame': frame, 'active': frame in active_keypoints} for frame in keypoints], key=lambda elem: elem['frame'])

    totalFramesUpdated = QtCore.pyqtSignal()
    @QtCore.pyqtProperty(int, notify=totalFramesUpdated)
    def total_frames(self):
        frame_map = self.get_frame_map()
        total = 0
        for frames in frame_map.values():
            total += len(frames)
        return total

    @QtCore.pyqtSlot(int)
    def go_to_frame(self, frame: int):
        self._glo.select_frame(frame)

    _local_model_selection: str = None
    modelNamesUpdated = QtCore.pyqtSignal()
    @QtCore.pyqtProperty(list, notify=modelNamesUpdated)
    def model_names(self):
        if self._glo.model_config is None:
            return []
        return list(self._glo.model_config.models.keys())
    @QtCore.pyqtSlot(str)
    def set_model_name(self, name: str):
        self._local_model_selection = name
        self.set_model_version("Latest")
        self.modelSelectionUpdated.emit()

    modelSelectionUpdated = QtCore.pyqtSignal()
    @QtCore.pyqtProperty(str, notify=modelSelectionUpdated)
    def model_name(self):
        if self._local_model_selection is None:
            return ""
        return self._local_model_selection
    @QtCore.pyqtProperty(list, notify=modelSelectionUpdated)
    def model_versions(self):
        if self._local_model_selection is None or self._glo.model_config is None:
            return ["Latest"]
        return ["Latest"] + self._glo.model_config.models[self._local_model_selection]

    modelVersionUpdated = QtCore.pyqtSignal()
    @QtCore.pyqtProperty(str, notify=modelVersionUpdated)
    def model_version(self):
        if self._glo.model_config is None:
            return "Latest"
        version = self._glo.model_config.model_version
        if version is None:
            version = "Latest"
        return str(version)
    @QtCore.pyqtSlot(str)
    def set_model_version(self, version: str):
        if version == "Latest":
            version = None
        else:
            try:
                version = int(version)
            except ValueError:
                version = None
        self._glo.model_config.set_model(self._local_model_selection, version)

    def __init__(self, engine: QtQuick.QQuickView):
        self.all_frames_map = {}
        self.some_frames_map = {}
        self.selected_keypoints_map = {}
        super().__init__(engine, "uis/LandmarkDetectionView.qml",
                         "Detect Landmarks")

        def on_config_change(conf: str):
            if conf == "model_config":
                self._local_model_selection = self._glo.model_config.model_name
                self.modelNamesUpdated.emit()
                self.modelSelectionUpdated.emit()
                self.modelVersionUpdated.emit()
        self._glo.onConfigChange.connect(on_config_change)
        self.videosUpdated.connect(self.refresh_maps)

    def show(self):
        super().show()
        self.videosUpdated.emit()
        self._local_model_selection = self._glo.model_config.model_name
        self.modelNamesUpdated.emit()
        self.modelSelectionUpdated.emit()
        self.modelVersionUpdated.emit()
        self.keypointsUpdated.emit()
        self._glo.video_config.keyPointsUpdated.connect(self.keypointsUpdated.emit)

    def refresh_maps(self):
        for video in self.videos_list:
            if video["name"] not in self.all_frames_map:
                self.all_frames_map[video["name"]] = False
            if video["name"] not in self.some_frames_map:
                self.some_frames_map[video["name"]] = ""
            if video["name"] not in self.selected_keypoints_map:
                self.selected_keypoints_map[video["name"]] = set()

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

    def get_frame_map(self):
        frame_map = {}
        for video in self.videos_list:
            all_frames = set()
            name, path = video["name"], video["path"]
            if self.all_frames_map[name]:
                cap = cv2.VideoCapture(path)
                if cap is not None and cap.isOpened():
                    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    all_frames.update(range(frame_count))
                    frame_map[path] = all_frames
                continue
            if len(self.some_frames_map[name]) > 0:
                all_frames.update(self.frame_desc_to_list(self.some_frames_map[name])[0])
            all_frames.update(
                set([frame for frame in self.selected_keypoints_map[name] if frame in self._glo.video_config.key_points[path]])
            )
            frame_map[path] = all_frames
        return frame_map

    def on_frame_done(self, _num_processed: int, progress: float):
        self.progressChanged.emit(progress)

    def on_video_done(self, video: str, landmarks: pd.DataFrame):
        """Updates landmarks with the new ones"""
        add_landmarks = False
        landmarks.set_index("Frame_number")
        landmark_file = self._glo.project.files_map[video]
        if landmark_file is None:
            add_landmarks = True
            old_landmarks = pd.DataFrame()
            landmark_file = os.path.splitext(video)[0] + ".csv"
        else:
            old_landmarks = pd.read_csv(landmark_file)
        if old_landmarks.empty:
            old_landmarks["Frame_number"] = []
        old_landmarks.set_index("Frame_number")
        new_frame: pd.DataFrame = pd.concat([landmarks, old_landmarks]).drop_duplicates(["Frame_number"], keep='last').sort_values(by=['Frame_number'], ascending=False)
        new_frame.to_csv(landmark_file)
        if add_landmarks:
            self._glo.project.set_landmarks(video, landmark_file)
        if self._glo.curr_file == video:
            self._glo.select_file(video)

    def on_inference_finished(self):
        self.send_message("Finished finding landmarks")
        self.detecting = False
        self.progressChanged.emit(1)
        self.all_frames_map = {}
        self.some_frames_map = {}
        self.selected_keypoints_map = {}
        self.videosUpdated.emit()
        self.videoIndexUpdated.emit()
        self.keypointsUpdated.emit()
        self.allFramesMapUpdated.emit()
        self.someFramesMapUpdated.emit()
        self.totalFramesUpdated.emit()

    @QtCore.pyqtSlot()
    def detect(self):
        self.detecting = True
        try:
            self.detector = LandmarkDetector(self.get_frame_map())
            has_errored, errors = self.detector.has_errored()
            if has_errored:
                message = f"Errors ({len(errors)}):\n" + "\n".join(errors)
                self.send_message(message)
                self.detecting = False
            else:
                self.detector.frame_done_signal.connect(self.on_frame_done)
                self.detector.landmarks_complete_signal.connect(self.on_video_done)
                self.detector.inference_done_signal.connect(self.on_inference_finished)
                self.detector.start()
        except Exception as err:
            self.detecting = False
            print("Failed to detect: ", err)

