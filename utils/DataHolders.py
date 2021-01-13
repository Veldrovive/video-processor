from enum import Enum
from PyQt5 import QtCore
from pathlib import Path
import shutil
from dataclasses import dataclass, field
import threading
from abc import ABCMeta, abstractmethod
import pandas as pd
from typing import Dict, Tuple, Optional, List, Union, Set
import logging
import numpy as np
import cv2
import json
from scipy.spatial.distance import cdist
import os
import uuid

# Section: Enumeration Definitions
class Position(Enum):
    """
    This allows me to more easily keep track of where the buffer should be
    reading from.
    """
    BEG = 0
    END = 1

class InteractionMode(Enum):
    """
    I use this to keep track of whether the selection should only be of one
    point or if it should get the centroid
    """
    POINT = 1
    AREA = 2

class MouseMode(Enum):
    """
    I use this to decide whether the scene should drag or select when a user
    clicks and holds their mouse
    """
    PAN = 1
    DRAG = 2

class MetricType(Enum):
    """
    This stores whether a metric is of distance or area
    """
    LENGTH = 1
    AREA = 2


# Could change these to use multiple inheritance to reduce amount of code
class ConfigDict(dict):
    """
    This dictionary alerts the program whenever it has been changed in order to
    allow for dynamic saving of configuration files
    """
    change_callback = None

    def __init__(self, setup_dict: Dict = None, change_callback=None):
        self.change_callback = change_callback
        super(ConfigDict, self).__init__(setup_dict)
        self.update_types()

    def _get_config_version(self, value: Union[dict, list]):
        """Returns the config version of the input"""
        if isinstance(value, dict):
            value = ConfigDict(value, self.change_callback)
        if isinstance(value, list):
            value = ConfigList(value, self.change_callback)
        return value

    def __setitem__(self, key, value):
        """Catches when a value changes and calls the change callback"""
        # If a list or dict or list is inserted into this, use the config one
        value = self._get_config_version(value)
        super(ConfigDict, self).__setitem__(key, value)
        if self.change_callback is not None:
            self.change_callback()

    def update_types(self):
        """
        Transforms all contained lists and dicts into their config versions
        """
        for (key, val) in self.items():
            if isinstance(val, dict):
                self[key] = ConfigDict(val, self.change_callback)
                continue
            if isinstance(val, list):
                self[key] = ConfigList(val, self.change_callback)
                continue

class ConfigList(list):
    """
    This list alerts the program whenever it has been changed in order to
    allow for dynamic saving of configuration files
    """
    change_callback = None

    def __init__(self, setup_list: List = None, change_callback=None):
        self.change_callback = change_callback
        super(ConfigList, self).__init__(setup_list)
        self.update_types()

    def _get_config_version(self, value: Union[dict, list]):
        """Returns the config version of the input"""
        if isinstance(value, dict):
            value = ConfigDict(value, self.change_callback)
        if isinstance(value, list):
            value = ConfigList(value, self.change_callback)
        return value

    def __setitem__(self, key, value):
        """Catches when a value changes and calls the change callback"""
        # If a list or dict or list is inserted into this, use the config one
        value = self._get_config_version(value)
        super(ConfigList, self).__setitem__(key, value)
        if self.change_callback is not None:
            self.change_callback()

    def append(self, object) -> None:
        """Overrides the append method to get callback capability"""
        object = self._get_config_version(object)
        super(ConfigList, self).append(object)
        if self.change_callback is not None:
            self.change_callback()

    def remove(self, object) -> None:
        """Overrides the remove method to get callback capability"""
        object = self._get_config_version(object)
        super(ConfigList, self).remove(object)
        if self.change_callback is not None:
            self.change_callback()

    def update_types(self):
        """
        Transforms all contained lists and dicts into their config versions
        """
        for i, val in enumerate(self):
            if isinstance(val, dict):
                self[i] = ConfigDict(val, self.change_callback)
                continue
            if isinstance(val, list):
                self[i] = ConfigList(val, self.change_callback)
                continue

class Config(QtCore.QObject):
    __metaclass__ = ABCMeta

    @property
    def saved_props(self):
        raise NotImplementedError("Saved Props must be defined by the user")
    saved_props: List[str]

    save_loc: str = None

    onChangeSignal = QtCore.pyqtSignal()

    def __init__(self, load_file: str = None, glo=None):
        super(Config, self).__init__()
        self._glo = glo
        self.set_defaults()
        if load_file is not None:
            self.load(load_file)
            self.set_save_loc(load_file)

    @abstractmethod
    def set_defaults(self):
        """
        This method must always be overwritten and set the default configuration
        as well as the saved_props attribute
        """
        pass

    def set_save_loc(self, file: str):
        """
        Sets the path that the visual config will save to
        :param file: The file path to save to
        :return:
        """
        self.save_loc = os.path.abspath(file)

    def load(self, file: str) -> bool:
        """
        Updates the visual config with info from the disk
        :param file: The path to the config json
        :return: The success
        """
        def convert_types(obj):
            """
            Used as an object hook in load to convert all dicts into configdicts
            :param obj:
            """
            if isinstance(obj, dict):
                return ConfigDict(obj, self.on_change)
            if isinstance(obj, list):
                return ConfigList(obj, self.on_change)
            return obj
        if not os.path.isfile(file):
            raise ValueError("The specified file path")
        try:
            recalled_data = json.load(open(file, "r"), object_hook=convert_types)
            self.__dict__.update(recalled_data)
            return True
        except (FileNotFoundError, AttributeError):
            # Then either the file doesnt exist or the json is not correct
            return False

    def __setattr__(self, key, value):
        """Catches changes made to the base dict"""
        if isinstance(value, dict):
            value = ConfigDict(value, self.on_change)
        if isinstance(value, list):
            value = ConfigList(value, self.on_change)
        super(Config, self).__setattr__(key, value)
        self.on_change()

    def on_change(self) -> bool:
        """
        Saves the visual config to disk
        :return: The success value
        """
        self.onChangeSignal.emit()
        if self.save_loc is not None:
            t = threading.Thread(name='save_thread', target=self.thread_save)
            t.start()
            return True
        return False

    def thread_save(self):
        dump_data = {key: value for (key, value) in self.__dict__.items() if
                     key in self.saved_props}
        json.dump(dump_data, open(self.save_loc, "w"), indent=4)

class VisualConfig(Config):
    """
    This class handles passing information
    """
    # Current video specific
    scale_factor: float = 1
    min_dim: int = 1920
    resolution: Tuple[int, int]
    fps: float
    video_length: int = -1

    selected_landmarks: Set[int] = set()
    excluded_landmarks: Set[int] = set()

    # Saved data
    saved_props = ["playback_speed", "show", "color_overrides", "highlight_color", "group_colors"]

    playback_speed: float
    show: Dict[str, bool]
    color_overrides: Dict[int, Tuple[int, int, int]]
    highlight_color: Tuple[int, int, int]
    group_colors: Dict[str, Tuple[int, int, int]]

    def set_defaults(self):
        self.playback_speed = 1
        self.show = {"land": True, "bound": False, "metrics": True}
        self.color_overrides = {}
        self.highlight_color = (247, 222, 59)
        self.group_colors = {
            "face": (224, 27, 70),
            "lower_eye": (30, 27, 224),
            "upper_mouth": (39, 193, 232),
            "lower_mouth": (232, 39, 181)
        }

    def get_video_specs(self, cap: cv2.VideoCapture):
        """Gets the scale factor, resolution, and fps for the current video"""
        self.fps = int(cap.get(5))
        self.resolution = (int(cap.get(3)), int(cap.get(4)))
        self.scale_factor = max(self.min_dim / max(self.resolution), 1)
        self.video_length = cap.get(cv2.CAP_PROP_FRAME_COUNT)

    def toggle_landmarks(self):
        """Toggles the landmark visibility"""
        self.show["land"] = not self.show["land"]
        return self.show["land"]

    def toggle_metrics(self):
        """Toggles the metric visibility"""
        self.show["metrics"] = not self.show["metrics"]
        return self.show["metrics"]

    def toggle_bounding(self):
        """Toggles the bounding box visibility"""
        self.show["bound"] = not self.show["bound"]
        return self.show["bound"]

    def set_scale(self, factor: float):
        """Sets the current scale factor"""
        self.scale_factor = factor

    def exclude(self, landmarks: List[int]) -> bool:
        """
        Adds the landmarks to the excluded list
        :param landmarks: A list of landmark indexes
        :return: Whether the excluded list updated correctly
        """
        try:
            self.excluded_landmarks.update(landmarks)
            return True
        except TypeError:
            return False

    def include(self, landmarks: List[int] = None) -> bool:
        """
        Removes the landmarks from the excluded list
        :param landmarks: The landmarks to remove. If None, reset the exclusions
        :return:
        """
        if landmarks is None:
            self.excluded_landmarks = set()
        else:
            self.excluded_landmarks.difference_update(landmarks)
        return True

    # Maybe add the ability to change the color override

class VideoConfig(Config):
    """
    This config stores information necessary for each video such as key points
    """
    curr_videos: Set[str] = set()  # All videos currently in the config
    video_dicts = [("key_points", list), ("curr_positions", lambda: 0)]

    saved_props = ["key_points", "curr_positions", "curr_video"]
    key_points: Dict[str, List[int]]  # Key points for each video
    curr_positions: Dict[str, int]  # Saves the last position in the video
    curr_video: str = None

    setPosition = QtCore.pyqtSignal(int)  # Emitted to change the video frame
    remarkFrames = QtCore.pyqtSignal()  # Emitted to remark the current frames
    keyPointsUpdated = QtCore.pyqtSignal()  # Emitted when a keypoint is added

    def load(self, file: str) -> bool:
        if not super(VideoConfig, self).load(file):
            return False
        # If some config was loaded, then update all config dicts
        for (dict_name, dict_type) in [dict_n for dict_n in self.video_dicts if dict_n[0] in self.__dict__]:
            for video in self.__dict__[dict_name].keys():
                self.setup_video(video)
        return True

    def set_defaults(self):
        self.key_points = {}
        self.curr_positions = {}

    def set_curr_video(self, file: str):
        if file not in self.curr_videos:
            self.setup_video(file)
        self.curr_video = file

    def get_curr_video(self) -> Union[None, str]:
        """Returns the current video file"""
        return self.curr_video

    def setup_video(self, file: str):
        """
        Adds a new video to the config dicts
        """
        self.curr_videos.add(file)
        for (dict_name, member_type) in self.video_dicts:
            curr_dict = self.__dict__[dict_name]
            if file not in curr_dict:
                curr_dict[file] = member_type()

    def get_keypoints(self) -> List[int]:
        """Gets the saved keypoints for the current video"""
        if self.curr_video is None:
            return []
        return self.key_points[self.curr_video]

    def get_next_keypoint(self, frame: int) -> int:
        """
        Gets the keypoint after the current frame
        :param frame: The current frame
        :return: The frame of the next keypoint or zero if there is no next
        """
        if self.curr_video is None:
            return 0
        keypoints = self.get_keypoints()
        future_keypoints = [f for f in keypoints if f > frame]
        future_keypoints.sort()
        if len(future_keypoints) < 1:
            return 0
        else:
            return future_keypoints[0]

    def get_previous_keypoint(self, frame: int) -> int:
        """
        Gets the keypoint after before current frame
        :param frame: The current frame
        :return: The last keypoint or zero if there is no previous
        """
        if self.curr_video is None:
            return 0
        keypoints = self.get_keypoints()
        past_keypoints = [f for f in keypoints if f < frame]
        past_keypoints.sort()
        if len(past_keypoints) < 1:
            return 0
        else:
            return past_keypoints[-1]

    def add_keypoint(self, frame: int) -> bool:
        """Adds the new keypoint to the video.
        :return: Whether the keypoint was actually added
        """
        if self.curr_video is None:
            return False
        if frame in self.key_points[self.curr_video]:
            return False
        self.key_points[self.curr_video].append(frame)
        self.keyPointsUpdated.emit()
        return True

    def remove_keypoint(self, frame: int) -> bool:
        """Removes the keypoint from the video
        :return: Whether a keypoint was removed
        """
        if self.curr_video is None or frame not in self.key_points[self.curr_video]:
            return False
        self.key_points[self.curr_video].remove(frame)
        self.keyPointsUpdated.emit()
        return True

    def toggle_keypoint(self, frame: int) -> Union[None, bool]:
        """
        Adds a keypoint if it doesnt exist or remove it if it does
        :param frame: The frame to toggle the keypoint on
        :return: None if the function failed. True if a keypoint was added.
            False is one was removed
        """
        if self.curr_video is None:
            return
        if frame in self.key_points[self.curr_video]:
            self.remove_keypoint(frame)
            return False
        else:
            self.add_keypoint(frame)
            return True

    def reset_keypoints(self) -> bool:
        """Resets the keypoints for the current video
        :return: Whether the reset was successful
        """
        if self.curr_video is None:
            return False
        self.key_points[self.curr_video] = []
        return True

    def seek_to(self, frame: int):
        """Emits a setPosition event to update the frame buffer"""
        self.setPosition.emit(frame)

    def refresh_frames(self):
        self.remarkFrames.emit()

    def set_position(self, frame: int):
        """Sets the last position for the current video"""
        self.curr_positions[self.curr_video] = frame

    def get_position(self) -> int:
        """Gets the saved position of the current video"""
        return self.curr_positions[self.curr_video]

class ModelConfig(Config):
    # Basic config
    model_name: str = "Default"  # Stores the model the user has currently selected
    model_version: Optional[int] = None  # Which version of the model to use. None is latest.

    # Retraining Config
    learning_rate: Optional[float] = None  # None means find a default value
    batch_size: Optional[int] = None  # None means find a default value
    epochs: int = 10

    proportion_val: float = 0.1
    proportion_test: float = 0.1
    @property
    def proportion_train(self) -> float:
        return 1-self.proportion_val-self.proportion_test
    
    # To Save
    saved_props = ["model_name", "model_version", "learning_rate", "batch_size", "epochs", "proportion_val", "proportion_test"]

    # Signals
    modelChanged = QtCore.pyqtSignal()  # When the model name or version changes
    lrChanged = QtCore.pyqtSignal()
    batchSizeChanged = QtCore.pyqtSignal()
    epochsChanged = QtCore.pyqtSignal()
    splitsChanged = QtCore.pyqtSignal()  # When prop_val or prop_test changes

    def set_defaults(self):
        self.model_name = "Default"
        self.model_version = None
        self.learning_rate = None
        self.batch_size = None
        self.proportion_val = 0.1
        self.proportion_test = 0.1

    @property
    def models(self) -> Dict[str, List[int]]:
        models = {"Default": []}
        models_dir = self._glo.project.fan_dir
        model_dirs = [f.name for f in os.scandir(models_dir) if f.is_dir()]
        for model_name in model_dirs:
            model_dir = os.path.join(models_dir, model_name)
            def get_version(name):
                try:
                    return int(os.path.splitext(name)[0][len(file_base):])
                except ValueError:
                    return -1
            file_base = f"{model_name}_model_v"
            versions = set([get_version(file) for file in os.listdir(model_dir) if file_base in file])
            versions.discard(-1)
            models[model_name] = sorted(versions)
        return models

    def set_model(self, name: str, version: Optional[int] = None) -> Optional[str]:  # Returns the error if there is one
        models = self.models
        if name not in models:
            return "No model by that name exists"
        if version is not None and version not in models[name]:
            return "No model of that version exists"
        self.model_name = name
        self.model_version = version
        self.modelChanged.emit()

    @property
    def model_path(self):
        if self.model_name == "Default":
            return self._glo.project.default_fan_path
        version = self.models[self.model_name][-1] if self.model_version is None else self.model_version
        return os.path.join(self._glo.project.fan_dir, self.model_name, f"{self.model_name}_model_v{version}.ptl")

    @property
    def s3fd_path(self):
        return self._glo.project.default_s3fd_path

    def set_lr(self, lr: Optional[float] = None):
        self.learning_rate = lr
        self.lrChanged.emit()

    def set_batch_size(self, batch_size: Optional[int] = None):
        self.batch_size = batch_size
        self.batchSizeChanged.emit()

    def set_epochs(self, epochs: int):
        self.epochs = epochs
        self.epochsChanged.emit()

    def set_proportions(self, prop_val: Optional[float] = None, prop_test: Optional[float] = None):
        if prop_val:
            self.proportion_val = min(prop_val, 1-prop_test)
        if prop_test:
            self.proportion_test = min(prop_test, 1-prop_val)
        self.splitsChanged.emit()




class BoundingBox:
    locations: Dict[int, Tuple[Tuple[int, int], Tuple[int, int]]] = None

    def __init__(self, landmarks: pd.DataFrame):
        self.locations = {}
        try:
            locations = landmarks[[
                "Frame_number", "bbox_top_x","bbox_top_y", "bbox_bottom_x", "bbox_bottom_y"
            ]].values
        except KeyError:
            return
        for frame in locations:
            self.locations[int(frame[0])] = (
                (frame[1], frame[2]),
                (frame[3], frame[4])
            )

    def get_location(self, frame_num: int) -> Optional[Tuple[Tuple[float, float], Tuple[float, float]]]:
        if frame_num in self.locations:
            return self.locations[frame_num]
        return None

@dataclass
class Landmark:
    index: int = -1
    group: str = ""
    locations: Dict[int, Tuple[float, float]] = field(default_factory=dict)

    def set_location(self, frame: int, pos: Tuple[float, float]) -> bool:
        try:
            self.locations[frame] = pos
            return True
        except IndexError:
            return False

    def get_index(self) -> int:
        return self.index

    def get_group(self) -> str:
        return self.group

    def get_location(self, frame_num: int) -> Tuple[float, float]:
        try:
            return self.locations[frame_num]
        except KeyError:
            return -1, -1

    def get_columns(self) -> Dict[str, List[float]]:
        return {
            f"landmark_{self.index}_x": [self.locations[frame][0] for frame in self.locations],
            f"landmark_{self.index}_y": [self.locations[frame][1] for frame in self.locations]
        }

    def get_frames(self):
        """
        Returns a list of all frame on which the landmark is defined
        """
        return sorted(self.locations.keys())

class Landmarks:
    _landmarks_frame: pd.DataFrame = None
    _landmarks_file: str = None
    _landmarks: List[Optional[Landmark]] = []
    _bounding_boxes: List[Optional[BoundingBox]]
    _n_landmarks: int = 0

    _has_bbox: bool = False
    _has_landmarks = False

    def __init__(self, landmarks: pd.DataFrame, file: str = None):
        self._landmarks_frame = landmarks
        self._landmarks_file = file
        self._n_landmarks = int(len([col for col in landmarks.columns if "landmark_" in col]) / 2)
        self._landmarks = []
        self.populate_landmarks()
        self.populate_bounding_boxes()

    def calculate_landmark_ranges(self) -> List[Tuple[int, int]]:
        ranges = []
        start = None
        last_landmark_frame = None
        for frame in sorted(self.get_frames()):
            if last_landmark_frame is None or frame > last_landmark_frame + 1:
                if start is not None:
                    ranges.append((start, last_landmark_frame))
                start = frame
            last_landmark_frame = frame
        if start is not None and last_landmark_frame is not None:
            ranges.append((start, last_landmark_frame))
        return ranges

    def has_bbox(self):
        return self._has_bbox

    def has_landmarks(self):
        return self._has_landmarks

    def populate_landmarks(self) -> bool:
        for i in range(self._n_landmarks):
            try:
                locations = self._landmarks_frame[[
                    "Frame_number", f"landmark_{i}_x", f"landmark_{i}_y"
                ]].values
                frame_locations = {}
                for frame in locations:
                    frame_locations[int(frame[0])] = tuple(frame[1:])
                landmark = Landmark(i, "face", frame_locations)
            except KeyError:
                landmark = None
            self._landmarks.append(landmark)
        self._has_landmarks = len([l for l in self._landmarks if l is None]) == 0
        return True

    def populate_bounding_boxes(self) -> bool:
        bbox = BoundingBox(self._landmarks_frame)
        if bbox.locations is not None:
            self._bounding_boxes = [bbox]
            self._has_bbox = True
        else:
            self._bounding_boxes = [None]
            self._has_bbox = False
        return True

    def get_bound_box_locs(self, frame: int) -> List[Tuple[Tuple[float, float], Tuple[float, float]]]:
        if not self._has_bbox:
            return []
        box_locs = []
        for bounding_box in self._bounding_boxes:
            loc = bounding_box.get_location(frame)
            if loc is not None:
                box_locs.append(bounding_box.get_location(frame))
        return box_locs

    def set_group(self, landmarks: Union[int, List[int]], group: str = "face") -> bool:
        for landmark in landmarks:
            try:
                self._landmarks[landmark].group = group
            except (IndexError, AttributeError):
                pass
        return True

    def set_location(self, frame: int, landmark_index: int, pos: Tuple[float, float]) -> bool:
        try:
            return self._landmarks[landmark_index].set_location(frame, pos)
        except IndexError:
            return False

    def get_landmark_locations(self, frame: int, landmarks: Optional[Union[int, List[int]]]=None, exclude_none=True) -> List[Optional[Tuple[float, float]]]:
        if not self._has_landmarks:
            return []
        if isinstance(landmarks, int):
            landmarks = [landmarks]
        if landmarks is None:
            landmarks = range(self._n_landmarks)
        locs = [self._landmarks[i].get_location(frame) for i in landmarks if self._landmarks[i] is not None]
        if exclude_none:
            locs = [loc for loc in locs if loc is not None]
        return locs

    def get_landmarks(self, frame: int, landmarks: Optional[Union[int, List[int]]]=None) -> List[Optional[Tuple[Tuple[float, float], str, int]]]:
        if not self._has_landmarks:
            return []
        if isinstance(landmarks, int):
            landmarks = [landmarks]
        if landmarks is None:
            landmarks = range(self._n_landmarks)
        res = []
        for index in landmarks:
            landmark = self._landmarks[index]
            if landmark is not None:
                res.append((
                    landmark.get_location(frame),
                    landmark.get_group(),
                    landmark.get_index()
                ))
        return res

    def get_centroid(self, frame: int, landmarks: List[int]) -> Tuple[float, float]:
        if not self._has_landmarks:
            return (-1, -1)
        locations = [location for location in self.get_landmark_locations(frame, landmarks) if location is not None]
        return tuple(np.sum(locations, axis=0)/len(locations))

    def get_nearest_point(self, frame: int, pos: Tuple[float, float], threshold: int=6) -> Optional[int]:
        if not self._has_landmarks:
            return None
        locations = self.get_landmark_locations(frame)
        distance = cdist(np.array(locations), np.array([pos]))[:, 0]
        min_index = np.argmin(distance)
        min_dist = distance[min_index]
        if min_dist < threshold or threshold == -1:
            return self._landmarks[min_index].index
        else:
            return None

    def get_point_area(self, frame: int, x_max: float, x_min: float, y_max: float, y_min: float) -> List[int]:
        if not self._has_landmarks:
            return []
        locations = self.get_landmark_locations(frame)
        selected_landmarks = []
        for index, location in enumerate(locations):
            in_x = x_max >= location[0] >= x_min
            in_y = y_max >= location[1] >= y_min
            if in_x and in_y:
                selected_landmarks.append(index)
        return selected_landmarks

    def get_frames(self) -> List[int]:
        """
        :return: A list of all frames that have landmarks calculated
        """
        try:
            return self._landmarks_frame["Frame_number"].to_list()
        except AttributeError:
            return []

    def get_dataframe(self) -> pd.DataFrame:
        landmark_frame = self._landmarks_frame.copy()
        for landmark in self._landmarks:
            if landmark is None:
                continue
            columns = landmark.get_columns()
            for column in columns:
                landmark_frame[column] = columns[column]
        return landmark_frame

    @staticmethod
    def load(file: str):
        """
        Tries to create a landmark object out of the given file
        :param file: The absolute path to the landmark file
        :return: The corresponding Landmark object
        """
        try:
            # landmarks = pd.read_csv(file, index_col=0)
            landmarks = pd.read_csv(file)
        except Exception as e:
            # TODO: make this exception handling more specific
            return None
        return Landmarks(landmarks)

    def save(self, file: str = None, save_copy: bool = True) -> bool:
        """
        Handles actions that should be taken when the user wants to save their edits
        """
        frame = self.get_dataframe()
        s_file = file if self._landmarks_file is None else self._landmarks_file
        if s_file is None:
            return False
        if save_copy and os.path.exists(s_file):
            name, ext = os.path.splitext(s_file)
            orig_file = name + "_orig" + ext
            if os.path.exists(orig_file):
                os.remove(orig_file)
            os.rename(s_file, orig_file)
        frame.to_csv(s_file, index=False)


class Project:
    name: str
    id: str
    files_map: Dict[str, Union[None, str]]  # {abs_vid_path: abs_landmark_path}
    save_loc: str = None  # The save location

    _copy_files: bool = True
    @property
    def copy_files(self):
        """Gets whether files should be copied into the data directory"""
        return self._copy_files

    @copy_files.setter
    def copy_files(self, val: bool):
        """Sets whether files should be copied into the data directory"""
        if not isinstance(val, bool):
            raise ValueError("Copy files value must be a boolean")
        self._copy_files = val
        if self.save_loc is not None and self._copy_files:
            self.copy_videos_to_project()

    @property
    def files(self):
        """
        Gets a list of absolute paths to video files
        """
        return list(self.files_map.keys())

    @property
    def file_names(self):
        """
        Gets a list of all file names
        """
        return [os.path.basename(file) for file in self.files]

    @property
    def landmarks(self):
        """
        Gets a list of absolute paths to landmark files
        """
        return list(self.files_map.values())

    @property
    def frames_dir(self) -> Optional[str]:
        if self.save_loc is None:
            return None
        return os.path.join(self.save_loc, "retraining_data", "frames")

    @property
    def landmarks_dir(self) -> Optional[str]:
        if self.save_loc is None:
            return None
        return os.path.join(self.save_loc, "retraining_data", "landmarks")

    @property
    def data_dir(self) -> Optional[str]:
        if self.save_loc is None:
            return None
        print("Data dir: ", os.path.join(self.save_loc, "data"))
        return os.path.join(self.save_loc, "data")

    @property
    def metrics_dir(self) -> Optional[str]:
        if self.save_loc is None:
            return None
        return os.path.join(self.save_loc, "metric_output")

    @property
    def config_dir(self) -> Optional[str]:
        if self.save_loc is None:
            return None
        return os.path.join(self.save_loc, "config")

    @property
    def models_dir(self) -> Optional[str]:
        if self.save_loc is None:
            return None
        return os.path.join(self.save_loc, "models")

    @property
    def fan_dir(self) -> Optional[str]:
        if self.models_dir is None:
            return None
        return os.path.join(self.models_dir, "FAN")

    default_fan_path: str = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../landmark_detection/models/default_fan.tar"))

    @property
    def s3fd_dir(self) -> Optional[str]:
        if self.models_dir is None:
            return None
        return os.path.join(self.models_dir, "s3fd")

    default_s3fd_path: str = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../landmark_detection/models/s3fd.pth"))

    def __init__(self, name: str, save_loc: Optional[str] = None, p_id: Optional[str] = None):
        # if len(name) < 1:
        #     raise AttributeError(f"Project must have a name")
        self.name = name
        self.id = uuid.uuid4().hex if p_id is None else p_id
        self.files_map = {}
        if save_loc is not None:
            self.set_save_loc(save_loc)

    def add_FAN(self, file_path: str):
        """Moves a new fan model into the project"""
        shutil.copy(file_path, self.get_new_FAN_path())

    def get_new_FAN_path(self) -> str:
        """Gets the path that the next FAN model will save to"""
        try:
            max_index = max([int(file.split("_")[1]) for file in os.listdir(self.fan_dir)])
        except ValueError:
            max_index = 0
        return os.path.join(self.fan_dir, f"FAN_{max_index+1}_.pth.tar")

    def get_FAN_path(self) -> Optional[str]:
        """Gets the most recent FAN model"""
        def get_num_model(model_path):
            try:
                return int(model_path.split("_")[1])
            except (ValueError, IndexError) as e:
                return -1

        try:
            max_index = max([get_num_model(file) for file in os.listdir(self.fan_dir)])
            if max_index < 0:
                return None
            return os.path.join(self.fan_dir, f"FAN_{max_index}_.pth.tar")
        except ValueError:
            return None

    def add_s3fd(self, file_path: str):
        """Moves a new fan model into the project"""
        try:
            max_index = max([int(file.split("_")[1]) for file in os.listdir(self.s3fd_dir)])
        except ValueError:
            max_index = 0
        shutil.copy(file_path, os.path.join(self.s3fd_dir, f"s3fd_{max_index}_.pth"))

    def get_s3fd_path(self) -> Optional[str]:
        """Gets the most recent s3fd model"""
        try:
            max_index = max([int(file.split("_")[1]) for file in os.listdir(self.s3fd_dir)])
            return os.path.join(self.s3fd_dir, f"s3fd_{max_index}_.pth")
        except ValueError:
            return None

    def set_save_loc(self, loc: str):
        """
        Sets the save location and created required directories
        :param loc: The new location for the project to be saved to
        """
        logging.info(f"Setting save directory for {self.name} to: {loc}")
        self.save_loc = os.path.abspath(loc)
        os.makedirs(self.frames_dir, exist_ok=True)
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.landmarks_dir, exist_ok=True)
        os.makedirs(self.config_dir, exist_ok=True)
        os.makedirs(self.fan_dir, exist_ok=True)
        os.makedirs(self.s3fd_dir, exist_ok=True)
        os.makedirs(self.metrics_dir, exist_ok=True)

    def add_file(self, path: str, copy_files: bool = True):
        """
        Adds a new file
        :param path: A path to the new file
        :param copy_files: Whether to copy files to the project folder
        """
        abs_path = os.path.abspath(path)
        if not os.path.isfile(abs_path):
            raise ValueError("Video file does not exist")
        if abs_path not in self.files_map:
            self.files_map[abs_path] = None
        if self.copy_files and copy_files:
            self.copy_videos_to_project()

    def add_model(self, path: str):
        """
        Copies a model into the project
        :param path: A path to the model
        """
        abs_path = os.path.abspath(path)
        if not os.path.isfile(abs_path):
            raise ValueError("Model file does not exist")
        model_name = os.path.splitext(os.path.basename(abs_path))[0]
        model_path = os.path.join(self.fan_dir, model_name)
        os.makedirs(model_path, exist_ok=True)
        shutil.copy(abs_path, os.path.join(model_path, f"{model_name}_model_v1.ptl"))

    def set_landmarks(self, vid_path: str, l_path: Union[str, None]):
        """
        Sets the landmark file in the map
        :param vid_path: The path to the video
        :param l_path: The path to the landmark file
        """
        print("Setting landmarks for",vid_path,"equal to", l_path)
        abs_vid_path = os.path.abspath(vid_path)
        if abs_vid_path not in self.files_map:
            return False
        if l_path is None:
            self.files_map[abs_vid_path] = None
            return True
        abs_l_path = os.path.abspath(l_path)
        if not os.path.isfile(abs_l_path):
            return False
        self.files_map[abs_vid_path] = abs_l_path
        if self.copy_files:
            self.copy_videos_to_project()
        self.save()
        return True

    def remove_video(self, vid_path, clean_disk: bool = False):
        """
        Removes a video from the landmark map
        :param vid_path: A path to the video
        :param clean_disk: Whether to remove the video from disk if it in the
            data directory
        """
        data_dir = Path(self.data_dir)
        abs_path = os.path.abspath(vid_path)
        path_dir = Path(abs_path)
        landmark_file = self.files_map.pop(abs_path, None)
        if data_dir in path_dir.parents and clean_disk:
            # Then we want to remove the video and landmark from disk
            if abs_path in self.files:
                # Then somehow there is another file in the map with the same path, this is a bug in and of itself
                return
            os.remove(abs_path)
            if landmark_file is not None and landmark_file not in self.landmarks:
                # Then this landmark file is no longer is usage and can be deleted
                os.remove(landmark_file)

    def copy_videos_to_project(self):
        """
        Moves all videos and landmarks that are currently outside of the project
        into the designated data folder
        """
        data_dir = Path(self.data_dir)
        items = list(self.files_map.items())
        for vid_file, l_file in items:
            vid_path, l_path = Path(vid_file), None if l_file is None else Path(l_file)
            if data_dir not in vid_path.parents:
                vid_name = vid_path.name
                new_path = data_dir / vid_name
                shutil.copy(vid_path, new_path)
                old_val = self.files_map.pop(vid_file)
                self.files_map[str(new_path.absolute())] = old_val
                vid_path = new_path
            if l_file is not None and data_dir not in l_path.parents:
                l_name = l_path.name
                new_path = data_dir / l_name
                shutil.copy(l_path, new_path)
                self.files_map[str(vid_path.absolute())] = str(new_path.absolute())

    def get_video_data(self, video_id: Union[int, str]) -> Tuple[Union[cv2.VideoCapture, None], Union[Landmarks, None]]:
        """
        Ease of use function. Gets the video capture and landmarks for a video
        :param video_id:
            If int - Then it is the place in the file list
            If str - Then it is the absolute path to the video
        :return:
            If the video does not exist - (None, None)
            If landmarks do not exist - (VideoCapture, None)
            If landmarks do exist - (VideoCapture, Landmarks)
        """
        if isinstance(video_id, int):
            if video_id >= len(self.files) or video_id < 0:
                return None, None
            video_path = self.files[video_id]
        else:
            if video_id not in self.files_map:
                return None, None
            video_path = video_id
        landmark_path = self.files_map[video_path]
        video = cv2.VideoCapture(video_path)
        if video is None or not video.isOpened():
            return None, None
        landmarks = None
        if landmark_path is not None:
            landmarks = Landmarks.load(landmark_path)
        return video, landmarks

    def __getitem__(self, video_id: Union[int, str]) -> Tuple[Union[cv2.VideoCapture, None], Union[Landmarks, None]]:
        """
        The same as get_video_data but allows for ease of use with:
            project["video.mp4"] or project[1]
        """
        return self.get_video_data(video_id)

    def save(self):
        """
        Saves the project to disk
        """
        if self.save_loc is None:
            return False
        # This contains info about the videos in the project
        logging.info(f"Saving project {self.name} to: {self.save_loc}")
        video_df = pd.DataFrame({
            "abs_video_paths": list(self.files_map.keys()),
            "rel_video_paths": [os.path.relpath(path, self.save_loc) for path in self.files_map.keys()],
            "abs_landmark_paths": list(self.files_map.values()),
            "rel_landmark_paths": [os.path.relpath(path, self.save_loc) if path is not None else None for path in self.files_map.values()]
        })
        meta_df = pd.DataFrame({
            "name": self.name,
            "p_id": self.id
        }, index=["name"])
        video_data_path = os.path.join(self.save_loc, "video_data.csv")
        meta_data_path = os.path.join(self.save_loc, "meta_data.csv")
        if os.path.exists(video_data_path):
            os.remove(video_data_path)
        if os.path.exists(meta_data_path):
            os.remove(meta_data_path)
        video_df.to_csv(video_data_path)
        meta_df.to_csv(meta_data_path)
        return True

    @staticmethod
    def nan_to_none(obj):
        try:
            if np.isnan(obj):
                return None
            return obj
        except TypeError:
            return obj

    @staticmethod
    def load(project_dir: str, fail_to_none: bool = False):
        """
        Get project files back from the disk
        :param project_dir:
        """
        project_dir = os.path.join(project_dir)
        if not os.path.isdir(project_dir):
            if fail_to_none:
                return None
            raise ValueError("Project Directory must be a directory")
        video_data_path = os.path.join(project_dir, "video_data.csv")
        meta_data_path = os.path.join(project_dir, "meta_data.csv")
        if not os.path.isfile(video_data_path) or not os.path.isfile(meta_data_path):
            if fail_to_none:
                return None
            raise ValueError("Could not find video data or meta data file")

        video_df = pd.read_csv(video_data_path)
        meta_df = pd.read_csv(meta_data_path)
        try:
            # Extract project metadata to initialize the new project
            name, p_id = map(lambda elem: elem[0], meta_df.T.values[1:])
            project = Project(name, p_id=p_id, save_loc=project_dir)

            # Gets the paths to video and landmark files from disk
            # We have access to both absolute and relative paths so that files
            # saved inside the project directory can be recalled even if the
            # project has changed directories or even moved to another computer
            paths = zip(*map(list, video_df[["abs_video_paths", "rel_video_paths", "abs_landmark_paths", "rel_landmark_paths"]].values.transpose()))
            for vid_path, rel_vid_path, l_path, rel_l_path in paths:
                # pandas saves None as nan and we translate that back
                l_path, rel_l_path = Project.nan_to_none(l_path), Project.nan_to_none(rel_l_path)

                # We translate the relative paths to
                abs_vid_path = os.path.abspath(os.path.join(project_dir, rel_vid_path))
                abs_l_path = None if rel_l_path is None else os.path.abspath(os.path.join(project_dir, rel_l_path))
                if os.path.isfile(abs_vid_path):
                    # Then we found the video at its relative path
                    vid = abs_vid_path
                elif os.path.isfile(vid_path):
                    # Then we found the video at its absolute path
                    vid = vid_path
                else:
                    # Then the video has been removed from its place on disk
                    continue
                project.add_file(vid)

                if abs_l_path is not None and os.path.isfile(abs_l_path):
                    # Then we found the landmarks at their relative path
                    project.set_landmarks(vid, abs_l_path)
                elif l_path is not None and os.path.isfile(l_path):
                    # Then we found the landmarks at their absolute path
                    project.set_landmarks(vid, l_path)
            return project
        except KeyError:
            # Then something has changed in the CSVs
            if fail_to_none:
                return None
            raise ValueError("Project path has malformed meta CSVs")



@dataclass
class Metric:
    name: str = ""
    type: MetricType = MetricType.LENGTH
    landmarks: List[Union[int, List[int]]] = field(default_factory=list)

default_metrics = [
    Metric(
        "Inter-Canthil Distance",
        MetricType.LENGTH,
        [39, 42]
    ),
    Metric(
        "Left Mouth Area",
        MetricType.AREA,
        list(range(51, 57+1))
    ),
    Metric(
        "Right Mouth Area",
        MetricType.AREA,
        [57, 58, 59, 48, 49, 50, 51]
    ),
    Metric(
        "Left Eyebrow-Nose Distance",
        MetricType.LENGTH,
        [list(range(17, 21+1)), 30]
    ),
    Metric(
        "Right Eyebrow-Nose Distance",
        MetricType.LENGTH,
        [list(range(22, 26 + 1)), 30]
    ),
    Metric(
        "Mouth Vertical Range",
        MetricType.LENGTH,
        [51, 57]
    )
]
class MetricContainer(QtCore.QObject):
    metrics: List[Metric] = []
    working_metric: Metric = None
    save_loc: str = None

    @property
    def metric_names(self):
        return [metric.name for metric in self.metrics]

    metricChangedSignal = QtCore.pyqtSignal()
    workingChangedSignal = QtCore.pyqtSignal()

    def __init__(self, disk_loc: str):
        super().__init__()
        self.set_disk_loc(disk_loc)
        self.metrics = default_metrics
        self.working_metric = Metric()
        self.metricChangedSignal.connect(self.save_to_disk)

    def rename(self, old: str, new: str) -> bool:
        """
        Renames a metric
        :param old: The metric's old name. Used for identification
        :param new: The metric's new name
        :return: Whether a metric was actually changed
        """
        if new in self.metric_names:
            return False
        for metric in self.metrics:
            if metric.name == old:
                metric.name = new
                self.metricChangedSignal.emit()

    def get_all(self):
        """:returns: The list of all metrics"""
        return self.metrics

    def get(self, metric_name: str) -> Optional[Metric]:
        """
        :param metric_name: The name of the requested metric
        :return: The metric object corresponding to the name
        """
        for metric in self.metrics:
            if metric.name == metric_name:
                return metric

    def add(self, metric: Metric) -> bool:
        """
        Adds a new metric to the metric list
        :param metric: A metric object to add
        :return: Whether a metric was actually added
        """
        if metric.name in self.metric_names:
            return False
        self.metrics.append(metric)
        self.metricChangedSignal.emit()

    def append_to_working(self, landmark: Union[int, List[int]]):
        """
        Adds a new point to the working metric
        :param landmark: A id or list of ids corresponding to landmarks
        """
        self.working_metric.landmarks.append(landmark)
        self.workingChangedSignal.emit()

    def remove_from_working(self, landmark: Union[int, List[int]]) -> bool:
        """
        Removes the metric from the working landmarks
        :param landmark: The landmark to remove from the working metric
        :return: Whether the landmark was removed
        """
        try:
            if self.check_working_complete(landmark):
                return True
            self.working_metric.landmarks.remove(landmark)
            self.workingChangedSignal.emit()
            return True
        except ValueError:
            return False

    def set_working_type(self, m_type: MetricType):
        """
        Sets the working metric's type
        :param m_type: The new metric type
        """
        self.working_metric.type = m_type
        self.workingChangedSignal.emit()

    def reset_working(self):
        """
        Sets the working metric to an empty metric
        """
        self.working_metric = Metric()
        self.workingChangedSignal.emit()

    def check_working_complete(self, last_landmark: Union[int, List[int]]) -> bool:
        """Checks if the working metric is complete"""
        working_landmarks = self.working_metric.landmarks
        if len(working_landmarks) > 1 and last_landmark == working_landmarks[0]:
            self.add_working_to_metrics()

    def add_working_to_metrics(self):
        """
        Appends the current working metric to the metrics list
        """
        self.working_metric.name = ','.join([str(landmark) for landmark in self.working_metric.landmarks])
        self.metrics.append(self.working_metric)
        self.reset_working()
        self.metricChangedSignal.emit()

    def set_metric_type(self, metric_name: str, m_type: MetricType):
        """Sets the type of an existing metric"""
        metric = self.get(metric_name)
        if metric is not None:
            metric.type = m_type
            self.metricChangedSignal.emit()

    def remove(self, metric: Union[str, Metric]) -> bool:
        """
        Removes a metric from the metric list
        :param metric: The metric name or the metric object
        :return: Whether a metric was actually removed
        """
        if isinstance(metric, Metric):
            metric = metric.name
        metric_names = [metric.name for metric in self.metrics]
        if metric in metric_names:
            index = metric_names.index(metric)
            self.metrics.pop(index)
            self.metricChangedSignal.emit()
            return True
        return False

    def set_disk_loc(self, loc: str):
        """Sets the location that should be written to and from"""
        self.save_loc = loc

    def save_to_disk(self) -> bool:
        """Writes the metrics to the disk so that they can be recalled"""
        abs_path = os.path.abspath(self.save_loc)
        if os.path.isdir(os.path.dirname(abs_path)):
            dump_data = [{"name": metric.name, "type": metric.type.name, "landmarks": metric.landmarks} for metric in self.metrics]
            json.dump(dump_data, open(abs_path, "w"), indent=4)
            return True
        else:
            return False

    def recall_metrics(self) -> bool:
        """Reads a metric list from disk"""
        abs_path = os.path.abspath(self.save_loc)
        old_metrics = self.metrics
        if not os.path.isfile(abs_path):
            raise ValueError("The specified file path")
        try:
            recalled_data = json.load(open(abs_path, "r"))
            self.metrics = []
            for metric_data in recalled_data:
                metric = Metric(metric_data["name"], MetricType[metric_data["type"]], metric_data["landmarks"])
                self.metrics.append(metric)
            return True
        except (FileNotFoundError, AttributeError):
            # Then either the file doesnt exist or the json is not correct
            self.metrics = old_metrics
            return False


if __name__ == "__main__":
    # print("Testing Dataholders")
    # test_files = [
    #     "/Users/aidandempster/projects/uhn/vidProc/demo/example.mp4",
    #     "/Users/aidandempster/projects/uhn/vidProc/demo/preprocessed.mp4"
    # ]
    # test_project = Project("test_project", save_loc="/Users/aidandempster/projects/uhn/vidProc", files=test_files)
    # test_project.save()

    # rec_project = Project.load("/Users/aidandempster/projects/uhn/vidProc/utils/project_test_project")
    # print("Done testing Dataholders")

    vc = VisualConfig()
    vc.set_save_loc("./test_vc.json")
    print(vc.playback_speed)
    print(type(vc.__dict__))
    vc.playback_speed = 1.2
    print(vc.playback_speed)
    # print(vc.save())
    # rec_vc = vc

    # rec_vc = VisualConfig("./test_vc.json")
    # print(rec_vc.playback_speed)

    # rec_vc.playback_speed = 1
    # rec_vc.show["land"] = True
