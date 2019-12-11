from enum import Enum
from dataclasses import dataclass, field
import pandas as pd
from typing import Dict, Tuple, Optional, List, Union
import numpy as np
from scipy.spatial.distance import cdist

class Position(Enum):
    BEG = 0
    END = 1

class InteractionMode(Enum):
    POINT = 1
    AREA = 2

class MouseMode(Enum):
    PAN = 1
    DRAG = 2

class MetricType(Enum):
    LENGTH = 1
    AREA = 2

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
class Landmarks:
    _landmarks_frame: pd.DataFrame = None
    _landmarks: List[Optional[Landmark]] = []
    _bounding_boxes: List[Optional[BoundingBox]]
    _n_landmarks: int = 0

    _has_bbox: bool = False
    _has_landmarks = False

    def __init__(self, landmarks: pd.DataFrame, n_landmarks: int = 68):
        self._landmarks_frame = landmarks
        self._n_landmarks = n_landmarks
        self._landmarks = []
        self.populate_landmarks()
        self.populate_bounding_boxes()

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

    def get_landmark_locations(self, frame: int, landmarks: Optional[Union[int, List[int]]]=None, exlude_none=True) -> List[Optional[Tuple[float, float]]]:
        if not self._has_landmarks:
            return []
        if isinstance(landmarks, int):
            landmarks = [landmarks]
        if landmarks is None:
            landmarks = range(self._n_landmarks)
        locs = [self._landmarks[i].get_location(frame) for i in landmarks if self._landmarks[i] is not None]
        if exlude_none:
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



@dataclass
class Metric:
    name: str = ""
    type: MetricType = MetricType.LENGTH
    landmarks: List[Union[int, List[int]]] = field(default_factory=list)
