from typing import Dict, List, Set, Tuple, Union, Optional
from dataclasses import dataclass, field
from scipy.spatial.distance import cdist
import pandas as pd
import numpy as np
import cv2
import config
from enum import Enum


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
    locations: Dict[int, Tuple[Tuple[int, int], Tuple[int, int]]]

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
    _bounding_boxes: List[BoundingBox]
    _n_landmarks: int = 0

    def __init__(self, landmarks: pd.DataFrame, n_landmarks: int = 68):
        self._landmarks_frame = landmarks
        self._n_landmarks = n_landmarks
        self._landmarks = []
        self.populate_landmarks()
        self.populate_bounding_boxes()

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
        return True

    def populate_bounding_boxes(self) -> bool:
        self._bounding_boxes = [BoundingBox(self._landmarks_frame)]
        return True

    def get_bound_box_locs(self, frame: int) -> List[Tuple[Tuple[float, float], Tuple[float, float]]]:
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
        if isinstance(landmarks, int):
            landmarks = [landmarks]
        if landmarks is None:
            landmarks = range(self._n_landmarks)
        locs = [self._landmarks[i].get_location(frame) for i in landmarks]
        if exlude_none:
            locs = [loc for loc in locs if loc is not None]
        return locs

    def get_landmarks(self, frame: int, landmarks: Optional[Union[int, List[int]]]=None) -> List[Optional[Tuple[Tuple[float, float], str, int]]]:
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
        locations = [location for location in self.get_landmark_locations(frame, landmarks) if location is not None]
        return tuple(np.sum(locations, axis=0)/len(locations))

    def get_nearest_point(self, frame: int, pos: Tuple[float, float], threshold: int=6) -> Optional[int]:
        locations = self.get_landmark_locations(frame)
        distance = cdist(np.array(locations), np.array([pos]))[:, 0]
        min_index = np.argmin(distance)
        min_dist = distance[min_index]
        if min_dist < threshold or threshold == -1:
            return self._landmarks[min_index].index
        else:
            return None

    def get_point_area(self, frame: int, x_max: float, x_min: float, y_max: float, y_min: float) -> List[int]:
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
            columns = landmark.get_columns()
            for column in columns:
                landmark_frame[column] = columns[column]
        return landmark_frame



@dataclass
class Metric:
    name: str = ""
    type: MetricType = MetricType.LENGTH
    landmarks: List[Union[int, List[int]]] = field(default_factory=list)
class ImageMarker:
    _scale_factor: float = 1

    _landmarks: Optional[Landmarks] = None
    _show: Dict[str, bool] = {"land": True, "bound": False, "metrics": True}
    _selected: Set[int] = set()
    _excluded: Set[int] = set()
    _color_overrides: Dict[int, Tuple[int, int, int]] = {}

    _config: config.Config = None

    _metrics: List[Metric] = []
    _working_metrics: List[Metric] = []

    def __init__(self, landmarks: Landmarks=None, scale_factor: float=1, group_config: config.Config = None):
        if group_config is None:
            self._config = config.Config()
        else:
            self._config = group_config
        self._landmarks = landmarks
        self._scale_factor = scale_factor

    def set_scale_factor(self, scale_factor: float) -> bool:
        self._scale_factor = scale_factor
        return True

    def set_landmarks(self, landmarks: Landmarks) -> bool:
        self._landmarks = landmarks
        return True

    # For managing visibility
    def toggle_landmarks(self) -> bool:
        self._show["land"] = not self._show["land"]
        return self._show["land"]

    def toggle_bounding_box(self) -> bool:
        self._show["bound"] = not self._show["bound"]
        return self._show["bound"]

    def toggle_metrics(self) -> bool:
        self._show["metrics"] = not self._show["metrics"]
        return self._show["metrics"]

    # Metric Management
    def set_metrics(self, metrics: Optional[List[Metric]] = None, working_metrics: Optional[List[Metric]] = None) -> bool:
        if working_metrics is not None:
            self._working_metrics = working_metrics
        if metrics is not None:
            self._metrics = metrics
        return True

    # Group management
    def add_to_group(self, name: str, indices: List[int]) -> bool:
        if name in self._config.group.colors:
            self._landmarks.set_group(indices, name)
            return True
        return False

    def set_group_color(self, name: str, color: Tuple[int, int, int]) -> bool:
        return self._config.group.update_group(name, color)

    def select(self, landmarks: List[int]) -> bool:
        try:
            self._selected.update(landmarks)
            return True
        except TypeError:
            return False

    def deselect(self, landmarks: Optional[List[int]]=None) -> bool:
        if landmarks is None:
            self._selected = set()
        else:
            self._selected.difference_update(landmarks)
        return True

    def exclude(self, landmarks: List[int]) -> bool:
        try:
            self._excluded.update(landmarks)
            return True
        except TypeError:
            return False

    def include(self, landmarks: List[int]=None) -> bool:
        if landmarks is None:
            self._excluded = set()
        else:
            self._excluded.difference_update(landmarks)
        return True

    def add_override(self, index: int, color: Tuple[int, int, int]) -> bool:
        self._color_overrides[index] = color
        return True

    def remove_override(self, index) -> bool:
        if index in self._color_overrides:
            del self._color_overrides[index]
            return True
        return False

    # For marking up frames
    @staticmethod
    def _cast_pos(loc: Tuple[float, float], scale_factor: float) -> Tuple[int, ...]:
        return tuple([int(round(coord * scale_factor)) for coord in loc])

    def _get_point(self, frame_num: int, landmark_def: Union[int, List[int]]):
        is_centroid = False
        if isinstance(landmark_def, list):
            point = self._landmarks.get_centroid(frame_num, landmark_def)
            is_centroid = True
        else:
            point = self._landmarks.get_landmark_locations(frame_num, landmark_def)[0]
        return point, is_centroid

    def markup_image(self, img: np.ndarray, frame_num: int) -> np.ndarray:
        img = img.copy()
        h, w, _ = img.shape
        circle_rad = int(round(max(h, w) / 450))
        if self._show["bound"]:
            for bound_box_points in self._landmarks.get_bound_box_locs(frame_num):
                try:
                    lt_point, rb_point = [self._cast_pos(pos, self._scale_factor) for pos in bound_box_points]
                    cv2.rectangle(img, lt_point, rb_point, (255, 0, 0), 2)
                except TypeError:
                    pass

        if self._show["metrics"]:
            for metric in self._metrics+self._working_metrics:
                if len(metric.landmarks) < 1:
                    continue
                last_point = self._cast_pos(self._get_point(frame_num, metric.landmarks[0])[0], self._scale_factor)
                for landmark_def in metric.landmarks:
                    curr_point = self._cast_pos(self._get_point(frame_num, landmark_def)[0], self._scale_factor)
                    cv2.line(img, last_point, curr_point, self._config.group.highlight_color, 1)
                    cv2.circle(img=img, center=last_point, radius=circle_rad+1, color=self._config.group.highlight_color, thickness=1)
                    last_point = curr_point
                cv2.circle(img=img, center=last_point, radius=circle_rad + 1, color=self._config.group.highlight_color, thickness=1)
                if metric.type == MetricType.AREA:
                    curr_point = self._cast_pos(self._get_point(frame_num, metric.landmarks[0])[0], self._scale_factor)
                    cv2.line(img, last_point, curr_point, self._config.group.highlight_color, 1)

        if self._show["land"]:
            for landmark in self._landmarks.get_landmarks(frame_num):
                loc, group, index = landmark
                if index in self._excluded:
                    continue
                color = (0, 0, 0)
                group_colors = self._config.group.colors
                if group in group_colors:
                    color = group_colors[group]
                if index in self._color_overrides:
                    color = self._color_overrides[index]
                if index in self._selected:
                    color = self._config.group.highlight_color
                true_pos = self._cast_pos(loc, self._scale_factor)
                cv2.circle(img=img, center=true_pos, radius=circle_rad, color=color, thickness=-1)
                cv2.putText(img,
                            str(index),
                            (true_pos[0] - 2, true_pos[1] - 2),
                            cv2.FONT_HERSHEY_DUPLEX,
                            0.125 * circle_rad,
                            (0, 0, 0), 1
                            )
        return img
