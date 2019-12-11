from typing import Dict, List, Set, Tuple, Union, Optional
import DataHolders
import numpy as np
import cv2
import persistentConfig


class ImageMarker:
    _scale_factor: float = 1

    _landmarks: Optional[DataHolders.Landmarks] = None
    _show: Dict[str, bool] = {"land": True, "bound": False, "metrics": True}
    _selected: Set[int] = set()
    _excluded: Set[int] = set()
    _color_overrides: Dict[int, Tuple[int, int, int]] = {}

    _config: persistentConfig.Config

    _metrics: List[DataHolders.Metric] = []
    _working_metrics: List[DataHolders.Metric] = []

    def __init__(self, landmarks: DataHolders.Landmarks=None, scale_factor: float=1, config: persistentConfig.Config = None):
        self._config = config
        self._landmarks = landmarks
        self._scale_factor = scale_factor

    def set_scale_factor(self, scale_factor: float) -> bool:
        self._scale_factor = scale_factor
        return True

    def set_landmarks(self, landmarks: DataHolders.Landmarks) -> bool:
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
    def set_metrics(self, metrics: Optional[List[DataHolders.Metric]] = None, working_metrics: Optional[List[DataHolders.Metric]] = None) -> bool:
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
        if self._show["bound"] and self._landmarks.has_bbox():
            for bound_box_points in self._landmarks.get_bound_box_locs(frame_num):
                try:
                    lt_point, rb_point = [self._cast_pos(pos, self._scale_factor) for pos in bound_box_points]
                    cv2.rectangle(img, lt_point, rb_point, (255, 0, 0), 2)
                except TypeError:
                    pass

        if self._show["metrics"] and self._landmarks.has_landmarks():
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
                if metric.type == DataHolders.MetricType.AREA:
                    curr_point = self._cast_pos(self._get_point(frame_num, metric.landmarks[0])[0], self._scale_factor)
                    cv2.line(img, last_point, curr_point, self._config.group.highlight_color, 1)

        if self._show["land"] and self._landmarks.has_landmarks():
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
