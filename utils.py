from typing import Dict, List, Tuple, Union, Optional
from dataclasses import dataclass, field
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

@dataclass
class BoundingBox:
    point1: Tuple[int, int] = (-1, -1)
    point2: Tuple[int, int] = (-1, -1)
@dataclass
class Landmark:
    index: int = -1
    group: str = ""
    location: Tuple[int, int] = (-1, -1)
@dataclass
class FaceLandmarks:
    bounding_box: BoundingBox = field(default_factory=BoundingBox)
    lines: List[List[Landmark]] = field(default_factory=list)
    landmarks: Dict[str, List[Landmark]] = field(default_factory=dict)

@dataclass
class ShowStatus:
    landmarks: bool = True
    bounding_box: bool = True
    metrics: bool = True
@dataclass
class LandmarkFeatures:
    show: ShowStatus = field(default_factory=ShowStatus)
    groups: Dict[str, List[int]] = field(default_factory=dict)
    selected: List[int] = field(default_factory=list)
    excluded: List[int] = field(default_factory=list)
    lines: List[List[int]] = field(default_factory=list)
    color_overrides: Tuple[List[int], List[Tuple[int, int, int]]] = field(default_factory=list)

@dataclass
class LandmarkGroup:
    landmarks: List[Landmark] = field(default_factory=list)
@dataclass
class Metric:
    name: str = ""
    type: MetricType = MetricType.LENGTH
    landmarks: List[Union[Landmark, LandmarkGroup]] = field(default_factory=list)


def get_centroid(points: List[Tuple[float, float]]):
    x_avg = sum([coord[0] for coord in points])/len(points)
    y_avg = sum([coord[1] for coord in points])/len(points)
    return (x_avg, y_avg)

def landmark_frame_to_shapes(landmark_frame: pd.DataFrame, features: LandmarkFeatures) -> Optional[FaceLandmarks]:
    shape_defs = features.groups
    bounding_values = landmark_frame[["bbox_top_x", "bbox_top_y", "bbox_bottom_x", "bbox_bottom_y"]].values[0]
    bounding_box = BoundingBox(
        (int(round(bounding_values[0])), int(round(bounding_values[1]))),
        (int(round(bounding_values[2])), int(round(bounding_values[3])))
    )
    face_landmarks = FaceLandmarks(bounding_box, [], {})

    indices = [shape_defs[group] for group in shape_defs if len(shape_defs[group]) > 0]
    max_index = np.max([np.max(l) for l in indices]) if len(indices) > 0 else 0
    all_vals = []
    for i in range(max_index):
        all_vals.extend([f"landmark_{i}_x", f"landmark_{i}_y"])
    try:
        landmark_values = landmark_frame[all_vals].values[0]
    except Exception as e:
        # TODO: Make this exception handling more specific
        return None

    face_landmarks.landmarks = {}
    for group in shape_defs:
        face_landmarks.landmarks[group] = []
        landmark_indices = shape_defs[group]
        for index in landmark_indices:
            landmark = Landmark(
                index,
                group,
                (
                    int(landmark_values[2 * (index - 1)]),
                    int(landmark_values[2 * (index - 1) + 1])
                )
            )
            face_landmarks.landmarks[group].append(landmark)

    face_landmarks.lines = []
    for line in features.lines:
        face_landmarks.lines.append(list(map(
            lambda landmark_index: Landmark(landmark_index, "lines", (
                    int(landmark_values[2 * (landmark_index - 1)]),
                    int(landmark_values[2 * (landmark_index - 1) + 1])
                    )
                ), line
            ))
        )

    return face_landmarks

def rescale_pos(loc: Tuple[int, int], scale_factor: float) -> Tuple[int, ...]:
    return tuple([int(coord*scale_factor) for coord in loc])

def markup_image(img: np.ndarray,
                 face_landmarks: FaceLandmarks = FaceLandmarks(),
                 landmark_features: LandmarkFeatures = LandmarkFeatures(),
                 resolution: Tuple[int, int] = (1920, 1080),
                 scale_factor: float = 1
                 ):
    color_override = landmark_features.color_overrides
    excluded_landmarks = landmark_features.excluded
    img = img.copy()
    h, w, _ = img.shape
    base_colors = config.group_colors

    if landmark_features.show.bounding_box:
        bounding_box = face_landmarks.bounding_box
        cv2.rectangle(img, rescale_pos(bounding_box.point2, scale_factor),
                      rescale_pos(bounding_box.point1, scale_factor), (255, 0, 0),
                      2)

    if landmark_features.show.metrics:
        lines = face_landmarks.lines
        for line in lines:
            for i in range(len(line) - 1):
                point_one = tuple(
                    [int(round(coord * scale_factor)) for coord in line[i].location])
                point_two = tuple(
                    [int(round(coord * scale_factor)) for coord in line[i + 1].location])
                cv2.line(img, point_one, point_two, config.highlight_color, 1)

    if landmark_features.show.landmarks:
        for i, group in enumerate(face_landmarks.landmarks):
            if group in base_colors:
                # TODO: Make config configurable
                color = base_colors[group]
            else:
                r_seed = sum([ord(s) for s in group])
                np.random.seed(r_seed)
                color = [int(i) for i in np.random.randint(0, 256, 3)]
            landmarks = face_landmarks.landmarks[group]
            for landmark in landmarks:
                # TODO: get the size automatically so it is not fixed
                # TODO: Numpy types dont seem to work for color, fix that
                try:
                    override_pos = color_override[0].index(landmark.index)
                    landmark_color = color_override[1][override_pos]
                except (ValueError, IndexError) as e:
                    landmark_color = color

                if landmark.index in landmark_features.selected:
                    # TODO: Make config configurable
                    landmark_color = config.highlight_color

                if excluded_landmarks is None or landmark.index not in excluded_landmarks:
                    add_landmark_indicator(img, landmark, landmark_color, resolution, scale_factor)
    return img

def add_landmark_indicator(frame, landmark, color, resolution: Tuple[int, int], scale_factor: float):
    max_side = max(resolution)*scale_factor
    circle_rad = int(round(max_side/450))
    true_loc = rescale_pos(landmark.location, scale_factor)
    cv2.circle(img=frame, center=true_loc, radius=circle_rad,
               color=color, thickness=-1)
    cv2.putText(frame,
                str(landmark.index),
                (true_loc[0] - 2, true_loc[1] - 2),
                cv2.FONT_HERSHEY_DUPLEX,
                0.125 * circle_rad,
                (0, 0, 0), 1
                )

