from typing import Dict, List, Tuple, Union, Optional
from dataclasses import dataclass, field
import pandas as pd
import numpy as np
import cv2
import config


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
    landmarks: Dict[str, List[Landmark]] = field(default_factory=dict)


def landmark_frame_to_shapes(landmark_frame: pd.DataFrame, shape_defs: Dict[str, List[int]]) -> Optional[FaceLandmarks]:
    bounding_values = landmark_frame[["bbox_top_x", "bbox_top_y", "bbox_bottom_x", "bbox_bottom_y"]].values[0]
    bounding_box = BoundingBox(
        (int(bounding_values[0]), int(bounding_values[1])),
        (int(bounding_values[2]), int(bounding_values[3]))
    )
    face_landmarks = FaceLandmarks(bounding_box, {})

    indices = [shape_defs[group] for group in shape_defs if len(shape_defs[group]) > 0]
    max_index = np.max([np.max(l) for l in indices])
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
    return face_landmarks


def markup_image(img: np.ndarray,
                 face_landmarks: FaceLandmarks = FaceLandmarks(),
                 lines: List[Tuple[Landmark, Landmark]] = None,
                 color_override=None,
                 excluded_landmarks: Optional[List[int]]=None
                 ):
    if color_override is None:
        color_override = [[], []]
    img = img.copy()
    h, w, _ = img.shape
    base_colors = config.group_colors
    for i, group in enumerate(face_landmarks.landmarks):
        if group in base_colors:
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
            except ValueError:
                landmark_color = color

            if excluded_landmarks is None or landmark.index not in excluded_landmarks:
                cv2.circle(img=img, center=landmark.location, radius=5, color=landmark_color, thickness=-1)
                cv2.putText(img,
                            str(landmark.index),
                            (landmark.location[0] - 2, landmark.location[1] - 2),
                            cv2.FONT_HERSHEY_DUPLEX,
                            0.125 * 5,
                            (0, 0, 0), 1
                            )

    bounding_box = face_landmarks.bounding_box
    cv2.rectangle(img, bounding_box.point2, bounding_box.point1, (255, 0, 0), 4)
    return img

