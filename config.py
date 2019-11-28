from typing import Dict, Tuple
from dataclasses import dataclass, field

#TODO: Mouth and Eyes top/bottom different color
# group_colors = {
#     "left_eye": (230, 30, 30),
#     "right_eye": (230, 30, 30),
#     "nose": (59, 201, 40),
#     "inner_mouth": (39, 78, 232),
#     "outer_mouth": (232, 39, 181),
#     "left_eyebrow": (39, 193, 232),
#     "right_eyebrow": (39, 193, 232),
#     "chin_outline": (149, 77, 158)
# }

group_colors = {
    "face": (224, 27, 70),
    "lower_eye": (30, 27, 224),
    "upper_mouth": (39, 193, 232),
    "lower_mouth": (232, 39, 181)
}

highlight_color = (247, 222, 59)

@dataclass
class GroupConfig:
    colors: Dict[str, Tuple[int, int, int]] = field(default_factory=group_colors)
    highlight_color: Tuple[int, int, int] = highlight_color

@dataclass
class PlaybackConfig:
    speed: float = 1.0

@dataclass
class Config:
    group: GroupConfig = field(default_factory=GroupConfig)
    playback: PlaybackConfig = field(default_factory=PlaybackConfig)

