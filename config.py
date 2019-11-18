from typing import Dict, Tuple
from dataclasses import dataclass, field

group_colors = {
    "left_eye": (230, 30, 30),
    "right_eye": (230, 30, 30),
    "nose": (59, 201, 40),
    "inner_mouth": (39, 78, 232),
    "outer_mouth": (232, 39, 181),
    "left_eyebrow": (39, 193, 232),
    "right_eyebrow": (39, 193, 232),
    "chin_outline": (149, 77, 158)
}

@dataclass
class GroupConfig:
    colors: Dict[str, Tuple[int, int, int]] = field(default_factory=dict)

@dataclass
class PlaybackConfig:
    speed: float = 1.0

@dataclass
class Config:
    group: GroupConfig = field(default_factory=GroupConfig)
    playback: PlaybackConfig = field(default_factory=PlaybackConfig)

