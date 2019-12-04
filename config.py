from typing import Dict, Tuple
from dataclasses import dataclass, field

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
    colors: Dict[str, Tuple[int, int, int]] = field(default_factory=dict)
    highlight_color: Tuple[int, int, int] = highlight_color

    def __post_init__(self):
        self.colors = group_colors.copy()

    def update_group(self, name: str, color: Tuple[int, int, int]) -> bool:
        self.colors[name] = color
        return True

    def remove_group(self, name: str) -> bool:
        if name in self.colors:
            del self.colors[name]
            return True
        return False

@dataclass
class PlaybackConfig:
    speed: float = 1.0

@dataclass
class Config:
    group: GroupConfig = field(default_factory=GroupConfig)
    playback: PlaybackConfig = field(default_factory=PlaybackConfig)

