import DataHolders
from typing import Dict, Tuple, List, Union, Optional
from dataclasses import dataclass, field
from popups.Confirmation import Confirmation
import tempfile
import pickle

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

# TODO: Flesh this out so it can just act as a persistent config
#       Specifically, store user defined metrics over restart

group_colors = {
    "face": (224, 27, 70),
    "lower_eye": (30, 27, 224),
    "upper_mouth": (39, 193, 232),
    "lower_mouth": (232, 39, 181)
}

default_metrics = [
    DataHolders.Metric(
        "Inter-Canthil Distance",
        DataHolders.MetricType.LENGTH,
        [39, 42]
    ),
    DataHolders.Metric(
        "Left Mouth Area",
        DataHolders.MetricType.AREA,
        list(range(51, 57+1))
    ),
    DataHolders.Metric(
        "Right Mouth Area",
        DataHolders.MetricType.AREA,
        [57, 58, 59, 48, 49, 50, 51]
    ),
    DataHolders.Metric(
        "Left Eyebrow-Nose Distance",
        DataHolders.MetricType.LENGTH,
        [list(range(17, 21+1)), 30]
    ),
    DataHolders.Metric(
        "Right Eyebrow-Nose Distance",
        DataHolders.MetricType.LENGTH,
        [list(range(22, 26 + 1)), 30]
    ),
    DataHolders.Metric(
        "Mouth Vertical Range",
        DataHolders.MetricType.LENGTH,
        [51, 57]
    )
]

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
class MetricConfig:
    metrics: List[DataHolders.Metric] = field(default_factory=list)

    def __post_init__(self):
        self.metrics = default_metrics

    def rename(self, old: str, new: str) -> bool:
        metric_names = [metric.name for metric in self.metrics]
        if new in metric_names:
            return False
        for metric in self.metrics:
            if metric.name == old:
                metric.name = new

    def get_all(self):
        return self.metrics

    def get(self, metric_name: str) -> Optional[DataHolders.Metric]:
        for metric in self.metrics:
            if metric.name == metric_name:
                return metric

    def add(self, metric: DataHolders.Metric) -> bool:
        metric_names = [metric.name for metric in self.metrics]
        if metric.name in metric_names:
            return False
        self.metrics.append(metric)

    def remove(self, metric: Union[str, DataHolders.Metric]) -> bool:
        if isinstance(metric, DataHolders.Metric):
            metric = metric.name
        metric_names = [metric.name for metric in self.metrics]
        if metric in metric_names:
            index = metric_names.index(metric)
            self.metrics.pop(index)
            return True
        return False

class Config:
    conf: Confirmation

    group: GroupConfig
    playback: PlaybackConfig
    metrics: MetricConfig

    save_path: str

    def __init__(self, recall: bool = True, pickle_path: str = None):
        if pickle_path is not None:
            self.save_path = pickle_path
        else:
            self.save_path = tempfile.gettempdir()+"/VidProcConfig.p"
        if recall:
            # Then we deserialize all data
            try:
                with open(self.save_path, "rb") as f:
                    self.group, self.playback, self.metrics = pickle.load(f)
                    # self.conf = Confirmation("Loaded config", f"Config loaded from: {self.save_path}", can_deny=False)
            except (FileNotFoundError, EOFError):
                self.set_defaults()
        else:
            # Then we get default values
            self.set_defaults()

    def set_defaults(self):
        self.conf = Confirmation("Using default values", "No past config found", can_deny=False)
        self.group = GroupConfig()
        self.playback = PlaybackConfig()
        self.metrics = MetricConfig()

    def save(self):
        objs = (self.group, self.playback, self.metrics)
        with open(self.save_path, "wb") as f:
            pickle.dump(objs, f)

