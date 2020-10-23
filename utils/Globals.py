from typing import Dict, Tuple, List, Union, Optional
from utils import DataHolders
from PyQt5 import QtCore, QtWidgets
import os
import numpy as np
import cv2

from typing import Optional, Tuple
import logging

class FrameGrabber(QtCore.QRunnable):
    frame: int

    def __init__(self, glo, frame_num: int):
        super(FrameGrabber, self).__init__()
        self.frame = frame_num
        self._glo = glo

    def run(self):
        cap = self._glo.get_video()
        cap.set(cv2.CAP_PROP_POS_FRAMES, self.frame)
        ret, frame = cap.read()
        if ret is False:
            self._glo.got_frame(None, self.frame)
        else:
            self._glo.got_frame(frame, self.frame)

class Globals(QtWidgets.QWidget):
    """
    Contains references to objects that are utilized throughout the program
    """
    __instance = None

    project: DataHolders.Project = None
    metrics: DataHolders.MetricContainer = None
    curr_landmarks: Optional[DataHolders.Landmarks] = None
    curr_file: str = None
    curr_file_index: int = None
    curr_landmark_file = None
    video_loaded: bool = False  # Whether a video is loaded correctly

    # Configs
    configs: Dict[str, DataHolders.Config] = {
                "visual_config": DataHolders.VisualConfig,
                "video_config": DataHolders.VideoConfig,
                "model_config": DataHolders.ModelConfig
               }
    visual_config: DataHolders.VisualConfig = None
    visual_config_path: str
    video_config: DataHolders.VideoConfig = None
    video_config_path: str
    model_config: DataHolders.ModelConfig = None
    model_config_path: str

    # Signals do not have a reference to the changed object since leeches are
    # intended to only access globals through the globals object
    onProjectChange = QtCore.pyqtSignal()
    onLandmarksChange = QtCore.pyqtSignal()
    onFileChange = QtCore.pyqtSignal()
    onConfigChange = QtCore.pyqtSignal(str)  # Returns the type of config changed
    onMetricsChange = QtCore.pyqtSignal()

    _thread_pool: QtCore.QThreadPool

    def __init__(self, initial_project_dir: str = None):
        super(Globals, self).__init__()

        self._thread_pool = QtCore.QThreadPool()
        if initial_project_dir is not None:
            self.select_project(initial_project_dir)
        if Globals.__instance is None:
            Globals.__instance = self
        else:
            raise RuntimeError("Only one instance of globals is allowed")

    @staticmethod
    def get():
        if Globals.__instance is not None:
            return Globals.__instance
        else:
            return Globals()

    def get_config_change_interaction(self, config_name: str):
        """Gets the function that emits the correct config change"""
        return lambda: self.onConfigChange.emit(config_name)

    def load_config(self):
        """
        Iterates through all configs and loads them
        """
        for (name, c_type) in self.configs.items():
            try:
                self.__dict__[name] = c_type(self.__dict__[name+"_path"], glo=self)
            except ValueError:
                self.__dict__[name] = c_type(glo=self)
                self.__dict__[name].set_save_loc(self.__dict__[name+"_path"])
                self.__dict__[name].on_change()
            self.__dict__[name].onChangeSignal.connect(self.get_config_change_interaction(name))

    def get_video(self, file_override: str = None) -> Optional[cv2.VideoCapture]:
        """
        Tries to open the video stream the current file and updates the visual
        configurations
        :return: The video capture object or None
        """
        try:
            if file_override is None:
                cap = cv2.VideoCapture(self.curr_file)
            else:
                cap = cv2.VideoCapture(file_override)
            if cap is None or not cap.isOpened():
                return None
            self.visual_config.get_video_specs(cap)
        except cv2.error as e:
            return None
        return cap

    def ready(self) -> bool:
        return self.video_loaded

    def select_landmark(self, landmark: Union[int, List[int]]) -> bool:
        """
        Handles the action taken when a user clicks on a landmark to select it
        :param landmark: The id of the clicked landmark
        :return: True if the landmark was added and false if it was removed
        """
        if landmark in self.metrics.working_metric.landmarks:
            # Then remove it from the working metric
            self.metrics.remove_from_working(landmark)
            return False
        else:
            # Then we need to add the landmark to the working metric
            self.metrics.append_to_working(landmark)
            return True

    def deselect_landmarks(self):
        """
        Reset the working landmark
        """
        self.metrics.reset_working()

    def move_landmark(self, frame: int, landmark: int, new_pos: Tuple[float, float]) -> bool:
        """Handles actions taken when a landmark is moved to a new position"""
        if not self.ready():
            return False
        if self.curr_landmarks.set_location(frame, landmark, new_pos):
            self.onLandmarksChange.emit()
            self.curr_landmarks.save(self.curr_landmark_file)
            self.save_for_retrain(frame)
            return True

    def save_for_retrain(self, frame: int):
        frame_grabber = FrameGrabber(self, frame)
        self._thread_pool.start(frame_grabber)

    def got_frame(self, frame: Union[np.ndarray, None], frame_num: int):
        """
        A callback for a save to retrain
        :param frame: The cv2 frame to save
        :param frame_num: The frame number to save for retraining
        """
        if frame is None:
            print("Failed to save frame for retraining")
            return
        cv2.imwrite(os.path.join(self.project.frames_dir, "{}-{}.jpg".format(os.path.basename(os.path.splitext(self.curr_file)[0]), frame_num)), frame)
        landmarks_df = self.curr_landmarks.get_dataframe()
        frame_landmarks = landmarks_df[landmarks_df['Frame_number'] == frame_num]
        frame_landmarks.to_csv(os.path.join(self.project.landmarks_dir, "{}-{}.csv".format(os.path.basename(os.path.splitext(self.curr_file)[0]), frame_num)))

    def load_metrics(self):
        """
        This should eventually load metrics metrics the user has defined from
        their project files.
        :return:
        """
        self.metrics = DataHolders.MetricContainer(os.path.join(self.project.config_dir, "metrics.json"))
        try:
            self.metrics.recall_metrics()
        except ValueError:
            # Then the file did not exist. It will be created when metrics change from default
            pass
        self.metrics.metricChangedSignal.connect(self.onMetricsChange.emit)
        self.onMetricsChange.emit()

    def select_frame(self, frame: int):
        self.video_config.seek_to(frame)

    def select_file(self, file_path: Union[str, int]):
        """Called to select a file in the current project"""
        if isinstance(file_path, int):
            file_path = self.project.files[file_path % len(self.project.files)]
        if file_path not in self.project.files:
            # Maybe prompt user to add it? Maybe this situation never occurs
            raise ValueError("Video file not in project")
        landmark_file = self.project.files_map[file_path]
        landmarks = DataHolders.Landmarks.load(landmark_file)
        cap = self.get_video(file_path)  # To update the video specs
        self.video_config.setup_video(file_path)  # Update the video config
        self.video_config.set_curr_video(file_path)
        self.video_config.add_keypoint(0)
        self.video_config.add_keypoint(int(self.visual_config.video_length)-1)
        if cap is None:
            # Then the video did not load correctly
            self.video_loaded = False
            return None
        self.video_loaded = True
        self.curr_landmark_file = landmark_file
        self.curr_file = file_path
        self.curr_file_index = self.project.files.index(self.curr_file)
        self.curr_landmarks = landmarks
        self.onFileChange.emit()
        self.onLandmarksChange.emit()

    def select_project(self, project_path: str) -> bool:
        """
        Sets up a new project
        :param project_path: The path to the project directory
        :return: True if the project opened correctly otherwise false
        """
        try:
            new_project = DataHolders.Project.load(project_path)
            self.project = new_project
            # Load visual config. Maybe put this inside the project object?
            self.visual_config_path = os.path.join(self.project.config_dir, "visual_config.json")
            self.video_config_path = os.path.join(self.project.config_dir, "video_config.json")
            self.model_config_path = os.path.join(self.project.config_dir, "model_config.json")
            self.load_config()
            self.load_metrics()

            self.onProjectChange.emit()
            current_vid = self.video_config.get_curr_video()
            if current_vid is None or current_vid not in new_project.files:
                self.select_file(new_project.files[0])
            else:
                self.select_file(current_vid)
            logging.info(f"Opened new project {self.project.name}: {project_path}")
            return True
        except ValueError as e:
            print("Failed to open project:", e)
            logging.exception("Failed to open project because")
            # Then the project failed to load correctly. TODO: Maybe open up a config window with the error.
            return False
