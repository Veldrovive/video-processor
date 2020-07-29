from utils.qmlBase import WindowHandler
from PyQt5 import QtQuick, QtCore
from PyQt5.QtCore import QObject
from utils.Globals import Globals
import os
from typing import Dict, List, Union, Optional
from enum import Enum

import numpy as np
from tensorboard import program
from landmark_detection.face_alignment.models import LightningFAN
from pytorch_lightning import Trainer
from pytorch_lightning import loggers
from pytorch_lightning import Callback
from pytorch_lightning.callbacks import ProgressBarBase, ModelCheckpoint


SplitType = Enum("SplitType", "ALL TRAIN NONE")

class Signaler(QObject):
    fit_start_signal = QtCore.pyqtSignal()
    fit_end_signal = QtCore.pyqtSignal()
    test_start_signal = QtCore.pyqtSignal()
    test_end_signal = QtCore.pyqtSignal()
    epoch_start_signal = QtCore.pyqtSignal(int, int)  # (current epoch, max epoch)
    epoch_end_signal = QtCore.pyqtSignal(int, int)
    batch_start_signal = QtCore.pyqtSignal(str, int, int, float, int, int, float)  # (type, idx, max_idx, prop, epoch, max_epoch, epoch_percent)
    batch_end_signal = QtCore.pyqtSignal(str, int, int, float, int, int, float)

class StepCallbacks(ProgressBarBase):
    def disable(self):
        pass

    def enable(self):
        pass

    def __init__(self):
        super().__init__()
        self.enable = False

    signaler = Signaler()

    def on_batch_end(self, trainer, pl_module):
        super().on_batch_end(trainer, pl_module)
        total = self.total_train_batches + self.total_val_batches
        prop = self.train_batch_idx / total
        self.signaler.batch_end_signal.emit("train", self.train_batch_idx, total, prop, trainer.current_epoch + 1, trainer.max_epochs, (trainer.current_epoch + 1) / trainer.max_epochs)

    def on_batch_start(self, trainer, pl_module):
        super().on_batch_start(trainer, pl_module)
        total = self.total_train_batches + self.total_val_batches
        prop = self.train_batch_idx / total
        self.signaler.batch_start_signal.emit("train", self.train_batch_idx, total, prop, trainer.current_epoch, trainer.max_epochs, trainer.current_epoch / trainer.max_epochs)

    def on_epoch_end(self, trainer, pl_module):
        super().on_epoch_end(trainer, pl_module)
        self.signaler.epoch_end_signal.emit(trainer.current_epoch, trainer.max_epochs)

    def on_epoch_start(self, trainer, pl_module):
        super().on_epoch_start(trainer, pl_module)
        self.signaler.epoch_start_signal.emit(trainer.current_epoch, trainer.max_epochs)

    def on_fit_end(self, trainer):
        super().on_fit_end(trainer)
        self.signaler.fit_end_signal.emit()

    def on_fit_start(self, trainer):
        super().on_fit_start(trainer)
        self.signaler.fit_start_signal.emit()

    def on_test_batch_end(self, trainer, pl_module):
        super().on_test_batch_end(trainer, pl_module)
        prop = self.test_batch_idx / self.total_test_batches
        self.signaler.batch_end_signal.emit("test", self.test_batch_idx, self.total_test_batches, prop, trainer.current_epoch + 1, trainer.max_epochs, (trainer.current_epoch + 1) / trainer.max_epochs)

    def on_test_batch_start(self, trainer, pl_module):
        super().on_test_batch_start(trainer, pl_module)
        prop = self.test_batch_idx / self.total_test_batches
        self.signaler.batch_start_signal.emit("test", self.test_batch_idx, self.total_test_batches, prop, trainer.current_epoch, trainer.max_epochs, trainer.current_epoch / trainer.max_epochs)

    def on_test_end(self, trainer, pl_module):
        super().on_test_end(trainer, pl_module)
        self.signaler.test_end_signal.emit()

    def on_test_start(self, trainer, pl_module):
        super().on_test_start(trainer, pl_module)
        self.signaler.test_start_signal.emit()

    def on_validation_batch_end(self, trainer, pl_module):
        super().on_validation_batch_end(trainer, pl_module)
        idx = self.train_batch_idx + self.val_batch_idx
        total = self.total_train_batches + self.total_val_batches
        prop = idx / total
        self.signaler.batch_end_signal.emit("val", idx, total, prop, trainer.current_epoch + 1, trainer.max_epochs, (trainer.current_epoch + 1) / trainer.max_epochs)

    def on_validation_batch_start(self, trainer, pl_module):
        super().on_validation_batch_start(trainer, pl_module)
        idx = self.train_batch_idx + self.val_batch_idx
        total = self.total_train_batches + self.total_val_batches
        prop = idx / total
        self.signaler.batch_start_signal.emit("val", idx, total, prop, trainer.current_epoch, trainer.max_epochs, trainer.current_epoch / trainer.max_epochs)

class Retrainer(QtCore.QThread):
    _glo: Globals

    name: str

    trainingStarted = QtCore.pyqtSignal()
    testingStarted = QtCore.pyqtSignal()
    trainingFinished = QtCore.pyqtSignal()

    @property
    def model_dir(self):
        return os.path.join(self._glo.project.fan_dir, f"{self.name}")

    @property
    def checkpoint_dir(self):
        return os.path.join(self.model_dir, "checkpoints")

    @property
    def log_dir(self):
        return os.path.join(self.model_dir, "logs")

    def __init__(self, name: str, video_splits, total_frames: int, val_prop: float = 0.2, test_prop: float = 0.2, split_type: SplitType = SplitType.NONE):
        self._glo = Globals.get()
        self.sets = None
        self.split_type = split_type
        self.splits = video_splits
        self.frames = total_frames
        self.val_prop, self.test_prop, self.train_prop = val_prop, test_prop, 1 - (val_prop + test_prop)
        self.val_frames = int(round(self.frames * val_prop))
        self.test_frames = int(round(self.frames * test_prop))
        self.train_frames = self.frames - (self.val_frames + self.test_frames)

        if self.sets is None:
            self.get_splits()
        self.model = LightningFAN(
            self.split_to_map(self.sets["train"]),
            self.split_to_map(self.sets["val"]),
            self.split_to_map(self.sets["test"]),
            learning_rate=2.7542287033381663e-05, batch_size=1,
            FAN_path=self._glo.project.default_fan_path)
        self.set_name(name)
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.tb_logger = loggers.TensorBoardLogger(self.log_dir, name=self.name)
        self.callbacks = StepCallbacks()
        self.signals = self.callbacks.signaler

        self.checkpoint_callback = ModelCheckpoint(
            filepath=self.checkpoint_dir,
            save_top_k=1,
            monitor='loss',
            mode='min',
        )

        self.trainer = Trainer(
            logger=self.tb_logger,
            max_epochs=2,
            checkpoint_callback=self.checkpoint_callback,
            callbacks=[self.callbacks]
        )
        # print("Old Batch Size: ", model.hparams.batch_size)
        # new_batch_size = trainer.scale_batch_size(model)
        # print(new_batch_size)
        # model.hparams.batch_size = new_batch_size
        # lr_finder = trainer.lr_find(model)
        # new_lr = lr_finder.suggestion()
        # print(new_lr, lr_finder.results)
        # model.hparams.learning_rate = new_lr

        super().__init__()

    def load_checkpoint(self, checkpoint_path: str):
        model = LightningFAN.load_from_checkpoint(checkpoint_path)
        self.model = model
        trainer = Trainer(resume_from_checkpoint=checkpoint_path, max_epochs=2, logger=self.tb_logger, checkpoint_callback=self.checkpoint_callback, callbacks=[self.callbacks])
        self.trainer = trainer
        try:
            self.set_name(model.train_map["name"])
        except KeyError:
            self.set_name("")

    def cost(self, sample: Dict[str, list]):
        """
        Returns the difference between the optimal split and the current one
        :param sample: {"train": [video_name...], "val": [video_name...], "test": [video_name...]} or
            {"train": [video_name...], "test": [video_name...]
        """
        try:
            val_videos, test_videos, train_videos = sample["val"], sample["test"], sample["train"]
            val_total = sum([val["num_frames"] for key, val in self.splits.items() if key in val_videos])
            test_total = sum([val["num_frames"] for key, val in self.splits.items() if key in test_videos])
            train_total = sum([val["num_frames"] for key, val in self.splits.items() if key in train_videos])
            return abs(val_total - self.val_frames) + abs(test_total - self.test_frames) + abs(train_total - self.train_frames)
        except KeyError:
            test_videos, train_videos = sample["test"], sample["train"]
            test_total = sum([val["num_frames"] for key, val in self.splits.items() if key in test_videos])
            train_total = sum([val["num_frames"] for key, val in self.splits.items() if key in train_videos])
            return abs(test_total - self.test_frames) + abs(train_total - (self.train_frames + self.val_frames))

    def rand_swap(self, to_swap: Dict[str, list]):
        """
        Takes a dictionary of lists and swaps an element from one list to another randomly
        :param to_swap: {"train": [...], "val": [...], "test": [...]}
        :return:
        """
        new_swap = {}
        for key, elem in to_swap.items():
            new_swap[key] = elem.copy()
        reverse_map = []
        for key in new_swap.keys():
            for video in new_swap[key]:
                reverse_map.append((key, video))
        choice_key, choice_video = reverse_map[np.random.randint(len(reverse_map))]
        potential_keys = list(new_swap.keys())
        potential_keys.remove(choice_key)
        new_key = np.random.choice(potential_keys)
        new_swap[choice_key].remove(choice_video)
        new_swap[new_key].append(choice_video)
        return new_swap

    def optimize(self, seed: Dict[str, list], iterations: int = 1000, timeout: int = None):
        """
        Selects a good split for val, test, and train sets
        :param seed: A starting point to optimize off of:
            {"train": [video_name...], "val": [video_name...], "test": [video_name...]} or
            {"train": [video_name...], "test": [video_name...]
        :param iterations: How many iterations to randomize for
        :param timeout: How many iterations of no improvement to go for before quitting
        :return: {"val": [], "test": [], "train": []} map for optimal
        """
        curr = seed
        curr_cost = self.cost(curr)
        timeout_num = 0
        for i in range(iterations):
            swapped = self.rand_swap(curr)
            new_cost = self.cost(swapped)
            delta = new_cost - curr_cost
            timeout_num += 1
            if delta < 0:
                curr = swapped
                timeout_num = 0
                curr_cost = new_cost
            if timeout is not None and timeout_num > timeout:
                break
        return self.cost(curr), curr

    def get_rand(self, split_type: SplitType):
        """
        Gets a seed for the cooresponding split type
        :param split_type: The current split_type
        :return:
        """
        if split_type == SplitType.ALL:
            rand_map = {"train": [], "val": [], "test": []}
            for video in self.splits.keys():
                rand_map[list(rand_map.keys())[np.random.randint(0, 3)]].append(video)
            return rand_map

        if split_type == SplitType.TRAIN:
            rand_map = {"train": [], "test": []}
            for video in self.splits.keys():
                rand_map[list(rand_map.keys())[np.random.randint(0, 2)]].append(video)
            return rand_map

    def split_data(self, split_type: SplitType):
        """
        This takes video splits and returns the frames that should go in each dataset
        :param split_type:
            NONE: Then all sets are completely randomized
            TRAIN: Then train and test sets come from different videos and train is the split into train and val
            ALL: Then all sets must come from different videos
        """
        sets = {"train": [], "test": [], "val": []} # One entry will be {video: str, frame: str, landmarks: str}
        if split_type == SplitType.NONE:
            # Then all frames are split up randomly
            splitter = [("train" if i < self.train_frames else ("test" if i < self.train_frames + self.test_frames else "val")) for i in range(self.frames)]
            np.random.shuffle(splitter)
            for video, frames in self.splits.items():
                for frame in frames["frames"]:
                    group = splitter.pop(0)
                    sets[group].append({"video": video, "frame": frame["img_file"], "landmarks": frame["landmark_file"]})
            return sets

        seed = self.get_rand(split_type)
        cost, split = self.optimize(seed, iterations=1000, timeout=100)

        for video, frames in self.splits.items():
            group = None
            for pot_group in split.keys():
                if video in split[pot_group]:
                    group = pot_group
            for frame in frames["frames"]:
                sets[group].append({"video": video, "frame": frame["img_file"], "landmarks": frame["landmark_file"]})

        if split_type == SplitType.TRAIN:
            num_train = len(sets["train"])
            opt_val = int(round(self.frames * self.val_prop - len(sets["test"]) * self.val_prop))
            splitter = ["val" if i < opt_val else "train" for i in range(num_train)]
            np.random.shuffle(splitter)
            train_set = sets["train"]
            sets["train"] = []
            for i, frame in enumerate(train_set):
                sets[splitter[i]].append(frame)

        return sets

    def get_splits(self, split_type = None):
        if split_type is not None:
            self.split_type = split_type
        self.sets = self.split_data(self.split_type)
        return self.sets

    def split_to_map(self, split):
        """Takes a split and turns it into a map that the dataset is built to understand"""
        frame_map = {}
        for frame in split:
            frame_map[frame["frame"]] = frame["landmarks"]
        return frame_map

    def set_name(self, name: str):
        self.name = name
        if self.model is not None:
            self.model.hparams.name = name

    def get_save_path(self):
        def get_version(name):
            try:
                return int(os.path.splitext(name)[0][len(file_base):])
            except ValueError:
                return -1
        file_base = f"{self.name}_model_v"
        past_versions = [get_version(file) for file in os.listdir(self.model_dir) if file_base in file] + [0]
        new_version = max(past_versions) + 1
        return os.path.join(self.model_dir, f"{self.name}_model_v{new_version}.ptl")

    def run(self):
        self.trainingStarted.emit()
        self.model.train_freeze()
        self.trainer.fit(self.model)
        self.testingStarted.emit()
        self.trainer.test()
        self.trainingFinished.emit()

        self.trainer.save_checkpoint(self.get_save_path())


class RetrainerView(WindowHandler):
    _glo: Globals

    _name: str = ""
    val_split = 0.2
    test_split = 0.2
    split_type: SplitType = SplitType.NONE

    retrainer: Optional[Retrainer] = None

    _current_video_index: int = -1
    _current_img_index: int = -1

    batchCompleted = QtCore.pyqtSignal(int, int, float, arguments=["current", "max", "progress"])  # (current, max, progress)
    epochCompleted = QtCore.pyqtSignal(int, int, float, arguments=["current", "max", "progress"])  # (current, max, progress)
    trainingStarted = QtCore.pyqtSignal()
    trainingFinished = QtCore.pyqtSignal()

    @QtCore.pyqtSlot(name="getCheckpointPath")
    def get_checkpoint_path(self):
        if self._glo.project is None:
            return ""
        return os.path.join(self._glo.project.fan_dir, "checkpoints")

    checkpointPathUpdated = QtCore.pyqtSignal()
    @QtCore.pyqtProperty(str, notify=checkpointPathUpdated)
    def checkpointPath(self):
        if self._glo.project is None:
            return ""
        return os.path.join(self._glo.project.fan_dir, "checkpoints")

    nameUpdated = QtCore.pyqtSignal()
    @QtCore.pyqtProperty(str, notify=nameUpdated)
    def name(self):
        return self._name

    @name.setter
    def name(self, name):
        self._name = name
        self.nameUpdated.emit()

    videosUpdated = QtCore.pyqtSignal()
    @QtCore.pyqtProperty(list, notify=videosUpdated)
    def videos_list(self):
        if self._glo.project is None:
            return []
        return [{"name": os.path.basename(file), "path": file} for file in self._glo.project.files]

    @QtCore.pyqtSlot(int, name="updateCurrentVideo")
    def update_current_video(self, index: int):
        print("New index: ", index)
        self._current_video_index = index
        new_video = self.videos_list[index]
        self._glo.select_file(new_video["path"])
        self.framesUpdated.emit()

    framesUpdated = QtCore.pyqtSignal()
    @QtCore.pyqtProperty(int, notify=framesUpdated)
    def currentVideoIndex(self):
        return self._current_video_index

    @QtCore.pyqtProperty(list, notify=framesUpdated)
    def frames_list(self):
        if self._current_video_index < 0:
            return []
        splits = self.video_splits[0]
        video_name = os.path.splitext(self.videos_list[self._current_video_index]["name"])[0]
        try:
            return sorted(splits[video_name]["frames"], key=lambda elem: elem["frame_number"])
        except KeyError:
            return []

    currentImgUpdated = QtCore.pyqtSignal()
    @QtCore.pyqtProperty(object, notify=currentImgUpdated)
    def current_img(self):
        try:
            return self.frames_list[self.current_img_index]
        except IndexError:
            return None

    @QtCore.pyqtSlot(int, name="setCurrentFrame")
    def set_curr_frame(self, frame: int):
        self._current_img_index = frame
        self._glo.select_frame(frame)
        self.currentImgUpdated.emit()

    @QtCore.pyqtSlot(int, name="deleteFrame")
    def delete_frame(self, index: int):
        frames = self.frames_list
        frame = frames[index]
        os.remove(frame["img_file"])
        os.remove(frame["landmark_file"])
        if self._current_img_index == index and index == len(frames) - 1:
            self._current_img_index -= 1
            self.currentImgUpdated.emit()
        self.framesUpdated.emit()

    @property
    def video_splits(self):
        total_frames = 0
        video_map = {}
        if self._glo.project is None:
            return video_map
        f_dir = self._glo.project.frames_dir
        l_dir = self._glo.project.landmarks_dir
        frame_list = os.listdir(f_dir)
        landmarks_list = os.listdir(l_dir)
        for file in frame_list:
            coor_landmark = os.path.splitext(file)[0] + ".csv"
            if coor_landmark in landmarks_list:
                file_name = os.path.splitext(file)[0]
                video_name, frame_number = file_name.split("-")
                frame_number = int(frame_number)
                if video_name not in video_map:
                    video_map[video_name] = {"num_frames": 0, "frames": []}
                video_map[video_name]["num_frames"] += 1
                total_frames += 1
                video_map[video_name]["frames"].append({"img_file": os.path.join(f_dir, file), "landmark_file": os.path.join(l_dir, coor_landmark), "frame_number": frame_number})
        return video_map, total_frames

    @property
    def videos(self):
        return list(self.video_splits[0].keys())

    def generate_paths(self, videos: List[str]):
        img_map = {}
        own_videos = self.videos
        own_splits = self.video_splits[0]
        for video in videos:
            if video not in own_videos:
                continue
            for frame in own_splits[video]["frames"]:
                img_map[frame["img_file"]] = frame["landmark_file"]
        return img_map

    def __init__(self, engine: QtQuick.QQuickView):
        self._glo = Globals.get()
        super().__init__(engine, "uis/RetrainerView.qml", "Retrain Network")

    def show(self):
        super().show()
        self.videosUpdated.emit()
        self.framesUpdated.emit()
        self.checkpointPathUpdated.emit()

    @QtCore.pyqtSlot(str, name="loadCheckpoint")
    def load_checkpoint(self, checkpoint_path: str):
        print("Loading checkpoint at path: ", checkpoint_path)
        video_splits, total_frames = self.video_splits
        self.retrainer = Retrainer(self.name, video_splits, total_frames, test_prop=self.test_split, val_prop=self.val_split, split_type=SplitType.NONE)
        self.retrainer.load_checkpoint(checkpoint_path)
        self.connect_retrainer()
        self.name = self.retrainer.name
        self.nameUpdated.emit()

    def connect_retrainer(self):
        if self.retrainer is not None:
            def reemit(_batch_type, idx, max_idx, prop, epoch, max_epoch, epoch_percent):
                self.batchCompleted.emit(idx, max_idx, prop)
                self.epochCompleted.emit(epoch, max_epoch, epoch_percent)

            def on_finished():
                self.trainingFinished.emit()
                reemit("", -1, -1, 0, -1, -1, 0)
                self.send_message("Finished Retraining")
                self.retrainer = None

            def on_started():
                self.trainingStarted.emit()
                reemit("", 0, 0, 0, 0, 0, 0)
                self.start_tensorboard()

            self.retrainer.signals.batch_end_signal.connect(reemit)
            self.retrainer.trainingStarted.connect(on_started)
            self.retrainer.trainingFinished.connect(on_finished)

    def start_tensorboard(self):
        tb = program.TensorBoard()
        tb.configure(argv=[None, '--logdir', self.retrainer.log_dir])
        url = tb.launch()
        self.send_message(f"Tensorboard started at: <br/> <a href='{url}'>{url}</a>")

    @QtCore.pyqtSlot(name="retrain")
    def retrain(self, reset: bool = False):
        if len(self.name) < 1:
            self.send_message("New model must have a name")
            return
        video_splits, total_frames = self.video_splits
        if reset or self.retrainer is None:
            self.retrainer = Retrainer(self.name, video_splits, total_frames, test_prop=self.test_split, val_prop=self.val_split, split_type=SplitType.NONE)
            self.connect_retrainer()
        self.retrainer.set_name(self.name)
        self.retrainer.start()
