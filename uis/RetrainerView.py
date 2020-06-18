from utils.qmlBase import WindowHandler
from PyQt5 import QtQuick, QtCore
from utils.Globals import Globals
import os
import cv2
import pandas as pd
from typing import Dict, List

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

class Loader(Dataset):
    """
    A dataset that passes the retraining data
    """
    img_map: Dict[str, str]
    img_list: List[str]

    def __init__(self, img_map: Dict[str, str]):
        self.img_map = img_map
        self.img_list = list(img_map.keys())
        super().__init__()

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        img_path = self.img_list[idx]
        landmark_path = self.img_map[img_path]
        img = cv2.imread(img_path)
        landmarks = pd.read_csv(landmark_path)
        # Crop down the image, resize it, construct heatmaps of the same size, return
        bbox_cols = ["bbox_top_x", "bbox_top_y", "bbox_bottom_x", "bbox_bottom_y"]
        lm_cols = [col for col in landmarks.columns if "landmark_" in col]
        num_landmarks = len(lm_cols) / 2

class RetrainerView(WindowHandler):
    _glo: Globals

    @property
    def img_mark_map(self):
        img_map = {}
        if self._glo.project is None:
            return img_map
        f_dir = self._glo.project.frames_dir
        l_dir = self._glo.project.landmarks_dir
        frame_list = os.listdir(f_dir)
        landmarks_list = os.listdir(l_dir)
        for file in frame_list:
            coor_landmark = os.path.splitext(file)[0]+".csv"
            if coor_landmark in landmarks_list:
                frame_path = os.path.join(f_dir, file)
                landmark_path = os.path.join(l_dir, coor_landmark)
                img_map[frame_path] = landmark_path
        return img_map

    def __init__(self, engine: QtQuick.QQuickView):
        self._glo = Globals.get()
        super().__init__(engine, "uis/RetrainerView.qml", "Retrain Network")

    def show(self):
        super().show()

    @QtCore.pyqtSlot(name="retrain")
    def retrain(self):
        print("Retraining: ", self.img_mark_map)
