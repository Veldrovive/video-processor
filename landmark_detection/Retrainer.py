from PyQt5 import QtCore
import torch

class Retrainer(QtCore.QThread):
    _face_alignment_model_path: str = "./landmark_detection/models/2DFAN4-11f355bf06.pth.tar"
    _face_detector_model_path: str = "./landmark_detection/models/s3fd-619a316812.pth"

    def __init__(self):
        super(Retrainer, self).__init__()
        pass
