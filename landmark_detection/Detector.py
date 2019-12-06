from PyQt5 import QtCore
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from queue import Queue

from landmark_detection.face_alignment.utils import *
from landmark_detection.face_alignment import api as face_alignment
from landmark_detection.face_alignment.models import FAN
from landmark_detection.face_alignment.detection.sfd import sfd_detector
from landmark_detection.face_alignment.detection.sfd.sfd_detector import s3fd
from landmark_detection.face_alignment.detection.sfd.bbox import *

from typing import Union, Tuple, List

class LandmarkDetector(QtCore.QThread):
    _face_alignment_model_path: str = "./landmark_detection/models/2DFAN4-11f355bf06.pth.tar"
    _face_detector_model_path: str = "./landmark_detection/models/s3fd-619a316812.pth"
    _device: str = None

    _network_size: int
    _det_thresh: float
    _num_parallel_frames: int

    _video_queue: Queue
    _video_handler: cv2.VideoCapture

    _face_alignment_net: FAN = None
    _face_detector_net: s3fd = None

    frame_done_signal = QtCore.pyqtSignal(int, float)
    landmarks_complete_signal = QtCore.pyqtSignal(str, pd.DataFrame)
    video_added_to_queue_signal = QtCore.pyqtSignal(str)
    new_video_started_signal = QtCore.pyqtSignal(str)

    _running: bool = True

    def __init__(self, network_size: int=4, det_thresh: float=0.3, FAN_model_path: str=None, detector_model_path: str=None, num_frames: int=1):
        super(LandmarkDetector, self).__init__()
        self._video_queue = Queue()
        self._network_size = network_size
        self._det_thresh = det_thresh
        self._num_parallel_frames = num_frames
        if FAN_model_path is not None:
            self._face_alignment_model_path = FAN_model_path
        if detector_model_path is not None:
            self._face_detector_model_path = detector_model_path

        self._device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

        # Initialize Face Detection
        self._face_alignment_net = FAN(self._network_size)
        self.load_weights(self._face_alignment_net, self._face_alignment_model_path)
        self._face_alignment_net.to(self._device)
        self._face_alignment_net.eval()

        # Initialize Landmark Localization
        self._face_detector_net = s3fd()
        self.load_weights(self._face_detector_net, self._face_detector_model_path)
        self._face_detector_net.to(self._device)
        self._face_detector_net.eval()
        self._face_detector_net.reference_scale = 195

    def add_video(self, video_name: str, handler: cv2.VideoCapture) -> bool:
        self._video_queue.put((video_name, handler))
        self.video_added_to_queue_signal.emit(video_name)
        return True

    def load_weights(self, model: Union[FAN, s3fd], filename: str) -> bool:
        sd = torch.load(filename, map_location=lambda storage, loc: storage)
        names = set(model.state_dict().keys())
        for n in list(sd.keys()):
            if n not in names and n + '_raw' in names:
                if n + '_raw' not in sd:
                    sd[n + '_raw'] = sd[n]
                del sd[n]
        model.load_state_dict(sd)
        return True

    def detect_bbox(self, net: Union[FAN, s3fd], img: torch.Tensor) -> np.ndarray:
        if 'cuda' in self._device:
            torch.backends.cudnn.benchmark = True

        with torch.no_grad():
            olist = net(img)

        bboxlist = []
        for i in range(len(olist) // 2):
            olist[i * 2] = F.softmax(olist[i * 2], dim=1)
        olist = [oelem.data.cpu() for oelem in olist]
        for i in range(len(olist) // 2):
            ocls, oreg = olist[i * 2], olist[i * 2 + 1]
            FB, FC, FH, FW = ocls.size()  # feature map size
            stride = 2 ** (i + 2)  # 4,8,16,32,64,128
            anchor = stride * 4
            poss = zip(*np.where(ocls[:, 1, :, :] > 0.05))
            for Iindex, hindex, windex in poss:
                axc, ayc = stride / 2 + windex * stride, stride / 2 + hindex * stride
                score = ocls[0, 1, hindex, windex]
                loc = oreg[0, :, hindex, windex].contiguous().view(1, 4)
                priors = torch.Tensor([[axc / 1.0, ayc / 1.0, stride * 4 / 1.0,
                                        stride * 4 / 1.0]])
                variances = [0.1, 0.2]
                box = decode(loc, priors, variances)
                x1, y1, x2, y2 = box[0] * 1.0
                # cv2.rectangle(imgshow,(int(x1),int(y1)),(int(x2),int(y2)),(0,0,255),1)
                bboxlist.append([x1, y1, x2, y2, score])
        bboxlist = np.array(bboxlist)
        if 0 == len(bboxlist):
            bboxlist = np.zeros((1, 5))

        return bboxlist

    def nms(self, dets: np.ndarray) -> List[float]:
        if 0 == len(dets):
            return []
        x1, y1, x2, y2, scores = dets[:, 0], dets[:, 1], dets[:, 2], dets[:,
                                                                     3], dets[:,
                                                                         4]
        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        order = scores.argsort()[::-1]

        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            xx1, yy1 = np.maximum(x1[i], x1[order[1:]]), np.maximum(y1[i], y1[
                order[1:]])
            xx2, yy2 = np.minimum(x2[i], x2[order[1:]]), np.minimum(y2[i], y2[
                order[1:]])

            w, h = np.maximum(0.0, xx2 - xx1 + 1), np.maximum(0.0,
                                                              yy2 - yy1 + 1)
            ovr = w * h / (areas[i] + areas[order[1:]] - w * h)

            inds = np.where(ovr <= self._det_thresh)[0]
            order = order[inds + 1]

        return keep

    def get_preds_fromhm_subpixel(self, hm, center=None, scale=None):
        # TODO: Figure out if this is ever used
        """Similar to `get_preds_fromhm` Except it tries to estimate the coordinates of the mode of the distribution.
        """
        max, idx = torch.max(
            hm.view(hm.size(0), hm.size(1), hm.size(2) * hm.size(3)), 2)
        preds = idx.view(idx.size(0), idx.size(1), 1).repeat(1, 1, 2).float()
        preds[..., 0] = (preds[..., 0]) % hm.size(3)
        preds[..., 1].div_(hm.size(2)).floor_()
        eps = torch.tensor(0.0000000001).to(hm.device)
        # This is a magic number as far as understand.
        # 0.545 reduces the quantization error to exactly zero when `scale` is ~1.
        # 0.555 reduces the quantization error to exactly zero when `scale` is ~3.
        # 0.560 reduces the quantization error to exactly zero when `scale` is ~4.
        # 0.565 reduces the quantization error to exactly zero when `scale` is ~5.
        # 0.580 reduces the quantization error to exactly zero when `scale` is ~10.
        # 0.5825 reduces the quantization error to <0.002RMSE  when `scale` is ~100.
        sigma = 0.55
        for i in range(preds.size(0)):
            for j in range(preds.size(1)):
                hm_ = hm[i, j, :]
                pX, pY = int(preds[i, j, 0]), int(preds[i, j, 1])
                x0 = pX
                y0 = pY
                p0 = torch.max(hm_[pY, pX], eps)
                if pX < 63:
                    p1 = torch.max(hm_[pY, pX + 1], eps)
                    x1 = x0 + 1
                    y1 = y0
                    x = (3 * sigma) ** 2 * (torch.log(p1) - torch.log(p0)) - (
                            x0 ** 2 - x1 ** 2 + y0 ** 2 - y1 ** 2) / 2
                    if pY < 63:
                        p2 = torch.max(hm_[pY + 1, pX], eps)
                        x2 = x0
                        y2 = y0 + 1
                        y = (3 * sigma) ** 2 * (
                                    torch.log(p2) - torch.log(p0)) - (
                                    x0 ** 2 - x2 ** 2 + y0 ** 2 - y2 ** 2) / 2
                    else:
                        p2 = torch.max(hm_[pY - 1, pX], eps)
                        x2 = x0
                        y2 = y1 - 1
                        y = (3 * sigma) ** 2 * (
                                    torch.log(p2) - torch.log(p0)) - (
                                    x0 ** 2 - x2 ** 2 + y0 ** 2 - y2 ** 2) / 2
                else:
                    p1 = torch.max(hm_[pY, pX - 1], eps)
                    x1 = x0 - 1
                    y1 = y0
                    x = (3 * sigma) ** 2 * (torch.log(p1) - torch.log(p0)) - (
                            x0 ** 2 - x1 ** 2 + y0 ** 2 - y1 ** 2) / 2
                    if pY < 63:
                        p2 = torch.max(hm_[pY + 1, pX], eps)
                        x2 = x0
                        y2 = y0 + 1
                        y = (3 * sigma) ** 2 * (
                                    torch.log(p2) - torch.log(p0)) - (
                                    x0 ** 2 - x2 ** 2 + y0 ** 2 - y2 ** 2) / 2
                    else:
                        p2 = torch.max(hm_[pY - 1, pX])
                        x2 = x0
                        y2 = y1 - 1
                        y = (3 * sigma) ** 2 * (
                                    torch.log(p2) - torch.log(p0)) - (
                                    x0 ** 2 - x2 ** 2 + y0 ** 2 - y2 ** 2) / 2
                preds[i, j, 0] = x
                preds[i, j, 1] = y
        preds_orig = torch.zeros(preds.size())
        if center is not None and scale is not None:
            for i in range(hm.size(0)):
                for j in range(hm.size(1)):
                    preds_orig[i, j] = transform(
                        preds[i, j] + 0.5, center, scale, hm.size(2), True)
        return preds, preds_orig

    def find_landmarks(self, video_handler: cv2.VideoCapture, number_of_frames=10) -> pd.DataFrame:
        # TODO: Allow user to specify which frames to run detection on
        # define the DataFrame that will be used to store the landmark and boundingbox data
        df_cols = ["bbox_top_x", "bbox_top_y", "bbox_bottom_x", "bbox_bottom_y"]
        for i in range(0, 68):
            num = str(i)
            xx = 'landmark_' + num + '_x'
            yy = 'landmark_' + num + '_y'
            df_cols.append(xx)
            df_cols.append(yy)
        LandmarkDataFrame = pd.DataFrame(columns=df_cols)

        # get some information from video
        video_width = int(video_handler.get(cv2.CAP_PROP_FRAME_WIDTH))
        video_height = int(video_handler.get(cv2.CAP_PROP_FRAME_HEIGHT))
        video_channels = 3

        # video_length = int(video_handler.get(cv2.CAP_PROP_FRAME_COUNT))
        video_length = -1
        ret = True
        while ret:
            ret, frame = video_handler.read()
            video_length += 1
        video_handler.set(0, 0)

        if number_of_frames > video_length:
            number_of_frames = video_length

        # create the sequence of frames that will be read from file, this takes
        # into account that the last tensor might have less frames
        sequence = [number_of_frames] * (video_length // number_of_frames)
        if video_length % number_of_frames > 0:
            sequence.append(video_length % number_of_frames)

        # scale factor for face localization, 4 seems to work just fine
        scale_factor = 4
        count = 0
        total_count = video_length // number_of_frames
        for k in sequence:
            self.frame_done_signal.emit(count*number_of_frames, count/total_count)
            # st = time.time()
            data = []  # dict of dicts containing information for all frames

            # here we will stack a bunch of frames to proceess them in parallel
            framesstack = np.zeros(
                (k, video_height, video_width, video_channels))
            failed = False
            for f in range(k):
                success, frame = video_handler.read()
                if success:
                    # the first frame of the stack will be used to localize the face, the same face position will be used to all the stack
                    if f == 0:
                        frame_for_boundingbox = cv2.resize(frame, (
                        video_width // scale_factor,
                        video_height // scale_factor),
                                                           interpolation=cv2.INTER_AREA)
                        frame_for_boundingbox = frame_for_boundingbox - np.array(
                            [104, 117,
                             123])  # normalization required by the network
                        frame_for_boundingbox = frame_for_boundingbox.transpose(
                            2, 0, 1)
                        frame_for_boundingbox = frame_for_boundingbox.reshape(
                            (1,) + frame_for_boundingbox.shape)

                    framesstack[f, :, :, :] = frame
                else:
                    print('skipped frame')
                    failed = True

                    # Detect faces in the first frame of the stack
            bboxlist = self.detect_bbox(self._face_detector_net, torch.from_numpy(
                frame_for_boundingbox).float().to(self._device))
            keep = self.nms(bboxlist)
            bboxlist = bboxlist[keep, :]
            bboxlist = [x for x in bboxlist if x[-1] > 0.5]

            # find center of face and a 'scale' factor
            for i, d in enumerate(bboxlist):
                d[0:4] = d[0:4] * scale_factor
                center = torch.FloatTensor(
                    [d[2] - (d[2] - d[0]) / 2.0, d[3] - (d[3] - d[1]) / 2.0])
                center[1] = center[1] - (d[3] - d[1]) * 0.12
                scale = (d[2] - d[0] + d[3] - d[
                    1]) / self._face_detector_net.reference_scale

            # crop images so that face in centered and image dimensions as 256x256
            cropped_images = []
            for i in range(k):
                cropped_images.append(torch.tensor(
                    crop(framesstack[i, :, :, :], center.numpy(), scale)))

            img_cropped = torch.stack(cropped_images, 0)
            # permute color axis because
            # numpy image: H x W x C
            # torch image: C X H X W
            img_cropped = img_cropped.permute((0, 3, 1, 2)).float()
            img_cropped = img_cropped.to(self._device)
            img_cropped.div_(255)

            # find landmarks in cropped images --  this function returns a set of heatmaps, one for each landmark
            output = self._face_alignment_net(img_cropped)[-1]

            # retreive landmark positions from heatmaps
            pts, pts_img = get_preds_fromhm(output, center, scale)

            # save everything in a DataFrame
            for f in range(k):
                datum = []  # dict containing information for one frame

                datum.append(d[0])
                datum.append(d[1])
                datum.append(d[2])
                datum.append(d[3])

                for x, y in pts_img[f, :, :].numpy():
                    datum.append(x), datum.append(
                        y)  # x and y position of each landmark

                LandmarkDataFrame = LandmarkDataFrame.append(
                    pd.Series(datum, index=df_cols),
                    ignore_index=True)
            count += 1

        LandmarkDataFrame.insert(loc=0, column='Frame_number',
                                 value=range(video_length))

        return LandmarkDataFrame

    def stop(self) -> bool:
        self._running = False
        return True

    def run(self):
        while self._running:
            if not self._video_queue.empty():
                name, handler = self._video_queue.get()
                self.new_video_started_signal.emit(name)
                landmarks = self.find_landmarks(handler, self._num_parallel_frames)
                self.landmarks_complete_signal.emit(name, landmarks)

