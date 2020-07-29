from PyQt5 import QtCore
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torchvision import transforms
from pytorch_lightning.core.lightning import LightningModule
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam, lr_scheduler, RMSprop
from typing import Dict, List, Union
import numpy as np
import pandas as pd
from PIL import Image


def conv3x3(in_planes, out_planes, strd=1, padding=1, bias=False):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3,
                     stride=strd, padding=padding, bias=bias)


class ConvBlock(nn.Module):
    def __init__(self, in_planes, out_planes):
        super(ConvBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = conv3x3(in_planes, int(out_planes / 2))
        self.bn2 = nn.BatchNorm2d(int(out_planes / 2))
        self.conv2 = conv3x3(int(out_planes / 2), int(out_planes / 4))
        self.bn3 = nn.BatchNorm2d(int(out_planes / 4))
        self.conv3 = conv3x3(int(out_planes / 4), int(out_planes / 4))

        if in_planes != out_planes:
            self.downsample = nn.Sequential(
                nn.BatchNorm2d(in_planes),
                nn.ReLU(True),
                nn.Conv2d(in_planes, out_planes,
                          kernel_size=1, stride=1, bias=False),
            )
        else:
            self.downsample = None

    def forward(self, x):
        residual = x

        out1 = self.bn1(x)
        out1 = F.relu(out1, True)
        out1 = self.conv1(out1)

        out2 = self.bn2(out1)
        out2 = F.relu(out2, True)
        out2 = self.conv2(out2)

        out3 = self.bn3(out2)
        out3 = F.relu(out3, True)
        out3 = self.conv3(out3)

        out3 = torch.cat((out1, out2, out3), 1)

        if self.downsample is not None:
            residual = self.downsample(residual)

        out3 += residual

        return out3


class Bottleneck(nn.Module):

    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class HourGlass(nn.Module):
    def __init__(self, num_modules, depth, num_features):
        super(HourGlass, self).__init__()
        self.num_modules = num_modules
        self.depth = depth
        self.features = num_features

        self._generate_network(self.depth)

    def _generate_network(self, level):
        self.add_module('b1_' + str(level), ConvBlock(self.features, self.features))

        self.add_module('b2_' + str(level), ConvBlock(self.features, self.features))

        if level > 1:
            self._generate_network(level - 1)
        else:
            self.add_module('b2_plus_' + str(level), ConvBlock(self.features, self.features))

        self.add_module('b3_' + str(level), ConvBlock(self.features, self.features))

    def _forward(self, level, inp):
        # Upper branch
        up1 = inp
        up1 = self._modules['b1_' + str(level)](up1)

        # Lower branch
        low1 = F.avg_pool2d(inp, 2, stride=2)
        low1 = self._modules['b2_' + str(level)](low1)

        if level > 1:
            low2 = self._forward(level - 1, low1)
        else:
            low2 = low1
            low2 = self._modules['b2_plus_' + str(level)](low2)

        low3 = low2
        low3 = self._modules['b3_' + str(level)](low3)

        up2 = F.interpolate(low3, scale_factor=2, mode='nearest')

        return up1 + up2

    def forward(self, x):
        return self._forward(self.depth, x)


class FAN(nn.Module):

    def __init__(self, num_modules=1):
        super(FAN, self).__init__()
        self.num_modules = num_modules

        # Base part
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = ConvBlock(64, 128)
        self.conv3 = ConvBlock(128, 128)
        self.conv4 = ConvBlock(128, 256)

        # Stacking part
        for hg_module in range(self.num_modules):
            self.add_module('m' + str(hg_module), HourGlass(1, 4, 256))
            self.add_module('top_m_' + str(hg_module), ConvBlock(256, 256))
            self.add_module('conv_last' + str(hg_module),
                            nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0))
            self.add_module('bn_end' + str(hg_module), nn.BatchNorm2d(256))
            self.add_module('l' + str(hg_module), nn.Conv2d(256,
                                                            68, kernel_size=1, stride=1, padding=0))

            if hg_module < self.num_modules - 1:
                self.add_module(
                    'bl' + str(hg_module), nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0))
                self.add_module('al' + str(hg_module), nn.Conv2d(68,
                                                                 256, kernel_size=1, stride=1, padding=0))

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)), True)
        x = F.avg_pool2d(self.conv2(x), 2, stride=2)
        x = self.conv3(x)
        x = self.conv4(x)

        previous = x

        outputs = []
        for i in range(self.num_modules):
            hg = self._modules['m' + str(i)](previous)

            ll = hg
            ll = self._modules['top_m_' + str(i)](ll)

            ll = F.relu(self._modules['bn_end' + str(i)]
                        (self._modules['conv_last' + str(i)](ll)), True)

            # Predict heatmaps
            tmp_out = self._modules['l' + str(i)](ll)
            outputs.append(tmp_out)

            if i < self.num_modules - 1:
                ll = self._modules['bl' + str(i)](ll)
                tmp_out_ = self._modules['al' + str(i)](tmp_out)
                previous = previous + ll + tmp_out_

        return outputs

class AdaptiveWingLoss(torch.nn.Module):
    def __init__(self, w=15, e=1):
        self.w = w
        self.e = e
        super(AdaptiveWingLoss, self).__init__()

    def forward(self, pre_hm, gt_hm):
        """Code comes from: https://github.com/tankrant/Adaptive-Wing-Loss/blob/master/ApativeWingLoss.py"""
        theta = 0.5
        alpha = 2.1
        A = self.w * (1 / (1 + torch.pow(theta / self.e, alpha - gt_hm))) * (alpha - gt_hm) * torch.pow(theta / self.e, alpha - gt_hm - 1) * (1 / self.e)
        C = (theta * A - self.w * torch.log(1 + torch.pow(theta / self.e, alpha - gt_hm)))

        # batch_size = gt_hm.size()[0]
        # hm_num = gt_hm.size()[1]
        #
        # mask = torch.zeros_like(gt_hm)
        # # self.w = 10
        # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        # for i in range(batch_size):
        #     img_list = []
        #     for j in range(hm_num):
        #         img_list.append(np.round(gt_hm[i][j].cpu().numpy() * 255))
        #     img_merge = cv2.merge(img_list)
        #     img_dilate = cv2.morphologyEx(img_merge, cv2.MORPH_DILATE, kernel)
        #     img_dilate[img_dilate < 51] = 1  # 0*self.w+1
        #     img_dilate[img_dilate >= 51] = 11  # 1*self.w+1
        #     img_dilate = np.array(img_dilate, dtype=np.int)
        #     img_dilate = img_dilate.transpose(2, 0, 1)
        #     mask[i] = torch.from_numpy(img_dilate)

        diff_hm = torch.abs(gt_hm - pre_hm)
        AWingLoss = A * diff_hm - C
        idx = diff_hm < theta
        AWingLoss[idx] = self.w * torch.log(1 + torch.pow(diff_hm / self.e, alpha - gt_hm))[idx]

        # AWingLoss *= mask
        mean_loss = torch.mean(AWingLoss)
        # all_pixel = torch.sum(mask)
        # mean_loss = sum_loss / all_pixel

        return mean_loss

def _gaussian_fast(size=3, sigma=0.25, amplitude=1., offset=[0., 0.], device='cpu'):
    coordinates = torch.stack(torch.meshgrid(torch.arange(-size // 2 + 1. -offset[0], size // 2 + 1. -offset[0], step=1),
                                   torch.arange(-size // 2 + 1. -offset[1], size // 2 + 1. -offset[1], step=1))).to(device)
    coordinates = coordinates / (sigma * size)
    gauss = amplitude * torch.exp(-(coordinates**2 / 2).sum(dim=0))
    return gauss.permute(1, 0)

def draw_gaussian(image, point, sigma, offset=False):
    # Check if the gaussian is inside
    ul = [math.floor(point[0] - 3 * sigma), math.floor(point[1] - 3 * sigma)]
    br = [math.floor(point[0] + 3 * sigma), math.floor(point[1] + 3 * sigma)]
    if (ul[0] > image.shape[1] or ul[1] > image.shape[0] or br[0] < 1 or br[1] < 1):
        return image
    size = 6 * sigma + 1
    # g = torch.Tensor(_gaussian(size, offset=point%1) if offset else _gaussian(size), device=image.device)
    g = _gaussian_fast(size, offset=point%1, device=image.device) if offset else _gaussian_fast(size, device=image.device)
    g_x = [int(max(1, -ul[0])), int(min(br[0], image.shape[1])) - int(max(1, ul[0])) + int(max(1, -ul[0]))]
    g_y = [int(max(1, -ul[1])), int(min(br[1], image.shape[0])) - int(max(1, ul[1])) + int(max(1, -ul[1]))]
    img_x = [int(max(1, ul[0])), int(min(br[0], image.shape[1]))]
    img_y = [int(max(1, ul[1])), int(min(br[1], image.shape[0]))]
    assert (g_x[0] > 0 and g_y[1] > 0)
    image[img_y[0] - 1:img_y[1], img_x[0] - 1:img_x[1]
          ] = image[img_y[0] - 1:img_y[1], img_x[0] - 1:img_x[1]] + g[g_y[0] - 1:g_y[1], g_x[0] - 1:g_x[1]]
    image[image > 1] = 1
    return image

class FaceLandmarksDataset(Dataset):
    """Facial landmarks dataset
        input: CSV file
        output : Image, landmarks, angles, boundingbox
    """
    img_map: Dict[str, pd.DataFrame]
    img_list: List[str]

    def _get_landmarks(self, df: pd.DataFrame):
        landmarks = df[[col for col in df.columns if "landmark_" in col]].to_numpy()[0].reshape(-1, 2)
        # landmarks = [(landmarks[2 * i], landmarks[2 * i + 1]) for i in range(len(landmarks) // 2)]
        return np.array([landmarks])

    def __init__(self, img_map: Dict[str, str], transforms=None):
        self.transforms = transforms

        self.img_map = {img: self._get_landmarks(pd.read_csv(csv)) for img, csv in img_map.items()}
        self.img_list = list(self.img_map.keys())

    def __len__(self):
        return len(self.img_list)

    def create_bounding_box(self, target_landmarks, expansion_factor=0.0):
        """
        gets a batch of landmarks and calculates a bounding box that includes all the landmarks per set of landmarks in
        the batch
        :param target_landmarks: batch of landmarks of dim (n x 68 x 2). Where n is the batch size
        :param expansion_factor: expands the bounding box by this factor. For example, a `expansion_factor` of 0.2 leads
        to 20% increase in width and height of the boxes
        :return: a batch of bounding boxes of dim (n x 4) where the second dim is ('face_x', 'face_y', 'face_width', 'face_height')
        """
        n_landmarks = target_landmarks.shape[1]
        target_landmarks = torch.Tensor(target_landmarks)
        # Calc bounding box
        x_y_min, _ = target_landmarks.reshape(-1, n_landmarks, 2).min(dim=1)
        x_y_max, _ = target_landmarks.reshape(-1, n_landmarks, 2).max(dim=1)
        # expanding the bounding box
        expansion_factor /= 2
        bb_expansion_x = (x_y_max[:, 0] - x_y_min[:, 0]) * expansion_factor
        bb_expansion_y = (x_y_max[:, 1] - x_y_min[:, 1]) * expansion_factor
        x_y_min[:, 0] -= bb_expansion_x
        x_y_max[:, 0] += bb_expansion_x
        x_y_min[:, 1] -= bb_expansion_y
        x_y_max[:, 1] += bb_expansion_y
        # centers = ((x_y_max - x_y_min) / 2) + x_y_min
        # distances = x_y_max - x_y_min
        return torch.cat([x_y_min, x_y_max], dim=1).numpy()

    def __getitem__(self, idx):
        img_name = self.img_list[idx]
        image = Image.open(img_name)

        landmarks = self.img_map[img_name]

        bounding_box = self.create_bounding_box(landmarks, expansion_factor=0.0)

        sample = {"image": np.array(image), "bounding_box": bounding_box[0], "landmarks": landmarks[0]}
        if self.transforms is not None:
            sample = self.transforms(sample)

        return sample


class CropToBoundingbox(object):
    """
    Crop and image and adjust its size
    """

    def __init__(self, resolution=256):
        self.resolution = resolution

    def __call__(self, sample):

        image, bb, landmarks = sample['image'], sample['bounding_box'], sample['landmarks']

        # cropping the image
        # this was adapted from Adrian Bulat's API <https://github.com/1adrianb/face-alignment>
        reference_scale = 195
        center = [bb[0] + (bb[2] / 2.0), bb[1] + (bb[3] / 2.0)]
        center = center - (bb[3] * 0.12)  # Not sure where 0.12 comes from
        scale = (bb[2] + bb[3]) / reference_scale

        cropped_image = self.crop(image, center, scale, resolution=self.resolution)
        heatmaps = self.create_target_heatmap(landmarks, center, scale, subpixel=True)

        for n, p in enumerate(landmarks):
            if (p[0] <= 0) or (p[1] <= 0):
                landmarks[n, :] = p
            else:
                temp = self.transform(p, center, scale, self.resolution, invert=False, integer=False)
                landmarks[n, :] = temp.numpy()

        return {"image": cropped_image, "heatmaps": heatmaps.numpy(), "bounding_box": bb, "landmarks": landmarks}

    def create_target_heatmap(self, target_landmarks, centers, scales, subpixel=False):
        """
        Receives a batch of landmarks and returns a set of heatmaps for each set of 68 landmarks in the batch
        :param target_landmarks: the batch is expected to have the dim (n x 68 x 2). Where n is the batch size
        :param subpixel: whether to create subpixel heatmaps or floor the coordinates
        :return: returns a (n x 68 x 64 x 64) batch of heatmaps
        """
        n_landmarks = target_landmarks.shape[0]
        heatmaps = torch.zeros((n_landmarks, 64, 64))
        for p in range(n_landmarks):
            landmark_cropped_coor = self.transform(target_landmarks[p], centers, scales, 64, invert=False, integer=not subpixel)
            landmark_cropped_coor = landmark_cropped_coor + 0.5 if subpixel else landmark_cropped_coor + 1
            heatmaps[p] = draw_gaussian(heatmaps[p], landmark_cropped_coor, 1, offset=subpixel)
        return heatmaps

    def crop(self, image, center, scale, resolution):
        """Center crops an image or set of heatmaps
        Note: Tried moving this to GPU, but realized it doesn't make sense.
        Arguments:
            image {numpy.array} -- an rgb image
            center {numpy.array} -- the center of the object, usually the same as of the bounding box
            scale {float} -- scale of the face

        Keyword Arguments:
            resolution {float} -- the size of the output cropped image (default: {256.0})

        Returns:
            [type] -- [description]
        """  # Crop around the center point
        """ Crops the image around the center. Input is expected to be an np.ndarray """

        ul = self.transform([1, 1], center, scale, resolution, True)
        br = self.transform([resolution, resolution], center, scale, resolution, True)
        # pad = math.ceil(torch.norm((ul - br).float()) / 2.0 - (br[0] - ul[0]) / 2.0)
        if image.ndim > 2:
            newDim = np.array([br[1] - ul[1], br[0] - ul[0], image.shape[2]], dtype=np.int32)
            newImg = np.zeros(newDim, dtype=np.uint8)
        else:
            newDim = np.array([br[1] - ul[1], br[0] - ul[0]], dtype=np.int)
            newImg = np.zeros(newDim, dtype=np.uint8)
        ht = image.shape[0]
        wd = image.shape[1]
        newX = np.array([max(0, -ul[0]), min(br[0], wd) - ul[0]], dtype=np.int32)
        newY = np.array([max(0, -ul[1]), min(br[1], ht) - ul[1]], dtype=np.int32)
        oldX = np.array([max(0, ul[0]), min(br[0], wd)], dtype=np.int32)
        oldY = np.array([max(0, ul[1]), min(br[1], ht)], dtype=np.int32)
        newImg[newY[0]:newY[1], newX[0]:newX[1]] = image[oldY[0]:oldY[1], oldX[0]:oldX[1], :]

        newImg = np.array(Image.fromarray(newImg).resize((int(resolution), int(resolution)), Image.ANTIALIAS))
        # newImg = cv2.resize(newImg, dsize=(int(resolution), int(resolution)), interpolation=cv2.INTER_LINEAR)
        return newImg

    def transform(self, point, center, scale, resolution, invert=False, integer=True):
        """Generate and affine transformation matrix.

        Given a set of points, a center, a scale and a targer resolution, the
        function generates and affine transformation matrix. If invert is ``True``
        it will produce the inverse transformation.

        Arguments:
            point {torch.tensor} -- the input 2D point
            center {torch.tensor or numpy.array} -- the center around which to perform the transformations
            scale {float} -- the scale of the face/object
            resolution {float} -- the output resolution

        Keyword Arguments:
            invert {bool} -- define wherever the function should produce the direct or the
            inverse transformation matrix (default: {False})
        """

        _pt = torch.ones(3)
        _pt[0] = point[0]
        _pt[1] = point[1]

        h = 200.0 * scale
        t = torch.eye(3)
        t[0, 0] = resolution / h
        t[1, 1] = resolution / h
        t[0, 2] = resolution * (-center[0] / h + 0.5)
        t[1, 2] = resolution * (-center[1] / h + 0.5)

        if invert:
            t = torch.inverse(t)

        new_point = (torch.matmul(t, _pt))[0:2]

        return new_point.int() if integer else new_point


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, heatmaps, bounding_box, landmarks = sample['image'], sample['heatmaps'], sample['bounding_box'], sample['landmarks']

        return {'image': transforms.ToTensor()(image),
                'heatmaps': torch.from_numpy(heatmaps),
                "bounding_box": torch.from_numpy(bounding_box),
                'landmarks': torch.from_numpy(landmarks)}


class Normalize(object):
    """Normalize an tensor image with mean and standard deviation.

    Given mean: (R, G, B) and std: (R, G, B),
    will normalize each channel of the torch.*Tensor, i.e.
    channel = (channel - mean) / std

    Args:
        mean (sequence): Sequence of means for R, G, B channels respecitvely.
        std (sequence): Sequence of standard deviations for R, G, B channels
            respecitvely.
    """

    def __init__(self, mean, std, inplace=False):
        self.mean = mean
        self.std = std
        self.inplace = inplace

    def __call__(self, sample):
        image, heatmaps, bounding_box, landmarks = sample['image'], sample['heatmaps'], sample['bounding_box'], sample['landmarks']

        return {'image': transforms.Normalize(self.mean, self.std)(image),
                'heatmaps': heatmaps,
                "bounding_box": bounding_box,
                'landmarks': landmarks}


def de_norm(tensor, mean, std):
    """
    function that gets a normalized tensor and returns a PIL image
    """
    npimg = np.transpose(tensor.numpy(), (1, 2, 0))
    npimg[:, :, 0] = npimg[:, :, 0] * std[0] + mean[0]
    npimg[:, :, 1] = npimg[:, :, 1] * std[1] + mean[1]
    npimg[:, :, 2] = npimg[:, :, 2] * std[2] + mean[2]
    return npimg

class LightningFAN(LightningModule):
    def __init__(self, training_map=None, validation_map=None, test_map=None, learning_rate=1e-5, batch_size=1, FAN_path=None):
        # FAN.__init__(self, num_modules=4)
        LightningModule.__init__(self)
        if training_map is None:
            training_map = {}
        if validation_map is None:
            validation_map = {}
        if test_map is None:
            test_map = {}
        self._fan = FAN(num_modules=4)
        if FAN_path:
            self.load_FAN_weights(self._fan, FAN_path)
        # Currently, the autofinders for batch size and learning rate seem a bit buggy. The batch size one requires that
        # self.batch_size is set even though the docs show self.hparams.batch_size is the one it needs. The learning
        # rate one requires that self.hparams.learning_rate is set even though the docs say either that or
        # self.learning_rate works. I just do both for now.
        self.hparams.batch_size = batch_size
        self.batch_size = batch_size
        self.hparams.learning_rate = learning_rate
        self.learning_rate = learning_rate
        self.train_map = training_map
        self.val_map = validation_map
        self.test_map = test_map

        # self.criterion = AdaptiveWingLoss(15, 1)
        self.criterion = torch.nn.MSELoss()

        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        self.comp_transforms = transforms.Compose([
            CropToBoundingbox(resolution=256),
            ToTensor(),
            Normalize(mean=mean, std=std)
        ])

    def load_FAN_weights(self, model: FAN, filename: str) -> bool:
        sd = torch.load(filename, map_location=lambda storage, loc: storage)
        names = set(model.state_dict().keys())
        for n in list(sd.keys()):
            if n not in names and n + '_raw' in names:
                if n + '_raw' not in sd:
                    sd[n + '_raw'] = sd[n]
                del sd[n]
        model.load_state_dict(sd)
        return True

    def train_freeze(self):
        """
        Freeze all but the last hourglass for training.
        """
        for i, param in enumerate(self._fan.parameters()):
            if i < 168:
                param.requires_grad = False
            param.requires_grad = False
        # trainable = [
        #     self._fan.l3, self._fan.m3, self._fan.top_m_3, self._fan.conv_last3, self._fan.bn_end3
        # ]
        trainable = [
            self._fan.l3, self._fan.conv_last3, self._fan.bn_end3
        ]
        for module in trainable:
            for param in module.parameters():
                if not param.requires_grad:
                    print("Unfreezing param")
                param.requires_grad = True
        for i, param in enumerate(self._fan.parameters()):
            print(f"{i}: {param.requires_grad}")

        for key, val in self._fan.__dict__["_modules"].items():
            print("Key: ", key)
            try:
                print("Weight Requires Grad: ", val.weight.requires_grad)
            except Exception:
                pass
            try:
                print("Weight Grad Function: ", val.weight.grad_fn, val.weight.grad)
            except Exception:
                pass
            try:
                print("Bias Requires Grad: ", val.bias.requires_grad)
            except Exception:
                pass
            try:
                print("Bias Grad Function: ", val.bias.grad_fn, val.bias.grad)
            except Exception:
                pass
            print()


    def forward(self, x):
        return self._fan.forward(x)

    def train_dataloader(self) -> DataLoader:
        print("Training with batch size:", self.hparams.batch_size, self.batch_size)
        return DataLoader(FaceLandmarksDataset(self.train_map, self.comp_transforms), batch_size=self.batch_size)

    def training_step(self, batch, batch_idx):
        imgs, heatmaps, bb, landmarks = batch['image'], batch['heatmaps'], batch['bounding_box'], batch['landmarks']
        y_hat = self.forward(imgs)[0]
        # loss = torch.nn.MSELoss()(heatmaps.float(), y_hat.float())
        loss = self.criterion(heatmaps.float(), y_hat.float())

        tensorboard_logs = {'train_loss': loss}
        return {'loss': loss, 'log': tensorboard_logs}

    def val_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(FaceLandmarksDataset(self.val_map, self.comp_transforms), batch_size=self.batch_size)

    def validation_step(self, batch, batch_idx):
        imgs, heatmaps, bb, landmarks = batch['image'], batch['heatmaps'], batch['bounding_box'], batch['landmarks']
        y_hat = self.forward(imgs)[0]
        # loss = torch.nn.MSELoss()(heatmaps.float(), y_hat.float())
        loss = self.criterion(heatmaps.float(), y_hat.float())
        # Try a cross entropy loss

        tensorboard_logs = {'validation_loss': loss}
        return {'loss': loss, 'log': tensorboard_logs}

    def test_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(FaceLandmarksDataset(self.test_map, self.comp_transforms), batch_size=self.batch_size)

    def test_step(self, batch, batch_idx):
        imgs, heatmaps, bb, landmarks = batch['image'], batch['heatmaps'], batch['bounding_box'], batch['landmarks']
        y_hat = self.forward(imgs)[0]
        # loss = torch.nn.MSELoss()(heatmaps.float(), y_hat.float())
        loss = self.criterion(heatmaps.float(), y_hat.float())
        # Try a cross entropy loss

        tensorboard_logs = {'validation_loss': loss}
        return {'loss': loss, 'log': tensorboard_logs}

    def configure_optimizers(self):
        print("Optimizing with lr of:", self.hparams.learning_rate, self.learning_rate)
        print("Using parameters that require grad: ", len(list(filter(lambda p: p.requires_grad, self.parameters()))))
        optimizer = RMSprop(self._fan.parameters(), lr=self.hparams.learning_rate, weight_decay=0.0)
        # scheduler = lr_scheduler.StepLR(optimizer, step_size=1)
        return optimizer


class ResNetDepth(nn.Module):

    def __init__(self, block=Bottleneck, layers=[3, 8, 36, 3], num_classes=68):
        self.inplanes = 64
        super(ResNetDepth, self).__init__()
        self.conv1 = nn.Conv2d(3 + 68, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x
