from torch.utils.data import Dataset
from PIL import Image
import os
import torch
import numpy as np
from torchvision import transforms
import json
import torch.nn.functional as F

from utils import random_crop


class EvturbDataset(Dataset):
    def __init__(self, root, transform=None, mode='train', img_size=(128, 128), second_stage=False, spatial_offset=None, temporal_offset=None):
        self.root = root
        self.frame_dir = os.path.join(root, 'frame')
        self.one_dir = os.path.join(root, 'first_stage')
        self.event_dir = os.path.join(root, 'event')
        self.gt_dir = os.path.join(root, 'gt')
        self.vp_dir = os.path.join(root, 'var')
        self.mode = mode
        self.frames = []
        self.events = []
        self.gts = []
        self.vps = []
        self.img_size = img_size
        self.ss = second_stage
        self.spatial_offset = spatial_offset
        self.temporal_offset = temporal_offset

        self.load_data()

    def load_data(self):
        if self.mode == 'train':
            with open(os.path.join(self.root, 'train.json'), 'r') as f:
                sample_list = json.load(f)
        else:
            with open(os.path.join(self.root, 'test.json'), 'r') as f:
                sample_list = json.load(f)

        frame_dir = self.one_dir if self.ss else self.frame_dir
        for item in sample_list:
            self.frames.append(os.path.join(frame_dir, item + '.png'))
            self.events.append(os.path.join(self.event_dir, item + '.npz'))
            self.gts.append(os.path.join(self.gt_dir, item + '.png'))
            self.vps.append(os.path.join(self.vp_dir, item + '.npz'))

    def __len__(self):
        return len(self.frames)

    def _shift_spatial_pad(self, x, dy: int, dx: int, pad_value: float = 0.0):
        H, W = x.shape[-2], x.shape[-1]
        top = max(dy, 0)
        bottom = max(-dy, 0)
        left = max(dx, 0)
        right = max(-dx, 0)
        x = F.pad(x, (left, right, top, bottom), mode="constant", value=pad_value)
        return x[..., top:top + H, left:left + W]

    def _shift_temporal_pad(self, x, dt: int, pad_value: float = 0.0):
        if dt == 0:
            return x
        T = x.shape[0]
        if dt > 0:
            pad = x.new_full((dt, *x.shape[1:]), pad_value)
            x = torch.cat([pad, x], dim=0)
            return x[:T]
        dt = -dt
        pad = x.new_full((dt, *x.shape[1:]), pad_value)
        x = torch.cat([x, pad], dim=0)
        return x[dt:dt + T]

    def __getitem__(self, idx):
        frame = transforms.ToTensor()(Image.open(self.frames[idx]))
        gt = transforms.ToTensor()(Image.open(self.gts[idx]))
        event = torch.from_numpy(np.load(self.events[idx])['arr_0']).float()
        vp = torch.from_numpy(np.load(self.vps[idx])['arr_0']).unsqueeze(0).float()

        if self.spatial_offset is not None and self.spatial_offset != (0, 0):
            dx = int(self.spatial_offset[0])
            dy = int(self.spatial_offset[1])
            vp = self._shift_spatial_pad(vp, dy=dy, dx=dx, pad_value=0.0)
            event = self._shift_spatial_pad(event, dy=dy, dx=dx, pad_value=0.0)

        if self.temporal_offset is not None and self.temporal_offset != 0:
            event = self._shift_temporal_pad(event, dt=int(self.temporal_offset), pad_value=0.0)

        if self.mode == 'train':
            cnt = 0
            while True:
                concat = random_crop(torch.cat([frame, gt, vp, event], dim=0), self.img_size)
                frame_crop = concat[:3]
                gt_crop = concat[3:6]
                vp_crop = concat[6:7]
                event_crop = concat[7:]
                if torch.sum(abs(event_crop)) > self.img_size[0] * self.img_size[1] / 2 or cnt > 10:
                    frame = frame_crop
                    gt = gt_crop
                    event = event_crop
                    vp = vp_crop
                    break
                cnt += 1

        return {'frame': frame, 'event': event, 'gt': gt, 'vp': vp, 'filename': self.frames[idx]}
