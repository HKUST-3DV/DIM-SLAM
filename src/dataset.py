import os
import glob

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


class replica(Dataset):
    def __init__(self, args) -> None:
        super().__init__()
        self.args = args
        self.interval = args.dataset.ds_interval
        self.start = args.dataset.ds_start
        self.device = args.device
        self.scale = args.scale

        self.depth_scale = args.dataset.depth_scale

        self.crop_size = args.color_cam.crop_size
        self.crop_edge = args.color_cam.crop_edge

        self.input_folder = args.dataset.input_folder
        self.poses = []
        with open(os.path.join(self.input_folder, "traj.txt"), "r") as fin:
            for line in fin.readlines():
                c2w = np.array(list(map(float, line.split()))).reshape(4, 4)
                c2w[:3, 1] *= -1  # use the same representation of NICE-SLAM
                c2w[:3, 2] *= -1
                c2w = torch.from_numpy(c2w).float()
                self.poses.append(c2w)

        self.input_folder = os.path.join(self.input_folder, "results")
        self.color_paths = sorted(
            glob.glob(os.path.join(self.input_folder, "frame*.jpg"))
        )
        self.depth_paths = sorted(
            glob.glob(os.path.join(self.input_folder, "depth*.png"))
        )
        self.n_imgs = len(self.color_paths)


    def __len__(self):
        return (self.n_imgs - self.start) // self.interval

    def __getitem__(self, index):
        target_idx = index * self.interval + self.start
        color_path = self.color_paths[target_idx]
        depth_path = self.depth_paths[target_idx]
        color_img = torch.from_numpy(
            cv2.cvtColor(cv2.imread(color_path), cv2.COLOR_BGR2RGB) / 255.0
        ).to(torch.float32)
        depth_img = torch.from_numpy(
            cv2.imread(depth_path, cv2.IMREAD_UNCHANGED) / self.depth_scale * self.scale
        ).to(torch.float32)

        if self.crop_size is not None:
            color_img = (
                F.interpolate(
                    color_img.permute(2, 0, 1)[None].contiguous(),
                    self.crop_size,
                    mode="bilinear",
                    align_corners=True,
                )[0]
                .permute(1, 2, 0)
                .contiguous()
            )
            depth_img = F.interpolate(
                depth_img[None, None].contiguous(), self.crop_size, mode="nearest"
            )[0, 0].contiguous()
        if self.crop_edge is not None:
            color_img = color_img[
                self.crop_edge : -self.crop_edge, self.crop_edge : -self.crop_edge
            ]
            depth_img = depth_img[
                self.crop_edge : -self.crop_edge, self.crop_edge : -self.crop_edge
            ]

        c2w = self.poses[target_idx].clone()
        c2w[:3, 3] *= self.scale

        return (
            index,
            color_img.to(self.device),
            depth_img.to(self.device),
            c2w.to(self.device),
        )
