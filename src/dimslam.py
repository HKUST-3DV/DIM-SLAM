import torch
import torch.nn as nn

from src.nerf import NeRF
from src.render import BaseRenderer
from src.sfm import DIMSfM
from src.dataset import replica

from easydict import EasyDict as edict

class DIMSLAM(nn.Module):

    def __init__(self, args) -> None:
        super().__init__()
        self.args = args
        self.args.updated_color_cam = edict()
        self.args.updated_color_cam.H = self.args.color_cam.H
        self.args.updated_color_cam.W = self.args.color_cam.W
        self.args.updated_color_cam.fx = self.args.color_cam.fx
        self.args.updated_color_cam.fy = self.args.color_cam.fy
        self.args.updated_color_cam.cx = self.args.color_cam.cx
        self.args.updated_color_cam.cy = self.args.color_cam.cy
        if self.args.color_cam.crop_size is not None:
            sH, sW = self.args.color_cam.crop_size[0] // self.args.color_cam.H, self.args.color_cam.crop_size[1] // self.args.color_cam.W
            self.args.updated_color_cam.H = self.args.color_cam.crop_size[0]
            self.args.updated_color_cam.W = self.args.color_cam.crop_size[1]
            self.args.updated_color_cam.fx = self.args.color_cam.fx * sW
            self.args.updated_color_cam.fy = self.args.color_cam.fy * sH
            self.args.updated_color_cam.cx = self.args.color_cam.cx * sW
            self.args.updated_color_cam.cy = self.args.color_cam.cy * sH
        if self.args.color_cam.crop_edge is not None:
            self.args.updated_color_cam.H -= self.args.color_cam.crop_edge[0] * 2
            self.args.updated_color_cam.W -= self.args.color_cam.crop_edge[1] * 2
            self.args.updated_color_cam.cx -= self.args.color_cam.crop_edge[1]
            self.args.updated_color_cam.cy -= self.args.color_cam.crop_edge[0]
        


        self.model = NeRF(self.args)
        self.renderer = BaseRenderer(self.args, self.model)

        self.sfm = DIMSfM(self.args, self.model, self.renderer)

        self.stream = replica(self.args)

        self.device = args.device

    def start(self):
        
        self.init(15)

        for i in range(15, len(self.stream)):
            pass

    def init(self, init_num_frames=15):
        
        indexs = []
        images = []
        depths = []
        start_poses = []
        gt_poses = []
        for i in range(init_num_frames):
            index, color_img, depth_img, gt_pose = self.stream[i]
            indexs.append(index)
            images.append(color_img)
            depths.append(depth_img)
            start_poses.append(gt_pose if i <=1 else start_poses[-1].clone())
            gt_poses.append(gt_pose)
        
        indexs = torch.tensor(indexs).long().to(self.device)
        images = torch.stack(images).to(self.device)
        depths = torch.stack(depths).to(self.device)
        start_poses = torch.stack(start_poses).to(self.device)
        gt_poses = torch.stack(gt_poses).to(self.device)

        self.sfm.ba(indexs, images, depths, start_poses, gt_poses)
    
    def build_kf_graph(self):
        pass

    def select_kf_graph(self):
        pass

