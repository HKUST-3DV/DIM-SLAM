import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import os

from matplotlib import pyplot as plt

from src.loss import SSIM
from src.utils import *
from src.utils import evaluate as eval_ate


class DIMSfM(nn.Module):
    def __init__(self, args, model, renderer) -> None:
        super().__init__()

        self.args = args
        self.model = model
        self.renderer = renderer
        self.H, self.W, self.fx, self.fy, self.cx, self.cy = (
            args.updated_color_cam.H,
            args.updated_color_cam.W,
            args.updated_color_cam.fx,
            args.updated_color_cam.fy,
            args.updated_color_cam.cx,
            args.updated_color_cam.cy,
        )

        self.device = args.device
        self.pixels = args.sfm.pixels

        self.patch_size = 11

        self.ssim_loss_5 = SSIM(5).to(self.device)
        self.ssim_loss_3 = SSIM(3).to(self.device)

    def ba(self, indexs, images, depths, start_pose, gt_pose, iters=1500, init=True):
        H, W, fx, fy, cx, cy = self.H, self.W, self.fx, self.fy, self.cx, self.cy
        cam_last_row = torch.tensor([0, 0, 0, 1]).float().to(self.device)

        fix_cam = sorted(indexs)[:2]

        grid_var_list = self.model.grids_feat
        decoder_param = [
            {"params": self.model.alpha_decoder.parameters(), "lr": 0.0001},
            {"params": self.model.color_decoder.parameters(), "lr": 0.0001},
        ]

        cam_var_list = []
        for index, pose in zip(indexs, start_pose):
            cam_var = get_tensor_from_camera(pose.detach().cpu())
            cam_var = Variable(cam_var.to(self.device), requires_grad=True)
            cam_var_list.append(cam_var)

        optim_list = [{"params": gf, "lr": 0} for gf in self.model.grids_feat]
        optim_list.append({"params": cam_var_list, "lr": 0})
        optim_list += decoder_param
        optimizer = torch.optim.AdamW(optim_list)

        n_img = len(indexs)
        expand_scale = self.args.sfm.pixels // n_img
        pixels_num = self.args.sfm.pixels // n_img * n_img

        for opt_iter in range(iters):
            for ii in range(len(self.model.grids_feat)):
                # optimizer.param_groups[int(ii)]["lr"] = ((self.args.sfm.lr["grid_%d"%int(ii)] * opt_iter + 0.5 *self.args.sfm.lr["grid_%d"%int(ii)]* (iters - opt_iter))/iters) if opt_iter >= 150 else self.args.sfm.lr["grid_%d"%int(ii)]
                optimizer.param_groups[int(ii)]["lr"] = self.args.sfm.lr["grid_%d"%int(ii)] 
            optimizer.param_groups[-3]["lr"] = (
                (
                    (
                        self.args.sfm.lr.BA_cam / 2 * (iters - opt_iter)
                        + self.args.sfm.lr.BA_cam  * (opt_iter)
                    )
                    / iters
                )
                if opt_iter >= 150
                else 0.0
            )
            optimizer.param_groups[-1]["lr"] = self.args.sfm.lr.decoder
            optimizer.param_groups[-2]["lr"] = self.args.sfm.lr.decoder

            optimizer.zero_grad()

            cur_c2ws = []
            expand_current_c2w = []
            expand_indexs = []
            frame_indexs = []
            for index, pose, s_pose in zip(indexs, cam_var_list, start_pose):
                c2w = s_pose if index in fix_cam else get_camera_from_tensor(pose)
                if c2w.shape[0] == 3:
                    c2w = torch.cat([c2w, cam_last_row.view(1, 4)], dim=0)

                cur_c2ws.append(c2w)
                expand_current_c2w.append(c2w.unsqueeze(0).repeat(expand_scale, 1, 1))
                expand_indexs.append(
                    torch.ones(expand_scale, dtype=torch.long).to(self.device) * index
                )
                frame_indexs.append(index)
            cur_c2ws = torch.stack(cur_c2ws, dim=0)
            expand_current_c2w = torch.cat(expand_current_c2w, dim=0)
            expand_indexs = torch.cat(expand_indexs, dim=0)
            frame_indexs = torch.tensor(frame_indexs).long().to(self.device)
            (
                batch_rays_o,
                batch_rays_d,
                batch_gt_depth,
                batch_gt_color,
                batch_gt_uv,
            ) = get_samples_batch(
                10,
                H - 10,
                10,
                W - 10,
                pixels_num,
                H,
                W,
                fx,
                fy,
                cx,
                cy,
                expand_current_c2w,
                expand_indexs,
                depths,
                images,
                self.device,
                return_uv=True,
            )

            batch_patch_uv = (
                batch_gt_uv.clone()
                .view(1, *batch_gt_uv.shape)
                .repeat(self.patch_size * self.patch_size, 1, 1)
            )  # 121 pp 2
            offset_kernel = (
                torch.stack(
                    torch.meshgrid(
                        torch.arange(0, self.patch_size) - self.patch_size // 2,
                        torch.arange(0, self.patch_size) - self.patch_size // 2,
                    ),
                    dim=-1,
                )
                .view(self.patch_size * self.patch_size, 1, 2)
                .repeat(1, batch_patch_uv.shape[1], 1)
                .to(self.device)
            )

            batch_patch_uv = batch_patch_uv + offset_kernel
            batch_patch_uv[:, :, 0] = batch_patch_uv[:, :, 0] / W * 2 - 1.0
            batch_patch_uv[:, :, 1] = batch_patch_uv[:, :, 1] / H * 2 - 1.0

            (
                batch_patch_rays_o,
                batch_patch_rays_d,
                batch_patch_gt_depth,
                batch_patch_gt_color,
            ) = get_samples_by_indices_batch(
                0,
                H,
                0,
                W,
                H,
                W,
                fx,
                fy,
                cx,
                cy,
                expand_current_c2w,
                expand_indexs,
                depths,
                images,
                batch_patch_uv,
                self.device,
            )
            batch_patch_gt_depth = batch_patch_gt_depth.view(
                self.patch_size * self.patch_size, -1
            )
            batch_patch_gt_color = batch_patch_gt_color.view(
                self.patch_size * self.patch_size, -1, 3
            )

            with torch.no_grad():
                det_rays_o = batch_rays_o.clone().detach().unsqueeze(-1)  # (N, 3, 1)
                det_rays_d = batch_rays_d.clone().detach().unsqueeze(-1)  # (N, 3, 1)
                t = (
                    self.renderer.bound.unsqueeze(0).to(self.device) - det_rays_o
                ) / det_rays_d
                t, _ = torch.min(torch.max(t, dim=2)[0], dim=1)

                inside_mask = t >= 0

            batch_patch_uv = batch_patch_uv[:, inside_mask, :]
            batch_patch_rays_o = batch_patch_rays_o[:, inside_mask, :]
            batch_patch_rays_d = batch_patch_rays_d[:, inside_mask, :]
            batch_patch_gt_depth = batch_patch_gt_depth[:, inside_mask]
            batch_patch_gt_color = batch_patch_gt_color[:, inside_mask, :]

            batch_rays_d = batch_rays_d[inside_mask]
            batch_rays_o = batch_rays_o[inside_mask]
            batch_gt_depth = batch_gt_depth[inside_mask]
            batch_gt_color = batch_gt_color[inside_mask]

            ret = self.renderer.render_batch_ray(
                batch_rays_d,
                batch_rays_o,
                self.device,
                gt_depth=False,
                depth_max=10,
                detach_color_weight=True,
            )
            depth, uncertainty, color, extra_ret, density = ret

            loss = 0.0

            # depth loss
            depth_mask = batch_gt_depth > 0
            depth_loss = torch.nn.functional.smooth_l1_loss(
                batch_gt_depth[depth_mask].flatten(),
                depth[depth_mask].flatten(),
                beta=0.1,
                reduction="sum",
            )

            fake_depth_loss = torch.nn.functional.smooth_l1_loss(
                depth, torch.ones_like(depth) * 1.5, beta=0.1, reduction="sum"
            )
            if init and opt_iter < 150:
                loss += fake_depth_loss

            patch_loss = 0

            patch_3d_pts = (
                batch_patch_rays_o + batch_patch_rays_d * depth[None, :, None]
            )
            patch_3d_pts = patch_3d_pts.float()

            uv, z = project_point3d_to_image_batch(
                cur_c2ws, patch_3d_pts.view(-1, 3, 1), fx, fy, cx, cy, self.device
            )
            edge = 5
            uv = uv.view(
                patch_3d_pts.shape[0], patch_3d_pts.shape[1], cur_c2ws.shape[0], 2
            )  # [ws*ws, pn, Cn, 2]

            mask = (
                (uv[(self.patch_size * self.patch_size) // 2, :, :, 0] < W - edge)
                * (uv[(self.patch_size * self.patch_size) // 2, :, :, 0] > edge)
                * (uv[(self.patch_size * self.patch_size) // 2, :, :, 1] < H - edge)
                * (uv[(self.patch_size * self.patch_size) // 2, :, :, 1] > edge)
            )  # [Pn, Cn]
            mask = mask & (
                z.view(
                    patch_3d_pts.shape[0], patch_3d_pts.shape[1], cur_c2ws.shape[0], 1
                )[(self.patch_size * self.patch_size) // 2, :, :, 0]
                < 0
            )
            mask = mask & (frame_indexs[None, :] != expand_indexs[:, None])
            mask[mask.sum(dim=1) < 4] = False
            pixel_weight = [50 if x in fix_cam else 1  for x in expand_indexs]
            frame_weight = [50 if x in fix_cam else 1  for x in frame_indexs]
            fix_cam_weight = torch.tensor(frame_weight)[None, :].float().to(self.device) * torch.tensor(pixel_weight)[:,None].float().to(self.device)
            windows_reproj_idx = uv.permute(2, 1, 0, 3)  # Cn, pn,sz*sz,, 2
            windows_reproj_idx[..., 0] = windows_reproj_idx[..., 0] / W * 2.0 - 1.0
            windows_reproj_idx[..., 1] = windows_reproj_idx[..., 1] / H * 2.0 - 1.0
            windows_reproj_gt_color = torch.nn.functional.grid_sample(
                images.permute(0, 3, 1, 2).float(),
                windows_reproj_idx,
                padding_mode="border",
            ).permute(
                2, 0, 3, 1
            )  # [Pn, cn, sz*sz, 3]
            tmp_windows_reproj_gt_color = windows_reproj_gt_color
            tmp_batch_gt_color = batch_patch_gt_color.permute(1, 0, 2)

            patch_loss += (
                0.2
                * (self.ssim_loss_5(tmp_windows_reproj_gt_color, tmp_batch_gt_color)*fix_cam_weight)[
                    mask
                ].sum()
            )

            tmp_reproj_gt_color = (
                windows_reproj_gt_color.view(
                    windows_reproj_gt_color.shape[0],
                    windows_reproj_gt_color.shape[1],
                    self.patch_size,
                    self.patch_size,
                    3,
                )[:, :, 2 : self.patch_size - 2, 2 : self.patch_size - 2, :]
                .contiguous()
                .view(
                    windows_reproj_gt_color.shape[0],
                    windows_reproj_gt_color.shape[1],
                    (self.patch_size - 4) * (self.patch_size - 4),
                    3,
                )
            )
            tmp_batch_patch_gt_color = (
                batch_patch_gt_color.permute(1, 0, 2)
                .view(
                    windows_reproj_gt_color.shape[0],
                    self.patch_size,
                    self.patch_size,
                    -1,
                )[:, 2 : self.patch_size - 2, 2 : self.patch_size - 2, :]
                .contiguous()
                .view(
                    tmp_reproj_gt_color.shape[0],
                    (self.patch_size - 4) * (self.patch_size - 4),
                    -1,
                )
            )

            patch_loss += (
                0.6
                * (self.ssim_loss_3(tmp_reproj_gt_color, tmp_batch_patch_gt_color)*fix_cam_weight)[
                    mask
                ].sum()
            )

            tmp_windows_reproj_gt_color = windows_reproj_gt_color[:,:,(self.patch_size*self.patch_size)//2,:]
            tmp_batch_gt_color = batch_patch_gt_color.permute(1,0,2)[:,(self.patch_size*self.patch_size)//2,:]
                
            forward_reproj_loss = torch.nn.functional.smooth_l1_loss(tmp_windows_reproj_gt_color[mask],
                        tmp_batch_gt_color.unsqueeze(1).repeat(1,cur_c2ws.shape[0],1)[mask], beta=0.1,reduction="sum") * 1.0
            loss += forward_reproj_loss * 0.1

            loss += (
                patch_loss.sum()
                * (0.5 if (init and opt_iter > 150) or (not init) else 0)
                * 1.0
            )

            color_loss = torch.nn.functional.smooth_l1_loss(
                batch_gt_color.flatten(), color.flatten(), beta=0.1, reduction="sum"
            )
            loss += color_loss * 0.1 if opt_iter > 800 else 0.0

            if opt_iter % 55 == 0:
                print(
                    "iters: %d, color_loss %.3f depth_loss %.3f fake_depth: %.3f patch_loss %.3f "
                    % (opt_iter, color_loss, depth_loss, fake_depth_loss, patch_loss)
                )

                cur_c2ws_d = cur_c2ws.detach().clone()
                cur_c2ws_d = torch.stack(
                    [get_tensor_from_camera(c2w, True, "quat") for c2w in cur_c2ws_d]
                )
                gt_pose_d = gt_pose.detach().clone()
                gt_pose_d = torch.stack(
                    [get_tensor_from_camera(c2w, True, "quat") for c2w in gt_pose_d]
                )
                plot_path = os.path.join(self.args.output, "keyframe_ape.jpg")
                os.makedirs(self.args.output, exist_ok=True)
                res = eval_ate(gt_pose_d, cur_c2ws_d, plot_path)
                print(res)

                idx = -1
                depth, uncertainty, color = self.renderer.render_img(
                    cur_c2ws[idx].detach().clone(),
                    self.device,
                )
                depth_np = depth.detach().cpu().numpy()
                color_np = color.detach().cpu().numpy()
                gt_depth_np = depths[idx].detach().cpu().numpy()
                gt_color_np = images[idx].detach().cpu().numpy()
                depth_residual = np.abs(gt_depth_np - depth_np)
                depth_residual[gt_depth_np == 0.0] = 0.0
                color_residual = np.abs(gt_color_np - color_np)
                color_residual[gt_depth_np == 0.0] = 0.0

                fig, axs = plt.subplots(2, 3)
                fig.tight_layout()
                max_depth = np.max(gt_depth_np)
                max_depth = 2 if max_depth == 0 else max_depth
                axs[0, 0].imshow(gt_depth_np, cmap="plasma",
                                 vmin=0, vmax=max_depth)
                axs[0, 0].set_title('Input Depth')
                axs[0, 0].set_xticks([])
                axs[0, 0].set_yticks([])
                axs[0, 1].imshow(depth_np, cmap="plasma",
                                 vmin=0, vmax=max_depth)
                axs[0, 1].set_title('Generated Depth')
                axs[0, 1].set_xticks([])
                axs[0, 1].set_yticks([])
                axs[0, 2].imshow(depth_residual, cmap="plasma",
                                 vmin=0, vmax=max_depth)
                axs[0, 2].set_title('Depth Residual')
                axs[0, 2].set_xticks([])
                axs[0, 2].set_yticks([])
                gt_color_np = np.clip(gt_color_np, 0, 1)
                color_np = np.clip(color_np, 0, 1)
                color_residual = np.clip(color_residual, 0, 1)
                axs[1, 0].imshow(gt_color_np, cmap="plasma")
                axs[1, 0].set_title('Input RGB')
                axs[1, 0].set_xticks([])
                axs[1, 0].set_yticks([])
                axs[1, 1].imshow(color_np, cmap="plasma")
                axs[1, 1].set_title('Generated RGB')
                axs[1, 1].set_xticks([])
                axs[1, 1].set_yticks([])
                axs[1, 2].imshow(color_residual, cmap="plasma")
                axs[1, 2].set_title('RGB Residual')
                axs[1, 2].set_xticks([])
                axs[1, 2].set_yticks([])
                plt.subplots_adjust(wspace=0, hspace=0)
                plt.savefig(
                    f'{self.args.output}/{opt_iter:04d}.jpg', bbox_inches='tight', pad_inches=0.2)
                plt.clf()

            loss.backward()
            optimizer.step()

        end_c2ws = []
        for index, pose, s_pose in zip(indexs, cam_var_list, start_pose):
            c2w = (
                s_pose
                if index in fix_cam
                else get_camera_from_tensor(pose).detach().clone()
            )
            if c2w.shape[0] == 3:
                c2w = torch.cat([c2w, cam_last_row.view(1, 4)], dim=0)

            end_c2ws.append(c2w)
        return torch.cat(end_c2ws)
