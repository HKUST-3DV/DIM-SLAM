import torch
import numpy as np

from collections import defaultdict


class BaseRenderer(object):
    def __init__(
        self, args, model, points_batch_size=500000, ray_batch_size=100000
    ) -> None:
        self.args = args

        self.ray_batch_size = ray_batch_size
        self.points_batch_size = points_batch_size

        self.scale = args.scale
        self.model = model
        self.bound = model.bound

        # read the camera intrinsic from the image and K?
        self.H, self.W, self.fx, self.fy, self.cx, self.cy = (
            args.updated_color_cam.H,
            args.updated_color_cam.W,
            args.updated_color_cam.fx,
            args.updated_color_cam.fy,
            args.updated_color_cam.cx,
            args.updated_color_cam.cy,
        )

    def eval_points(self, points, **kwargs):
        """_summary_

        Args:
            points (_type_): _description_

        Returns:
            _type_: _description_
        """
        points_splits = torch.split(points, self.points_batch_size, dim=0)
        bound = self.bound

        rets = []
        extra_rets = defaultdict(list)

        for ps in points_splits:

            mask = (
                (ps[:, 0] < bound[0][1])
                & (ps[:, 0] > bound[0][0])
                & (ps[:, 1] < bound[1][1])
                & (ps[:, 1] > bound[1][0])
                & (ps[:, 2] < bound[2][1])
                & (ps[:, 2] > bound[2][0])
            )
            ps = ps.unsqueeze(0)
            ret = self.model(ps)
            ret[~mask, 3] = 100
            rets.append(ret)
        ret = torch.cat(rets, dim=0)
        extra_ret = {k: torch.stack(v, dim=0) for k, v in extra_rets.items()}
        return ret, extra_ret

    def render_batch_ray(self, rays_d, rays_o, device, depth=None, **kwargs):
        """_summary_

        Args:
            rays_d (_type_): _description_
            rays_o (_type_): _description_
            device (_type_): _description_
            depth (_type_, optional): _description_. Defaults to None.

        Returns:
            _type_: _description_
        """

        N_rays = rays_o.shape[0]

        near = 0.01
        with torch.no_grad():
            det_rays_o = rays_o.clone().detach().unsqueeze(-1)
            det_rays_d = rays_d.clone().detach().unsqueeze(-1)
            t = (
                self.bound.unsqueeze(0).to(device) - det_rays_o
            ) / det_rays_d  # (N, 3, 2)
            far_bb, _ = torch.min(torch.max(t, dim=2)[0], dim=1)
            far_bb = far_bb.unsqueeze(-1)
            far_bb += 0.01

        far = torch.clamp(
            far_bb, torch.zeros_like(far_bb), torch.ones_like(far_bb).float() * 5
        )

        # this is the multi stage sampling
        sampling_stages = [32, 32, 32, 32]
        for le, samples_num in enumerate(sampling_stages):
            if le < len(sampling_stages) - 1:
                with torch.no_grad():
                    t_vals = torch.linspace(0.0, 1.0, steps=samples_num, device=device)

                    if le == 0:
                        z_vals = near * (1.0 - t_vals) + far * (t_vals)

                        # get intervals between samples
                        mids = 0.5 * (z_vals[..., 1:] + z_vals[..., :-1])
                        upper = torch.cat([mids, z_vals[..., -1:]], -1)
                        lower = torch.cat([z_vals[..., :1], mids], -1)
                        t_rand = torch.rand(z_vals.shape).to(device)
                        z_vals = lower + (upper - lower) * t_rand
                    else:
                        z_vals_mid = 0.5 * (z_vals[..., 1:] + z_vals[..., :-1])
                        z_samples = sample_pdf(
                            z_vals_mid,
                            weights[..., 1:-1],
                            samples_num,
                            det=False,
                            device=device,
                        )
                        z_samples = z_samples.detach()
                        z_vals, _ = torch.sort(torch.cat([z_vals, z_samples], -1), -1)

                    pts = (
                        rays_o[..., None, :]
                        + rays_d[..., None, :] * z_vals[..., :, None]
                    )
                    pointsf = pts.reshape(-1, 3)

                    raw, extra_ret = self.eval_points(pointsf, **kwargs)
                    raw = raw.reshape(N_rays, z_vals.shape[1], -1)
                    (
                        depth,
                        uncertainty,
                        color,
                        weights,
                        density,
                    ) = render_equation(raw, z_vals, rays_d, device)
            else:
                t_vals = torch.linspace(0.0, 1.0, steps=samples_num, device=device)

                if le == 0:
                    z_vals = near * (1.0 - t_vals) + far * (t_vals)
                    # get intervals between samples
                    mids = 0.5 * (z_vals[..., 1:] + z_vals[..., :-1])
                    upper = torch.cat([mids, z_vals[..., -1:]], -1)
                    lower = torch.cat([z_vals[..., :1], mids], -1)
                    # stratified samples in those intervals
                    t_rand = torch.rand(z_vals.shape).to(device)
                    z_vals = lower + (upper - lower) * t_rand
                else:
                    z_vals_mid = 0.5 * (z_vals[..., 1:] + z_vals[..., :-1])
                    z_samples = sample_pdf(
                        z_vals_mid,
                        weights[..., 1:-1],
                        samples_num,
                        det=False,
                        device=device,
                    )
                    z_samples = z_samples.detach()
                    z_vals, _ = torch.sort(torch.cat([z_vals, z_samples], -1), -1)

                pts = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., :, None]
                pointsf = pts.reshape(-1, 3)

                raw, extra_ret = self.eval_points(pointsf, **kwargs)
                raw = raw.reshape(N_rays, z_vals.shape[1], -1)

                (
                    depth,
                    uncertainty,
                    color,
                    weights,
                    density,
                ) = render_equation(raw, z_vals, rays_d, device)

        return depth, uncertainty, color, extra_ret, density

    def render_img(self, c2w, device, gt_depth=None, **kwargs):
        """
        Renders out depth, uncertainty, and color images.

        Args:
            c (dict): feature grids.
            decoders (nn.module): decoders.
            c2w (tensor): camera to world matrix of current frame.
            device (str): device name to compute on.
            stage (str): query stage.
            gt_depth (tensor, optional): sensor depth image. Defaults to None.

        Returns:
            depth (tensor, H*W): rendered depth image.
            uncertainty (tensor, H*W): rendered uncertainty image.
            color (tensor, H*W*3): rendered color image.
        """
        with torch.no_grad():
            H = self.H
            W = self.W
            rays_o, rays_d = get_rays(
                H, W, self.fx, self.fy, self.cx, self.cy, c2w, device
            )
            rays_o = rays_o.reshape(-1, 3)
            rays_d = rays_d.reshape(-1, 3)

            depth_list = []
            uncertainty_list = []
            color_list = []

            ray_batch_size = self.ray_batch_size
            if gt_depth is not None:
                gt_depth = gt_depth.reshape(-1)

            if "render_size" in kwargs:
                skip = rays_d.shape[0] // kwargs["render_size"]
                kwargs["skip"] = skip

            if "skip" in kwargs:
                rays_d = rays_d[:: kwargs["skip"]]
                rays_o = rays_o[:: kwargs["skip"]]

            for i in range(0, rays_d.shape[0], ray_batch_size):
                rays_d_batch = rays_d[i : i + ray_batch_size]
                rays_o_batch = rays_o[i : i + ray_batch_size]
                gt_depth_batch = (
                    gt_depth[i : i + ray_batch_size] if gt_depth is not None else None
                )
                ret = self.render_batch_ray(
                    rays_d_batch,
                    rays_o_batch,
                    device,
                    gt_depth=gt_depth_batch,
                    **kwargs
                )

                depth, uncertainty, color, extra_ret, density = ret
                depth_list.append(depth.double())
                uncertainty_list.append(uncertainty.double())
                color_list.append(color)

            depth = torch.cat(depth_list, dim=0)
            uncertainty = torch.cat(uncertainty_list, dim=0)
            color = torch.cat(color_list, dim=0)

            if "skip" not in kwargs or kwargs["skip"] == 1:
                depth = depth.reshape(H, W)
                uncertainty = uncertainty.reshape(H, W)
                color = color.reshape(H, W, 3)
            return depth, uncertainty, color

    def cam_center_regulation(
        self,
        rays_d,
        rays_o,
        device,
    ):
        """the space near the camera center is not well sampled, so we add some random rays to the center of the camera, to make sure the space is empty.

        Args:
            rays_d (_type_): _description_
            rays_o (_type_): _description_
            device (_type_): _description_

        Returns:
            _type_: _description_
        """
        rand_rays_d = torch.rand(rays_d.shape) * 2 - 1
        # rand_rays_d = rand_rays_d.normalize(dim=-1, p=2)
        rand_rays_d = torch.nn.functional.normalize(rand_rays_d, dim=-1, p=2).to(device)

        t = torch.rand(rays_d.shape[0], 1) * 0.16
        t = t.to(device)
        pts = rays_o[..., None, :] + rand_rays_d[..., None, :] * t[..., :, None]
        pointsf = pts.reshape(-1, 3)
        raw, extra_ret = self.eval_points(pointsf)
        sigma = raw[:, -1]
        return sigma


def get_rays(H, W, fx, fy, cx, cy, c2w, device):
    """
    Get rays for a whole image.

    """
    if isinstance(c2w, np.ndarray):
        c2w = torch.from_numpy(c2w)
    c2w = c2w.to(device)
    # pytorch's meshgrid has indexing='ij'
    i, j = torch.meshgrid(torch.linspace(0, W - 1, W), torch.linspace(0, H - 1, H))
    i = i.t()  # transpose
    j = j.t()
    # This is the same as the nice slam
    dirs = torch.stack([(i - cx) / fx, -(j - cy) / fy, -torch.ones_like(i)], -1).to(
        device
    )
    dirs = dirs.reshape(H, W, 1, 3)
    rays_d = torch.sum(dirs * c2w[:3, :3], -1)
    rays_o = c2w[:3, -1].expand(rays_d.shape)
    return rays_o, rays_d


def sample_pdf(bins, weights, N_samples, det=False, device="cuda:0"):
    """
    Hierarchical sampling in NeRF paper (section 5.2).

    """
    # Get pdf
    weights = weights + 1e-5  # prevent nans
    pdf = weights / torch.sum(weights, -1, keepdim=True)
    cdf = torch.cumsum(pdf, -1)
    # (batch, len(bins))
    cdf = torch.cat([torch.zeros_like(cdf[..., :1]), cdf], -1)

    # Take uniform samples
    if det:
        u = torch.linspace(0.0, 1.0, steps=N_samples)
        u = u.expand(list(cdf.shape[:-1]) + [N_samples])
    else:
        u = torch.rand(list(cdf.shape[:-1]) + [N_samples])

    u = u.to(device)
    # Invert CDF
    u = u.contiguous()
    try:
        # this should work fine with the provided environment.yaml
        inds = torch.searchsorted(cdf, u, right=True)
    except:
        # for lower version torch that does not have torch.searchsorted,
        # you need to manually install from
        # https://github.com/aliutkus/torchsearchsorted
        from torchsearchsorted import searchsorted

        inds = searchsorted(cdf, u, side="right")
    below = torch.max(torch.zeros_like(inds - 1), inds - 1)
    above = torch.min((cdf.shape[-1] - 1) * torch.ones_like(inds), inds)
    inds_g = torch.stack([below, above], -1)  # (batch, N_samples, 2)

    matched_shape = [inds_g.shape[0], inds_g.shape[1], cdf.shape[-1]]
    cdf_g = torch.gather(cdf.unsqueeze(1).expand(matched_shape), 2, inds_g)
    bins_g = torch.gather(bins.unsqueeze(1).expand(matched_shape), 2, inds_g)

    denom = cdf_g[..., 1] - cdf_g[..., 0]
    denom = torch.where(denom < 1e-5, torch.ones_like(denom), denom)
    t = (u - cdf_g[..., 0]) / denom
    samples = bins_g[..., 0] + t * (bins_g[..., 1] - bins_g[..., 0])

    return samples


def render_equation(raw, z_vals, rays_d, device="cuda:0"):
    dists = z_vals[..., 1:] - z_vals[..., :-1]
    dists = dists.float()
    dists = torch.cat(
        [dists, torch.Tensor([1e10]).float().to(device).expand(dists[..., :1].shape)],
        -1,
    )  # [N_rays, N_samples]

    # different ray angle corresponds to different unit length
    dists = dists * torch.norm(rays_d[..., None, :], dim=-1)
    rgb = raw[..., :-1]
    raw[..., 3] = torch.sigmoid(10 * raw[..., -1])
    alpha = raw[..., -1]
    weights = (
        alpha.float()
        * torch.cumprod(
            torch.cat(
                [
                    torch.ones((alpha.shape[0], 1)).to(device).float(),
                    (1.0 - alpha + 1e-10).float(),
                ],
                -1,
            ).float(),
            -1,
        )[:, :-1]
    )
    rgb = torch.sum(weights[..., None].detach() * rgb, -2)
    depth = torch.sum(weights * z_vals, -1)
    tmp = z_vals - depth.unsqueeze(-1)
    depth_var = torch.sum(weights * tmp * tmp, dim=1)
    return depth, depth_var, rgb, weights, alpha
