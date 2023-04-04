from src.eval_pose import evaluate_ate
import torch
import numpy as np

# from mathutils import Matrix


def evaluate(poses_gt, poses_est, plot=None):

    poses_gt = poses_gt.detach().cpu().numpy()
    poses_est = poses_est.detach().cpu().numpy()

    N = poses_gt.shape[0]
    poses_gt = dict([(i, poses_gt[i]) for i in range(N)])
    poses_est = dict([(i, poses_est[i]) for i in range(N)])

    results = evaluate_ate(poses_gt, poses_est, plot)
    print(results)
    return results


def project_point3d_to_image_batch(c2ws, pts3d, fx, fy, cx, cy, device="cuda:0"):
    if pts3d.shape[-2] == 3:
        pts3d_homo = torch.cat(
            [pts3d, torch.ones_like(pts3d[:, 0].view(-1, 1, 1))], dim=-2
        )
    elif pts3d.shape[-2] == 4:
        pts3d_homo = pts3d
    else:
        raise NotImplementedError

    pts3d_homo = pts3d_homo.to(device)
    bottom = torch.tensor([0.0, 0.0, 0.0, 1.0], dtype=torch.float32).to(c2ws.device)
    if c2ws.shape[-2:] != (4, 4):
        c2ws = torch.cat(
            [
                c2ws,
                bottom.to(device)
                if (c2ws.shape == 2)
                else bottom.view(1, 4, 1).repeat(c2ws.shape[0], 1, 1),
            ],
            dim=-1,
        ).to(device)
    w2cs = torch.inverse(c2ws)

    pts2d_homo = (
        w2cs @ pts3d_homo[:, None, :, :]
    )  # [Cn, 4, 4] @ [Pn, 1, 4, 1] = [Pn, Cn, 4, 1]
    pts2d = pts2d_homo[:, :, :3]
    K = (
        torch.from_numpy(
            np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]]).reshape(3, 3)
        )
        .to(device)
        .float()
    )
    pts2d[:, :, 0] *= -1
    uv = K @ pts2d  # [3,3] @ [Pn, Cn, 3, 1] = [Pn, Cn, 3, 1]
    z = uv[:, :, -1:] + 1e-5
    uv = uv[:, :, :2] / z  # [Pn, Cn, 2, 1]

    uv = uv.float()
    return uv, z


def select_uv_batch(i, j, n, idxs, depth, color, device="cuda:0"):
    """
    Select n uv from dense uv.

    """
    all_size = i.reshape(-1).shape[0]
    i = i.reshape(-1)
    j = j.reshape(-1)
    indices = torch.randint(i.shape[0], (n,), device=device).long()
    indices = (indices.clamp(0, i.shape[0])).long()
    assert indices.max() < i.shape[0], "out 1"
    assert indices.max() < j.shape[0], "out 2"
    i = i[indices]  # (n)
    j = j[indices]  # (n)

    indices = indices + idxs.long() * all_size
    indices = indices.long()
    color = color.reshape(-1, 3)  # [b,h,w,3]
    assert indices.max() < color.shape[0], "out 3"
    color = color[indices]  # (n,3)
    if depth is not None:
        depth = depth.reshape(-1)
        assert indices.max() < depth.shape[0], "out 4"
        depth = depth[indices]  # (n)
    else:
        # no depth
        pass
    return i, j, depth, color


def get_sample_uv_batch(
    H0, H1, W0, W1, n, idx, depth, color, device="cuda:0", indices=None
):
    """
    Sample n uv coordinates from an image region H0..H1, W0..W1

    """
    if depth is not None:
        depth_ori = depth.clone()
        depth = depth[:, H0:H1, W0:W1]
    else:
        # no depth
        pass
    color_ori = color.clone()
    color = color[:, H0:H1, W0:W1]
    if indices is None:
        # select random uv
        i, j = torch.meshgrid(
            torch.linspace(W0, W1 - 1, W1 - W0).to(device),
            torch.linspace(H0, H1 - 1, H1 - H0).to(device),
        )
        i = i.t()  # transpose
        j = j.t()
        i, j, depth, color = select_uv_batch(i, j, n, idx, depth, color, device=device)
    else:
        i = indices[:, 0].view(-1)
        j = indices[:, 1].view(-1)
        assert i.max() < color.shape[0], "out 7"
        assert j.max() < color.shape[1], "out 8"
        depth = depth[i, j]
        color = color[i, j]
    return i, j, depth, color


def get_rays_from_uv_batch(i, j, c2w, H, W, fx, fy, cx, cy, device):
    """
    Get corresponding rays from input uv.

    """
    if isinstance(c2w, np.ndarray):
        c2w = torch.from_numpy(c2w).to(device)

    dirs = torch.stack([(i - cx) / fx, -(j - cy) / fy, -torch.ones_like(i)], -1).to(
        device
    )
    dirs = dirs.reshape(i.shape[0], i.shape[1], 1, 3)
    # Rotate ray directions from camera frame to the world frame
    # dot product, equals to: [c2w.dot(dir) for dir in dirs]
    rays_d = torch.sum(dirs * c2w[None, :, :3, :3], -1)
    rays_o = c2w[:, :3, -1].expand(rays_d.shape)
    return rays_o, rays_d


def get_samples_batch(
    H0,
    H1,
    W0,
    W1,
    n,
    H,
    W,
    fx,
    fy,
    cx,
    cy,
    c2ws,
    idxs,
    depth,
    color,
    device,
    indices=None,
    return_uv=False,
):
    i, j, sample_depth, sample_color = get_sample_uv_batch(
        H0, H1, W0, W1, n, idxs, depth, color, device=device, indices=indices
    )
    i = i.view(-1, c2ws.shape[0])
    j = j.view(-1, c2ws.shape[0])
    rays_o, rays_d = get_rays_from_uv_batch(i, j, c2ws, H, W, fx, fy, cx, cy, device)
    if return_uv:
        return (
            rays_o.contiguous().view(-1, 3),
            rays_d.contiguous().view(-1, 3),
            sample_depth.view(-1),
            sample_color.view(-1, 3),
            torch.stack([i.view(-1), j.view(-1)], dim=1).to(device),
        )
    else:
        return (
            rays_o.view(-1, 3),
            rays_d.view(-1, 3),
            sample_depth.view(-1),
            sample_color.view(-1, 3),
        )


def get_sample_uv_by_indices_batch(
    H0, H1, W0, W1, depth, color, indices, idxs, device="cuda:0"
):
    """
    Sample n uv coordinates from an image region H0..H1, W0..W1

    """
    if depth is not None:
        depth = depth[:, H0:H1, W0:W1]
    else:
        # no depth
        pass
    color = color[:, H0:H1, W0:W1]

    # compute new idxs

    indices = indices
    if depth is not None:
        depth = (
            torch.nn.functional.grid_sample(
                depth.view(list(depth.shape[0:]) + [1]).permute(0, 3, 1, 2),
                indices.view(indices.shape[0], depth.shape[0], -1, 2).permute(
                    1, 0, 2, 3
                ),
                mode="nearest",
            )
            .permute(2, 0, 3, 1)
            .reshape(-1)
        )
    color = (
        torch.nn.functional.grid_sample(
            color.permute(0, 3, 1, 2),
            indices.view(indices.shape[0], color.shape[0], -1, 2).permute(1, 0, 2, 3),
            mode="bilinear",
        )
        .permute(2, 0, 3, 1)
        .contiguous()
        .view(-1, 3)
    )

    # color = color[]

    i = indices[..., 0].view(-1)
    j = indices[..., 1].view(-1)
    return i, j, depth, color


def get_samples_by_indices_batch(
    H0,
    H1,
    W0,
    W1,
    H,
    W,
    fx,
    fy,
    cx,
    cy,
    c2w,
    idxs,
    depth,
    color,
    indices,
    device="cuda:0",
    return_uv=False,
):
    """get n rays from the image region
    c2w is the camera pose and depth/color is the corresponding image tensor.
    select the rays by the given indices

    Args:
        H0 (_type_): _description_
        H1 (_type_): _description_
        W0 (_type_): _description_
        W1 (_type_): _description_
        n (_type_): _description_
        H (_type_): _description_
        W (_type_): _description_
        fx (_type_): _description_
        fy (_type_): _description_
        cx (_type_): _description_
        cy (_type_): _description_
        c2w (_type_): _description_
        depth (_type_): _description_
        color (_type_): _description_
        indices (_type_): [-1,1] , WH , for grid sample
        device (_type_): _description_
    """

    i, j, sample_depth, sample_color = get_sample_uv_by_indices_batch(
        H0, H1, W0, W1, depth, color, indices, device
    )
    i = (indices[..., 0].view(-1) + 1) / 2.0 * W
    j = (indices[..., 1].view(-1) + 1) / 2.0 * H
    i = i.view(-1, c2w.shape[0])
    j = j.view(-1, c2w.shape[0])
    rays_o, rays_d = get_rays_from_uv_batch(i, j, c2w, H, W, fx, fy, cx, cy, device)
    if not return_uv:
        return rays_o, rays_d, sample_depth, sample_color
    else:
        return (
            rays_o,
            rays_d,
            sample_depth,
            sample_color,
            torch.stack([i, j], dim=1).to(device),
        )


def normalize_vector(v, return_mag=False):
    batch = v.shape[0]
    v_mag = torch.sqrt(v.pow(2).sum(1))  # batch
    v_mag = torch.max(
        v_mag, torch.autograd.Variable(torch.FloatTensor([1e-8]).to(v.device))
    )
    v_mag = v_mag.view(batch, 1).expand(batch, v.shape[1])
    v = v / v_mag
    if return_mag == True:
        return v, v_mag[:, 0]
    else:
        return v


def cross_product(u, v):
    batch = u.shape[0]
    # print (u.shape)
    # print (v.shape)
    i = u[:, 1] * v[:, 2] - u[:, 2] * v[:, 1]
    j = u[:, 2] * v[:, 0] - u[:, 0] * v[:, 2]
    k = u[:, 0] * v[:, 1] - u[:, 1] * v[:, 0]

    out = torch.cat(
        (i.view(batch, 1), j.view(batch, 1), k.view(batch, 1)), 1
    )  # batch*3

    return out


def compute_ortho6d_from_rotation_matrix(matrix: np.ndarray):
    # if matrix.shape[-2] != (3, 3):
    #     raise "3 x 3 matrix only"
    is_batch = True if len(matrix.shape) == 3 else False

    if not is_batch:
        # matrix = matrix.unsqueeze
        matrix = np.expand_dims(matrix, 0)
    matrix = matrix[:, :3, :3]
    # ortho6d = torch.cat([matrix[:, :, 0], matrix[:, :, 1]], dim=-1).view(-1, 6)
    ortho6d = np.concatenate([matrix[:, :, 0], matrix[:, :, 1]], 0).reshape(-1, 6)
    if not is_batch:
        ortho6d = ortho6d.squeeze(0)
    return ortho6d


def compute_rotation_matrix_from_ortho6d(ortho6d):
    is_batch = True
    if len(ortho6d.shape) != 2:
        is_batch = False
        ortho6d = ortho6d.unsqueeze(0)
    x_raw = ortho6d[:, 0:3]  # batch*3
    y_raw = ortho6d[:, 3:6]  # batch*3

    x = normalize_vector(x_raw)  # batch*3
    z = cross_product(x, y_raw)  # batch*3
    z = normalize_vector(z)  # batch*3
    y = cross_product(z, x)  # batch*3

    x = x.view(-1, 3, 1)
    y = y.view(-1, 3, 1)
    z = z.view(-1, 3, 1)
    matrix = torch.cat((x, y, z), 2)  # batch*3*3
    if not is_batch:
        matrix = matrix[0]
    return matrix


def get_tensor_from_camera(RT, Tquad=False, method="quat"):
    """
    Convert transformation matrix to quaternion and translation.

    """
    gpu_id = -1
    if type(RT) == torch.Tensor:
        if RT.get_device() != -1:
            RT = RT.detach().cpu()
            gpu_id = RT.get_device()
        RT = RT.numpy()
    # from mathutils import Matrix
    R, T = RT[:3, :3], RT[:3, 3]
    if method == "quat":
        # rot = Matrix(R)
        # quad = rot.to_quaternion()
        quad = (
            rot2quaternion(torch.from_numpy(R).unsqueeze(0))
            .squeeze(0)
            .detach()
            .cpu()
            .numpy()
        )
    elif method == "ortho6d":
        quad = compute_ortho6d_from_rotation_matrix(R)
    else:
        raise NotImplementedError
    if Tquad:
        tensor = np.concatenate([T, quad], 0)
    else:
        tensor = np.concatenate([quad, T], 0)
    tensor = torch.from_numpy(tensor).float()
    if gpu_id != -1:
        tensor = tensor.to(gpu_id)
    return tensor


def quad2rotation(quad):
    """
    Convert quaternion to rotation in batch. Since all operation in pytorch, support gradient passing.

    Args:
        quad (tensor, batch_size*4): quaternion.

    Returns:
        rot_mat (tensor, batch_size*3*3): rotation.
    """
    bs = quad.shape[0]
    qr, qi, qj, qk = quad[:, 0], quad[:, 1], quad[:, 2], quad[:, 3]
    two_s = 2.0 / (quad * quad).sum(-1)
    rot_mat = torch.zeros(bs, 3, 3).to(quad.get_device())
    rot_mat[:, 0, 0] = 1 - two_s * (qj**2 + qk**2)
    rot_mat[:, 0, 1] = two_s * (qi * qj - qk * qr)
    rot_mat[:, 0, 2] = two_s * (qi * qk + qj * qr)
    rot_mat[:, 1, 0] = two_s * (qi * qj + qk * qr)
    rot_mat[:, 1, 1] = 1 - two_s * (qi**2 + qk**2)
    rot_mat[:, 1, 2] = two_s * (qj * qk - qi * qr)
    rot_mat[:, 2, 0] = two_s * (qi * qk - qj * qr)
    rot_mat[:, 2, 1] = two_s * (qj * qk + qi * qr)
    rot_mat[:, 2, 2] = 1 - two_s * (qi**2 + qj**2)
    return rot_mat


def rot2quaternion(rot_mat, eps=1e-6):
    """Convert 3x4 rotation matrix to 4d quaternion vector [From torchgeometrc library]
    This algorithm is based on algorithm described in
    https://github.com/KieranWynn/pyquaternion/blob/master/pyquaternion/quaternion.py#L201
    Args:
        rot_mat (Tensor): the rotation matrix to convert.
    Return:
        Tensor: the rotation in quaternion
    Shape:
        - Input: :math:`(N, 3, 4)`
        - Output: :math:`(N, 4)`
    Example:
        >>> input = torch.rand(4, 3, 4)  # Nx3x4
        >>> output = tgm.rot_mat_to_quaternion(input)  # Nx4
    """
    if not torch.is_tensor(rot_mat):
        raise TypeError(
            "Input type is not a torch.Tensor. Got {}".format(type(rot_mat))
        )

    if len(rot_mat.shape) > 3:
        raise ValueError(
            "Input size must be a three dimensional tensor. Got {}".format(
                rot_mat.shape
            )
        )

    rot_mat = rot_mat[:, :3, :3]
    rmat_t = torch.transpose(rot_mat, 1, 2)

    mask_d2 = (rmat_t[:, 2, 2] < eps).float()
    mask_d0_d1 = (rmat_t[:, 0, 0] > rmat_t[:, 1, 1]).float()
    mask_d0_nd1 = (rmat_t[:, 0, 0] < -rmat_t[:, 1, 1]).float()

    t0 = 1 + rmat_t[:, 0, 0] - rmat_t[:, 1, 1] - rmat_t[:, 2, 2]
    q0 = torch.stack(
        [
            rmat_t[:, 1, 2] - rmat_t[:, 2, 1],
            t0,
            rmat_t[:, 0, 1] + rmat_t[:, 1, 0],
            rmat_t[:, 2, 0] + rmat_t[:, 0, 2],
        ],
        -1,
    )
    t0_rep = t0.repeat(4, 1).t()

    t1 = 1 - rmat_t[:, 0, 0] + rmat_t[:, 1, 1] - rmat_t[:, 2, 2]
    q1 = torch.stack(
        [
            rmat_t[:, 2, 0] - rmat_t[:, 0, 2],
            rmat_t[:, 0, 1] + rmat_t[:, 1, 0],
            t1,
            rmat_t[:, 1, 2] + rmat_t[:, 2, 1],
        ],
        -1,
    )
    t1_rep = t1.repeat(4, 1).t()

    t2 = 1 - rmat_t[:, 0, 0] - rmat_t[:, 1, 1] + rmat_t[:, 2, 2]
    q2 = torch.stack(
        [
            rmat_t[:, 0, 1] - rmat_t[:, 1, 0],
            rmat_t[:, 2, 0] + rmat_t[:, 0, 2],
            rmat_t[:, 1, 2] + rmat_t[:, 2, 1],
            t2,
        ],
        -1,
    )
    t2_rep = t2.repeat(4, 1).t()

    t3 = 1 + rmat_t[:, 0, 0] + rmat_t[:, 1, 1] + rmat_t[:, 2, 2]
    q3 = torch.stack(
        [
            t3,
            rmat_t[:, 1, 2] - rmat_t[:, 2, 1],
            rmat_t[:, 2, 0] - rmat_t[:, 0, 2],
            rmat_t[:, 0, 1] - rmat_t[:, 1, 0],
        ],
        -1,
    )
    t3_rep = t3.repeat(4, 1).t()

    mask_c0 = mask_d2 * mask_d0_d1
    mask_c1 = mask_d2 * (1 - mask_d0_d1)
    mask_c2 = (1 - mask_d2) * mask_d0_nd1
    mask_c3 = (1 - mask_d2) * (1 - mask_d0_nd1)
    mask_c0 = mask_c0.view(-1, 1).type_as(q0)
    mask_c1 = mask_c1.view(-1, 1).type_as(q1)
    mask_c2 = mask_c2.view(-1, 1).type_as(q2)
    mask_c3 = mask_c3.view(-1, 1).type_as(q3)

    q = q0 * mask_c0 + q1 * mask_c1 + q2 * mask_c2 + q3 * mask_c3
    q /= torch.sqrt(
        t0_rep * mask_c0
        + t1_rep * mask_c1
        + t2_rep * mask_c2  # noqa
        + t3_rep * mask_c3
    )  # noqa
    q *= 0.5
    return q


def get_camera_from_tensor(inputs):

    N = len(inputs.shape)
    if N == 1:
        inputs = inputs.unsqueeze(0)
    rot_len = 4
    quad, T = inputs[:, :rot_len], inputs[:, rot_len:]
    R = quad2rotation(quad)
    RT = torch.cat([R, T[:, :, None]], 2)
    if N == 1:
        RT = RT[0]
    return RT
