import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable


class DenseLayer(nn.Linear):
    def __init__(
        self, in_dim: int, out_dim: int, activation: str = "relu", *args, **kwargs
    ) -> None:
        self.activation = activation
        super().__init__(in_dim, out_dim, *args, **kwargs)

    def reset_parameters(self) -> None:
        torch.nn.init.xavier_uniform_(
            self.weight, gain=torch.nn.init.calculate_gain(self.activation)
        )
        if self.bias is not None:
            torch.nn.init.zeros_(self.bias)


class basic_MLP(nn.Module):
    def __init__(
        self, input_dim=3, hidden_dim=32, output_dim=3, n_blocks=5, skips=[2]
    ) -> None:
        super().__init__()

        self.linears = nn.ModuleList()
        layer_input_dim = input_dim
        layer_output_dim = None
        for i in range(n_blocks):
            layer_input_dim = input_dim if i == 0 else hidden_dim
            if i in skips:
                layer_input_dim += input_dim
            layer_output_dim = output_dim if i == n_blocks - 1 else hidden_dim

            layer = DenseLayer(
                layer_input_dim,
                layer_output_dim,
                activation="relu" if i != n_blocks - 1 else "linear",
            )
            self.linears.append(layer)

        self.skips = skips
        self.dp = nn.Dropout(p=0.2)

    def forward(self, input):
        x = input
        # x = self.dp(x)
        for i, l in enumerate(self.linears):
            if i in self.skips:
                x = torch.cat([x, input], dim=-1)
            x = l(x)
            if i != len(self.linears) - 1:
                x = F.relu(x)
        return x


class NeRF(nn.Module):
    def __init__(self, configs, **kwargs) -> None:
        super().__init__()

        self.args = configs
        self.device = self.args.device
        self.grids_lens = (
            [0.64, 0.48, 0.32, 0.24, 0.16, 0.12, 0.08]
            if self.args.nerf.grids_lens is None
            else self.args.nerf.grids_lens
        )
        self.bound = torch.from_numpy(
            np.asarray(
                self.args.nerf.bound
                if self.args.nerf.bound is not None
                else [[-1, 1], [-1, 1], [1, 1]]
            )
        )
        self.grids_dim = (
            self.args.nerf.grids_dim if self.args.nerf.grids_dim is not None else 4
        )
        self.xyz_len = (self.bound[:, 1] - self.bound[:, 0]).float()
        self.grids_shape = []
        self.grids_feat = []
        self.grids_level = len(self.grids_lens)
        self.grids_xyz = []
        self.grids_coord_norm_fns = []
        for ii, grid_len in enumerate(self.grids_lens):
            grid_shape = list(map(int, (self.xyz_len / grid_len).tolist()))
            grid_shape[0], grid_shape[2] = grid_shape[2], grid_shape[0]

            grid_true_shape = [1, self.grids_dim] + grid_shape
            grid = Variable(
                torch.zeros(grid_true_shape).to(self.device), requires_grad=True
            )
            self.grids_feat.append(grid)
            self.grids_shape.append(grid_shape)
            self.grids_xyz.append(
                (torch.tensor(grid_shape).float() * grid_len).to(self.device)
            )

        # self.decoder = base if self.args.nerf.use_decoder else None
        self.color_decoder = nn.Module()
        self.alpha_decoder = nn.Module()
        if self.args.nerf.decoder == "basic_MLP":
            self.alpha_decoder = basic_MLP(
                self.grids_level,
                self.args.nerf.hidden_dim,
                1,
                self.args.nerf.n_blocks,
                self.args.nerf.skips,
            ).to(self.device)
            self.color_decoder = basic_MLP(
                6,
                self.args.nerf.hidden_dim,
                3,
                self.args.nerf.n_blocks,
                self.args.nerf.skips,
            ).to(self.device)
        else:
            self.alpha_decoder = lambda x: x
            self.color_decoder = lambda x: x

    def normalize_3d_coordinate(self, p, grid_len, grid_shape):
        grid_xyz = torch.tensor(grid_shape).float() * grid_len
        p = p.reshape(-1, 3)
        p[:, 0] = ((p[:, 0] - self.bound[0, 0]) / (grid_xyz[2])) * 2 - 1.0
        p[:, 1] = ((p[:, 1] - self.bound[1, 0]) / (grid_xyz[1])) * 2 - 1.0
        p[:, 2] = ((p[:, 2] - self.bound[2, 0]) / (grid_xyz[0])) * 2 - 1.0
        return p

    def forward(self, input, **kwargs):
        p = input

        raw_rgb_input = []
        raw_alpha_input = []
        for i in range(self.grids_level):
            p_norm = self.normalize_3d_coordinate(
                p.clone(), self.grids_lens[i], self.grids_shape[i]
            ).unsqueeze(0)
            c = (
                F.grid_sample(
                    self.grids_feat[i],
                    p_norm[:, :, None, None].float(),
                    align_corners=True,
                    mode="bilinear",
                )
                .squeeze(-1)
                .squeeze(-1)
                .transpose(1, 2)
                .squeeze(0)
            )
            raw_alpha_input.append(c[:, -1].view(-1, 1))
            raw_rgb_input.append(c[:, :3])
        raw_alpha_input = torch.cat(raw_alpha_input, dim=-1)
        raw_rgb_input = torch.cat(raw_rgb_input[-2:], dim=-1)
        alpha = self.alpha_decoder(raw_alpha_input)
        rgb = self.color_decoder(raw_rgb_input)
        return torch.cat([rgb, alpha], dim=-1)
