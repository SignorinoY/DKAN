from __future__ import annotations

import torch
import torch.nn as nn


class DKANLinear(nn.Module):

    def __init__(
        self,
        in_features: int,
        out_features: int,
        grid_size: int = 5,
        spline_degree: int = 3,
        alpha: float = 0.01,
        base_activation=torch.nn.SiLU,
        momentum: float = 0.1,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_degree = spline_degree
        self.base_activation = base_activation()
        self.momentum = momentum

        knot = torch.linspace(alpha / 2, 1 - alpha / 2, grid_size + 2)
        knot = torch.distributions.Normal(0, 1).icdf(knot)

        spline_order = spline_degree + 1
        eps = 1e-3
        grid = torch.concat([
            knot[0] - torch.linspace(0, eps * spline_order, spline_order).flip(-1),
            knot[1:-1],
            knot[-1] + torch.linspace(0, eps * spline_order, spline_order),
        ])
        grid = grid.expand(in_features, -1)
        self.register_buffer("knot", knot)
        self.register_buffer("grid", grid)

        self.batch_norm = nn.BatchNorm1d(in_features, affine=False, momentum=momentum)
        self.residual = nn.Linear(in_features, out_features)
        self.bspline = nn.Linear(in_features*(grid_size+spline_order+2), out_features)

    def forward(self, x):
        residual = self.residual(self.base_activation(x))

        knot: torch.Tensor = (self.knot)
        grid: torch.Tensor = (self.grid)
        x = self.batch_norm(x).unsqueeze(-1)
        bases = ((x >= grid[:, :-1]) & (x < grid[:, 1:])).to(x.dtype)
        for k in range(1, self.spline_degree + 1):
            bases = (
                (x - grid[:, : - (k + 1)])
                / (grid[:, k:-1] - grid[:, : -(k+1)])
                * bases[:, :, :-1]
            ) + (
                (grid[:, (k + 1):] - x)
                / (grid[:, (k + 1):] - grid[:, 1:(-k)])
                * bases[:, :, 1:]
            )
        out1 = nn.functional.relu(- x + knot[0])
        out2 = nn.functional.relu(x - knot[-1])
        bases = torch.cat([bases, out1, out2], dim=2)
        bspline = self.bspline(bases.view(x.size(0), -1))
        return bspline + residual

class DKAN(nn.Module):
    def __init__(
        self,
        layers_hidden,
        grid_size: int = 5,
        spline_degree: int = 3,
        alpha: float = 0.01,
        base_activation=torch.nn.SiLU,
        momentum: float = 0.1,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList()
        for in_features, out_features in zip(layers_hidden, layers_hidden[1:]):
            self.layers.append(
                DKANLinear(
                    in_features, out_features,
                    grid_size=grid_size,
                    spline_degree=spline_degree,
                    alpha=alpha,
                    base_activation=base_activation,
                    momentum=momentum
                    )
                )

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
