import torch
import torch.nn as nn
import torch.nn.functional as F


class KANLinear(nn.Module):

    def __init__(
        self,
        in_features: int,
        out_features: int,
        grid_size: int = 5,
        spline_order: int = 3,
        scale_noise: float = 0.1,
        scale_base: float = 1.0,
        scale_spline: float = 1.0,
        enable_standalone_scale_spline: bool = True,
        base_activation=nn.SiLU,
        grid_eps: float = 0.02,
        grid_range: list = [-1, 1],
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order
        self.base_activation = base_activation()

        # B-spline grid: (in_features, grid_size + 2*spline_order + 1)
        h = (grid_range[1] - grid_range[0]) / grid_size
        grid = (
            torch.arange(-spline_order, grid_size + spline_order + 1) * h
            + grid_range[0]
        ).expand(in_features, -1).contiguous()
        self.register_buffer("grid", grid)

        # Learnable parameters
        self.base_weight = nn.Parameter(torch.empty(out_features, in_features))
        self.spline_weight = nn.Parameter(
            torch.empty(out_features, in_features, grid_size + spline_order)
        )

        if enable_standalone_scale_spline:
            self.spline_scaler = nn.Parameter(torch.empty(out_features, in_features))
        else:
            self.spline_scaler = None

        self.scale_base = scale_base
        self.scale_spline = scale_spline
        self.scale_noise = scale_noise

        self._init_weights()

    def _init_weights(self):
        nn.init.kaiming_uniform_(self.base_weight, a=5 ** 0.5)

        with torch.no_grad():
            noise = (
                torch.rand(self.grid_size + 1, self.in_features, self.out_features) - 0.5
            ) * self.scale_noise / self.grid_size

            self.spline_weight.data.copy_(
                self.scale_spline
                * self._curve2coeff(
                    self.grid.T[self.spline_order: -self.spline_order],
                    noise,
                )
            )

        if self.spline_scaler is not None:
            nn.init.kaiming_uniform_(self.spline_scaler, a=5 ** 0.5)

    def b_splines(self, x: torch.Tensor) -> torch.Tensor:
        assert x.dim() == 2 and x.shape[1] == self.in_features

        grid = self.grid
        x = x.unsqueeze(-1)

        bases = ((x >= grid[:, :-1]) & (x < grid[:, 1:])).float()

        for k in range(1, self.spline_order + 1):
            bases = (
                (x - grid[:, : -(k + 1)])
                / (grid[:, k:-1] - grid[:, : -(k + 1)] + 1e-8)
                * bases[:, :, :-1]
            ) + (
                (grid[:, k + 1:] - x)
                / (grid[:, k + 1:] - grid[:, 1:(-k)] + 1e-8)
                * bases[:, :, 1:]
            )

        return bases.contiguous()

    def _curve2coeff(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        A = self.b_splines(x).transpose(0, 1)
        B = y.transpose(0, 1)
        solution = torch.linalg.lstsq(A, B).solution
        result = solution.permute(2, 0, 1)
        return result.contiguous()

    def scaled_spline_weight(self) -> torch.Tensor:
        if self.spline_scaler is not None:
            return self.spline_weight * self.spline_scaler.unsqueeze(-1)
        return self.spline_weight

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        original_shape = x.shape
        x_2d = x.reshape(-1, self.in_features)

        base_out = F.linear(self.base_activation(x_2d), self.scale_base * self.base_weight)

        spline_out = F.linear(
            self.b_splines(x_2d).view(x_2d.shape[0], -1),
            self.scaled_spline_weight().view(self.out_features, -1),
        )

        out = base_out + spline_out
        return out.reshape(original_shape[:-1] + (self.out_features,))

    def regularisation_loss(
        self,
        regularise_activation: float = 1.0,
        regularise_entropy: float = 1.0,
    ) -> torch.Tensor:

        w = self.scaled_spline_weight()
        l1 = w.abs().mean(dim=-1)

        l1_loss = l1.mean()
        p = l1 / (l1.sum() + 1e-8)
        entropy_loss = -(p * (p + 1e-8).log()).sum()

        return regularise_activation * l1_loss + regularise_entropy * entropy_loss
