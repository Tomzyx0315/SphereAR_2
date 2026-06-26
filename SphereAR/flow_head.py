import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def _inverse_softplus(x: float) -> float:
    return math.log(math.exp(x) - 1.0)


def _sum_except_batch(x):
    return x.reshape(x.shape[0], -1).sum(dim=1)


def _gather_bins(values, indices):
    return values.gather(-1, indices.unsqueeze(-1)).squeeze(-1)


def rational_quadratic_spline(
    inputs,
    unnormalized_widths,
    unnormalized_heights,
    unnormalized_derivatives,
    inverse=False,
    tail_bound=6.0,
    min_bin_width=1e-3,
    min_bin_height=1e-3,
    min_derivative=1e-3,
):
    num_bins = unnormalized_widths.shape[-1]
    left, right = -tail_bound, tail_bound
    bottom, top = -tail_bound, tail_bound
    width = right - left
    height = top - bottom

    if min_bin_width * num_bins > width:
        raise ValueError("min_bin_width is too large for the number of bins")
    if min_bin_height * num_bins > height:
        raise ValueError("min_bin_height is too large for the number of bins")

    inside = (inputs >= left) & (inputs <= right)
    clipped_inputs = inputs.clamp(left + 1e-6, right - 1e-6)

    widths = F.softmax(unnormalized_widths, dim=-1)
    widths = min_bin_width + (width - min_bin_width * num_bins) * widths
    cumwidths = torch.cumsum(widths, dim=-1)
    cumwidths = F.pad(cumwidths, pad=(1, 0), mode="constant", value=0.0)
    cumwidths = cumwidths + left
    cumwidths[..., 0] = left
    cumwidths[..., -1] = right

    heights = F.softmax(unnormalized_heights, dim=-1)
    heights = min_bin_height + (height - min_bin_height * num_bins) * heights
    cumheights = torch.cumsum(heights, dim=-1)
    cumheights = F.pad(cumheights, pad=(1, 0), mode="constant", value=0.0)
    cumheights = cumheights + bottom
    cumheights[..., 0] = bottom
    cumheights[..., -1] = top

    derivatives = min_derivative + F.softplus(unnormalized_derivatives)
    derivatives = torch.cat(
        [
            torch.ones_like(derivatives[..., :1]),
            derivatives[..., 1:-1],
            torch.ones_like(derivatives[..., -1:]),
        ],
        dim=-1,
    )

    if inverse:
        bin_idx = torch.searchsorted(cumheights, clipped_inputs.unsqueeze(-1)).squeeze(
            -1
        )
    else:
        bin_idx = torch.searchsorted(cumwidths, clipped_inputs.unsqueeze(-1)).squeeze(
            -1
        )
    bin_idx = (bin_idx - 1).clamp(min=0, max=num_bins - 1)

    input_cumwidths = _gather_bins(cumwidths, bin_idx)
    input_bin_widths = _gather_bins(widths, bin_idx)
    input_cumheights = _gather_bins(cumheights, bin_idx)
    input_heights = _gather_bins(heights, bin_idx)
    input_delta = input_heights / input_bin_widths
    input_derivatives = _gather_bins(derivatives, bin_idx)
    input_derivatives_plus_one = _gather_bins(derivatives, bin_idx + 1)

    if inverse:
        y_minus_cumheight = clipped_inputs - input_cumheights
        common = input_derivatives + input_derivatives_plus_one - 2 * input_delta
        a = y_minus_cumheight * common + input_heights * (
            input_delta - input_derivatives
        )
        b = input_heights * input_derivatives - y_minus_cumheight * common
        c = -input_delta * y_minus_cumheight
        discriminant = (b.square() - 4 * a * c).clamp_min(0.0)
        root = (2 * c) / (-b - torch.sqrt(discriminant).clamp_min(1e-12))
        theta = root.clamp(0.0, 1.0)
        outputs = theta * input_bin_widths + input_cumwidths
    else:
        theta = (clipped_inputs - input_cumwidths) / input_bin_widths
        theta = theta.clamp(0.0, 1.0)
        theta_one_minus_theta = theta * (1 - theta)
        numerator = input_heights * (
            input_delta * theta.square() + input_derivatives * theta_one_minus_theta
        )
        denominator = input_delta + (
            input_derivatives + input_derivatives_plus_one - 2 * input_delta
        ) * theta_one_minus_theta
        outputs = input_cumheights + numerator / denominator.clamp_min(1e-12)

    theta_one_minus_theta = theta * (1 - theta)
    derivative_numerator = input_delta.square() * (
        input_derivatives_plus_one * theta.square()
        + 2 * input_delta * theta_one_minus_theta
        + input_derivatives * (1 - theta).square()
    )
    derivative_denominator = (
        input_delta
        + (input_derivatives + input_derivatives_plus_one - 2 * input_delta)
        * theta_one_minus_theta
    ).square()
    logabsdet = torch.log(derivative_numerator.clamp_min(1e-12)) - torch.log(
        derivative_denominator.clamp_min(1e-12)
    )
    if inverse:
        logabsdet = -logabsdet

    outputs = torch.where(inside, outputs, inputs)
    logabsdet = torch.where(inside, logabsdet, torch.zeros_like(logabsdet))
    return outputs, logabsdet


class ResidualMLP(nn.Module):
    def __init__(self, dim, hidden_dim, out_dim, depth):
        super().__init__()
        self.in_proj = nn.Linear(dim, hidden_dim)
        self.blocks = nn.ModuleList(
            [
                nn.Sequential(
                    nn.LayerNorm(hidden_dim),
                    nn.Linear(hidden_dim, hidden_dim * 2),
                    nn.SiLU(),
                    nn.Linear(hidden_dim * 2, hidden_dim),
                )
                for _ in range(depth)
            ]
        )
        self.out_norm = nn.LayerNorm(hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, out_dim)

    def forward(self, x):
        x = self.in_proj(x)
        for block in self.blocks:
            x = x + block(x)
        return self.out_proj(self.out_norm(x))


class InvertibleLinear(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.weight_raw = nn.Parameter(torch.randn(dim, dim) * 0.02)
        self.log_scale = nn.Parameter(torch.zeros(dim))

    def _weight(self):
        skew = self.weight_raw - self.weight_raw.t()
        rotation = torch.matrix_exp(skew)
        return rotation * torch.exp(self.log_scale).view(1, -1)

    def forward(self, x, inverse=False):
        weight = self._weight()
        if inverse:
            weight = torch.linalg.inv(weight)
        y = x @ weight.t()
        logabsdet = self.log_scale.float().sum()
        if inverse:
            logabsdet = -logabsdet
        return y, logabsdet.to(x).expand(x.shape[0])


class RQSCoupling(nn.Module):
    def __init__(
        self,
        dim,
        cond_dim,
        hidden_dim,
        num_bins,
        mask,
        conditioner_depth,
        tail_bound,
    ):
        super().__init__()
        self.dim = dim
        self.num_bins = num_bins
        self.tail_bound = tail_bound
        self.register_buffer("mask", mask.view(1, dim))
        self.cond_proj = nn.Sequential(
            nn.LayerNorm(cond_dim),
            nn.Linear(cond_dim, hidden_dim),
            nn.SiLU(),
        )
        params_per_dim = 3 * num_bins + 1
        self.net = ResidualMLP(
            dim + hidden_dim,
            hidden_dim,
            dim * params_per_dim,
            conditioner_depth,
        )
        nn.init.zeros_(self.net.out_proj.weight)
        nn.init.zeros_(self.net.out_proj.bias)
        deriv_bias = _inverse_softplus(1.0 - 1e-3)
        with torch.no_grad():
            bias = self.net.out_proj.bias.view(dim, params_per_dim)
            bias[:, 2 * num_bins :] = deriv_bias

    def initialize_identity(self):
        nn.init.zeros_(self.net.out_proj.weight)
        nn.init.zeros_(self.net.out_proj.bias)
        deriv_bias = _inverse_softplus(1.0 - 1e-3)
        with torch.no_grad():
            bias = self.net.out_proj.bias.view(self.dim, 3 * self.num_bins + 1)
            bias[:, 2 * self.num_bins :] = deriv_bias

    def _params(self, x, cond):
        masked_x = x * self.mask
        h = torch.cat([masked_x, self.cond_proj(cond)], dim=-1)
        params = self.net(h).view(x.shape[0], self.dim, 3 * self.num_bins + 1)
        widths = params[..., : self.num_bins]
        heights = params[..., self.num_bins : 2 * self.num_bins]
        derivatives = params[..., 2 * self.num_bins :]
        return widths, heights, derivatives

    def forward(self, x, cond, inverse=False):
        widths, heights, derivatives = self._params(x, cond)
        transformed, logabsdet = rational_quadratic_spline(
            x,
            widths,
            heights,
            derivatives,
            inverse=inverse,
            tail_bound=self.tail_bound,
        )
        transform_mask = 1.0 - self.mask
        y = x * self.mask + transformed * transform_mask
        logabsdet = _sum_except_batch(logabsdet * transform_mask)
        return y, logabsdet


class FlowStep(nn.Module):
    def __init__(
        self,
        dim,
        cond_dim,
        hidden_dim,
        num_bins,
        mask,
        conditioner_depth,
        tail_bound,
    ):
        super().__init__()
        self.coupling = RQSCoupling(
            dim,
            cond_dim,
            hidden_dim,
            num_bins,
            mask,
            conditioner_depth,
            tail_bound,
        )
        self.linear = InvertibleLinear(dim)

    def forward(self, x, cond):
        x, logdet_coupling = self.coupling(x, cond, inverse=False)
        x, logdet_linear = self.linear(x, inverse=False)
        return x, logdet_coupling + logdet_linear

    def inverse(self, x, cond):
        x, logdet_linear = self.linear(x, inverse=True)
        x, logdet_coupling = self.coupling(x, cond, inverse=True)
        return x, logdet_linear + logdet_coupling


class ConditionalSplineFlow(nn.Module):
    def __init__(
        self,
        dim,
        cond_dim,
        hidden_dim,
        num_layers,
        num_bins,
        conditioner_depth,
        tail_bound,
        base_scale_bound,
    ):
        super().__init__()
        self.dim = dim
        self.base_scale_bound = base_scale_bound
        self.base_net = ResidualMLP(cond_dim, hidden_dim, dim * 2, depth=2)
        nn.init.zeros_(self.base_net.out_proj.weight)
        nn.init.zeros_(self.base_net.out_proj.bias)
        self.steps = nn.ModuleList()
        base_mask = (torch.arange(dim) % 2).float()
        for i in range(num_layers):
            mask = base_mask if i % 2 == 0 else 1.0 - base_mask
            self.steps.append(
                FlowStep(
                    dim,
                    cond_dim,
                    hidden_dim,
                    num_bins,
                    mask,
                    conditioner_depth,
                    tail_bound,
                )
            )

    def initialize_weights(self):
        nn.init.zeros_(self.base_net.out_proj.weight)
        nn.init.zeros_(self.base_net.out_proj.bias)
        for step in self.steps:
            step.coupling.initialize_identity()

    def _base_stats(self, cond):
        mean, log_scale = self.base_net(cond).chunk(2, dim=-1)
        log_scale = self.base_scale_bound * torch.tanh(log_scale / self.base_scale_bound)
        return mean, log_scale

    def log_prob(self, x, cond):
        z = x
        logdet = x.new_zeros(x.shape[0])
        for step in reversed(self.steps):
            z, step_logdet = step.inverse(z, cond)
            logdet = logdet + step_logdet

        mean, log_scale = self._base_stats(cond)
        inv_scale = torch.exp(-log_scale)
        log_base = -0.5 * ((z - mean) * inv_scale).square()
        log_base = log_base - log_scale - 0.5 * math.log(2 * math.pi)
        return _sum_except_batch(log_base) + logdet

    def sample(self, cond, temperature=1.0):
        mean, log_scale = self._base_stats(cond)
        eps = torch.randn_like(mean)
        z = mean + eps * torch.exp(log_scale) * temperature
        x = z
        for step in self.steps:
            x, _ = step(x, cond)
        return x


class FlowHead(nn.Module):
    def __init__(
        self,
        ch_target,
        ch_cond,
        ch_latent,
        num_layers=8,
        num_bins=16,
        conditioner_depth=2,
        tail_bound=6.0,
        noise_std=0.01,
        base_scale_bound=2.0,
        grad_checkpointing=False,
    ):
        super().__init__()
        del grad_checkpointing
        self.ch_target = ch_target
        self.noise_std = noise_std
        self.flow = ConditionalSplineFlow(
            dim=ch_target,
            cond_dim=ch_cond,
            hidden_dim=ch_latent,
            num_layers=num_layers,
            num_bins=num_bins,
            conditioner_depth=conditioner_depth,
            tail_bound=tail_bound,
            base_scale_bound=base_scale_bound,
        )

    def forward(self, target, z):
        with torch.autocast(device_type=target.device.type, enabled=False):
            target = target.float()
            z = z.float()
            if self.noise_std > 0.0 and self.training:
                target = target + self.noise_std * torch.randn_like(target)
            loss = -self.flow.log_prob(target, z).mean()
        return loss

    def sample(self, z, temperature=1.0):
        with torch.autocast(device_type=z.device.type, enabled=False):
            return self.flow.sample(z.float(), temperature=temperature)

    def initialize_weights(self):
        self.flow.initialize_weights()
