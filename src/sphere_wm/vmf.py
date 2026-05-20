"""Hyperspherical Uniform + von Mises-Fisher distributions on S^(m-1).

Faithful port of cells 7-9 in colab_codes/ablation_sphere_vae.ipynb
(based on nicola-decao/s-vae-pytorch). Kept self-contained: no external
dependencies beyond torch + scipy + numpy.
"""
from __future__ import annotations

import math
from numbers import Number

import numpy as np
import scipy.special
import torch
import torch.distributions


class IveFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, v, z):
        assert isinstance(v, Number), "v must be a scalar"
        ctx.save_for_backward(z)
        ctx.v = v
        z_cpu = z.data.cpu().numpy()
        if np.isclose(v, 0):
            output = scipy.special.i0e(z_cpu, dtype=z_cpu.dtype)
        elif np.isclose(v, 1):
            output = scipy.special.i1e(z_cpu, dtype=z_cpu.dtype)
        else:
            output = scipy.special.ive(v, z_cpu, dtype=z_cpu.dtype)
        return torch.Tensor(output).to(z.device)

    @staticmethod
    def backward(ctx, grad_output):
        z = ctx.saved_tensors[-1]
        return None, grad_output * (ive(ctx.v - 1, z) - ive(ctx.v, z) * (ctx.v + z) / z)


ive = IveFunction.apply


def ive_fraction_approx2(v, z, eps=1e-20):
    def delta_a(a):
        lamb = v + (a - 1.0) / 2.0
        return (v - 0.5) + lamb / (
            2 * torch.sqrt(torch.clamp(lamb ** 2 + z ** 2, min=eps))
        )

    d0 = delta_a(0.0)
    d2 = delta_a(2.0)
    B0 = z / (d0 + torch.sqrt(torch.clamp(d0 ** 2 + z ** 2, min=eps)))
    B2 = z / (d2 + torch.sqrt(torch.clamp(d2 ** 2 + z ** 2, min=eps)))
    return (B0 + B2) / 2.0


class HypersphericalUniform(torch.distributions.Distribution):
    arg_constraints = {}
    support = torch.distributions.constraints.real
    has_rsample = False
    _mean_carrier_measure = 0

    def __init__(self, dim, validate_args=None, device="cpu"):
        super().__init__(torch.Size([dim]), validate_args=validate_args)
        self._dim = dim
        self._device = (
            torch.device(device) if not isinstance(device, torch.device) else device
        )

    @property
    def dim(self):
        return self._dim

    @property
    def device(self):
        return self._device

    def sample(self, shape=torch.Size()):
        shape = shape if isinstance(shape, torch.Size) else torch.Size([shape])
        out = (
            torch.distributions.Normal(0, 1)
            .sample(shape + torch.Size([self._dim + 1]))
            .to(self.device)
        )
        return out / out.norm(dim=-1, keepdim=True)

    def entropy(self):
        return self._log_surface_area()

    def log_prob(self, x):
        return -torch.ones(x.shape[:-1], device=self.device) * self._log_surface_area()

    def _log_surface_area(self):
        lgamma = torch.lgamma(torch.tensor([(self._dim + 1) / 2]).to(self.device))
        return math.log(2) + ((self._dim + 1) / 2) * math.log(math.pi) - lgamma


class VonMisesFisher(torch.distributions.Distribution):
    arg_constraints = {
        "loc": torch.distributions.constraints.real,
        "scale": torch.distributions.constraints.positive,
    }
    support = torch.distributions.constraints.real
    has_rsample = True
    _mean_carrier_measure = 0

    def __init__(self, loc, scale, validate_args=None, k=1):
        self.dtype = loc.dtype
        self.loc = loc
        self.scale = scale
        self.device = loc.device
        self.__m = loc.shape[-1]
        self.__e1 = torch.Tensor([1.0] + [0] * (loc.shape[-1] - 1)).to(self.device)
        self.k = k
        super().__init__(self.loc.size(), validate_args=validate_args)

    @property
    def mean(self):
        return self.loc * ive_fraction_approx2(self.__m / 2, self.scale)

    @property
    def stddev(self):
        return self.scale

    def sample(self, shape=torch.Size()):
        with torch.no_grad():
            return self.rsample(shape)

    def rsample(self, shape=torch.Size()):
        shape = shape if isinstance(shape, torch.Size) else torch.Size([shape])
        w = (
            self.__sample_w3(shape=shape)
            if self.__m == 3
            else self.__sample_w_rej(shape=shape)
        )
        v = (
            torch.distributions.Normal(0, 1)
            .sample(shape + torch.Size(self.loc.shape))
            .to(self.device)
            .transpose(0, -1)[1:]
        ).transpose(0, -1)
        v = v / v.norm(dim=-1, keepdim=True)
        w_ = torch.sqrt(torch.clamp(1 - w ** 2, 1e-10))
        x = torch.cat((w, w_ * v), -1)
        z = self.__householder_rotation(x)
        return z.type(self.dtype)

    def __sample_w3(self, shape):
        shape = shape + torch.Size(self.scale.shape)
        u = torch.distributions.Uniform(0, 1).sample(shape).to(self.device)
        self.__w = (
            1
            + torch.stack(
                [torch.log(u), torch.log(1 - u) - 2 * self.scale], dim=0
            ).logsumexp(0)
            / self.scale
        )
        return self.__w

    def __sample_w_rej(self, shape):
        c = torch.sqrt((4 * (self.scale ** 2)) + (self.__m - 1) ** 2)
        b_true = (-2 * self.scale + c) / (self.__m - 1)
        b_app = (self.__m - 1) / (4 * self.scale)
        s = torch.min(
            torch.max(
                torch.tensor([0.0], dtype=self.dtype, device=self.device),
                self.scale - 10,
            ),
            torch.tensor([1.0], dtype=self.dtype, device=self.device),
        )
        b = b_app * s + b_true * (1 - s)
        a = (self.__m - 1 + 2 * self.scale + c) / 4
        d = (4 * a * b) / (1 + b) - (self.__m - 1) * math.log(self.__m - 1)
        self.__b, (self.__e, self.__w) = b, self.__while_loop(b, a, d, shape, k=self.k)
        return self.__w

    @staticmethod
    def first_nonzero(x, dim, invalid_val=-1):
        mask = x > 0
        idx = torch.where(
            mask.any(dim=dim),
            mask.float().argmax(dim=1).squeeze(),
            torch.tensor(invalid_val, device=x.device),
        )
        return idx

    def __while_loop(self, b, a, d, shape, k=20, eps=1e-20):
        b, a, d = [
            e.repeat(*shape, *([1] * len(self.scale.shape))).reshape(-1, 1)
            for e in (b, a, d)
        ]
        w, e, bool_mask = (
            torch.zeros_like(b).to(self.device),
            torch.zeros_like(b).to(self.device),
            (torch.ones_like(b) == 1).to(self.device),
        )
        sample_shape = torch.Size([b.shape[0], k])
        shape = shape + torch.Size(self.scale.shape)
        while bool_mask.sum() != 0:
            con1 = torch.tensor((self.__m - 1) / 2, dtype=torch.float64)
            con2 = torch.tensor((self.__m - 1) / 2, dtype=torch.float64)
            e_ = (
                torch.distributions.Beta(con1, con2)
                .sample(sample_shape)
                .to(self.device)
                .type(self.dtype)
            )
            u = (
                torch.distributions.Uniform(0 + eps, 1 - eps)
                .sample(sample_shape)
                .to(self.device)
                .type(self.dtype)
            )
            w_ = (1 - (1 + b) * e_) / (1 - (1 - b) * e_)
            t = (2 * a * b) / (1 - (1 - b) * e_)
            accept = ((self.__m - 1.0) * t.log() - t + d) > torch.log(u)
            accept_idx = self.first_nonzero(accept, dim=-1, invalid_val=-1).unsqueeze(1)
            accept_idx_clamped = accept_idx.clamp(0)
            w_ = w_.gather(1, accept_idx_clamped.view(-1, 1))
            e_ = e_.gather(1, accept_idx_clamped.view(-1, 1))
            reject = accept_idx < 0
            accept = ~reject
            w[bool_mask * accept] = w_[bool_mask * accept]
            e[bool_mask * accept] = e_[bool_mask * accept]
            bool_mask[bool_mask * accept] = reject[bool_mask * accept]
        return e.reshape(shape), w.reshape(shape)

    def __householder_rotation(self, x):
        u = self.__e1 - self.loc
        u = u / (u.norm(dim=-1, keepdim=True) + 1e-5)
        return x - 2 * (x * u).sum(-1, keepdim=True) * u

    def entropy(self):
        out = -self.scale * ive_fraction_approx2(self.__m / 2, self.scale)
        return out.view(*(out.shape[:-1])) + self._log_normalization()

    def log_prob(self, x):
        return self._log_unnormalized_prob(x) - self._log_normalization()

    def _log_unnormalized_prob(self, x):
        out = self.scale * (self.loc * x).sum(-1, keepdim=True)
        return out.view(*(out.shape[:-1]))

    def _log_normalization(self):
        out = -(
            (self.__m / 2 - 1) * torch.log(self.scale)
            - (self.__m / 2) * math.log(2 * math.pi)
            - (self.scale + torch.log(ive(self.__m / 2 - 1, self.scale) + 1e-30))
        )
        return out.view(*(out.shape[:-1]))


def kl_vmf_uniform(vmf: VonMisesFisher, hyu: HypersphericalUniform) -> torch.Tensor:
    return -vmf.entropy() + hyu.entropy()
