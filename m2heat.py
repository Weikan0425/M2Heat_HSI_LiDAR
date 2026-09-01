"""M2Heat: multimodal hyperspectral and auxiliary-image classifier.

The implementation keeps the active modules of the original M2Heat model in
one file. Inputs use channels-first tensors with shape ``[B, C, P, P]``.
"""

from __future__ import annotations

import math
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F


class LayerNorm2d(nn.LayerNorm):
    """LayerNorm over channels for a channels-first feature map."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 3, 1).contiguous()
        x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        return x.permute(0, 3, 1, 2).contiguous()


class Mlp(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: int | None = None,
        out_features: int | None = None,
        channels_first: bool = False,
    ) -> None:
        super().__init__()
        hidden_features = hidden_features or in_features
        out_features = out_features or in_features
        linear = partial(nn.Conv2d, kernel_size=1) if channels_first else nn.Linear
        self.fc1 = linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = linear(hidden_features, out_features)
        self.drop = nn.Dropout(0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.drop(self.act(self.fc1(x)))
        return self.drop(self.fc2(x))


class StemLayer(nn.Module):
    """Two convolutions used independently by the two input modalities."""

    def __init__(self, in_chans: int, out_chans: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_chans, out_chans // 2, 3, padding=1)
        self.norm1 = nn.BatchNorm2d(out_chans // 2)
        self.act = nn.GELU()
        self.conv2 = nn.Conv2d(out_chans // 2, out_chans, 3, padding=1)
        self.norm2 = nn.BatchNorm2d(out_chans)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act(self.norm1(self.conv1(x)))
        return self.norm2(self.conv2(x))


class Classifier(nn.Module):
    """Global-average classifier matching the original implementation."""

    def __init__(self, dim: int, num_classes: int) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(dim, dim // 2, 1),
            nn.BatchNorm2d(dim // 2),
            nn.LeakyReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.head = nn.Conv2d(dim // 2, num_classes, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.head(self.features(x)).flatten(1)
        # Kept for compatibility with the released training protocol.
        return F.softmax(x, dim=1)


class HCBlock(nn.Module):
    """Heat-conduction operator implemented in the 2-D cosine basis."""

    def __init__(self, res: int, dim: int, hidden_dim: int) -> None:
        super().__init__()
        if dim != hidden_dim:
            raise ValueError("M2Heat uses depthwise HCBlock channels: dim must equal hidden_dim")
        self.res = res
        self.hidden_dim = hidden_dim
        self.dwconv = nn.Conv2d(dim, hidden_dim, 3, padding=1, groups=hidden_dim)
        self.linear = nn.Linear(hidden_dim, 2 * hidden_dim)
        self.out_norm = nn.LayerNorm(hidden_dim)
        self.out_linear = nn.Linear(hidden_dim, hidden_dim)
        self.to_k = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU())

    @staticmethod
    def cosine_map(size: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        positions = (torch.arange(size, device=device, dtype=dtype)[None, :] + 0.5) / size
        frequencies = torch.arange(size, device=device, dtype=dtype)[:, None]
        result = torch.cos(frequencies * positions * torch.pi) * math.sqrt(2.0 / size)
        result[0] /= math.sqrt(2.0)
        return result

    @staticmethod
    def decay_map(height: int, width: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        rows = torch.linspace(0, torch.pi, height + 1, device=device, dtype=dtype)[:height, None]
        cols = torch.linspace(0, torch.pi, width + 1, device=device, dtype=dtype)[:width, None].T
        return torch.exp(-(rows.square() + cols.square()))

    def forward(self, x: torch.Tensor, freq_embed: torch.Tensor) -> torch.Tensor:
        batch, channels, height, width = x.shape
        x = self.dwconv(x)
        x = self.linear(x.permute(0, 2, 3, 1).contiguous())
        x, gate = x.chunk(2, dim=-1)

        cosine_h = self.cosine_map(height, x.device, x.dtype)
        cosine_w = self.cosine_map(width, x.device, x.dtype)
        decay = self.decay_map(height, width, x.device, x.dtype)
        k_eff = self.to_k(freq_embed)

        modes_h, modes_w = cosine_h.shape[0], cosine_w.shape[0]
        x = F.conv1d(x.reshape(batch, height, -1), cosine_h.reshape(modes_h, height, 1))
        x = F.conv1d(x.reshape(-1, width, channels), cosine_w.reshape(modes_w, width, 1))
        x = x.reshape(batch, modes_h, modes_w, channels)
        x = torch.einsum("bnmc,nmc->bnmc", x, decay[:, :, None].pow(k_eff))
        x = F.conv1d(x.reshape(batch, modes_h, -1), cosine_h.T.reshape(height, modes_h, 1))
        x = F.conv1d(x.reshape(-1, modes_w, channels), cosine_w.T.reshape(width, modes_w, 1))
        x = x.reshape(batch, height, width, channels)

        x = self.out_norm(x) * F.silu(gate)
        return self.out_linear(x).permute(0, 3, 1, 2).contiguous()


class HCLayer(nn.Module):
    """Stacked heat-conduction blocks with residual feed-forward updates."""

    def __init__(self, res: int, dim: int, hidden_dim: int, depth: int = 2) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            nn.ModuleDict(
                {
                    "norm1": LayerNorm2d(hidden_dim),
                    "hco": HCBlock(res, dim, hidden_dim),
                    "norm2": LayerNorm2d(hidden_dim),
                    "ffn": Mlp(dim, dim // 2, dim, channels_first=True),
                }
            )
            for _ in range(depth)
        )

    def forward(self, x: torch.Tensor, freq_embed: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = x + layer["hco"](layer["norm1"](x), freq_embed)
            x = x + layer["ffn"](layer["norm2"](x))
        return x


class CFF(nn.Module):
    """Cross-frequency fusion of the two modality streams."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.pre1 = nn.Conv2d(dim, dim, 1)
        self.pre2 = nn.Conv2d(dim, dim, 1)
        self.amp_fuse = nn.Sequential(
            nn.Conv2d(dim * 2, dim, 1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(dim, dim, 1),
        )
        self.pha_fuse = nn.Sequential(
            nn.Conv2d(dim * 2, dim, 1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(dim, dim, 1),
        )
        self.post = nn.Conv2d(dim, dim, 1)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        height, width = x1.shape[-2:]
        x1f = torch.fft.rfft2(self.pre1(x1) + 1e-8, norm="backward")
        x2f = torch.fft.rfft2(self.pre2(x2) + 1e-8, norm="backward")
        amp = self.amp_fuse(torch.cat((x1f.abs(), x2f.abs()), dim=1))
        phase = self.pha_fuse(torch.cat((x1f.angle(), x2f.angle()), dim=1))
        fused = torch.complex(amp * torch.cos(phase) + 1e-8, amp * torch.sin(phase) + 1e-8)
        fused = torch.fft.irfft2(fused, s=(height, width), norm="backward").abs()
        return self.post(fused)


def _initialize_fve(tensor: torch.Tensor, mode: str, value: float, std: float) -> None:
    mode = mode.lower()
    with torch.no_grad():
        if mode in {"trunc_normal", "truncated_normal"}:
            nn.init.trunc_normal_(tensor, std=std)
        elif mode in {"constant", "fixed"}:
            tensor.fill_(value)
        elif mode in {"zero", "zeros"}:
            tensor.zero_()
        else:
            raise ValueError("fves_init must be trunc_normal, constant, or zero")


class M2Heat(nn.Module):
    """Complete M2Heat classifier for HSI plus LiDAR/DSM/SAR auxiliary data."""

    def __init__(
        self,
        patch_size: int,
        num_classes: int,
        num_patches: tuple[int, int] | list[int],
        dim: int = 64,
        hidden_dim: int = 64,
        depth: int = 2,
        fves_init: str = "trunc_normal",
        fves_value: float = 0.0,
        fves_std: float = 0.02,
        fves_trainable: bool = True,
    ) -> None:
        super().__init__()
        if dim != hidden_dim:
            raise ValueError("M2Heat requires dim == hidden_dim")

        self.stem_hsi = StemLayer(num_patches[0], dim * 2)
        self.stem_aux = StemLayer(num_patches[1], dim * 2)
        self.hclayer_hsi = HCLayer(patch_size, dim, hidden_dim, depth)
        self.hclayer_aux = HCLayer(patch_size, dim, hidden_dim, depth)
        self.hclayer_fuse = HCLayer(patch_size, dim * 2, hidden_dim * 2, depth)
        self.cff = CFF(dim)
        self.proj = nn.Conv2d(dim * 2, dim, 1)

        for name, channels in (("fves_hsi", dim), ("fves_aux", dim), ("fves_fuse", dim * 2)):
            tensor = torch.zeros(patch_size, patch_size, channels)
            _initialize_fve(tensor, fves_init, fves_value, fves_std)
            if fves_trainable:
                setattr(self, name, nn.Parameter(tensor))
            else:
                self.register_buffer(name, tensor)

        self.cof1 = nn.Parameter(torch.ones(1))
        self.cof2 = nn.Parameter(torch.ones(1))
        self.cls = Classifier(dim, num_classes)

    def forward(self, hsi: torch.Tensor, auxiliary: torch.Tensor) -> torch.Tensor:
        hsi = self.stem_hsi(hsi)
        auxiliary = self.stem_aux(auxiliary)
        hsi, hsi_gate = hsi.chunk(2, dim=1)
        auxiliary, aux_gate = auxiliary.chunk(2, dim=1)
        fused = torch.cat((hsi_gate, aux_gate), dim=1)

        hsi = self.hclayer_hsi(hsi, self.fves_hsi)
        auxiliary = self.hclayer_aux(auxiliary, self.fves_aux)
        fused = self.hclayer_fuse(fused, self.fves_fuse)
        output = self.cff(hsi, auxiliary) * self.cof1 + self.proj(fused) * self.cof2
        return self.cls(output)


__all__ = ["M2Heat"]
