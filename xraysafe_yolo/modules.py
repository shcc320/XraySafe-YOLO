from __future__ import annotations

from typing import Iterable, List, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBNAct(nn.Module):
    """Small convolution block used by the custom modules.

    This block avoids importing Ultralytics internals so that the proposed
    modules can be unit-tested independently of a specific Ultralytics release.
    """

    def __init__(self, c1: int, c2: int, k: int = 1, s: int = 1, p: int | None = None, groups: int = 1):
        super().__init__()
        if p is None:
            p = k // 2
        self.conv = nn.Conv2d(c1, c2, k, s, p, groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


class DWConvBNAct(nn.Module):
    """Depthwise-separable convolution used by LSAFF for low overhead."""

    def __init__(self, c1: int, c2: int, k: int = 3):
        super().__init__()
        self.depthwise = ConvBNAct(c1, c1, k=k, groups=c1)
        self.pointwise = ConvBNAct(c1, c2, k=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pointwise(self.depthwise(x))


class CGOA(nn.Module):
    """Contour-Guided Occlusion Attention.

    CGOA is a single-input/single-output module.  It keeps the feature tensor
    shape unchanged unless c1 != c2, in which case a 1x1 projection is used.
    The module has two branches:

    * a lightweight local-variation branch that estimates a contour-sensitive
      spatial response map without additional contour annotations;
    * a channel-recalibration branch using average and max pooled descriptors.

    The output matches the manuscript equation:
        F' = F * (1 + sigmoid(M_c)) * A_ch,
    followed by an optional 1x1 projection when the channel count changes.
    """

    def __init__(self, c1: int, c2: int | None = None, reduction: int = 16, residual_scale: float = 1.0):
        super().__init__()
        c2 = c1 if c2 is None else int(c2)
        c1 = int(c1)
        hidden = max(c1 // int(reduction), 8)
        self.residual_scale = float(residual_scale)

        self.contour = nn.Sequential(
            nn.Conv2d(c1, c1, kernel_size=3, stride=1, padding=1, groups=c1, bias=False),
            nn.BatchNorm2d(c1),
            nn.SiLU(inplace=True),
            nn.Conv2d(c1, 1, kernel_size=1, stride=1, padding=0, bias=True),
        )
        self.shared_mlp = nn.Sequential(
            nn.Conv2d(c1, hidden, kernel_size=1, bias=False),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, c1, kernel_size=1, bias=True),
        )
        self.project = nn.Identity() if c1 == c2 else ConvBNAct(c1, c2, k=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        spatial = torch.sigmoid(self.contour(x))
        avg = F.adaptive_avg_pool2d(x, 1)
        mx = F.adaptive_max_pool2d(x, 1)
        channel = torch.sigmoid(self.shared_mlp(avg) + self.shared_mlp(mx))
        y = x * (1.0 + self.residual_scale * spatial) * channel
        return self.project(y)


class LSAFF(nn.Module):
    """Lightweight Scale-Adaptive Feature Fusion.

    Ultralytics calls this module with a list of feature maps.  Each instance
    produces one output feature scale.  Therefore a full three-scale LSAFF neck
    is represented by three YAML layers that share the same input layers but use
    target_index=0, 1, and 2 respectively.

    Args:
        c1: List of input channel counts, e.g. [P3_channels, P4_channels, P5_channels].
        c2: Output channel count for the selected output scale.
        target_index: Which input scale defines the output spatial size.
        use_post: Whether to apply an additional lightweight post-fusion block.
        residual: If true, preserve the target-scale feature and add the
            cross-scale fusion through a learnable gate. This keeps the initial
            behavior close to the YOLO/CGOA feature path and reduces destructive
            feature replacement.
        gate_init: Initial gate logit used when residual=True. -2.0 gives a
            sigmoid gate of about 0.12.
        spatial_fusion: If true, predict per-location scale weights instead of
            using one global weight per input scale.
    """

    def __init__(
        self,
        c1: Sequence[int] | int,
        c2: int,
        target_index: int = 0,
        use_post: bool = True,
        residual: bool = False,
        gate_init: float = -2.0,
        spatial_fusion: bool = False,
    ):
        super().__init__()
        if isinstance(c1, int):
            c1 = [c1]
        self.c1 = [int(c) for c in c1]
        self.c2 = int(c2)
        self.target_index = int(target_index)
        self.residual = bool(residual)
        self.spatial_fusion = bool(spatial_fusion)
        if not (0 <= self.target_index < len(self.c1)):
            raise ValueError(f"target_index={self.target_index} is invalid for {len(self.c1)} inputs")
        self.align = nn.ModuleList([DWConvBNAct(c, self.c2, k=3) for c in self.c1])
        if self.spatial_fusion:
            self.scale_predictors = nn.ModuleList([nn.Conv2d(self.c2, 1, kernel_size=1) for _ in self.c1])
            for pred in self.scale_predictors:
                nn.init.zeros_(pred.weight)
                nn.init.zeros_(pred.bias)
        else:
            self.scale_logits = nn.Parameter(torch.zeros(len(self.c1), dtype=torch.float32))
        self.post = DWConvBNAct(self.c2, self.c2, k=3) if use_post else nn.Identity()
        if self.residual:
            target_c = self.c1[self.target_index]
            self.identity = nn.Identity() if target_c == self.c2 else ConvBNAct(target_c, self.c2, k=1)
            self.gate_logit = nn.Parameter(torch.tensor(float(gate_init), dtype=torch.float32))

    def forward(self, xs: Sequence[torch.Tensor] | torch.Tensor) -> torch.Tensor:
        if isinstance(xs, torch.Tensor):
            xs = [xs]
        xs = list(xs)
        if len(xs) != len(self.align):
            raise ValueError(f"LSAFF expected {len(self.align)} input tensors, got {len(xs)}")
        target_hw = xs[self.target_index].shape[-2:]
        aligned: List[torch.Tensor] = []
        for x, block in zip(xs, self.align):
            y = block(x)
            if y.shape[-2:] != target_hw:
                y = F.interpolate(y, size=target_hw, mode="nearest")
            aligned.append(y)

        if self.spatial_fusion:
            logits = torch.stack([pred(y) for pred, y in zip(self.scale_predictors, aligned)], dim=1)
            weights = torch.softmax(logits, dim=1)
            fused = None
            for i, y in enumerate(aligned):
                weighted = y * weights[:, i]
                fused = weighted if fused is None else fused + weighted
        else:
            weights = torch.softmax(self.scale_logits, dim=0)
            fused = None
            for y, w in zip(aligned, weights):
                y = y * w
                fused = y if fused is None else fused + y

        fused = self.post(fused)
        if self.residual:
            identity = self.identity(xs[self.target_index])
            if identity.shape[-2:] != target_hw:
                identity = F.interpolate(identity, size=target_hw, mode="nearest")
            return identity + torch.sigmoid(self.gate_logit) * fused
        return fused
