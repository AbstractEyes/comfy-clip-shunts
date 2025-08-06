from dataclasses import dataclass

import torch
from typing import Optional


import torch.nn.functional as F
import math

# --- helper for numerically‑stable slerp ---------------------------------
def _slerp(a: torch.Tensor, b: torch.Tensor, alpha: torch.Tensor, eps=1e-6) -> torch.Tensor:
    a_norm, b_norm = F.normalize(a, dim=-1, eps=eps), F.normalize(b, dim=-1, eps=eps)
    dot = (a_norm * b_norm).sum(dim=-1, keepdim=True).clamp(-1 + eps, 1 - eps)
    omega = torch.acos(dot)
    sin_omega = torch.sin(omega).clamp_min(eps)
    t1 = torch.sin((1 - alpha) * omega) / sin_omega
    t2 = torch.sin(alpha * omega) / sin_omega
    return t1 * a + t2 * b

class FoldingKernel:
    def apply(self,
              a: torch.Tensor,
              b: torch.Tensor,
              t: torch.Tensor,
              alpha: Optional[torch.Tensor] = None,
              context: Optional[dict] = None) -> torch.Tensor:
        raise NotImplementedError("All folding kernels must implement the `apply` method.")


class RigidFolding(FoldingKernel):
    def apply(self, a, b, t, alpha=None, context=None):
        return a


class FoldFolding(FoldingKernel):
    def apply(self, a, b, t, alpha=None, context=None):
        return a + (b - a) * t.unsqueeze(-1)


class ZipperFolding(FoldingKernel):
    def apply(self, a, b, t, alpha=None, context=None):
        mask = torch.arange(a.size(1), device=a.device) % 2 == 0
        return torch.where(mask.unsqueeze(0).unsqueeze(-1), a, b)


class RippleFolding(FoldingKernel):
    def apply(self, a, b, t, alpha=None, context=None):
        freq = context.get("ripple_freq", 2.0) if context else 2.0
        ripple = torch.sin(freq * torch.pi * t).unsqueeze(-1)
        return a + ripple * (b - a)


class SurgeFolding(FoldingKernel):
    def apply(self, a, b, t, alpha=None, context=None):
        intensity = context.get("surge_intensity", 5.0) if context else 5.0
        surge = 1 - torch.exp(-intensity * t)
        return a + (b - a) * surge.unsqueeze(-1)


class CollapseFolding(FoldingKernel):
    def apply(self, a, b, t, alpha=None, context=None):
        rate = context.get("collapse_rate", 1.0) if context else 1.0
        collapse = 1 - torch.exp(-rate * (1 - t))
        return b * collapse.unsqueeze(-1)


import torch.nn as nn

class ConcatFlattenFolding(FoldingKernel):
    def __init__(self):
        super().__init__()
        self.proj = None
        self.last_dim = None

    def apply(self, a: torch.Tensor, b: torch.Tensor, t=None, alpha=None, context=None):
        if a.ndim != 3 or b.ndim != 3:
            raise ValueError(f"[ConcatFlattenFolding] Expected [B,T,D] tensors, got {a.shape} and {b.shape}")

        B, T, D = a.shape
        key_dim = D

        if self.proj is None or self.last_dim != key_dim:
            self.last_dim = key_dim
            self.proj = nn.Linear(D * 2, D).to(a.device)

        concat = torch.cat([a, b], dim=-1)          # [B, T, 2D]
        flat = concat.view(B * T, -1)               # [B*T, 2D]
        out = self.proj(flat)                       # [B*T, D]
        return out.view(B, T, D)                    # [B, T, D]


class ZeusFolding(FoldingKernel):
    def apply(self, a, b, t, alpha=None, context=None):
        sharpness = context.get("zeus_force", 10.0) if context else 10.0
        mask = torch.sigmoid(sharpness * (t - 0.5)).unsqueeze(-1)
        return a * (1 - mask) + b * mask


class HeliosFolding(FoldingKernel):
    def apply(self, a, b, t, alpha=None, context=None):
        weight = torch.sin(torch.pi * t).unsqueeze(-1)
        return a + weight * (b - a)


class CascadeFolding(FoldingKernel):
    def apply(self, a, b, t, alpha=None, context=None):
        steps = context.get("cascade_steps", 4.0) if context else 4.0
        gate = torch.clamp(steps * t - 1, 0.0, 1.0).unsqueeze(-1)
        return a + gate * (b - a)


class InterpolateFolding(FoldingKernel):
    def apply(self, a, b, t, alpha=None, context=None):
        return a + (b - a) * t.unsqueeze(-1)


class SurgeFoldFolding(FoldingKernel):
    def apply(self, a, b, t, alpha=None, context=None):
        surge = SurgeFolding().apply(a, b, t, context=context)
        return FoldFolding().apply(a, surge, t, context=context)


class SlerpFolding(FoldingKernel):
    def apply(self, a, b, t, alpha=None, context=None):
        # `alpha` preferred (better scheduler resolution); fallback to `t`
        mix = alpha if alpha is not None else t
        return _slerp(a, b, mix.unsqueeze(-1))


# -- 1. Shiva: Icy gradient decay ---------------------------------------------
class ShivaFolding(FoldingKernel):
    def apply(self, a, b, t, alpha=None, context=None):
        decay_rate = context.get("shiva_cool", 4.0) if context else 4.0
        cold = torch.exp(-decay_rate * t).unsqueeze(-1)
        return a + (b - a) * (1.0 - cold)


# -- 2. Ifrit: Fiery waveform spikes ------------------------------------------
class IfritFolding(FoldingKernel):
    def apply(self, a, b, t, alpha=None, context=None):
        freq = context.get("ifrit_freq", 4.0)
        amp = context.get("ifrit_amp", 1.0)
        fire = (torch.sin(freq * torch.pi * t) ** 2).unsqueeze(-1) * amp
        return a + fire * (b - a)


# -- 3. Gilgamesh: Multi-vector projection from alpha -------------------------
class GilgameshFolding(FoldingKernel):
    def apply(self, a, b, t, alpha=None, context=None):
        beta_axes = context.get("gilgamesh_axes", [0.25, 0.33, 0.5, 0.66, 0.75])
        out = a
        for w in beta_axes:
            weight = torch.tensor(w, device=a.device).view(1, 1, 1)
            delta = weight * (b - a)
            out = out + delta * torch.sigmoid(alpha.unsqueeze(-1))
        return out


# -- 4. Hive: Internal scheduler cascade --------------------------------------
class HiveFolding(FoldingKernel):
    def apply(self, a, b, t, alpha=None, context=None):
        step_count = context.get("steps", 8)
        thresholds = context.get("hive_thresholds", [0.2, 0.4, 0.6, 0.8])
        schedulers = context.get("hive_kernels", ["rigid", "fold", "ripple", "zeus"])
        selected = "rigid"
        for th, name in zip(thresholds, schedulers):
            if t.mean().item() <= th:
                selected = name
                break
        return get_folding_kernel(selected).apply(a, b, t, alpha, context)


# -- 5. A_Walk: Dream‑based time walk ----------------------------------------
class AWalkFolding(FoldingKernel):
    def apply(self, a, b, t, alpha=None, context=None):
        walk_random = context.get("walk_random", 0.03) if context else 0.03
        walk_speed = context.get("walk_speed", 3) if context else 3
        drift = torch.sin(t * math.pi).unsqueeze(-1) ** walk_speed
        noise = torch.randn_like(a) * walk_random
        return a + drift * (b - a + noise)



class SlipFolding(FoldingKernel):
    """
    Implements the Slip Principle: entropic‑phase gating.
    Context must carry 'delta' (provided automatically by FieldWalker).
    """
    def apply(self, a, b, t, alpha=None, context=None):
        mix = alpha if alpha is not None else t
        delta = (context or {}).get("delta", b - a)
        phase = (delta * b).sum(dim=-1, keepdim=True)
        phase_gate = torch.sigmoid(phase)          # 0‑1 weighting
        adj = mix.unsqueeze(-1) * phase_gate
        return a + adj * delta



FOLDING_KERNELS: dict[str, FoldingKernel] = {
    "shiva": ShivaFolding(),
    "ifrit": IfritFolding(),
    "gilgamesh": GilgameshFolding(),
    "hive": HiveFolding(),
    "a_walk": AWalkFolding(),
    "rigid": RigidFolding(),
    "fold": FoldFolding(),
    "zipper": ZipperFolding(),
    "ripple": RippleFolding(),
    "surge": SurgeFolding(),
    "collapse": CollapseFolding(),
    "concat_flatten": ConcatFlattenFolding(),
    "zeus": ZeusFolding(),
    "helios": HeliosFolding(),
    "cascade": CascadeFolding(),
    "interpolate": InterpolateFolding(),
    "surge_fold": SurgeFoldFolding(),
    "slerp": SlerpFolding(),
    "slip": SlipFolding(),  # entropy‑weighted
}

@dataclass
class FoldingKernels:
    """
    A collection of available folding kernels.
    """

    shiva: str = "shiva"
    ifrit: str = "ifrit"
    gilgamesh: str = "gilgamesh"
    hive: str = "hive"
    a_walk: str = "a_walk"
    rigid: str = "rigid"
    fold: str = "fold"
    zipper: str = "zipper"
    ripple: str = "ripple"
    surge: str = "surge"
    collapse: str = "collapse"
    concat: str = "concat"
    zeus: str = "zeus"
    helios: str = "helios"
    cascade: str = "cascade"
    interpolate: str = "interpolate"
    surge_fold: str = "surge_fold"
    slerp: str = "slerp"
    slip: str = "slip"

    @staticmethod
    def to_list() -> list[str]:
        """
        Returns a sorted list of all folding kernel names.
        """
        return sorted([
            "shiva", "ifrit", "gilgamesh", "hive", "a_walk",
            "rigid", "fold", "zipper", "ripple", "surge",
            "collapse", "concat_flatten", "zeus", "helios",
            "cascade", "interpolate", "surge_fold", "slerp",
            "slip"
        ])


def get_folding_kernel(mode: str) -> FoldingKernel:
    return FOLDING_KERNELS.get(mode.lower(), RigidFolding())

