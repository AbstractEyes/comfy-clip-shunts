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
        freq = context.get("freq", 2.0) if context else 2.0
        ripple = torch.sin(freq * torch.pi * t).unsqueeze(-1)
        return a + ripple * (b - a)


class SurgeFolding(FoldingKernel):
    def apply(self, a, b, t, alpha=None, context=None):
        intensity = context.get("surge_intensity", 5.0) if context else 5.0
        surge = 1 - torch.exp(-intensity * t)
        return a + (b - a) * surge.unsqueeze(-1)


class CollapseFolding(FoldingKernel):
    def apply(self, a, b, t, alpha=None, context=None):
        rate = context.get("rate", 1.0) if context else 1.0
        collapse = 1 - torch.exp(-rate * (1 - t))
        return b * collapse.unsqueeze(-1)


class ConcatFlattenFolding(FoldingKernel):
    def apply(self, a, b, t, alpha=None, context=None):
        concat = torch.cat([a, b], dim=-1)
        proj = torch.nn.Linear(concat.size(-1), a.size(-1)).to(a.device)
        return proj(concat)


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
    "rigid": RigidFolding(),
    "fold": FoldFolding(),
    "zipper": ZipperFolding(),
    "ripple": RippleFolding(),
    "surge": SurgeFolding(),
    "collapse": CollapseFolding(),
    "concat-flatten": ConcatFlattenFolding(),
    "zeus": ZeusFolding(),
    "helios": HeliosFolding(),
    "cascade": CascadeFolding(),
    "interpolate": InterpolateFolding(),
    "surge-fold": SurgeFoldFolding(),
    "slerp": SlerpFolding(),
    "slip": SlipFolding(),  # entropy‑weighted
}


def get_folding_kernel(mode: str) -> FoldingKernel:
    return FOLDING_KERNELS.get(mode.lower(), RigidFolding())
