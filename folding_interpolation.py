# folding_interpolation.py
import torch
import math
from .schedules import ConditioningScheduler  # Ensure schedules.py is in same directory or adjust import


class FoldingInterpolator:
    """
    Symbolic interpolator with support for scheduler-driven alpha weighting
    and optional folding strategies that manipulate the blending pattern.
    """

    def __init__(self, scheduler_mode: str = "none", folding_mode: str = "interpolate"):
        self.scheduler = ConditioningScheduler(scheduler_mode)
        self.folding_mode = folding_mode.lower()

    def interpolate(self, masked: torch.Tensor, full: torch.Tensor, t: float) -> torch.Tensor:
        """
        Interpolates between masked and full embeddings using alpha(t) from scheduler,
        then optionally applies a folding effect to modify the output.
        """
        alpha = self.scheduler.alpha(t)

        # Base linear interpolation
        interpolated = (1 - alpha) * masked + alpha * full

        # Folding logic
        if self.folding_mode == "zipper":
            # Interleave masked/full dimensions
            interp = interpolated.clone()
            interp[..., ::2] = masked[..., ::2]
            interp[..., 1::2] = full[..., 1::2]
            return interp

        elif self.folding_mode == "ripple":
            ripple = torch.sin(2 * torch.pi * torch.linspace(0, 1, interpolated.size(-1), device=interpolated.device))
            return interpolated + 0.1 * ripple * (full - masked)

        elif self.folding_mode == "surge":
            delta = full - masked
            gain = torch.where(alpha > 0.7, alpha ** 2, alpha)
            return masked + gain * delta

        elif self.folding_mode == "collapse":
            return masked + (alpha ** 0.5) * (full - masked)

        elif self.folding_mode == "concat-flatten":
            concat = torch.cat([masked, full], dim=-1)
            projection = torch.eye(concat.size(-1), masked.size(-1), device=concat.device)
            return torch.nn.functional.linear(concat, projection)

        elif self.folding_mode == "zeus":
            threshold = 0.75
            return torch.where(alpha > threshold, full, masked)

        elif self.folding_mode == "helios":
            # Gradual boost to weak components across time
            weight = torch.linspace(0, 1, full.size(-1), device=full.device)
            weak_mask = (full.abs() < 0.25).float()
            boost = (alpha ** 2) * weak_mask * weight
            return interpolated + boost * (full - masked)

        elif self.folding_mode == "cascade":
            steps = interpolated.size(-1) // 4
            mask = torch.zeros_like(interpolated)
            idx = int(min(3, int(alpha * 4)))
            mask[..., idx * steps:(idx + 1) * steps] = 1.0
            return mask * full + (1 - mask) * masked

        elif self.folding_mode == "interpolate":
            return interpolated

        else:
            return interpolated
