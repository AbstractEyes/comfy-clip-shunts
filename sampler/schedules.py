# schedules.py
import torch
import math
from .modes import ConditioningSchedulerTypes



class ConditioningScheduler:
    """
    Computes scheduler-based alpha values for symbolic interpolation.
    Given a mode and timestep t in [0, 1], returns alpha(t).
    """

    def __init__(self, mode: str = "none"):
        self.mode = mode.lower()

    def alpha(self, t: float) -> torch.Tensor:
        t = torch.clamp(torch.tensor(t), 0.0, 1.0)

        if self.mode == ConditioningSchedulerTypes.TAU:
            return 1 - torch.exp(-5 * t)

        elif self.mode == ConditioningSchedulerTypes.TOP_K:
            return torch.where(t > 0.5, torch.tensor(1.0), torch.tensor(0.0))

        elif self.mode == ConditioningSchedulerTypes.TOP_20K:
            return torch.where(t > 0.2, torch.tensor(1.0), torch.tensor(0.0))

        elif self.mode == ConditioningSchedulerTypes.TOP_50K:
            return torch.where(t > 0.5, torch.tensor(1.0), torch.tensor(0.0))

        elif self.mode == ConditioningSchedulerTypes.COS:
            return torch.cos(math.pi * t)

        elif self.mode == ConditioningSchedulerTypes.SINE:
            return torch.sin(math.pi * t)

        elif self.mode == ConditioningSchedulerTypes.COSINE:
            return 0.5 * (1 - torch.cos(math.pi * t))

        elif self.mode == ConditioningSchedulerTypes.WAVE:
            return torch.sin(2 * math.pi * t)

        elif self.mode == ConditioningSchedulerTypes.PULSE:
            return torch.sin(10 * math.pi * t) * (1 - t)

        elif self.mode == ConditioningSchedulerTypes.SHOCKWAVE:
            return torch.exp(-((t - 0.5) ** 2) / 0.01)

        elif self.mode == ConditioningSchedulerTypes.CASCADE:
            return torch.clamp(4 * t - 1, 0.0, 1.0)  # slow ramp-up

        else:  # "none" or fallback
            return torch.tensor(1.0)

    def __call__(self, t: float) -> torch.Tensor:
        return self.alpha(t)
