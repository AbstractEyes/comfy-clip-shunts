import torch
import math
import torch.nn.functional as F
from typing import Optional, Callable, Union


class FormulaFunction:
    """
    Base class for formula modules. Each subclass implements `__call__(t, a, b, config)`.
    """
    def __call__(self,
                 t: torch.Tensor,
                 a: Optional[torch.Tensor] = None,
                 b: Optional[torch.Tensor] = None,
                 config: Optional[dict] = None) -> torch.Tensor:
        raise NotImplementedError("FormulaFunction subclasses must implement __call__.")

    def __repr__(self):
        return self.__class__.__name__


class TauInterpolation(FormulaFunction):
    def __call__(self, t, a, b, config=None):
        tau = (config or {}).get("tau", 1.0)
        sigma_fn = (config or {}).get("sigma_fn", None)
        sigma = sigma_fn(t) if sigma_fn else torch.sin(math.pi * t)
        tau_scale = 1 - torch.exp(-tau * t)
        delta = b - a
        return a + delta * sigma.unsqueeze(-1) * tau_scale.unsqueeze(-1)


class ThresholdGate(FormulaFunction):
    def __init__(self, threshold):
        self.threshold = threshold

    def __call__(self, t, a=None, b=None, config=None):
        return torch.where(t > self.threshold,
                           torch.tensor(1.0, device=t.device),
                           torch.tensor(0.0, device=t.device))


class CosineEnvelope(FormulaFunction):
    def __call__(self, t, a=None, b=None, config=None):
        return 0.5 * (1 - torch.cos(math.pi * t))


class WaveFunction(FormulaFunction):
    def __call__(self, t, a=None, b=None, config=None):
        freq = (config or {}).get("wave_freq", 2.0)
        return torch.sin(freq * math.pi * t)


class PulseFunction(FormulaFunction):
    def __call__(self, t, a=None, b=None, config=None):
        freq = (config or {}).get("pulse_freq", 10.0)
        return torch.sin(freq * math.pi * t) * (1 - t)


class ShockwaveFunction(FormulaFunction):
    def __call__(self, t, a=None, b=None, config=None):
        center = (config or {}).get("center", 0.5)
        variance = (config or {}).get("variance", 0.01)
        return torch.exp(-((t - center) ** 2) / variance)


class CascadeFunction(FormulaFunction):
    def __call__(self, t, a=None, b=None, config=None):
        rate = (config or {}).get("cascade_rate", 4.0)
        return torch.clamp(rate * t - 1, 0.0, 1.0)


class ConstantFunction(FormulaFunction):
    def __init__(self, value: Union[float, torch.Tensor] = 1.0):
        self.value = torch.tensor(value) if not isinstance(value, torch.Tensor) else value

    def __call__(self, t, a=None, b=None, config=None):
        return self.value.to(device=t.device)


class PhaseSlip(FormulaFunction):
    def __call__(self, t, a=None, b=None, config=None):
        # expects context to contain 'delta'
        ctx = config or {}
        delta = ctx.get("delta", b - a) if (a is not None and b is not None) else None
        if delta is None:
            return t       # graceful fallback
        entropy = delta.var(dim=-1)                  # [B,T]
        entropy_norm = (entropy - entropy.min()) / (entropy.max() - entropy.min() + 1e-8)
        base = 0.5 * (1 - torch.cos(math.pi * t))
        return torch.clamp(base + entropy_norm * (1 - base), 0.0, 1.0)




class FormulaScheduler:
    def __init__(self, mode: str, config: Optional[dict] = None):
        self.mode = mode
        self.config = config or {}
        self.registry = self._register_formulas()

    def _register_formulas(self) -> dict[str, Callable]:
        return {
            "tau": TauInterpolation(),
            "top_k": ThresholdGate(threshold=self.config.get("top_k", 0.5)),
            "top_20k": ThresholdGate(threshold=0.2),
            "top_50k": ThresholdGate(threshold=0.5),
            "cosine": CosineEnvelope(),
            "cos": WaveFunction(),
            "sine": WaveFunction(),  # alias
            "wave": WaveFunction(),
            "pulse": PulseFunction(),
            "shockwave": ShockwaveFunction(),
            "cascade": CascadeFunction(),
            "phase_slip": ShockwaveFunction(),
            "none": ConstantFunction(value=1.0),
        }

    def compute_alpha(self,
                      t: torch.Tensor,
                      a: Optional[torch.Tensor] = None,
                      b: Optional[torch.Tensor] = None,
                      context: Optional[dict] = None) -> torch.Tensor:
        fn = self.registry.get(self.mode, self.registry["none"])
        return fn(t, a, b, context or self.config)

    def available_modes(self) -> list:
        return list(self.registry.keys())


SCHEDULER_MODES = [
    "tau", "top_k", "top_20k", "top_50k",
    "cosine", "cos", "sine", "wave",
    "pulse", "shockwave", "cascade",
    "phase_slip", "none"
]