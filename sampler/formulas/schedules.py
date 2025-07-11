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
    def __init__(self, tau: Optional[float] = None, sigma_fn: Optional[Callable] = None):
        """
        Initializes the TauInterpolation with an optional sigma function.
        If no sigma function is provided, it defaults to a sine function.
        """
        self.sigma_fn = sigma_fn
        self.tau = tau
    def __call__(self, t, a, b, config=None):
        tau = (config or {}).get("tau", 1.0) or self.tau or 1.0
        sigma_fn = (config or {}).get("sigma_fn", None) or self.sigma_fn
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
    def __init__(self, wave_freq: Optional[float]):
        self.wave_freq = wave_freq
    def __call__(self, t, a=None, b=None, config=None):
        freq = (config or {}).get("wave_freq", 0.27195) or self.wave_freq
        return torch.sin(freq * math.pi * t)


class PulseFunction(FormulaFunction):
    def __init__(self, pulse_freq: Optional[float]):
        self.pulse_freq = pulse_freq

    def __call__(self, t, a=None, b=None, config=None):
        freq = (config or {}).get("pulse_freq", 5.0) or self.pulse_freq
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
            "top_k": ThresholdGate(threshold=self.config.get("top_k", 0.8)),
            "top_20k": ThresholdGate(threshold=0.2),
            "top_50k": ThresholdGate(threshold=0.5),
            "cosine": CosineEnvelope(),
            "cos": WaveFunction(wave_freq=0.5),  # alias for cosine
            "sine": WaveFunction(wave_freq=1.0),  # alias for sine
            "wave": WaveFunction(wave_freq=0.27195),
            "pulse": PulseFunction(pulse_freq=5.0),
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


class SchedulerModes:
    """
    List of available scheduling modes for the FormulaScheduler.
    These modes define how the interpolation is computed.
    """
    TAU = "tau"
    TOP_K = "top_k"
    TOP_20K = "top_20k"
    TOP_50K = "top_50k"
    COSINE = "cosine"
    COS = "cos"  # alias
    SINE = "sine"  # alias
    WAVE = "wave"
    PULSE = "pulse"
    SHOCKWAVE = "shockwave"
    CASCADE = "cascade"
    PHASE_SLIP = "phase_slip"
    NONE = "none"  # Default mode

    @staticmethod
    def to_list() -> list[str]:
        """
        Returns a list of all available scheduler modes.
        """
        return [
            SchedulerModes.TAU,
            SchedulerModes.TOP_K,
            SchedulerModes.TOP_20K,
            SchedulerModes.TOP_50K,
            SchedulerModes.COSINE,
            SchedulerModes.COS,
            SchedulerModes.SINE,
            SchedulerModes.WAVE,
            SchedulerModes.PULSE,
            SchedulerModes.SHOCKWAVE,
            SchedulerModes.CASCADE,
            SchedulerModes.PHASE_SLIP,
            SchedulerModes.NONE
        ]