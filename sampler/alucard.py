# folding_interpolation.py
from dataclasses import dataclass
from typing import Optional
import logging
import torch
from torch import nn

from .formulas.schedules import FormulaScheduler   # Ensure schedules.py is in same directory or adjust import
from .formulas.folding import FoldingKernel, get_folding_kernel  # Ensure folding.py is in same directory or adjust import
from .formulas.padding import FoldingModifier, FoldingModifierConfig  # Ensure padding.py is in same directory or adjust import
from .formulas.modes import FoldingTypes, FoldingPaddingTypes, ConditioningSchedulerTypes, FoldingPoolingTypes
from .alucard_exceptions import validate_shapes  # Ensure alucard_error.py is in same directory or adjust import

logger = logging.getLogger(__name__)

@dataclass
class FieldWalkerConfig:
    folding_mode: str = FoldingTypes.FOLD
    scheduler_mode: str = ConditioningSchedulerTypes.TAU
    t_steps: int = 6
    padding_mode: str = FoldingPaddingTypes.INTERPOLATE
    pooling_mode: str = FoldingPoolingTypes.AVERAGE
    scheduler_config: Optional[dict] = None
    context_overrides: Optional[dict] = None


class SamplerCore(nn.Module):

    def sample(
            self,
            a: torch.Tensor,  # base field: [B, T, D]
            b: torch.Tensor,  # target field: [B, T, D]
            d: torch.Tensor,  # delta field: [B, T, D] (precomputed or guided shift)
            t_steps: int,  # total interpolation steps
            scheduler: FormulaScheduler,  # provides alpha, tau, etc.
            kernel: FoldingKernel,  # folding mode executor
            modifier: FoldingModifier,  # padding/pooling control
            pad_mask: Optional[torch.Tensor] = None,  # [B, T] bool
            context: Optional[dict] = None  # extra runtime info
    ) -> torch.Tensor:
        """
        Performs a folding schedule from embedding A to B using the scheduler & kernel logic.
        Delta field `d` allows guided interpolation from base → target.
        Returns either stacked, pooled, or concatenated embeddings.
        """
        with torch.autocast(device_type=a.device.type, enabled=a.device.type != 'cpu'):
            validate_shapes(a, b)
            B, T, D = a.shape
            folds = []
            context = context or {}
            context["delta"] = d  # Inject delta into shared execution context

            for step in range(t_steps):
                t_scalar = step / (t_steps - 1)
                t = torch.full((B, T), t_scalar, device=a.device)

                # -- Step 1: Compute Alpha (scheduler can now use delta)
                alpha = scheduler.compute_alpha(t, a, b, context)

                # -- Step 2: Fold using Kernel (can now use delta from context)
                folded = kernel.apply(a=a, b=b, alpha=alpha, t=t, context=context)

                # -- Step 3: Apply Padding Policy
                if pad_mask is not None:
                    folded = modifier.apply_padding(a, folded, pad_mask)

                folds.append(folded)

            # -- Step 4: Aggregate via Pooling
            result = modifier.apply_pooling(folds)
            return result


class FieldWalker:
    def __init__(self, config: FieldWalkerConfig):
        self.config = config
        self.scheduler = FormulaScheduler(config.scheduler_mode, config.scheduler_config or {})
        self.kernel = get_folding_kernel(config.folding_mode)
        self.modifier = FoldingModifier(
            FoldingModifierConfig(
                padding_mode=config.padding_mode,
                pooling_mode=config.pooling_mode
            )
        )
        self.core = SamplerCore()

    def walk(self, a: torch.Tensor, b: torch.Tensor, pad_mask: Optional[torch.Tensor] = None,
             d: Optional[torch.Tensor] = None) -> torch.Tensor:
        logger.info(
            f"[Alucard] Walking: a {a.shape}, b {b.shape}, d {d.shape if d is not None else 'computed'}, t_steps={self.config.t_steps}")

        d = d if d is not None else (b - a)
        context = self.config.context_overrides or {}
        return self.core.sample(
            a=a, b=b, d=d,
            t_steps=self.config.t_steps,
            scheduler=self.scheduler,
            kernel=self.kernel,
            modifier=self.modifier,
            pad_mask=pad_mask,
            context=context
        )
