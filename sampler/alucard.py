"""
    Alucard the Field Walker
    Author: AbstractPhil
    Date: 2025-7-10

    This module implements the Alucard field walker, which performs guided interpolation and folding through
    a complex sequence of operations. It uses a scheduler to compute interpolation parameters, a kernel to apply
    folding logic, and a modifier to handle padding and pooling of the resulting embeddings.

    Alucard is a hivemind interpolation formula that allows for flexible and guided transformations in tensor fields
    using many different folding and padding strategies. It is designed to work with symbolic fields.

    Many of these operations simply do not work. Many are not properly implemented yet.
    Many are likely going to be removed entirely in the future or replaced with something more efficient and useful.

    Alucard however, will stay. This is a foundational piece of the ABS framework, allowing for guided interpolation
    Integra regulates him, and Alucard goes on his walks - never understanding the big picture, only caring
    about the immediate task at hand.

"""
# alucard.py
from dataclasses import dataclass
from typing import Optional
import logging
import torch
from torch import nn

from comfy import model_management
from .formulas.schedules import FormulaScheduler   # Ensure schedules.py is in same directory or adjust import
from .formulas.folding import FoldingKernel, get_folding_kernel  # Ensure folding.py is in same directory or adjust import
from .formulas.padding import FoldingModifier  # Ensure padding.py is in same directory or adjust import
from .formulas.modes import FoldingPaddingTypes, FoldingPoolingTypes
from .formulas.pooling import WindowPooling
from .formulas.folding import FoldingKernels
from .formulas.schedules import SchedulerModes
from .alucard_exceptions import validate_shapes  # Ensure alucard_error.py is in same directory or adjust import

logger = logging.getLogger(__name__)

@dataclass
class FieldWalkerConfig:
    name: str = ""
    folding_mode: str = FoldingKernels.gilgamesh
    scheduler_mode: str = SchedulerModes.TAU
    t_steps: int = 6
    padding_mode: str = FoldingPaddingTypes.INTERPOLATE
    pooling_mode: str = FoldingPoolingTypes.AVERAGE
    scheduler_config: Optional[dict] = None
    context_overrides: Optional[dict] = None
    window_managed_externally: bool = True


class SamplerCore(nn.Module):

    def sample(
            self,
            a: torch.Tensor,  # base field: [B, T, D]
            b: torch.Tensor,  # target field: [B, T, D]
            d: torch.Tensor,  # delta field: [B, T, D] (precomputed or guided shift)
            t_steps: int,  # total interpolation steps
            scheduler: FormulaScheduler,  # provides alpha, tau, etc.
            kernel: FoldingKernel,  # folding mode executor
            padding: FoldingModifier,  # padding/pooling control
            pooling: WindowPooling,  # pooling strategy, may implement again later
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
            #they're ready, lets clone them.
            a = a.clone()
            b = b.clone()
            d = d.clone() if d is not None else (b - a).clone()
            B, T, D = a.shape
            folds = []
            context = context or {}
            context["delta"] = d  # Inject delta into shared execution context

            for step in range(t_steps):
                model_management.throw_exception_if_processing_interrupted()
                t_scalar = step / (t_steps - 1)
                t = torch.full((B, T), t_scalar, device=a.device)

                # -- Step 1: Compute Alpha (scheduler can now use delta)
                alpha = scheduler.compute_alpha(t, a, b, context)

                # -- Step 2: Fold using Kernel (can now use delta from context)
                folded = kernel.apply(a=a, b=b, alpha=alpha, t=t, context=context)

                # -- Step 3: Apply Padding Policy
                if pad_mask is not None:
                    folded = padding.apply_padding(a, folded, pad_mask)

                folds.append(folded)

            # -- Step 4: Aggregate via Pooling
            #result = pooling.apply(self, folds)

            # Removes the pooling behavior, as he is incapable of seeing the big picture.
            #torch.stack(folds)
            # replaces the pooling behavior with a simple stack for Integra to process.
            return torch.stack(folds)


class FieldWalker:
    def __init__(self, config: FieldWalkerConfig):
        self.config = config
        self.name = config.name or "Alucard"
        self.scheduler = FormulaScheduler(config.scheduler_mode, config.scheduler_config or {})
        self.kernel = get_folding_kernel(config.folding_mode)
        self.padding = FoldingModifier({"padding_mode":config.padding_mode,})
        self.pooling = WindowPooling({"pooling_mode": config.pooling_mode})
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
            pooling=self.pooling,
            padding=self.padding,
            pad_mask=pad_mask,
            context=context
        )
