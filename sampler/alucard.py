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
from comfy.utils import ProgressBar
from .formulas.schedules import FormulaScheduler   # Ensure schedules.py is in same directory or adjust import
from .formulas.folding import FoldingKernel, get_folding_kernel  # Ensure folding.py is in same directory or adjust import
from .formulas.padding import FoldingModifier  # Ensure padding.py is in same directory or adjust import
from .formulas.modes import FoldingPaddingTypes, FoldingPoolingTypes
from .formulas.pooling import WindowPooling
from .formulas.folding import FoldingKernels
from .formulas.schedules import SchedulerModes
from .alucard_exceptions import validate_shapes  # Ensure alucard_error.py is in same directory or adjust import

logger = logging.getLogger(__name__)

from ..utils.conditioning_shifter import ConditioningShifter
from ..utils.rose_util import rose_score_magnitude, legacy_entropy as compute_entropy

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

    def __copy__(self):
        # Create a shallow copy of the dataclass
        return FieldWalkerConfig(
            name=self.name,
            folding_mode=self.folding_mode,
            scheduler_mode=self.scheduler_mode,
            t_steps=self.t_steps,
            padding_mode=self.padding_mode,
            pooling_mode=self.pooling_mode,
            scheduler_config=self.scheduler_config.copy() if self.scheduler_config else None,
            context_overrides=self.context_overrides.copy() if self.context_overrides else None,
            window_managed_externally=self.window_managed_externally
        )


class Alucard(nn.Module):


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
            context: Optional[dict] = None,  # extra runtime info
            config: Optional[FieldWalkerConfig] = None,  # configuration for the sampler
            pbar: Optional[ProgressBar] = None,  # progress bar for tracking
    ) -> torch.Tensor:
        """
        Performs a folding schedule from embedding A to B using the scheduler & kernel logic.
        Delta field `d` allows guided interpolation from base → target.
        Returns either stacked, pooled, or concatenated embeddings.
        """
        global we_running_it
        with torch.autocast(device_type=a.device.type, enabled=a.device.type != 'cpu'):
            validate_shapes(a, b)
            #they're ready, lets clone them.
            a = a.clone()
            b = b.clone()
            d = d.clone() if d is not None else (b - a).clone()
            # -- Inject resonance potential --

            context = {} if context is None else context
            #logger.info(f"[EncoderSampler] Sampling with context: {config}")
            if config.context_overrides.get("enable_rope_spiral", False):
                #logger.info("[EncoderSampler] Computing resonance potential...")

                potential = ConditioningShifter.compute_resonance_potential(
                    embedding=a,
                    attention_mask=torch.ones(a.shape[:2], dtype=torch.bool, device=a.device),
                    offsets=config.context_overrides.get("rope_phase_offsets", None),
                    mode=config.context_overrides.get("rope_potential_mode", "harmonic_std")
                )
                context["resonance_potential"] = potential  # [B, T, 1] — used inside IntegraOrchestrator
                #logger.info(f"[EncoderSampler] Resonance potential computed with shape: {potential}")
                if config.context_overrides.get("rope_potential_mode", "harmonic_std"):
                    d = d * potential.unsqueeze(-1)  # Apply potential to delta
                elif config.context_overrides.get("rope_potential_mode", "spiral_gate"):
                    # Spiral gate mode, apply potential as a gating mechanism
                    d = d * potential.unsqueeze(-1)
                else:
                    d = d # No potential applied, just use delta as is

            B, T, D = a.shape
            folds = []
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
                    #logger.info(f"[Alucard] Applying padding with mask shape: {pad_mask.shape}")
                    temp_pad_mask = pad_mask.clone().to(a.device)

                    if context.get("use_entropy_scaling", False):
                        entropy = compute_entropy(temp_pad_mask)  # [B, T]
                        entropy_scalar = entropy.mean(dim=-1, keepdim=True)  # [B, 1]

                        entropy_weight = torch.sigmoid((entropy_scalar - context.get("entropy_scale_center", 0.5)) * context.get("entropy_scale_magnitude", 5.0))  # sharpen center around 0.5
                        temp_pad_mask = temp_pad_mask * entropy_weight.unsqueeze(1)  # [B, T, 1] scaled

                    folded = padding.apply_padding(a, folded, temp_pad_mask)


                folds.append(folded)
                if pbar is not None:
                    pbar.update(1)

            # -- Step 4: Aggregate via Pooling
            #result = pooling.apply(self, folds)

            # Removes the pooling behavior, as he is incapable of seeing the big picture.
            #torch.stack(folds)
            # replaces the pooling behavior with a simple stack for Integra to process.
            return torch.stack(folds)



class FieldWalker:
    def __init__(self, config: FieldWalkerConfig):
        self.config: FieldWalkerConfig = config
        self.name = config.name or "Alucard"
        self.scheduler = FormulaScheduler(config.scheduler_mode, config.scheduler_config or {})
        self.kernel = get_folding_kernel(config.folding_mode)
        self.padding = FoldingModifier({"padding_mode":config.padding_mode,})
        self.pooling = WindowPooling({"pooling_mode": config.pooling_mode})
        self.core = Alucard()

    from ..utils.rose_util import rose_score_magnitude, entropy

    def walk(
            self,
            a: torch.Tensor,
            b: torch.Tensor,
            pad_mask: Optional[torch.Tensor] = None,
            d: Optional[torch.Tensor] = None,
            pbar: Optional = None,
            context: dict = None,
    ) -> torch.Tensor:
        """
        Walks symbolic delta from `a` to `b` using Rose magnitude-based masking.
        Will auto-inject pad_mask using rose_score_magnitude(a, b, d, purpose).
        """
        d = d if d is not None else (b - a)
        context = context or {}

        if pad_mask is None and context.get("use_alpha_mask", False) and context.get("use_rose_similarity", False):
            try:
                relation = d
                purpose = context.get("rose_purpose", torch.ones_like(a))  # fallback = identity purpose

                pad_mask = rose_score_magnitude(
                    x=a,
                    need=b,
                    relation=relation,
                    purpose=purpose,
                ).unsqueeze(-1)  # shape: [B, T, 1]
                #pad_mask = pad_mask.clamp(0.0, 1.0)

            except Exception as e:
                import logging
                logging.warning(f"[FieldWalker] Rose magnitude mask failed: {e}")
                pad_mask = None

        return self.core.sample(
            a=a,
            b=b,
            d=d,
            t_steps=self.config.t_steps,
            scheduler=self.scheduler,
            kernel=self.kernel,
            pooling=self.pooling,
            padding=self.padding,
            pad_mask=pad_mask,
            context=context,
            pbar=pbar,
            config=self.config,
        )

