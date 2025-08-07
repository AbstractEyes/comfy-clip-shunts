import math

import torch
from typing import Tuple, List, Dict, Optional
from dataclasses import dataclass

from comfy import model_management
from .alucard import FieldWalker, FieldWalkerConfig
from .sliding_window import ShuntStackConfig
import logging
import torch.nn.functional as F
from .formulas.schedules import FormulaScheduler   # Ensure schedules.py is in same directory or adjust import
from .formulas.folding import FoldingKernel, get_folding_kernel  # Ensure folding.py is in same directory or adjust import
from .formulas.padding import FoldingModifier  # Ensure padding.py is in same directory or adjust import
from .formulas.modes import FoldingPaddingTypes, FoldingPoolingTypes
from .formulas.pooling import WindowPooling
from .formulas.folding import FoldingKernels
from .formulas.schedules import SchedulerModes
from .alucard_exceptions import validate_shapes  # Ensure alucard_error.py is in same directory or adjust import

from ..utils.rose_util import rose_score

from comfy.utils import ProgressBar

logger = logging.getLogger(__name__)


@dataclass
class IntegraConfig:
    walker_config: FieldWalkerConfig
    stack_config: ShuntStackConfig
    trace_folds: bool = False  # Optional: store each windowed fold for debug
    enforce_projection: bool = True  # Optional: auto-project symbolic fields if needed
    enable_clip_alignment: bool = True  # Optional: perform scheduler-aware comparison to CLIP
    use_rose_similarity: bool = False  # Optional: use Rose similarity for window selection

    def __copy__(self):
        return IntegraConfig(
            walker_config=self.walker_config.__copy__(),
            stack_config=self.stack_config.__copy__(),
            trace_folds=self.trace_folds,
            enforce_projection=self.enforce_projection,
            enable_clip_alignment=self.enable_clip_alignment,
            use_rose_similarity=self.use_rose_similarity
        )



class IntegraOrchestrator:
    def __init__(self, config: IntegraConfig):
        self.config = config
        self.walker = FieldWalker(config.walker_config)
        #self.scheduler = FormulaScheduler(config.walker_config.scheduler_mode, config.walker_config.scheduler_config or {})
        self.kernel = get_folding_kernel(config.walker_config.folding_mode)
        self.padding = FoldingModifier({"padding_mode": config.walker_config.padding_mode,})
        self.pooling = WindowPooling({"pooling_mode": config.walker_config.pooling_mode})
        stack = config.stack_config
        self.window_size = stack.sliding_window_size
        self.stride = stack.sliding_window_stride
        self.max_length = stack.max_length
        self.override_context_window = stack.override_context_window
        self.context_window_size = stack.context_window_size

    def walk_encoder_field(
            self,
            a: torch.Tensor,
            b: torch.Tensor,
            d: torch.Tensor,
            context=None  # ✅ Accept context
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Walks a full symbolic encoder field via sliding windows, governed by Integra.
        Returns recombined tensor and orchestration report.
        """
        with torch.autocast(device_type=a.device.type, enabled=a.device.type != 'cpu'):
            a, b, d = upscale_trio(a, b, d)
            B, T_full, D = a.shape
            limit = min(T_full, self.max_length)
            T = min(limit, T_full)

            a = a[:, :T, :]
            b = b[:, :T, :]
            d = d[:, :T, :]

            folds = []
            starts = self._compute_window_starts(T, a.squeeze(0))
            steps = self.config.walker_config.t_steps
            passes = context.get("passes", 1) if context else 1
            pbar = ProgressBar(len(starts * steps * passes))

            for p in range(passes):
                for i, start in enumerate(starts):
                    model_management.throw_exception_if_processing_interrupted()
                    end = start + self.window_size
                    if end > T:
                        end = T
                        start = max(0, end - self.window_size)

                    a_win = a[:, start:end, :].clone()
                    b_win = b[:, start:end, :].clone()
                    d_win = d[:, start:end, :].clone()

                    # ✅ Now passes context to walker
                    folded = self.walker.walk(a=a_win, b=b_win, d=d_win, pbar=pbar, context=context)

                    pbar.update(1)
                    folds.append((start, end, folded))

            aggregated = self.aggregate(folds, T, config=context)

            return aggregated, {
                "tokens_processed": T,
                "tokens_total": T_full,
                "folds": len(folds),
                "stride": self.stride,
                "window_size": self.window_size,
                "override_context_window": self.override_context_window
            }

    def aggregate(self, folds, _, config=None) -> torch.Tensor:
        collapsed = [chunk.mean(0) for _, _, chunk in folds]
        return self.pooling.apply(self, collapsed, config=config)

    # --- helper: build pad-mask for one window ----------------------------
    def _build_pad_mask(self,
                        ids_len: int,
                        cls_pos: int = 0,
                        eof_pos: Optional[int] = None) -> torch.Tensor:
        """
        Returns a [1, ids_len] bool Tensor where False ⇒ token should be
        *ignored* by the fold (padding), True ⇒ keep it.
        • CLS (and optional EOF) are kept.
        • If override_context is on, we also drop first/last token.
        """
        mask = torch.ones(1, ids_len, dtype=torch.bool)
        # keep CLS / EOF
        mask[:, cls_pos] = True
        if eof_pos is not None and eof_pos < ids_len:
            mask[:, eof_pos] = True
        # when override_context, silent-drop first/last token
        if self.override_context_window:
            mask[:, 0] = False
            mask[:, -1] = False
        return mask

    # ------------------------------------------------------------------
    # choose at most `max_windows` windows whose start positions are
    # either (A) evenly spaced, or (B) chosen at local-min similarity
    # ------------------------------------------------------------------
    def _compute_window_starts(self,
                               T: int,
                               embeddings: Optional[torch.Tensor] = None) -> List[int]:
        """
        Return list of window-start indices (len ≤ max_windows).
        """
        if self.window_size >= T:
            return [0]  # one window == whole prompt

        W = self.config.stack_config.max_windows
        if W is None or W <= 1:  # ➊ guard: only one window wanted
            return [0]  # just start at 0

        # --- evenly spaced baseline ---------------------------------
        stride = math.ceil((T - self.window_size) / (W - 1))
        stride = max(stride, 1)
        starts = list(range(0, T - self.window_size + 1, stride))
        starts = starts[:W]  # clamp in case we overshot

        # --- similarity refinement (optional) -----------------------
        if embeddings is not None:
            with torch.no_grad():
                if self.config.use_rose_similarity:
                    if embeddings.ndim == 3:
                        need = embeddings.mean(dim=1)
                        relation = embeddings[:, 1:, :].mean(dim=1)
                        purpose = embeddings[:, :-1, :].mean(dim=1)
                        sims = rose_score(embeddings[:, :-1, :], need, relation, purpose)
                    else:
                        # Fallback for [T, D] shape (single batch already squeezed)
                        need = embeddings.mean(dim=0, keepdim=True)
                        relation = embeddings[1:, :].mean(dim=0, keepdim=True)
                        purpose = embeddings[:-1, :].mean(dim=0, keepdim=True)
                        sims = rose_score(embeddings[:-1, :], need, relation, purpose)
                else:
                    sims = F.cosine_similarity(embeddings[:-1], embeddings[1:], dim=-1)
                for i in range(1, len(starts) - 1):
                    seg = slice(max(0, starts[i] - 4), min(T - 1, starts[i] + 4))
                    local_min = sims[seg].argmin().item() + seg.start
                    starts[i] = max(0, min(local_min, T - self.window_size))
            starts = sorted(set(starts))

        return starts

DTYPE_PECKING_ORDER = {
    torch.float64: 1,  # we prioritize lowest to upscale to
    torch.float32: 2,
    torch.float16: 3,
    torch.bfloat16: 4,
}

def upscale_trio(
        base: torch.Tensor,  # [B, T, D]
        folded: torch.Tensor,  # [B, T, D]
        mask: torch.Tensor  # [B, T] | [B, T, 1] | [B, T, D]
) -> (torch.Tensor, torch.Tensor, Optional[torch.Tensor], list):
    """
    Upscales base and folded tensors to the highest precision between them.
    Returns upscaled base and folded tensors.
    """
    if not isinstance(base, torch.Tensor) or not isinstance(folded, torch.Tensor):
        raise TypeError(
            f"[Alucard] Expected base and folded to be torch.Tensor, "
            f"got {type(base)} and {type(folded)}"
        )

    # Determine the highest precision dtype
    dtypes = [base.dtype, folded.dtype, mask.dtype] if isinstance(mask, torch.Tensor) else [base.dtype,
                                                                                            folded.dtype]
    target_dtype = min(dtypes, key=lambda x: DTYPE_PECKING_ORDER[x])

    # Upscale both tensors to the highest dtype
    base = base.clone().to(target_dtype) if base.dtype != target_dtype else base
    folded = folded.clone().to(target_dtype) if folded.dtype != target_dtype else folded
    if isinstance(mask, torch.Tensor):
        mask = mask.clone().to(target_dtype) if mask.dtype != target_dtype else mask

    return base, folded, mask  # just upscale it all in uniform

#def aggregate(self, folds, total_tokens):
#    B, _, D = folds[0][2].shape
#    device = folds[0][2].device
#    acc = torch.zeros(B, total_tokens, D, device=device)
#    wsum = torch.zeros(B, total_tokens, 1, device=device)
#
#    for start, end, chunk in folds:
#        length = end - start
#        tri = torch.linspace(0, 1, length, device=device).unsqueeze(0).unsqueeze(-1)
#        tri = torch.minimum(tri, 1 - tri) * 2
#        acc[:, start:end, :] += chunk * tri
#        wsum[:, start:end, :] += tri
#
#    return acc / wsum.clamp(min=1e-6)