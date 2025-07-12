from dataclasses import dataclass
from typing import List, Optional
import torch
import torch.nn.functional as F
from torch import nn, sort
import logging
logger = logging.getLogger(__name__)


from .modes import FoldingPaddingTypes, FoldingPoolingTypes
# --------------------------------------------------------------------- #

class FoldingModifier:
    def __init__(self, config: Optional[dict] = None):
        if config is None:
            config = {}
        self.padding_mode = config.get("padding_mode", FoldingPaddingTypes.NONE)

    # --- PADDING ---
    # --- PADDING ---
    def apply_padding(
            self,
            base: torch.Tensor,        # [B, T, D]
            folded: torch.Tensor,      # [B, T, D]
            mask: torch.Tensor,        # [B, T] | [B, T, 1] | [B, T, D]
            config: Optional[dict] = None
    ) -> torch.Tensor:
        """
        `mask` meaning depends on its rank:

        • [B, T]      → bool/float  padding flag per token
        • [B, T, 1]   → broadcast weight for all features
        • [B, T, D]   → full gradient / gate per‑feature
        Values should be in **[0,1]** where 0 shuts folded out,
        1 keeps folded fully, and anything in‑between blends.
        """
        if not isinstance(base, torch.Tensor) or not isinstance(folded, torch.Tensor):
            raise TypeError(
                f"[Alucard] Expected base and folded to be torch.Tensor, "
                f"got {type(base)} and {type(folded)}"
            )
        if config is None:
            config = {}
        self.padding_mode = config.get("padding_mode", self.padding_mode)
        if self.padding_mode is None:
            self.padding_mode = FoldingPaddingTypes.NONE

        # ---------- 1 · shape harmonisation ----------
        if mask.dim() == 2:
            mask = mask.unsqueeze(-1).float()           # → [B,T,1]
        elif mask.dim() == 3 and mask.shape[-1] == 1:
            mask = mask.float()                         # already broadcast
        elif mask.dim() == 3 and mask.shape[-1] == base.shape[-1]:
            mask = mask.float()                         # gradient mask
        else:
            raise ValueError(
                f"[Alucard] Invalid mask shape {mask.shape}; "
                f"expected [B,T], [B,T,1] or [B,T,D={base.shape[-1]}]"
            )

        if mask.shape[0:2] != base.shape[0:2]:
            raise ValueError(
                f"[Alucard] Mask token dimensions {mask.shape[:2]} ≠ base {base.shape[:2]}"
            )

        # ---------- 2 · mode‑specific behaviour ----------
        mode = self.padding_mode

        # NONE ­­­→ leave folded unchanged
        if mode == FoldingPaddingTypes.NONE:
            return folded

        # INTERPOLATE ­­­→ blend base & folded by mask
        if mode == FoldingPaddingTypes.INTERPOLATE:
            return mask * folded + (1.0 - mask) * base

        # REPLACE ­­­→ hard switch when mask>0.5
        if mode == FoldingPaddingTypes.REPLACE:
            return torch.where(mask > 0.5, folded, base)

        # GAPPED ­­­→ zero‑out where mask>0.5
        if mode == FoldingPaddingTypes.GAPPED:
            return torch.where(mask > 0.5, torch.zeros_like(base), base)

        # SPARSE ­­­→ keep only masked portions of folded
        if mode == FoldingPaddingTypes.SPARSE:
            out = torch.zeros_like(base)
            out[mask > 0.5] = folded[mask > 0.5]
            return out

        # BLEND ­­­→ blend base & folded by mask, but keep base where mask<0.5
        if mode == FoldingPaddingTypes.BLEND:
            # blend folded into base, but only where mask < 0.5
            blended = base * (1.0 - mask) + folded * mask
            return blended

        # BLEND2 ­­­→ blend folded into base, but only where mask < 0.5
        if mode == FoldingPaddingTypes.BLEND2:
            # blend folded into base, but only where mask < 0.5
            blended = base * (1.0 - mask) + folded * mask
            # ensure that the base is not zeroed out
            blended[mask < 0.5] = base[mask < 0.5]
            return blended

        if mode == FoldingPaddingTypes.MASK_EDGES:
            # mask edges of folded based on the mask
            #adjust mask so only the top 25% and bottom 25% are masked
            # should we preserve the top_k?
            if mask.shape[1] < 8:
                logger.warning("[Alucard] Mask edges mode requires at least 8 tokens, skipping masking.")
                return folded
            mask = mask.clone().detach()
            mask[:, :mask.shape[1] // 8] = -100.0
            mask[:, -mask.shape[1] // 8:] = -100.0
            folded_masked = folded * mask
            # ensure that the base is not zeroed out
            folded_masked[mask < 0.5] = base[mask < 0.5]
            return folded_masked

        if mode == FoldingPaddingTypes.MASK_TOP_K:
            # mask top K tokens of folded based on the mask
            k = int(mask.shape[1] * 0.25)
            # we want to preserve the top K tokens of folded without changing the base
            top_k_mask = torch.zeros_like(mask, dtype=torch.bool)
            top_k_mask[:, :k] = True
            folded_masked = folded * top_k_mask.float()
            # ensure that the base is not zeroed out
            folded_masked[top_k_mask < 0.5] = base[top_k_mask < 0.5]
            return folded_masked


        # Fallback: return folded unchanged
        return folded


