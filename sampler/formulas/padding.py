from dataclasses import dataclass
from typing import List
import torch
import torch.nn.functional as F

from .modes import FoldingPaddingTypes, FoldingPoolingTypes
# --------------------------------------------------------------------- #

@dataclass
class FoldingModifierConfig:
    padding_mode: str = FoldingPaddingTypes.NONE
    pooling_mode: str = FoldingPoolingTypes.NONE


class FoldingModifier:
    def __init__(self, config: FoldingModifierConfig):
        self.padding_mode = config.padding_mode
        self.pooling_mode = config.pooling_mode

    # --- PADDING ---
    # --- PADDING ---
    def apply_padding(
            self,
            base: torch.Tensor,        # [B, T, D]
            folded: torch.Tensor,      # [B, T, D]
            mask: torch.Tensor         # [B, T] | [B, T, 1] | [B, T, D]
    ) -> torch.Tensor:
        """
        `mask` meaning depends on its rank:

        • [B, T]      → bool/float  padding flag per token
        • [B, T, 1]   → broadcast weight for all features
        • [B, T, D]   → full gradient / gate per‑feature
        Values should be in **[0,1]** where 0 shuts folded out,
        1 keeps folded fully, and anything in‑between blends.
        """

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

        # Fallback: return folded unchanged
        return folded


    # --- POOLING ---
    def apply_pooling(self, embeddings: List[torch.Tensor]) -> torch.Tensor:
        """
        Args:
            embeddings: list of [B, T, D] or [B, D] tensors to pool across
        """
        if self.pooling_mode == FoldingPoolingTypes.NONE:
            return torch.cat(embeddings, dim=1)

        stack = torch.stack(embeddings)

        if self.pooling_mode == FoldingPoolingTypes.AVERAGE:
            return stack.mean(dim=0)

        elif self.pooling_mode == FoldingPoolingTypes.MAX:
            return stack.max(dim=0).values

        elif self.pooling_mode == FoldingPoolingTypes.SUM:
            return stack.sum(dim=0)

        # --- new modes ---
        elif self.pooling_mode == FoldingPoolingTypes.SLERP:
            # spherical mean across stacked steps
            out = stack[0]
            for i in range(1, stack.size(0)):
                alpha = (i + 1) / stack.size(0)
                out = F.normalize(out, dim=-1) * (1 - alpha) + \
                      F.normalize(stack[i], dim=-1) * alpha
            return out

        elif self.pooling_mode == FoldingPoolingTypes.TRIANGULAR:
            # smooth overlap‑add
            steps = stack.size(0)
            weights = torch.linspace(0.0, 1.0, steps, device=stack.device)
            weights = torch.minimum(weights, 1 - weights) * 2          # triangle
            weighted = stack * weights.view(-1, 1, 1, 1)               # broadcast
            return weighted.sum(dim=0) / (weights.sum() + 1e-6)

        else:
            return torch.cat(embeddings, dim=1)  # fallback
