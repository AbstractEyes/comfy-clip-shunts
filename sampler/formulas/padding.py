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
    def apply_padding(
            self,
            base: torch.Tensor,
            folded: torch.Tensor,
            pad_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            base: original embedding [B, T, D]
            folded: folded output [B, T, D]
            pad_mask: bool mask where True indicates a PAD token [B, T]
        """

        # --- 🔐 Defensive checks
        if pad_mask.dim() != 2:
            raise ValueError(f"pad_mask must be [B, T], got shape {pad_mask.shape}")
        if base.shape != folded.shape:
            raise ValueError(f"Shape mismatch: base {base.shape}, folded {folded.shape}")
        if pad_mask.shape[0] != base.shape[0] or pad_mask.shape[1] != base.shape[1]:
            raise ValueError(f"pad_mask {pad_mask.shape} does not align with base {base.shape}")

        # --- Expand mask to [B, T, 1] for broadcasting
        alpha = pad_mask.unsqueeze(-1).float()

        if self.padding_mode == FoldingPaddingTypes.NONE:
            return folded

        elif self.padding_mode == FoldingPaddingTypes.REPLACE:
            return torch.where(alpha.bool(), folded, base)

        elif self.padding_mode == FoldingPaddingTypes.INTERPOLATE:
            return alpha * folded + (1.0 - alpha) * base

        elif self.padding_mode == FoldingPaddingTypes.GAPPED:
            gapped = base.clone()
            gapped[pad_mask] = 0.0
            return gapped

        elif self.padding_mode == FoldingPaddingTypes.SPARSE:
            sparse = torch.zeros_like(base)
            sparse[~pad_mask] = folded[~pad_mask]
            return sparse

        else:
            return folded  # fallback

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
