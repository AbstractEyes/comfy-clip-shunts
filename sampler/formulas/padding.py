from dataclasses import dataclass
from typing import List
import torch
import torch.nn.functional as F
from torch import nn, sort
import logging
logger = logging.getLogger(__name__)


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
            mask = mask.clone().detach()
            mask[:, :mask.shape[1] // 4] = 0.0
            mask[:, -mask.shape[1] // 4:] = 0.0
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


    # --- POOLING ---
    def apply_pooling(self, embeddings: List[torch.Tensor]) -> torch.Tensor:
        """
        Args:
            embeddings: list of [B, T, D] or [B, D] tensors to pool across
        """


        stack = torch.stack(embeddings)
        logger.info(f"[Alucard] Pooling {len(embeddings)} embeddings of shape {stack.shape} with mode {self.pooling_mode}")

        if self.pooling_mode == FoldingPoolingTypes.NONE: # if none default to mean pooling
            return stack.mean(dim=0)

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

        elif self.pooling_mode == FoldingPoolingTypes.CONV2:
            # 2D convolution pooling
            steps, A, B, D = stack.shape
            stack = stack.unsqueeze(1)
            kernel = torch.ones(1, 1, 2, 2, device=stack.device) / 4.0
            pooled = F.conv2d(stack, kernel, padding=1, groups=B)
            return pooled.squeeze(1).mean(dim=0)
        elif self.pooling_mode == FoldingPoolingTypes.CONV3:
            # 3D convolution pooling

            steps, A, B, D = stack.shape
            stack = stack.unsqueeze(1)  # [B, 1, T, A, B, D]
            kernel = torch.ones(1, 1, 2, 2, 2, device=stack.device) / 8.0
            pooled = F.conv3d(stack, kernel, padding=1, groups=B)
            return pooled.squeeze(1).mean(dim=0)
        elif self.pooling_mode == FoldingPoolingTypes.CONV4:
            # 4D convolution pooling
            steps, A, B, D = stack.shape
            stack = stack.unsqueeze(1)
            kernel = torch.ones(1, 1, 2, 2, 2, 2, device=stack.device) / 16.0
            pooled = F.conv3d(stack, kernel, padding=1, groups=B)
            return pooled.squeeze(1).mean(dim=0)

        elif self.pooling_mode == FoldingPoolingTypes.SIMILARITY_O: # similarity overlap pooling
            # overlap pooling based on similarity
            steps, A, B, D = stack.shape
            out = torch.zeros(B, D, device=stack.device)
            for i in range(steps):
                sim = F.cosine_similarity(stack[:, i, :].unsqueeze(1), stack[:, :i+1, :], dim=-1)
                out += sim.mean(dim=1) * stack[:, i, :]
            T = steps * (steps + 1) // 2  # total number of steps
            return out / T

        elif self.pooling_mode == FoldingPoolingTypes.SIMILARITY_X: # cross-similarity pooling
            # cross similarity pooling, reorders based on the highest similarity before merging
            steps, A, B, D = stack.shape
            pool = torch.zeros(B, D, device=stack.device)
            for i in range(steps):
                sim = F.cosine_similarity(stack[:, i, :].unsqueeze(1), stack, dim=-1)
                idx = sim.argmax(dim=1)
                pool += stack[torch.arange(B), idx, :]
            sorted = sort(pool, dim=1, descending=True)
            return sorted.values.mean(dim=1)

        elif self.pooling_mode == FoldingPoolingTypes.SIMILARITY_MASK: # similarity mask pooling
            # determines the similarity based on the delta masks
            steps, A, B, D = stack.shape
            out = torch.zeros(B, D, device=stack.device)
            for i in range(steps):
                sim = F.cosine_similarity(stack[:, i, :].unsqueeze(1), stack[:, :i+1, :], dim=-1)
                mask = (sim > 0.5).float()
                out += (mask * stack[:, i, :]).mean(dim=1)
            return out / steps

        elif self.pooling_mode == FoldingPoolingTypes.BILINEAR:
            # bilinear pooling
            steps, A, B, D = stack.shape
            out = torch.zeros(B, D, device=stack.device)
            for i in range(steps):
                out += stack[:, i, :] * stack[:, :i+1, :].mean(dim=1)
            return out / T
        elif self.pooling_mode == FoldingPoolingTypes.FLOOD:
            # flood fill pooling
            steps, A, B, D = stack.shape
            flood = torch.zeros(B, T, D, device=stack.device)
            for i in range(steps):
                flood[:, i, :] = stack[:, :i+1, :].mean(dim=1)
            return flood

        else:
            return torch.cat(embeddings, dim=1)  # fallback
