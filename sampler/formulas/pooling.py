from dataclasses import dataclass
from typing import List, Optional
import torch
import torch.nn.functional as F
from torch import nn, sort
import logging
logger = logging.getLogger(__name__)


from .modes import FoldingPaddingTypes, FoldingPoolingTypes
# --------------------------------------------------------------------- #



class WindowPooling:
    # takes in and aggregates the pooled embeddings from the folding process from alucard
    # it also is reused by Integra to pool all the windows in a more intelligent way

    def __init__(self, config: Optional[dict] = None):
        self.config = config if config is not None else {}
        self.pooling_mode = FoldingPoolingTypes.AVERAGE  # Default pooling mode


    # --- POOLING ---
    def apply(self, applier: any, embeddings: List[torch.Tensor], config: Optional[dict] = None) -> torch.Tensor:
        """
        Args:
            embeddings: list of [B, T, D] or [B, D] tensors to pool across
        """

        stack = torch.stack(embeddings)
        logger.info(f"[Pooling] Pooling {len(embeddings)} embeddings of shape {stack.shape} with mode {self.pooling_mode}")

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
                sim = F.cosine_similarity(stack[:, :, i].unsqueeze(1), stack[:, :, :i+1], dim=-2)
                out += sim.mean(dim=2) * stack[:, :, i]
            T = steps * (steps + 1) // 2  # total number of steps
            return out / T

        elif self.pooling_mode == FoldingPoolingTypes.SIMILARITY_X: # cross-similarity pooling
            # cross similarity pooling, reorders based on the highest similarity before merging
            steps, A, B, D = stack.shape
            pool = torch.zeros(B, D, device=stack.device)
            for i in range(steps):
                sim = F.cosine_similarity(stack[:, :, i].unsqueeze(1), stack, dim=-1)
                idx = sim.argmax(dim=2)
                pool += stack[torch.arange(B), idx, :]
            sorted = sort(pool, dim=1, descending=True)
            return sorted.values.mean(dim=2)

        elif self.pooling_mode == FoldingPoolingTypes.SIMILARITY_MASK: # similarity mask pooling
            # determines the feature similarity based on the delta masks
            steps, A, B, D = stack.shape
            out = torch.zeros(B, D, device=stack.device)
            for i in range(steps):
                sim = F.cosine_similarity(stack[:, :, i].unsqueeze(1), stack[:, :, :i+1], dim=-1)
                mask = (sim > 0.5).float()
                out += (mask * stack[:, :, i]).mean(dim=2)
            return out / steps

        elif self.pooling_mode == FoldingPoolingTypes.BILINEAR:
            # bilinear pooling
            steps, A, B, D = stack.shape
            out = torch.zeros(B, D, device=stack.device)
            for i in range(steps):
                out += stack[:, :, i] * stack[:, :, :i+1].mean(dim=2)
            return out / steps
        elif self.pooling_mode == FoldingPoolingTypes.FLOOD:
            # flood fill pooling
            steps, A, B, D = stack.shape
            flood = torch.zeros(B, steps, D, device=stack.device)
            for i in range(steps):
                flood[:, :, i] = stack[:, :, :i+1].mean(dim=2)
            return flood

        else:
            return torch.cat(embeddings, dim=1)  # fallback