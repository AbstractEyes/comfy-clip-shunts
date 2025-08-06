from dataclasses import dataclass
from typing import List, Optional
import torch
import torch.nn.functional as F
from torch import nn, sort
import logging
logger = logging.getLogger(__name__)


from .modes import FoldingPaddingTypes, FoldingPoolingTypes
# --------------------------------------------------------------------- #

from scipy.cluster.hierarchy import linkage, leaves_list


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
        if config is not None:
            self.config = config
            self.pooling_mode = self.config.get("pooling_mode", FoldingPoolingTypes.AVERAGE)

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
            # 2D convolution over symbolic steps × tokens
            steps, A, B, D = stack.shape

            # Rearrange to [B, steps, A * D]
            reshaped = stack.permute(2, 0, 1, 3)  # [B, steps, A, D]
            B, S, A, D = reshaped.shape
            reshaped = reshaped.reshape(B, S, A * D)  # [B, steps, flat]
            reshaped = reshaped.unsqueeze(1)  # [B, 1, steps, flat]

            # Convolution kernel
            kernel = torch.ones(1, 1, 2, 1, device=stack.device) / 2.0

            # Apply 2D convolution over (steps, features)
            conv_out = F.conv2d(reshaped, kernel, padding=(1, 0))  # [B, 1, steps+1, A*D]

            # Pad flat feature dim if needed
            flat = conv_out.shape[-1]
            pad = (D - (flat % D)) % D
            if pad != 0:
                conv_out = F.pad(conv_out, (0, pad), mode="constant", value=0.0)
                flat += pad

            # Restore shape
            A_out = flat // D
            conv_out = conv_out.squeeze(1).reshape(B, -1, A_out, D)  # [B, steps+1, A, D]

            # Average across steps, then permute to [A, B, D]
            return conv_out.mean(dim=1).permute(1, 0, 2)

        elif self.pooling_mode == FoldingPoolingTypes.CONV3:
            # 3D convolution across (steps, tokens, features)
            steps, A, B, D = stack.shape

            # Rearrange: [steps, A, B, D] → [B, 1, steps, A, D]
            reshaped = stack.permute(2, 0, 1, 3).unsqueeze(1)  # [B, 1, steps, A, D]

            # 3D kernel over (steps × A), leave D untouched
            kernel = torch.ones(1, 1, 3, 3, 1, device=stack.device) / 9.0
            padded = F.pad(reshaped, (0, 0, 1, 1, 1, 1))  # pad steps and A

            conv_out = F.conv3d(padded, kernel)  # [B, 1, steps, A, D]

            result = conv_out.squeeze(1).mean(dim=1)  # [B, A, D]
            return result.permute(1, 0, 2)  # → [A, B, D]

        elif self.pooling_mode == FoldingPoolingTypes.CONV4:
            # 4D convolution pooling
            # Simulated 4D convolution over (steps, tokens, features)
            steps, A, B, D = stack.shape

            reshaped = stack.permute(2, 0, 1, 3).unsqueeze(1)  # [B, 1, steps, A, D]
            padded = F.pad(reshaped, (1, 1, 1, 1, 1, 1))  # pad D, A, steps

            kernel = torch.ones(1, 1, 3, 3, 3, device=stack.device) / 27.0

            conv_out = F.conv3d(padded, kernel)  # [B, 1, steps, A, D]
            result = conv_out.squeeze(1).mean(dim=1)  # [B, A, D]
            return result.permute(1, 0, 2)  # → [A, B, D]


        elif self.pooling_mode == FoldingPoolingTypes.SIMILARITY_O: # similarity overlap pooling
            # overlap pooling based on similarity
            steps, A, B, D = stack.shape
            pool = torch.zeros(A, B, D, device=stack.device)
            for i in range(steps):
                ref = stack[i]  # [A, B, D]
                sim = torch.zeros_like(ref)
                for j in range(i + 1):
                    sim += F.cosine_similarity(ref, stack[j], dim=-1).unsqueeze(-1) * stack[j]
                pool += sim / (i + 1)
            return pool / steps  # [A, B, D]

        elif self.pooling_mode == FoldingPoolingTypes.SIMILARITY_X: # cross-similarity pooling
            # cross similarity pooling, reorders based on the highest similarity before merging
            steps, A, B, D = stack.shape
            pool = torch.zeros(A, B, D, device=stack.device)
            for s in range(steps):
                ref = stack[s].mean(dim=0, keepdim=True)  # [1, B, D]
                sim = F.cosine_similarity(stack[s], ref.expand_as(stack[s]), dim=-1)  # [A, B]
                idx = sim.argmax(dim=0)  # [B]
                gathered = torch.stack([
                    stack[s, idx[b], b] for b in range(B)
                ])  # [B, D]
                pool[:, torch.arange(B)] += gathered.unsqueeze(0).expand(A, -1, -1)
            return pool / steps  # [A, B, D]


        elif self.pooling_mode == FoldingPoolingTypes.SIMILARITY_MASK: # similarity mask pooling
            # determines the feature similarity based on the delta masks
            steps, A, B, D = stack.shape
            pool = torch.zeros(A, B, D, device=stack.device)

            for s in range(steps):
                ref = stack[s]  # [A, B, D]
                sim_sum = torch.zeros_like(ref)
                for t in range(s + 1):
                    sim = F.cosine_similarity(ref, stack[t], dim=-1)  # [A, B]
                    mask = (sim > 0.5).float().unsqueeze(-1)  # [A, B, 1]
                    sim_sum += mask * stack[t]  # [A, B, D]
                pool += sim_sum / (s + 1)

            return pool / steps  # → [A, B, D]


        elif self.pooling_mode == FoldingPoolingTypes.BILINEAR:
            # bilinear pooling
            steps, A, B, D = stack.shape
            out = torch.zeros(B, D, device=stack.device)
            for i in range(steps):
                out += stack[:, :, i] * stack[:, :, :i+1].mean(dim=2)
            return out / steps
        elif self.pooling_mode == FoldingPoolingTypes.FLOOD:
            # cumulative mean fill over steps
            steps, A, B, D = stack.shape
            flood = torch.zeros(steps, A, B, D, device=stack.device)

            for s in range(steps):
                flood[s] = stack[:s + 1].mean(dim=0)  # mean over previous steps
            return flood.mean(dim=0)  # final output: [A, B, D]
        elif self.pooling_mode == FoldingPoolingTypes.SIMILARITY_TREE:
            """
            Tree-based pooling: reorders segments via pairwise cosine similarity
            and pools them in dendrogram-leaf order.
            """
            steps, B, T, D = stack.shape

            # Pool each segment
            pooled = stack.mean(dim=2)  # [steps, B, D]

            out = []
            for b in range(B):
                vectors = pooled[:, b]  # [steps, D]
                sim_matrix = F.cosine_similarity(vectors.unsqueeze(1), vectors.unsqueeze(0), dim=-1)
                dist = 1.0 - sim_matrix.cpu().numpy()
                linkage_matrix = linkage(dist, method='centroid', optimal_ordering=True)
                order = leaves_list(linkage_matrix)

                ordered_stack = stack[order, b]  # [steps, T, D]
                pooled_b = ordered_stack.mean(dim=0)  # [T, D]
                out.append(pooled_b)

            return torch.stack(out, dim=0)  # [B, T, D]
        else:
            return torch.cat(embeddings, dim=1)  # fallback