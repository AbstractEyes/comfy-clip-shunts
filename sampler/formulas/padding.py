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
    def apply_padding(
            self,
            base: torch.Tensor,  # [B, T, D]
            folded: torch.Tensor,  # [B, T, D]
            mask: torch.Tensor,  # [B, T] | [B, T, 1] | [B, T, D]
            config: Optional[dict] = None
    ) -> torch.Tensor:
        """
        Applies padding logic between base and folded embeddings using the selected padding mode.
        Each mode manipulates `folded` based on a `mask` that reflects token weights, similarity, or importance.

        Args:
            base:    Original input tensor [B, T, D]
            folded:  Modified or processed tensor [B, T, D]
            mask:    Control mask; shape determines application style
            config:  Optional config dict with padding_mode and parameters

        Returns:
            Modified tensor after applying the selected padding mode
        """
        if not isinstance(base, torch.Tensor) or not isinstance(folded, torch.Tensor):
            raise TypeError(f"[Alucard] Expected base/folded to be torch.Tensor, got {type(base)} and {type(folded)}")

        if config is None:
            config = {}
        self.padding_mode = config.get("padding_mode", self.padding_mode or FoldingPaddingTypes.NONE)

        # --- Normalize and validate mask shape ---
        if mask.dim() == 2:
            mask = mask.unsqueeze(-1).float()  # [B,T] → [B,T,1]
        elif mask.dim() == 3:
            mask = mask.float()  # [B,T,1] or [B,T,D]
        else:
            raise ValueError(f"[Alucard] Invalid mask shape {mask.shape}")

        if mask.shape[:2] != base.shape[:2]:
            raise ValueError(f"[Alucard] Mask shape {mask.shape[:2]} != base shape {base.shape[:2]}")

        # --- Grab params ---
        mode = self.padding_mode
        B, T, D = base.shape

        # ========== BASIC MODES ==========

        # NONE → return folded unchanged
        if mode == FoldingPaddingTypes.NONE:
            return folded

        # INTERPOLATE → blend folded and base using [0,1] mask
        if mode == FoldingPaddingTypes.INTERPOLATE:
            m = torch.clamp(mask, 0.0, 1.0)
            if torch.allclose(m, torch.zeros_like(m)) or torch.allclose(m, torch.ones_like(m)):
                return folded
            return m * folded + (1.0 - m) * base

        # REPLACE → use folded if mask > 0.5, otherwise base
        if mode == FoldingPaddingTypes.REPLACE:
            return torch.where(mask > 0.5, folded, base)

        # GAPPED → zero out positions where mask > 0.5
        if mode == FoldingPaddingTypes.GAPPED:
            return torch.where(mask > 0.5, torch.zeros_like(base), base)

        # SPARSE → keep only masked values of folded, zero elsewhere
        if mode == FoldingPaddingTypes.SPARSE:
            out = torch.zeros_like(base)
            out[mask > 0.5] = folded[mask > 0.5]
            return out

        # --- BLEND family -----------------------------------------------------------
        def _prep_mask(m: torch.Tensor, blur_sigma: int | float | None = None) -> torch.Tensor:
            """Clamp, de-NaN and (optionally) Gaussian-blur a [B,T,1] mask."""
            m = torch.nan_to_num(m, nan=0., posinf=1., neginf=0.)  # finite
            m = torch.clamp(m, 0., 1.)  # [0,1]
            if blur_sigma:  # simple 1-D blur
                B, T, _ = m.shape
                k = torch.arange(T, device=m.device).float()
                g = torch.exp(-0.5 * ((k - T // 2) / blur_sigma) ** 2)
                g = (g / g.sum()).view(1, 1, -1)  # [1,1,T]
                m = F.conv1d(m.transpose(1, 2), g, padding=T // 2).transpose(1, 2)
            return m

        # inside apply_padding …
        if mode in (FoldingPaddingTypes.BLEND, FoldingPaddingTypes.BLEND2):
            # 1️⃣ collapse any [B,T,D] mask to scalar weight
            if mask.shape[-1] != 1:
                m = mask.mean(dim=-1, keepdim=True)
            else:
                m = mask
            # 2️⃣ clean & optionally soften
            m = _prep_mask(m, config.get("blur_sigma", 0))
            thresh = config.get("thresh", .5)
            if mode == FoldingPaddingTypes.BLEND2:
                m = torch.where(m >= thresh, m, m.new_zeros(()))  # force base below threshold
            # 3️⃣ up-cast for accuracy, blend, then restore dtype
            out = m.float() * folded.float() + (1. - m.float()) * base.float()
            return out.to(base.dtype)
        ## BLEND → continuous blend from base to folded
        #if mode == FoldingPaddingTypes.BLEND:
        #    return base * (1.0 - mask) + folded * mask

        ## BLEND2 → blend only where mask ≥ 0.5, otherwise retain base
        #if mode == FoldingPaddingTypes.BLEND2:
        #    blended = base * (1.0 - mask) + folded * mask
        #    blended[mask < 0.5] = base[mask < 0.5]
        #    return blended

        # ========== EDGE MASKING FIX ==========

        # MASK_EDGES → smooth fade-out at front and back (cosine taper)
        if mode == FoldingPaddingTypes.MASK_EDGES:
            if T < 8:
                logger.warning("[Alucard] mask_edges: sequence too short")
                return folded
            m = torch.ones_like(mask)
            edge_len = T // 8
            edge = torch.linspace(0, 1, steps=edge_len, device=base.device).view(1, -1, 1)
            edge = (1 - torch.cos(torch.pi * edge)) / 2  # cosine bell
            m[:, :edge_len] *= edge
            m[:, -edge_len:] *= torch.flip(edge, dims=[1])
            return base * (1.0 - m) + folded * m

        # MASK_TOP_K → preserve top-k of folded, overwrite rest with base
        if mode == FoldingPaddingTypes.MASK_TOP_K:
            k = int(T * 0.25)
            top_k_mask = torch.zeros_like(mask, dtype=torch.bool)
            top_k_mask[:, :k] = True
            out = folded.clone()
            out[top_k_mask < 0.5] = base[top_k_mask < 0.5]
            return out

        # ========== NEW MODES ==========

        # SHUFFLE → randomly shuffle token dimension for each batch
        if mode == FoldingPaddingTypes.SHUFFLE:
            indices = torch.argsort(torch.rand(B, T, device=base.device), dim=1)
            return torch.gather(folded, dim=1, index=indices.unsqueeze(-1).expand(-1, -1, D))

        # MASK_BOTTOM_K → suppress bottom-k values based on mask scores
        if mode == FoldingPaddingTypes.MASK_BOTTOM_K:
            k_frac = config.get("bottom_k_frac", 0.25)
            hard = config.get("hard", False)
            k = int(T * k_frac)
            mask_flat = mask.squeeze(-1)
            sorted_mask, indices = torch.sort(mask_flat, dim=1)
            mask_new = mask.clone()
            for b in range(B):
                bottom_k = indices[b, :k]
                if hard:
                    mask_new[b, bottom_k] = 0.0
                else:
                    mask_new[b, bottom_k] *= 0.25
            return mask_new * folded + (1.0 - mask_new) * base

        # --- BELL_MOLD ---------------------------------------------------------------
        if mode == FoldingPaddingTypes.BELL_MOLD:
            """
            Re-orders tokens so those with the **lowest pad-mask weight** move to the
            edges and those with the **highest weight** move toward the centre
            in a bell-curve pattern.

            Accepts every valid mask rank:
              • [B, T]      (boolean / weight per token)
              • [B, T, 1]   (broadcast weight)
              • [B, T, D]   (per-feature mask)   ← this was the crashing case
            """
            # ---------- normalise mask to [B, T] scalar weights ----------
            if mask.dim() == 3:
                if mask.shape[-1] == 1:  # [B,T,1] → [B,T]
                    scores = mask.squeeze(-1)
                else:  # [B,T,D] → mean over D
                    scores = mask.mean(dim=-1)
            else:  # already [B,T]
                scores = mask

            B, T, D = folded.shape
            if T < 2:
                logger.warning("[BELL_MOLD] sequence too short – returning folded")
                return folded

            # bell pattern index: centre-first, edges-last  e.g. [3,2,4,1,5,0,6…]
            bell_pattern = torch.argsort(
                torch.abs(torch.arange(T, device=folded.device) - (T // 2))
            )  # [T]

            output = torch.empty_like(folded)

            for b in range(B):
                token_weights = scores[b]  # [T]
                #  low-weight → 0, high-weight → 1   (already in [0,1])
                sort_idx = torch.argsort(token_weights)  # [T] ascending
                perm_idx = sort_idx[bell_pattern]  # [T] bell-re-ordered
                output[b] = folded[b][perm_idx.long()]  # [T,D]

            return output

        # Fallback: return unchanged folded
        return folded



