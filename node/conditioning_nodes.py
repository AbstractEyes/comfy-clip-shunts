"""
    Conditioning Nodes
    Author: AbstractPhil

    This houses a uniquely designed, flexible, and highly potent conditioning node setup for ComfyUI.
    These are designed to be used with everything from simple text encoders to complex multimodal models.
    The conditioning system does not require any specific model to be used, and only requires encoded tensors.

    Core Conditioner:
        Replaces the current conditioning pipeline entirely with a new one.

    Conditioning Configuration:
        Includes a complex series of conditioning configurations that allow the encoders to be targeted and configured.
        Each encoder can be assigned a unique identifier, and each identifier can have it's own unqiue configuration.
        These include features like:
            * full scheduling and formula-capable internal components
            * advanced tokenization and encoder management capabilities
            * the standard simple and advanced conditioning nodes
            * soft and hard attention masks for conditioning shaping including formula access
            * dtype conversion and management for the conditioning tensors at runtime
            * shunt conditioning shaping using a multitude of experimental and advanced paper-driven techniques
            * runtime token classification for multi-shot capable conditioning using additional encoders

    Standard Conventional Object;
        This is the annoying ComfyUI object that is used to represent conditioning in ComfyUI, so we'll use that.

        [
            [
                combined_tensor,
                pooled_tensor,
                config_dict: Optional[dict],
            ], ...
        ]
        Nested lists... in a system built with excess power and dictionaries in mind. What a damn joke.

    Our Custom Encoder Conditioning Object; far more clean and usable.
        [
            {
                "encoder_id": "", # which encoder this came from
                "tensors": {
                    "source_encoder": str, # source of the tensor, e.g. clip's name, encoder's name, etc.
                    "conditioning": torch.Tensor, # the conditioning tensor
                    "modulation": torch.Tensor, # the modulation tensor, meant to be used for conditioning shaping
                    "cond_masks":
            }
"""
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from ..utils.alignment import match_feature_dims
import comfy


class ConditioningStackMultipleNode:
    """
    A node to stack up and organize up to 5 conditioning pipes.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "conditioning_1": ("CONDITIONING", {}),
                "conditioning_2": ("CONDITIONING", {}),
            },
            "optional": {
                "conditioning_3": ("CONDITIONING", {}),
                "conditioning_4": ("CONDITIONING", {}),
                "conditioning_5": ("CONDITIONING", {}),
            },
        }
    RETURN_TYPES = ("CONDITIONING",)
    RETURN_NAMES = ("conditionings",)
    FUNCTION = "stack_conditionings"
    CATEGORY = "utils/conditioning"

    def stack_conditionings(self, conditioning_1: list, conditioning_2: list, conditioning_3=None, conditioning_4=None, conditioning_5=None):
        """
        Stacks up to 5 conditioning pipes into a single conditioning stack.
        This allows for complex conditioning setups to be managed easily.
        """
        conditionings = []
        conditionings.extend(conditioning_1)
        conditionings.extend(conditioning_2)
        if conditioning_3 is not None:
            conditionings.extend(conditioning_3)
        if conditioning_4 is not None:
            conditionings.extend(conditioning_4)
        if conditioning_5 is not None:
            conditionings.extend(conditioning_5)

        return (conditionings,)

class NormalizeConditioningToMasksNode:
    """
    A node to create normalization soft-masks for conditioning tensors.
    This is a series of normalized multipliers that can be used to deterministically alter conditioning tensors of
    matching types, shapes, and scales.

        Standard Conditioning Object;
        This is the annoying ComfyUI object that is used to represent conditioning in ComfyUI, so we'll use that.

        [
            [
                combined_tensor,
                pooled_tensor,
                config_dict: Optional[dict],
            ], ...
        ]
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "conditioning": ("CONDITIONING", {}),
                "normalize_type": (["minmax", "zscore"], {"default": "minmax", "tooltip": "Normalization type to apply to the conditioning tensors."}),
                "individual": ("BOOLEAN", {"default": False, "tooltip": "If true, each conditioning tensor will be normalized individually."}),
                "norm_conds": ("BOOLEAN", {"default": True, "tooltip": "If true, the conditioning tensors will be normalized."}),
                "norm_pools": ("BOOLEAN", {"default": True, "tooltip": "If true, the pooled tensors will also be normalized."}),
            },
        }

    RETURN_TYPES = ("CONDITIONING",)
    RETURN_NAMES = ("normalized_conditioning",)
    FUNCTION = "normalize_conditioning"
    CATEGORY = "utils/conditioning"

    def normalize_conditioning(
            self,
            conditioning,  # List[[Tensor, Tensor, Optional[dict]]]  OR single such triple
            normalize_type="minmax",
            individual=False,
            norm_conds=True,
            norm_pools=True
    ):
        eps = 1e-6

        # 1) unify to list
        items = conditioning if isinstance(conditioning, list) else [conditioning]

        # 2) extract all conds & pools
        conds = [itm[0].clone() for itm in items]
        pools = [itm[1].clone() for itm in items]

        def _norm(x):
            if normalize_type == "minmax":
                lo = x.amin(dim=list(range(1, x.dim())), keepdim=True)
                hi = x.amax(dim=list(range(1, x.dim())), keepdim=True)
                return (x - lo) / (hi - lo + eps)
            # zscore
            mu = x.mean(dim=list(range(1, x.dim())), keepdim=True)
            sd = x.std(dim=list(range(1, x.dim())), keepdim=True)
            return (x - mu) / (sd + eps)

        # 3) either per‐item or across‐list
        if individual:
            new_conds = [_norm(c) if norm_conds else c for c in conds]
            new_pools = [_norm(p) if norm_pools else p for p in pools]
        else:
            new_conds, new_pools = conds, pools
            if norm_conds:
                stacked = torch.stack(conds, dim=0)  # [N,B,T,D]
                weights = stacked / (stacked.sum(dim=0, keepdim=True) + eps)
                new_conds = [stacked[i] * weights[i] for i in range(len(conds))]
            if norm_pools:
                stacked = torch.stack(pools, dim=0)  # [N,B,D']
                weights = stacked / (stacked.sum(dim=0, keepdim=True) + eps)
                new_pools = [stacked[i] * weights[i] for i in range(len(pools))]

        # 4) repackage, preserving cfg
        out = []
        for i, itm in enumerate(items):
            cfg = itm[2] if len(itm) > 2 else {}
            out.append([new_conds[i], new_pools[i], cfg])

        # 5) if original was single, unwrap
        return (out if isinstance(conditioning, list) else out[0],)


# 2. ConditioningProjectMultiple
class ConditioningProjectMultiple:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "conditioning": ("CONDITIONING", {}),
                "reference":    ("CONDITIONING", {}),
            }
        }
    RETURN_TYPES = ("CONDITIONING",)
    RETURN_NAMES = ("projected",)
    FUNCTION = "project_multiple"
    CATEGORY = "utils/conditioning"

    def project_multiple(self, conditioning: list, reference: list):
        """
        For each [combined, info] in conditioning, resample combined’s last dim
        to match reference[0][0].
        """
        if not reference or not isinstance(reference[0], list):
            return (conditioning,)
        ref_tensor = reference[0][0]
        out = []
        for combined, info in conditioning:
            proj = match_feature_dims(combined, ref_tensor)
            out.append([proj, info])
        return (out,)


# 3. ConditioningScaleMultiple
class ConditioningScaleMultiple:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "conditioning": ("CONDITIONING", {}),
                "scale":        ("FLOAT", {"default": 1.0, "min": 0.0}),
            }
        }
    RETURN_TYPES = ("CONDITIONING",)
    RETURN_NAMES = ("scaled",)
    FUNCTION = "scale_multiple"
    CATEGORY = "utils/conditioning"

    def scale_multiple(self, conditioning: list, scale: float):
        """
        Multiply each combined tensor and its pooled_output (if present) by scale.
        """
        out = []
        for combined, info in conditioning:
            c = combined * scale
            i = {}
            for k, v in info.items():
                i[k] = v * scale if isinstance(v, torch.Tensor) else v
            out.append([c, i])
        return (out,)


# 4. ConditioningSeparateMultiple
class ConditioningSeparateMultiple:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"conditioning": ("CONDITIONING", {})}}
    RETURN_TYPES = ("CONDITIONING", "CONDITIONING")
    RETURN_NAMES = ("combined_list", "pooled_list")
    FUNCTION = "separate_multiple"
    CATEGORY = "utils/conditioning"

    def separate_multiple(self, conditioning: list):
        """
        Split into two CONDITIONING lists:
        - combined_list: [[combined, {}], ...]
        - pooled_list:   [[pooled_output, {}], ...]
        """
        comb, pool = [], []
        for combined, info in conditioning:
            comb.append([combined, {}])
            pooled = info.get("pooled_output")
            pool.append([pooled, {}] if isinstance(pooled, torch.Tensor) else [torch.zeros_like(combined[..., :1]), {}])
        return (comb, pool)


# 5. ConditioningSaveLatents
class ConditioningSaveLatents:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "conditioning": ("CONDITIONING", {}),
                "path":         ("STRING", {}),
            }
        }
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("saved_path",)
    FUNCTION = "save_latents"
    CATEGORY = "utils/conditioning"

    def save_latents(self, conditioning: list, path: str):
        """
        Save all combined tensors to disk as a list of tensors.
        """
        tensors = [item[0].cpu() for item in conditioning]
        torch.save(tensors, path)
        return (path,)


# 6. ConditioningLoadLatents
class ConditioningLoadLatents:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"path": ("STRING", {})}}
    RETURN_TYPES = ("CONDITIONING",)
    RETURN_NAMES = ("conditioning",)
    FUNCTION = "load_latents"
    CATEGORY = "utils/conditioning"

    def load_latents(self, path: str):
        """
        Load a list of tensors from disk and wrap into CONDITIONING format.
        """
        tensors = torch.load(path)
        cond = [[t, {}] for t in tensors]
        return (cond,)


# 7. ConditioningApplySoftMask
class ConditioningApplySoftMask:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "conditioning": ("CONDITIONING", {}),
                "mask":         ("MASK", {}),
            }
        }
    RETURN_TYPES = ("CONDITIONING",)
    RETURN_NAMES = ("masked",)
    FUNCTION = "apply_soft_mask"
    CATEGORY = "utils/conditioning"

    def apply_soft_mask(self, conditioning: list, mask: torch.Tensor):
        """
        Multiply each combined tensor by mask[..., None].
        """
        out = []
        m = mask.unsqueeze(-1)
        m.to("cpu")
        for combined, info in conditioning:
            dev = combined.device
            combined = combined.to("cpu")
            modified = combined * m

            out.append([modified.to(dev), info])
        return (out,)


# 8. ConditioningApplyHardMask
class ConditioningApplyHardMask:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "conditioning": ("CONDITIONING", {}),
                "mask":         ("MASK", {}),
            }
        }
    RETURN_TYPES = ("CONDITIONING",)
    RETURN_NAMES = ("masked",)
    FUNCTION = "apply_hard_mask"
    CATEGORY = "utils/conditioning"

    def apply_hard_mask(self, conditioning: list, mask: torch.Tensor):
        """
        Binarize mask (>=0.5) then apply.
        """
        binm = (mask >= 0.5).float().unsqueeze(-1)
        out = []
        for combined, info in conditioning:
            out.append([combined * binm, info])
        return (out,)


# 9. ConditioningMultiply
class ConditioningMultiply:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "a": ("CONDITIONING", {}),
                "b": ("CONDITIONING", {}),
            }
        }
    RETURN_TYPES = ("CONDITIONING",)
    RETURN_NAMES = ("result",)
    FUNCTION = "multiply"
    CATEGORY = "utils/conditioning"

    def multiply(self, a: list, b: list):
        """
        Elementwise multiply two CONDITIONING lists of equal length.
        """
        out = []
        for (ca, ia), (cb, ib) in zip(a, b):
            info = {}
            # merge pooled outputs if present
            pa = ia.get("pooled_output"); pb = ib.get("pooled_output")
            if isinstance(pa, torch.Tensor) and isinstance(pb, torch.Tensor):
                info["pooled_output"] = pa * pb
            out.append([ca * cb, info])
        return (out,)


# 10. ConditioningFromEncoderConditioning
class ConditioningFromEncoderConditioning:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"encoder_conditioning": ("ENCODER_CONDITIONING", {})}}
    RETURN_TYPES = ("CONDITIONING",)
    RETURN_NAMES = ("conditioning",)
    FUNCTION = "from_encoder"
    CATEGORY = "utils/conditioning"

    def from_encoder(self, encoder_conditioning: list):
        """
        Convert ENCODER_CONDITIONING dict-list to standard CONDITIONING list.
        """
        out = []
        for obj in encoder_conditioning:
            cond = obj["tensors"].get("conditioning")
            mod  = obj["tensors"].get("modulation")
            info = {}
            if isinstance(mod, torch.Tensor):
                info["pooled_output"] = mod
            out.append([cond, info])
        return (out,)


# 11. ConditioningToEncoderConditioning
class ConditioningToEncoderConditioning:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "conditioning": ("CONDITIONING", {}),
                "encoder_id":   ("STRING", {}),
            }
        }
    RETURN_TYPES = ("ENCODER_CONDITIONING",)
    RETURN_NAMES = ("encoder_conditioning",)
    FUNCTION = "to_encoder"
    CATEGORY = "utils/conditioning"

    def to_encoder(self, conditioning: list, encoder_id: str):
        """
        Wrap standard CONDITIONING into ENCODER_CONDITIONING dict-list.
        """
        out = []
        for combined, info in conditioning:
            pooled = info.get("pooled_output")
            obj = {
                "encoder_id": encoder_id,
                "tensors": {
                    "conditioning": combined,
                    "modulation": pooled,
                    "cond_masks": None
                }
            }
            out.append(obj)
        return (out,)

import torch
import torch.nn.functional as F


# 1. Extract Token Mask
class ConditioningExtractTokenMaskNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "encoder_obj": ("ENCODER_CONDITIONING", {}),
                "clip_obj": ("CONDITIONING", {}),
                "top_k": ("INT", {"default": 1, "min": 1}),
            }
        }

    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("mask",)
    FUNCTION = "extract_token_mask"
    CATEGORY = "utils/encoder/conditioning"

    def extract_token_mask(self, encoder_obj, clip_obj, top_k):
        # 1) unify to lists
        encs = encoder_obj if isinstance(encoder_obj, list) else [encoder_obj]
        clips = clip_obj if isinstance(clip_obj, list) else [clip_obj]

        masks = []
        for e_obj, c_obj in zip(encs, clips):
            E = e_obj["tensors"]["conditioning"]  # [B,T,D]
            C = c_obj["tensors"]["conditioning"]  # [B,T,D]
            sim = F.cosine_similarity(E, C, dim=-1)  # [B,T]
            k = min(top_k, sim.size(-1))
            vals, idx = torch.topk(sim, k, dim=-1)
            m = torch.zeros_like(sim)
            m.scatter_(-1, idx, 1.0)
            masks.append(m)

        # 2) unwrap for single-input
        return (masks if isinstance(encoder_obj, list) else masks[0],)


# 2. Schedule Mask Over Time
class ConditioningScheduleMaskOverTimeNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "conditionings": ("ENCODER_CONDITIONING", {}),
                "encoder_id":    ("STRING", {"default": ""}),
                "steps":         ("INT",    {"default": 10, "min": 1}),
                "ramp_in":       ("INT",    {"default": 2,  "min": 0}),
                "ramp_out":      ("INT",    {"default": 2,  "min": 0}),
            }
        }
    RETURN_TYPES = ("CONDITIONING",)
    RETURN_NAMES = ("scheduled",)
    FUNCTION = "schedule_mask_over_time"
    CATEGORY = "utils/encoder/conditioning"

    def schedule_mask_over_time(self, conditionings, encoder_id, steps, ramp_in, ramp_out):
        # find matching encoder
        obj = next(o for o in conditionings if o["encoder_id"] == encoder_id)
        mask = obj["tensors"].get("cond_masks")
        if mask is None:
            mask = torch.ones_like(obj["tensors"]["conditioning"][...,0])  # [B,T]
        # build ramp
        lin   = torch.linspace(0, 1, steps)
        ramps = lin.clone()
        ramps[:ramp_in] = lin[:ramp_in]
        ramps[-ramp_out:] = lin[:ramp_out].flip(0)
        seq = []
        for r in ramps:
            new = obj.copy()
            new = {**new, "tensors": {**new["tensors"], "cond_masks": mask * r}}
            seq.append(new)
        return (seq,)


# 3. Compute Causal Delta
class ConditioningComputeCausalDeltaNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "before":    ("ENCODER_CONDITIONING", {}),
                "after":     ("ENCODER_CONDITIONING", {}),
                "normalize": ("BOOLEAN", {"default": False}),
            }
        }
    RETURN_TYPES = ("TENSOR",)
    RETURN_NAMES = ("delta",)
    FUNCTION = "compute_causal_delta"
    CATEGORY = "utils/encoder/conditioning"

    def compute_causal_delta(self, before, after, normalize):
        a0 = before["tensors"]["conditioning"]
        a1 = after["tensors"]["conditioning"]
        d  = a1 - a0
        if normalize:
            lo = d.amin(dim=list(range(1, d.dim())), keepdim=True)
            hi = d.amax(dim=list(range(1, d.dim())), keepdim=True)
            d  = (d - lo) / (hi - lo + 1e-6)
        return (d,)


# 4. Mix Gate
class ConditioningMixGateNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "obj":            ("ENCODER_CONDITIONING", {}),
                "hard_threshold": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0}),
                "soft_blur":      ("INT",   {"default": 3,   "min": 1}),
            }
        }
    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("gate",)
    FUNCTION = "mix_gate"
    CATEGORY = "utils/encoder/conditioning"

    def mix_gate(self, obj, hard_threshold, soft_blur):
        mask = obj["tensors"].get("cond_masks")
        if mask is None:
            mask = torch.ones_like(obj["tensors"]["conditioning"][...,0])
        gate = (mask >= hard_threshold).float().unsqueeze(1)  # [B,1,T]
        if soft_blur > 1:
            gate = F.avg_pool1d(gate, kernel_size=soft_blur, stride=1, padding=soft_blur//2)
        return (gate.squeeze(1),)


# 5. Low-Pass Filter
class ConditioningLowPassFilterNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "obj":    ("ENCODER_CONDITIONING", {}),
                "cutoff": ("INT", {"default": 1, "min": 0}),
            }
        }
    RETURN_TYPES = ("TENSOR",)
    RETURN_NAMES = ("filtered",)
    FUNCTION = "low_pass_filter"
    CATEGORY = "utils/encoder/conditioning"

    def low_pass_filter(self, obj, cutoff):
        cond = obj["tensors"]["conditioning"]  # [B,T,D]
        freq = torch.fft.rfft(cond, dim=1)
        freq[:, cutoff+1:] = 0
        out = torch.fft.irfft(freq, n=cond.size(1), dim=1)
        return (out,)


# 6. PCA Reduce
class ConditioningPCAReduceNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "conditionings": ("ENCODER_CONDITIONING", {}),
                "n_components":  ("INT", {"default": 2, "min": 1}),
            }
        }
    RETURN_TYPES = ("TENSOR",)
    RETURN_NAMES = ("reduced",)
    FUNCTION = "pca_reduce"
    CATEGORY = "utils/encoder/conditioning"

    def pca_reduce(self, conditionings, n_components):
        mats  = [o["tensors"]["conditioning"] for o in conditionings]  # list of [B,T,D]
        stack = torch.stack(mats, dim=0)                                  # [N,B,T,D]
        N,B,T,D = stack.shape
        X = stack.permute(1,2,0,3).reshape(B*T, N*D)
        U,S,V = torch.svd(X)
        W = V[:, :n_components]
        proj = X @ W
        out = proj.reshape(B, T, n_components)
        return (out,)


# 7. Reweight by Attention
class ConditioningReweightByAttentionNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "source": ("ENCODER_CONDITIONING", {}),
                "target": ("ENCODER_CONDITIONING", {}),
            }
        }
    RETURN_TYPES = ("TENSOR",)
    RETURN_NAMES = ("reweighted",)
    FUNCTION = "reweight_by_attention"
    CATEGORY = "utils/encoder/conditioning"

    def reweight_by_attention(self, source, target):
        A  = source["tensors"]["conditioning"]
        Bm = target["tensors"]["conditioning"]
        D  = A.size(-1)
        scores  = (A @ Bm.transpose(-2,-1)) / (D**0.5)
        weights = F.softmax(scores, dim=-1)
        out     = weights @ Bm
        return (out,)


# 8. Summary Statistics
class ConditioningSummaryStatsNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "conditioning": ("ENCODER_CONDITIONING", {}),
            }
        }
    RETURN_TYPES = ("TENSOR",)
    RETURN_NAMES = ("stats",)
    FUNCTION = "summary_stats"
    CATEGORY = "utils/encoder/conditioning"

    def summary_stats(self, conditioning):
        cond = conditioning["tensors"]["conditioning"]  # [B,T,D]
        mu   = cond.mean(dim=1)
        var  = cond.var(dim=1, unbiased=False)
        std  = var.sqrt() + 1e-6
        m3   = ((cond - mu.unsqueeze(1))**3).mean(dim=1)
        skew = m3 / (std**3)
        m4   = ((cond - mu.unsqueeze(1))**4).mean(dim=1)
        kurt = m4 / (var**2) - 3
        stats = torch.stack([mu, var, skew, kurt], dim=-1)
        return (stats,)


# 9. Dynamic Threshold
class ConditioningDynamicThresholdNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "conditioning": ("ENCODER_CONDITIONING", {}),
                "percentile":  ("FLOAT", {"default": 50.0, "min": 0.0, "max": 100.0}),
            }
        }
    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("mask",)
    FUNCTION = "dynamic_threshold"
    CATEGORY = "utils/encoder/conditioning"

    def dynamic_threshold(self, conditioning, percentile):
        scores = conditioning["tensors"]["modulation"].mean(dim=-1)
        thresh = scores.quantile(percentile/100, dim=-1, keepdim=True)
        mask = (scores >= thresh).float()
        return (mask,)


# 10. Morphological Filter
class ConditioningMorphologicalFilterNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "conditioning": ("ENCODER_CONDITIONING", {}),
                "operation":    (["erode", "dilate"], {"default": "erode"}),
                "kernel_size":  ("INT", {"default": 3, "min": 1}),
            }
        }
    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("mask",)
    FUNCTION = "morphological_filter"
    CATEGORY = "utils/encoder/conditioning"

    def morphological_filter(self, conditioning, operation, kernel_size):
        mask = conditioning["tensors"].get("cond_masks")
        if mask is None:
            mask = torch.ones_like(conditioning["tensors"]["conditioning"][...,0])
        x = mask.unsqueeze(1)
        if operation == "erode":
            y = F.max_pool1d(1 - x, kernel_size, stride=1, padding=kernel_size//2)
            out = 1 - y
        else:
            out = F.max_pool1d(x, kernel_size, stride=1, padding=kernel_size//2)
        return (out.squeeze(1),)


# 11. Exponential Ramp
class ConditioningExpRampNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "conditioning": ("ENCODER_CONDITIONING", {}),
                "alpha":        ("FLOAT", {"default": 1.0}),
                "beta":         ("FLOAT", {"default": 1.0}),
            }
        }
    RETURN_TYPES = ("TENSOR",)
    RETURN_NAMES = ("ramped",)
    FUNCTION = "exp_ramp"
    CATEGORY = "utils/encoder/conditioning"

    def exp_ramp(self, conditioning, alpha, beta):
        x = conditioning["tensors"]["conditioning"]
        out = torch.exp(-alpha * x.abs()) - torch.exp(-beta * x.abs())
        return (out,)


# 12. Blend Over Steps
class ConditioningBlendOverStepsNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "objA":     ("ENCODER_CONDITIONING", {}),
                "objB":     ("ENCODER_CONDITIONING", {}),
                "schedule": ("TENSOR", {}),
            }
        }
    RETURN_TYPES = ("CONDITIONING",)
    RETURN_NAMES = ("sequence",)
    FUNCTION = "blend_over_steps"
    CATEGORY = "utils/encoder/conditioning"

    def blend_over_steps(self, objA, objB, schedule):
        A = objA["tensors"]["conditioning"]
        B = objB["tensors"]["conditioning"]
        seq = []
        for w in schedule:
            cond = (1 - w) * A + w * B
            new = objA.copy()
            new = {**new, "tensors": {**new["tensors"], "conditioning": cond}}
            seq.append(new)
        return (seq,)
