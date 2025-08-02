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
import logging

logger = logging.getLogger(__name__)


from ..utils.conditioning_shifter import ConditioningShifter
from ..utils.rose_util import rose_score

import torch
from ..utils.rose_util import rose_score

class ApplyRoseScoreNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "encodings": ("MODULATION_CONDITIONING", {}),
                "trajectory": ("CONDITIONING", {}),
                "similarity": ("CONDITIONING", {}),
                "conditionings": ("CONDITIONING", {}),
                "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0}),
            }
        }

    RETURN_TYPES = ("CONDITIONING",)
    RETURN_NAMES = ("conditioning",)
    FUNCTION = "apply_rose_score"
    CATEGORY = "utils/conditioning"

    def apply_rose_score(self, encodings, trajectory, similarity, conditionings, strength):
        output = []

        # Extract shared vectors from input
        need     = trajectory[0][0][0, 0]                        # (D,)
        relation = similarity[0][0][0, 0]                        # (D,)
        purpose  = encodings[0]["tensors"]["modulation"][0]     # (D,)

        for combined, info in conditionings:
            rose = rose_score(combined, need, relation, purpose).unsqueeze(-1).unsqueeze(-1)  # (B,T,1)
            aligned = (1 - strength) * combined + strength * (combined * rose)
            output.append([aligned, info])

        return (output,)




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
                #"time_start": ("INT", {"default": 0.0, "min": 0, "max": 1.0, "tooltip": "Start time for the first conditioning."}),
                #"time_end": ("INT", {"default": 1.0, "min": 0, "max": 1.0, "tooltip": "End time for the last conditioning."}),
                #"cond_strength": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "tooltip": "Strength of the conditioning stack."}),
                #"pool_strength": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "tooltip": "Strength of the pooled output."}),
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

    def stack_conditionings(self,
                            conditioning_1: list,
                            conditioning_2: list,
                            #time_start: float = 0,
                            #time_end: float = 1.0,
                            #cond_strength: float = 1.0,
                            #pool_strength: float = 1.0,
                            conditioning_3=None,
                            conditioning_4=None,
                            conditioning_5=None):
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

        #ConditioningShifter.conditioning_set_values(conditionings, {"time_start", time_start, "time_start", time_end })
        #if cond_strength != 1.0 or pool_strength != 1.0:
        #    conditionings = ConditioningShifter.conditioning_set_strength(conditionings, cond_strength, pool_strength)

        return (conditionings,)

class ConditioningSetDtypeNode:
    """
    A node to set the dtype of all conditioning tensors.
    This is useful for ensuring that the conditioning tensors are in the correct dtype for the model.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "conditioning": ("CONDITIONING", {}),
                "dtype":       (["float64", "float32", "float16", "bfloat16"], {"default": "float32", "tooltip": "Dtype to set the conditioning tensors to."}),
            }
        }
    RETURN_TYPES = ("CONDITIONING",)
    RETURN_NAMES = ("conditioning",)
    FUNCTION = "set_dtype_conds"
    CATEGORY = "utils/conditioning"

    def set_dtype_conds(self, conditioning: list, dtype: str):
        """
        Set the dtype for all conditioning tensors.
        This is useful for ensuring that the conditioning tensors are in the correct dtype.
        """
        out = []
        for combined, info in conditioning:
            combined = combined.to(dtype)
            for k, v in info.items():
                if isinstance(v, torch.Tensor):
                    info[k] = v.to(dtype)
            out.append([combined, info])
        return (out,)

class ConditioningSetDeviceNode:
    # guarantee the device is set for the conditioning tensors
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "conditioning": ("CONDITIONING", {}),
                "device":       (["cpu", "cuda", "mps"], {"default": "cpu", "tooltip": "Device to set the conditioning tensors to."}),
            }
        }
    RETURN_TYPES = ("CONDITIONING",)
    RETURN_NAMES = ("conditioning",)
    FUNCTION = "set_device_conds"
    CATEGORY = "utils/conditioning"
    def set_device_conds(self, conditioning: list, device: str):
        """
        Set the device for all conditioning tensors.
        This is useful for ensuring that the conditioning tensors are on the correct device.
        """
        out = []
        for combined, info in conditioning:
            combined = combined.to(device)
            for k, v in info.items():
                if isinstance(v, torch.Tensor):
                    info[k] = v.to(device)
                if k == "device":
                    info[k] = device
            out.append([combined, info])
        return (out,)

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
                "proj_method": (["linear", "bilinear", "bicubic", "trilinear"], {"default": "linear",
                                                                                 "tooltip": "Method to use for projecting the conditioning tensors."}),
                "primary_batch":      ("INT", {"default": 1, "min": -1, "max": 100, "tooltip": "Number of conditioning tensors to project."}),
                "primary_tokens":   ("INT", {"default": 77, "min": -1, "max": 1000, "tooltip": "Number of tokens to project."}),
                "primary_features": ("INT", {"default": 4096, "min": -1, "max": 8192, "tooltip": "Number of features to project."}),

                "pooled_batch": ("INT", {"default": 1, "min": -1, "max": 100, "tooltip": "Number of pooled tensors to project."}),
                "pooled_tokens": ("INT", {"default": 1, "min": -1, "max": 100, "tooltip": "Number of pooled tokens to project."}),
                "pooled_features": ("INT", {"default": 768, "min": -1, "max": 4096, "tooltip": "Number of pooled features to project."}),

            }
        }
    RETURN_TYPES = ("CONDITIONING",)
    RETURN_NAMES = ("projected",)
    FUNCTION = "project_multiple"
    CATEGORY = "utils/conditioning"

    def project_multiple(self, conditioning: list,
                         proj_method: str = "linear",
                         primary_batch: int = -1,
                         primary_tokens: int = -1,
                         primary_features: int = -1,
                         pooled_tokens: int = -1,
                         pooled_batch: int = -1,
                         pooled_features: int = -1):
        """
        For each [combined, info] in conditioning, resample combined’s last dim
        to match reference[0][0].
        """
        try:
            conditioning = list(conditioning)
            out = []
            for combined, info in conditioning:
                # 1) get reference shape
                logger.info(f"Projecting conditioning with method {proj_method}")
                combined = combined.clone()
                info = dict(info)  # ensure info is a dict
                ref = combined.shape
                if primary_batch < 0: primary_batch = ref[0]
                if primary_tokens < 0: primary_tokens = ref[1]
                if primary_features < 0: primary_features = ref[2]

                # 2) resample combined
                if proj_method == "linear":
                    combined = match_feature_dims(combined, torch.zeros((primary_batch, primary_tokens, primary_features), device=combined.device), mode="linear")
                elif proj_method == "bilinear":
                    combined = match_feature_dims(combined, torch.zeros((primary_batch, primary_tokens, primary_features), device=combined.device), mode="bilinear")
                elif proj_method == "bicubic":
                    combined = match_feature_dims(combined, torch.zeros((primary_batch, primary_tokens, primary_features), device=combined.device), mode="bicubic")
                elif proj_method == "trilinear":
                    combined = match_feature_dims(combined, torch.zeros((primary_batch, primary_tokens, primary_features), device=combined.device), mode="trilinear")

                # 3) resample pooled
                pooled = info.get("pooled_output", None)
                if pooled is not None:
                    # clone it
                    pooled = pooled.clone()
                    if pooled_batch < 0: pooled_batch = pooled.shape[0]
                    if pooled_tokens < 0: pooled_tokens = pooled.shape[1]
                    if pooled_features < 0: pooled_features = pooled.shape[2]
                    if proj_method == "linear":
                        pooled = match_feature_dims(pooled, torch.zeros((pooled_batch, pooled_tokens, pooled_features), device=pooled.device), mode="linear")
                    elif proj_method == "bilinear":
                        pooled = match_feature_dims(pooled, torch.zeros((pooled_batch, pooled_tokens, pooled_features), device=pooled.device), mode="bilinear")
                    elif proj_method == "bicubic":
                        pooled = match_feature_dims(pooled, torch.zeros((pooled_batch, pooled_tokens, pooled_features), device=pooled.device), mode="bicubic")
                    elif proj_method == "trilinear":
                        pooled = match_feature_dims(pooled, torch.zeros((pooled_batch, pooled_tokens, pooled_features), device=pooled.device), mode="trilinear")
                else:
                    # create empty tensor with correct shape
                    pooled = torch.zeros((pooled_batch, pooled_tokens, pooled_features), device=combined.device)
                # 4) repackage
                info = info.copy()
                info["pooled_output"] = pooled
                out.append([combined, info])
        except Exception as e:
            logger.error(f"Error in ConditioningProjectMultiple: {e}")
            return (conditioning, )

        return (out,)

# 4. ConditioningSeparateMultiple
class ConditioningSeparateMultiple:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"conditioning": ("CONDITIONING", {})}}
    RETURN_TYPES = ("CONDITIONING", "CONDITIONING", "CONDITIONING", "CONDITIONING", "CONDITIONING", "CONDITIONING", "CONDITIONING")
    RETURN_NAMES = ("combined_list", "pooled_list", "individual_1", "individual_2", "individual_3", "individual_4", "individual_5")
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


"""
    WAS Conditioning Blend Node
    # complements of the WAS extras pack, ported for utility
"""

import torch
import math


#def normalize(latent, target_min=None, target_max=None):
#    """
#    Normalize a tensor `latent` between `target_min` and `target_max`.
#
#    Args:
#        latent (torch.Tensor): The input tensor to be normalized.
#        target_min (float, optional): The minimum value after normalization.
#            - When `None` min will be tensor min range value.
#        target_max (float, optional): The maximum value after normalization.
#            - When `None` max will be tensor max range value.
#
#    Returns:
#        torch.Tensor: The normalized tensor
#    """
#    min_val = latent.min()
#    max_val = latent.max()
#
#    if target_min is None:
#        target_min = min_val
#    if target_max is None:
#        target_max = max_val
#
#    normalized = (latent - min_val) / (max_val - min_val)
#    scaled = normalized * (target_max - target_min) + target_min
#    return scaled




class ConditioningBlenderSampler:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pos_conditionings": ("CONDITIONING", {}),
                "folding_mode": (list(blending_modes.keys()), {"default": "lerp", "tooltip": "Blending mode to use."}),
            }
        }

    RETURN_TYPES = ("CONDITIONING",)
    RETURN_NAMES = ("conditioning",)
    FUNCTION = "combine"
    CATEGORY = "conditioning"

    def combine(self,
                conditioning_a,
                blending_mode,
                blending_strength,
                seed,
                squash=False,
                amount_blended=-1,
                a_pool_strength=0.5,
                b_pool_strength=0.5,
                extrapolate_pooled=False,
                conditioning_b=[], device="cpu"):
        ...



class ConditioningSelector:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "conditioning": ("CONDITIONING", {}),
                "index": ("INT", {"default": 1, "min": 0, "step": 1}),
                "only_one": ("BOOLEAN", {"default": False, "tooltip": "If true, only return the selected conditioning."}),
            }
        }

    RETURN_TYPES = ("CONDITIONING", "CONDITIONING")
    RETURN_NAMES = ("before", "after")
    FUNCTION = "split"
    CATEGORY = "conditioning"

    def split(self, conditioning, index, only_one=False):
        uc = UsefulConditioning(conditioning).clone()

        # use built-in slicing logic (class handles validation)
        if only_one:
            return [uc[index]], None
        before = uc.slice(0, index)
        after = uc.slice(index)

        return before, after




import torch
import math
import torch.nn.functional as F

# Spherical linear interpolation (slerp), arc-preserving over cosine similarity
def true_slerp(a, b, t):
    dot = F.cosine_similarity(a, b, dim=-1).clamp(-0.9995, 0.9995)
    theta = torch.acos(dot)
    sin_theta = torch.sin(theta)
    s1 = torch.sin((1 - t) * theta) / sin_theta
    s2 = torch.sin(t * theta) / sin_theta
    return a * s1.unsqueeze(-1) + b * s2.unsqueeze(-1)

# Barycentric-style 5-point interpolation for rose/pentachoron blends
def pentachoron_blend(points, weights):
    return sum(w * p for w, p in zip(weights, points))

blending_modes = {
    'lerp': lambda a, b, t: a * (1 - t) + b * t,  # Linear interpolation
    'slerp': true_slerp,                         # Spherical interpolation over cosine arc
    'cosine': lambda a, b, t: (a + b - (a - b) * torch.cos(t * math.pi)) / 2,  # Cosine-eased LERP
    'cuberp': lambda a, b, t: a + (b - a) * (3 * t ** 2 - 2 * t ** 3),         # Smooth cubic interpolation
    'exclusion': lambda a, b, t: (a + b - 2 * a * b) * t,                      # Exclusive dissimilarity blend
    'inject': lambda a, b, t: a + b * t,                                       # Additive delta injection
    'random': lambda a, b, t: a + (torch.rand_like(b) * (b - a)) * t,          # Random perturbation toward b
    'pentachoron': pentachoron_blend,                                          # 5-point latent polytope blend
}

pooled_blending_modes = {
    'lerp': lambda a, b, t: a * (1 - t) + b * t,
    'slerp': true_slerp,
    'cosine': lambda a, b, t: a + (b - a) * (1 - torch.cos(t * math.pi)),
    'cuberp': lambda a, b, t: a + (b - a) * (3 * t ** 2 - 2 * t ** 3),
    'exclusion': lambda a, b, t: a + (b - a) * t - 2 * a * b * t,
    'inject': lambda a, b, t: a + b * t,
    'random': lambda a, b, t: a + (torch.rand_like(b) * (b - a)) * t,
}

class ABS_WAS_ConditioningBlend:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "conditioning_a": ("CONDITIONING",),
                "blending_mode": (list(blending_modes.keys()),),
                "blending_strength": ("FLOAT", {"default": 0.5, "min": -10.0, "max": 10.0, "step": 0.001}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "squash": ("BOOLEAN", {"default": False, "tooltip": "Average each input group before blending."}),
                "amount_blended": ("INT", {"default": -1, "min": -1, "max": 1000, "step": 1}),
                "a_pool_strength": ("FLOAT", {
                    "default": 0.5, "min": -10.0, "max": 10.0, "step": 0.01,
                    "tooltip": "How strongly to apply delta from pooled B into pooled A."
                }),
                "b_pool_strength": ("FLOAT", {
                    "default": 0.5, "min": -10.0, "max": 10.0, "step": 0.01,
                    "tooltip": "How strongly to apply delta from pooled B into pooled A."
                }),
                "extrapolate_pooled": ("BOOLEAN", {"default": False, "tooltip": "Extrapolate pooled outputs or avoid."}),
            },
            "optional": {
                "conditioning_b": ("CONDITIONING", {"default": []}),
                "device": (["cpu", "cuda", "mps"], {"default": "cpu", "tooltip": "Device to run the blending on."}),
            }
        }

    RETURN_TYPES = ("CONDITIONING",)
    RETURN_NAMES = ("conditioning",)
    FUNCTION = "combine"
    CATEGORY = "conditioning"
    DEPRECATED = True  # This node is deprecated, use the new ConditioningBlenderSampler instead.

    def combine(self,
                conditioning_a,
                blending_mode,
                blending_strength,
                seed,
                squash=False,
                amount_blended=-1,
                a_pool_strength=0.5,
                b_pool_strength=0.5,
                extrapolate_pooled=False,
                conditioning_b=[], device="cpu"):

        if seed > 0:
            torch.manual_seed(seed)

        if not conditioning_a and not conditioning_b:
            return ([],)
        if not conditioning_b:
            return (conditioning_a,)

        conditioning_a = UsefulConditioning(conditioning_a) if isinstance(conditioning_a, list) else conditioning_a
        conditioning_b = UsefulConditioning(conditioning_b) if isinstance(conditioning_b, list) else conditioning_b

        conditioning_a = conditioning_a.clone()
        conditioning_b = conditioning_b.clone()

        blend_weight = torch.tensor(blending_strength, device=device)
        t = blend_weight.item() if blend_weight.numel() == 1 else float(blend_weight)

        # --- SQUASHED BLEND ---
        if squash:
            a_avg = self.project_to_dominant_length([e[0] for e in conditioning_a], device)
            pa_avg = torch.stack([e[1]["pooled_output"] for e in conditioning_a if "pooled_output" in e[1]]) \
                .mean(dim=0) if extrapolate_pooled else None

            b_avg = self.project_to_dominant_length([e[0] for e in conditioning_b], device)
            pb_avg = torch.stack([e[1]["pooled_output"] for e in conditioning_b if "pooled_output" in e[1]]) \
                .mean(dim=0) if extrapolate_pooled else None

            a_proj, b_proj = self.align_pair_length(a_avg, b_avg, device)

            if blending_mode == "difference":
                delta = torch.abs(a_proj - b_proj)
                cond = a_proj + delta * blend_weight
            elif blending_mode == "difference_exclude":
                delta = torch.abs(a_proj - b_proj)
                cond = a_proj - delta * blend_weight
            elif blending_mode == "pentachoron":
                points = [a_proj, b_proj, a_proj.clone(), b_proj.clone(), (a_proj + b_proj) / 2]
                weights = torch.tensor([0.2] * 5, device=device)
                cond = F.normalize(pentachoron_blend(points, weights))
            else:
                blend_fn = blending_modes[blending_mode]
                cond = F.normalize(blend_fn(a_proj, b_proj, 1 - blend_weight))

            pooled = None
            if extrapolate_pooled and pa_avg is not None and pb_avg is not None:
                pooled_fn = pooled_blending_modes.get(blending_mode, lambda a, b, t: (a + b) / 2)
                pooled = pooled_fn(pa_avg, pb_avg, t)
                pooled = F.normalize(pooled, dim=-1)
            elif pa_avg is not None or pb_avg is not None:
                pooled = pa_avg if pa_avg is not None else pb_avg

            return ([[cond, {"pooled_output": pooled}]],)

        # --- PAIRWISE BLEND ---
        result = []
        num_blend = len(conditioning_a) if amount_blended == -1 else min(amount_blended, len(conditioning_a))

        if not conditioning_b:
            conditioning_b = [[a.clone(), meta] for a, meta in conditioning_a]

        for i in range(num_blend):
            a, meta_a = conditioning_a[i]
            pa = meta_a.get("pooled_output", None)

            for b, meta_b in conditioning_b:
                pb = meta_b.get("pooled_output", None)

                a, b = a.to(device), b.to(device)
                pa = pa.to(device).clone() if pa is not None else None
                pb = pb.to(device).clone() if pb is not None else None
                pa = (a_pool_strength * pa) if pa is not None else None
                pb = (b_pool_strength * pb) if pb is not None else None

                a_proj, b_proj = self.align_pair_length(a, b, device)

                if blending_mode == "difference":
                    delta = torch.abs(a_proj - b_proj)
                    cond = a_proj + delta * blend_weight
                elif blending_mode == "difference_exclude":
                    delta = torch.abs(a_proj - b_proj)
                    cond = a_proj - delta * blend_weight
                elif blending_mode == "pentachoron":
                    points = [a_proj, b_proj, a_proj.clone(), b_proj.clone(), (a_proj + b_proj) / 2]
                    weights = torch.tensor([0.75] * 5, device=device)
                    cond = F.normalize(pentachoron_blend(points, weights))
                else:
                    blend_fn = blending_modes[blending_mode]
                    cond = F.normalize(blend_fn(a_proj, b_proj, 1 - blend_weight))

                pooled = None
                if extrapolate_pooled and pa is not None and pb is not None:
                    pooled_fn = pooled_blending_modes.get(blending_mode, lambda a, b, t: (a + b) / 2)
                    pooled = pooled_fn(pa, pb, t)
                    pooled = F.normalize(pooled, dim=-1)
                elif pa is not None or pb is not None:
                    pooled = pa if pa is not None else pb

                result.append([cond, {"pooled_output": pooled}])

        return (result,)

    def project_to_dominant_length(self, tensors: list[torch.Tensor], device):
        lengths = [t.shape[1] for t in tensors]
        dominant_len = max(set(lengths), key=lengths.count)

        def resize(t):
            t = t.to(device)
            L = t.shape[1]
            if L == dominant_len:
                return t
            elif L > dominant_len:
                return t[:, :dominant_len]
            else:
                pad = torch.zeros((t.shape[0], dominant_len - L, t.shape[2]), device=t.device)
                return torch.cat([t, pad], dim=1)

        return torch.stack([resize(t) for t in tensors]).mean(dim=0)

    def align_pair_length(self, a: torch.Tensor, b: torch.Tensor, device):
        T = max(a.shape[1], b.shape[1])

        def pad_to(t):
            L = t.shape[1]
            if L == T:
                return t
            elif L > T:
                return t[:, :T]
            else:
                pad = torch.zeros((t.shape[0], T - L, t.shape[2]), device=t.device)
                return torch.cat([t, pad], dim=1)

        return pad_to(a.to(device)), pad_to(b.to(device))

# ----------------------------------------------------------------------
# Encoder Blender Sampler
# ----------------------------------------------------------------------




# ─────────────────────────────────────────────────────────────────────
# ComfyUI Node
# ─────────────────────────────────────────────────────────────────────
class RoseSimilarityConditioning:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "x_input":   ("CONDITIONING",),
                "need":      ("CONDITIONING",),
                "relation":  ("CONDITIONING",),
                "purpose":   ("CONDITIONING",),
                "normalize_result": ("BOOLEAN", {"default": True}),
                "pooled_output_source": (
                    ["x_input", "need", "relation", "purpose", "none"],
                    {"default": "x_input"}
                ),
            }
        }

    RETURN_TYPES = ("CONDITIONING",)
    RETURN_NAMES = ("rose_resonated",)
    FUNCTION = "resonate"
    CATEGORY = "conditioning/rose"

    def resonate(self, x_input, need, relation, purpose, normalize_result=True, pooled_output_source="x_input"):
        result = []

        # Determine pooled output source
        pooled = {
            "x_input":  x_input[0][1].get("pooled_output", None),
            "need":     need[0][1].get("pooled_output", None),
            "relation": relation[0][1].get("pooled_output", None),
            "purpose":  purpose[0][1].get("pooled_output", None),
            "none":     None
        }.get(pooled_output_source, None)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # move all to the correct device
        x_input = [[x.clone().to(device), meta] for x, meta in x_input]
        need = [[n.clone().to(device), meta] for n, meta in need]
        relation = [[r.clone().to(device), meta] for r, meta in relation]
        purpose = [[p.clone().to(device), meta] for p, meta in purpose]
        for i, (x, meta) in enumerate(x_input):
            n = need[i % len(need)][0]
            r = relation[i % len(relation)][0]
            p = purpose[i % len(purpose)][0]

            # Align lengths
            T = max(x.shape[1], n.shape[1], r.shape[1], p.shape[1])
            def align(t): return F.pad(t, (0, 0, 0, T - t.shape[1])) if t.shape[1] < T else t[:, :T]
            x, n, r, p = [align(t) for t in [x, n, r, p]]
            rose_vals = rose_score(x, n, r, p).unsqueeze(-1)

            modes = []
            modes.append(x * rose_vals)
            modes.append(x + rose_vals * (n - x))
            modes.append((1 - rose_vals) * x + rose_vals * n)
            modes.append(x + rose_vals * ((n + r) - p))
            modes.extend([n, r, p])

            averaged = torch.stack([F.normalize(m) for m in modes]).mean(dim=0)
            if normalize_result:
                averaged = F.normalize(averaged)

            result.append([averaged, {"pooled_output": pooled}])

        return (result,)


from ..utils.conditioning_helper import ConditioningHelper, UsefulConditioning


class TestNewCondTypeNode:
    """
        This node should convert the original cond into a more useful object for access.
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "conditioning": ("CONDITIONING", {}),
            }
        }

    RETURN_TYPES = ("CONDITIONING",)
    RETURN_NAMES = ("new_conditioning",)

    FUNCTION = "convert_conditioning"
    CATEGORY = "utils/conditioning"
    def convert_conditioning(self, conditioning):
        """
        Convert CONDITIONING to a more useful format.
        """
        out = []
        for combined, info in conditioning:
            # Convert to a more useful format
            new_combined = ConditioningHelper.convert_conditioning(combined)

        return (new_combined,)











