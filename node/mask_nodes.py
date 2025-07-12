"""
    Clip-suite masking nodes for ComfyUI, designed to handle various masking operations on tensors, images, and conditioning data.
    Author: AbstractPhil

    Most of these are fail-friendly - meaning gracefully if the input is not as expected, rather than crashing the node.

    This module provides nodes for masking operations, including:
    - CreateEmptyMask:
            Generates an empty mask tensor with specified dimensions.
            Can be initialized with noise patterns, zeroes, a mask value, segmented images, or more.
            This has a dropdown for known mask types, such as:
                Latent, Image, Conditioning, Prompt, and more will be added in the future.


    - CreateMaskFromImage:
            Creates a mask tensor from an input image, pick your color.
            Has a heuristic detection system and is linked to SAM for auto image-mask extraction if available.
    - CreateMaskFromLatent:
            Generates a mask with specified size dimensions from a tensor.
            This can be used to create masks from tensors of various shapes in a streamlined and efficient manner.
    - CreateMaskFromConditioning:
            Generates a mask tensor from conditioning data.
            Can be used to create masks based on conditioning information, such as text prompts or other conditioning inputs.
            This is useful for generating masks that are conditioned on specific inputs.
    - CreateMaskFromPrompt:
            Generates a mask tensor based on a text prompt.
            This can be used to create masks that are influenced by specific text prompts, allowing for dynamic masking based on user input.

    - BlurMask:
            Applies a node-based blur to any input mask input.

    - BlurLatentWithMask:
            Blurs an image using a mask tensor instead of completely obliterating it, like the standard ComfyUI mask node.
            Uses similar logic to the ComfyUI mask node but allows for more nuanced control over the blurring process.

    - BlurImageWithMask:
            Blurs the masked region of an image using a mask tensor instead of completely destroying it.
            Allows specific region control and is useful for creating more nuanced image processing effects.

    - InterpolateMaskOverTime:
            Accepts a batched mask tensor and a time parameter.
            Configured by batch and can be used to interpolate a mask tensor over time.
            Interpolates a mask tensor over time, allowing for smooth transitions between different mask states.
            This is useful for creating dynamic masks that change over time, such as in animations or video processing.

"""

import math
import torch
import torch.nn.functional as F
import torchvision.transforms as T
import comfy
import logging

log = logging.getLogger(__name__)

# ─── Helpers ──────────────────────────────────────────────────────────── #

def _ensure_mask(tensor: torch.Tensor) -> torch.Tensor:
    """
    Ensure mask has shape [B,1,...] and dtype float32 on correct device.
    Supports 1D ([B,T]), 2D ([H,W]), 3D ([B,H,W]), 4D ([B,C,H,W]).
    """
    m = tensor.clone().float()
    # ensure batch dimension
    if m.dim() == 2:       # [B,T] → [B,1,T]
        m = m.unsqueeze(1)
    elif m.dim() == 3:     # [B,H,W] → [B,1,H,W]
        m = m.unsqueeze(1)
    elif m.dim() == 3 and tensor.size(0) != 1:
        # ambiguous, leave
        pass
    elif m.dim() == 2 and tensor.size(0) == 1:
        # [H,W] → [1,1,H,W]
        m = m.unsqueeze(0).unsqueeze(0)
    elif m.dim() == 4:
        # [B,C,H,W] → assume C=1 or drop C
        if m.size(1) != 1:
            m = m.mean(dim=1, keepdim=True)
    return m

def _clamp_mask(mask: torch.Tensor) -> torch.Tensor:
    return mask.clamp(0.0, 1.0)

# ─── Nodes ────────────────────────────────────────────────────────────── #

class CreateEmptyMask:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "shape": ("INT", {"multi": True}),
                "mask_value": ("FLOAT", {"default": 0.0}),
                "noise": ("BOOLEAN", {"default": False}),
                "noise_scale": ("FLOAT", {"default": 1.0}),
                "softness": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 10.0}),
            }
        }
    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("mask",)
    FUNCTION = "create_empty_mask"
    CATEGORY = "mask"

    def create_empty_mask(self, shape, mask_value, noise, noise_scale, softness):
        """
        Generates a mask of given shape, optionally with noise, raised to 'softness' power.
        """
        try:
            shp = tuple(shape)
            m = torch.full(shp, mask_value, dtype=torch.float32)
            if noise:
                m = m + torch.randn_like(m) * noise_scale
            m = _clamp_mask(m) ** softness
            return (_ensure_mask(m),)
        except Exception as e:
            log.warning(f"CreateEmptyMask failed: {e}")
            return (_ensure_mask(torch.zeros(tuple(shape), dtype=torch.float32)),)


class CreateMaskFromImage:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {}),
                "color": ("COLOR", {"default": [255,255,255]}),
                "threshold": ("FLOAT", {"default": 0.1}),
                "threshold_mode": (["absolute","percentile","top_k"], {"default":"absolute"}),
                "percentile": ("FLOAT", {"default":80.0, "min":0.0, "max":100.0}),
                "top_k": ("INT", {"default":10, "min":1}),
                "softness": ("FLOAT", {"default":1.0, "min":0.1, "max":10.0}),
            }
        }
    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("mask",)
    FUNCTION = "create_mask_from_image"
    CATEGORY = "mask"

    def create_mask_from_image(self, image, color, threshold, threshold_mode, percentile, top_k, softness):
        """
        Create a soft mask by color similarity.
        Modes:
         - absolute: diff < threshold
         - percentile: keep top X% most similar pixels
         - top_k: keep K most similar pixels
        """
        try:
            img = image.float()
            tgt = torch.tensor(color, device=img.device, dtype=img.dtype).view(-1,1,1)/255.0
            diff = (img - tgt).abs().mean(dim=0)  # [H,W]
            sim = 1 - diff / (diff.max()+1e-6)    # similarity [0,1]
            if threshold_mode == "absolute":
                mask = (sim >= threshold).float()
            elif threshold_mode == "percentile":
                th = torch.quantile(sim.flatten(), 1 - percentile/100)
                mask = (sim >= th).float()
            else:  # top_k
                flat = sim.flatten()
                k = min(top_k, flat.numel())
                val, idx = torch.topk(flat, k)
                mask = torch.zeros_like(flat)
                mask[idx] = 1.0
                mask = mask.view(sim.shape)
            mask = _clamp_mask(mask ** softness)
            return (_ensure_mask(mask),)
        except Exception as e:
            log.warning(f"CreateMaskFromImage failed: {e}")
            H,W = image.shape[-2:]
            return (_ensure_mask(torch.zeros((1,1,H,W), dtype=torch.float32)),)


class CreateMaskFromLatent:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "latent": ("LATENT", {}),
                "threshold_mode": (["none","percentile"],{"default":"none"}),
                "percentile":("FLOAT",{"default":50.0}),
                "softness":("FLOAT",{"default":1.0,"min":0.1,"max":10.0}),
            }
        }
    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("mask",)
    FUNCTION = "create_mask_from_latent"
    CATEGORY = "mask"

    def create_mask_from_latent(self, latent, threshold_mode, percentile, softness):
        """
        Binarize latent energy map or produce full mask.
         - none: full mask of ones
         - percentile: keep top X% channels’ mean energy
        """
        try:
            B,C,H,W = latent.shape
            energy = latent.abs().mean(dim=1)  # [B,H,W]
            if threshold_mode=="percentile":
                ths = torch.quantile(energy.flatten(1), 1-percentile/100, dim=1)
                mask = (energy >= ths.view(B,1,1)).float()
            else:
                mask = torch.ones_like(energy)
            mask = _clamp_mask(mask ** softness)
            return (_ensure_mask(mask),)
        except Exception as e:
            log.warning(f"CreateMaskFromLatent failed: {e}")
            return (_ensure_mask(torch.zeros((1,1,1,1), dtype=torch.float32)),)


class CreateMaskFromConditioning:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "conditioning": ("CONDITIONING", {}),
                "threshold_mode": (["absolute","percentile"],{"default":"absolute"}),
                "threshold":("FLOAT",{"default":0.0}),
                "percentile":("FLOAT",{"default":80.0}),
                "softness":("FLOAT",{"default":1.0,"min":0.1,"max":10.0}),
            }
        }
    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("mask",)
    FUNCTION = "create_mask_from_conditioning"
    CATEGORY = "mask"

    def create_mask_from_conditioning(self, conditioning, threshold_mode, threshold, percentile, softness):
        """
        Build mask from aggregated conditioning scores over tokens.
        """
        try:
            # stack cond.mean over D → [N,B,T]
            scores = torch.stack([c[0].mean(dim=-1) for c in conditioning], dim=0).sum(dim=0)  # [B,T]
            if threshold_mode=="absolute":
                mask = (scores >= threshold).float()
            else:
                th = torch.quantile(scores, 1-percentile/100, dim=1, keepdim=True)
                mask = (scores >= th).float()
            mask = _clamp_mask(mask ** softness)
            return (_ensure_mask(mask),)
        except Exception as e:
            log.warning(f"CreateMaskFromConditioning failed: {e}")
            return (_ensure_mask(torch.zeros((1,1,1), dtype=torch.float32)),)


class CreateMaskFromPrompt:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING",{"default":""}),
                "mode":(["uniform","triangular"],{"default":"triangular"}),
                "softness":("FLOAT",{"default":1.0,"min":0.1,"max":10.0}),
            }
        }
    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("mask",)
    FUNCTION = "create_mask_from_prompt"
    CATEGORY = "mask"

    def create_mask_from_prompt(self, prompt, mode, softness):
        """
        Build per-token mask from prompt length.
        """
        try:
            tokens = prompt.split()
            T = max(len(tokens),1)
            if mode=="uniform":
                m = torch.ones((1,T))
            else:
                x = torch.linspace(0,1,T)
                m = 1 - (x-0.5).abs()*2
                m = m.unsqueeze(0)
            m = _clamp_mask(m ** softness)
            return (_ensure_mask(m),)
        except Exception as e:
            log.warning(f"CreateMaskFromPrompt failed: {e}")
            return (_ensure_mask(torch.ones((1,1), dtype=torch.float32)),)


class BlurMask:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mask": ("MASK",{}),
                "kernel_size":("INT",{"default":3,"min":1}),
                "gaussian":("BOOLEAN",{"default":False}),
                "sigma":("FLOAT",{"default":1.0}),
            }
        }
    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("mask",)
    FUNCTION = "blur_mask"
    CATEGORY = "mask"

    def blur_mask(self, mask, kernel_size, gaussian, sigma):
        """
        Blur a mask with average or Gaussian. Input mask → [B,1,...].
        """
        try:
            m = _ensure_mask(mask)
            if gaussian:
                gb = T.GaussianBlur(kernel_size, sigma=(sigma,sigma))
                # flatten spatial dims into 2D image if needed
                shape = m.shape
                B,C,*dims = shape
                m_flat = m.view(B,C, *dims)
                blurred = gb(m_flat)
            else:
                dims = m.dim()-2
                if dims==1:
                    blurred = F.avg_pool1d(m, kernel_size, stride=1, padding=kernel_size//2)
                else:
                    blurred = F.avg_pool2d(m, kernel_size, stride=1, padding=kernel_size//2)
            return (_clamp_mask(blurred),)
        except Exception as e:
            log.warning(f"BlurMask failed: {e}")
            return (_ensure_mask(mask),)


class ErodeDilateMask:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mask":("MASK",{}),
                "operation":(["erode","dilate"],{"default":"erode"}),
                "kernel_size":("INT",{"default":3,"min":1}),
            }
        }
    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("mask",)
    FUNCTION = "erode_dilate_mask"
    CATEGORY = "mask"

    def erode_dilate_mask(self, mask, operation, kernel_size):
        """
        1D or 2D morphological erosion/dilation.
        """
        try:
            m = _ensure_mask(mask)
            if m.dim()==3:  # [B,1,T]
                if operation=="erode":
                    out = 1 - F.max_pool1d(1-m, kernel_size, stride=1, padding=kernel_size//2)
                else:
                    out = F.max_pool1d(m, kernel_size, stride=1, padding=kernel_size//2)
            else:  # [B,1,H,W]
                if operation=="erode":
                    out = 1 - F.max_pool2d(1-m, kernel_size, stride=1, padding=kernel_size//2)
                else:
                    out = F.max_pool2d(m, kernel_size, stride=1, padding=kernel_size//2)
            return (_clamp_mask(out),)
        except Exception as e:
            log.warning(f"ErodeDilateMask failed: {e}")
            return (_ensure_mask(mask),)


class InterpolateMaskOverTime:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mask_or_list":("MASK_LIST",{}),  # accepts single mask or list
                "steps":("INT",{"default":10,"min":1}),
                "mode":(["linear","cosine"],{"default":"linear"}),
            }
        }
    RETURN_TYPES = ("MASK_LIST",)
    RETURN_NAMES = ("sequence",)
    FUNCTION = "interpolate_mask_over_time"
    CATEGORY = "mask"

    def interpolate_mask_over_time(self, mask_or_list, steps, mode):
        """
        Interpolate between multiple mask states or ramp a single mask.
        mask_or_list: tensor [B,1,...] or list thereof
        Returns list of masks length=steps.
        """
        try:
            if isinstance(mask_or_list, list):
                # stack and linear interpolate pairwise across list
                states = [ _ensure_mask(m) for m in mask_or_list ]
                N = len(states)
                seq = []
                for i in range(N-1):
                    A, Bm = states[i], states[i+1]
                    for t in torch.linspace(0,1,steps):
                        seq.append(_clamp_mask((1-t)*A + t*Bm))
                return (seq,)
            else:
                m = _ensure_mask(mask_or_list)
                if mode=="cosine":
                    t = torch.linspace(0, math.pi/2, steps)
                    w = torch.sin(t)
                else:
                    w = torch.linspace(0,1,steps)
                shape = [steps] + [1]*m.dim()
                seq = [(w[i].view(shape[1:]) * m) for i in range(steps)]
                return (seq,)
        except Exception as e:
            log.warning(f"InterpolateMaskOverTime failed: {e}")
            # fallback flat replication
            m = _ensure_mask(mask_or_list if not isinstance(mask_or_list,list) else mask_or_list[0])
            return ([m.clone() for _ in range(steps)],)
