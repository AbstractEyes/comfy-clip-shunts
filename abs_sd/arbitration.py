# arbitration.py
# --------------------
# Bridge between ABS symbolic model manager and ComfyUI's VRAM arbitration

import torch
import logging
from comfy import model_management as comfy_mm

logger = logging.getLogger(__name__)

# --------------------
# Device Arbitration
# --------------------

def get_device(name_hint: str = "text_encoder") -> torch.device:
    """
    Resolves appropriate device for text encoders or other components.
    """
    if name_hint == "text_encoder":
        return comfy_mm.text_encoder_device()
    elif name_hint == "unet":
        return comfy_mm.get_torch_device()
    elif name_hint == "vae":
        return comfy_mm.vae_device()
    return comfy_mm.get_torch_device()


def get_offload_device(name_hint: str = "text_encoder") -> torch.device:
    if name_hint == "text_encoder":
        return comfy_mm.text_encoder_offload_device()
    elif name_hint == "vae":
        return comfy_mm.vae_offload_device()
    return torch.device("cpu")


# --------------------
# Dtype Arbitration
# --------------------

def get_dtype(name_hint: str = "text_encoder") -> torch.dtype:
    """Resolve proper dtype for a given model class."""
    if name_hint == "text_encoder":
        return comfy_mm.text_encoder_dtype()
    elif name_hint == "vae":
        return comfy_mm.vae_dtype()
    elif name_hint == "unet":
        return comfy_mm.unet_dtype()
    return torch.float32


def pick_compatible_dtype(candidate: torch.dtype, fallback: torch.dtype) -> torch.dtype:
    """
    Given a candidate and fallback dtype, apply ComfyUI compatibility check.
    """
    device = comfy_mm.get_torch_device()
    return comfy_mm.pick_weight_dtype(candidate, fallback, device=device)


# --------------------
# Memory Hooks
# --------------------

def load_model_comfy_runtime(model_patchable):
    """
    Use ComfyUI's smart GPU loader on patchable model (e.g. with .patcher).
    """
    if hasattr(model_patchable, "patcher") and model_patchable.patcher is not None:
        logger.info(f"[arbitration] Registering patcher model {model_patchable.__class__.__name__} with Comfy")
        comfy_mm.load_model_gpu(model_patchable.patcher)
        return True
    return False


def unload_all_comfy_models():
    comfy_mm.unload_all_models()


def get_max_vram_allocation(device: torch.device = None) -> int:
    return comfy_mm.maximum_vram_for_weights(device)


def soft_empty_cache():
    comfy_mm.soft_empty_cache()


def is_supported_dtype(device: torch.device, dtype: torch.dtype) -> bool:
    return comfy_mm.supports_dtype(device, dtype)


def get_free_memory(device: torch.device = None) -> int:
    return comfy_mm.get_free_memory(device)


# --------------------
# Inference Safety Check
# --------------------

def throw_if_interrupted():
    comfy_mm.throw_exception_if_processing_interrupted()
