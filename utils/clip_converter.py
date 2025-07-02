import torch
import types
import comfy.sd
import importlib
import logging

logger = logging.getLogger(__name__)



def translate_comfy_clip_to_multiclip(comfy_clip):
    """
    Converts a native ComfyUI CLIP instance into a MultiClip-compatible dictionary structure.

    Arguments:
    ----------
    - comfy_clip (ComfyUI.CLIP) : the standard loaded comfy clip model

    Returns:
    --------
    - dict: { sub_clip_name (str) : {full internal dict structure} }
    """

    multi_clip_dict = {}

    # Extract the core model
    cond_stage_model = comfy_clip.cond_stage_model
    tokenizer = getattr(comfy_clip, "tokenizer", None)
    patcher = getattr(comfy_clip, "patcher", None)
    device_info = {}

    # Check device mappings if patcher exists
    if patcher:
        device_info = {
            "load_device": patcher.load_device,
            "offload_device": patcher.offload_device,
        }

    # === SECTION 1: Identify Submodels ===
    for name in dir(cond_stage_model):
        logger.info(f"Checking attribute: {name}")
        attr = getattr(cond_stage_model, name)
        if isinstance(attr, torch.nn.Module):
            # Skip any obvious non-clip modules
            if name.startswith("_") or isinstance(attr, (torch.nn.Parameter,)):
                continue
            # Filter only clip components based on known keywords
            if any(key in name.lower() for key in ["clip", "t5", "vision", "llm"]):
                multi_clip_dict[name] = {
                    "model": attr,
                    "name": name,
                    "tokenizer": tokenizer,
                    "patcher": patcher,
                    "device_info": device_info,
                    "metadata": {
                        "source": "extracted",
                        "original_comfy_clip": type(comfy_clip).__name__,
                    },
                    "config": {
                        "has_tokenizer": tokenizer is not None,
                        "expected_dtype": str(attr.parameters().__next__().dtype) if any(attr.parameters()) else "unknown",
                    }
                }

    # === SECTION 2: Fallback - If no split, treat full model as "base" ===
    if not multi_clip_dict:
        multi_clip_dict["base_clip"] = {
            "model": cond_stage_model,
            "name": "base_clip",
            "tokenizer": tokenizer,
            "patcher": patcher,
            "device_info": device_info,
            "metadata": {
                "source": "fallback",
                "original_comfy_clip": type(comfy_clip).__name__,
            },
            "config": {
                "has_tokenizer": tokenizer is not None,
                "expected_dtype": str(cond_stage_model.parameters().__next__().dtype) if any(cond_stage_model.parameters()) else "unknown",
            }
        }

    return multi_clip_dict

def reconstruct_comfy_clip_from_multiclip(multi_clip_dict, base_class_path="guess", constructor_type="class"):
    """
    Reconstructs a ComfyUI-compatible CLIP object from the internal MultiClip dictionary.

    Arguments:
    ----------
    - multi_clip_dict (dict): structure of submodels extracted
    - base_class_path (str): "package.module.ClassName" to instantiate original comfy clip class
    - constructor_type (str): "class" or "function" for how to instantiate

    Returns:
    --------
    - comfy_clip (CLIP) : full ready-to-use ComfyUI CLIP object
    """

    # === SECTION 1: Dynamic Import ===
    module_path, class_name = base_class_path.rsplit(".", 1)
    imported_module = importlib.import_module(module_path)
    clip_class = getattr(imported_module, class_name)

    # Instantiate
    if constructor_type == "class":
        cond_stage_model = clip_class()
    elif constructor_type == "function":
        cond_stage_model = clip_class()
    else:
        raise ValueError(f"Unsupported constructor_type '{constructor_type}'")

    # === SECTION 2: Reattach submodules ===
    for name, data in multi_clip_dict.items():
        submodel = data.get("model", None)
        if submodel is not None:
            setattr(cond_stage_model, name, submodel)

    # === SECTION 3: Create wrapper comfy.CLIP object ===
    device_info = next(iter(multi_clip_dict.values())).get("device_info", {})
    load_device = device_info.get("load_device", None)
    offload_device = device_info.get("offload_device", None)

    from comfy.model_patcher import ModelPatcher
    comfy_clip = comfy.sd.CLIP(no_init=True)
    comfy_clip.cond_stage_model = cond_stage_model
    comfy_clip.tokenizer = next(iter(multi_clip_dict.values())).get("tokenizer", None)
    comfy_clip.patcher = ModelPatcher(cond_stage_model, load_device=load_device, offload_device=offload_device)
    comfy_clip.patcher.is_clip = True

    return comfy_clip