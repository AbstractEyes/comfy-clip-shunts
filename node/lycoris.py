import torch
import os
import logging

import folder_paths
import comfy.utils
import comfy.lora_convert
import comfy.model_patcher

# Import custom implementations
from ..weight_adapter.load_lora import load_lora, model_lora_keys_unet, model_lora_keys_clip

logger = logging.getLogger(__name__)


def load_lora_with_metadata(lora_path):
    """
    Load a LoRA/LyCORIS file and preserve its metadata through conversion.
    Returns:
        tuple: (converted_lora_dict, metadata_dict)
    """
    import json
    import safetensors.torch

    # Load the file
    lora_data = comfy.utils.load_torch_file(lora_path, safe_load=True)
    metadata = {}

    # Extract metadata from safetensors
    if lora_path.endswith('.safetensors'):
        with safetensors.torch.safe_open(lora_path, framework="pt") as f:
            metadata = dict(f.metadata())

    # Parse network args if present
    network_args = None
    if 'ss_network_args' in metadata:
        try:
            network_args = json.loads(metadata['ss_network_args'])
            metadata['network_args_parsed'] = network_args
            logger.info(f"Detected algorithm: {network_args.get('algo', 'unknown')}")
        except json.JSONDecodeError:
            logger.warning("Failed to parse ss_network_args")

    # Convert lora format
    converted = comfy.lora_convert.convert_lora(lora_data)

    # Preserve metadata by attaching it to the converted dict
    class LoraWithMetadata(dict):
        def __init__(self, data, metadata):
            super().__init__(data)
            self.metadata = metadata

    return LoraWithMetadata(converted, metadata), metadata


class LycorisLoaderNode:
    """Load LyCORIS adapters (LoHA, LoKr, LoCon, etc.) into ComfyUI models"""

    CATEGORY = "loaders"
    RETURN_TYPES = ("MODEL", "CLIP")
    RETURN_NAMES = ("model", "clip")
    FUNCTION = "load_lycoris"
    OUTPUT_NODE = False

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "clip": ("CLIP",),
                "lora_name": (folder_paths.get_filename_list("loras"),),
                "strength_model": ("FLOAT", {"default": 1.0, "min": -100.0, "max": 100.0, "step": 0.01}),
                "strength_clip": ("FLOAT", {"default": 1.0, "min": -100.0, "max": 100.0, "step": 0.01}),
            }
        }

    def load_lycoris(self, model, clip, lora_name, strength_model, strength_clip):
        if strength_model == 0 and strength_clip == 0:
            return (model, clip)

        lora_path = folder_paths.get_full_path("loras", lora_name)
        if not os.path.exists(lora_path):
            logger.error(f"LoRA file not found: {lora_path}")
            return (model, clip)

        try:
            # Use our new loader function
            lora_data, metadata = load_lora_with_metadata(lora_path)
            # Build key map
            key_map = {}
            if model is not None and strength_model != 0:
                key_map = model_lora_keys_unet(model.model, key_map)
            if clip is not None and strength_clip != 0:
                key_map = model_lora_keys_clip(clip.cond_stage_model, key_map)

            # Load patches - lora_data now has metadata attached
            loaded = load_lora(lora_data, key_map)

            # Apply patches to models
            new_model = model
            new_clip = clip

            if strength_model != 0 and model is not None:
                new_model = model.clone()
                new_model.add_patches(loaded, strength_model)
            if strength_clip != 0 and clip is not None:
                new_clip = clip.clone()
                new_clip.add_patches(loaded, strength_clip)

            logger.info(f"Successfully loaded LyCORIS: {lora_name}")
            return (new_model, new_clip)

        except Exception as e:
            logger.error(f"Failed to load LyCORIS {lora_path}: {e}", exc_info=True)
            return (model, clip)