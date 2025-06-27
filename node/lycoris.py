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
                "strength_model": ("FLOAT", {"default": 1.0, "min": -20.0, "max": 20.0, "step": 0.01}),
                "strength_clip": ("FLOAT", {"default": 1.0, "min": -20.0, "max": 20.0, "step": 0.01}),
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
            # Load the lora file
            lora_data = comfy.utils.load_torch_file(lora_path, safe_load=True)

            # Load metadata from safetensors BEFORE convert_lora
            metadata = {}
            if lora_path.endswith('.safetensors'):
                import safetensors.torch
                with safetensors.torch.safe_open(lora_path, framework="pt") as f:
                    metadata = f.metadata()
                logger.info(f"Loaded metadata: {list(metadata.keys())[:10]}")
                if 'ss_network_args' in metadata:
                    logger.info(f"Network args: {metadata['ss_network_args']}")

            # Create wrapper to preserve metadata through convert_lora
            class LoraDataWithMetadata(dict):
                def __init__(self, data, metadata):
                    super().__init__(data)
                    self.metadata = metadata

            lora_data = LoraDataWithMetadata(lora_data, metadata)

            # Convert if needed - this preserves our metadata attribute
            lora_data = comfy.lora_convert.convert_lora(lora_data)

            # Check raw loaded file
            raw_data = comfy.utils.load_torch_file(lora_path, safe_load=True)
            logger.info(f"Raw data type: {type(raw_data)}")
            if hasattr(raw_data, 'metadata'):
                logger.info("Raw data HAS metadata attribute!")

            # Build key map
            key_map = {}
            if model is not None and strength_model != 0:
                key_map = model_lora_keys_unet(model.model, key_map)
            if clip is not None and strength_clip != 0:
                key_map = model_lora_keys_clip(clip.cond_stage_model, key_map)

            logger.info(f"Built key map with {len(key_map)} entries")

            # Load patches using custom load_lora
            # This now returns pre-calculated patches in ComfyUI format
            loaded = load_lora(lora_data, key_map)

            # Log patch details
            logger.info(f"Loaded {len(loaded)} patches")
            patch_types = {}
            for k, v in loaded.items():
                patch_type = v[0] if isinstance(v, tuple) else "unknown"
                patch_types[patch_type] = patch_types.get(patch_type, 0) + 1
            logger.info(f"Patch types: {patch_types}")

            # Apply patches to models
            new_model = model
            new_clip = clip

            if strength_model != 0 and model is not None:
                new_model = model.clone()
                k_model = new_model.add_patches(loaded, strength_model)
                logger.info(f"Applied {len(k_model)} patches to model")
            if strength_clip != 0 and clip is not None:
                new_clip = clip.clone()
                k_clip = new_clip.add_patches(loaded, strength_clip)
                logger.info(f"Applied {len(k_clip)} patches to CLIP")
            # Log any unloaded keys
            all_patched = set()
            if strength_model != 0 and 'k_model' in locals():
                all_patched.update(k_model)
            if strength_clip != 0 and 'k_clip' in locals():
                all_patched.update(k_clip)

            for k in loaded:
                if k not in all_patched:
                    logger.warning(f"Patch not applied: {k}")

            logger.info(f"Successfully loaded LyCORIS: {lora_name}")
            return (new_model, new_clip)

        except Exception as e:
            logger.error(f"Failed to load LyCORIS {lora_path}: {e}", exc_info=True)
            return (model, clip)