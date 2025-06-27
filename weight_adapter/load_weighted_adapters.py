import json
import logging
from safetensors.torch import load_file
import comfy.lora
import comfy.lora_convert

from . import adapters

from .load_lora import load_lora, model_lora_keys_unet, model_lora_keys_clip

logger = logging.getLogger(__name__)

def resolve_weight_adapter(x, lora, alpha, dora_scale, network_args=None):
    """
    Dispatch the correct adapter class from registered types.
    Uses ss_network_args['algo'] to prioritize matching if needed.
    """
    algo = None
    if network_args and isinstance(network_args, dict):
        algo = network_args.get("algo", "").lower()

    for adapter_cls in adapters:
        # Filter by known name if algo is specified
        if algo and adapter_cls.name.lower() != algo:
            continue
        adapter = adapter_cls.load(x, lora, alpha, dora_scale)
        if adapter is not None:
            return adapter
    return None


def load_adapters_for_models(model, clip, lora_path, strength_model=1.0, strength_clip=1.0):
    """
    Drop-in replacement for load_lora_for_models but uses enhanced adapters.
    """
    lora = load_file(lora_path)
    logger.info(f"[load_adapters] Loaded LoRA from {lora_path} with {len(lora.keys())} keys")

    # Try to extract network_args
    """
        # this is in lora.metadata, or it should be
        ss_network_args	{"preset": "full", "conv_dim": "1", "conv_alpha": "1", "use_tucker": "False", "rank_dropout": "0", "bypass_mode": "False", "dora_wd": "False", "module_dropout": "0", "use_scalar": "False", "rank_dropout_scale": "False", "algo": "locon", "train_norm": "False"}
    """
    network_args = None
    if hasattr(lora, 'metadata') and isinstance(lora.metadata, dict):
        algo = lora.metadata.get("ss_network_args", "")
        if isinstance(algo, str) and algo.strip().startswith("{"):
            try:
                network_args = json.loads(algo)
                logger.info(f"[load_adapters] Network args: {network_args}")
            except json.JSONDecodeError as e:
                logger.warning(f"[load_adapters] Failed to parse ss_network_args: {e}")

    # Build key map from model and clip
    key_map = {}
    if model is not None:
        key_map = model_lora_keys_unet(model.model, key_map)
    if clip is not None:
        key_map = model_lora_keys_clip(clip.cond_stage_model, key_map)

    patches = {}
    loaded_keys = set()

    for lora_key in lora.keys():
        lora_key_base = lora_key.split(".")[0]
        if lora_key_base in loaded_keys:
            continue

        model_key = key_map.get(lora_key_base)
        if model_key is None:
            continue

        adapter = resolve_weight_adapter(lora_key_base, lora, alpha=1.0, dora_scale=None, network_args=network_args)
        if adapter is not None:
            patches[model_key] = adapter
            loaded_keys.update(adapter.loaded_keys)

    if model is not None:
        model = model.clone()
        k_model = model.add_patches(patches, strength_model)
    else:
        k_model = ()

    if clip is not None:
        clip = clip.clone()
        k_clip = clip.add_patches(patches, strength_clip)
    else:
        k_clip = ()

    # Warn about unmatched adapter modules
    patched = set(k_model) | set(k_clip)
    for key in patches:
        if key not in patched:
            logger.warning(f"[load_adapters] Adapter key not applied: {key}")

    return model, clip

