# model_loader_registry.py

import torch
from typing import Callable, Dict
from .model_manager_wrapper import ClipPipelineConfig
from .abs_clip.multi_clip_registry import MultiClipEntry


# Central registry for model loading functions
MODEL_LOADER_REGISTRY: Dict[str, Callable[[ClipPipelineConfig], MultiClipEntry]] = {}


def register_model_loader(model_type: str):
    """
    Decorator to register a loader for a given model type.
    """
    def wrapper(fn):
        if model_type in MODEL_LOADER_REGISTRY:
            raise ValueError(f"Loader for model_type '{model_type}' already registered.")
        MODEL_LOADER_REGISTRY[model_type] = fn
        return fn
    return wrapper


def load_model_from_registry(config: ClipPipelineConfig) -> MultiClipEntry:
    loader = MODEL_LOADER_REGISTRY.get(config.clip_type)
    if not loader:
        raise ValueError(f"No loader registered for clip_type '{config.clip_type}'.")
    return loader(config)


# === EXAMPLE REGISTERED LOADERS ===

@register_model_loader("clip_l")
def load_clip_l(config: ClipPipelineConfig) -> MultiClipEntry:
    from transformers import CLIPTextModel
    model = CLIPTextModel.from_pretrained(config.model_path, torch_dtype=config.dtype)
    return MultiClipEntry(
        model=model,
        name="clip_l",
        clip_type="clip_l",
        dtype=config.dtype,
        tokenizer=None,
        source=config.source or "hf/clip_l"
    )


@register_model_loader("t5")
def load_t5(config: ClipPipelineConfig) -> MultiClipEntry:
    from transformers import T5EncoderModel
    model = T5EncoderModel.from_pretrained(config.model_path, torch_dtype=config.dtype)
    return MultiClipEntry(
        model=model,
        name="t5",
        clip_type="t5",
        dtype=config.dtype,
        tokenizer=None,
        source=config.source or "hf/t5"
    )
