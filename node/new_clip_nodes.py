from dataclasses import dataclass
from typing import Optional

import torch
import logging
import comfy

from comfy.sd import CLIP
from ..utils.clip_converter import translate_comfy_clip_to_multiclip, reconstruct_comfy_clip_from_multiclip
from ..model.model_manager import get_model_manager
from ..abs_sd.CLIP import load_clip, CLIPType

import folder_paths


logger = logging.getLogger(__name__)

model_manager = get_model_manager() # singleton instance of the model manager
"""
class CLIPType(Enum):
    NOVELAI_V2 = 100
    STABLE_DIFFUSION = 1
    STABLE_CASCADE = 2
    SD3 = 3
    STABLE_AUDIO = 4
    HUNYUAN_DIT = 5
    FLUX = 6
    MOCHI = 7
    LTXV = 8
    HUNYUAN_VIDEO = 9
    PIXART = 10
    COSMOS = 11
    LUMINA2 = 12
    WAN = 13
    HIDREAM = 14
    CHROMA = 15
    ACE = 16
    OMNIGEN2 = 17
"""


class TextEncoderType:
    """
    Enum for different text encoder types.
    This is used to specify the type of text encoder in the pipeline configuration.
    """
    CLIP_L = "clip_l"
    CLIP_G = "clip_g"
    CLIP_H = "clip_h"
    T5 = "t5"
    LLM = "llm"


SUPPORTED_CLIP_TYPES = [TextEncoderType.__dict__.values() if isinstance(v, str) else v for v in TextEncoderType.__dict__.values() if not v.startswith("__")]
logging.info(f"AllEncoders: {SUPPORTED_CLIP_TYPES}")


@dataclass
class ClipPipelineConfig:
    clip_type: TextEncoderType
    model_path: str
    dtype: torch.dtype
    tokenizer: Optional[str] = None
    source: Optional[str] = None


class ClipTarget:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip_type": (SUPPORTED_CLIP_TYPES, {"default": TextEncoderType.CLIP_L}),
                "model_path": ("STRING", {"default": "", "multiline": False}),
                "dtype": (["bfloat16", "float16", "float32", "float64"], {"default": "float16"}),
                "source": ("STRING", {"default": "", "multiline": False}),
            },
        }

    RETURN_TYPES = ("CLIP_PIPELINE", )
    RETURN_NAMES = ("unloaded_pipeline", )
    FUNCTION = "create_clip_target"
    CATEGORY = "utils/clip"

    def create_clip_target(self, clip_type: TextEncoderType, model_path: str,
                           dtype: str = "float16", source: str = "") -> tuple:
        dtype = getattr(torch, dtype)
        config = ClipPipelineConfig(
            clip_type=clip_type,
            model_path=model_path,
            dtype=dtype,
            source=source
        )
        return (config, )


from ..abs_sd.multi_clip_registry import MultiClipRegistry
from ..abs_sd.CLIP import load_clip, CLIPType
from ..model.model_manager import get_model_manager
from ..utils.clip_converter import translate_comfy_clip_to_multiclip
import torch
import folder_paths


from typing import Optional, Dict, Any
import torch
import folder_paths
from ..abs_sd.CLIP import load_clip, CLIPType
from ..abs_sd.model_manager_wrapper import get_extended_model_manager
from ..abs_sd.multi_clip_registry import MultiClipRegistry


from ..abs_sd.CLIP import load_clip, CLIPType
from ..abs_sd.multi_clip_registry import MultiClipRegistry
#from ..abs_sd.model_manager_wrapper import resolve_clip_allocation
from ..utils.clip_converter import translate_comfy_clip_to_multiclip

import folder_paths


#lass ABS_ModelSelector:
#
#   @classmethod
#   def INPUT_TYPES(cls):
#       return {
#           "required": {
#               "clip_declaration": ("CLIP_PIPELINE",),
#               "attach_tokenizer": (["yes", "no"], {"default": "yes"}),
#               "patch_mode": (["min_vram", "max_speed", "custom"], {"default": "min_vram"}),
#           }
#       }

#   RETURN_TYPES = ("CLIP", "CLIP_PIPELINE", "CLIP_ROUTER", "MULTICLIP_REGISTRY", "DICT")
#   RETURN_NAMES = ("clip", "pipeline", "clip_router", "multi_clip_dict", "model_options")
#   FUNCTION = "build_clip_model"
#   CATEGORY = "ABS/CLIP"

#   def build_clip_model(self, clip_declaration, attach_tokenizer, patch_mode):
#       config = clip_declaration

#       # Resolve target hardware assignment using our wrapper logic
#       alloc = resolve_clip_allocation(
#           model_path=config.model_path,
#           dtype=config.dtype,
#           patch_mode=patch_mode
#       )

#       # Resolve comfy CLIPType enum from our lowercase descriptor
#       clip_type_enum = getattr(CLIPType, config.clip_type.upper(), CLIPType.STABLE_DIFFUSION)

#       # Load and patch the clip using proper options
#       clip = load_clip(
#           ckpt_paths=[config.model_path],
#           embedding_directory=folder_paths.get_folder_paths("embeddings"),
#           clip_type=clip_type_enum,
#           model_options=alloc.model_options
#       )

#       # Generate our router + registry
#       entries_raw = MultiClipRegistry.extract_from_comfy(clip)
#       registry = MultiClipRegistry()
#       for entry in entries_raw.values():
#           registry.add_entry(entry)

#       router = registry.to_cliprouter_dict()

#       # Compose pipeline metadata object
#       pipeline = {
#           "registry": registry,
#           "router": router,
#           "source_model": clip,
#           "model_type": config.clip_type,
#           "tokenizer_attached": attach_tokenizer == "yes"
#       }

#       return (clip, pipeline, router, registry.entries, alloc.model_options)
