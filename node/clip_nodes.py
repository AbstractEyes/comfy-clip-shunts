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


class ACLIPLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "clip_name": (folder_paths.get_filename_list("text_encoders"), ),
                              "type": (["stable_diffusion", "stable_cascade", "sd3", "stable_audio", "mochi", "ltxv", "pixart", "cosmos", "lumina2", "wan", "hidream", "chroma", "ace", "omnigen2"], ),
                              },
                "optional": {
                              "device": (["default", "cpu"], {"advanced": True}),
                             }}
    RETURN_TYPES = ("CLIP",)
    FUNCTION = "load_clip_internal"

    CATEGORY = "advanced/loaders"

    DESCRIPTION = "[Recipes]\n\nstable_diffusion: clip-l\nstable_cascade: clip-g\nsd3: t5 xxl/ clip-g / clip-l\nstable_audio: t5 base\nmochi: t5 xxl\ncosmos: old t5 xxl\nlumina2: gemma 2 2B\nwan: umt5 xxl\n hidream: llama-3.1 (Recommend) or t5\nomnigen2: qwen vl 2.5 3B"

    def load_clip_internal(self, clip_name, type="stable_diffusion", device="default"):
        clip_type = getattr(CLIPType, type.upper(), CLIPType.STABLE_DIFFUSION)

        model_options = {}
        if device == "cpu":
            model_options["load_device"] = model_options["offload_device"] = torch.device("cpu")

        clip_path = folder_paths.get_full_path_or_raise("text_encoders", clip_name)
        clip = load_clip(ckpt_paths=[clip_path], embedding_directory=folder_paths.get_folder_paths("embeddings"), clip_type=clip_type, model_options=model_options)
        return (clip,)

class ADualCLIPLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "clip_name1": (folder_paths.get_filename_list("text_encoders"), ),
                              "clip_name2": (folder_paths.get_filename_list("text_encoders"), ),
                              "type": (["sdxl", "sd3", "flux", "hunyuan_video", "hidream"], ),
                              },
                "optional": {
                              "device": (["default", "cpu"], {"advanced": True}),
                             }}
    RETURN_TYPES = ("CLIP",)
    FUNCTION = "load_clip_internal"

    CATEGORY = "advanced/loaders"

    DESCRIPTION = "[Recipes]\n\nsdxl: clip-l, clip-g\nsd3: clip-l, clip-g / clip-l, t5 / clip-g, t5\nflux: clip-l, t5\nhidream: at least one of t5 or llama, recommended t5 and llama"

    def load_clip_internal(self, clip_name1, clip_name2, type, device="default"):
        clip_type = getattr(CLIPType, type.upper(), CLIPType.STABLE_DIFFUSION)

        clip_path1 = folder_paths.get_full_path_or_raise("text_encoders", clip_name1)
        clip_path2 = folder_paths.get_full_path_or_raise("text_encoders", clip_name2)

        model_options = {}
        if device == "cpu":
            model_options["load_device"] = model_options["offload_device"] = torch.device("cpu")

        clip = load_clip(ckpt_paths=[clip_path1, clip_path2], embedding_directory=folder_paths.get_folder_paths("embeddings"), clip_type=clip_type, model_options=model_options)
        return (clip,)


class ATripleCLIPLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                    "clip_name1": (folder_paths.get_filename_list("text_encoders"), ),
                    "clip_name2": (folder_paths.get_filename_list("text_encoders"), ),
                    "clip_name3": (folder_paths.get_filename_list("text_encoders"), )
                }}
    RETURN_TYPES = ("CLIP",)
    FUNCTION = "load_clip_internal"

    CATEGORY = "advanced/loaders"

    DESCRIPTION = "[Recipes]\n\nsd3: clip-l, clip-g, t5"

    def load_clip_internal(self, clip_name1, clip_name2, clip_name3):
        clip_path1 = folder_paths.get_full_path_or_raise("text_encoders", clip_name1)
        clip_path2 = folder_paths.get_full_path_or_raise("text_encoders", clip_name2)
        clip_path3 = folder_paths.get_full_path_or_raise("text_encoders", clip_name3)
        clip = load_clip(ckpt_paths=[clip_path1, clip_path2, clip_path3], embedding_directory=folder_paths.get_folder_paths("embeddings"))
        return (clip,)


class AQuadrupleCLIPLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                              "clip_name1": (folder_paths.get_filename_list("text_encoders"), ),
                              "clip_name2": (folder_paths.get_filename_list("text_encoders"), ),
                              "clip_name3": (folder_paths.get_filename_list("text_encoders"), ),
                              "clip_name4": (folder_paths.get_filename_list("text_encoders"), )
                            }}
    RETURN_TYPES = ("CLIP",)
    FUNCTION = "load_clip_internal"

    CATEGORY = "advanced/loaders"

    DESCRIPTION = "[Recipes]\n\nhidream: long clip-l, long clip-g, t5xxl, llama_8b_3.1_instruct"

    def load_clip_internal(self, clip_name1, clip_name2, clip_name3, clip_name4):
        clip_path1 = folder_paths.get_full_path_or_raise("text_encoders", clip_name1)
        clip_path2 = folder_paths.get_full_path_or_raise("text_encoders", clip_name2)
        clip_path3 = folder_paths.get_full_path_or_raise("text_encoders", clip_name3)
        clip_path4 = folder_paths.get_full_path_or_raise("text_encoders", clip_name4)
        clip = load_clip(ckpt_paths=[clip_path1, clip_path2, clip_path3, clip_path4], embedding_directory=folder_paths.get_folder_paths("embeddings"))
        return (clip,)



# This is a placeholder for the EncodeConditioning class.
class EncoderEncodeConditioning:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip": ("CLIP",),
                "clip_pipeline": ("CLIP_PIPELINE", {"default": None}),
            }
        }
    RETURN_TYPES = ("CONDITIONING", "CONDITIONING_PIPELINE")
    RETURN_NAMES = ("conditioning", "conditioning_pipeline")
    FUNCTION = "encode_conditioning"

    CATEGORY = "clip-suite/conditioning"
    def encode_conditioning(self, clip, clip_pipeline=None):
        """
        Encodes the conditioning from the provided CLIP model and pipeline.
        """
        if not isinstance(clip, CLIP):
            raise ValueError(f"{self.__class__.__name__}: Provided clip is not a valid CLIP instance.")

        # Convert the comfy CLIP to a MultiClip dictionary
        multiclip_dict = translate_comfy_clip_to_multiclip(clip)

        # Reconstruct the comfy CLIP from the MultiClip dictionary
        reconstructed_clip = reconstruct_comfy_clip_from_multiclip(multiclip_dict)

        # If a clip_pipeline is provided, use it; otherwise, return None
        conditioning_pipeline = clip_pipeline if clip_pipeline else None

        return (reconstructed_clip.conditioning, conditioning_pipeline)



class AbsClipSplitter:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip": ("CLIP",),
            },
            "optional": {
                "clip_pipeline": ("CLIP_PIPELINE", {"default": None}),
            }
        }

    RETURN_TYPES = (
        "ABS_CLIP_L",
        "ABS_CLIP_G",
        "ABS_CLIP_H",
        "ABS_CLIP_VISION",
        "ABS_T5",
        "ABS_LLM",
        "ABS_UNKNOWN_SD",
    )
    RETURN_NAMES = (
        "clip_l",
        "clip_g",
        "clip_h",
        "clip_vision",
        "t5",
        "llm",
        "unknown",
    )

    FUNCTION = "split"
    CATEGORY = "clip/custom_pipeline"

    def split(self, clip):
        # Step 1: Translate comfy CLIP to MultiClip dictionary
        translated = translate_comfy_clip_to_multiclip(clip)

        # Step 2: Prepare clean assignment dictionary
        output = {
            "clip_l": None,
            "clip_g": None,
            "clip_h": None,
            "clip_vision": None,
            "t5": None,
            "llm": None,
            "unknown": {},
        }

        # Step 3: Distribute parts properly
        for subclip_name, subclip_data in translated.items():
            lowered = subclip_name.lower()
            if "clip_l" in lowered:
                output["clip_l"] = subclip_data
            elif "clip_g" in lowered:
                output["clip_g"] = subclip_data
            elif "clip_h" in lowered:
                output["clip_h"] = subclip_data
            elif "vision" in lowered:
                output["clip_vision"] = subclip_data
            elif "t5" in lowered:
                output["t5"] = subclip_data
            elif "bert" in lowered:
                output["bert"] = subclip_data
            elif "llama" in lowered:
                output["llama"] = subclip_data
            elif "llm" in lowered:
                output["llm"] = subclip_data
            else:
                output["unknown"][subclip_name] = subclip_data

        # Step 4: Return ordered tuple
        return (
            output["clip_l"],
            output["clip_g"],
            output["clip_h"],
            output["clip_vision"],
            output["t5"],
            output["llm"],
            output["unknown"] if output["unknown"] else None,
        )