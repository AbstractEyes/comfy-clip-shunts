from dataclasses import dataclass

import torch
import logging
import comfy

from comfy.sd import CLIP
from ..model.model_manager import get_model_manager
from ..abs_sd.CLIP import load_clip, CLIPType

import folder_paths


logger = logging.getLogger(__name__)

model_manager = get_model_manager() # singleton instance of the model manager


class EmptyClipLatent:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip": ("CLIP",),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 128, "step": 1, "tooltip": "Batch size for the empty conditioning."}),
                "override": (["yes", "no"], {"default": "no", "tooltip": "Override the default empty conditioning with a custom one."}),
                "amount": ("INT", {"default": 1, "min": 1, "max": 128, "step": 1}),
                "length": ("INT", {"default": 77, "min": 1, "max": 2048, "step": 1}),
                "dims": ("INT", {"default": 768, "min": 1, "max": 2048, "step": 1}),
            }
        }

    RETURN_TYPES = ("CONDITIONING", )
    RETURN_NAMES = ("empty_conditionings",)
    FUNCTION = "get_empty_conditioning"

    CATEGORY = "clip-suite/conditioning"

    def get_empty_conditioning(self, clip, override, batch_size, length, dims):
        """
        Generates an empty conditioning tensor for the specified CLIP model.
        This is useful for initializing conditioning tensors when no input is provided.
        """

        # Check the clip models for their size and create the empty conditioning accordingly

        # Create an empty conditioning tensor with the specified dimensions
        empty_conditioning = torch.zeros(batch_size, length, dims, dtype=torch.float64)

        # Return the empty conditioning as a tuple
        return (empty_conditioning,)


import torch
from .pipes import ConditionPipe, ConditionEmbeddingNode
from ..utils.conditioning_shifter import ConditioningShifter, ShiftConfig

class CLIPPipelineToEncoderPipe:
    """
    Converts a CLIP_PIPELINE into one or more ENCODER_PIPE entries.
    Useful for routing into standard LoRA or conditioning systems.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip_pipeline": ("CLIP_PIPELINE", {}),
                "max_length": ("INT", {"default": 77, "min": 1, "max": 8192}),
                "padding": (["max_length", "longest", "do_not_pad"], {"default": "max_length"}),
                "device": (["cpu", "cuda", "mps"], {"default": "cuda" if torch.cuda.is_available() else "cpu"}),
            }
        }

    RETURN_TYPES = ("ENCODER_PIPE",)
    RETURN_NAMES = ("encoder_pipe",)
    FUNCTION = "convert"
    CATEGORY = "encoder/bridge"

    def convert(self, clip_pipeline, max_length, padding, device):
        device = torch.device(device)
        registry = clip_pipeline.get("registry", [])
        if not registry:
            raise ValueError("CLIP_PIPELINE has no valid registry entries.")

        encoder_pipes = []
        for idx, entry in enumerate(registry):
            model = entry.get("model")
            tokenizer = entry.get("tokenizer")
            name = entry.get("name", f"clip_{idx}")
            model_type = entry.get("type", "clip")

            if model is None or tokenizer is None:
                continue

            config_dict = {
                "model_id": f"{model_type}_{name}",
                "model_type": model_type,
                "model_name": name,
                "source": "clip_pipeline",
                "device": str(device),
                "trust_remote_code": False,
                "config": {
                    "max_length": max_length,
                    "padding": padding
                }
            }

            encoder_pipe_entry = {
                "model": model,
                "tokenizer": tokenizer,
                "config": config_dict
            }
            encoder_pipes.append(encoder_pipe_entry)

        return (encoder_pipes,)



def make_clip_pipeline(clip, model_type: str = "unknown") -> dict:
    """
    Constructs a symbolic CLIP_PIPELINE wrapper from a legacy CLIP object.
    This does NOT use MultiClipRegistry. It manually builds a minimal pipeline interface.
    """

    return {
        "source_model": clip,
        "model_type": model_type
    }



from ..abs_sd.CLIP import CLIPType, load_clip


def attention_multiply(attn, model, q, k, v, out):
    m = model.clone()
    sd = model.model_state_dict()

    for key in sd:
        if key.endswith("{}.to_q.bias".format(attn)) or key.endswith("{}.to_q.weight".format(attn)):
            m.add_patches({key: (None,)}, 0.0, q)
        if key.endswith("{}.to_k.bias".format(attn)) or key.endswith("{}.to_k.weight".format(attn)):
            m.add_patches({key: (None,)}, 0.0, k)
        if key.endswith("{}.to_v.bias".format(attn)) or key.endswith("{}.to_v.weight".format(attn)):
            m.add_patches({key: (None,)}, 0.0, v)
        if key.endswith("{}.to_out.0.bias".format(attn)) or key.endswith("{}.to_out.0.weight".format(attn)):
            m.add_patches({key: (None,)}, 0.0, out)
    return m



class ClipEncoderLoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip_name": (
                    folder_paths.get_filename_list("text_encoders"),
                ),
                "model_type": (
                    [
                        "stable_diffusion", "novelai_v2", "stable_cascade", "sd3",
                        "stable_audio", "mochi", "ltxv", "pixart", "cosmos",
                        "lumina2", "wan", "hidream", "chroma", "ace", "omnigen2",
                        "flux", "undefined"
                    ],
                    {"default": "undefined", "tooltip": "Which model format this CLIP file was trained for"}
                ),
                "encoder_type": (
                    [
                        "clip_l", "clip_g", "clip_h",
                        "t5", "t5_unchained", "llama", "vision"
                    ],
                    {"default": "clip_l", "tooltip": "Symbolic encoder role for this CLIP (used for downstream interpretation)"}
                ),
                "clip_set_last_layer": ("INT", {"default": -2, "min": -69696, "max": -1, "step": -1,
                                                "tooltip": "Set the last layer of the CLIP model to this value. 0 means no change."}),
                "alpha_influence": ("FLOAT", {"default": 1.000, "min": -10000.000, "max": 10000.000, "step": 0.001,}),
                "trajectory_impact": ("FLOAT", {"default": 1.000, "min": -10000.000, "max": 10000.000, "step": 0.001,}),
                "low_attn_weight": ("FLOAT", {"default": 1.000, "min": -10000.000, "max": 10000.000, "step": 0.001,}),
                "mid_attn_weight": ("FLOAT", {"default": 1.000, "min": -10000.000, "max": 10000.000, "step": 0.001,}),
                "high_attn_weight": ("FLOAT", {"default": 1.000, "min": -10000.000, "max": 10000.000, "step": 0.001,}),
                "out_attn_weight": ("FLOAT", {"default": 1.000, "min": -10000.000, "max": 10000.000, "step": 0.001,}),
            },
            "optional": {
                "device": (
                    ["default", "cpu", "cuda"],
                    {"default": "default", "advanced": True}
                ),

            }
        }

    RETURN_TYPES = ("CLIP", "ENCODER_PIPE", "CLIP")
    RETURN_NAMES = ("clip", "encoder_pipe", "attn_clip")
    FUNCTION = "load_clip_internal"
    CATEGORY = "advanced/loaders"
    DESCRIPTION = "[ABS] Loads a single CLIP and returns both CLIP and ENCODER_PIPE for symbolic pipeline use."


    def load_clip_internal(self,
                           clip_name,
                           model_type,
                           encoder_type,
                           device="default",
                           clip_set_last_layer=-1,
                           alpha_influence=1.0,
                           trajectory_impact=1.0,
                           low_attn_weight=1.0,
                           mid_attn_weight=1.0,
                           high_attn_weight=1.0,
                           out_attn_weight=1.0):
        clip_type = getattr(CLIPType, model_type.upper(), CLIPType.STABLE_DIFFUSION)

        model_options = {}
        if device == "cpu":
            model_options["load_device"] = model_options["offload_device"] = torch.device("cpu")

        clip_path = folder_paths.get_full_path_or_raise("text_encoders", clip_name)
        clip = load_clip(
            ckpt_paths=[clip_path],
            embedding_directory=folder_paths.get_folder_paths("embeddings"),
            clip_type=clip_type,
            model_options=model_options
        )
        clip.clip_layer(clip_set_last_layer)  # Set to -1 by default, meaning no change
        m = clip.clone()

        sd = m.patcher.model_state_dict()

        for key in sd:
            if key.endswith("self_attn.q_proj.weight") or key.endswith("self_attn.q_proj.bias"):
                m.add_patches({key: (None,)}, 0.0, low_attn_weight)
            if key.endswith("self_attn.k_proj.weight") or key.endswith("self_attn.k_proj.bias"):
                m.add_patches({key: (None,)}, 0.0, mid_attn_weight)
            if key.endswith("self_attn.v_proj.weight") or key.endswith("self_attn.v_proj.bias"):
                m.add_patches({key: (None,)}, 0.0, high_attn_weight)
            if key.endswith("self_attn.out_proj.weight") or key.endswith("self_attn.out_proj.bias"):
                m.add_patches({key: (None,)}, 0.0, out_attn_weight)

        encoder_pipe = [{
            "clip": m,
            "type": encoder_type,
            "tokenizer": m.tokenizer,
            "config": {
                "alpha_influence": alpha_influence,
                "trajectory_impact": trajectory_impact,
            },
        }]

        return clip, encoder_pipe, m



class ACLIPLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "clip_name": (folder_paths.get_filename_list("text_encoders"), ),
                              "type": (["stable_diffusion", "novelai_v2", "stable_cascade", "sd3", "stable_audio", "mochi", "ltxv", "pixart", "cosmos", "lumina2", "wan", "hidream", "chroma", "ace", "omnigen2"], ),
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
        return {
            "required": {
                "clip_name1": (folder_paths.get_filename_list("text_encoders"), ),
                "clip_name2": (folder_paths.get_filename_list("text_encoders"), ),
                "type": (["sdxl", "sd3", "flux", "hunyuan_video", "hidream"], {
                    "default": "sdxl",
                    "tooltip": "Type of the CLIP model to load."
                }),
            },
            "optional": {
                "device": (["default", "cpu", "cuda"], {"advanced": True}),
            }
        }

    RETURN_TYPES = ("CLIP", )
    RETURN_NAMES = ("clip", )
    FUNCTION = "load_clip_internal"
    CATEGORY = "advanced/loaders"

    def load_clip_internal(self, clip_name1, clip_name2, type="sdxl", device="default"):
        clip_type = getattr(CLIPType, type.upper(), CLIPType.STABLE_DIFFUSION)

        clip_path1 = folder_paths.get_full_path_or_raise("text_encoders", clip_name1)
        clip_path2 = folder_paths.get_full_path_or_raise("text_encoders", clip_name2)

        model_options = {}
        if device == "cpu":
            model_options["load_device"] = model_options["offload_device"] = torch.device("cpu")

        clip = load_clip(
            ckpt_paths=[clip_path1, clip_path2],
            embedding_directory=folder_paths.get_folder_paths("embeddings"),
            clip_type=clip_type,
            model_options=model_options
        )

        return (clip,)


class ATripleCLIPLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "clip_name1": (folder_paths.get_filename_list("text_encoders"), ),
                "clip_name2": (folder_paths.get_filename_list("text_encoders"), ),
                "clip_name3": (folder_paths.get_filename_list("text_encoders"), )
            },
            "optional": {
                "model_type": (["sd3"], {"default": "sd3", "tooltip": "Type of the CLIP model to load."}),
                "device": (["default", "cpu", "cuda"], {"advanced": True}),
            }
        }

    RETURN_TYPES = ("CLIP",)
    RETURN_NAMES = ("clip",)
    FUNCTION = "load_clip_internal"
    CATEGORY = "advanced/loaders"

    def load_clip_internal(self, clip_name1, clip_name2, clip_name3, model_type="sd3", device="default"):
        clip_type = getattr(CLIPType, model_type.upper(), CLIPType.SD3)

        clip_path1 = folder_paths.get_full_path_or_raise("text_encoders", clip_name1)
        clip_path2 = folder_paths.get_full_path_or_raise("text_encoders", clip_name2)
        clip_path3 = folder_paths.get_full_path_or_raise("text_encoders", clip_name3)

        model_options = {}
        if device == "cpu":
            model_options["load_device"] = model_options["offload_device"] = torch.device("cpu")

        clip = load_clip(
            ckpt_paths=[clip_path1, clip_path2, clip_path3],
            embedding_directory=folder_paths.get_folder_paths("embeddings"),
            clip_type=clip_type,
            model_options=model_options
        )

        return (clip,)


class AQuadrupleCLIPLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "clip_name1": (folder_paths.get_filename_list("text_encoders"), ),
                "clip_name2": (folder_paths.get_filename_list("text_encoders"), ),
                "clip_name3": (folder_paths.get_filename_list("text_encoders"), ),
                "clip_name4": (folder_paths.get_filename_list("text_encoders"), )
            },
            "optional": {
                "model_type": (["hidream"], {"default": "hidream", "tooltip": "Type of the CLIP model to load."}),
                "device": (["default", "cpu", "cuda"], {"advanced": True}),
            }
        }

    RETURN_TYPES = ("CLIP", "CLIP_PIPELINE")
    RETURN_NAMES = ("clip", "clip_pipeline")
    FUNCTION = "load_clip_internal"
    CATEGORY = "advanced/loaders"

    def load_clip_internal(self, clip_name1, clip_name2, clip_name3, clip_name4, model_type="hidream", device="default"):
        clip_type = getattr(CLIPType, model_type.upper(), CLIPType.HIDREAM)

        clip_path1 = folder_paths.get_full_path_or_raise("text_encoders", clip_name1)
        clip_path2 = folder_paths.get_full_path_or_raise("text_encoders", clip_name2)
        clip_path3 = folder_paths.get_full_path_or_raise("text_encoders", clip_name3)
        clip_path4 = folder_paths.get_full_path_or_raise("text_encoders", clip_name4)

        model_options = {}
        if device == "cpu":
            model_options["load_device"] = model_options["offload_device"] = torch.device("cpu")

        clip = load_clip(
            ckpt_paths=[clip_path1, clip_path2, clip_path3, clip_path4],
            embedding_directory=folder_paths.get_folder_paths("embeddings"),
            clip_type=clip_type,
            model_options=model_options
        )

        return (clip,)




from typing import Tuple



class ClipSetDtypeNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip": ("CLIP",),
                "dtype": (["float32", "float16", "bfloat16"], {"default": "float32"}),
                "device": (["default", "cpu", "cuda"], {"default": "default", "advanced": True}),
            }
        }

    RETURN_TYPES = ("CLIP",)
    RETURN_NAMES = ("clip",)
    FUNCTION = "set_dtype"
    CATEGORY = "ABS/CLIP"

    def set_dtype(self, clip, dtype: str, device: str) -> Tuple:
        """
        Sets the dtype of the provided CLIP model.
        """
        if dtype == "float32":
            clip.to(torch.float32)
        elif dtype == "float16":
            clip.to(torch.float16)
        elif dtype == "bfloat16":
            clip.to(torch.bfloat16)

        if device == "cuda":
            if torch.cuda.is_available():
                clip.to("cuda")
            else:
                raise RuntimeError("CUDA is not available on this system.")
        else:
            clip.to("cpu")

        return (clip,)