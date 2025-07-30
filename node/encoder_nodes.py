import logging
from typing import Optional

import torch
from comfy.sd import CLIP
from ..model.configs import ShuntUtil

logger = logging.getLogger(__name__)

import hashlib
from ..model.model_manager import get_model_manager
from ..model.configs import ENCODER_CONFIGS, ShuntData, EncoderData
from ..utils.conditioning_shifter import ConditioningShifter

from ..sampler.formulas.folding import FoldingKernels
from ..sampler.formulas.schedules import SchedulerModes


class EncoderStackerNode:
    """
    Collects multiple encoder objects into a list stack.
    Ensures all entries are flat encoder dicts, not nested lists.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "encoder1": ("ENCODER_PIPE", {}),
            },
            "optional": {
                "encoder2": ("ENCODER_PIPE", {}),
                "encoder3": ("ENCODER_PIPE", {}),
                "encoder4": ("ENCODER_PIPE", {}),
                "encoder5": ("ENCODER_PIPE", {}),
            }
        }

    RETURN_TYPES = ("ENCODER_PIPE",)
    RETURN_NAMES = ("encoders",)
    FUNCTION = "stack"
    CATEGORY = "encoder/core"

    def stack(self, encoder1, encoder2=None, encoder3=None, encoder4=None, encoder5=None):
        raw_inputs = [encoder1, encoder2, encoder3, encoder4, encoder5]
        encoders = []

        for entry in raw_inputs:
            if entry is None:
                continue
            if isinstance(entry, list):
                encoders.extend(entry)
            else:
                encoders.append(entry)

        return (encoders,)




from ..utils.conditioning_shifter import ConditioningShifter, ShiftConfig
from .pipes import ConditionPipe, ConditionEmbeddingNode

class EncodingGeneratorNode:
    """
    Processes a stacked list of encoders and generates a ConditionPipe.
    Each encoder is sampled using ConditioningShifter and wrapped as a ConditionEmbeddingNode.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "encoder_stack": ("ENCODER_STACK", {}),
                "prompt": ("STRING", {"default": "a photo of a robot.", "multiline": True}),
                "device": (["cpu", "cuda", "mps"], {"default": "cuda" if torch.cuda.is_available() else "cpu"}),
            }
        }

    RETURN_TYPES = ("CONDITION_PIPE",)
    RETURN_NAMES = ("condition_pipe",)
    FUNCTION = "generate"
    CATEGORY = "encoder/core"

    def generate(self, encoder_stack, prompt, device):
        device = torch.device(device)
        shift_cfg = ShiftConfig(prompt=prompt)

        pipe = ConditionPipe()

        for idx, encoder_entry in enumerate(encoder_stack):
            model = encoder_entry.get("model")
            tokenizer = encoder_entry.get("tokenizer")
            config = encoder_entry.get("config", {})
            encoder_id = config.get("model_name", f"encoder_{idx}")

            # Extract tensor [B,T,D]
            tensor = ConditioningShifter.extract_encoder_embeddings(encoder_entry, device, shift_cfg)

            # Generate placeholder masks (can be enhanced)
            B, T, D = tensor.shape
            default_mask = torch.ones((B, T), device=device)
            symbolic = torch.mean(tensor, dim=1)  # pooled
            symbolic_mask = torch.ones_like(symbolic)

            # Wrap
            node = ConditionEmbeddingNode(
                name=f"{encoder_id}_cond",
                embedding=tensor,
                embedding_mask=default_mask,
                trajectory=tensor.clone(),  # use same for now
                trajectory_mask=default_mask,
                symbolic=symbolic,
                symbolic_mask=symbolic_mask,
                config={"encoder_id": encoder_id, "role": "conditioning"}
            )
            pipe.add(node)

        return (pipe,)


class EncoderEmbeddings:
    # a representative class for any reusable encoder embeddings created.
    # meant to be passed down the pipeline and used by any shunt-suite shaping nodes
    # these are generated every time an encoder is run, and can be used to store embeddings
    # this is a raw encoder output, not a conditioning, a very big distinction.
    # these must also be cloned before use to modify them without affecting the original embeddings.
    def __init__(self, embeddings: torch.Tensor, config: Optional[dict] = None):
        """
        Initialize the EncoderEmbeddings with a tensor and optional configuration.
        :param embeddings: A tensor of shape [B, T, D] representing the embeddings.
        :param config: Optional configuration dictionary for the embeddings.
        """
        self.embeddings = embeddings
        self.config = config if config is not None else {}


class EncodeEmbeddings:
    # uses the encoder pipeline and returns a set of embeddings, very straightforward.
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "encoder": ("ENCODER_PIPE", {"default": {}}),
                "encoder_config": ("ENCODER_EMBEDDING_CONFIG", {"default": {}}),
                "seed": ("INT", {"default": 420, "min": 0, "max": 100000000}),
                "device": (["cpu", "cuda"], {"default": "cpu", "tooltip": "Device to run the encoding on."}),
            }
        }

    RETURN_TYPES = ("ENCODER_EMBEDDINGS",)
    RETURN_NAMES = ("embeddings",)
    FUNCTION = "encode"
    CATEGORY = "encoder/embeddings"
    def encode(self, encoder, encoder_config, device):
        """
        Encodes the input using the specified encoder pipeline and configuration.
        :param encoder: The encoder pipeline to use for encoding.
        :param encoder_config: Configuration for the encoder embeddings.
        :return: An EncoderEmbeddings object containing the encoded embeddings.
        """
        # Ensure the encoder is valid
        if not encoder:
            raise ValueError("Encoder pipeline must be provided.")
        device = torch.device(device)
        # Extract embeddings using the ConditioningShifter
        embeddings = ConditioningShifter.extract_encoder_embeddings(encoder, device=torch.device(), config=encoder_config)

        return ([EncoderEmbeddings(embeddings, config=encoder_config)],)



class EncoderSamplerSimple:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clips": ("CLIP", {"default": {}}),
                "encoders": ("ENCODER_PIPE", {"default": {}}),
                "steps": ("INT", {"default": 4, "min": 1, "max": 1000}),
                "cfg_scale": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 100.0}),
                "guidance_scale": ("FLOAT", {"default": 5, "min": 0.0, "max": 100.0}),
            }
        }

    RETURN_TYPES = ("CONDITIONING", )
    RETURN_NAMES = ("conditioning", )
    FUNCTION = "sample"

    CATEGORY = "encoder/sampler"

    def sample(self, clips, encoders, steps, cfg_scale, guidance_scale):
        """
        Sample a conditioning from the provided CLIP and encoder models.
        This is a simplified version that does not require configuration for the various parameters.
        """
        # Ensure clips and encoders are valid
        if not clips or not encoders:
            raise ValueError("Both clips and encoders must be provided.")




        return ( )

class EncoderFoldingSchedulerConfig:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "folding_scheduler": ([
                    "none", "tau", "top_k", "top_20k", "top_50k",
                    "cosine", "cascade", "cos", "sine",
                    "shockwave", "pulse", "wave"
                ], {"default": "none"}),
                "tau": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0}),
                "top_k": ("FLOAT", {"default": 50, "min": 0.0, "max": 10000}),
                "top_p": ("FLOAT", {"default": 0.9, "min": 0.0, "max": 1.0}),
            }
        }

    RETURN_TYPES = ("ENCODER_FOLDING_SCHEDULER_CONFIG",)
    RETURN_NAMES = ("encoder_folding_scheduler_config",)
    FUNCTION = "configure"
    CATEGORY = "encoder/scheduler"

    def configure(self, folding_scheduler, tau, top_k, top_p):
        """Prepare the configuration dict with the provided parameters."""
        return ({
            "folding_scheduler": folding_scheduler,
            "tau": tau,
            "top_k": top_k,
            "top_p": top_p
        },)

class EncoderEmbeddingConfig:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pos_embedding": (["none", "cos", "sine", "cosine"], {"default": "cos"}),
                "normalization_anchor": (["none", "l2", "l1", "heun", "surge", "sigma", "delta", "gate", "bong"], {"default": "surge"}),
            }
        }

    RETURN_TYPES = ("ENCODER_EMBEDDING_CONFIG",)
    RETURN_NAMES = ("encoder_embedding_config",)
    FUNCTION = "configure"
    CATEGORY = "encoder/embedding"

    def configure(self, pos_embedding, normalization_anchor):
        """Prepare the configuration dict with the provided parameters."""
        return ({
            "pos_embedding": pos_embedding,
            "normalization_anchor": normalization_anchor
        },)


class EncoderProjectionConfig:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "force_projection_in": ("BOOLEAN", {"default": False, "tooltip": "Force projection of context window to model's max length."}),
                "projection_dims_in": ("INT", {"default": 768, "min": 1, "max": 8192}),
                "interpolation_method_in": (["lerp", "slerp", "cosine", "sine", "linear", "mixed"], {"default": "slerp", "tooltip": "Method to use for interpolating projections."}),
                "force_projection_out": ("BOOLEAN", {"default": False, "tooltip": "Force projection of model output to context window size."}),
                "projection_dims_out": ("INT", {"default": 768, "min": 1, "max": 8192}),
                "interpolation_method_out": (["lerp", "slerp", "cosine", "sine", "linear", "mixed"], {"default": "slerp", "tooltip": "Method to use for interpolating model output projections."}),
            }
        }

    RETURN_TYPES = ("ENCODER_PROJECTION_CONFIG",)
    RETURN_NAMES = ("encoder_projection_config",)
    FUNCTION = "configure"
    CATEGORY = "encoder/projection"

    def configure(self, force_projection_in, projection_dims_in, interpolation_method_in,
                  force_projection_out, projection_dims_out, interpolation_method_out):
        """Prepare the configuration dict with the provided parameters."""
        return ({
            "force_projection_in": force_projection_in,
            "projection_dims_in": projection_dims_in,
            "interpolation_method_in": interpolation_method_in,
            "force_projection_out": force_projection_out,
            "projection_dims_out": projection_dims_out,
            "interpolation_method_out": interpolation_method_out
        },)


from ..sampler.formulas.modes import (
    FoldingPoolingTypes,
    FoldingPaddingTypes,
)

class EncoderSamplerConfig:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                # passes the context window downstream with a flag that determines if it overrides the default
                "override_context_window": ("BOOLEAN", {"default": True}),
                # a helper point, necessary for the alucard sampler to work
                "context_window": ("STRING", {
                    "default": "a photo of a robot.",
                    "multiline": True
                }), # the downstream respects the override flag, and if downstream has no prompt it uses the default
                # a core fundamental trait of alucard
                "steps": ("INT", {"default": 100, "min": 1, "max": 100000}),

                # Completely unimplemented, likely to never be used
                "cfg_scale": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 100.0}),
                "guidance_scale": ("FLOAT", {"default": 5, "min": 0.0, "max": 100.0}),

                # Absolutely fantastic set of formulas.
                "folding": (FoldingKernels.to_list(), {"default": FoldingKernels.shiva, "tooltip": "Folding mode to use for the encoder."}),

                # implemented for testing, some schedulers fail and require edge cases to be handled
                "folding_scheduler": (SchedulerModes.to_list(), {"default": SchedulerModes.TAU, "tooltip": "Folding scheduler to use for the encoder."}),

                # implemented for testing, disabled for debugging
                "padding_mode": (FoldingPaddingTypes.to_list(), {"default": FoldingPaddingTypes.SPARSE, "tooltip": "Padding mode to use for the encoder."}),
                # implemented for testing, works splendidly
                "pooling_mode": (FoldingPoolingTypes.to_list(),{"default": FoldingPoolingTypes.BILINEAR, "tooltip": "Pooling mode to use for the encoder."}),
                # predominantly ignored, but used in some places
                "use_alpha_mask": ("BOOLEAN", {"default": True}),
                # todo - flag implemented not respected, is implemented elsewhere
                "cosine_similarity_gate": ("BOOLEAN", {"default": False}),

                # todo - not implemented yet
                "pos_embedding": (["none", "cos", "sine", "cosine"], {"default": "none"}),
                "normalization_anchor": (["none", "l2", "l1", "heun", "surge", "sigma", "delta", "gate", "bong"], {"default": "none"}),

                # doesn't work correctly, connected but not implemented everywhere
                "top_k": ("FLOAT", {"default": 50.0, "min": 0.0, "max": 10000.0}),
                # connected, but not implemented everywhere
                "top_p": ("FLOAT", {"default": 0.9, "min": 0.0, "max": 1.0}),
                # connected, but only on some code paths
                "temperature": ("FLOAT", {"default": 5.0, "min": 0.0, "max": 50.0}),
                # not connected
                "tau": ("FLOAT", {"default": 5.0, "min": 0.0, "max": 50.0}),
                #not connected, requires cache
                "beams": ("INT", {"default": 4, "min": 1, "max": 32}),

                #works
                "max_windows": ("INT", {"default": 64, "min": 1, "max": 2048}),
                "context_window_size": ("INT", {"default": 2048, "min": 77, "max": 8192}),
                "sliding_window_size": ("INT", {"default": 77, "min": 1, "max": 8192}),
                "sliding_window_stride": ("INT", {"default": 77, "min": 1, "max": 2048}),

                # projection in works, projection out is not implemented for model specifics yet but will work.
                "force_projection_in": ("BOOLEAN", {"default": False, "tooltip": "Force projection of context window to model's max length."}),
                "projection_dims_in": ("INT", {"default": 768, "min": 1, "max": 8192}),
                "interpolation_method_in": (["linear"], {"default": "linear", "tooltip": "Method to use for interpolating projections."}),
                "force_projection_out": ("BOOLEAN", {"default": False, "tooltip": "Force projection of model output to context window size."}),
                "projection_dims_out": ("INT", {"default": 768, "min": 1, "max": 8192}),
                "interpolation_method_out": (["linear"], {"default": "linear", "tooltip": "Method to use for interpolating model output projections."}),
                "use_rose_similarity": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Use ROSE-based symbolic similarity instead of cosine similarity for resonance evaluation."
                }),
                "device": (["cpu", "cuda", "mps"], {"default": "cuda" if torch.cuda.is_available() else "cpu", "tooltip": "Device to run the encoder on."}),
            }
        }

    RETURN_TYPES = ("ENCODER_SAMPLER_CONFIG",)
    RETURN_NAMES = ("encoder_sampler_config",)
    FUNCTION = "configure"
    CATEGORY = "encoder/sampler"

    def configure(self,
                    override_context_window,
                    context_window,
                    steps,
                    cfg_scale,
                    guidance_scale,
                    folding,
                    folding_scheduler,
                    padding_mode,
                    pooling_mode,
                    use_alpha_mask,
                    cosine_similarity_gate,
                    pos_embedding,
                    normalization_anchor,
                    top_k,
                    top_p,
                    temperature,
                    tau,
                    beams,
                    max_windows,
                    context_window_size,
                    sliding_window_size,
                    sliding_window_stride,
                    force_projection_in,
                    projection_dims_in,
                    interpolation_method_in,
                    force_projection_out,
                    projection_dims_out,
                    interpolation_method_out,
                    use_rose_similarity,
                    device):
        """Prepare the configuration dict with the provided parameters."""
        return ({
            "override_context_window": override_context_window,
            "context_window": context_window,
            "steps": steps,
            "cfg_scale": cfg_scale,
            "guidance_scale": guidance_scale,
            "folding": folding,
            "folding_scheduler": folding_scheduler,
            "padding_mode": padding_mode,
            "pooling_mode": pooling_mode,
            "use_alpha_mask": use_alpha_mask,
            "cosine_similarity_gate": cosine_similarity_gate,
            "pos_embedding": pos_embedding,
            "normalization_anchor": normalization_anchor,
            "top_k": top_k,
            "top_p": top_p,
            "temperature": temperature,
            "tau": tau,
            "beams": beams,
            "max_windows": max_windows,
            "context_window_size": context_window_size,
            "sliding_window_size": sliding_window_size,
            "sliding_window_stride": sliding_window_stride,
            "force_projection_in": force_projection_in,
            "projection_dims_in": projection_dims_in,
            "interpolation_method_in": interpolation_method_in,
            "force_projection_out": force_projection_out,
            "projection_dims_out": projection_dims_out,
            "interpolation_method_out": interpolation_method_out,
            "use_rose_similarity": use_rose_similarity,
            "device": device,
        },)

from ..sampler.alucard import FieldWalker, FieldWalkerConfig
from ..sampler.conditionings import ConditioningData, ConditioningContainer

from ..sampler.conditionings import ConditioningData


import torch
import torch.nn.functional as F
from ..sampler.alucard import FieldWalkerConfig
from ..sampler.integra import IntegraConfig, IntegraOrchestrator
from ..sampler.sliding_window import ShuntStackConfig
from ..utils.conditioning_shifter import ConditioningShifter, ShiftConfig
from ..sampler.formulas.schedules import FormulaScheduler
from ..utils.alignment import match_feature_dims, match_tokens
from ..sampler.alucard_exceptions import AlucardShapeError

import torch.nn.functional as F



from ..utils.alignment import match_feature_dims, match_tokens
from ..sampler.alucard_exceptions import AlucardShapeError



BEATRIX_SPECIAL_TOKENS_AND_SHUNTS = [
    "<subject>","<subject1>","<subject2>","<pose>","<emotion>","<surface>","<lighting>","<material>","<accessory>",
    "<footwear>", "<upper_body_clothing>","<hair_style>","<hair_length>","<headwear>","<texture>","<pattern>","<grid>",
    "<zone>","<offset>","<object_left>","<object_right>","<relation>","<intent>","<style>","<fabric>","<jewelry>",
    "[SHUNT_1000000]","[SHUNT_1000001]","[SHUNT_1000002]","[SHUNT_1000003]","[SHUNT_1000004]",
    "[SHUNT_1000005]","[SHUNT_1000006]","[SHUNT_1000007]","[SHUNT_1000008]","[SHUNT_1000009]","[SHUNT_1000010]",
    "[SHUNT_1000011]","[SHUNT_1000012]","[SHUNT_1000013]","[SHUNT_1000014]","[SHUNT_1000015]","[SHUNT_1000016]",
    "[SHUNT_1000017]","[SHUNT_1000018]","[SHUNT_1000019]","[SHUNT_1000020]","[SHUNT_1000021]","[SHUNT_1000022]",
    "[SHUNT_1000023]","[SHUNT_1000024]","[SHUNT_1000025]","<EOF>","<START>","<END>","<PAD>","<MASK>","[CLS]","[SEP]","[PAD]",
    "<|startoftext|>", "<|endoftext|>", "<|startofimage|>", "<|endofimage|>", "<|startofvideo|>", "<|endofvideo|>",
    "<|startofaudio|>", "<|endofaudio|>", "<|startofdocument|>", "<|endofdocument|>", "<|startofcode|>", "<|endofcode|>",
    "<|startofchat|>", "<|endofchat|>", "<|startofquestion|>", "<|endofquestion|>", "<|startofanswer|>", "<|endofanswer|>",
    "<|startofparagraph|>", "<|endofparagraph|>", "<|startofsentence|>", "<|endofsentence|>", "<|startofphrase|>", "<|endofphrase|>",
    "<|startofline|>", "<|endofline|>", "<|startofword|>", "<|endofword|>", "<|startofcharacter|>", "<|endofcharacter|>",
    "<|startofentity|>", "<|endofentity|>", "<|startofrelation|>", "<|endofrelation|>", "<|startofattribute|>", "<|endofattribute|>",
    "<|startofproperty|>", "<|endofproperty|>", "<|startofaction|>", "<|endofaction|>", "<|startofevent|>", "<|endofevent|>",
    "<|startofconcept|>", "<|endofconcept|>", "<|startoftopic|>", "<|endoftopic|>", "<|startoftheme|>", "<|endoftheme|>",
    "<|startofgenre|>", "<|endofgenre|>", "<|startofstyle|>", "<|endofstyle|>", "<|startofmood|>", "<|endofmood|>",
    "<|startofemotion|>", "<|endofemotion|>", "<|startoffeeling|>", "<|endoffeeling|>", "<|startofopinion|>", "<|endofopinion|>",
    "<|startofbelief|>", "<|endofbelief|>", "<|startofattitude|>", "<|endofattitude|>", "<|startofperspective|>", "<|endofperspective|>",
    "<|startofviewpoint|>", "<|endofviewpoint|>", "<|startofstance|>", "<|endofstance|>", "<|startofposition|>", "<|endofposition|>",
    "[MASK]", "[PAD]", "[CLS]", "[SEP]", "<|startoftext|>", "<|endoftext|>", "<|startofimage|>", "<|endofimage|>",
    "<MASK>", "<PAD>", "<CLS>", "<SEP>", "<|startofvideo|>", "<|endofvideo|>",
    "<mask>", "<pad>", "<cls>", "<sep>", "<|startofaudio|>", "<|endofaudio|>",
]

CLIP_ONLY_TOKENS = [ # tokens we only want clip to see.
    "</w>", "<w>"
]


def remove_special_tokens(prompt, remove_clip=False):
    """
    Removes known special tokens from the prompt.
    This is useful for cleaning up prompts before processing.
    """
    tokens = BEATRIX_SPECIAL_TOKENS_AND_SHUNTS.copy() if not remove_clip else CLIP_ONLY_TOKENS
    # find the array of tokens within the prompt to replace
    for token in tokens:
        if token in prompt:
            # remove the prompt
            prompt = prompt.replace(token, "")
    return (prompt,)

class RemoveSpecialTokens:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("cleaned_prompt",)
    FUNCTION = "rremove_special_tokens"
    CATEGORY = "encoder/special_tokens"
    def rremove_special_tokens(self, prompt):
        """
        Removes known special tokens from the prompt.
        This is useful for cleaning up prompts before processing.
        """
        # find the array of tokens within the prompt to replace
        for token in BEATRIX_SPECIAL_TOKENS_AND_SHUNTS:
            if token in prompt:
                # remove the prompt
                prompt = prompt.replace(token, "")
        return (prompt,)

import torch
import logging
from typing import List, Dict, Any

from ..sampler.alucard import FieldWalkerConfig
from ..sampler.integra import IntegraConfig, IntegraOrchestrator
from ..sampler.sliding_window import ShuntStackConfig
from ..utils.conditioning_shifter import ConditioningShifter, ShiftConfig
from ..utils.alignment import match_project, match_feature_dims, match_tokens
from ..sampler.formulas.schedules import FormulaScheduler
from ..sampler.alucard_exceptions import AlucardShapeError



logger = logging.getLogger(__name__)


class LegacyEncoderModelUsageConfig:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "use_clip_l": ("BOOLEAN", {"default": True, "tooltip": "Use the CLIP-L variant."}),
                "use_clip_g": ("BOOLEAN", {"default": True, "tooltip": "Use the CLIP-G variant."}),
                "use_t5": ("BOOLEAN", {"default": True, "tooltip": "Use the T5 variant."}),
                "use_llama": ("BOOLEAN", {"default": False, "tooltip": "Use the Llama variant."}),
                "use_clip_l_mask": ("BOOLEAN", {"default": True, "tooltip": "Use the CLIP-L mask for the conditioning."}),
                "use_clip_g_mask": ("BOOLEAN", {"default": True, "tooltip": "Use the CLIP-G mask for the conditioning."}),
                "use_t5_mask": ("BOOLEAN", {"default": True, "tooltip": "Use the T5 mask for the conditioning."}),
                "use_llama_mask": ("BOOLEAN", {"default": False, "tooltip": "Use the Llama mask for the conditioning."}),
            }
        }

    RETURN_TYPES = ("ENCODER_GATE_CONFIG",)
    RETURN_NAMES = ("encoder_gate_config",)
    FUNCTION = "configure"
    CATEGORY = "encoder/gate"

    def configure(self,
                    use_clip_l: bool = True,
                    use_clip_g: bool = True,
                    use_t5: bool = True,
                    use_llama: bool = False,
                    use_clip_l_mask: bool = True,
                    use_clip_g_mask: bool = True,
                    use_t5_mask: bool = True,
                    use_llama_mask: bool = False):
        """Prepare the configuration dict with the provided parameters."""
        return ({
            "use_clip_l": use_clip_l,
            "use_clip_g": use_clip_g,
            "use_t5": use_t5,
            "use_llama": use_llama,
            "use_clip_l_mask": use_clip_l_mask,
            "use_clip_g_mask": use_clip_g_mask,
            "use_t5_mask": use_t5_mask,
            "use_llama_mask": use_llama_mask
        },)


class RecklessEncoderConfig:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "slice_to_fill": ("BOOL", {"default": False, "tooltip": "Flail as it fills the whole context window. Good luck."}),
                "shuffle_tokens": ("BOOL", {"default": False, "tooltip": "Shuffle the tokens in the input. This is a bad idea."}),
                "shuffle_seed": ("INT", {"default": 420, "min": 0, "max": 10000000, "tooltip": "Seed for the random shuffle. This is a bad idea."}),
                "obliterate": ("BOOL", {"default": False, "tooltip": "Fun mode. Anything I feel like doing."}),
                "slice_context_window": ("INT", {"default": -1, "min": -1, "max": 16384, "tooltip": "The context window size to slice the input to."}),
                "refold_truncated": ("BOOL", {"default": False, "tooltip": "Ignore the maximum length of the clips and instead refold the difference."}),
                "use_wrong_clips": ("BOOL", {"default": False, "tooltip": "Use clips that are not compatible with the model."}),
                "square_hole": ("BOOL", {"default": False, "tooltip": "It goes in the square hole. Everything does."}),
                "use_all_masks": ("BOOL", {"default": False, "tooltip": "Use all available masks for the conditioning."}),
                "use_no_masks": ("BOOL", {"default": False, "tooltip": "Use no masks for the conditioning."}),
            }
        }

    RETURN_TYPES = ("RECKLESS_ENCODER_CONFIG",)
    RETURN_NAMES = ("reckless_encoder_config",)
    FUNCTION = "configure"
    CATEGORY = "encoder/reckless"

    def configure(self, slice_to_fill: bool = False,
                        shuffle_tokens: bool = False,
                        shuffle_seed: int = 420,
                        obliterate: bool = False,
                        slice_context_window: int = -1,
                        refold_truncated: bool = False,
                        use_wrong_clips: bool = False,
                        square_hole: bool = False,
                        use_all_masks: bool = False,
                        use_no_masks: bool = False):
        """Prepare the configuration dict with the provided parameters."""
        return ({
            "slice_to_fill": slice_to_fill,
            "shuffle_tokens": shuffle_tokens,
            "shuffle_seed": shuffle_seed,
            "obliterate": obliterate,
            "slice_context_window": slice_context_window,
            "refold_truncated": refold_truncated,
            "use_wrong_clips": use_wrong_clips,
            "square_hole": square_hole,
            "use_all_masks": use_all_masks,
            "use_no_masks": use_no_masks
        },)



ENCODER_SUPPORTED_CLIP_TYPES = [
    "clip_l", "clip_g", "clip_h", "clip_vision", "t5", "llama", "t5_unchained"
    # these are the main supported types for the internal CLIP structure, there will be more.
]

ENCODER_SUPPORTED_MODEL_TYPES = [
    "sdxl", "sd1", "flux", "hidream", "full_no_pool", "full_pool",
    # currently only supports these four modes, but WILL be extended in the future.
]

class EncoderSampler:
    MODES = ["sdxl", "sd1", "flux", "hidream"]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "encoders": ("ENCODER_PIPE", {}),
                "clip": ("CLIP", {}),
                "config": ("ENCODER_SAMPLER_CONFIG", {}),
                "mode": (ENCODER_SUPPORTED_MODEL_TYPES, {"default": "sdxl"}),
            },
            "optional": {
                "negative_config": ("ENCODER_SAMPLER_CONFIG", {}),
                "prompt_in": ("STRING", {"default": None, "multiline": True}),
                "reckless_config": ("RECKLESS_ENCODER_CONFIG", {"default": {}}),
                "clip_gate_config": ("ENCODER_GATE_CONFIG", {"default": {}}),
                "negative_prompt_in": ("STRING", {"default": None, "multiline": True}),
            }
        }

    RETURN_TYPES = ("CONDITIONING", "CONDITIONING", "DICT", "CONDITIONING", "CONDITIONING", "DICT")
    RETURN_NAMES = (
        "pos_sampled_conditioning",
        "pos_raw_conditioning",
        "pos_debug",
        "neg_sampled_conditioning",
        "neg_raw_conditioning",
        "neg_debug")
    FUNCTION = "sample"
    CATEGORY = "encoder/sampler"

    def sample(
        self,
        encoders,
        clip,
        config,
        mode="sdxl",
        negative_config=None,
        prompt_in=None,
        reckless_config=None,
        clip_gate_config=None,
        negative_prompt_in=None,
    ):
        device = config.get("device", "cpu")
        positive_prompt = prompt_in or (config.get("context_window", None) and config.get("override_context_window", False))
        negative_prompt = negative_prompt_in or (negative_config and negative_config.get("context_window", None) and negative_config.get("override_context_window", False))

        pos_sampled_conditioning, pos_raw_conditioning, pos_debug = self.__prepare_conditioning(
            positive_prompt, encoders, clip, device, mode, config=config, reckless_config=reckless_config, clip_gate_config=clip_gate_config
        )

        if not negative_prompt:
            # if no negative prompt is provided, we can just ignore it. negative is optional and we don't need to run it.
            neg_sampled_conditioning = []
            neg_raw_conditioning = []
            neg_debug = {}
        else:
            neg_sampled_conditioning, neg_raw_conditioning, neg_debug = self.__prepare_conditioning(
                negative_prompt, encoders, clip, device, mode, config=(negative_config or config), reckless_config=reckless_config,
                clip_gate_config=clip_gate_config
            )

        return (pos_sampled_conditioning, pos_raw_conditioning, pos_debug, neg_sampled_conditioning, neg_raw_conditioning, neg_debug)



    def __prepare_conditioning(self, prompt, encoders, clip, device, mode, config=None, reckless_config=None, clip_gate_config=None):
        #negative_prompt = negative_prompt_in or config.get("negative_prompt", None)

        logger.info(f"[EncoderSampler] Sampling with mode: {mode} on device {device}")#, prompt: {prompt}, encoders: {encoders}")
        a_raws = [self.__extract_symbolic(encoder, prompt, device) for encoder in encoders]

        logger.info(f"[EncoderSampler] Pre-removal prompt {prompt}.")
        encoder_prompt = remove_special_tokens(prompt, remove_clip=True)[0] if prompt else None
        logger.info(f"[EncoderSampler] Cleaned prompt: {encoder_prompt}")
        cond, data = self.__schedule_and_extract_conds(clip, encoder_prompt, device, mode=mode)
        orig_clip_full = cond.clone()
        orig_features_full = data.get("features", None)
        orig_features_full = orig_features_full.clone() if orig_features_full is not None else None

        pooled_output = data.get("pooled_output", None)
        orig_pool_full = pooled_output.clone() if pooled_output is not None else None

        clip_slices = self.__slice_conds(cond.clone(), orig_pool_full, orig_features_full, mode=mode)

        folded_outputs: Dict[str, List[torch.Tensor]] = {}

        for a_raw in a_raws:
            for cond_name, slice in clip_slices.items():
                folded = self._run_path(a_raw, cond_name, slice, config, device)
                folded_outputs.setdefault(cond_name, []).append(folded)

        if not folded_outputs:
            raise RuntimeError("No encoder-condition slices folded successfully.")
        else:
            logger.info(f"[EncoderSampler] Folded outputs: {list(folded_outputs.keys())}")

        conditioning = self._pack_conditioning_from_registry(
            folded_outputs=folded_outputs,
            cfg=config,
            device=device,
            mode=mode,
        )
        pooled = conditioning[0][1].get("pooled_output", None)
        orig_pool_ready = {
            "pooled_output": (orig_pool_full.clone() if orig_pool_full is not None else None)
        }
        raw_conditioning = [[orig_clip_full, orig_pool_ready]]
        # log the conds and the pools of both outputs

        return conditioning, raw_conditioning, {}


    def __extract_symbolic(self, pipe, prompt, device):
        encoder_type = pipe.get("type", "unknown").lower()
        clip_like = encoder_type in ENCODER_SUPPORTED_CLIP_TYPES
        # a small set of tests to run, so far showing promise

        if clip_like:
            prepared_prompt = remove_special_tokens(prompt, remove_clip=False)[0] if prompt else None
            clip_model = pipe["clip"]
            if encoder_type == "t5":
                max_tokens = 512
            else:
                max_tokens = 77
            #clip_model.load_model()
            tokens = clip_model.tokenize(prepared_prompt, tokenizer_options={
                "padding": "max_length",
                "truncation": True,
                "max_tokens": max_tokens
            })


            with torch.no_grad():
                full_cond = clip_model.encode_from_tokens_scheduled(tokens)
                cond = full_cond[0][0]  # Extract the first element which is the condition tensor
                if cond is None:
                    raise ValueError("Could not extract CLIP condition from scheduled encoding.")
            return cond.to(device)


        else:
            # model isn't clip like, we need to extract special tokens leaving the rest to the model
            if "model" in pipe:
                pipe["model"].to(device)
            #prepared_prompt = remove_special_tokens(prompt, remove_clip=True)[0] if prompt else None
            shift_cfg = ShiftConfig(prompt=prompt)
            return ConditioningShifter.extract_encoder_embeddings(pipe, device, shift_cfg).to(device)

    def __schedule_and_extract_conds(self, clip, prompt, device, mode):
        clip_l_tokens = None
        if mode == "flux":
            tokens = clip.tokenize(
                prompt,
                tokenizer_options={
                    "padding": "max_length",
                    "truncation": True,
                    "max_tokens": 512
                }
            )
            full_cond = clip.encode_from_tokens_scheduled(tokens, use_full=True)

            # Extract known-good tensors
            cond = full_cond.get("t5")  # ← symbolic field
            features = full_cond.get("clip_l")  # ← clip_l token stream

            # Place upstream pool in the pool dict - it's the clip_l features without the tokens
            logger.info(f"[EncoderSampler] Using Flux mode, cond shape: {cond.shape if cond is not None else 'None'}")
            pool = None
        elif mode == "full_no_pool": # many models use this mode, so we can assume it is a full model without pooling
            # extract from the clip and return a pool with "pool": None
            # assume the tokenizer already knows the max length and whatever else
            tokens = clip.tokenize(prompt)
            raw = clip.encode_from_tokens_scheduled(tokens)
            cond = raw[0][0]
            # no features either, we don't need them
            return cond.to(device), { "pooled_output": None }
        else:
            tokens = clip.tokenize(
                prompt,
                tokenizer_options={"padding": "max_length", "min_length": 512, "max_length": 512, "max_tokens": 512, "truncation": True}
            )

            raw = clip.encode_from_tokens_scheduled(tokens)
            cond = raw[0][0]
            pool = raw[0][1]
            features = None #raw[0][2] if len(raw[0]) > 2 else None

        return cond.to(device), {"features": features, "pooled_output": pool.get("pooled_output", None) if pool else None}

    def __slice_conds(self, clip_full, pool_dict, other=None, mode="sdxl"):
        slices = {}
        if mode == "sd1":
            slices["clip_l"] = clip_full
        elif mode == "sdxl":
            slices["clip_l"] = clip_full[:, :, :768]
            slices["clip_g"] = clip_full[:, :, 768:]
        elif mode == "flux":
            slices["t5"] = clip_full
            slices["clip_l"] = other
        elif mode == "hidream":
            slices["clip_l"] = clip_full[:, :, :768]
            slices["clip_g"] = clip_full[:, :, 768:2048]
        elif mode == "full_no_pool":
            slices["clip_l"] = clip_full # we'll assume this as a similar to sd1 mode
        return slices


    def _run_path(self, a_raw, cond_name, clip_slice, cfg, device):
        if a_raw.size(-1) != clip_slice.size(-1) or cfg.get("force_projection_in", False):
            a_proj = match_project(a_raw, clip_slice, mode=cfg.get("interpolation_method_in", "linear"))
        else:
            a_proj = a_raw

        a_feat = match_feature_dims(a_proj, clip_slice)
        b = match_tokens(clip_slice, a_feat.shape[1])
        delta = b - a_proj


        # -- Inject resonance potential --
        context = {}
        if cfg.get("enable_rope_spiral", False):
            potential = ConditioningShifter.compute_resonance_potential(
                embedding=a_feat,
                attention_mask=torch.ones(a_feat.shape[:2], dtype=torch.bool, device=a_feat.device),
                offsets=cfg.get("rope_phase_offsets", [1, 2, 4, 8, 16]),
                mode=cfg.get("rope_potential_mode", "spiral_gate")
            )
            context["resonance_potential"] = potential  # [B, T, 1] — used inside IntegraOrchestrator

        integra = self._build_integra(cfg)
        try:
            raw_folded, _ = integra.walk_encoder_field(
                a_feat, b, delta, context=context
            )
        except AlucardShapeError as e:
            raise RuntimeError(f"[EncoderSampler] Shape error: {e}") from e

        return match_tokens(raw_folded, clip_slice.shape[1]).to(device)

    def _build_integra(self, cfg):
        walker_cfg = FieldWalkerConfig(
            name=cfg.get("name", "Alucard"),
            folding_mode=cfg["folding"],
            scheduler_mode=cfg["folding_scheduler"],
            t_steps=cfg["steps"],
            padding_mode=cfg["padding_mode"],
            pooling_mode=cfg["pooling_mode"],
            scheduler_config={
                "tau": cfg["tau"],
                "top_k": cfg["top_k"],
                "top_p": cfg["top_p"],
            },
            context_overrides={
                "use_alpha_mask": cfg["use_alpha_mask"],
                "cosine_gate": cfg["cosine_similarity_gate"],
            },
        )
        stack_cfg = ShuntStackConfig(
            sliding_window_size=cfg["sliding_window_size"],
            sliding_window_stride=cfg["sliding_window_stride"],
            context_window_size=cfg["context_window_size"],
            override_context_window=cfg["override_context_window"],
            context_window=cfg["context_window"],
            max_windows=cfg["max_windows"],
        )
        return IntegraOrchestrator(IntegraConfig(
            walker_config=walker_cfg,
            stack_config=stack_cfg,
            trace_folds=False,
            enforce_projection=cfg["force_projection_in"],
            enable_clip_alignment=cfg["cosine_similarity_gate"],
            use_rose_similarity=cfg["use_rose_similarity"],
        ))

    def _pack_conditioning_from_registry(
        self,
        folded_outputs: Dict[str, List[torch.Tensor]],
        cfg: Dict[str, Any],
        device: torch.device,
        mode: str = "sdxl",
    ):
        """
        Assemble conditioning tensors per encoder, preserving alignment per source.
        Output shape: List of [B,T,D] folded embeddings, one per encoder.
        """
        # Validate input
        if not folded_outputs:
            raise RuntimeError("No folded outputs to assemble.")

        # Determine slice keys used
        keys = {
            "sdxl": ["clip_l", "clip_g"],
            "sd1": ["clip_l"],
            "flux": ["t5"],
        }.get(mode, list(folded_outputs.keys()))


        # Infer encoder count
        encoder_count = max(len(folded_outputs.get(k, [])) for k in keys)

        conditioning = []
        for i in range(encoder_count):
            parts = []
            for key in keys:
                if i < len(folded_outputs.get(key, [])):
                    parts.append(folded_outputs[key][i])

            if not parts:
                continue

            # Align token count
            token_lengths = [p.shape[1] for p in parts]
            target_T = min(token_lengths)
            parts = [p[:, :target_T, :] for p in parts]

            # Concatenate
            folded = torch.cat(parts, dim=-1)
            B, T, D = folded.shape

            # Patch CLS/END
            start = torch.zeros(B, 1, D, device=device)
            end = torch.zeros(B, 1, D, device=device)
            body = folded[:, 1:-1, :]
            patched = torch.cat([start, body, end], dim=1)

            # Pooled strategy
            if mode == "sdxl":
                pooled = patched[:, -1, 768:2048]
            elif mode == "sd1":
                pooled = patched[:, -1, :768]
            elif mode == "flux":
                pooled = patched[:, -1, :]
            else:
                pooled = patched[:, -1, :]

            conditioning.append([patched.cpu().clone(), {"pooled_output": pooled.cpu().clone()}])

        return conditioning

    def _pool_clip_l_tokens(self, walked_clip_l: torch.Tensor, strategy: str = "last") -> torch.Tensor:
        if strategy == "mean":
            return walked_clip_l.mean(dim=1)
        elif strategy == "first":
            return walked_clip_l[:, 0, :]
        elif strategy == "last":
            return walked_clip_l[:, -1, :]
        raise ValueError(f"Unknown pooling strategy: {strategy}")

class ClipStacker:
    # takes in clip pipelines and converts them into a stacked multi-clip conditioning
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip1": ("CLIP", {}),
            },
            "optional": {
                "clip2": ("CLIP", {}),
                "clip3": ("CLIP", {}),
                "clip4": ("CLIP", {}),
                "clip5": ("CLIP", {}),
            }
        }
    RETURN_TYPES = ("CLIP_PIPELINE",)
    RETURN_NAMES = ("clip_pipeline",)
    FUNCTION = "stack_clips"
    CATEGORY = "encoder/sampler"
    def stack_clips(self, clip1, clip2=None, clip3=None, clip4=None, clip5=None):
        """
        Stack multiple CLIP pipelines into a single conditioning.
        This is useful for combining multiple CLIP models into one conditioning.
        """
        clips = [clip1, clip2, clip3, clip4, clip5]
        # filter out None values
        clips = [clip for clip in clips if clip is not None]
        if not clips:
            raise ValueError("At least one CLIP pipeline must be provided.")

        # create a stacked conditioning
        return (clips,)

class UnstackClip:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip_pipeline": ("CLIP_PIPELINE", {}),
            }
        }

    RETURN_TYPES = ("CLIP", "CLIP", "CLIP", "CLIP", "CLIP")
    RETURN_NAMES = ("clip1", "clip2", "clip3", "clip4", "clip5")
    FUNCTION = "unstack_clip"
    CATEGORY = "encoder/sampler"

    def unstack_clip(self, clip_pipeline):
        """
        Unstack a stacked CLIP pipeline into individual CLIP models.
        This is useful for extracting individual CLIP models from a stacked conditioning.
        """
        if not isinstance(clip_pipeline, list) or not all(isinstance(clip, CLIP) for clip in clip_pipeline):
            raise ValueError("Input must be a list of CLIP models.")

        # Ensure the list has exactly 5 elements, filling with None if necessary
        while len(clip_pipeline) < 5:
            clip_pipeline.append(None)

        return tuple(clip_pipeline[:5])




class LegacyEncoderSampler:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "encoder_pipe": ("ENCODER_PIPE", {}),
                "clip": ("CLIP", {}),
                "config": ("ENCODER_SAMPLER_CONFIG", {}),
            }
        }

    RETURN_TYPES = ("CONDITIONING", "DICT")
    RETURN_NAMES = ("conditioning", "debug_report")
    FUNCTION = "sample"
    CATEGORY = "encoder/sampler"

    # ------------------------------------------------------------------ #

    def sample(self, encoder_pipe, clip, config):
        # ---------- sanity ----------
        if not encoder_pipe or not clip:
            raise ValueError("Both encoder_pipe and clip must be provided.")
        device = torch.device(encoder_pipe["config"]["device"])
        encoder_pipe = dict(encoder_pipe)
        clip = clip.clone()
        cfg = config#.get("config", {})
        prompt = config.get("context_window", "a photo of a robot.")

        # ------------------------------------------------------------------ #
        # 1 · Symbolic encoder field (anchor A)
        # ------------------------------------------------------------------ #
        shift_cfg = ShiftConfig(prompt=prompt)
        a_raw = ConditioningShifter.extract_encoder_embeddings(encoder_pipe, device, shift_cfg)  # [B,T_enc,768]

        force_projection_in_dims = cfg.get("projection_dims_in", 2048)
        force_projection_in = cfg.get("force_projection_in", True)
        force_projection_out_dims = cfg.get("projection_dims_out", 1280)
        force_projection_out = cfg.get("force_projection_out", False)
        #logger.info(f"[EncoderSampler] A raw shape after projection: {a_raw.shape}")
        logger.info(f"[ForceProjection] Force projection in: {force_projection_in}, dims: {force_projection_in_dims}")
        # ------------------------------------------------------------------ #
        # 2 · CLIP perspective slices
        # ------------------------------------------------------------------ #
        clip_tokens = clip.tokenize(prompt, tokenizer_options={
            "padding": "max_length", "max_length": 77, "truncation": True
        })
        clip_cond = clip.encode_from_tokens_scheduled(clip_tokens)
        logger.info(f"[EncoderSampler] CLIP pooled_output shape: {clip_cond[0][1].get('pooled_output').shape}")  # [B,2048]
        logger.info(f"[EncoderSampler] CLIP condition shape: {clip_cond[0][0].shape}")  # [B,77,2048]
        clip_full = clip_cond[0][0]  # [B,77,2048]

        clip_l_slice = clip_full[:, :, :768]  # CLIP‑L 768‑D
        clip_g_slice = clip_full[:, :, 768:]  # CLIP‑G 1280‑D
        if force_projection_in:
            # just flat upscale the raw_a's third dimension to the defined size given in the config
            base_a = a_raw.clone()
            logger.info(f"[EncoderSampler] Projection dims: {force_projection_in_dims}, force_projection_in: {force_projection_in}")
            reference = torch.zeros(base_a.shape[0], base_a.shape[1], force_projection_in_dims, device=device)
            logger.info(f"[EncoderSampler] A raw shape before input projection: {a_raw.shape}")
            a_raw = match_project(base_a, reference, mode=cfg.get("interpolation_method_in", "linear"))
            logger.info(f"[EncoderSampler] A raw shape after input projection: {a_raw.shape}")
            base_a.detach()  # detach the base_a to avoid memory leaks

        # ------------------------------------------------------------------ #
        # 3 · Path L  (works entirely in 768‑D)
        # ------------------------------------------------------------------ #
        a_up_l = match_feature_dims(a_raw, clip_l_slice).to(device)  # feature align (no change)
        b_l = match_tokens(a_up_l, a_raw.shape[1]).to(device)  # token align → [B,T_enc,768]
        d_l = b_l - a_raw  # delta_L

        # ------------------------------------------------------------------ #
        # 4 · Path G  (encoder up‑scaled to 1280‑D)
        # ------------------------------------------------------------------ #
        a_up_g = match_feature_dims(a_raw, clip_g_slice).to(device)  # 768 → 1280
        b_g = match_tokens(clip_g_slice, a_up_g.shape[1]).to(device)  # [B,T_enc,1280]
        d_g = b_g - a_up_g  # delta_G

        # ---------------- device transfer --------------------------------- #
        to_dev = lambda x: x.to(device) if device.type == "cuda" else x.cpu()
        a_raw, a_up_g = map(to_dev, (a_raw, a_up_g))
        b_l, b_g = map(to_dev, (b_l, b_g))
        d_l, d_g = map(to_dev, (d_l, d_g))
        clip_l_slice, clip_g_slice = map(to_dev, (clip_l_slice, clip_g_slice))

        # ------------------------------------------------------------------ #
        # 5 · Walker & Stack configs (unchanged)
        # ------------------------------------------------------------------ #
        walker_cfg = FieldWalkerConfig(
            name="Alucard",
            folding_mode=cfg.get("folding", "surge-fold"),
            scheduler_mode=cfg.get("folding_scheduler", "tau"),
            t_steps=cfg.get("steps", 4),
            padding_mode=cfg.get("padding_mode", "none"),
            pooling_mode=cfg.get("pooling_mode", "average"),
            scheduler_config={
                "tau": cfg.get("tau", 1.0),
                "top_k": cfg.get("top_k", 50.0),
                "top_p": cfg.get("top_p", 0.9),
            },
            context_overrides={
                "use_alpha_mask": cfg.get("use_alpha_mask", True),
                "cosine_gate": cfg.get("cosine_similarity_gate", False),
            }
        )

        stack_cfg = ShuntStackConfig(
            sliding_window_size=cfg.get("sliding_window_size", 77),
            sliding_window_stride=cfg.get("sliding_window_stride", 33),
            context_window_size=cfg.get("context_window_size", 2048),
            override_context_window=cfg.get("override_context_window", True),
            context_window=prompt,
            max_windows=cfg.get("max_windows", 8),
        )

        integra = IntegraOrchestrator(
            IntegraConfig(
                walker_config=walker_cfg,
                stack_config=stack_cfg,
                trace_folds=False,
                enforce_projection=cfg.get("force_projection_in", False),
                enable_clip_alignment=cfg.get("cosine_similarity_gate", False),
            )
        )

        # ------------------------------------------------------------------ #
        # 6 · Walk each path separately
        # ------------------------------------------------------------------ #
        try:
            raw_folded_l, meta_l = integra.walk_encoder_field(a_raw, b_l, d_l)
            raw_folded_g, meta_g = integra.walk_encoder_field(a_up_g, b_g, d_g)
        except AlucardShapeError as e:
            raise RuntimeError(str(e))

        # resize token count back to 77, feature dims already correct
        folded_l = match_tokens(raw_folded_l, clip_l_slice.shape[1])  # [B,77,768]
        folded_g = match_tokens(raw_folded_g, clip_g_slice.shape[1])  # [B,77,1280]

        # ------------------------------------------------------------------ #
        # 7 · Optional cosine diagnostics
        # ------------------------------------------------------------------ #
        def cosine_track(f_proj, c_ref, delta):
            sched = FormulaScheduler(walker_cfg.scheduler_mode, walker_cfg.scheduler_config)
            f_n = torch.nn.functional.normalize(f_proj, dim=-1)
            c_n = torch.nn.functional.normalize(c_ref, dim=-1)
            sims, alphas = [], []
            for step in range(walker_cfg.t_steps):
                t = torch.full_like(f_proj[..., 0], step / (walker_cfg.t_steps - 1))
                alpha = sched.compute_alpha(t, f_proj, c_ref, context={"delta": delta})
                sims.append((f_n * c_n).sum(dim=-1).mean().item())
                alphas.append(alpha.mean().item())
            return {"sim": sims, "alpha": alphas}
        report = {"meta_l": meta_l, "meta_g": meta_g}
        if cfg.get("cosine_similarity_gate", False):
            report["clip_l"] = cosine_track(folded_l, clip_l_slice, d_l)
            report["clip_g"] = cosine_track(folded_g, clip_g_slice, d_g)
        # ------------------------------------------------------------------ #
        # 8 · Pack final conditioning
        # ------------------------------------------------------------------ #
        folded_full = torch.cat([folded_l, folded_g], dim=-1)  # [B,77,2048]
        logger.info(f"[EncoderSampler] Folded full shape: {folded_full.shape}")

        # Assume 77 tokens, swap in canonical START and END tokens
        B, T, D = folded_full.shape
        start_token = torch.zeros((B, 1, D), device=folded_full.device)  # [CLS]-like vector (zeros)
        end_token = torch.zeros((B, 1, D), device=folded_full.device)  # [EOS]-like vector (zeros)

        # Snip interior (e.g., tokens 1 to T-2), then prepend/append swapped tokens
        snipped = folded_full[:, 1:-1, :]  # [B, T-2, D]
        folded_patched = torch.cat([start_token, snipped, end_token], dim=1)  # [B, T, D]

        # Sanity check
        assert folded_patched.shape == folded_full.shape, f"Shape mismatch: {folded_patched.shape} vs {folded_full.shape}"

        # Extract pooled output from EOS (position 76)
        pooled_output = folded_patched[:, -1, 768:2048]  # [B, 1280]

        conditioning = [[folded_patched.double().cpu(), {"pooled_output": pooled_output.double().cpu()}]]
        logger.info(f"[EncoderSampler] Final conditioning shape: {raw_folded_g.shape} {raw_folded_g.get_device()}")
        logger.info(f"[EncoderSampler] Pooled output shape: {pooled_output.shape} {pooled_output.get_device()}")
        return conditioning, None#report


class EncoderConfigNode:
    """
    A node to configure and manage encoder models, creates a pipeline dict if none is provided.
    """
    @classmethod
    def INPUT_TYPES(cls):
        manager = get_model_manager()
        return {
            "required": {
                "max_length": ("INT", {"default": 512, "min": 1, "max": 8192}),
                "pos_embedding": (["none", "cos", "sine", "cosine_sine_product"], {"default": "none"}),
                "padding": (["max_length", "longest", "do_not_pad"], {"default": "max_length"}),
                "dtype": (["default", "float32", "float16", "bfloat16"], {"default": "default"}),
                "device": (["cpu", "cuda", "mps"], {"default": "cuda" if torch.cuda.is_available() else "cpu"}),
                "trust_remote_code": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Allow execution of remote model code. Use only with trusted sources."
                }),
            },
        }

    RETURN_TYPES = ("ENCODER_CONFIG",)
    RETURN_NAMES = ("encoder_config",)
    FUNCTION = "configure"

    def configure(self,
             context_window,
             override_context_window,
             context_window_size,
             sliding_window_size,
             sliding_window_stride,
             max_length,
             folding,
             folding_scheduler,
             pos_embedding,
             padding,
             dtype,
             device,
             trust_remote_code):
        """Prepare the configuration dict with the provided parameters."""


class EncoderLoader:
    """
    Loads T5 or BERT encoder model and prepares tokenized context window.
    Returns a complete CONDITIONING_PIPE config block for downstream interpolation, masking, and scheduling.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_name": (
                    list(ENCODER_CONFIGS.keys()),
                    {"default": "bert-beatrix-2048", "tooltip": "Select the encoder model to load."}
                ),
                "local_path": ("STRING", {"default": ""}),
                "max_length": ("INT", {"default": 512, "min": 1, "max": 8192}),
                "padding": (["max_length", "longest", "do_not_pad"], {"default": "max_length"}),
                "dtype": (["default", "float32", "float16", "bfloat16"], {"default": "default"}),
                "device": (["cpu", "cuda", "mps"], {"default": "cuda" if torch.cuda.is_available() else "cpu"}),
                "trust_remote_code": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Allow execution of remote model code. Use only with trusted sources."
                }),
            }
        }


    RETURN_TYPES = ("ENCODER_PIPE",)
    RETURN_NAMES = ("encoder_pipe",)
    FUNCTION = "load"
    CATEGORY = "adapter/testing"
    DEPRECATED = False

    def load(self,
             model_name,
             local_path,
             max_length,
             padding,
             dtype,
             device,
             trust_remote_code):

        model_manager = get_model_manager()
        device_obj = torch.device(device)

        # Determine source
        model_config = ENCODER_CONFIGS.get(model_name, {})
        model_type = model_config.get("type", "unknown")
        model_source = local_path or model_config.get("repo_name", model_name)
        model_id = f"{model_type}_{model_name}_{hashlib.sha1(model_source.encode()).hexdigest()[:10]}"

        dtype = torch.get_default_dtype() if dtype == "default" else {
            "float64": torch.float64,
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16
        }[dtype]


        # Load model/tokenizer
        result = model_manager.load_encoder_model(
            model_type=model_type,
            model_id=model_id,
            model_name_or_path=model_source,
            device=device_obj,
            dtype=dtype,
            force_reload=False,
            trust_remote_code=trust_remote_code,
            config=model_config
        )
        if not result:
            raise RuntimeError(f"Failed to load encoder model: {model_name}")
        model, tokenizer = result

        # Build config dictionary for downstream control
        config_dict = {
            "model_id": model_id,
            "model_type": model_type,
            "model_name": model_name,
            "source": model_source,
            "device": str(device),
            "trust_remote_code": trust_remote_code,
            #"context_window": context_window,
            "config": {
                #"override_context_window": override_context_window,
                #"context_window_size": context_window_size,
                #"sliding_window_size": sliding_window_size,
                #"sliding_window_stride": sliding_window_stride,
                "max_length": max_length,
                #"folding": folding,
                #"folding_scheduler": folding_scheduler,
                #"pos_embedding": pos_embedding,
                "padding": padding
            }
        }

        return ([{ # this is the encoder_pipe paradigm
            "model": model,
            "tokenizer": tokenizer,
            "config": config_dict,
        }],)

class SimpleEncoderLoader:
    """
    Loads a simple encoder model and prepares tokenized context window.
    Returns a complete CONDITIONING_PIPE config block for downstream interpolation, masking, and scheduling.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_name": (list(ENCODER_CONFIGS.keys()), {"default": "beatrix-bert-2048"}),
                "device": (["cpu", "cuda", "mps"], {"default": "cuda" if torch.cuda.is_available() else "cpu"}),
                "trust_remote_code": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Allow execution of remote model code. Use only with trusted sources."
                }),
            }
        }

    RETURN_TYPES = ("ENCODER_PIPE",)
    RETURN_NAMES = ("encoder_pipe",)
    FUNCTION = "load"
    CATEGORY = "adapter/testing"
    DEPRECATED = False

    def load(self, model_name, device, trust_remote_code):
        """Load an encoder with defaulted configuration to allow simplistic loading process.
            This still loads all the more advanced features from the advanced system, but it's all set to default.
        """

        model_manager = get_model_manager()
        device_obj = torch.device(device)
        # Determine source
        model_config: Optional[EncoderData] = ShuntUtil.get_encoder_by_model_name(model_name)
        if not model_config:
            raise ValueError(f"No configuration found for model '{model_name}'.")

        """
        class EncoderData:
            def __init__(self,
                         name: str,
                         file: str,
                         repo: str,
                         config: dict,
                         type: str = "t5"):
                self.name = name
                self.file = file
                self.repo = repo
                self.config = config
                self.type = type

        """

        # load using the model managers load_encoder_model method
        # using the correct model archetype and paradigm we
        model_type = model_config.type
        model_source = model_config.repo

        model_id = f"{model_type}_{model_name}_{hashlib.sha1(model_source.encode()).hexdigest()[:10]}"
        result = model_manager.load_encoder_model(
            model_type=model_type,
            model_id=model_id,
            model_name_or_path=model_source,
            device=device_obj,
            dtype=torch.float32,  # Default dtype
            force_reload=False,
            trust_remote_code=trust_remote_code,  # Default to not trusting remote code
            config=model_config.config if model_config.config else {}
        )
        if not result:
            raise RuntimeError(f"Failed to load encoder model: {model_name}")
        model, tokenizer = result
        # Build config dictionary for downstream control
        config_dict = {
            "model_id": model_id,
            "model_type": model_type,
            "model_name": model_name,
            "source": model_source,
            "device": str(device),
            "trust_remote_code": False,  # Default to not trusting remote code
            "config": {
                # attempt to seek the config details from the model config, then default if not found
                "max_length": model_config.config.get("max_length", 77),
                "padding": model_config.config.get("padding", "max_length"),
                "sliding_window_size": model_config.config.get("sliding_window_size", 77),
                "sliding_window_stride": model_config.config.get("sliding_window_stride", 33),
                "override_context_window": model_config.config.get("override_context_window", True),
                "context_window_size": model_config.config.get("context_window_size", 77),
                "folding": model_config.config.get("folding", "surge-fold"),
                "folding_scheduler": model_config.config.get("folding_scheduler", "none"),
                "pos_embedding": model_config.config.get("pos_embedding", "none"),
                "min_slices": model_config.config.get("min_slices", 1),
                "max_slices": model_config.config.get("max_slices", 10),
                "truncate_option": model_config.config.get("truncate_option", "fold"),
                "sliding_window": model_config.config.get("sliding_window", True),
            }
        }
        return ({
            "model": model,
            "tokenizer": tokenizer,
            "config": config_dict,
        },)


class T5LoaderTest:
    """
    Loads T5 encoder-decoder model and prepares tokenized context for adapters.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_name": (list(ENCODER_CONFIGS.keys()), {"default": "google/flan-t5-base"}),
                "local_path": ("STRING", {"default": "", "tooltip": "Local path override. If empty, use HuggingFace."}),
                "context_window": ("STRING", {"default": "a photo of a robot.", "multiline": True}),
                "override_context_window": ("BOOLEAN", {"default": True}),
                "sliding_window_size": ("INT", {"default": 512, "min": 1, "max": 2048}),
                "sliding_window_stride": ("INT", {"default": 256, "min": 1, "max": 2048}),
                "max_length": ("INT", {"default": 77, "min": 1, "max": 512}),
                "padding": (["max_length", "longest", "do_not_pad"], {"default": "max_length"}),
                "max_slices": ("INT", {"default": 10, "min": 1, "max": 100}),
                "min_slices": ("INT", {"default": 1, "min": 1, "max": 100}),
                "truncate_option": (["fold", "interpolate", "collapse", "zipper"], {"default": "fold"}),
                "device": (["cpu", "cuda", "mps"], {"default": "cuda" if torch.cuda.is_available() else "cpu"})
            }
        }

    RETURN_TYPES = ("ENCODER_PIPE",)
    RETURN_NAMES = ("encoder_pipe",)
    FUNCTION = "load"
    CATEGORY = "adapter/testing"
    DEPRECATED = True

    def load(self, model_name, local_path, context_window, override_context_window,
             sliding_window_size, sliding_window_stride, max_length, padding,
             max_slices, min_slices, truncate_option, device):
        """Load the T5 model and tokenizer, and encode a sample context window."""

        # Get model manager
        model_manager = get_model_manager()

        # Determine model source
        model_config = ENCODER_CONFIGS.get(model_name, {})
        model_source = local_path or model_config.get("repo_name", "")

        if not model_source:
            raise ValueError(f"No path found for model '{model_name}'.")

        # Create unique model ID
        model_id = f"t5_{model_name}_{hash(model_source)}"

        # Load model and tokenizer through manager
        device_obj = torch.device(device)
        result = model_manager.load_t5_model(
            model_id=model_id,
            model_name_or_path=model_source,
            device=device_obj,
            dtype=torch.float32
        )

        if not result:
            raise RuntimeError(f"Failed to load T5 model: {model_name}")

        model, tokenizer = result


        # Tokenize context
        input_ids, attention_mask = None, None
        if override_context_window:
            tokens = tokenizer(
                context_window,
                return_tensors="pt",
                padding=padding if padding != "do_not_pad" else False,
                truncation=True,
                max_length=max_length
            )
            input_ids = tokens["input_ids"].to(device_obj)
            attention_mask = tokens["attention_mask"].to(device_obj)

        return ([{
                    "model": model,
                    "tokenizer": tokenizer,
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "prompt": context_window,
                    "max_slices": max_slices,
                    "min_slices": min_slices,
                    "sliding_window_size": sliding_window_size,
                    "sliding_window_stride": sliding_window_stride,
                    "override_context_window": override_context_window,
                    "max_length": max_length,
                    "padding": padding,
                    "truncate_option": truncate_option,
                    "device": str(device),
                    "model_id": model_id  # Include for tracking
                }],)
