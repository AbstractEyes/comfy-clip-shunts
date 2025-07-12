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
        embeddings = ConditioningShifter.extract_encoder_embeddings(encoder, device=device, config=encoder_config)

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
                "override_context_window": ("BOOLEAN", {"default": True}),
                "context_window": ("STRING", {
                    "default": "a photo of a robot.",
                    "multiline": True
                }),
                "steps": ("INT", {"default": 250, "min": 1, "max": 100000}),
                "cfg_scale": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 100.0}),
                "guidance_scale": ("FLOAT", {"default": 5, "min": 0.0, "max": 100.0}),

                "folding": (FoldingKernels.to_list(), {"default": FoldingKernels.a_walk, "tooltip": "Folding mode to use for the encoder."}),

                # implemented for testing not fully functional
                "folding_scheduler": (SchedulerModes.to_list(), {"default": SchedulerModes.TAU, "tooltip": "Folding scheduler to use for the encoder."}),

                #implemented for testing not fully functional
                "padding_mode": (FoldingPaddingTypes.to_list(), {"default": FoldingPaddingTypes.INTERPOLATE, "tooltip": "Padding mode to use for the encoder."}),
                #implemented for testing not fully functional
                "pooling_mode": (FoldingPoolingTypes.to_list(),{"default": FoldingPoolingTypes.AVERAGE, "tooltip": "Pooling mode to use for the encoder."}),
                #doesn't work correctly yet
                "use_alpha_mask": ("BOOLEAN", {"default": True}),
                #todo
                "cosine_similarity_gate": ("BOOLEAN", {"default": False}),

                #todo
                "pos_embedding": (["none", "cos", "sine", "cosine"], {"default": "none"}),
                "normalization_anchor": (["none", "l2", "l1", "heun", "surge", "sigma", "delta", "gate", "bong"], {"default": "none"}),

                #doesn't work correctly
                "top_k": ("FLOAT", {"default": 50.0, "min": 0.0, "max": 10000.0}),
                #todo
                "top_p": ("FLOAT", {"default": 0.9, "min": 0.0, "max": 1.0}),
                #todo
                "temperature": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0}),
                #todo
                "tau": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0}),
                #works, i think
                "beams": ("INT", {"default": 4, "min": 1, "max": 32}),

                #works
                "max_windows": ("INT", {"default": 32, "min": 1, "max": 2048}),
                "context_window_size": ("INT", {"default": 2048, "min": 77, "max": 8192}),
                "sliding_window_size": ("INT", {"default": 128, "min": 1, "max": 8192}),
                "sliding_window_stride": ("INT", {"default": 16, "min": 1, "max": 2048}),

                #todo
                "force_projection_in": ("BOOLEAN", {"default": False, "tooltip": "Force projection of context window to model's max length."}),
                "projection_dims_in": ("INT", {"default": 768, "min": 1, "max": 8192}),
                "interpolation_method_in": (["lerp", "slerp", "cosine", "sine", "linear", "mixed"], {"default": "slerp", "tooltip": "Method to use for interpolating projections."}),
                "force_projection_out": ("BOOLEAN", {"default": False, "tooltip": "Force projection of model output to context window size."}),
                "projection_dims_out": ("INT", {"default": 768, "min": 1, "max": 8192}),
                "interpolation_method_out": (["lerp", "slerp", "cosine", "sine", "linear", "mixed"], {"default": "slerp", "tooltip": "Method to use for interpolating model output projections."}),
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
                    interpolation_method_out):
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
            "interpolation_method_out": interpolation_method_out
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
    FUNCTION = "remove_special_tokens"
    CATEGORY = "encoder/special_tokens"
    def remove_special_tokens(self, prompt):
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

class EncoderSampler:
    CLIP_VARIATIONS = ["clip_l", "clip_g"]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                # now accepts a list of encoder pipelines
                "encoders": ("ENCODER_PIPE", {}),
                "clip":       ("CLIP", {}),
                "config":     ("ENCODER_SAMPLER_CONFIG", {}),
            }
        }

    RETURN_TYPES = ("CONDITIONING", "DICT")
    RETURN_NAMES = ("conditioning", "debug_report")
    FUNCTION = "sample"
    CATEGORY = "encoder/sampler"


    def sample(
        self,
        encoders: List[Dict[str,Any]],
        clip,
        config: Dict[str,Any],
    ):
        device = torch.device(config.get("device", "cpu"))
        prompt = config["context_window"]

        # 1) Extract symbolic embeddings from each encoder in the pipeline
        a_raws = [
            self._extract_symbolic(pipe, prompt, device)
            for pipe in encoders
        ]

        # 2) Run the CLIP model once and slice into your two variants
        clip_full   = self._extract_clip(clip, prompt, device)     # [B, T_clip, D_clip]
        clip_slices = self._slice_clip(clip_full)                  # {"clip_l":…, "clip_g":…}

        # 3) For every (encoder × clip-variant) pair, run the fold/walk
        folded_list = []
        for a_raw in a_raws:
            for var in self.CLIP_VARIATIONS:
                folded = self._run_path(a_raw, clip_slices[var], config, device)
                folded_list.append(folded)

        # 4) Concatenate all outputs along the feature dimension
        folded_full = torch.cat(folded_list, dim=-1)  # [B, T_clip, sum(D_i)]

        # 5) Package into ComfyUI conditioning format
        conditioning = self._pack_conditioning(folded_full, config, device)
        debug_report = None  # or collect meta if you like
        return conditioning, debug_report


    # ————— Helpers ————— #

    def _extract_symbolic(self, pipe: Dict, prompt: str, device: torch.device) -> torch.Tensor:
        """
        Run your ConditioningShifter against one encoder_pipe,
        returning a [B, T_enc, D_enc] tensor on `device`.
        """
        # move the model to the correct device
        if "model" in pipe:
            pipe["model"].to(device)
        shift_cfg = ShiftConfig(prompt=prompt)
        a = ConditioningShifter.extract_encoder_embeddings(pipe, device, shift_cfg)
        return a.to(device)


    def _extract_clip(self, clip, prompt: str, device: torch.device) -> torch.Tensor:
        """
        Tokenize + scheduled-encode via CLIP, returning [B, T_clip, D_clip].
        """
        tokens = clip.tokenize(prompt,
            tokenizer_options={"padding":"max_length","max_length":77,"truncation":True}
        )
        cond = clip.encode_from_tokens_scheduled(tokens)[0][0]  # [B,77,2048]
        return cond.to(device)


    def _slice_clip(self, clip_full: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Hard-coded for now: split into CLIP-L (first 768 dims)
        and CLIP-G (last 1280 dims).
        """
        return {
            "clip_l": clip_full[:, :, :768],
            "clip_g": clip_full[:, :, 768:]
        }

    def _run_path(
            self,
            a_raw: torch.Tensor,
            clip_slice: torch.Tensor,
            cfg: Dict[str, Any],
            device: torch.device
    ) -> torch.Tensor:
        """
        1) Ensure feature dimensions match between a_raw and clip_slice
        2) Align tokens
        3) Build and run IntegraOrchestrator
        4) Re-tokenize back to clip token count
        """
        # 1) Ensure feature dims match
        if a_raw.size(-1) != clip_slice.size(-1):
            a_proj = match_project(
                a_raw,
                clip_slice,
                mode=cfg.get("interpolation_method_in", "linear")
            )
        elif cfg.get("force_projection_in", False):
            a_proj = match_project(
                a_raw,
                clip_slice,
                mode=cfg.get("interpolation_method_in", "linear")
            )
        else:
            a_proj = a_raw

        # 2) Align sequence and feature dimensions
        a_feat = match_feature_dims(a_proj, clip_slice)
        b = match_tokens(clip_slice, a_feat.shape[1])
        delta = b - a_proj

        # 3) Create the Integra orchestrator
        integra = self._build_integra(cfg)

        # 4) Perform the folding walk
        try:
            raw_folded, _ = integra.walk_encoder_field(a_feat, b, delta)
        except AlucardShapeError as e:
            raise RuntimeError(f"[EncoderSampler] Shape error: {e}") from e

        # 5) Resize back to original CLIP token count
        folded = match_tokens(raw_folded, clip_slice.shape[1])
        return folded.to(device)

    def _build_integra(self, cfg: Dict[str,Any]) -> IntegraOrchestrator:
        """Map `cfg` into your FieldWalkerConfig + ShuntStackConfig → IntegraOrchestrator."""
        walker_cfg = FieldWalkerConfig(
            name=cfg.get("name","Alucard"),
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
                "use_alpha_mask":         cfg["use_alpha_mask"],
                "cosine_gate":            cfg["cosine_similarity_gate"],
            }
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
            stack_config= stack_cfg,
            trace_folds=False,
            enforce_projection=cfg["force_projection_in"],
            enable_clip_alignment=cfg["cosine_similarity_gate"],
        ))


    def _pack_conditioning(
        self,
        folded: torch.Tensor,
        cfg: Dict[str,Any],
        device: torch.device
    ):
        """
        Swap in CLS/EOS if needed, then
        return [[folded.cpu(), {"pooled_output": ...}]]
        """
        B, T, D = folded.shape
        start = torch.zeros(B,1,D, device=device)
        end   = torch.zeros(B,1,D, device=device)
        body  = folded[:,1:-1,:]
        patched = torch.cat([start, body, end], dim=1)

        pooled = patched[:, -1, 768:2048]  # for example
        return [[patched.cpu(), {"pooled_output": pooled.cpu()}]]



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

