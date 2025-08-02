import logging
from typing import Optional, List, Dict, Any

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
from ..sampler.formulas.modes import FoldingPoolingTypes, FoldingPaddingTypes

from ..utils.conditioning_helper import ConditioningHelper, UsefulConditioning, ModelSlicer


import torch.nn.functional as F
from ..sampler.alucard import FieldWalkerConfig
from ..sampler.integra import IntegraConfig, IntegraOrchestrator
from ..sampler.sliding_window import ShuntStackConfig
from ..utils.conditioning_shifter import ConditioningShifter, ShiftConfig
from ..sampler.formulas.schedules import FormulaScheduler
from ..utils.alignment import match_feature_dims, match_tokens, match_project
from ..sampler.alucard_exceptions import AlucardShapeError


# ─────────────────────────────────────────────────────────────────────────────
#  CorePromptConfig  – prompt & device
# ─────────────────────────────────────────────────────────────────────────────



BEATRIX_SPECIAL_TOKENS_AND_SHUNTS = [
    "<subject>","<subject1>","<subject2>","<pose>","<emotion>","<surface>","<lighting>","<material>","<accessory>",
    "<footwear>", "<upper_body_clothing>","<hair_style>","<hair_length>","<headwear>","<texture>","<pattern>","<grid>",
    "<zone>","<offset>","<object_left>","<object_right>","<relation>","<intent>","<style>","<fabric>","<jewelry>",
    "[SHUNT_1000000]","[SHUNT_1000001]","[SHUNT_1000002]","[SHUNT_1000003]","[SHUNT_1000004]",
    "[SHUNT_1000005]","[SHUNT_1000006]","[SHUNT_1000007]","[SHUNT_1000008]","[SHUNT_1000009]","[SHUNT_1000010]",
    "[SHUNT_1000011]","[SHUNT_1000012]","[SHUNT_1000013]","[SHUNT_1000014]","[SHUNT_1000015]","[SHUNT_1000016]",
    "[SHUNT_1000017]","[SHUNT_1000018]","[SHUNT_1000019]","[SHUNT_1000020]","[SHUNT_1000021]","[SHUNT_1000022]",
    "[SHUNT_1000023]","[SHUNT_1000024]","[SHUNT_1000025]","<EOF>","<START>","<END>","<PAD>","<MASK>","[CLS]","[SEP]","[PAD]",
    "<|startofimage|>", "<|endofimage|>", "<|startofvideo|>", "<|endofvideo|>",
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
    "</w>", "<w>", "<|startoftext|>", "<|endoftext|>",
]


def remove_special_tokens(prompt, remove_for_clip=False):
    """
    Removes known special tokens from the prompt.
    This is useful for cleaning up prompts before processing.
    """
    if remove_for_clip:
        # we want to remove everything that is not a special token for CLIP, so we want the special tokens
        tokens = BEATRIX_SPECIAL_TOKENS_AND_SHUNTS.copy()
    else:
        tokens = CLIP_ONLY_TOKENS.copy()
    # find the array of tokens within the prompt to replace
    for token in tokens:
        if token in prompt:
            # remove the prompt
            prompt = prompt.replace(token, "")
    return (prompt,)

class CorePromptConfig:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "override_context_window": ("BOOLEAN", {"default": True}),
                "context_window": ("STRING", {
                    "default": "a photo of a robot.",
                    "multiline": True
                }),
                "context_window_size": ("INT", {"default": 2048, "min": 77, "max": 8192}),
                "device": (["cpu", "cuda", "mps"], {
                    "default": "cuda" if torch.cuda.is_available() else "cpu"
                }),
            }
        }

    RETURN_TYPES = ("CORE_PROMPT_CONFIG",)
    RETURN_NAMES = ("core_prompt_cfg",)
    FUNCTION     = "configure"
    CATEGORY     = "encoder/config"

    def configure(self,
                  override_context_window,
                  context_window,
                  context_window_size,
                  device):
        return ({
            "override_context_window": override_context_window,
            "context_window":          context_window,
            "context_window_size":     context_window_size,
            "device":                  device,
        },)



# ─────────────────────────────────────────────────────────────────────────────
#  SlidingWindowConfig – three shunt-stack controls
# ─────────────────────────────────────────────────────────────────────────────
class SlidingWindowConfig:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "max_windows":          ("INT", {"default": 64, "min": 1, "max": 2048}),
                "sliding_window_size":  ("INT", {"default": 77, "min": 1, "max": 8192}),
                "sliding_window_stride":("INT", {"default": 77, "min": 1, "max": 2048}),
            }
        }

    RETURN_TYPES = ("SLIDING_WINDOW_CONFIG",)
    RETURN_NAMES = ("sliding_window_cfg",)
    FUNCTION     = "configure"
    CATEGORY     = "encoder/config"

    def configure(self,
                  max_windows,
                  sliding_window_size,
                  sliding_window_stride):
        return ({
            "max_windows":          max_windows,
            "sliding_window_size":  sliding_window_size,
            "sliding_window_stride":sliding_window_stride,
        },)



# ─────────────────────────────────────────────────────────────────────────────
#  FoldingStackConfig – folding / padding / pooling + steps
# ─────────────────────────────────────────────────────────────────────────────
class FoldingStackConfig:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "folding_formula":   (FoldingKernels.to_list(), {"default": FoldingKernels.shiva}),
                "folding_scheduler": (SchedulerModes.to_list(), {"default": SchedulerModes.TAU}),
                "padding_mode":      (FoldingPaddingTypes.to_list(), {"default": FoldingPaddingTypes.SPARSE}),
                "pooling_mode":      (FoldingPoolingTypes.to_list(), {"default": FoldingPoolingTypes.BILINEAR}),
                "steps":             ("INT", {"default": 100, "min": 1, "max": 100_000}),
            }
        }

    RETURN_TYPES = ("FOLDING_STACK_CONFIG",)
    RETURN_NAMES = ("folding_stack_cfg",)
    FUNCTION     = "configure"
    CATEGORY     = "encoder/config"

    def configure(self,
                  folding_formula,
                  folding_scheduler,
                  padding_mode,
                  pooling_mode,
                  steps):
        return ({
            "folding":           folding_formula,
            "folding_scheduler": folding_scheduler,
            "padding_mode":      padding_mode,
            "pooling_mode":      pooling_mode,
            "steps":             steps,
        },)



# ─────────────────────────────────────────────────────────────────────────────
#  SchedulerHyperConfig – top-k / top-p / temp / tau / beams
# ─────────────────────────────────────────────────────────────────────────────
class SchedulerHyperConfig:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "top_k":      ("FLOAT", {"default": 50.0, "min": 0.0, "max": 10_000.0}),
                "top_p":      ("FLOAT", {"default": 0.9,  "min": 0.0, "max": 1.0}),
                "temperature":("FLOAT", {"default": 5.0,  "min": 0.0, "max": 50.0}),
                "tau":        ("FLOAT", {"default": 5.0,  "min": 0.0, "max": 50.0}),
                "beams":      ("INT",   {"default": 4,    "min": 1,   "max": 32}),
            }
        }

    RETURN_TYPES = ("SCHEDULER_HYPER_CONFIG",)
    RETURN_NAMES = ("scheduler_hyper_cfg",)
    FUNCTION     = "configure"
    CATEGORY     = "encoder/config"

    def configure(self,
                  top_k,
                  top_p,
                  temperature,
                  tau,
                  beams):
        return ({
            "top_k":      top_k,
            "top_p":      top_p,
            "temperature":temperature,
            "tau":        tau,
            "beams":      beams,
        },)



# ─────────────────────────────────────────────────────────────────────────────
#  ProjectionConfig – projection + interpolation
# ─────────────────────────────────────────────────────────────────────────────
class ProjectionConfig:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "force_projection_in":   ("BOOLEAN", {"default": False}),
                "projection_dims_in":    ("INT", {"default": 768, "min": 1, "max": 8192}),
                "interpolation_method_in": (["linear"], {"default": "linear"}),
                "force_projection_out":  ("BOOLEAN", {"default": False}),
                "projection_dims_out":   ("INT", {"default": 768, "min": 1, "max": 8192}),
                "interpolation_method_out":(["linear"], {"default": "linear"}),
            }
        }

    RETURN_TYPES = ("PROJECTION_CONFIG",)
    RETURN_NAMES = ("projection_cfg",)
    FUNCTION     = "configure"
    CATEGORY     = "encoder/config"

    def configure(self,
                  force_projection_in,
                  projection_dims_in,
                  interpolation_method_in,
                  force_projection_out,
                  projection_dims_out,
                  interpolation_method_out):
        return ({
            "force_projection_in":   force_projection_in,
            "projection_dims_in":    projection_dims_in,
            "interpolation_method_in":  interpolation_method_in,
            "force_projection_out":  force_projection_out,
            "projection_dims_out":   projection_dims_out,
            "interpolation_method_out": interpolation_method_out,
        },)



# ─────────────────────────────────────────────────────────────────────────────
#  ExperimentalFlags – all remaining feature-flags
# ─────────────────────────────────────────────────────────────────────────────
class ExperimentalFlags:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "use_alpha_mask":        ("BOOLEAN", {"default": True}),
                "cosine_similarity_gate":("BOOLEAN", {"default": False}),
                "use_rose_similarity":   ("BOOLEAN", {"default": False}),
                "use_rope_resonance":    ("BOOLEAN", {"default": False}),
                "cfg_scale":             ("FLOAT",   {"default": 1.0, "min": 0.0, "max": 100.0}),
                "guidance_scale":        ("FLOAT",   {"default": 5.0, "min": 0.0, "max": 100.0}),
                "pos_embedding":         (["none", "cos", "sine", "cosine"], {"default": "none"}),
                "normalization_anchor":  ([
                    "none", "l2", "l1", "heun", "surge", "sigma",
                    "delta", "gate", "bong"
                ], {"default": "none"}),
            }
        }

    RETURN_TYPES = ("EXPERIMENTAL_FLAGS",)
    RETURN_NAMES = ("experimental_cfg",)
    FUNCTION     = "configure"
    CATEGORY     = "encoder/config"

    def configure(self,
                  use_alpha_mask,
                  cosine_similarity_gate,
                  use_rose_similarity,
                  use_rope_resonance,
                  cfg_scale,
                  guidance_scale,
                  pos_embedding,
                  normalization_anchor):
        return ({
            "use_alpha_mask":         use_alpha_mask,
            "cosine_similarity_gate": cosine_similarity_gate,
            "use_rose_similarity":    use_rose_similarity,
            "use_rope_resonance":     use_rope_resonance,
            "cfg_scale":              cfg_scale,
            "guidance_scale":         guidance_scale,
            "pos_embedding":          pos_embedding,
            "normalization_anchor":   normalization_anchor,
        },)


class TextEncoderSamplerConfig:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "use_bert_wildcards": ("BOOLEAN", {"default": False, "tooltip": "Use BERT for wildcarding the prompt."}),
                "symbolic_logic": ("BOOLEAN", {"default": True, "tooltip": "Use symbolic logic for encoding."}),
            }
        }

    RETURN_TYPES = ("TEXT_ENCODER_SAMPLER_CONFIG",)
    RETURN_NAMES = ("text_encoder_sampler_cfg",)
    FUNCTION     = "configure"
    CATEGORY     = "encoder/config"

    def configure(self, clip_type, model_type, device):
        return ({
            "clip_type": clip_type,
            "model_type": model_type,
            "device": device,
        },)

class ClipTextEncodeSampled:
    # The node that canonically converts text to CLIP embeddings as the original comfyui system has trained people to do.

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip": ("CLIP", {}),
                "prompt": ("STRING", {"default": "", "multiline": True}),
                "device": (["cpu", "cuda", "mps"], {"default": "cuda" if torch.cuda.is_available() else "cpu"}),
            }
        }



ENCODER_SUPPORTED_CLIP_TYPES = [
    "clip_l", "clip_g", "clip_h", "clip_vision", "t5", "llama", "t5_unchained"
    # these are the main supported types for the internal CLIP structure, there will be more.
]

ENCODER_SUPPORTED_MODEL_TYPES = [
    "sdxl", "sd1", "flux", "hidream", "full_no_pool", "full_pool",
    # currently only supports these four modes, but WILL be extended in the future.
]

class ClipSampler:
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

    def __prepare_conditioning(
        self, prompt, encoders, clip, device, mode,
        config=None, reckless_config=None, clip_gate_config=None
    ):
        logger.info(f"[EncoderSampler] Sampling with mode: {mode} on device {device}")

        # ── 1. raw encoder embeddings ───────────────────────────────────────
        a_raws = [
            (encoder, self.__extract_symbolic(encoder, prompt, device, config))
            for encoder in encoders
        ]

        # ── 2. baseline CLIP conditioning + pooled ──────────────────────────
        encoder_prompt = remove_special_tokens(prompt, remove_for_clip=True)[0] if prompt else None
        clip_tensor, clip_meta = self.__schedule_and_extract_conds(
            clip, encoder_prompt, device, mode=mode
        )
        orig_clip_full = clip_tensor.clone()
        orig_pool_full = clip_meta.get("pooled_output")            # ← no bool-test
        if orig_pool_full is not None:
            orig_pool_full = orig_pool_full.clone()

        # ── 3. symbolic slicing via ModelSlicer ─────────────────────────────
        base_uc = UsefulConditioning(
            [[clip_tensor.clone(), {"pooled_output": orig_pool_full}]]
        )
        clip_uc = ModelSlicer.slice(base_uc, model_type=mode, device=device)

        # ── 4. fold each slice through Integra ──────────────────────────────
        folded_outputs: Dict[str, List[torch.Tensor]] = {}
        for encoder, a_raw in a_raws:
            for idx in range(len(clip_uc)):
                slice_tensor = clip_uc.get_tensor(idx)
                key = clip_uc.get_field(idx, "slicer_info")["key"]

                folded = self._run_integra(a_raw, key, slice_tensor, config, device, encoder)
                folded_outputs.setdefault(key, []).append(folded)

        if not folded_outputs:
            raise RuntimeError("No encoder-condition slices folded successfully.")
        logger.info(f"[EncoderSampler] Folded outputs: {list(folded_outputs.keys())}")

        # ── 5. assemble final conditioning bundle ───────────────────────────
        conditioning = self._pack_conditioning_from_registry(
            folded_outputs, cfg=config, device=device, mode=mode
        )
        raw_conditioning = [[orig_clip_full, {"pooled_output": orig_pool_full}]]

        return conditioning, raw_conditioning, {}

    def __extract_symbolic(self, pipe, prompt, device, config=None):
        encoder_type = pipe.get("type", "unknown").lower()
        clip_like = encoder_type in ENCODER_SUPPORTED_CLIP_TYPES
        # a small set of tests to run, so far showing promise

        if clip_like:
            prepared_prompt = remove_special_tokens(prompt, remove_for_clip=False)[0] if prompt else None
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
            shift_cfg = ShiftConfig(prompt=prompt, **config if config is dict else {})
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


    def _run_integra(self, a_raw, cond_name, clip_slice, cfg, device, encoder):
        if a_raw.size(-1) != clip_slice.size(-1) or cfg.get("force_projection_in", False):
            a_proj = match_project(a_raw, clip_slice, mode=cfg.get("interpolation_method_in", "linear"))
        else:
            a_proj = a_raw

        a_feat = match_feature_dims(a_proj, clip_slice)
        b = match_tokens(clip_slice, a_feat.shape[1])
        delta = b - a_proj


        integra = self._build_integra(encoder, cfg)
        try:
            raw_folded, _ = integra.walk_encoder_field(
                a_feat, b, delta,
            )
        except AlucardShapeError as e:
            raise RuntimeError(f"[EncoderSampler] Shape error: {e}") from e

        return match_tokens(raw_folded, clip_slice.shape[1]).to(device)



    def _build_integra(self, encoder, cfg):
        #logger.info(f"[EncoderSampler] Building Integra with config: {cfg}")
        #logger.info(f"[EncoderSampler] Encoder type: {encoder.get("config", {}).keys()}")
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
                "encoder_name": encoder.get("config", {}).get("model_name", "unknown").lower(),
                "encoder_type": encoder.get("config", {}).get("model_type", "unknown").lower(),
                "use_alpha_mask": cfg.get("use_alpha_mask", True),
                "cosine_gate": cfg.get("cosine_similarity_gate", False),
                "use_rose_similarity": cfg.get("use_rose_similarity", True),

                "enable_rope_spiral": cfg.get("rope_phase_offsets", False),
                "rope_phase_offsets": cfg.get("rope_phase_offsets", None),
                "spiral_probe_token": cfg.get("spiral_probe_token", None),
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

