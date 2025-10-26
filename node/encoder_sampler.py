import logging
from typing import Optional, List, Dict, Any

import torch
from comfy.sd import CLIP
from ..model.configs import ShuntUtil

logger = logging.getLogger(__name__)

from ..sampler.formulas.folding import FoldingKernels
from ..sampler.formulas.schedules import SchedulerModes
from ..sampler.formulas.modes import FoldingPoolingTypes, FoldingPaddingTypes

from ..utils.conditioning_helper import ConditioningHelper, UsefulConditioning, ModelSlicer


import torch.nn.functional as F
from ..sampler.alucard import FieldWalkerConfig
from ..sampler.integra import IntegraConfig, IntegraOrchestrator
from ..sampler.sliding_window import ShuntStackConfig
from ..utils.conditioning_shifter import ConditioningShifter
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


class ClipPromptConfig:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                # Positive prompt
                "override_context_window": ("BOOLEAN", {"default": True}),
                "context_window": ("STRING", {
                    "default": "a photo of a robot.",
                    "multiline": True
                }),
                "context_window_size": ("INT", {
                    "default": 77, "min": 1, "max": 8192
                }),

                # Negative prompt (mirrored fields)
                "override_negative_context_window": ("BOOLEAN", {"default": False}),
                "negative_context_window": ("STRING", {
                    "default": "",
                    "multiline": True
                }),

                # Device routing
                "device": (["cpu", "cuda", "mps"], {
                    "default": "cuda" if torch.cuda.is_available() else "cpu"
                }),
            }
        }

    RETURN_TYPES = ("CORE_PROMPT_CONFIG",)
    RETURN_NAMES = ("core_prompt_cfg",)
    FUNCTION = "configure"
    CATEGORY = "encoder/config"

    def configure(
        self,
        override_context_window,
        context_window,
        context_window_size,
        override_negative_context_window,
        negative_context_window,
        device,
    ):
        return ({
            "override_context_window": override_context_window,
            "context_window": context_window,
            "context_window_size": context_window_size,
            "override_negative_context_window": override_negative_context_window,
            "negative_context_window": negative_context_window,
            "device": device,
        },)


# ─────────────────────────────────────────────────────────────────────────────
#  ClipSlidingWindowConfig – three shunt-stack controls
# ─────────────────────────────────────────────────────────────────────────────
class ClipSlidingWindowConfig:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "max_windows":              ("INT", {"default": 32, "min": 1, "max": 2048}),
                "sliding_window_size":      ("INT", {"default": 77, "min": 1, "max": 8192}),
                "sliding_window_stride":    ("INT", {"default": 77, "min": 1, "max": 2048}),
                "use_alpha_mask":           ("BOOLEAN", {"default": True}),
                "cosine_similarity_gate":   ("BOOLEAN", {"default": False}),
                "use_rose_similarity":      ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("SLIDING_WINDOW_CONFIG",)
    RETURN_NAMES = ("sliding_window_cfg",)
    FUNCTION     = "configure"
    CATEGORY     = "encoder/config"

    def configure(self,
                  max_windows,
                  sliding_window_size,
                  sliding_window_stride,
                  use_alpha_mask=True,
                  cosine_similarity_gate=False,
                  use_rose_similarity=False):
        return ({
            "max_windows":              max_windows,
            "sliding_window_size":      sliding_window_size,
            "sliding_window_stride":    sliding_window_stride,
            "use_alpha_mask":           use_alpha_mask,
            "cosine_similarity_gate":   cosine_similarity_gate,
            "use_rose_similarity":      use_rose_similarity,
        },)



# ─────────────────────────────────────────────────────────────────────────────
#  ClipFoldingStackConfig – folding / padding / pooling + steps
# ─────────────────────────────────────────────────────────────────────────────
class ClipFoldingStackConfig:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "folding_formula":      ( FoldingKernels.to_list(), {"default": FoldingKernels.shiva} ),
                "folding_scheduler":    ( SchedulerModes.to_list(), {"default": SchedulerModes.TAU} ),
                "padding_mode":         ( FoldingPaddingTypes.to_list(), {"default": FoldingPaddingTypes.SPARSE} ),
                "pooling_mode":         ( FoldingPoolingTypes.to_list(), {"default": FoldingPoolingTypes.NEAREST} ),
                "steps":                ( "INT", {"default": 10, "min": 1, "max": 100_000} ),
                "passes":               ( "INT", {"default": 10, "min": 1, "max": 100_000} ),
                "pool_frozen":          ( "BOOLEAN", {"default": False, "tooltip": "This enables returning only one frozen pool."} ),
                "conv_dim":             ( "INT", {"default": 2, "min": 2, "max": 4} ),
                "similarity_threshold": ( "FLOAT", {"default": 0.5} ),
                "tree_linkage_method":  ( ["centroid", "single", "complete"], {"default": "centroid"} ),
                "blur_sigma":           ( "FLOAT", {"default": 0.0} ),
                "thresh":               ( "FLOAT", {"default": 0.5} ),
                "bottom_k_frac":        ( "FLOAT", {"default": 0.25} ),
                "hard_bottom_k":        ( "BOOLEAN", {"default": False} ),
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
                  steps,
                  passes=1,
                  pool_frozen=False,
                  conv_dim=2,
                  similarity_threshold=0.5,
                  tree_linkage_method="centroid",
                  blur_sigma=0.0,
                  thresh=0.5,
                  bottom_k_frac=0.25,
                  hard_bottom_k=False):
        return ({
            "folding":              folding_formula,
            "folding_scheduler":    folding_scheduler,
            "padding_mode":         padding_mode,
            "pooling_mode":         pooling_mode,
            "steps":                steps,
            "passes":               passes,
            "pool_frozen":          pool_frozen,
            "conv_dim":             conv_dim,
            "similarity_threshold": similarity_threshold,
            "tree_linkage_method":  tree_linkage_method,
            "blur_sigma":           blur_sigma,
            "thresh":               thresh,
            "bottom_k_frac":        bottom_k_frac,
            "hard_bottom_k":        hard_bottom_k,
        },)



# ─────────────────────────────────────────────────────────────────────────────
#  ClipHyperConfig – top-k / top-p / temp / tau / beams
# ─────────────────────────────────────────────────────────────────────────────
class ClipHyperConfigNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "top_k":                ("FLOAT", {"default": 50.00, "min": 0.0, "max": 10_000.0}),
                "top_p":                ("FLOAT", {"default": 0.90,  "min": 0.0, "max": 1.0}),
                "temperature":          ("FLOAT", {"default": 5.00,  "min": 0.0, "max": 50.0}),
                "tau":                  ("FLOAT", {"default": 5.00,  "min": 0.0, "max": 50.0}),
                "walk_random":          ("FLOAT", {"default": 0.03, "step": 0.01, "tooltip": "Enable random walk during sampling."}),
                "walk_speed":           ("FLOAT", {"default": 3.00, "tooltip": "Speed of random walk."}),
                "surge_intensity":      ("FLOAT", {"default": 5.00, "min": 0.00, "max": 100.00, "tooltip": "Intensity of surge effect."}),
                "cascade_steps":        ("FLOAT", {"default": 4.00}),
                "shockwave_center":     ("FLOAT", {"default": 5.00}),
                "shockwave_variance":   ("FLOAT", {"default": 0.01, "step": 0.01}),
                "shiva_cool":           ("FLOAT", {"default": 4.00, "tooltip": "Enable shiva cool mode for folding."}),
                "ifrit_freq":           ("FLOAT", {"default": 4.00, "tooltip": "Frequency for Ifrit mode."}),
                "ifrit_amp":            ("FLOAT", {"default": 1.00, "tooltip": "Amplitude for Ifrit mode."}),
                "gilgamesh_axes_count": ("INT",   {"default": 3, "min": 1, "max": 10, "tooltip": "Number of axes for Gilgamesh mode."}),
                "collapse_rate":        ("FLOAT", {"default": 1.00, "tooltip": "Rate of collapse for Collapse mode."}),
                "ripple_freq":          ("FLOAT", {"default": 2.00, "tooltip": "Frequency for Ripple mode."}),
                "zeus_force":           ("FLOAT", {"default": 10.00, "tooltip": "Force of Zeus mode."}),
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
                    walk_random=0.03,
                    walk_speed=3.0,
                    surge_intensity=5.0,
                    cascade_steps=4.0,
                    shockwave_center=0.5,
                    shockwave_variance=0.01,
                    shiva_cool=4.0,
                    ifrit_freq=4.0,
                    ifrit_amp=2.0,
                    gilgamesh_axes_count=5,
                    collapse_rate=1.0,
                    ripple_freq=2.0,
                    zeus_force=10.0):
        return ({
            "top_k":      top_k,
            "top_p":      top_p,
            "temperature":temperature,
            "tau":        tau,
            "walk_random": walk_random,
            "walk_speed": walk_speed,
            "surge_intensity": surge_intensity,
            "cascade_steps": cascade_steps,
            "shockwave_center": shockwave_center,
            "shockwave_variance": shockwave_variance,
            "shiva_cool": shiva_cool,
            "ifrit_freq": ifrit_freq,
            "ifrit_amp":  ifrit_amp,
            "gilgamesh_axes_count": gilgamesh_axes_count,
            "collapse_rate": collapse_rate,
            "ripple_freq": ripple_freq,
            "zeus_force": zeus_force,
        },)



# ─────────────────────────────────────────────────────────────────────────────
#  ClipProjectionConfigNode – projection + interpolation
# ─────────────────────────────────────────────────────────────────────────────
class ClipProjectionConfigNode:
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
#  ClipExperimentalConfigNode – all remaining feature-flags
# ─────────────────────────────────────────────────────────────────────────────
class ClipExperimentalConfigNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "use_entropy_scaling":      ("BOOLEAN", {"default": False}),
                "enable_rope_spiral":       ("BOOLEAN", {"default": False}),
                "entropy_scale_center":     ("FLOAT", {"default": 0.5, "min": -10000.0, "max": 10000.0}),
                "entropy_scale_magnitude":  ("FLOAT", {"default": 5.0, "min": -10000.0, "max": 10000.0}),
                "cfg_scale":                ("FLOAT",   {"default": 1.0, "min": 0.0, "max": 1000.0}),
                "guidance_scale":           ("FLOAT",   {"default": 5.0, "min": 0.0, "max": 1000.0}),
                "pos_embedding":            (["none", "cos", "sine", "cosine"], {"default": "none"}),
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
                  use_entropy_scaling,
                  enable_rope_spiral,
                  entropy_scale_center,
                  entropy_scale_magnitude,
                  cfg_scale,
                  guidance_scale,
                  pos_embedding,
                  normalization_anchor):
        return ({
            "use_entropy_scaling":    use_entropy_scaling,
            "enable_rope_spiral":     enable_rope_spiral,
            "entropy_scale_center":   entropy_scale_center,
            "entropy_scale_magnitude": entropy_scale_magnitude,
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

    LAST_ACTIVATED_CACHE = {}
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
            (encoder, ConditioningHelper.extract_symbolic_field(encoder, prompt, device, config))
            for encoder in encoders
        ]

        # ── 2. baseline CLIP conditioning + pooled ──────────────────────────
        encoder_prompt = remove_special_tokens(prompt, remove_for_clip=True)[0] if prompt else None
        clip_tensor, clip_meta = ConditioningHelper.schedule_and_extract_clip_conditioning(
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
        conditioning = ConditioningHelper.pack_conditioning_bundle(
            folded_outputs, cfg=config, device=device, mode=mode
        )
        raw_conditioning = [[orig_clip_full, {"pooled_output": orig_pool_full}]]

        return conditioning, raw_conditioning, {}


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




from ..sampler.processor import ClipSamplerProcessor


class ClipSamplerConfigured:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "encoders": ("ENCODER_PIPE", {}),
                "clip": ("CLIP", {}),
                "prompt_override": ("STRING", {"default": "", "multiline": True}),
                "negative_prompt": ("STRING", {"default": "", "multiline": True}),
                "mode": (
                    ["sdxl", "sd1", "flux", "full_no_pool", "full_pool"],
                    {"default": "sdxl"}
                ),
            },
            "optional": {
                "prompt_config": ("CORE_PROMPT_CONFIG", {}),
                "sliding_window_cfg": ("SLIDING_WINDOW_CONFIG", {}),
                "folding_stack_cfg": ("FOLDING_STACK_CONFIG", {}),
                "scheduler_hyper_cfg": ("SCHEDULER_HYPER_CONFIG", {}),
                "projection_cfg": ("PROJECTION_CONFIG", {}),
                "experimental_cfg": ("EXPERIMENTAL_FLAGS", {}),
            }
        }


    RETURN_TYPES = (
        "CONDITIONING", "CONDITIONING", "DICT",
        "CONDITIONING", "CONDITIONING", "DICT"
    )
    RETURN_NAMES = (
        "pos_sampled_conditioning", "pos_raw_conditioning", "pos_debug",
        "neg_sampled_conditioning", "neg_raw_conditioning", "neg_debug"
    )
    FUNCTION = "sample"
    CATEGORY = "encoder/sampler"

    def sample(
        self,
        encoders,
        clip,
        prompt_override,
        negative_prompt,
        mode,
        prompt_config=None,
        sliding_window_cfg=None,
        folding_stack_cfg=None,
        scheduler_hyper_cfg=None,
        projection_cfg=None,
        experimental_cfg=None,
    ):
        # ─────────────────────────────────────────────────────────────
        # Ensure config defaults
        # ─────────────────────────────────────────────────────────────
        prompt_config = prompt_config or {
            "override_context_window": False,
            "context_window": prompt_override,
            "context_window_size": 2048,
            "override_negative_context_window": False,
            "negative_context_window": negative_prompt,
            "device": "cuda" if torch.cuda.is_available() else "cpu",
        }
        logger.info(f"[EncoderSampler] Using prompt config: {prompt_config}")


        sliding_window_cfg  = sliding_window_cfg or {
            "max_windows": 64,
            "sliding_window_size": 77,
            "sliding_window_stride": 77,
        }

        folding_stack_cfg = folding_stack_cfg or {
            "folding": "shiva",
            "folding_scheduler": "tau",
            "padding_mode": "sparse",
            "pooling_mode": "bilinear",
            "steps": 100,
            "passes": 1,
            "conv_dim": 2,
            "similarity_threshold": 0.5,
            "tree_linkage_method": "centroid",
            "blur_sigma": 0.0,
            "thresh": 0.5,
            "bottom_k_frac": 0.25,
            "hard_bottom_k": False,
            "pool_frozen": False,
        }

        scheduler_hyper_cfg = scheduler_hyper_cfg or {
            "top_k": 50.0,
            "top_p": 0.9,
            "temperature": 5.0,
            "tau": 5.0,
            "walk_random": 0.03,
            "walk_speed": 3.0,
            "surge_intensity": 5.0,
            "cascade_steps": 4.0,
            "shockwave_center": 0.5,
            "shockwave_variance": 0.01,
            "shiva_cool": 4.0,
            "ifrit_freq": 4.0,
            "ifrit_amp": 1.0,
            "gilgamesh_axes_count": 5,
            "collapse_rate": 1.0,
            "ripple_freq": 2.0,
            "zeus_force": 10.0,
        }

        projection_cfg = projection_cfg or {
            "force_projection_in": False,
            "projection_dims_in": 768,
            "interpolation_method_in": "linear",
            "force_projection_out": False,
            "projection_dims_out": 768,
            "interpolation_method_out": "linear",
        }

        experimental_cfg = experimental_cfg or {
            "use_entropy_scaling": False,
            "use_alpha_mask": False,
            "cosine_similarity_gate": False,
            "use_rose_similarity": False,
            "use_rope_resonance": False,
            "cfg_scale": 1.0,
            "guidance_scale": 5.0,
            "pos_embedding": "none",
            "normalization_anchor": "none",
        }

        # ─────────────────────────────────────────────────────────────
        # Prompt resolution logic
        # ─────────────────────────────────────────────────────────────
        pos_prompt = (
            prompt_override if prompt_config.get("override_context_window", False)
            else prompt_config.get("context_window", "")
        )


        neg_prompt = (
            negative_prompt if prompt_config.get("override_negative_context_window", False)
            else prompt_config.get("negative_context_window", "")
        )

        processor = ClipSamplerProcessor(
            prompt=pos_prompt,
            negative_prompt=neg_prompt or None,
            encoders=encoders,
            clip=clip,
            prompt_config=prompt_config,
            sliding_window_cfg=sliding_window_cfg,
            folding_cfg=folding_stack_cfg,
            scheduler_hyper_cfg=scheduler_hyper_cfg,
            projection_cfg=projection_cfg,
            experimental_cfg=experimental_cfg,
            mode=mode,
        )

        return processor.run()




import torch
from typing import Optional, Tuple
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import logging

from ..model.model_manager import get_model_manager

class T5SummarizeCaption:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"default": "", "multiline": True}),
                "encoder_pipe": ("ENCODER_PIPE", {}),
            },
            "optional": {
                "command": ("STRING", {"default": "Compress and keep visual clues and object positions:"}),
                "max_tokens": ("INT", {"default": 512, "min": 8, "max": 512}),
                "min_length": ("INT", {"default": 10, "min": 1, "max": 512}),
                "do_sample": ("BOOLEAN", {"default": False, "tooltip": "Use sampling instead of greedy decoding."}),
                "regen_on_short": ("BOOLEAN", {"default": True}),
                "max_repeats": ("INT", {"default": 3, "min": 1, "max": 10, "tooltip": "Maximum number of times to repeat the generation."}),
                "folding_repeats": ("BOOLEAN", {"default": False, "tooltip": "Use folding to repeat the generation."}),
                "seed": ("INT", {"default": -1, "min": -1, "max": 2**32-1}),
                "early_stopping": ( "BOOLEAN", {"default": True, "tooltip": "Stop generation early if the model predicts the end token."}),
                "length_penalty": ( "FLOAT", {"default": 2.0, "min": 0.0, "max": 10.0, "tooltip": "Penalty for longer sequences."}),
                "no_repeat_ngram_size": ( "INT", {"default": 3, "min": 1, "max": 10, "tooltip": "Prevent repetition of n-grams of this size."}),
                "num_beams": ( "INT", {"default": 4, "min": 1, "max": 10, "tooltip": "Number of beams for beam search."}),
                "device": (["cpu", "cuda", "mps"], { "default": "cuda" if torch.cuda.is_available() else "cpu" }),
                "offload_device": (["cpu", "cuda", "mps"], { "default": "cpu" }),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("summary",)
    FUNCTION = "summarize"
    CATEGORY = "ABS/Captioning"

    def summarize(self,
                  prompt: str,
                  encoder_pipe,
                  command: str = "Compress and keep visual clues and object positions:",
                  max_tokens: int = 512,
                  min_length: int = 10,
                  do_sample: bool = False,
                  regen_on_short: bool = True,
                  max_repeats: int = 3,
                  folding_repeats: bool = False,
                  seed: int = -1,
                  early_stopping = True,
                  length_penalty = 2.0,
                  no_repeat_ngram_size = 3,
                  num_beams = 4,
                  device = "cuda",
                  offload_device="cpu") -> Tuple[str]:

        device_obj = torch.device(device)
        model_manager = get_model_manager()

        model_name = encoder_pipe[0].get("config", {}).get("source", "unknown")
        logger.info(f"[T5SummarizeCaption] Using model: {model_name}")

        tokenizer = encoder_pipe[0].get("tokenizer", None)
        model = encoder_pipe[0].get("model", None)
        model.to(device_obj)

        tasks = encoder_pipe[0].get("config", {}).get("model_config", {}).get("config", {}).get("task_specific_params", {})
        logger.info(f"[T5SummarizeCaption] Task-specific parameters: {tasks}")
        task = tasks.get(command.strip().replace(":", ""), {})
        logger.info(f"[T5SummarizeCaption] Using command: {command.strip()}")
        if task is not None:
            logger.info(f"[T5SummarizeCaption] Found task-specific parameters for command: {command.strip()}")
            logger.info(f"[T5SummarizeCaption] Task parameters: {task}")

        prompt_full = f"{command.strip()} {prompt.strip()}"
        if seed != -1:
            torch.manual_seed(seed)

        input_ids = tokenizer(prompt_full, return_tensors="pt", truncation=True).input_ids.to(device_obj)

        summary = ""
        retries = 0

        while retries < max_repeats:
            output_ids = model.generate(
                input_ids,
                max_length=task.get("max_length", max_tokens),
                min_length=task.get("min_length", min_length),
                early_stopping=task.get("early_stopping", early_stopping),
                length_penalty=task.get("length_penalty", length_penalty),
                no_repeat_ngram_size=task.get("no_repeat_ngram_size", no_repeat_ngram_size),
                num_beams=task.get("num_beams", num_beams),
                do_sample=task.get("do_sample", do_sample),
                num_return_sequences=1
            )[0]

            summary = tokenizer.decode(output_ids, skip_special_tokens=True).strip()
            logger.info(f"[T5SummarizeCaption] Generated summary: {summary}")
            logger.info(f"[T5SummarizeCaption]")

            if not regen_on_short:
                break
            if len(summary.split()) >= min_length:
                break

            retries += 1
            logger.info(f"[T5SummarizeCaption] Retry {retries}/{max_repeats} — Too short: {len(summary.split())} tokens")

            # If using folding mode, concatenate original prompt
            if folding_repeats:
                length_penalty = length_penalty * 0.9  # Adjust length penalty for folding
                no_repeat_ngram_size = no_repeat_ngram_size + 1  # Increase n-gram size to avoid repetition
                num_beams = num_beams + 1  # Increase beams to explore more options
                logger.info(f"[T5SummarizeCaption] Using folding mode, concatenating original prompt {prompt}")
                summary = summary.replace(command.strip(), "").strip()
                logger.info(f"[T5SummarizeCaption] Summary after folding: {summary}")
                prompt_full = f"{command.strip()} {summary} {prompt*retries}"
                logger.info(f"[T5SummarizeCaption] New prompt for next iteration: {prompt_full}")
                input_ids = tokenizer(prompt_full, return_tensors="pt", truncation=False).input_ids.to(device_obj)

        model_manager.unload_model(model_name)
        model_manager.clear_all()

        return (summary,)

