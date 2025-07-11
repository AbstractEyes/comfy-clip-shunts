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
    ConditioningSchedulerTypes,
    CONDITIONING_SCHEDULERS,
    FoldingPoolingTypes,
    FOLDING_POOLING_TYPES,
    FoldingPaddingTypes,
    FOLDING_PADDING_TYPES,
    FoldingTypes,
    FOLDING_MODES
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
                "steps": ("INT", {"default": 4, "min": 1, "max": 1000}),
                "cfg_scale": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 100.0}),
                "guidance_scale": ("FLOAT", {"default": 5, "min": 0.0, "max": 100.0}),

                "folding": ([
                                "zeus", "helios", "surge", "surge-fold", "fold", "interpolate",
                                "collapse", "zipper", "concat-flatten", "cascade", "ripple",
                                "slerp", "slip",  # <<< new
                                "hard_truncate", "soft_truncate", "truncate", "none"
                            ], {"default": "slerp"}),

                "folding_scheduler": ([
                                          "none", "tau", "top_k", "top_20k", "top_50k",
                                          "cosine", "cascade", "cos", "sine",
                                          "shockwave", "pulse", "wave"
                                      ], {"default": "none"}),

                "padding_mode": (FOLDING_PADDING_TYPES,
                                 {"default": FOLDING_PADDING_TYPES[0]}),
                "pooling_mode": (FOLDING_POOLING_TYPES,
                                 {"default": FOLDING_POOLING_TYPES[0]}),
                "use_alpha_mask": ("BOOLEAN", {"default": True}),
                "cosine_similarity_gate": ("BOOLEAN", {"default": False}),

                "pos_embedding": (["none", "cos", "sine", "cosine"], {"default": "cos"}),
                "normalization_anchor": (["none", "l2", "l1", "heun", "surge", "sigma", "delta", "gate", "bong"], {"default": "surge"}),

                "top_k": ("FLOAT", {"default": 50.0, "min": 0.0, "max": 10000.0}),
                "top_p": ("FLOAT", {"default": 0.9, "min": 0.0, "max": 1.0}),
                "temperature": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0}),
                "tau": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0}),
                "beams": ("INT", {"default": 4, "min": 1, "max": 32}),
                "max_windows": ("INT", {"default": 4, "min": 1, "max": 256}),
                "context_window_size": ("INT", {"default": 2048, "min": 77, "max": 8192}),
                "sliding_window_size": ("INT", {"default": 77, "min": 1, "max": 2048}),
                "sliding_window_stride": ("INT", {"default": 33, "min": 1, "max": 2048}),

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

def match_project(tensor: torch.Tensor, reference: torch.Tensor, mode: str = "linear") -> torch.Tensor:
    if tensor.ndim == 4:
        tensor = tensor.squeeze(1)
    if reference.ndim == 4:
        reference = reference.squeeze(1)
    if tensor.shape[-1] == reference.shape[-1]:
        return tensor
    return F.interpolate(tensor.transpose(1, 2), size=reference.shape[-1], mode=mode, align_corners=False).transpose(1, 2)

from ..utils.alignment import match_feature_dims, match_tokens
from ..sampler.alucard_exceptions import AlucardShapeError

class EncoderSampler:
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
        cfg = config#.get("config", {})
        prompt = config.get("context_window", "a photo of a robot.")

        # ------------------------------------------------------------------ #
        # 1 · Symbolic encoder field (anchor A)
        # ------------------------------------------------------------------ #
        shift_cfg = ShiftConfig(prompt=prompt)
        a_raw = ConditioningShifter.extract_encoder_embeddings(encoder_pipe, device, shift_cfg)  # [B,T_enc,768]

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

        # ------------------------------------------------------------------ #
        # 3 · Path L  (works entirely in 768‑D)
        # ------------------------------------------------------------------ #
        b_l = match_feature_dims(clip_l_slice, a_raw).to(device)  # feature align (no change)
        b_l = match_tokens(b_l, a_raw.shape[1]).to(device)  # token align → [B,T_enc,768]
        d_l = b_l - a_raw  # delta_L

        # ------------------------------------------------------------------ #
        # 4 · Path G  (encoder up‑scaled to 1280‑D)
        # ------------------------------------------------------------------ #
        a_up = match_feature_dims(a_raw, clip_g_slice).to(device)  # 768 → 1280
        b_g = match_tokens(clip_g_slice, a_up.shape[1]).to(device)  # [B,T_enc,1280]
        d_g = b_g - a_up  # delta_G

        # ---------------- device transfer --------------------------------- #
        to_dev = lambda x: x.to(device) if device.type == "cuda" else x.cpu()
        a_raw, a_up = map(to_dev, (a_raw, a_up))
        b_l, b_g = map(to_dev, (b_l, b_g))
        d_l, d_g = map(to_dev, (d_l, d_g))
        clip_l_slice, clip_g_slice = map(to_dev, (clip_l_slice, clip_g_slice))

        # ------------------------------------------------------------------ #
        # 5 · Walker & Stack configs (unchanged)
        # ------------------------------------------------------------------ #
        walker_cfg = FieldWalkerConfig(
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
            raw_folded_g, meta_g = integra.walk_encoder_field(a_up, b_g, d_g)
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
        return conditioning, report


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

        return ({ # this is the encoder_pipe paradigm
            "model": model,
            "tokenizer": tokenizer,
            "config": config_dict,
        },)

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

        return ({
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
                },)
