import logging
from typing import Dict, List, Optional, Tuple, Any

import torch
from ..utils.conditioning_helper import ConditioningHelper, UsefulConditioning, ModelSlicer
from ..sampler.integra import IntegraConfig, IntegraOrchestrator
from ..sampler.sliding_window import ShuntStackConfig
from ..sampler.alucard import FieldWalkerConfig
from ..utils.alignment import match_project, match_feature_dims, match_tokens
from ..sampler.alucard_exceptions import AlucardShapeError

logger = logging.getLogger(__name__)


def apply_alpha_blend_on_slice(
        base_tensor: torch.Tensor,
        injected_tensor: torch.Tensor,
        start: int,
        end: int,
        alpha: float,
) -> torch.Tensor:
    """
    Applies an alpha blend from `injected_tensor` into `base_tensor`,
    only over token positions [start:end]. Assumes [B, T, D] shape.
    """
    if injected_tensor.shape[1] != (end - start):
        raise ValueError(
            f"Injected slice shape mismatch: expected length {(end - start)}, got {injected_tensor.shape[1]}")

    blended = base_tensor.clone()
    blended[:, start:end, :] = torch.lerp(
        base_tensor[:, start:end, :],
        injected_tensor,
        alpha
    )
    return blended

class ClipSamplerProcessor:
    def __init__(
        self,
        prompt: str,
        negative_prompt: Optional[str],
        encoders: List[Dict],
        clip,
        prompt_config: dict,
        sliding_window_cfg: dict,
        folding_cfg: dict,
        scheduler_hyper_cfg: dict,
        projection_cfg: dict,
        experimental_cfg: dict,
        mode: str = "sdxl"
    ):
        self.prompts = {
            "positive": prompt,
            "negative": negative_prompt or None,
        }

        self.encoders = encoders
        self.clip = clip
        self.mode = mode

        self.config = {
            **prompt_config,
            **sliding_window_cfg,
            **folding_cfg,
            **scheduler_hyper_cfg,
            **projection_cfg,
            **experimental_cfg,
        }

        self.device = self.config.get("device", "cpu")

    def run(self) -> Tuple:
        results = self._process_prompts(self.prompts)

        # Extract return values explicitly
        pos = results.get("positive", ([], [], {}))
        neg = results.get("negative", ([], [], {}))

        return (*pos, *neg)

    def _process_prompts(self, prompts: Dict[str, Optional[str]]) -> Dict[str, Tuple]:
        """
        Processes multiple prompt roles (e.g. positive, negative) in one pass.
        Returns: {role: (sampled, raw, debug)}
        """
        output = {}

        for role, prompt in prompts.items():
            if not prompt:
                output[role] = ([], [], {})
                continue

            logger.info(f"[ClipSamplerProcessor] Processing '{role}' prompt with mode {self.mode}")

            # 1. Symbolic extraction per encoder
            a_raws = [
                (encoder, ConditioningHelper.extract_symbolic_field(
                    encoder, prompt, self.config, self.device
                ))
                for encoder in self.encoders
            ]

            # 2. CLIP baseline
            clip_tensor, clip_meta = ConditioningHelper.schedule_and_extract_clip_conditioning(
                self.clip, prompt, self.config, self.device, mode=self.mode
            )
            orig_clip = clip_tensor.clone()
            orig_pooled = clip_meta.get("pooled_output")
            if orig_pooled is not None:
                orig_pooled = orig_pooled.clone()

            # 3. Slice clip tensor
            base_uc = UsefulConditioning([[clip_tensor.clone(), {"pooled_output": orig_pooled}]])
            clip_uc = ModelSlicer.slice(base_uc, model_type=self.mode, device=self.device)

            # 4. Fold all symbolic slices
            folded_outputs: Dict[str, List[torch.Tensor]] = {}
            for encoder, a_raw in a_raws:
                for idx in range(len(clip_uc)):
                    slice_tensor = clip_uc.get_tensor(idx)
                    meta = clip_uc.get_all_metadata()[idx]
                    slicer_info = meta.get("slicer_info", {})
                    key = slicer_info.get("key", f"unk_{idx}")

                    folded = self._run_integra(
                        a_raw,
                        key,
                        slice_tensor,
                        encoder,
                        slicer_info=slicer_info  # Pass for scoped alpha
                    )

                    folded_outputs.setdefault(key, []).append(folded)

            if not folded_outputs:
                raise RuntimeError(f"No folded outputs for '{role}' prompt.")

            # 5. Assemble conditioning output
            conditioning = ConditioningHelper.pack_conditioning_bundle(
                folded_outputs, cfg=self.config, device=self.device, mode=self.mode,
            )
            if self.config.get("pool_frozen", False):
                del conditioning[0][1]["pooled_output"]  # Use pooled output directly
                conditioning[0][1]["pooled_output"] = orig_pooled.clone()  # Attach original pooled output


            raw = [[orig_clip, {"pooled_output": orig_pooled}]]

            output[role] = (conditioning, raw, {})  # TODO: attach debug if needed

        return output

    def _run_integra(self, a_raw, cond_name, clip_slice, encoder, slicer_info=None):
        cfg = self.config
        if a_raw.size(-1) != clip_slice.size(-1) or cfg.get("force_projection_in", False):
            a_proj = match_project(a_raw, clip_slice, mode=cfg.get("interpolation_method_in", "linear"))
        else:
            a_proj = a_raw

        a_feat = match_feature_dims(a_proj, clip_slice)
        b = match_tokens(clip_slice, a_feat.shape[1])
        delta = b - a_proj
        integra = self._build_integra(encoder)
        try:
            # ✅ Now passes full config as context
            folded, _ = integra.walk_encoder_field(a_feat, b, delta, context=self.config)
        except AlucardShapeError as e:
            raise RuntimeError(f"[ClipSamplerProcessor] Shape error: {e}") from e

        return match_tokens(folded, clip_slice.shape[1]).to(self.device)

    def _build_integra(self, encoder):
        cfg = self.config
        walker_cfg = FieldWalkerConfig(
            name=cfg.get("name", "Alucard"),
            folding_mode=cfg.get("folding", "shiva"),
            scheduler_mode=cfg.get("folding_scheduler", "tau"),
            t_steps=cfg.get("steps", 100),
            padding_mode=cfg.get("padding_mode", "sparse"),
            pooling_mode=cfg.get("pooling_mode", "bilinear"),
            scheduler_config={
                "tau": cfg.get("tau", 5.0),
                "top_k": cfg.get("top_k", 50.0),
                "top_p": cfg.get("top_p", 0.9),
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
            sliding_window_size=cfg.get("sliding_window_size", 77),
            sliding_window_stride=cfg.get("sliding_window_stride", 77),
            context_window_size=cfg.get("context_window_size", 2048),
            override_context_window=cfg.get("override_context_window", False),
            context_window=cfg.get("context_window", ""),
            max_windows=cfg.get("max_windows", 64),
        )


        return IntegraOrchestrator(IntegraConfig(
            walker_config=walker_cfg,
            stack_config=stack_cfg,
            trace_folds=False,
            enforce_projection=cfg.get("force_projection_in", True),
            enable_clip_alignment=cfg.get("cosine_similarity_gate", True),
            use_rose_similarity=cfg.get("use_rose_similarity", True),
        ))
