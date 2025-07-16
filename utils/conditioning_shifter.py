import torch
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass

from ..model.dual_stream_adapter_model import ConditionModulationShuntAdapter, reshape_for_shunt

logger = logging.getLogger(__name__)

@dataclass
class ShiftConfig:
    """Unified configuration for all modifications"""
    prompt: str = ""
    seed: int = -1  # -1 means no seed, use random
    strength: float = 1.0
    delta_mean: float = 0.0
    delta_scale: float = 1.0
    sigma_scale: float = 0.0
    gate_probability: float = 1.0
    gate_threshold: float = 0.1
    noise_injection: float = 0.0
    use_anchor: bool = True
    pool_method: str = "weighted_average"  # "sequential" or "weighted_average"
    # Top-K parameters
    use_topk: bool = False
    topk_percentage: float = 100.0  # Percentage of tokens to keep
    tau_temperature: float = 1.0  # Temperature scaling for tau
    topk_mode: str = "attention"  # "attention", "gate", "combined", "tau_softmax"
    guidance_scale: float = 1.0,
    max_tokens: int = 77  # Maximum number of tokens to process
    # Optimization parameters for batch processing
    batch_size: int = 4 # use no more than 4 for now, as without this it causes ram explosions with pooled outputs
    ram_capacity: float = 0.1  # Percentage of RAM to use, we divide up the pipeline into slices of ram based on batches
    vram_capacity: float = 0.1  # Percentage of VRAM to use
    eager_offloading: bool = True  # Whether to offload tensors to CPU after processing, or cache to disc if overwhelmed
    squash_similarity: bool = True  #

    # The models are very small, but the entire structure around calculating the embeddings is quite large;
    # This requires that we need to be careful with memory usage.


@dataclass
class AdapterOutput:
    """Raw output from adapter forward pass"""
    anchor: torch.Tensor
    delta: torch.Tensor  # Note: already has gate multiplied in!
    log_sigma: torch.Tensor
    tau: torch.Tensor
    g_pred: torch.Tensor
    gate: torch.Tensor
    adapter_type: str
    slice_range: Tuple[int, int]
    # Add attention weights for top-k
    attn_c2m: Optional[torch.Tensor] = None
    attn_m2c: Optional[torch.Tensor] = None


class ConditioningShifter:
    @staticmethod
    def extract_encoder_embeddings(
        encoder_pipe: Dict[str, Any],
        device: torch.device,
        shift_config: Optional[ShiftConfig | dict[str, Any]] = None,
        sampler_cfg: Dict[str, Any] = None
    ) -> torch.Tensor:
        """
        1) Clean prompt of any shunt tokens
        2) Tokenize + encode via T5/BERT
        3) Optionally project to sampler_cfg['projection_dims_in']
        """
        # 1) prompt cleanup
        if isinstance(shift_config, dict):
            shift_config = ShiftConfig(**shift_config)
        raw_prompt = shift_config.prompt
        prompt = raw_prompt#RemoveSpecialTokens.remove_special_tokens(raw_prompt)

        # 2) tokenize & encode
        tokenizer = encoder_pipe["tokenizer"]
        model     = encoder_pipe["model"]
        cfg       = encoder_pipe["config"]["config"]  # your existing mini‐config

        tokens = tokenizer(
            prompt,
            return_tensors="pt",
            padding=cfg.get("padding","max_length"),
            truncation=True,
            max_length=cfg.get("max_tokens",cfg.get("max_length", 512)),
        )
        input_ids      = tokens["input_ids"].to(device)
        attention_mask = tokens["attention_mask"].to(device)

        with torch.no_grad():
            model.to(device)
            mtype = encoder_pipe["config"].get("model_type","")
            if "t5" in mtype:
                embeddings = model.encoder(input_ids=input_ids,
                                           attention_mask=attention_mask
                ).last_hidden_state
            elif mtype in ("bert","nomic_bert"):
                embeddings = model(input_ids=input_ids,
                                   attention_mask=attention_mask,
                                   return_dict=True
                ).last_hidden_state
            else:
                raise ValueError(f"Unsupported encoder type {mtype!r}")
            model.to("cpu")  # free GPU memory

        # 3) optional input‐projection to match CLIP dims
        if sampler_cfg and sampler_cfg.get("force_projection_in", False):
            target_dims = sampler_cfg["projection_dims_in"]
            embeddings = ConditioningShifter._project_embeddings(
                embeddings, target_dims, sampler_cfg["interpolation_method_in"]
            )

        return embeddings.to(device)


    @staticmethod
    def _project_embeddings(
        embeddings: torch.Tensor,
        target_dim: int,
        mode: str
    ) -> torch.Tensor:
        """
        Interpolate the last dimension from D→target_dim via F.interpolate,
        preserving batch & sequence dims.
        """
        B, T, D = embeddings.shape
        if D == target_dim:
            return embeddings

        # [B*T, 1, D] → interpolate → [B*T, 1, target_dim] → back to [B,T,target_dim]
        flat = embeddings.reshape(B*T, 1, D)
        proj = torch.nn.functional.interpolate(
            flat.float(),
            size=target_dim,
            mode=mode,
            align_corners=(mode in {"linear","bilinear","trilinear"})
        )
        return proj.reshape(B, T, target_dim)

    @staticmethod
    def run_adapter(adapter_model: ConditionModulationShuntAdapter,
                    encoder_embeddings: torch.Tensor,
                    clip_slice: torch.Tensor,
                    guidance_scale: float,
                    adapter_type: str,
                    slice_range: Tuple[int, int]) -> AdapterOutput:
        """Run adapter and package output"""
        gen_config = {"max_guidance": guidance_scale if guidance_scale > 0 else 1.0}

        #encoder_embeddings, clip_slice = reshape_for_shunt(encoder_embeddings, clip_slice, adapter_model)

        with torch.no_grad():
            outputs = adapter_model(encoder_embeddings.float(), clip_slice.float(), config=gen_config)

            if isinstance(outputs, tuple) and len(outputs) == 8:
                anchor, delta, log_sigma, attn_c2m, attn_m2c, tau, g_pred, gate = outputs
                return AdapterOutput(
                    anchor=anchor,
                    delta=delta,  # Already has gate multiplied!
                    log_sigma=log_sigma,
                    tau=tau,
                    g_pred=g_pred,
                    gate=gate,
                    adapter_type=adapter_type,
                    slice_range=slice_range,
                    attn_c2m=attn_c2m,
                    attn_m2c=attn_m2c
                )
            else:
                raise ValueError(f"Unexpected adapter output format: {type(outputs)}")

    @staticmethod
    def apply_topk_selection(output: AdapterOutput, config: ShiftConfig) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply top-k selection using tau and attention weights.
        Returns mask and selection scores for CLIP tokens.
        """
        if not config.use_topk:
            # Return full mask matching gate dimensions
            return torch.ones_like(output.gate.squeeze(-1)), None

        # Calculate selection scores based on mode
        if config.topk_mode == "attention":
            # Use modulation->condition attention (how much each CLIP token attends to encoder)
            # Sum across encoder dimension to get importance score per CLIP token
            scores = output.attn_m2c.mean(dim=1).sum(dim=-1)  # [batch, seq_clip]
        elif config.topk_mode == "attention_collaborative":
            # Use modulation->condition attention (how much each CLIP token attends to encoder)
            # Sum across encoder dimension to get importance score per CLIP token
            # compare and normalize using the c2m attention as a soft mask
            scores = output.attn_m2c.mean(dim=1).sum(dim=-1)
            c2m_scores = output.attn_c2m.mean(dim=1).sum(dim=-1)  # [batch, seq_clip]
            # soft mask weaken and strengthen scores based on c2m_scores
            scores = (scores - c2m_scores.min()) / (c2m_scores.max() - c2m_scores.min() + 1e-8)


        elif config.topk_mode == "gate":
            # Use gate values directly (already in CLIP space)
            scores = output.gate.squeeze(-1)  # [batch, seq_clip]

        elif config.topk_mode == "combined":
            # Combine attention and gate scores
            attn_score = output.attn_m2c.mean(dim=1).sum(dim=-1)  # [batch, seq_clip]
            gate_score = output.gate.squeeze(-1)

            # Normalize and combine
            attn_score = (attn_score - attn_score.min()) / (attn_score.max() - attn_score.min() + 1e-8)
            gate_score = (gate_score - gate_score.min()) / (gate_score.max() - gate_score.min() + 1e-8)

            scores = (attn_score + gate_score) / 2

        elif config.topk_mode == "tau_softmax":
            # Use tau as temperature for softmax selection
            attn_score = output.attn_m2c.mean(dim=1).sum(dim=-1)  # [batch, seq_clip]

            # Apply tau temperature scaling
            tau_value = output.tau.mean().item() * config.tau_temperature
            scores = torch.nn.functional.softmax(attn_score / tau_value, dim=-1)
        else:
            scores = output.gate.squeeze(-1)

        # Calculate k
        k = int(scores.size(-1) * (config.topk_percentage / 100.0))
        k = max(1, min(k, scores.size(-1)))

        # Get top-k indices
        topk_values, topk_indices = torch.topk(scores, k, dim=-1)

        # Create sparse mask
        mask = torch.zeros_like(scores)
        mask.scatter_(-1, topk_indices, 1.0)

        return mask, scores

    @staticmethod
    def apply_modifications(clip_slice: torch.Tensor, outputs: List[AdapterOutput],
                            config: ShiftConfig) -> torch.Tensor:
        """Apply modifications based on config.pool_method"""
        torch.manual_seed(config.seed if config.seed >= 0 else torch.randint(0, 2**32, (1,)).item())

        modified = clip_slice.clone()
        if config.pool_method == "sequential":
            # Apply each adapter sequentially
            for output in outputs:
                modified = ConditioningShifter._apply_single(modified, output, config)
            return modified

        elif config.pool_method == "weighted_average":
            # Pool all adapters then apply once
            if len(outputs) == 1:
                return ConditioningShifter._apply_single(modified, outputs[0], config)

            pooled = ConditioningShifter._pool_outputs(outputs)
            return ConditioningShifter._apply_single(clip_slice, pooled, config)

        else:
            raise ValueError(f"Unknown pool_method: {config.pool_method}")

    @staticmethod
    def _apply_single(clip_slice: torch.Tensor, output: AdapterOutput,
                      config: ShiftConfig) -> torch.Tensor:
        """Apply a single adapter output with optional top-k selection"""

        # Apply top-k selection if enabled
        topk_mask, scores = ConditioningShifter.apply_topk_selection(output, config)

        # Preprocess (but remember delta already has gate!)
        delta = output.delta * config.delta_scale + config.delta_mean

        gate_scaled = output.gate * config.gate_probability
        gate_mask = (gate_scaled > config.gate_threshold).float()
        gate_masked = gate_scaled * gate_mask

        # Apply top-k mask to gate and delta
        if config.use_topk:
            # Expand mask to match dimensions
            topk_mask_expanded = topk_mask.unsqueeze(-1)
            gate_masked = gate_masked * topk_mask_expanded
            delta = delta * topk_mask_expanded

        # Apply strength
        delta_final = delta

        # Apply based on anchor mode
        if config.use_anchor:
            # Blend original with anchor, then add delta
            blended = clip_slice * (1 - gate_masked) + output.anchor * gate_masked
            clip_modified = blended + delta_final
        else:
            # Simple additive
            clip_modified = clip_slice + delta_final

        # Apply noise
        if config.sigma_scale > 0 and config.noise_injection > 0:
            sigma = torch.exp(output.log_sigma * config.sigma_scale)
            clip_modified += torch.randn_like(clip_modified) * sigma * config.noise_injection
        elif config.noise_injection > 0:
            clip_modified += torch.randn_like(clip_modified) * config.noise_injection

        return clip_modified

    @staticmethod
    def _pool_outputs(outputs: List[AdapterOutput]) -> AdapterOutput:
        """Pool multiple adapter outputs into one"""
        # Simple weighted average
        total_weight = len(outputs)

        pooled_anchor = sum(o.anchor for o in outputs) / total_weight
        pooled_delta = sum(o.delta for o in outputs) / total_weight
        pooled_log_sigma = sum(o.log_sigma for o in outputs) / total_weight

        # Handle tau with different head counts
        if all(o.tau is not None for o in outputs):
            # Take mean across heads for each adapter, then average
            tau_values = [o.tau.mean().item() for o in outputs]
            pooled_tau_value = sum(tau_values) / total_weight
            # Create scalar tensor on same device
            pooled_tau = torch.tensor(pooled_tau_value, device=outputs[0].tau.device)
        else:
            pooled_tau = None

        pooled_g_pred = sum(o.g_pred for o in outputs) / total_weight if outputs[0].g_pred is not None else None
        pooled_gate = sum(o.gate for o in outputs) / total_weight

        # Pool attention weights if available - handle different head counts
        pooled_attn_c2m = None
        pooled_attn_m2c = None
        if all(o.attn_c2m is not None for o in outputs):
            # First, average across heads for each adapter to get [batch, seq_c, seq_m]
            attn_c2m_list = []
            attn_m2c_list = []

            for o in outputs:
                # Average across heads dimension
                attn_c2m_avg = o.attn_c2m.mean(dim=1)  # [batch, seq_c, seq_m]
                attn_m2c_avg = o.attn_m2c.mean(dim=1)  # [batch, seq_m, seq_c]
                attn_c2m_list.append(attn_c2m_avg)
                attn_m2c_list.append(attn_m2c_avg)

            # Now average across adapters
            pooled_attn_c2m = sum(attn_c2m_list) / total_weight
            pooled_attn_m2c = sum(attn_m2c_list) / total_weight

            # Add back a dummy heads dimension for compatibility
            pooled_attn_c2m = pooled_attn_c2m.unsqueeze(1)  # [batch, 1, seq_c, seq_m]
            pooled_attn_m2c = pooled_attn_m2c.unsqueeze(1)  # [batch, 1, seq_m, seq_c]

        return AdapterOutput(
            anchor=pooled_anchor,
            delta=pooled_delta,
            log_sigma=pooled_log_sigma,
            tau=pooled_tau,
            g_pred=pooled_g_pred,
            gate=pooled_gate,
            adapter_type=outputs[0].adapter_type,
            slice_range=outputs[0].slice_range,
            attn_c2m=pooled_attn_c2m,
            attn_m2c=pooled_attn_m2c
        )

    @staticmethod
    def conditioning_set_values(conditioning, values={}, append=False):
        """
        Set values in conditioning based on provided values.
        Original set values was provided by comfyui node_helpers.py

        """
        c = []
        for t in conditioning:
            n = [t[0], t[1].copy()]
            for k in values:
                val = values[k]
                if append:
                    old_val = n[1].get(k, None)
                    if old_val is not None:
                        val = old_val + val

                n[1][k] = val
            c.append(n)

        return

    @staticmethod
    def conditioning_set_strength(conditioning, cond_strength: float, pool_strength: float = 1.0):
        """
        Set strength in conditioning based on provided strength - we need to manually modify instead of setting values.
            [    [base_tensor, { "pooled_outputs": pool, ... other dict entries } ], ...    ]
        """
        c = []
        for t in conditioning:
            base_tensor = t[0].copy()
            # Set our usage strength, then find out if we have pooled outputs
            base_tensor *= cond_strength
            kwarg_dict = t[1].clone() if t[1] is not None else {} # copies the config params for later use

            # lets get and remove the pooled outputs if they exist
            pooled: Optional[None | torch.Tensor] = kwarg_dict.get("pooled_outputs", None)
            if pooled is not None:
                del kwarg_dict["pooled_outputs"]
                pooled = pooled.clone()
                # If we have pooled outputs, apply the pooled strength
                pooled *= pool_strength
                kwarg_dict["pooled_outputs"] = pooled

            c.append([base_tensor, kwarg_dict])



