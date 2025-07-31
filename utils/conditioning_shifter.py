import comfy
import torch
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import torch
import torch.nn.functional as F

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

    def join_dict(self, dict_in):
        # any key in dict_in that is not in this config will be ignored
        for key, value in dict_in.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                logger.warning(f"ShiftConfig: Ignoring unknown key {key!r} in join_dict")



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
            truncation=False,
            max_length=cfg.get("max_tokens",cfg.get("max_length", 512)),
        )

        if "bert" in encoder_pipe.get("name", ""):
            # BERT tokenizers use special tokens like [CLS] and [SEP]
            input_ids = ConditioningShifter.apply_gap_spacer(tokens["input_ids"], tokenizer, interval=77)
        else:
            input_ids = tokens["input_ids"]

        input_ids = input_ids.to(device)
        attention_mask = tokens["attention_mask"].to(device)


        with torch.no_grad():
            model.to(device)
            mtype = encoder_pipe["config"].get("model_type","")
            name = encoder_pipe.get("name", "").lower()
            if "t5" in mtype:
                embeddings = model.encoder(input_ids=input_ids,
                                           attention_mask=attention_mask
                ).last_hidden_state
            elif mtype in ("bert","nomic_bert"):
                # --- Auto-enable RoPE spiral if RoPE-aware encoder is used ---

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
    def apply_gap_spacer(input_ids: torch.Tensor, tokenizer, interval: int = 77) -> torch.Tensor:
        """
        Replaces [PAD] tokens with [MASK], and inserts [SEP] every `interval` tokens.
        This operation is destructive: it overwrites tokens at those exact positions.
        """
        input_ids = input_ids.clone()

        pad_id = tokenizer.pad_token_id
        mask_id = tokenizer.mask_token_id
        sep_id = tokenizer.sep_token_id
        cls_id = tokenizer.cls_token_id

        # set the 0th token to [CLS]
        input_ids[..., 0] = cls_id

        # Replace [PAD] with [MASK] every other pad token
        # we skip and leave one token, then apply mask to the next, and then check for pad
        for idx in range(1, input_ids.shape[-1]):
            if input_ids[..., idx] == pad_id:
                if (idx % 2) == 0:
                    input_ids[..., idx] = mask_id

        # Insert [SEP] tokens every `interval` step
        seq_len = input_ids.shape[-1]
        for idx in range(0, seq_len, interval):
            input_ids[..., idx] = sep_id

        return input_ids


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
        return c

    # [[ cond_tensor, { "pooled_outputs": pooled_tensor, ... } ], ...]
    @staticmethod
    def clone_conditionings(conditioning: List[List], device: str = torch.device("cpu")) -> List[List]:
        """
        Clone the conditioning list to avoid modifying the original.
        Standard format: [[ cond_tensor, { "pooled_output": pooled_tensor, ... } ], ...]
        Hidream format: [[ cond_tensor, { "pooled_output": pooled_tensor, ... }, hidream_extra_tensor, { hidream_metadata } ], ...]
        """
        if conditioning:
            for i in range(len(conditioning)):
                temp_cond = conditioning[i].copy()
                temp_cond[0] = temp_cond[0].clone().to(device)
                if len(temp_cond) > 1:
                    # check for dict
                    if isinstance(temp_cond[1], dict):
                        # if dict iterate and clone all tensors
                        cloned_dict = {k: v.clone().to(device) if isinstance(v, torch.Tensor) else v for k, v in temp_cond[1].items()}
                        temp_cond[1] = cloned_dict
                    elif isinstance(temp_cond[1], torch.Tensor):
                        # if tensor, clone it
                        temp_cond[1] = temp_cond[1].clone().to(device)
                if len(temp_cond) > 2:
                    # check for hidream extra tensor
                    if isinstance(temp_cond[2], torch.Tensor):
                        temp_cond[2] = temp_cond[2].clone().to(device)
                    elif isinstance(temp_cond[2], dict):
                        # if dict iterate and clone all tensors
                        cloned_dict = {k: v.clone().to(device) if isinstance(v, torch.Tensor) else v for k, v in temp_cond[2].items()}
                        temp_cond[2] = cloned_dict
                cloned = temp_cond
                conditioning[i] = cloned
            return conditioning
        else:
            raise ValueError("Conditioning list is empty, cannot clone.")


    @staticmethod
    def get_rope_resonance(
            embedding: torch.Tensor,
            attention_mask: torch.Tensor,
            offsets: List[int],
            anchor: Optional[torch.Tensor] = None,
            anchor_mode: str = "pooled"
    ) -> Dict[str, float]:
        """
        Computes RoPE-aware resonance statistics from token embeddings:
        - spiral_sim: average phase alignment across offsets
        - harmonic_trace: phase stability (std dev)
        - anchor_similarity: comparison with anchor vector
        """
        B, T, D = embedding.shape
        if B != 1:
            raise ValueError("Only batch size 1 supported for resonance")

        em = embedding[0][attention_mask[0].bool()]  # [T_active, D]
        T_active = em.shape[0]
        if T_active < 2:
            return {"spiral_sim": 0.0, "harmonic_trace": 0.0, "anchor_similarity": 0.0}

        sims = []
        for offset in offsets:
            if offset >= T_active:
                continue
            sims.append(torch.cosine_similarity(em[:-offset], em[offset:], dim=-1).mean())

        if not sims:
            return {"spiral_sim": 0.0, "harmonic_trace": 0.0, "anchor_similarity": 0.0}

        sim_tensor = torch.stack(sims)
        spiral_sim = sim_tensor.mean().item()
        harmonic_trace = sim_tensor.std().item()

        anchor_similarity = 0.0
        if anchor is not None:
            pooled = ConditioningShifter._select_anchor_vector(em.unsqueeze(0), anchor_mode=anchor_mode)[0]
            anchor_similarity = F.cosine_similarity(anchor.to(pooled.device).unsqueeze(0), pooled.unsqueeze(0),
                                                    dim=-1).item()

        return {
            "spiral_sim": spiral_sim,
            "harmonic_trace": harmonic_trace,
            "anchor_similarity": anchor_similarity
        }

    @staticmethod
    def _select_anchor_vector(
            embedding: torch.Tensor,
            anchor_mode: str = "pooled"
    ) -> torch.Tensor:
        """
        Chooses a vector to represent the prompt for anchor comparison.
        """
        if embedding.dim() == 2:
            embedding = embedding.unsqueeze(0)

        if anchor_mode in {"pooled", "mean"}:
            return embedding.mean(dim=1)
        elif anchor_mode == "first":
            return embedding[:, 0, :]
        elif anchor_mode == "last":
            return embedding[:, -1, :]
        else:
            raise ValueError(f"Unknown anchor_mode: {anchor_mode}")

    @staticmethod
    def compute_resonance_potential(
            embedding: torch.Tensor,
            attention_mask: torch.Tensor,
            offsets: Optional[List[int]] = None,
            mode: str = "harmonic_std"
    ) -> torch.Tensor:
        """
        Returns a [B, T, 1] potential mask for folding modulation.
        Modes:
            - "spiral_gate": mean cosine similarity across offsets
            - "harmonic_std": inverse stddev of similarity across offsets

        Args:
            embedding: Input embeddings [B, T, D]
            attention_mask: Attention mask [B, T]
            offsets: Position offsets for similarity computation. If None, uses adaptive defaults.
            mode: Computation mode for potential calculation

        Returns:
            Potential mask [B, T, 1] for modulation
        """

        # Input validation
        if embedding is None:
            raise ValueError("embedding cannot be None")
        if attention_mask is None:
            raise ValueError("attention_mask cannot be None")

        B, T, D = embedding.shape
        if B != 1:
            raise ValueError("Only batch size 1 is supported for resonance potential.")

        # Validate attention_mask shape
        if attention_mask.shape != (B, T):
            raise ValueError(
                f"attention_mask shape {attention_mask.shape} doesn't match embedding batch/seq dims ({B}, {T})")

        # Set adaptive offsets if None - optimized for nomic-bert-2048 RoPE patterns
        if offsets is None:
            active_count = attention_mask[0].sum().item()
            if active_count <= 8:
                offsets = [1, 2, 3]
            elif active_count <= 16:
                offsets = [1, 2, 3, 5]
            elif active_count <= 32:
                offsets = [1, 2, 3, 5, 8]
            elif active_count <= 64:
                offsets = [1, 2, 3, 5, 8, 13]
            else:
                # For longer sequences, add more sophisticated patterns
                offsets = [1, 2, 3, 5, 8, 13, 21, 34]

        if not isinstance(offsets, list) or len(offsets) == 0:
            raise ValueError("offsets must be a non-empty list")

        # Extract active tokens with safety checks
        mask_bool = attention_mask[0].bool()
        active_count = mask_bool.sum().item()

        if active_count == 0:
            # No active tokens - return all ones
            logger.warning("No active tokens found in attention_mask, returning default potentials")
            return torch.ones(1, T, 1, device=embedding.device, dtype=embedding.dtype)

        em = embedding[0][mask_bool]  # [T_active, D]
        T_active = em.shape[0]

        # Early return for insufficient tokens
        if T_active < 2:
            result = torch.ones(1, T_active, 1, device=embedding.device, dtype=embedding.dtype)
            if T_active < T:
                result = F.pad(result, (0, 0, 0, T - T_active), value=1.0)
            return result

        # Filter valid offsets
        valid_offsets = [offset for offset in offsets if 0 < offset < T_active]

        if not valid_offsets:
            logger.warning(f"No valid offsets found. T_active={T_active}, offsets={offsets}")
            result = torch.ones(1, T_active, 1, device=embedding.device, dtype=embedding.dtype)
            if T_active < T:
                result = F.pad(result, (0, 0, 0, T - T_active), value=1.0)
            return result

        # Compute similarities for valid offsets
        sim_matrix = []
        for offset in valid_offsets:
            try:
                # Calculate similarity between shifted sequences
                sim = F.cosine_similarity(em[:-offset], em[offset:], dim=-1)  # [T_active - offset]

                # Pad to full T_active length
                padded = F.pad(sim, (offset, 0), value=1.0)  # [T_active]
                sim_matrix.append(padded)

            except Exception as e:
                logger.error(f"Error computing similarity for offset {offset}: {e}")
                continue

        if not sim_matrix:
            logger.error("Failed to compute any similarities, returning default potentials")
            result = torch.ones(1, T_active, 1, device=embedding.device, dtype=embedding.dtype)
            if T_active < T:
                result = F.pad(result, (0, 0, 0, T - T_active), value=1.0)
            return result


        # Stack and compute potentials
        try:
            sim_stack = torch.stack(sim_matrix)  # [len(valid_offsets), T_active]

            if mode == "spiral_gate":
                potentials = sim_stack.mean(dim=0)  # [T_active]
            elif mode == "harmonic_std":
                std_vals = sim_stack.std(dim=0)  # [T_active]
                # Add small epsilon to prevent division by zero
                potentials = 1.0 / (1.0 + std_vals + 1e-8)  # [T_active]
            else:
                raise ValueError(f"Unknown resonance mode: {mode}")

            # Ensure valid range and reshape
            potentials = potentials.clamp(min=0.0, max=1.0).view(1, T_active, 1)

            # Pad back to original sequence length if needed
            if T_active < T:
                potentials = F.pad(potentials, (0, 0, 0, T - T_active), value=1.0)

            return potentials

        except Exception as e:
            logger.error(f"Error in final potential computation: {e}")
            # Fallback to default
            return torch.ones(1, T, 1, device=embedding.device, dtype=embedding.dtype)

    @staticmethod
    def set_device_conds(conditioning: list, device: Optional[str] = None):
        """
        Set the device for all conditioning tensors.
        This is useful for ensuring that the conditioning tensors are on the correct device.
        """
        out = []
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        for combined, info in conditioning:
            combined = combined.to(device)
            for k, v in info.items():
                if isinstance(v, torch.Tensor):
                    info[k] = v.to(device)
                if k == "device":
                    info[k] = device
            out.append([combined, info])
        return (out,)