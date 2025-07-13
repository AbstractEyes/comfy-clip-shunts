from typing import Tuple

import torch
import torch.nn as nn
from .configs import ENCODER_CONFIGS, HARMONIC_SHUNT_REPOS


class DualConversionNames:
    """
    Mapping from legacy dual adapter layer names to updated
    condition/modulation schema. Also supports delta/gate harmonization.
    """
    LAYER_NAMES = {
        # Projection remapping
        "t5_proj":       "condition_projection",
        "clip_proj":     "modulation_projection",

        # Cross attention
        "cross_t2c":     "cross_c2m",   # condition to modulation
        "cross_c2t":     "cross_m2c",   # modulation to condition

        # Output projections
        "anchor_proj":   "anchor_projection",
        "delta_proj":    "delta_projection",
        "logsig_proj":   "log_sigma_projection",

        # Gate and guidance
        "gate_proj":     "gate_projection",
        "guidance_proj": "guidance_projection",

        # Fuse block
        "fuse":          "fusion_block",

        # Pocket residual
        "pocket_blocks": "residual_pocket_block"
    }


# ─── Residual Pocket Block ───────────────────────────────────
class BottleneckResBlock(nn.Module):
    def __init__(self, dim, kernel=3, dropout=0.0):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.conv = nn.Conv1d(dim, dim, kernel_size=kernel, padding=kernel // 2, groups=1)
        self.proj = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Linear(dim * 2, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        residual = x
        x = self.norm(x)
        x = x.transpose(1, 2)
        x = self.conv(x).transpose(1, 2)
        return residual + self.proj(x)


class ConditionModulationShuntAdapter(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        self.config = config
        self.dtype = config.get("dtype", torch.float32)
        self.condition_dim = config.get("condition_encoders", [])[0].get("hidden_size", 768)
        self.modulation_dim = config.get("modulation_encoders", [])[0].get("hidden_size", 768)
        self.bneck = config["bottleneck"]
        self.heads = config["heads"]
        self.tau_init = config["tau_init"]
        self.max_guidance = config["max_guidance"]

        use_norm   = config.get("layer_norm", True)
        use_do     = config.get("use_dropout", True)
        do_p       = config.get("dropout", 0.0)
        proj_depth = config.get("proj_layers", 2)

        def build_projection(input_dim, output_dim):
            layers = []
            last_dim = input_dim
            if use_norm:
                layers.append(nn.LayerNorm(last_dim))
            for i in range(proj_depth):
                next_dim = self.bneck * (2 if i == 0 and proj_depth > 1 else 1)
                layers.append(nn.Linear(last_dim, next_dim))
                layers.append(nn.GELU())
                if use_do:
                    layers.append(nn.Dropout(do_p))
                last_dim = next_dim
            layers.append(nn.Linear(last_dim, output_dim))
            return nn.Sequential(*layers)

        # Projection layers
        self.condition_projection = build_projection(self.condition_dim, self.bneck)
        self.modulation_projection = build_projection(self.modulation_dim, self.bneck)

        # Cross attention blocks
        self.cross_c2m = nn.MultiheadAttention(self.bneck, self.heads, batch_first=True, dropout=do_p)
        self.cross_m2c = nn.MultiheadAttention(self.bneck, self.heads, batch_first=True, dropout=do_p)
        self.tau       = nn.Parameter(torch.full((self.heads, 1, 1), self.tau_init))

        # Residual processing block
        self.residual_pocket_block = nn.Sequential(
            BottleneckResBlock(self.bneck, dropout=do_p),
            BottleneckResBlock(self.bneck, dropout=do_p)
        )

        # Fusion pathway
        self.fusion_block = nn.Sequential(
            nn.LayerNorm(2 * self.bneck),
            nn.Linear(2 * self.bneck, self.bneck * 2),
            nn.GELU(),
            nn.Linear(self.bneck * 2, self.bneck)
        )

        # Output projections
        self.anchor_projection    = build_projection(self.bneck, self.modulation_dim)
        self.delta_projection     = build_projection(self.bneck, self.modulation_dim)
        self.log_sigma_projection = build_projection(self.bneck, self.modulation_dim)

        # Gate and guidance
        self.gate_projection = nn.Sequential(
            nn.LayerNorm(self.bneck),
            nn.Linear(self.bneck, self.bneck),
            nn.GELU(),
            nn.Linear(self.bneck, 1),
            nn.Tanh(),
            nn.Sigmoid()
        )
        self.guidance_projection = nn.Sequential(
            nn.LayerNorm(self.bneck),
            nn.Linear(self.bneck, 1),
            nn.Sigmoid()
        )

        # ─── Legacy Aliases (Version 1 Compatibility) ──────────────────────────
        self.proj_t5        = self.condition_projection
        self.proj_clip      = self.modulation_projection
        self.cross_t2c      = self.cross_c2m
        self.cross_c2t      = self.cross_m2c
        self.pocket_blocks  = self.residual_pocket_block
        self.fuse           = self.fusion_block
        self.anchor_proj    = self.anchor_projection
        self.delta_proj     = self.delta_projection
        self.logsig_proj    = self.log_sigma_projection
        self.gate_proj      = self.gate_projection
        self.guidance_proj  = self.guidance_projection

    def forward(self, cond_seq: torch.Tensor, mod_seq: torch.Tensor, config: dict = None):
        if self.config.get("assert_input_dims", True):
            assert cond_seq.size(-1) == self.condition_dim
            assert mod_seq.size(-1) == self.modulation_dim

        max_guidance = self.max_guidance if config is None else config.get("max_guidance", 0.0)
        if max_guidance <= 0:
            max_guidance = self.max_guidance
        if max_guidance <= 0:
            max_guidance = config.get("guidance_scale", 10.0)

        cond_b = self.condition_projection(cond_seq)
        mod_b  = self.modulation_projection(mod_seq)

        c2m, attn_c2m = self.cross_c2m(cond_b, mod_b, mod_b, need_weights=True, average_attn_weights=False)
        m2c, attn_m2c = self.cross_m2c(mod_b, cond_b, cond_b, need_weights=True, average_attn_weights=False)

        pocket = self.residual_pocket_block(c2m)
        pocket_mean = pocket.mean(1, keepdim=True).expand(-1, mod_b.size(1), -1)

        h = self.fusion_block(torch.cat([pocket_mean, m2c], dim=-1))

        anchor    = self.anchor_projection(h)
        delta     = self.delta_projection(h) * self.gate_projection(h)
        log_sigma = self.log_sigma_projection(h)

        g_tok  = self.guidance_projection(h).squeeze(-1)
        g_pred = g_tok.mean(1, keepdim=True) * max_guidance

        return anchor, delta, log_sigma, attn_c2m, attn_m2c, self.tau, g_pred, self.gate_projection(h)


# ─── V1 Original Two Stream Shunt Adapter ──────────────────────────────────────
class TwoStreamShuntAdapter(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        self.config = config
        self.dtype = config.get("dtype", torch.float32)
        self.t5_dim = config.get("condition_encoders", [])[0].get("hidden_size", 768)
        self.clip_dim = config.get("modulation_encoders", [])[0].get("hidden_size", 768)
        self.bneck = config["bottleneck"]
        self.heads = config["heads"]
        self.tau_init = config["tau_init"]
        self.max_guidance = config["max_guidance"]

        use_norm   = config.get("layer_norm", True)
        use_do     = config.get("use_dropout", True)
        do_p       = config.get("dropout", 0.0)
        proj_depth = config.get("proj_layers", 2)

        def build_projection(input_dim, output_dim):
            layers = []
            last_dim = input_dim
            if use_norm:
                layers.append(nn.LayerNorm(last_dim))
            for i in range(proj_depth):
                next_dim = self.bneck * (2 if i == 0 and proj_depth > 1 else 1)
                layers.append(nn.Linear(last_dim, next_dim))
                layers.append(nn.GELU())
                if use_do:
                    layers.append(nn.Dropout(do_p))
                last_dim = next_dim
            layers.append(nn.Linear(last_dim, output_dim))
            return nn.Sequential(*layers)

        # Projections
        self.proj_t5   = build_projection(self.t5_dim, self.bneck)
        self.proj_clip = build_projection(self.clip_dim, self.bneck)

        # Attention
        self.cross_t2c = nn.MultiheadAttention(self.bneck, self.heads, batch_first=True, dropout=do_p)
        self.cross_c2t = nn.MultiheadAttention(self.bneck, self.heads, batch_first=True, dropout=do_p)
        self.tau       = nn.Parameter(torch.full((self.heads, 1, 1), self.tau_init))

        # Residual Pocket
        self.pocket_blocks = nn.Sequential(
            BottleneckResBlock(self.bneck, dropout=do_p),
            BottleneckResBlock(self.bneck, dropout=do_p)
        )

        # Fuse
        self.fuse = nn.Sequential(
            nn.LayerNorm(2 * self.bneck),
            nn.Linear(2 * self.bneck, self.bneck * 2),
            nn.GELU(),
            nn.Linear(self.bneck * 2, self.bneck)
        )

        # Output Projections
        self.anchor_proj = build_projection(self.bneck, self.clip_dim)
        self.delta_proj  = build_projection(self.bneck, self.clip_dim)
        self.logsig_proj = build_projection(self.bneck, self.clip_dim)

        self.gate_proj = nn.Sequential(
            nn.LayerNorm(self.bneck),
            nn.Linear(self.bneck, self.bneck),
            nn.GELU(),
            nn.Linear(self.bneck, 1),
            nn.Tanh(),
            nn.Sigmoid()
        )

        self.guidance_proj = nn.Sequential(
            nn.LayerNorm(self.bneck),
            nn.Linear(self.bneck, 1),
            nn.Sigmoid()
        )

    def forward(self, t5_seq: torch.Tensor, clip_seq: torch.Tensor, config: dict = None):
        if self.config.get("assert_input_dims", True):
            assert t5_seq.size(-1) == self.t5_dim
            assert clip_seq.size(-1) == self.clip_dim

        max_guidance = self.max_guidance if config is None else config.get("max_guidance", 0.0)
        if max_guidance <= 0:
            max_guidance = self.max_guidance
        if max_guidance <= 0:
            max_guidance = 10
        max_guidance = config.get("guidance_scale", 5.0)

        t5_b   = self.proj_t5(t5_seq)
        clip_b = self.proj_clip(clip_seq)

        t2c, attn_t2c = self.cross_t2c(t5_b, clip_b, clip_b, need_weights=True, average_attn_weights=False)
        c2t, attn_c2t = self.cross_c2t(clip_b, t5_b, t5_b, need_weights=True, average_attn_weights=False)

        pocket = self.pocket_blocks(t2c)

        pocket_mean = pocket.mean(1, keepdim=True).expand(-1, clip_b.size(1), -1)
        h = self.fuse(torch.cat([pocket_mean, c2t], dim=-1))

        anchor    = self.anchor_proj(h)
        delta     = self.delta_proj(h) * self.gate_proj(h)
        log_sigma = self.logsig_proj(h)

        g_tok  = self.guidance_proj(h).squeeze(-1)
        g_pred = g_tok.mean(1, keepdim=True) * max_guidance

        return anchor, delta, log_sigma, attn_t2c, attn_c2t, self.tau, g_pred, self.gate_proj(h)




from safetensors.torch import save_file, load_file

def save_safetensors(adapter: nn.Module, path: str, metadata: dict = None):
    """
    Save the current adapter state to safetensors format.
    All tensors are moved to CPU and saved as float32 for compatibility.
    Optional metadata may be embedded (e.g., version, prompt_mode).
    """
    state = {k: v.float().cpu() for k, v in adapter.state_dict().items()}
    save_file(state, path, metadata=metadata or {})
    print(f"✅ Model saved to {path}")


def load_safetensors(adapter: nn.Module, path: str, map_location="cpu"):
    """
    Load a safetensors checkpoint into the adapter.
    Uses strict key matching. Tensors are loaded to the specified device.
    """
    state = load_file(path, device=map_location)
    adapter.load_state_dict(state, strict=True)
    print(f"✅ Model loaded from {path}")


def load_converted_safetensors(adapter: nn.Module, path: str, map_location="cpu"):
    """
    Load a legacy-format adapter into the updated dual-shunt schema.
    Converts key names according to DualConversionNames mapping.
    """
    state = load_file(path, device=map_location)
    new_state = {}

    rename_map = DualConversionNames.LAYER_NAMES
    matched, renamed, skipped = 0, 0, 0

    for key, tensor in state.items():
        found = False
        for old, new in rename_map.items():
            if old in key:
                new_key = key.replace(old, new)
                new_state[new_key] = tensor
                print(f"[MIGRATE] {key} → {new_key}")
                renamed += 1
                found = True
                break
        if not found:
            if key in adapter.state_dict():
                new_state[key] = tensor
                matched += 1
            else:
                print(f"[SKIP]   {key} not found in target adapter.")
                skipped += 1

    adapter.load_state_dict(new_state, strict=False)

    print(f"\n✅ Converted model loaded from {path}")
    print(f"   🔁 Renamed Keys: {renamed}")
    print(f"   ✅ Direct Matches: {matched}")
    print(f"   ⚠️  Skipped Keys: {skipped}")


def reshape_for_shunt(
    encoder_embeddings: torch.Tensor,
    clip_slice: torch.Tensor,
    adapter_model
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Ensures encoder_embeddings and clip_slice match the required dimensions
    for adapter_model: [B, adapter_seq, adapter_dim].

    Applies sequence interpolation and feature projection as needed.
    """
    return encoder_embeddings, clip_slice
    B, encoder_seq, encoder_dim = encoder_embeddings.shape
    B2, clip_seq, clip_dim = clip_slice.shape

    assert B == B2, "Batch sizes must match"

    # -- Step 1: Interpolate SEQUENCE LENGTH (dim=1) if needed --
    target_seq = max(adapter_model.condition_dim, adapter_model.modulation_dim)

    if clip_seq != target_seq:
        clip_slice = clip_slice.permute(0, 0, 2)  # [B, C, T]
        clip_slice = torch.nn.functional.interpolate(
            clip_slice.float(),
            size=target_seq,
            mode="nearest"
        )
        clip_slice = clip_slice.permute(0, 0, 2)  # [B, T, C]

    if encoder_seq != target_seq:
        encoder_embeddings = encoder_embeddings.permute(0, 0, 2)
        encoder_embeddings = torch.nn.functional.interpolate(
            encoder_embeddings.float(),
            size=target_seq,
            mode="nearest"
        )
        encoder_embeddings = encoder_embeddings.permute(0, 0, 2)

    # -- Step 2: Project FEATURE DIMENSION (dim=2) if needed --
    if clip_slice.size(-1) != adapter_model.condition_dim:
        projection_clip = torch.nn.Linear(
            clip_slice.size(-1),
            adapter_model.condition_dim,
            bias=True,
            device=clip_slice.device
        )
        clip_slice = projection_clip(clip_slice)
        del projection_clip

    if encoder_embeddings.size(-1) != adapter_model.modulation_dim:
        projection_encoder = torch.nn.Linear(
            encoder_embeddings.size(-1),
            adapter_model.modulation_dim,
            bias=True,
            device=encoder_embeddings.device
        )
        encoder_embeddings = projection_encoder(encoder_embeddings)
        del projection_encoder

    return encoder_embeddings, clip_slice
