
import torch
import torch.nn as nn
import torch.nn.functional as F
from .configs import T5_CONFIGS, T5_SHUNT_REPOS



# ─── Residual Pocket Block ───────────────────────────────────
class BottleneckResBlock(nn.Module):
    def __init__(self, dim, kernel=3, dropout=0.1):
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

# ─── Two Stream Shunt Adapter ──────────────────────────────────────
class TwoStreamShuntAdapter(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        self.config = config
        self.dtype = config.get("dtype", torch.float32)
        self.t5_dim = config["t5"]["hidden_size"]
        self.clip_dim = config["clip"]["hidden_size"]
        self.bneck = config["bottleneck"]
        self.heads = config["heads"]
        self.tau_init = config["tau_init"]
        self.max_guidance = config["max_guidance"]

        use_norm   = config.get("layer_norm", True)
        use_do     = config.get("use_dropout", True)
        do_p       = config.get("dropout", 0.1)
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

    def forward(self, t5_seq: torch.Tensor, clip_seq: torch.Tensor):
        if self.config.get("assert_input_dims", True):
            assert t5_seq.size(-1) == self.t5_dim
            assert clip_seq.size(-1) == self.clip_dim

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
        g_pred = g_tok.mean(1, keepdim=True) * self.max_guidance

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

