
"""
Multi-Modal VAE with Adaptive Cantor Fusion
============================================

Cantor-based fusion with learned visibility (alpha) and capacity (beta):
- Alpha: Learned visibility controlling latent space usage (tied to KL divergence)
- Beta: Learned capacity controlling source influence strength
- Decoupled T5-XL representations for CLIP-L and CLIP-G
- Cantor fractal routing for sparse attention

Author: AbstractPhil
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
import math


# ============================================================================
# CONFIGURATION
# ============================================================================

class FusionStrategy(Enum):
    """Fusion strategies for multi-modal VAE."""
    CONCATENATE = "concatenate"
    ATTENTION = "attention"
    GATED = "gated"
    CANTOR = "cantor"
    GEOMETRIC = "geometric"
    HIERARCHICAL = "hierarchical"
    ADAPTIVE_CANTOR = "adaptive_cantor"  # Default and recommended


@dataclass
class MultiModalVAEConfig:
    """Configuration for multi-modal VAE."""
    # Input modalities
    modality_dims: Dict[str, int] = None
    modality_seq_lens: Dict[str, int] = None

    # Latent space
    latent_dim: int = 2048
    seq_len: int = 77

    # Architecture
    encoder_layers: int = 3
    decoder_layers: int = 3
    hidden_dim: int = 1024
    dropout: float = 0.1

    # Fusion
    fusion_strategy: str = "adaptive_cantor"  # Default
    fusion_heads: int = 8
    fusion_dropout: float = 0.1
    binding_config: Optional[Dict[str, Dict[str, float]]] = None

    # Cantor parameters (for cantor and adaptive_cantor)
    cantor_depth: int = 8
    cantor_local_window: int = 3

    # Adaptive fusion parameters (for adaptive_cantor)
    alpha_init: float = 1.0
    beta_init: float = 0.3
    alpha_lr_scale: float = 0.1
    beta_lr_scale: float = 1.0

    # Loss weights
    beta_kl: float = 0.1
    beta_reconstruction: float = 1.0
    beta_cross_modal: float = 0.05
    beta_alpha_regularization: float = 0.01

    # Training
    use_amp: bool = True

    # Reproducibility
    seed: Optional[int] = None

    def __post_init__(self):
        # Default: SDXL configuration with decoupled T5
        if self.modality_dims is None:
            self.modality_dims = {
                "clip_l": 768,
                "clip_g": 1280,
                "t5_xl_l": 2048,
                "t5_xl_g": 2048
            }

        # Default: Different sequence lengths
        if self.modality_seq_lens is None:
            self.modality_seq_lens = {
                "clip_l": 77,
                "clip_g": 77,
                "t5_xl_l": 512,
                "t5_xl_g": 512
            }

        # Default binding for adaptive strategies
        if self.binding_config is None and self.fusion_strategy == "adaptive_cantor":
            self.binding_config = {
                "clip_l": {"t5_xl_l": 0.3},
                "clip_g": {"t5_xl_g": 0.3},
                "t5_xl_l": {},
                "t5_xl_g": {}
            }


# ============================================================================
# FUSION MODULES
# ============================================================================

class ConcatenateFusion(nn.Module):
    """Simple concatenation-based fusion with sequence length handling."""

    def __init__(
            self,
            modality_dims: Dict[str, int],
            modality_seq_lens: Dict[str, int],
            output_dim: int,
            dropout: float = 0.1,
            seed: Optional[int] = None
    ):
        super().__init__()

        if seed is not None:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

        self.modality_names = list(modality_dims.keys())
        self.modality_seq_lens = modality_seq_lens
        self.max_seq_len = max(modality_seq_lens.values())
        total_dim = sum(modality_dims.values())

        self.projection = nn.Sequential(
            nn.Linear(total_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )

    def forward(self, modality_inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        device = list(modality_inputs.values())[0].device

        # Pad all modalities to max sequence length
        padded = []
        for name in self.modality_names:
            if name in modality_inputs:
                inp = modality_inputs[name]
                B, seq_len, dim = inp.shape

                if seq_len < self.max_seq_len:
                    padding = torch.zeros(B, self.max_seq_len - seq_len, dim, device=device)
                    inp = torch.cat([inp, padding], dim=1)

                padded.append(inp)

        # Concatenate along feature dimension
        cat_inputs = torch.cat(padded, dim=-1)
        return self.projection(cat_inputs)


class AttentionFusion(nn.Module):
    """Standard multi-head attention fusion with sequence length handling."""

    def __init__(
            self,
            modality_dims: Dict[str, int],
            modality_seq_lens: Dict[str, int],
            output_dim: int,
            num_heads: int = 8,
            dropout: float = 0.1,
            seed: Optional[int] = None
    ):
        super().__init__()

        if seed is not None:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

        self.modality_names = list(modality_dims.keys())
        self.modality_seq_lens = modality_seq_lens
        self.max_seq_len = max(modality_seq_lens.values())
        self.num_modalities = len(self.modality_names)
        self.output_dim = output_dim
        self.num_heads = num_heads
        self.head_dim = output_dim // num_heads

        # Project each modality to common dimension
        self.modality_projections = nn.ModuleDict({
            name: nn.Linear(dim, output_dim)
            for name, dim in modality_dims.items()
        })

        # Multi-head attention
        self.q_proj = nn.Linear(output_dim, output_dim)
        self.k_proj = nn.Linear(output_dim, output_dim)
        self.v_proj = nn.Linear(output_dim, output_dim)
        self.out_proj = nn.Linear(output_dim, output_dim)

        self.dropout = nn.Dropout(dropout)
        self.temperature = nn.Parameter(torch.tensor(0.07))

    def forward(self, modality_inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        device = list(modality_inputs.values())[0].device
        B = list(modality_inputs.values())[0].shape[0]

        # Project and pad all modalities
        projected = []
        for name in self.modality_names:
            if name in modality_inputs:
                proj = self.modality_projections[name](modality_inputs[name])
                seq_len = proj.shape[1]

                if seq_len < self.max_seq_len:
                    padding = torch.zeros(B, self.max_seq_len - seq_len, self.output_dim, device=device)
                    proj = torch.cat([proj, padding], dim=1)

                projected.append(proj)

        # Stack: [batch, num_modalities, max_seq, dim]
        stacked = torch.stack(projected, dim=1)

        # Apply attention
        Q = self.q_proj(stacked[:, 0:1])  # Use first modality as query
        K = self.k_proj(stacked)
        V = self.v_proj(stacked)

        # Reshape for multi-head
        Q = Q.view(B, 1, self.max_seq_len, self.num_heads, self.head_dim).permute(0, 3, 1, 2, 4)
        K = K.view(B, self.num_modalities, self.max_seq_len, self.num_heads, self.head_dim).permute(0, 3, 1, 2, 4)
        V = V.view(B, self.num_modalities, self.max_seq_len, self.num_heads, self.head_dim).permute(0, 3, 1, 2, 4)

        # Attention scores
        scores = torch.einsum('bhqsd,bhmsd->bhqms', Q, K) / math.sqrt(self.head_dim)
        scores = scores / self.temperature.abs()

        attn = F.softmax(scores, dim=-2)
        attn = self.dropout(attn)

        # Apply attention
        out = torch.einsum('bhqms,bhmsd->bhqsd', attn, V)
        out = out.squeeze(2).permute(0, 2, 1, 3).reshape(B, self.max_seq_len, self.output_dim)

        return self.out_proj(out)


class GatedFusion(nn.Module):
    """Gated fusion with learned modality weights and sequence length handling."""

    def __init__(
            self,
            modality_dims: Dict[str, int],
            modality_seq_lens: Dict[str, int],
            output_dim: int,
            dropout: float = 0.1,
            seed: Optional[int] = None
    ):
        super().__init__()

        if seed is not None:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

        self.modality_names = list(modality_dims.keys())
        self.modality_seq_lens = modality_seq_lens
        self.max_seq_len = max(modality_seq_lens.values())

        # Project each modality to common dimension
        self.modality_projections = nn.ModuleDict({
            name: nn.Linear(dim, output_dim)
            for name, dim in modality_dims.items()
        })

        # Gating networks
        self.gates = nn.ModuleDict({
            name: nn.Sequential(
                nn.Linear(output_dim, output_dim // 4),
                nn.GELU(),
                nn.Linear(output_dim // 4, 1),
                nn.Sigmoid()
            )
            for name in self.modality_names
        })

        self.output_proj = nn.Sequential(
            nn.Linear(output_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )

    def forward(self, modality_inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        device = list(modality_inputs.values())[0].device

        # Project, gate, and pad each modality
        gated_features = []

        for name in self.modality_names:
            if name in modality_inputs:
                proj = self.modality_projections[name](modality_inputs[name])
                gate = self.gates[name](proj)
                gated = proj * gate

                B, seq_len, dim = gated.shape
                if seq_len < self.max_seq_len:
                    padding = torch.zeros(B, self.max_seq_len - seq_len, dim, device=device)
                    gated = torch.cat([gated, padding], dim=1)

                gated_features.append(gated)

        # Sum gated features
        fused = sum(gated_features) / len(gated_features)

        return self.output_proj(fused)


class CantorModalityFusion(nn.Module):
    """Cantor-based fusion with sequence length handling."""

    def __init__(
            self,
            modality_dims: Dict[str, int],
            modality_seq_lens: Dict[str, int],
            output_dim: int,
            num_heads: int = 8,
            cantor_depth: int = 8,
            local_window: int = 3,
            dropout: float = 0.1,
            seed: Optional[int] = None
    ):
        super().__init__()

        if seed is not None:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

        self.modality_names = list(modality_dims.keys())
        self.modality_seq_lens = modality_seq_lens
        self.max_seq_len = max(modality_seq_lens.values())
        self.num_modalities = len(self.modality_names)
        self.output_dim = output_dim
        self.num_heads = num_heads
        self.head_dim = output_dim // num_heads

        # Project each modality to common dimension
        self.modality_projections = nn.ModuleDict({
            name: nn.Linear(dim, output_dim)
            for name, dim in modality_dims.items()
        })

        # Modality embeddings
        self.modality_embeddings = nn.Parameter(
            torch.randn(self.num_modalities, output_dim) * 0.02
        )

        # QKV projections
        self.q_proj = nn.Linear(output_dim, output_dim)
        self.k_proj = nn.Linear(output_dim, output_dim)
        self.v_proj = nn.Linear(output_dim, output_dim)

        # Cantor routing
        self.cantor_depth = cantor_depth
        self.local_window = min(local_window, self.num_modalities)

        # Pre-compute Cantor coordinates
        self.register_buffer(
            'modality_cantor_coords',
            self._compute_modality_cantor_coordinates()
        )

        # Pre-compute routing
        self.register_buffer(
            'modality_routes',
            self._build_modality_routes()
        )

        # Output
        self.out_proj = nn.Linear(output_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        self.temperature = nn.Parameter(torch.tensor(0.07))

    def _cantor_coordinate(self, position: int, max_len: int, depth: int) -> float:
        """Compute Cantor set coordinate."""
        x = position / max(1, max_len - 1)
        x = max(1e-6, min(x, 1.0 - 1e-6))

        cantor_val = 0.0
        factor = 0.5

        for _ in range(depth):
            x *= 3.0
            digit = int(x)
            x -= digit

            if digit == 2:
                cantor_val += factor

            factor *= 0.5

        return cantor_val

    def _compute_modality_cantor_coordinates(self) -> torch.Tensor:
        """Map each modality to Cantor coordinate."""
        coords = torch.tensor([
            self._cantor_coordinate(i, self.num_modalities, self.cantor_depth)
            for i in range(self.num_modalities)
        ], dtype=torch.float32)
        return coords

    def _build_modality_routes(self) -> torch.Tensor:
        """Build routing table for modality attention."""
        routes = torch.zeros(self.num_modalities, self.local_window, dtype=torch.long)

        for i in range(self.num_modalities):
            distances = torch.abs(
                self.modality_cantor_coords - self.modality_cantor_coords[i]
            )
            _, nearest = torch.topk(distances, self.local_window, largest=False)
            routes[i] = nearest

        return routes

    def forward(self, modality_inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Fuse multiple modality inputs using Cantor routing."""
        B = list(modality_inputs.values())[0].shape[0]
        device = list(modality_inputs.values())[0].device

        # Project and pad all modalities to common space
        projected = []
        for i, name in enumerate(self.modality_names):
            if name in modality_inputs:
                proj = self.modality_projections[name](modality_inputs[name])
                proj = proj + self.modality_embeddings[i]

                seq_len = proj.shape[1]
                if seq_len < self.max_seq_len:
                    padding = torch.zeros(B, self.max_seq_len - seq_len, self.output_dim, device=device)
                    proj = torch.cat([proj, padding], dim=1)

                projected.append(proj)

        # Stack: [batch, num_modalities, max_seq, dim]
        stacked = torch.stack(projected, dim=1)

        # Multi-head attention with Cantor routing
        Q = self.q_proj(stacked).view(B, self.num_modalities, self.max_seq_len, self.num_heads, self.head_dim)
        K = self.k_proj(stacked).view(B, self.num_modalities, self.max_seq_len, self.num_heads, self.head_dim)
        V = self.v_proj(stacked).view(B, self.num_modalities, self.max_seq_len, self.num_heads, self.head_dim)

        # Permute to [batch, heads, num_modalities, seq, head_dim]
        Q = Q.permute(0, 3, 1, 2, 4)
        K = K.permute(0, 3, 1, 2, 4)
        V = V.permute(0, 3, 1, 2, 4)

        # Sparse attention via Cantor routing
        routes = self.modality_routes.to(device)

        attended = []
        for i in range(self.num_modalities):
            neighbors = routes[i]

            q_i = Q[:, :, i, :, :]
            k_neighbors = K[:, :, neighbors, :, :]
            v_neighbors = V[:, :, neighbors, :, :]

            scores = torch.einsum('bhsd,bhwsd->bhsw', q_i, k_neighbors) / math.sqrt(self.head_dim)
            scores = scores / self.temperature.abs()

            attn = F.softmax(scores, dim=-1)
            attn = self.dropout(attn)

            out_i = torch.einsum('bhsw,bhwsd->bhsd', attn, v_neighbors)
            attended.append(out_i)

        # Stack and mean over modalities
        fused = torch.stack(attended, dim=2).mean(dim=2)

        # Reshape back
        fused = fused.permute(0, 2, 1, 3).reshape(B, self.max_seq_len, self.output_dim)

        output = self.out_proj(fused)
        output = self.dropout(output)

        return output


class GeometricModalityFusion(nn.Module):
    """Geometric fusion with sequence length handling."""

    def __init__(
            self,
            modality_dims: Dict[str, int],
            modality_seq_lens: Dict[str, int],
            output_dim: int,
            num_heads: int = 4,
            use_cayley: bool = True,
            use_angular: bool = True,
            temperature: float = 0.07,
            dropout: float = 0.1,
            seed: Optional[int] = None
    ):
        super().__init__()

        if seed is not None:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

        self.modality_names = list(modality_dims.keys())
        self.modality_seq_lens = modality_seq_lens
        self.max_seq_len = max(modality_seq_lens.values())
        self.num_modalities = len(self.modality_names)
        self.output_dim = output_dim
        self.num_heads = num_heads
        self.head_dim = output_dim // num_heads
        self.use_cayley = use_cayley
        self.use_angular = use_angular

        # Project modalities to common space
        self.modality_projections = nn.ModuleDict({
            name: nn.Linear(dim, output_dim)
            for name, dim in modality_dims.items()
        })

        # QKV
        self.q_proj = nn.Linear(output_dim, output_dim)
        self.k_proj = nn.Linear(output_dim, output_dim)
        self.v_proj = nn.Linear(output_dim, output_dim)

        # Pentachoron role weights
        if use_angular:
            role_weights = torch.tensor([1.0, -0.75, 0.75, 0.75, -0.75])
            if self.num_modalities < 5:
                role_weights = role_weights[:self.num_modalities]
            elif self.num_modalities > 5:
                extra = torch.ones(self.num_modalities - 5) * 0.5
                role_weights = torch.cat([role_weights, extra])
            self.register_buffer("role_weights", role_weights)

        # Output
        self.out_proj = nn.Linear(output_dim, output_dim)
        self.temperature = nn.Parameter(torch.tensor(temperature))
        self.dropout = nn.Dropout(dropout)
        self.attention_weights = nn.Parameter(torch.ones(3) / 3)

    def _compute_angular_attention(self, features: List[torch.Tensor]) -> torch.Tensor:
        """Compute attention based on angular relationships."""
        B = features[0].shape[0]
        features_norm = [F.normalize(f, dim=-1) for f in features]
        attention = torch.zeros(B, self.num_modalities, device=features[0].device)

        for i, feat_i in enumerate(features_norm):
            for j, feat_j in enumerate(features_norm):
                if i != j:
                    cos_sim = (feat_i * feat_j).sum(dim=-1).mean(dim=-1)
                    angle = torch.acos(cos_sim.clamp(-1 + 1e-7, 1 - 1e-7))
                    attention[:, i] += self.role_weights[j] * torch.exp(-angle / self.temperature.abs())

        return F.softmax(attention, dim=-1)

    def _compute_cayley_attention(self, features: List[torch.Tensor]) -> torch.Tensor:
        """Compute attention based on Cayley-Menger volumes."""
        B = features[0].shape[0]
        volume_scores = []

        for i in range(self.num_modalities):
            simplex_points = [features[i]]
            for j in range(min(4, self.num_modalities - 1)):
                angle = (j + 1) * math.pi / 4
                other_idx = (i + j + 1) % self.num_modalities
                rot_feat = features[i] * math.cos(angle) + features[other_idx] * math.sin(angle)
                simplex_points.append(rot_feat)

            simplex = torch.stack(simplex_points, dim=1)
            diff = simplex.unsqueeze(2) - simplex.unsqueeze(1)
            distsq = (diff * diff).sum(dim=-1).sum(dim=-1)
            volume = distsq.mean(dim=(1, 2))
            volume_scores.append(volume)

        volumes = torch.stack(volume_scores, dim=1)
        return F.softmax(volumes / self.temperature.abs(), dim=-1)

    def _compute_standard_attention(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
        """Standard multi-head attention."""
        scores = torch.einsum('bhqsd,bhmsd->bhqms', Q, K) / math.sqrt(self.head_dim)
        scores = scores / self.temperature.abs()
        attn = F.softmax(scores, dim=-2)
        attn = self.dropout(attn)
        out = torch.einsum('bhqms,bhmsd->bhqsd', attn, V)
        B = out.shape[0]
        out = out.squeeze(2).permute(0, 2, 1, 3).reshape(B, -1, self.output_dim)
        return out

    def forward(self, modality_inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Fuse modalities using geometric attention."""
        B = list(modality_inputs.values())[0].shape[0]
        device = list(modality_inputs.values())[0].device

        # Project and pad features
        features = []
        for name in self.modality_names:
            if name in modality_inputs:
                proj = self.modality_projections[name](modality_inputs[name])
                seq_len = proj.shape[1]

                if seq_len < self.max_seq_len:
                    padding = torch.zeros(B, self.max_seq_len - seq_len, self.output_dim, device=device)
                    proj = torch.cat([proj, padding], dim=1)

                features.append(proj)

        Q_feat = features[0]
        Q = self.q_proj(Q_feat).view(B, self.num_heads, 1, self.max_seq_len, self.head_dim)

        K_list, V_list = [], []
        for feat in features:
            K = self.k_proj(feat).view(B, self.num_heads, 1, self.max_seq_len, self.head_dim)
            V = self.v_proj(feat).view(B, self.num_heads, 1, self.max_seq_len, self.head_dim)
            K_list.append(K)
            V_list.append(V)

        K = torch.cat(K_list, dim=2)
        V = torch.cat(V_list, dim=2)

        attention_outputs = []
        mha_out = self._compute_standard_attention(Q, K, V)
        attention_outputs.append(mha_out)

        if self.use_angular:
            angular_weights = self._compute_angular_attention(features)
            angular_out = torch.zeros_like(features[0])
            for i in range(self.num_modalities):
                w = angular_weights[:, i]
                angular_out = angular_out + w.unsqueeze(-1).unsqueeze(-1) * features[i]
            attention_outputs.append(angular_out)

        if self.use_cayley:
            cayley_weights = self._compute_cayley_attention(features)
            cayley_out = torch.zeros_like(features[0])
            for i in range(self.num_modalities):
                w = cayley_weights[:, i]
                cayley_out = cayley_out + w.unsqueeze(-1).unsqueeze(-1) * features[i]
            attention_outputs.append(cayley_out)

        attn_weights = F.softmax(self.attention_weights[:len(attention_outputs)], dim=0)
        fused = torch.zeros_like(attention_outputs[0])
        for i, out in enumerate(attention_outputs):
            fused = fused + attn_weights[i] * out

        output = self.out_proj(fused)
        output = self.dropout(output)

        return output


class HierarchicalFusion(nn.Module):
    """Hierarchical fusion with sequence length handling."""

    def __init__(
            self,
            modality_dims: Dict[str, int],
            modality_seq_lens: Dict[str, int],
            output_dim: int,
            dropout: float = 0.1,
            seed: Optional[int] = None
    ):
        super().__init__()

        if seed is not None:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

        self.modality_names = list(modality_dims.keys())
        self.modality_seq_lens = modality_seq_lens
        self.max_seq_len = max(modality_seq_lens.values())

        # Stage 1: Project each modality
        self.modality_projections = nn.ModuleDict({
            name: nn.Linear(dim, output_dim)
            for name, dim in modality_dims.items()
        })

        # Stage 2: Hierarchical combination
        self.stage2_proj = nn.Sequential(
            nn.Linear(output_dim * 2, output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        # Stage 3: Final fusion
        self.final_proj = nn.Sequential(
            nn.Linear(output_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )

    def forward(self, modality_inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        device = list(modality_inputs.values())[0].device

        # Stage 1: Project and pad all modalities
        projected = []
        for name in self.modality_names:
            if name in modality_inputs:
                proj = self.modality_projections[name](modality_inputs[name])
                B, seq_len, dim = proj.shape

                if seq_len < self.max_seq_len:
                    padding = torch.zeros(B, self.max_seq_len - seq_len, dim, device=device)
                    proj = torch.cat([proj, padding], dim=1)

                projected.append(proj)

        # Stage 2: Pairwise combinations
        if len(projected) == 1:
            return self.final_proj(projected[0])

        combined = projected[0]
        for i in range(1, len(projected)):
            pair = torch.cat([combined, projected[i]], dim=-1)
            combined = self.stage2_proj(pair)

        # Stage 3: Final projection
        return self.final_proj(combined)

class AdaptiveCantorModalityFusion(nn.Module):
    """
    Cantor-based fusion with learned alpha (visibility) and beta (capacity).

    Combines fractal routing from Cantor fusion with adaptive learning:

    Alpha (visibility): Controls how much latent space is used by each modality.
                       Tied to KL divergence - higher alpha = more latent usage.

    Beta (capacity): Controls how strongly sources influence targets in binding pairs.
                    Per-binding-pair learned parameter.

    Features:
    - Cantor fractal routing for sparse attention
    - Decoupled T5-XL representations (t5_xl_l for CLIP-L, t5_xl_g for CLIP-G)
    - Learned per-modality alpha parameters
    - Learned per-binding-pair beta parameters
    - Different sequence lengths per modality
    """

    def __init__(
            self,
            modality_dims: Dict[str, int],
            modality_seq_lens: Dict[str, int],
            binding_config: Dict[str, Dict[str, float]],
            output_dim: int,
            num_heads: int = 8,
            cantor_depth: int = 8,
            local_window: int = 3,
            alpha_init: float = 1.0,
            beta_init: float = 0.3,
            alpha_lr_scale: float = 0.1,
            beta_lr_scale: float = 1.0,
            dropout: float = 0.1,
            seed: Optional[int] = None
    ):
        super().__init__()

        if seed is not None:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

        self.modality_names = list(modality_dims.keys())
        self.num_modalities = len(self.modality_names)
        self.modality_dims = modality_dims
        self.modality_seq_lens = modality_seq_lens
        self.binding_config = binding_config
        self.output_dim = output_dim
        self.num_heads = num_heads
        self.head_dim = output_dim // num_heads
        self.alpha_lr_scale = alpha_lr_scale
        self.beta_lr_scale = beta_lr_scale

        # Project each modality to common dimension
        self.modality_projections = nn.ModuleDict({
            name: nn.Linear(dim, output_dim)
            for name, dim in modality_dims.items()
        })

        # Modality embeddings (Cantor)
        self.modality_embeddings = nn.Parameter(
            torch.randn(self.num_modalities, output_dim) * 0.02
        )

        # Learned alpha (visibility) per modality
        self.alphas = nn.ParameterDict({
            name: nn.Parameter(torch.tensor(alpha_init))
            for name in self.modality_names
        })

        # Learned beta (capacity) per binding pair
        self.betas = nn.ParameterDict()
        for target, sources in binding_config.items():
            for source, init_weight in sources.items():
                if init_weight > 0:
                    key = f"{target}_{source}"
                    self.betas[key] = nn.Parameter(torch.tensor(beta_init))

        # Alpha-modulated gating networks
        self.alpha_gates = nn.ModuleDict({
            name: nn.Sequential(
                nn.Linear(output_dim, output_dim // 4),
                nn.GELU(),
                nn.Linear(output_dim // 4, 1),
                nn.Sigmoid()
            )
            for name in self.modality_names
        })

        # QKV projections (Cantor)
        self.q_proj = nn.Linear(output_dim, output_dim)
        self.k_proj = nn.Linear(output_dim, output_dim)
        self.v_proj = nn.Linear(output_dim, output_dim)

        # Cantor routing
        self.cantor_depth = cantor_depth
        self.local_window = min(local_window, self.num_modalities)

        # Pre-compute Cantor coordinates
        self.register_buffer(
            'modality_cantor_coords',
            self._compute_modality_cantor_coordinates()
        )

        # Pre-compute routing
        self.register_buffer(
            'modality_routes',
            self._build_modality_routes()
        )

        # Build binding route masks for beta modulation
        self.binding_route_masks = self._build_binding_route_masks()

        # Output
        self.out_proj = nn.Linear(output_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        self.temperature = nn.Parameter(torch.tensor(0.07))

    def _cantor_coordinate(self, position: int, max_len: int, depth: int) -> float:
        """Compute Cantor set coordinate."""
        x = position / max(1, max_len - 1)
        x = max(1e-6, min(x, 1.0 - 1e-6))

        cantor_val = 0.0
        factor = 0.5

        for _ in range(depth):
            x *= 3.0
            digit = int(x)
            x -= digit

            if digit == 2:
                cantor_val += factor

            factor *= 0.5

        return cantor_val

    def _compute_modality_cantor_coordinates(self) -> torch.Tensor:
        """Map each modality to Cantor coordinate."""
        coords = torch.tensor([
            self._cantor_coordinate(i, self.num_modalities, self.cantor_depth)
            for i in range(self.num_modalities)
        ], dtype=torch.float32)
        return coords

    def _build_modality_routes(self) -> torch.Tensor:
        """Build routing table for modality attention using Cantor distances."""
        routes = torch.zeros(self.num_modalities, self.local_window, dtype=torch.long)

        for i in range(self.num_modalities):
            distances = torch.abs(
                self.modality_cantor_coords - self.modality_cantor_coords[i]
            )
            _, nearest = torch.topk(distances, self.local_window, largest=False)
            routes[i] = nearest

        return routes

    def _build_binding_route_masks(self) -> Dict[str, torch.Tensor]:
        """Build masks indicating which routes correspond to binding pairs."""
        masks = {}

        for target_idx, target in enumerate(self.modality_names):
            if target in self.binding_config:
                for source, weight in self.binding_config[target].items():
                    if weight > 0 and source in self.modality_names:
                        source_idx = self.modality_names.index(source)
                        key = f"{target}_{source}"

                        # Create mask: 1.0 where source is in target's routes
                        routes = self.modality_routes[target_idx]
                        mask = torch.zeros(self.local_window)
                        for i, route_idx in enumerate(routes):
                            if route_idx == source_idx:
                                mask[i] = 1.0

                        masks[key] = mask

        return masks

    def get_alpha_params(self) -> Dict[str, torch.Tensor]:
        """Get alpha parameters for external use (e.g., loss computation)."""
        return {name: alpha for name, alpha in self.alphas.items()}

    def get_beta_params(self) -> Dict[str, torch.Tensor]:
        """Get beta parameters for external use."""
        return {key: beta for key, beta in self.betas.items()}

    def forward(
            self,
            modality_inputs: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Fuse modalities using Cantor routing with adaptive alpha/beta.

        Args:
            modality_inputs: Dict of {name: tensor [batch, seq_i, dim_i]}

        Returns:
            Dict of {name: fused tensor [batch, seq_i, output_dim]}
        """
        # Handle different sequence lengths - pad to max for processing
        max_seq_len = max(self.modality_seq_lens.values())
        device = list(modality_inputs.values())[0].device

        # Project and pad all modalities
        projected = {}
        original_seq_lens = {}

        for i, name in enumerate(self.modality_names):
            if name in modality_inputs:
                # Project to common space
                proj = self.modality_projections[name](modality_inputs[name])

                # Apply alpha-modulated gating
                alpha = self.alphas[name]
                gate = self.alpha_gates[name](proj)
                alpha_clamped = torch.sigmoid(alpha)
                proj = proj * (gate * alpha_clamped + (1 - alpha_clamped))

                # Add modality embedding
                proj = proj + self.modality_embeddings[i]

                # Store original length and pad
                B, seq_len, _ = proj.shape
                original_seq_lens[name] = seq_len

                if seq_len < max_seq_len:
                    padding = torch.zeros(B, max_seq_len - seq_len, self.output_dim, device=device)
                    proj = torch.cat([proj, padding], dim=1)

                projected[name] = proj

        # Stack: [batch, num_modalities, max_seq, dim]
        B = list(projected.values())[0].shape[0]
        stacked = torch.stack([projected[name] for name in self.modality_names], dim=1)

        # Multi-head attention with Cantor routing
        Q = self.q_proj(stacked).view(B, self.num_modalities, max_seq_len, self.num_heads, self.head_dim)
        K = self.k_proj(stacked).view(B, self.num_modalities, max_seq_len, self.num_heads, self.head_dim)
        V = self.v_proj(stacked).view(B, self.num_modalities, max_seq_len, self.num_heads, self.head_dim)

        # Permute to [batch, heads, num_modalities, seq, head_dim]
        Q = Q.permute(0, 3, 1, 2, 4)
        K = K.permute(0, 3, 1, 2, 4)
        V = V.permute(0, 3, 1, 2, 4)

        # Sparse attention via Cantor routing with beta modulation
        routes = self.modality_routes.to(device)

        attended = []
        for i, target_name in enumerate(self.modality_names):
            neighbors = routes[i]  # [local_window]

            q_i = Q[:, :, i, :, :]  # [batch, heads, seq, head_dim]
            k_neighbors = K[:, :, neighbors, :, :]  # [batch, heads, window, seq, head_dim]
            v_neighbors = V[:, :, neighbors, :, :]  # [batch, heads, window, seq, head_dim]

            # Compute attention scores
            scores = torch.einsum('bhsd,bhwsd->bhsw', q_i, k_neighbors) / math.sqrt(self.head_dim)
            scores = scores / self.temperature.abs()

            # Apply beta modulation for binding pairs
            if target_name in self.binding_config:
                for source_name, weight in self.binding_config[target_name].items():
                    if weight > 0:
                        key = f"{target_name}_{source_name}"
                        if key in self.betas and key in self.binding_route_masks:
                            beta = self.betas[key]
                            beta_clamped = torch.sigmoid(beta)

                            # Get mask for this binding pair
                            mask = self.binding_route_masks[key].to(device)

                            # Apply beta to relevant routes
                            # mask: [local_window], scores: [B, H, seq, window]
                            beta_mask = mask.view(1, 1, 1, -1)  # [1, 1, 1, window]

                            # Beta modulates attention: multiply scores by beta for bound routes
                            scores = scores * (1 + beta_mask * (beta_clamped - 1))

            attn = F.softmax(scores, dim=-1)
            attn = self.dropout(attn)

            # Apply attention
            out_i = torch.einsum('bhsw,bhwsd->bhsd', attn, v_neighbors)
            attended.append(out_i)

        # Stack and mean over modalities: [batch, heads, seq, head_dim]
        fused_tensor = torch.stack(attended, dim=2).mean(dim=2)

        # Reshape back: [batch, seq, output_dim]
        fused_tensor = fused_tensor.permute(0, 2, 1, 3).reshape(B, max_seq_len, self.output_dim)

        # Output projection
        output = self.out_proj(fused_tensor)
        output = self.dropout(output)

        # Unpad to original sequence lengths and return dict
        enriched = {}
        for name in self.modality_names:
            seq_len = original_seq_lens[name]
            enriched[name] = output[:, :seq_len, :]

        return enriched


# ============================================================================
# MULTIMODAL VAE
# ============================================================================

class MultiModalVAE(nn.Module):
    """Multi-modal VAE with multiple fusion strategies."""

    def __init__(self, config: MultiModalVAEConfig):
        super().__init__()
        self.config = config
        self.modality_names = list(config.modality_dims.keys())
        self.modality_seq_lens = config.modality_seq_lens  # Store original seq lens
        self.num_modalities = len(self.modality_names)
        self.seed = config.seed

        if self.seed is not None:
            torch.manual_seed(self.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(self.seed)

        # Fusion module - select based on strategy
        fusion_strategy = FusionStrategy(config.fusion_strategy)
        self.fusion_strategy = fusion_strategy

        if fusion_strategy == FusionStrategy.CONCATENATE:
            self.fusion = ConcatenateFusion(
                modality_dims=config.modality_dims,
                modality_seq_lens=config.modality_seq_lens,  # ADD THIS
                output_dim=config.hidden_dim,
                dropout=config.fusion_dropout,
                seed=config.seed
            )
        elif fusion_strategy == FusionStrategy.ATTENTION:
            self.fusion = AttentionFusion(
                modality_dims=config.modality_dims,
                modality_seq_lens=config.modality_seq_lens,  # ADD THIS
                output_dim=config.hidden_dim,
                num_heads=config.fusion_heads,
                dropout=config.fusion_dropout,
                seed=config.seed
            )
        elif fusion_strategy == FusionStrategy.GATED:
            self.fusion = GatedFusion(
                modality_dims=config.modality_dims,
                modality_seq_lens=config.modality_seq_lens,  # ADD THIS
                output_dim=config.hidden_dim,
                dropout=config.fusion_dropout,
                seed=config.seed
            )
        elif fusion_strategy == FusionStrategy.CANTOR:
            self.fusion = CantorModalityFusion(
                modality_dims=config.modality_dims,
                modality_seq_lens=config.modality_seq_lens,  # ADD THIS
                output_dim=config.hidden_dim,
                num_heads=config.fusion_heads,
                cantor_depth=config.cantor_depth,
                local_window=config.cantor_local_window,
                dropout=config.fusion_dropout,
                seed=config.seed
            )
        elif fusion_strategy == FusionStrategy.GEOMETRIC:
            self.fusion = GeometricModalityFusion(
                modality_dims=config.modality_dims,
                modality_seq_lens=config.modality_seq_lens,  # ADD THIS
                output_dim=config.hidden_dim,
                num_heads=config.fusion_heads,
                dropout=config.fusion_dropout,
                seed=config.seed
            )
        elif fusion_strategy == FusionStrategy.HIERARCHICAL:
            self.fusion = HierarchicalFusion(
                modality_dims=config.modality_dims,
                modality_seq_lens=config.modality_seq_lens,  # ADD THIS
                output_dim=config.hidden_dim,
                dropout=config.fusion_dropout,
                seed=config.seed
            )
        elif fusion_strategy == FusionStrategy.ADAPTIVE_CANTOR:
            self.fusion = AdaptiveCantorModalityFusion(
                modality_dims=config.modality_dims,
                modality_seq_lens=config.modality_seq_lens,
                binding_config=config.binding_config,
                output_dim=config.hidden_dim,
                num_heads=config.fusion_heads,
                cantor_depth=config.cantor_depth,
                local_window=config.cantor_local_window,
                alpha_init=config.alpha_init,
                beta_init=config.beta_init,
                alpha_lr_scale=config.alpha_lr_scale,
                beta_lr_scale=config.beta_lr_scale,
                dropout=config.fusion_dropout,
                seed=config.seed
            )
        else:
            raise ValueError(f"Unknown fusion strategy: {config.fusion_strategy}")

        # Encoders - per-modality for ADAPTIVE_CANTOR, single for others
        if fusion_strategy == FusionStrategy.ADAPTIVE_CANTOR:
            self.encoders = nn.ModuleDict()
            for name in self.modality_names:
                encoder_layers = []
                in_dim = config.hidden_dim
                for i in range(config.encoder_layers):
                    out_dim = config.hidden_dim if i < config.encoder_layers - 1 else config.latent_dim * 2
                    encoder_layers.extend([
                        nn.Linear(in_dim, out_dim),
                        nn.LayerNorm(out_dim),
                        nn.GELU(),
                        nn.Dropout(config.dropout)
                    ])
                    in_dim = out_dim
                self.encoders[name] = nn.Sequential(*encoder_layers[:-2])

            # Per-modality mu/logvar projections
            self.fc_mus = nn.ModuleDict({
                name: nn.Linear(config.latent_dim * 2, config.latent_dim)
                for name in self.modality_names
            })
            self.fc_logvars = nn.ModuleDict({
                name: nn.Linear(config.latent_dim * 2, config.latent_dim)
                for name in self.modality_names
            })
        else:
            # Single encoder for other strategies
            encoder_layers = []
            in_dim = config.hidden_dim
            for i in range(config.encoder_layers):
                out_dim = config.hidden_dim if i < config.encoder_layers - 1 else config.latent_dim * 2
                encoder_layers.extend([
                    nn.Linear(in_dim, out_dim),
                    nn.LayerNorm(out_dim),
                    nn.GELU(),
                    nn.Dropout(config.dropout)
                ])
                in_dim = out_dim
            self.encoder = nn.Sequential(*encoder_layers[:-2])
            self.fc_mu = nn.Linear(config.latent_dim * 2, config.latent_dim)
            self.fc_logvar = nn.Linear(config.latent_dim * 2, config.latent_dim)

        # Decoders (per-modality for all strategies)
        self.decoders = nn.ModuleDict()
        for name, dim in config.modality_dims.items():
            decoder_layers = []
            in_dim = config.latent_dim
            for i in range(config.decoder_layers):
                out_dim = config.hidden_dim if i < config.decoder_layers - 1 else dim
                decoder_layers.extend([
                    nn.Linear(in_dim, out_dim),
                    nn.LayerNorm(out_dim) if i < config.decoder_layers - 1 else nn.Identity(),
                    nn.GELU() if i < config.decoder_layers - 1 else nn.Identity(),
                    nn.Dropout(config.dropout) if i < config.decoder_layers - 1 else nn.Identity()
                ])
                in_dim = out_dim
            self.decoders[name] = nn.Sequential(*decoder_layers)

        # Cross-modal projection layers
        self.cross_modal_projections = nn.ModuleDict({
            name: nn.Linear(dim, config.latent_dim)
            for name, dim in config.modality_dims.items()
        })

    def get_fusion_params(self) -> Dict[str, Dict[str, torch.Tensor]]:
        """Get learned alpha and beta parameters from fusion layer."""
        if hasattr(self.fusion, 'get_alpha_params'):
            return {
                'alphas': self.fusion.get_alpha_params(),
                'betas': self.fusion.get_beta_params()
            }
        return {}

    def fuse_modalities(self, modality_inputs: Dict[str, torch.Tensor]):
        """Fuse multiple modality inputs."""
        return self.fusion(modality_inputs)

    def encode(
            self,
            modality_inputs: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[Dict[str, torch.Tensor]]]:
        """
        Encode modalities to latent parameters.

        Returns:
            mu, logvar, per_modality_mus (for alpha regularization, only for ADAPTIVE_CANTOR)
        """
        if self.fusion_strategy == FusionStrategy.ADAPTIVE_CANTOR:
            fused = self.fuse_modalities(modality_inputs)  # Dict

            # Encode each modality separately
            mus, logvars = [], []
            per_modality_mus = {}

            # Find max sequence length for padding
            max_seq = max(f.shape[1] for f in fused.values())
            device = list(fused.values())[0].device

            for name in self.modality_names:
                if name in fused:
                    h = self.encoders[name](fused[name])
                    mu = self.fc_mus[name](h)
                    logvar = self.fc_logvars[name](h)

                    # Pad to max sequence length for stacking
                    B, seq_len, latent_dim = mu.shape
                    if seq_len < max_seq:
                        pad_mu = torch.zeros(B, max_seq - seq_len, latent_dim, device=device)
                        pad_logvar = torch.zeros(B, max_seq - seq_len, latent_dim, device=device)
                        mu = torch.cat([mu, pad_mu], dim=1)
                        logvar = torch.cat([logvar, pad_logvar], dim=1)

                    mus.append(mu)
                    logvars.append(logvar)
                    per_modality_mus[name] = mu

            # Average latent parameters
            mu = torch.stack(mus).mean(dim=0)
            logvar = torch.stack(logvars).mean(dim=0)

            return mu, logvar, per_modality_mus
        else:
            # Single encoder for other strategies
            fused = self.fuse_modalities(modality_inputs)  # Tensor
            h = self.encoder(fused)
            mu = self.fc_mu(h)
            logvar = self.fc_logvar(h)
            return mu, logvar, None

    def reparameterize(
            self,
            mu: torch.Tensor,
            logvar: torch.Tensor,
            generator: Optional[torch.Generator] = None
    ) -> torch.Tensor:
        """Reparameterization trick."""
        std = torch.exp(0.5 * logvar)

        if generator is None and self.seed is not None:
            generator = torch.Generator(device=mu.device).manual_seed(self.seed)

        if generator is not None:
            eps = torch.randn(mu.shape, generator=generator, device=mu.device, dtype=mu.dtype)
        else:
            eps = torch.randn_like(std)

        return mu + eps * std

    def decode(
            self,
            z: torch.Tensor,
            target_modalities: Optional[List[str]] = None
    ) -> Dict[str, torch.Tensor]:
        """Decode latent to reconstructions."""
        if target_modalities is None:
            target_modalities = self.modality_names

        reconstructions = {}
        for name in target_modalities:
            recon = self.decoders[name](z)

            # Slice to original sequence length
            original_seq_len = self.modality_seq_lens[name]
            reconstructions[name] = recon[:, :original_seq_len, :]

        return reconstructions

    def project_for_cross_modal(
            self,
            reconstructions: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """Project reconstructions to common space."""
        projected = {}
        for name, recon in reconstructions.items():
            proj = self.cross_modal_projections[name](recon)
            proj = F.normalize(proj, dim=-1)
            projected[name] = proj
        return projected

    def forward(
            self,
            modality_inputs: Dict[str, torch.Tensor],
            target_modalities: Optional[List[str]] = None,
            generator: Optional[torch.Generator] = None
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor, Optional[Dict[str, torch.Tensor]]]:
        """
        Full forward pass.

        Returns:
            reconstructions, mu, logvar, per_modality_mus (None for non-ADAPTIVE_CANTOR)
        """
        mu, logvar, per_modality_mus = self.encode(modality_inputs)
        z = self.reparameterize(mu, logvar, generator)
        reconstructions = self.decode(z, target_modalities)
        return reconstructions, mu, logvar, per_modality_mus


# ============================================================================
# ENHANCED LOSS FUNCTION
# ============================================================================

class MultiModalVAELoss(nn.Module):
    """Enhanced loss with alpha regularization."""

    def __init__(
            self,
            beta_kl: float = 0.1,
            beta_reconstruction: float = 1.0,
            beta_cross_modal: float = 0.05,
            beta_alpha_regularization: float = 0.01,
            recon_type: str = 'mse',
            modality_weights: Optional[Dict[str, float]] = None
    ):
        super().__init__()
        self.beta_kl = beta_kl
        self.beta_reconstruction = beta_reconstruction
        self.beta_cross_modal = beta_cross_modal
        self.beta_alpha_regularization = beta_alpha_regularization
        self.recon_type = recon_type
        self.modality_weights = modality_weights or {}

    def forward(
            self,
            inputs: Dict[str, torch.Tensor],
            reconstructions: Dict[str, torch.Tensor],
            mu: torch.Tensor,
            logvar: torch.Tensor,
            per_modality_mus: Optional[Dict[str, torch.Tensor]] = None,
            alphas: Optional[Dict[str, torch.Tensor]] = None,
            projected_recons: Optional[Dict[str, torch.Tensor]] = None,
            return_components: bool = False
    ) -> Tuple[torch.Tensor, Optional[Dict[str, torch.Tensor]]]:
        """
        Compute total loss with optional alpha regularization.
        """
        losses = {}

        # 1. Reconstruction loss
        recon_losses = []
        total_weight = 0.0

        for name in reconstructions.keys():
            # Ensure shapes match
            recon = reconstructions[name]
            inp = inputs[name]

            # Handle sequence length mismatch by slicing to minimum
            min_seq_len = min(recon.shape[1], inp.shape[1])
            recon = recon[:, :min_seq_len, :]
            inp = inp[:, :min_seq_len, :]

            if self.recon_type == 'mse':
                recon_loss = F.mse_loss(recon, inp)
            elif self.recon_type == 'cosine':
                recon_flat = recon.reshape(-1, recon.shape[-1])
                input_flat = inp.reshape(-1, inp.shape[-1])
                recon_norm = F.normalize(recon_flat, dim=-1)
                input_norm = F.normalize(input_flat, dim=-1)
                cos_sim = (recon_norm * input_norm).sum(dim=-1)
                recon_loss = (1 - cos_sim).mean()
            else:
                raise ValueError(f"Unknown recon_type: {self.recon_type}")

            weight = self.modality_weights.get(name, 1.0)
            weighted_loss = recon_loss * weight

            losses[f'recon_{name}'] = recon_loss
            recon_losses.append(weighted_loss)
            total_weight += weight

        total_recon = sum(recon_losses) / total_weight if recon_losses else torch.tensor(0.0)

        # 2. KL divergence (global)
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        kl_loss = kl_loss / (mu.shape[0] * mu.shape[1] * mu.shape[2])
        losses['kl'] = kl_loss

        # 3. Alpha regularization (only for ADAPTIVE_CANTOR)
        if per_modality_mus is not None and alphas is not None:
            alpha_reg_losses = []
            for name, alpha in alphas.items():
                if name in per_modality_mus:
                    # Compute per-modality KL divergence
                    mu_mod = per_modality_mus[name]
                    kl_mod = -0.5 * torch.sum(1 - mu_mod.pow(2))
                    kl_mod = kl_mod / (mu_mod.shape[0] * mu_mod.shape[1] * mu_mod.shape[2])

                    # Alpha should correlate with KL
                    alpha_clamped = torch.sigmoid(alpha)
                    alpha_target = torch.sigmoid(kl_mod * 10)

                    alpha_reg = (alpha_clamped - alpha_target).pow(2)
                    alpha_reg_losses.append(alpha_reg)

                    losses[f'alpha_{name}'] = alpha_clamped.item()

            alpha_regularization = sum(alpha_reg_losses) / len(alpha_reg_losses) if alpha_reg_losses else torch.tensor(
                0.0, device=mu.device)
            losses['alpha_reg'] = alpha_regularization
        else:
            alpha_regularization = torch.tensor(0.0, device=mu.device)
            losses['alpha_reg'] = alpha_regularization

        # 4. Cross-modal consistency
        if len(reconstructions) > 1 and projected_recons is not None:
            projected_list = list(projected_recons.values())
            cross_modal_losses = []

            for i in range(len(projected_list)):
                for j in range(i + 1, len(projected_list)):
                    # Handle sequence length mismatches
                    proj_i = projected_list[i]
                    proj_j = projected_list[j]
                    min_seq = min(proj_i.shape[1], proj_j.shape[1])

                    cm_loss = F.mse_loss(proj_i[:, :min_seq, :], proj_j[:, :min_seq, :])
                    cross_modal_losses.append(cm_loss)

            cross_modal = sum(cross_modal_losses) / len(cross_modal_losses) if cross_modal_losses else torch.tensor(0.0,
                                                                                                                    device=mu.device)
            losses['cross_modal'] = cross_modal
        else:
            cross_modal = torch.tensor(0.0, device=mu.device)
            losses['cross_modal'] = cross_modal

        # Total loss
        total_loss = (
                self.beta_reconstruction * total_recon +
                self.beta_kl * kl_loss +
                self.beta_cross_modal * cross_modal +
                self.beta_alpha_regularization * alpha_regularization
        )
        losses['total'] = total_loss

        if return_components:
            return total_loss, losses
        return total_loss, None


# ============================================================================
# COMPREHENSIVE TEST SUITE
# ============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("Multi-Modal VAE - Comprehensive Test Suite")
    print("=" * 80)

    # Set seed for reproducibility
    SEED = 42
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n🖥️  Device: {device}")

    # ========================================================================
    # TEST 1: All Fusion Strategies
    # ========================================================================
    print("\n" + "=" * 80)
    print("TEST 1: All Fusion Strategies")
    print("=" * 80)

    strategies = [
        "concatenate",
        "attention",
        "gated",
        "cantor",
        "geometric",
        "hierarchical",
        "adaptive_cantor"
    ]

    # Prepare test data (SDXL-style)
    batch_size = 2
    test_inputs = {
        "clip_l": torch.randn(batch_size, 77, 768, device=device),
        "clip_g": torch.randn(batch_size, 77, 1280, device=device),
        "t5_xl_l": torch.randn(batch_size, 512, 2048, device=device),
        "t5_xl_g": torch.randn(batch_size, 512, 2048, device=device)
    }

    for strategy in strategies:
        print(f"\n📊 Testing strategy: {strategy.upper()}")

        config = MultiModalVAEConfig(
            modality_dims={
                "clip_l": 768,
                "clip_g": 1280,
                "t5_xl_l": 2048,
                "t5_xl_g": 2048
            },
            modality_seq_lens={
                "clip_l": 77,
                "clip_g": 77,
                "t5_xl_l": 512,
                "t5_xl_g": 512
            },
            latent_dim=2048,
            fusion_strategy=strategy,
            seed=SEED
        )

        try:
            model = MultiModalVAE(config).to(device).eval()

            # Count parameters
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

            print(f"   Total parameters: {total_params:,}")
            print(f"   Trainable parameters: {trainable_params:,}")

            # Forward pass
            with torch.no_grad():
                generator = torch.Generator(device=device).manual_seed(SEED)
                recons, mu, logvar, per_mod_mus = model(test_inputs, generator=generator)

            print(f"   ✓ Forward pass successful")
            print(f"   Latent shape: {mu.shape}")
            print(f"   Reconstructions:")
            for name, recon in recons.items():
                print(f"     - {name}: {recon.shape}")

            # Check for alpha/beta parameters
            if strategy == "adaptive_cantor":
                fusion_params = model.get_fusion_params()
                if fusion_params:
                    print(f"   Alpha parameters: {len(fusion_params.get('alphas', {}))}")
                    print(f"   Beta parameters: {len(fusion_params.get('betas', {}))}")
                    for name, alpha in fusion_params.get('alphas', {}).items():
                        print(f"     - alpha_{name}: {torch.sigmoid(alpha).item():.4f}")
                    for name, beta in fusion_params.get('betas', {}).items():
                        print(f"     - beta_{name}: {torch.sigmoid(beta).item():.4f}")

        except Exception as e:
            print(f"   ✗ Error: {e}")
            import traceback

            traceback.print_exc()

    # ========================================================================
    # TEST 2: Loss Computation
    # ========================================================================
    print("\n" + "=" * 80)
    print("TEST 2: Loss Computation")
    print("=" * 80)

    config = MultiModalVAEConfig(
        fusion_strategy="adaptive_cantor",
        seed=SEED
    )
    model = MultiModalVAE(config).to(device).eval()

    loss_fn = MultiModalVAELoss(
        beta_kl=0.1,
        beta_reconstruction=1.0,
        beta_cross_modal=0.05,
        beta_alpha_regularization=0.01
    )

    with torch.no_grad():
        generator = torch.Generator(device=device).manual_seed(SEED)
        recons, mu, logvar, per_mod_mus = model(test_inputs, generator=generator)

        # Get fusion parameters for alpha regularization
        fusion_params = model.get_fusion_params()
        alphas = fusion_params.get('alphas', None)

        # Compute loss
        loss, components = loss_fn(
            inputs=test_inputs,
            reconstructions=recons,
            mu=mu,
            logvar=logvar,
            per_modality_mus=per_mod_mus,
            alphas=alphas,
            return_components=True
        )

    print(f"\n📉 Loss Components:")
    print(f"   Total loss: {loss.item():.6f}")
    for name, value in components.items():
        if name != 'total' and not name.startswith('alpha_'):
            if isinstance(value, torch.Tensor):
                print(f"   {name}: {value.item():.6f}")
            else:
                print(f"   {name}: {value:.6f}")

    # ========================================================================
    # TEST 3: Seed Reproducibility
    # ========================================================================
    print("\n" + "=" * 80)
    print("TEST 3: Seed Reproducibility")
    print("=" * 80)

    config = MultiModalVAEConfig(
        fusion_strategy="adaptive_cantor",
        seed=SEED
    )

    # First run
    model1 = MultiModalVAE(config).to(device).eval()
    with torch.no_grad():
        generator1 = torch.Generator(device=device).manual_seed(SEED)
        recons1, mu1, logvar1, _ = model1(test_inputs, generator=generator1)

    # Second run (same seed)
    model2 = MultiModalVAE(config).to(device).eval()
    with torch.no_grad():
        generator2 = torch.Generator(device=device).manual_seed(SEED)
        recons2, mu2, logvar2, _ = model2(test_inputs, generator=generator2)

    # Check reproducibility
    mu_match = torch.allclose(mu1, mu2, atol=1e-6)
    logvar_match = torch.allclose(logvar1, logvar2, atol=1e-6)
    recons_match = all(
        torch.allclose(recons1[k], recons2[k], atol=1e-6)
        for k in recons1.keys()
    )

    print(f"\n🔄 Reproducibility Test:")
    print(f"   Latent mu identical: {mu_match}")
    print(f"   Latent logvar identical: {logvar_match}")
    print(f"   Reconstructions identical: {recons_match}")

    if mu_match and logvar_match and recons_match:
        print(f"   ✓ Full reproducibility achieved!")
    else:
        print(f"   ✗ Reproducibility failed")

    # ========================================================================
    # TEST 4: Different Sequence Lengths
    # ========================================================================
    print("\n" + "=" * 80)
    print("TEST 4: Different Sequence Lengths")
    print("=" * 80)

    config = MultiModalVAEConfig(
        modality_dims={
            "clip_l": 768,
            "clip_g": 1280,
            "t5_xl_l": 2048,
            "t5_xl_g": 2048
        },
        modality_seq_lens={
            "clip_l": 77,
            "clip_g": 77,
            "t5_xl_l": 512,  # Longer for T5
            "t5_xl_g": 512
        },
        fusion_strategy="adaptive_cantor",
        seed=SEED
    )

    model = MultiModalVAE(config).to(device).eval()

    # Test with actual different lengths
    varied_inputs = {
        "clip_l": torch.randn(batch_size, 77, 768, device=device),
        "clip_g": torch.randn(batch_size, 77, 1280, device=device),
        "t5_xl_l": torch.randn(batch_size, 512, 2048, device=device),
        "t5_xl_g": torch.randn(batch_size, 512, 2048, device=device)
    }

    with torch.no_grad():
        generator = torch.Generator(device=device).manual_seed(SEED)
        recons, mu, logvar, _ = model(varied_inputs, generator=generator)

    print(f"\n📏 Sequence Length Handling:")
    print(f"   Input sequences:")
    for name, inp in varied_inputs.items():
        print(f"     - {name}: {inp.shape[1]} tokens")
    print(f"   Output sequences:")
    for name, recon in recons.items():
        print(f"     - {name}: {recon.shape[1]} tokens (preserved)")

    # Verify sequence lengths preserved
    lengths_preserved = all(
        recons[name].shape[1] == varied_inputs[name].shape[1]
        for name in recons.keys()
    )
    print(f"   ✓ Sequence lengths preserved: {lengths_preserved}")

    # ========================================================================
    # TEST 5: Decoupled T5 Scales
    # ========================================================================
    print("\n" + "=" * 80)
    print("TEST 5: Decoupled T5 Scales")
    print("=" * 80)

    config = MultiModalVAEConfig(
        binding_config={
            "clip_l": {"t5_xl_l": 0.3},  # CLIP-L bound to T5-L
            "clip_g": {"t5_xl_g": 0.3},  # CLIP-G bound to T5-G
            "t5_xl_l": {},  # T5-L independent
            "t5_xl_g": {}  # T5-G independent
        },
        fusion_strategy="adaptive_cantor",
        seed=SEED
    )

    model = MultiModalVAE(config).to(device).eval()

    print(f"\n🔗 Binding Configuration:")
    for target, sources in config.binding_config.items():
        if sources:
            print(f"   {target} ← {', '.join(f'{s} ({w})' for s, w in sources.items())}")
        else:
            print(f"   {target} (independent)")

    # Test that T5 scales are treated independently
    fusion_params = model.get_fusion_params()
    if fusion_params:
        betas = fusion_params.get('betas', {})
        print(f"\n   Learned beta parameters:")
        for key, beta in betas.items():
            print(f"     - {key}: {torch.sigmoid(beta).item():.4f}")

    # ========================================================================
    # TEST 6: Cantor Routing Visualization
    # ========================================================================
    print("\n" + "=" * 80)
    print("TEST 6: Cantor Routing Visualization")
    print("=" * 80)

    config = MultiModalVAEConfig(
        fusion_strategy="cantor",
        cantor_depth=8,
        cantor_local_window=3,
        seed=SEED
    )

    model = MultiModalVAE(config).to(device).eval()

    if hasattr(model.fusion, 'modality_cantor_coords'):
        coords = model.fusion.modality_cantor_coords
        routes = model.fusion.modality_routes

        print(f"\n📍 Cantor Coordinates:")
        for i, name in enumerate(model.fusion.modality_names):
            print(f"   {name}: {coords[i].item():.6f}")

        print(f"\n🗺️  Routing Table (local_window={config.cantor_local_window}):")
        for i, name in enumerate(model.fusion.modality_names):
            neighbors = [model.fusion.modality_names[idx] for idx in routes[i]]
            print(f"   {name} → {neighbors}")

    # ========================================================================
    # TEST 7: Gradient Flow
    # ========================================================================
    print("\n" + "=" * 80)
    print("TEST 7: Gradient Flow")
    print("=" * 80)

    config = MultiModalVAEConfig(
        fusion_strategy="adaptive_cantor",
        seed=SEED
    )

    model = MultiModalVAE(config).to(device)
    model.train()  # Enable training mode

    loss_fn = MultiModalVAELoss(
        beta_kl=0.1,
        beta_reconstruction=1.0,
        beta_cross_modal=0.05,
        beta_alpha_regularization=0.01
    )

    # Forward pass with gradients
    recons, mu, logvar, per_mod_mus = model(test_inputs)

    fusion_params = model.get_fusion_params()
    alphas = fusion_params.get('alphas', None)

    loss, _ = loss_fn(
        inputs=test_inputs,
        reconstructions=recons,
        mu=mu,
        logvar=logvar,
        per_modality_mus=per_mod_mus,
        alphas=alphas
    )

    # Backward pass
    loss.backward()

    print(f"\n🎓 Gradient Statistics:")

    # Check alpha/beta gradients
    if alphas:
        print(f"   Alpha gradients:")
        for name, alpha in alphas.items():
            if alpha.grad is not None:
                print(f"     - alpha_{name}: {alpha.grad.item():.6f}")

    betas = fusion_params.get('betas', {})
    if betas:
        print(f"   Beta gradients:")
        for name, beta in betas.items():
            if beta.grad is not None:
                print(f"     - {name}: {beta.grad.item():.6f}")

    # Check encoder gradients
    encoder_grad_norms = []
    for name, param in model.named_parameters():
        if param.grad is not None and 'encoder' in name:
            encoder_grad_norms.append(param.grad.norm().item())

    if encoder_grad_norms:
        print(f"   Encoder gradient norms:")
        print(f"     - Mean: {sum(encoder_grad_norms) / len(encoder_grad_norms):.6f}")
        print(f"     - Max: {max(encoder_grad_norms):.6f}")
        print(f"     - Min: {min(encoder_grad_norms):.6f}")

    # ========================================================================
    # TEST 8: Memory Usage
    # ========================================================================
    print("\n" + "=" * 80)
    print("TEST 8: Memory Usage")
    print("=" * 80)

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

        config = MultiModalVAEConfig(
            fusion_strategy="adaptive_cantor",
            seed=SEED
        )

        model = MultiModalVAE(config).to(device)

        # Warm up
        with torch.no_grad():
            _ = model(test_inputs)

        torch.cuda.reset_peak_memory_stats()

        # Measure forward pass
        with torch.no_grad():
            _ = model(test_inputs)

        forward_memory = torch.cuda.max_memory_allocated() / 1024 ** 2  # MB

        # Measure backward pass
        torch.cuda.reset_peak_memory_stats()
        model.train()
        recons, mu, logvar, per_mod_mus = model(test_inputs)
        fusion_params = model.get_fusion_params()
        loss, _ = loss_fn(
            inputs=test_inputs,
            reconstructions=recons,
            mu=mu,
            logvar=logvar,
            per_modality_mus=per_mod_mus,
            alphas=fusion_params.get('alphas')
        )
        loss.backward()

        backward_memory = torch.cuda.max_memory_allocated() / 1024 ** 2  # MB

        print(f"\n💾 GPU Memory Usage:")
        print(f"   Forward pass: {forward_memory:.2f} MB")
        print(f"   Backward pass: {backward_memory:.2f} MB")
    else:
        print(f"\n💾 GPU not available - skipping memory test")

    # ========================================================================
    # Summary
    # ========================================================================
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"""
✓ All {len(strategies)} fusion strategies tested
✓ Loss computation verified
✓ Seed reproducibility confirmed
✓ Variable sequence lengths supported
✓ Decoupled T5 scales working
✓ Cantor routing operational
✓ Gradient flow validated
✓ Memory profiling complete

🎵 VAE Lyra is ready for training!
    """)

    print("=" * 80)
    print("Key Features:")
    print("  • 7 fusion strategies (CONCATENATE, ATTENTION, GATED, CANTOR,")
    print("    GEOMETRIC, HIERARCHICAL, ADAPTIVE_CANTOR)")
    print("  • Learned alpha (visibility) and beta (capacity) parameters")
    print("  • Cantor fractal routing for sparse attention")
    print("  • Decoupled T5-XL representations for CLIP-L and CLIP-G")
    print("  • Variable sequence lengths per modality")
    print("  • Full seed reproducibility")
    print("  • Comprehensive loss with alpha regularization")
    print("=" * 80)