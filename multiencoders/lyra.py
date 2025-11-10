"""
Multi-Modal VAE with Advanced Fusion and Seed Control
======================================================

Single-input and multi-input VAE architectures with geometric fusion mechanisms
and deterministic seed control for reproducibility.

SEED CONTROL:
- Set seed in config for reproducible initialization
- Pass generator to forward() for deterministic latent sampling
- Use model.eval() for fully deterministic inference (disables dropout)

USAGE:
    # Reproducible training
    config = MultiModalVAEConfig(seed=42)
    vae = MultiModalVAE(config)

    # Deterministic evaluation
    vae.eval()
    generator = torch.Generator(device='cuda').manual_seed(42)
    for batch in data:
        recon, mu, logvar = vae(batch, generator=generator)

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


@dataclass
class MultiModalVAEConfig:
    """Configuration for multi-modal VAE."""
    # Input modalities
    modality_dims: Dict[str, int] = None  # e.g., {"clip": 768, "t5": 768}

    # Latent space
    latent_dim: int = 768
    seq_len: int = 77

    # Architecture
    encoder_layers: int = 3
    decoder_layers: int = 3
    hidden_dim: int = 1024
    dropout: float = 0.1

    # Fusion
    fusion_strategy: str = "cantor"
    fusion_heads: int = 8
    fusion_dropout: float = 0.1

    # Loss weights
    beta_kl: float = 0.1
    beta_reconstruction: float = 1.0
    beta_cross_modal: float = 0.05

    # Training
    use_amp: bool = True

    # Reproducibility
    seed: Optional[int] = None

    def __post_init__(self):
        if self.modality_dims is None:
            self.modality_dims = {"clip": 768, "t5": 768}


# ============================================================================
# SINGLE-INPUT VAE (Baseline)
# ============================================================================

class SingleModalityVAE(nn.Module):
    """
    Simple single-modality VAE with seed control.

    For baseline: just CLIP or just T5.
    """

    def __init__(
            self,
            input_dim: int = 768,
            latent_dim: int = 768,
            seq_len: int = 77,
            hidden_dim: int = 1024,
            num_layers: int = 3,
            dropout: float = 0.1,
            seed: Optional[int] = None
    ):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.seq_len = seq_len
        self.seed = seed

        # Set seed for reproducible initialization
        if seed is not None:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

        # Encoder
        encoder_layers = []
        in_dim = input_dim
        for i in range(num_layers):
            out_dim = hidden_dim if i < num_layers - 1 else latent_dim * 2
            encoder_layers.extend([
                nn.Linear(in_dim, out_dim),
                nn.LayerNorm(out_dim),
                nn.GELU(),
                nn.Dropout(dropout)
            ])
            in_dim = out_dim

        self.encoder = nn.Sequential(*encoder_layers[:-2])  # Remove last GELU/Dropout

        # Latent projection (outputs mu and logvar)
        self.fc_mu = nn.Linear(latent_dim * 2, latent_dim)
        self.fc_logvar = nn.Linear(latent_dim * 2, latent_dim)

        # Decoder
        decoder_layers = []
        in_dim = latent_dim
        for i in range(num_layers):
            out_dim = hidden_dim if i < num_layers - 1 else input_dim
            decoder_layers.extend([
                nn.Linear(in_dim, out_dim),
                nn.LayerNorm(out_dim) if i < num_layers - 1 else nn.Identity(),
                nn.GELU() if i < num_layers - 1 else nn.Identity(),
                nn.Dropout(dropout) if i < num_layers - 1 else nn.Identity()
            ])
            in_dim = out_dim

        self.decoder = nn.Sequential(*decoder_layers)

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode input to latent parameters.

        Args:
            x: Input [batch, seq, input_dim]

        Returns:
            mu, logvar: [batch, seq, latent_dim]
        """
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(
            self,
            mu: torch.Tensor,
            logvar: torch.Tensor,
            generator: Optional[torch.Generator] = None
    ) -> torch.Tensor:
        """
        Reparameterization trick with optional generator for seed control.

        Args:
            mu: Mean [batch, seq, latent_dim]
            logvar: Log variance [batch, seq, latent_dim]
            generator: Optional torch.Generator for reproducible sampling

        Returns:
            z: Sampled latent [batch, seq, latent_dim]
        """
        std = torch.exp(0.5 * logvar)

        # Create generator if seed is set but no generator provided
        if generator is None and self.seed is not None:
            generator = torch.Generator(device=mu.device).manual_seed(self.seed)

        # Sample epsilon
        if generator is not None:
            eps = torch.randn(mu.shape, generator=generator, device=mu.device, dtype=mu.dtype)
        else:
            eps = torch.randn_like(std)

        return mu + eps * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decode latent to reconstruction.

        Args:
            z: Latent [batch, seq, latent_dim]

        Returns:
            Reconstruction [batch, seq, input_dim]
        """
        return self.decoder(z)

    def forward(
            self,
            x: torch.Tensor,
            generator: Optional[torch.Generator] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Full forward pass.

        Args:
            x: Input [batch, seq, input_dim]
            generator: Optional generator for reproducible sampling

        Returns:
            recon, mu, logvar
        """
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar, generator)
        recon = self.decode(z)
        return recon, mu, logvar


# ============================================================================
# FUSION MODULES (Re-dubbed Lyra from David)
# ============================================================================

class CantorModalityFusion(nn.Module):
    """
    Cantor-based fusion for multiple modalities.
    Uses fractal routing to fuse embeddings from different sources.
    """

    def __init__(
            self,
            modality_dims: Dict[str, int],
            output_dim: int,
            num_heads: int = 8,
            cantor_depth: int = 8,
            local_window: int = 3,
            dropout: float = 0.1,
            seed: Optional[int] = None
    ):
        super().__init__()

        # Set seed for reproducible initialization
        if seed is not None:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

        self.modality_names = list(modality_dims.keys())
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

        # Temperature
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

    def forward(
            self,
            modality_inputs: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """
        Fuse multiple modality inputs.

        Args:
            modality_inputs: Dict of {name: tensor [batch, seq, dim]}

        Returns:
            Fused output [batch, seq, output_dim]
        """
        B, seq_len, _ = list(modality_inputs.values())[0].shape
        device = list(modality_inputs.values())[0].device

        # Project all modalities to common space
        projected = []
        for i, name in enumerate(self.modality_names):
            if name in modality_inputs:
                proj = self.modality_projections[name](modality_inputs[name])
                proj = proj + self.modality_embeddings[i]
                projected.append(proj)

        # Stack: [batch, num_modalities, seq, dim]
        stacked = torch.stack(projected, dim=1)

        # Multi-head attention with Cantor routing
        # Reshape for multi-head
        Q = self.q_proj(stacked).view(B, self.num_modalities, seq_len, self.num_heads, self.head_dim)
        K = self.k_proj(stacked).view(B, self.num_modalities, seq_len, self.num_heads, self.head_dim)
        V = self.v_proj(stacked).view(B, self.num_modalities, seq_len, self.num_heads, self.head_dim)

        # Permute to [batch, heads, num_modalities, seq, head_dim]
        Q = Q.permute(0, 3, 1, 2, 4)
        K = K.permute(0, 3, 1, 2, 4)
        V = V.permute(0, 3, 1, 2, 4)

        # Sparse attention via Cantor routing
        routes = self.modality_routes.to(device)

        # For each modality, gather its neighbors
        attended = []
        for i in range(self.num_modalities):
            neighbors = routes[i]  # [local_window]

            q_i = Q[:, :, i, :, :]  # [batch, heads, seq, head_dim]
            k_neighbors = K[:, :, neighbors, :, :]  # [batch, heads, window, seq, head_dim]
            v_neighbors = V[:, :, neighbors, :, :]  # [batch, heads, window, seq, head_dim]

            # Attention scores
            scores = torch.einsum('bhsd,bhwsd->bhsw', q_i, k_neighbors) / math.sqrt(self.head_dim)
            scores = scores / self.temperature.abs()

            attn = F.softmax(scores, dim=-1)
            attn = self.dropout(attn)

            # Apply attention
            out_i = torch.einsum('bhsw,bhwsd->bhsd', attn, v_neighbors)
            attended.append(out_i)

        # Stack and mean over modalities: [batch, heads, seq, head_dim]
        fused = torch.stack(attended, dim=2).mean(dim=2)

        # Reshape back: [batch, seq, output_dim]
        fused = fused.permute(0, 2, 1, 3).reshape(B, seq_len, self.output_dim)

        # Output projection
        output = self.out_proj(fused)
        output = self.dropout(output)

        return output


class GeometricModalityFusion(nn.Module):
    """
    Geometric fusion using pentachoron-inspired attention.
    """

    def __init__(
            self,
            modality_dims: Dict[str, int],
            output_dim: int,
            num_heads: int = 4,
            use_cayley: bool = True,
            use_angular: bool = True,
            temperature: float = 0.07,
            dropout: float = 0.1,
            seed: Optional[int] = None
    ):
        super().__init__()

        # Set seed for reproducible initialization
        if seed is not None:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

        self.modality_names = list(modality_dims.keys())
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

        # Pentachoron role weights for angular attention
        if use_angular:
            role_weights = torch.tensor([1.0, -0.75, 0.75, 0.75, -0.75])
            # Pad or truncate to num_modalities
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

        # Learnable combination of attention types
        self.attention_weights = nn.Parameter(torch.ones(3) / 3)

    def _compute_angular_attention(
            self,
            features: List[torch.Tensor]
    ) -> torch.Tensor:
        """Compute attention based on angular relationships."""
        B, seq_len, _ = features[0].shape

        # Normalize
        features_norm = [F.normalize(f, dim=-1) for f in features]

        # Compute pairwise angles
        attention = torch.zeros(B, self.num_modalities, device=features[0].device)

        for i, feat_i in enumerate(features_norm):
            for j, feat_j in enumerate(features_norm):
                if i != j:
                    cos_sim = (feat_i * feat_j).sum(dim=-1).mean(dim=-1)
                    angle = torch.acos(cos_sim.clamp(-1 + 1e-7, 1 - 1e-7))
                    attention[:, i] += self.role_weights[j] * torch.exp(-angle / self.temperature.abs())

        return F.softmax(attention, dim=-1)

    def _compute_cayley_attention(
            self,
            features: List[torch.Tensor]
    ) -> torch.Tensor:
        """Compute attention based on Cayley-Menger volumes."""
        B = features[0].shape[0]

        # Compute volume for each modality as center
        volume_scores = []

        for i in range(self.num_modalities):
            # Create simplex with modality i as anchor
            simplex_points = [features[i]]

            # Add other modalities with rotations
            for j in range(min(4, self.num_modalities - 1)):
                angle = (j + 1) * math.pi / 4
                other_idx = (i + j + 1) % self.num_modalities
                rot_feat = features[i] * math.cos(angle) + features[other_idx] * math.sin(angle)
                simplex_points.append(rot_feat)

            # Stack and compute volume proxy
            simplex = torch.stack(simplex_points, dim=1)  # [B, num_points, seq, dim]
            diff = simplex.unsqueeze(2) - simplex.unsqueeze(1)
            distsq = (diff * diff).sum(dim=-1).sum(dim=-1)  # Sum over seq and dim

            volume = distsq.mean(dim=(1, 2))
            volume_scores.append(volume)

        volumes = torch.stack(volume_scores, dim=1)
        return F.softmax(volumes / self.temperature.abs(), dim=-1)

    def _compute_standard_attention(
            self,
            Q: torch.Tensor,
            K: torch.Tensor,
            V: torch.Tensor
    ) -> torch.Tensor:
        """Standard multi-head attention."""
        # Q: [B, H, 1, seq, D]
        # K, V: [B, H, num_mod, seq, D]

        scores = torch.einsum('bhqsd,bhmsd->bhqms', Q, K) / math.sqrt(self.head_dim)
        scores = scores / self.temperature.abs()

        attn = F.softmax(scores, dim=-2)  # Over modalities
        attn = self.dropout(attn)

        out = torch.einsum('bhqms,bhmsd->bhqsd', attn, V)

        # Reshape: [B, H, 1, seq, D] -> [B, seq, H*D] = [B, seq, output_dim]
        B = out.shape[0]
        out = out.squeeze(2)  # [B, H, seq, D]
        out = out.permute(0, 2, 1, 3)  # [B, seq, H, D]
        out = out.reshape(B, -1, self.output_dim)  # [B, seq, H*D]
        return out

    def forward(
            self,
            modality_inputs: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """
        Args:
            modality_inputs: Dict of {name: [batch, seq, dim]}

        Returns:
            Fused output [batch, seq, output_dim]
        """
        B, seq_len, _ = list(modality_inputs.values())[0].shape

        # Project to common space
        features = []
        for name in self.modality_names:
            if name in modality_inputs:
                proj = self.modality_projections[name](modality_inputs[name])
                features.append(proj)

        # Multi-head projection
        Q_feat = features[0] if features else torch.zeros(B, seq_len, self.output_dim,
                                                          device=list(modality_inputs.values())[0].device)
        Q = self.q_proj(Q_feat).view(B, self.num_heads, 1, seq_len, self.head_dim)

        K_list, V_list = [], []
        for feat in features:
            K = self.k_proj(feat).view(B, self.num_heads, 1, seq_len, self.head_dim)
            V = self.v_proj(feat).view(B, self.num_heads, 1, seq_len, self.head_dim)
            K_list.append(K)
            V_list.append(V)

        K = torch.cat(K_list, dim=2)  # [B, H, num_mod, seq, D]
        V = torch.cat(V_list, dim=2)

        # Compute different attention types
        attention_outputs = []

        # 1. Standard MHA
        mha_out = self._compute_standard_attention(Q, K, V)
        attention_outputs.append(mha_out)

        # 2. Angular attention
        if self.use_angular:
            angular_weights = self._compute_angular_attention(features)  # [B, num_modalities]
            # Weight each modality's features
            angular_out = torch.zeros_like(features[0])
            for i in range(self.num_modalities):
                w = angular_weights[:, i]  # [B]
                angular_out = angular_out + w.unsqueeze(-1).unsqueeze(-1) * features[i]
            attention_outputs.append(angular_out)

        # 3. Cayley attention
        if self.use_cayley:
            cayley_weights = self._compute_cayley_attention(features)  # [B, num_modalities]
            cayley_out = torch.zeros_like(features[0])
            for i in range(self.num_modalities):
                w = cayley_weights[:, i]  # [B]
                cayley_out = cayley_out + w.unsqueeze(-1).unsqueeze(-1) * features[i]
            attention_outputs.append(cayley_out)

        # Combine with learnable weights
        attn_weights = F.softmax(self.attention_weights[:len(attention_outputs)], dim=0)
        fused = torch.zeros_like(attention_outputs[0])
        for i, out in enumerate(attention_outputs):
            fused = fused + attn_weights[i] * out

        # Output projection
        output = self.out_proj(fused)
        output = self.dropout(output)

        return output


# ============================================================================
# MULTI-MODAL VAE
# ============================================================================

# In vae_lyra.py - Update MultiModalVAE

class MultiModalVAE(nn.Module):
    """
    Multi-modal VAE with advanced fusion and seed control.

    Handles multiple text encoders (CLIP, T5, etc.) and fuses them
    into a unified latent space.
    """

    def __init__(self, config: MultiModalVAEConfig):
        super().__init__()
        self.config = config
        self.modality_names = list(config.modality_dims.keys())
        self.num_modalities = len(self.modality_names)
        self.seed = config.seed

        # Set seed for reproducible initialization
        if self.seed is not None:
            torch.manual_seed(self.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(self.seed)

        # Fusion module
        fusion_strategy = FusionStrategy(config.fusion_strategy)

        if fusion_strategy == FusionStrategy.CANTOR:
            self.fusion = CantorModalityFusion(
                modality_dims=config.modality_dims,
                output_dim=config.hidden_dim,
                num_heads=config.fusion_heads,
                dropout=config.fusion_dropout,
                seed=config.seed
            )
        elif fusion_strategy == FusionStrategy.GEOMETRIC:
            self.fusion = GeometricModalityFusion(
                modality_dims=config.modality_dims,
                output_dim=config.hidden_dim,
                num_heads=config.fusion_heads,
                dropout=config.fusion_dropout,
                seed=config.seed
            )
        elif fusion_strategy == FusionStrategy.CONCATENATE:
            total_dim = sum(config.modality_dims.values())
            self.fusion = nn.Sequential(
                nn.Linear(total_dim, config.hidden_dim),
                nn.LayerNorm(config.hidden_dim),
                nn.GELU(),
                nn.Dropout(config.dropout)
            )
        else:
            raise ValueError(f"Unknown fusion strategy: {config.fusion_strategy}")

        self.fusion_strategy = fusion_strategy

        # Encoder (from fused representation to latent)
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

        # Latent parameters
        self.fc_mu = nn.Linear(config.latent_dim * 2, config.latent_dim)
        self.fc_logvar = nn.Linear(config.latent_dim * 2, config.latent_dim)

        # Decoder (from latent to each modality)
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

        # Cross-modal projection layers (for comparing different dimensions)
        # These project all modalities to a common space for alignment
        self.cross_modal_projections = nn.ModuleDict({
            name: nn.Linear(dim, config.latent_dim)  # Project to latent_dim
            for name, dim in config.modality_dims.items()
        })

    def fuse_modalities(
            self,
            modality_inputs: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """Fuse multiple modality inputs."""
        if self.fusion_strategy == FusionStrategy.CONCATENATE:
            # Simple concatenation
            cat_inputs = torch.cat([
                modality_inputs[name] for name in self.modality_names
            ], dim=-1)
            return self.fusion(cat_inputs)
        else:
            # Advanced fusion (Cantor, Geometric, etc.)
            return self.fusion(modality_inputs)

    def encode(
            self,
            modality_inputs: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode modalities to latent parameters.

        Args:
            modality_inputs: Dict of {name: [batch, seq, dim]}

        Returns:
            mu, logvar: [batch, seq, latent_dim]
        """
        fused = self.fuse_modalities(modality_inputs)
        h = self.encoder(fused)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(
            self,
            mu: torch.Tensor,
            logvar: torch.Tensor,
            generator: Optional[torch.Generator] = None
    ) -> torch.Tensor:
        """
        Reparameterization trick with optional generator for seed control.

        Args:
            mu: Mean [batch, seq, latent_dim]
            logvar: Log variance [batch, seq, latent_dim]
            generator: Optional torch.Generator for reproducible sampling

        Returns:
            z: Sampled latent [batch, seq, latent_dim]
        """
        std = torch.exp(0.5 * logvar)

        # Create generator if seed is set but no generator provided
        if generator is None and self.seed is not None:
            generator = torch.Generator(device=mu.device).manual_seed(self.seed)

        # Sample epsilon
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
        """
        Decode latent to reconstructions.

        Args:
            z: Latent [batch, seq, latent_dim]
            target_modalities: Which modalities to decode (default: all)

        Returns:
            Dict of {name: reconstruction [batch, seq, dim]}
        """
        if target_modalities is None:
            target_modalities = self.modality_names

        reconstructions = {}
        for name in target_modalities:
            reconstructions[name] = self.decoders[name](z)

        return reconstructions

    def project_for_cross_modal(
            self,
            reconstructions: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Project reconstructions to common space for cross-modal comparison.

        Args:
            reconstructions: Dict of {name: [batch, seq, dim_i]}

        Returns:
            Dict of {name: [batch, seq, latent_dim]} - all same dimension
        """
        projected = {}
        for name, recon in reconstructions.items():
            proj = self.cross_modal_projections[name](recon)
            proj = F.normalize(proj, dim=-1)  # Normalize for stability
            projected[name] = proj

        return projected

    def forward(
            self,
            modality_inputs: Dict[str, torch.Tensor],
            target_modalities: Optional[List[str]] = None,
            generator: Optional[torch.Generator] = None
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
        """
        Full forward pass.

        Args:
            modality_inputs: Dict of input tensors
            target_modalities: Optional list of modalities to decode
            generator: Optional generator for reproducible sampling

        Returns:
            reconstructions, mu, logvar
        """
        mu, logvar = self.encode(modality_inputs)
        z = self.reparameterize(mu, logvar, generator)
        reconstructions = self.decode(z, target_modalities)
        return reconstructions, mu, logvar


# ============================================================================
# LOSS FUNCTION
# ============================================================================

class MultiModalVAELoss(nn.Module):
    """Loss for multi-modal VAE - now stateless."""

    def __init__(
            self,
            beta_kl: float = 0.1,
            beta_reconstruction: float = 1.0,
            beta_cross_modal: float = 0.05,
            recon_type: str = 'mse',
            modality_weights: Optional[Dict[str, float]] = None
    ):
        super().__init__()
        self.beta_kl = beta_kl
        self.beta_reconstruction = beta_reconstruction
        self.beta_cross_modal = beta_cross_modal
        self.recon_type = recon_type
        self.modality_weights = modality_weights or {}

    def forward(
            self,
            inputs: Dict[str, torch.Tensor],
            reconstructions: Dict[str, torch.Tensor],
            mu: torch.Tensor,
            logvar: torch.Tensor,
            projected_recons: Optional[Dict[str, torch.Tensor]] = None,  # <-- NEW
            return_components: bool = False
    ) -> Tuple[torch.Tensor, Optional[Dict[str, torch.Tensor]]]:
        """
        Compute total loss.

        Args:
            inputs: Original inputs {name: [batch, seq, dim]}
            reconstructions: Reconstructed outputs {name: [batch, seq, dim]}
            mu, logvar: Latent parameters
            projected_recons: Optional pre-projected reconstructions for cross-modal loss
            return_components: Return loss breakdown

        Returns:
            total_loss, optional components dict
        """
        losses = {}

        # 1. Reconstruction loss for each modality (with per-modality weights)
        recon_losses = []
        total_weight = 0.0

        for name in reconstructions.keys():
            if self.recon_type == 'mse':
                recon_loss = F.mse_loss(reconstructions[name], inputs[name])
            elif self.recon_type == 'cosine':
                recon_flat = reconstructions[name].reshape(-1, reconstructions[name].shape[-1])
                input_flat = inputs[name].reshape(-1, inputs[name].shape[-1])
                recon_norm = F.normalize(recon_flat, dim=-1)
                input_norm = F.normalize(input_flat, dim=-1)
                cos_sim = (recon_norm * input_norm).sum(dim=-1)
                recon_loss = (1 - cos_sim).mean()

            weight = self.modality_weights.get(name, 1.0)
            weighted_loss = recon_loss * weight

            losses[f'recon_{name}'] = recon_loss
            recon_losses.append(weighted_loss)
            total_weight += weight

        total_recon = sum(recon_losses) / total_weight

        # 2. KL divergence
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        kl_loss = kl_loss / (mu.shape[0] * mu.shape[1] * mu.shape[2])
        losses['kl'] = kl_loss

        # 3. Cross-modal consistency in common space
        if len(reconstructions) > 1 and projected_recons is not None:
            # Compare pre-projected reconstructions (all same dimension now)
            projected_list = list(projected_recons.values())
            cross_modal_losses = []

            for i in range(len(projected_list)):
                for j in range(i + 1, len(projected_list)):
                    cm_loss = F.mse_loss(projected_list[i], projected_list[j])
                    cross_modal_losses.append(cm_loss)

            cross_modal = sum(cross_modal_losses) / len(cross_modal_losses)
            losses['cross_modal'] = cross_modal
        else:
            cross_modal = torch.tensor(0.0, device=mu.device)
            losses['cross_modal'] = cross_modal

        # Total
        total_loss = (
                self.beta_reconstruction * total_recon +
                self.beta_kl * kl_loss +
                self.beta_cross_modal * cross_modal
        )
        losses['total'] = total_loss

        if return_components:
            return total_loss, losses
        return total_loss, None
