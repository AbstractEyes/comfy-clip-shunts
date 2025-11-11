# geovocab2/train/model/vae/loader.py

"""
VAE Lyra Model Loader - Intelligent Version Detection and Loading
=================================================================

Automatically detects and loads the correct VAE Lyra variant from HuggingFace Hub
based on configuration parameters.

Supported Variants:
- v1: Standard multi-modal VAE with fusion strategies (concatenate, attention,
      gated, cantor, geometric, hierarchical)
      Examples: vae-lyra, vae-lyra-sdxl-t5xl

- v2: Adaptive Cantor VAE with learned alpha/beta parameters, variable sequence
      lengths, decoupled T5 scales, and binding configuration
      Examples: vae-lyra-xl-adaptive-cantor

Known Models:
- AbstractPhil/vae-lyra: Original CLIP-L + T5-base (v1)
- AbstractPhil/vae-lyra-sdxl-t5xl: SDXL with CLIP-L + CLIP-G + T5-XL (v1)
- AbstractPhil/vae-lyra-xl-adaptive-cantor: Adaptive Cantor with decoupled T5 (v2)

Usage:
    from geovocab2.train.model.vae.loader import load_vae_lyra

    # Auto-detect version
    model = load_vae_lyra("AbstractPhil/vae-lyra-xl-adaptive-cantor")

    # List all known models
    from geovocab2.train.model.vae.loader import list_known_models
    list_known_models()

    # Get model info
    info = get_model_info("AbstractPhil/vae-lyra-sdxl-t5xl")
    print(f"Version: {info['version']}")

Author: AbstractPhil
"""

import torch
import json
from pathlib import Path
from typing import Optional, Dict, Any, Tuple, List
from huggingface_hub import hf_hub_download
from dataclasses import asdict

# ============================================================================
# MODEL REGISTRY
# ============================================================================

KNOWN_MODELS = {
    "AbstractPhil/vae-lyra": {
        "version": "v1",
        "description": "Original VAE Lyra with CLIP-L + T5-base",
        "modalities": ["clip", "t5"],
        "fusion": "cantor",
        "latent_dim": 768,
        "recommended_for": "General text embedding transformation"
    },
    "AbstractPhil/vae-lyra-sdxl-t5xl": {
        "version": "v1",
        "description": "SDXL-compatible with CLIP-L + CLIP-G + T5-XL",
        "modalities": ["clip_l", "clip_g", "t5_xl"],
        "fusion": "cantor",
        "latent_dim": 2048,
        "recommended_for": "SDXL text encoder replacement"
    },
    "AbstractPhil/vae-lyra-xl-adaptive-cantor": {
        "version": "v2",
        "description": "Adaptive Cantor with decoupled T5 scales and learned parameters",
        "modalities": ["clip_l", "clip_g", "t5_xl_l", "t5_xl_g"],
        "fusion": "adaptive_cantor",
        "latent_dim": 2048,
        "has_adaptive_params": True,
        "has_variable_seq_lens": True,
        "recommended_for": "Advanced SDXL with geometric consciousness"
    }
}


def list_known_models():
    """List all known VAE Lyra models with descriptions."""
    print("\n" + "=" * 80)
    print("KNOWN VAE LYRA MODELS")
    print("=" * 80)

    for repo_id, info in KNOWN_MODELS.items():
        print(f"\n📦 {repo_id}")
        print(f"   Version: {info['version']}")
        print(f"   Description: {info['description']}")
        print(f"   Modalities: {', '.join(info['modalities'])}")
        print(f"   Fusion: {info['fusion']}")
        print(f"   Latent dim: {info['latent_dim']}")
        if info.get('has_adaptive_params'):
            print(f"   🎯 Learned alpha/beta parameters")
        if info.get('has_variable_seq_lens'):
            print(f"   📏 Variable sequence lengths")
        print(f"   Use case: {info['recommended_for']}")

    print("\n" + "=" * 80 + "\n")


def get_known_model_info(repo_id: str) -> Optional[Dict[str, Any]]:
    """Get registry info for a known model."""
    return KNOWN_MODELS.get(repo_id)


# ============================================================================
# VERSION DETECTION
# ============================================================================

def detect_lyra_version(config: Dict[str, Any], repo_id: Optional[str] = None) -> str:
    """
    Detect VAE Lyra version from configuration.

    Args:
        config: Configuration dictionary
        repo_id: Optional repository ID for registry lookup

    Returns:
        Version string: "v1" or "v2"
    """
    # Check registry first if repo_id provided
    if repo_id and repo_id in KNOWN_MODELS:
        return KNOWN_MODELS[repo_id]['version']

    # v2 signature features
    v2_indicators = [
        'modality_seq_lens',  # Variable sequence lengths (v2)
        'binding_config',  # Binding configuration (v2)
        'alpha_init',  # Adaptive alpha parameters (v2)
        'beta_init',  # Adaptive beta parameters (v2)
        'alpha_lr_scale',  # Alpha learning rate scaling (v2)
        'beta_lr_scale',  # Beta learning rate scaling (v2)
        'beta_alpha_regularization'  # Alpha regularization (v2)
    ]

    # Check for v2 indicators
    v2_score = sum(1 for key in v2_indicators if key in config)

    # Also check for adaptive_cantor fusion strategy
    if config.get('fusion_strategy') == 'adaptive_cantor':
        v2_score += 3  # Strong indicator

    # Decision threshold
    if v2_score >= 2:
        return "v2"
    else:
        return "v1"


def validate_config_for_version(config: Dict[str, Any], version: str) -> Tuple[bool, str]:
    """
    Validate that config is compatible with specified version.

    Args:
        config: Configuration dictionary
        version: Target version ("v1" or "v2")

    Returns:
        (is_valid, error_message)
    """
    if version == "v1":
        # v1 should not have v2-specific features
        v2_only_keys = ['modality_seq_lens', 'binding_config', 'alpha_init',
                        'beta_init', 'alpha_lr_scale', 'beta_lr_scale']

        found_v2_keys = [k for k in v2_only_keys if k in config]

        if found_v2_keys:
            return False, f"Config contains v2-only keys: {found_v2_keys}. Use load_vae_lyra_v2() instead."

        # v1 should not use adaptive_cantor
        if config.get('fusion_strategy') == 'adaptive_cantor':
            return False, "adaptive_cantor fusion requires v2. Use load_vae_lyra_v2() instead."

        return True, ""

    elif version == "v2":
        # v2 with adaptive_cantor needs specific config
        if config.get('fusion_strategy') == 'adaptive_cantor':
            required_keys = ['modality_seq_lens', 'binding_config']
            missing_keys = [k for k in required_keys if k not in config]

            if missing_keys:
                return False, f"adaptive_cantor requires: {missing_keys}"

        return True, ""

    else:
        return False, f"Unknown version: {version}"


# ============================================================================
# MODEL LOADING
# ============================================================================

def load_vae_lyra(
        repo_id: str,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        force_version: Optional[str] = None
) -> torch.nn.Module:
    """
    Load VAE Lyra from HuggingFace Hub with automatic version detection.

    Args:
        repo_id: HuggingFace repository ID (e.g., "AbstractPhil/vae-lyra")
        device: Device to load model on
        force_version: Force specific version ("v1" or "v2"), otherwise auto-detect

    Returns:
        Loaded VAE Lyra model

    Examples:
    #   >>> # Original model
    #   >>> model = load_vae_lyra("AbstractPhil/vae-lyra")

    #   >>> # SDXL model
    #   >>> model = load_vae_lyra("AbstractPhil/vae-lyra-sdxl-t5xl")

    #   >>> # Adaptive Cantor model
    #   >>> model = load_vae_lyra("AbstractPhil/vae-lyra-xl-adaptive-cantor")
    """
    print(f"🔍 Loading VAE Lyra from: {repo_id}")

    # Show registry info if available
    registry_info = get_known_model_info(repo_id)
    if registry_info:
        print(f"📋 Known model: {registry_info['description']}")
        print(f"   Modalities: {', '.join(registry_info['modalities'])}")

    # Download config
    try:
        config_path = hf_hub_download(
            repo_id=repo_id,
            filename="config.json",
            repo_type="model"
        )
    except Exception as e:
        raise ValueError(f"Could not download config from {repo_id}: {e}")

    # Load config
    with open(config_path) as f:
        config_dict = json.load(f)

    # Detect version
    if force_version:
        version = force_version
        print(f"⚙️  Forced version: {version}")
    else:
        version = detect_lyra_version(config_dict, repo_id)
        print(f"✓ Detected version: {version}")

    # Validate config
    is_valid, error_msg = validate_config_for_version(config_dict, version)
    if not is_valid:
        raise ValueError(f"Config validation failed: {error_msg}")

    # Load appropriate version
    if version == "v1":
        return load_vae_lyra_v1(repo_id, device, config_dict)
    elif version == "v2":
        return load_vae_lyra_v2(repo_id, device, config_dict)
    else:
        raise ValueError(f"Unknown version: {version}")


def load_vae_lyra_v1(
        repo_id: str,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        config_dict: Optional[Dict[str, Any]] = None
) -> torch.nn.Module:
    """
    Load VAE Lyra v1 (standard fusion strategies).

    Supports:
    - AbstractPhil/vae-lyra (CLIP-L + T5-base)
    - AbstractPhil/vae-lyra-sdxl-t5xl (CLIP-L + CLIP-G + T5-XL)

    Args:
        repo_id: HuggingFace repository ID
        device: Device to load model on
        config_dict: Optional pre-loaded config dictionary

    Returns:
        VAE Lyra v1 model
    """
    from .lyra import (
        MultiModalVAE,
        MultiModalVAEConfig
    )

    print("📦 Loading VAE Lyra v1...")

    # Download config if not provided
    if config_dict is None:
        config_path = hf_hub_download(
            repo_id=repo_id,
            filename="config.json",
            repo_type="model"
        )
        with open(config_path) as f:
            config_dict = json.load(f)

    # Download model weights
    model_path = hf_hub_download(
        repo_id=repo_id,
        filename="model.pt",
        repo_type="model"
    )

    # Create config
    vae_config = MultiModalVAEConfig(
        modality_dims=config_dict.get('modality_dims', {"clip": 768, "t5": 768}),
        latent_dim=config_dict.get('latent_dim', 768),
        seq_len=config_dict.get('seq_len', 77),
        encoder_layers=config_dict.get('encoder_layers', 3),
        decoder_layers=config_dict.get('decoder_layers', 3),
        hidden_dim=config_dict.get('hidden_dim', 1024),
        dropout=config_dict.get('dropout', 0.1),
        fusion_strategy=config_dict.get('fusion_strategy', 'cantor'),
        fusion_heads=config_dict.get('fusion_heads', 8),
        fusion_dropout=config_dict.get('fusion_dropout', 0.1),
        beta_kl=config_dict.get('beta_kl', 0.1),
        beta_reconstruction=config_dict.get('beta_reconstruction', 1.0),
        beta_cross_modal=config_dict.get('beta_cross_modal', 0.05),
        seed=config_dict.get('seed')
    )

    # Load model
    checkpoint = torch.load(model_path, map_location=device)
    model = MultiModalVAE(vae_config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)

    print(f"✓ Loaded v1 model from step {checkpoint.get('global_step', 'unknown')}")
    print(f"✓ Best loss: {checkpoint.get('best_loss', 'unknown')}")
    print(f"✓ Fusion strategy: {vae_config.fusion_strategy}")
    print(f"✓ Modalities: {list(vae_config.modality_dims.keys())}")
    print(f"✓ Latent dimension: {vae_config.latent_dim}")

    return model


def load_vae_lyra_v2(
        repo_id: str,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        config_dict: Optional[Dict[str, Any]] = None
) -> torch.nn.Module:
    """
    Load VAE Lyra v2 (adaptive Cantor with learned parameters).

    Supports:
    - AbstractPhil/vae-lyra-xl-adaptive-cantor

    Args:
        repo_id: HuggingFace repository ID
        device: Device to load model on
        config_dict: Optional pre-loaded config dictionary

    Returns:
        VAE Lyra v2 model
    """
    from .lyra_v2 import (
        MultiModalVAE,
        MultiModalVAEConfig
    )

    print("📦 Loading VAE Lyra v2 (Adaptive Cantor)...")

    # Download config if not provided
    if config_dict is None:
        config_path = hf_hub_download(
            repo_id=repo_id,
            filename="config.json",
            repo_type="model"
        )
        with open(config_path) as f:
            config_dict = json.load(f)

    # Download model weights
    model_path = hf_hub_download(
        repo_id=repo_id,
        filename="model.pt",
        repo_type="model"
    )

    # Create config with v2-specific parameters
    vae_config = MultiModalVAEConfig(
        modality_dims=config_dict.get('modality_dims'),
        modality_seq_lens=config_dict.get('modality_seq_lens'),
        binding_config=config_dict.get('binding_config'),
        latent_dim=config_dict.get('latent_dim', 2048),
        seq_len=config_dict.get('seq_len', 77),
        encoder_layers=config_dict.get('encoder_layers', 3),
        decoder_layers=config_dict.get('decoder_layers', 3),
        hidden_dim=config_dict.get('hidden_dim', 1024),
        dropout=config_dict.get('dropout', 0.1),
        fusion_strategy=config_dict.get('fusion_strategy', 'adaptive_cantor'),
        fusion_heads=config_dict.get('fusion_heads', 8),
        fusion_dropout=config_dict.get('fusion_dropout', 0.1),
        cantor_depth=config_dict.get('cantor_depth', 8),
        cantor_local_window=config_dict.get('cantor_local_window', 3),
        alpha_init=config_dict.get('alpha_init', 1.0),
        beta_init=config_dict.get('beta_init', 0.3),
        alpha_lr_scale=config_dict.get('alpha_lr_scale', 0.1),
        beta_lr_scale=config_dict.get('beta_lr_scale', 1.0),
        beta_kl=config_dict.get('beta_kl', 0.1),
        beta_reconstruction=config_dict.get('beta_reconstruction', 1.0),
        beta_cross_modal=config_dict.get('beta_cross_modal', 0.05),
        beta_alpha_regularization=config_dict.get('beta_alpha_regularization', 0.01),
        seed=config_dict.get('seed')
    )

    # Load model
    checkpoint = torch.load(model_path, map_location=device)
    model = MultiModalVAE(vae_config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)

    # Show learned parameters
    fusion_params = model.get_fusion_params()

    print(f"✓ Loaded v2 model from step {checkpoint.get('global_step', 'unknown')}")
    print(f"✓ Best loss: {checkpoint.get('best_loss', 'unknown')}")
    print(f"✓ Fusion strategy: {vae_config.fusion_strategy}")
    print(f"✓ Modalities: {list(vae_config.modality_dims.keys())}")
    print(f"✓ Latent dimension: {vae_config.latent_dim}")
    print(f"✓ Sequence lengths: {vae_config.modality_seq_lens}")

    if fusion_params:
        print(f"\n📊 Learned Parameters:")
        if 'alphas' in fusion_params:
            print(f"   Alpha (visibility):")
            for name, alpha in fusion_params['alphas'].items():
                print(f"     • {name}: {torch.sigmoid(alpha).item():.4f}")
        if 'betas' in fusion_params:
            print(f"   Beta (capacity):")
            for name, beta in fusion_params['betas'].items():
                print(f"     • {name}: {torch.sigmoid(beta).item():.4f}")

    return model


# ============================================================================
# MODEL INFORMATION
# ============================================================================

def get_model_info(repo_id: str) -> Dict[str, Any]:
    """
    Get information about a VAE Lyra model without loading it.

    Args:
        repo_id: HuggingFace repository ID

    Returns:
        Dictionary with model information
    """
    print(f"🔍 Inspecting model: {repo_id}")

    # Check registry first
    registry_info = get_known_model_info(repo_id)
    if registry_info:
        print(f"📋 Found in registry: {registry_info['description']}")

    # Download config
    try:
        config_path = hf_hub_download(
            repo_id=repo_id,
            filename="config.json",
            repo_type="model"
        )
    except Exception as e:
        raise ValueError(f"Could not download config from {repo_id}: {e}")

    # Load config
    with open(config_path) as f:
        config_dict = json.load(f)

    # Detect version
    version = detect_lyra_version(config_dict, repo_id)

    # Compile info
    info = {
        'repo_id': repo_id,
        'version': version,
        'fusion_strategy': config_dict.get('fusion_strategy', 'unknown'),
        'modality_dims': config_dict.get('modality_dims', {}),
        'latent_dim': config_dict.get('latent_dim', 'unknown'),
        'has_adaptive_params': version == 'v2' and config_dict.get('fusion_strategy') == 'adaptive_cantor',
        'has_variable_seq_lens': 'modality_seq_lens' in config_dict,
        'has_binding_config': 'binding_config' in config_dict,
    }

    # Add registry info if available
    if registry_info:
        info['registry_description'] = registry_info['description']
        info['recommended_for'] = registry_info['recommended_for']

    # Add v2-specific info
    if version == 'v2':
        info['modality_seq_lens'] = config_dict.get('modality_seq_lens', {})
        info['binding_config'] = config_dict.get('binding_config', {})
        info['alpha_init'] = config_dict.get('alpha_init', 'N/A')
        info['beta_init'] = config_dict.get('beta_init', 'N/A')

    return info


def print_model_info(repo_id: str):
    """
    Print formatted information about a VAE Lyra model.

    Args:
        repo_id: HuggingFace repository ID
    """
    info = get_model_info(repo_id)

    print(f"\n{'=' * 80}")
    print(f"VAE LYRA MODEL INFO")
    print(f"{'=' * 80}")
    print(f"Repository: {info['repo_id']}")
    print(f"Version: {info['version']}")

    if 'registry_description' in info:
        print(f"Description: {info['registry_description']}")
        print(f"Recommended for: {info['recommended_for']}")

    print(f"\nArchitecture:")
    print(f"  Fusion Strategy: {info['fusion_strategy']}")
    print(f"  Latent Dimension: {info['latent_dim']}")

    print(f"\nModalities:")
    for name, dim in info['modality_dims'].items():
        print(f"  • {name}: {dim}d", end="")
        if info['has_variable_seq_lens']:
            seq_len = info['modality_seq_lens'].get(name, 'unknown')
            print(f" @ {seq_len} tokens")
        else:
            print()

    if info['has_adaptive_params']:
        print(f"\n🎯 Adaptive Parameters:")
        print(f"  Alpha init: {info['alpha_init']}")
        print(f"  Beta init: {info['beta_init']}")

    if info['has_binding_config']:
        print(f"\n🔗 Binding Configuration:")
        for target, sources in info['binding_config'].items():
            if sources:
                print(f"  {target} ← {', '.join(f'{s} ({w})' for s, w in sources.items())}")
            else:
                print(f"  {target} (independent)")

    print(f"{'=' * 80}\n")


# ============================================================================
# COMPATIBILITY UTILITIES
# ============================================================================

def convert_v1_to_v2_config(v1_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert a v1 config to v2 format (for migration/compatibility).

    Args:
        v1_config: v1 configuration dictionary

    Returns:
        v2-compatible configuration dictionary
    """
    v2_config = v1_config.copy()

    # Add v2-specific defaults
    if 'modality_seq_lens' not in v2_config:
        # Create uniform sequence lengths based on seq_len
        seq_len = v2_config.get('seq_len', 77)
        modality_dims = v2_config.get('modality_dims', {})
        v2_config['modality_seq_lens'] = {
            name: seq_len for name in modality_dims.keys()
        }

    if 'binding_config' not in v2_config:
        # Create empty binding config (all independent)
        modality_dims = v2_config.get('modality_dims', {})
        v2_config['binding_config'] = {
            name: {} for name in modality_dims.keys()
        }

    if 'alpha_init' not in v2_config:
        v2_config['alpha_init'] = 1.0

    if 'beta_init' not in v2_config:
        v2_config['beta_init'] = 0.3

    if 'alpha_lr_scale' not in v2_config:
        v2_config['alpha_lr_scale'] = 0.1

    if 'beta_lr_scale' not in v2_config:
        v2_config['beta_lr_scale'] = 1.0

    if 'beta_alpha_regularization' not in v2_config:
        v2_config['beta_alpha_regularization'] = 0.01

    return v2_config


def is_compatible_checkpoint(model_state_dict: Dict[str, Any], config: Dict[str, Any]) -> Tuple[bool, str]:
    """
    Check if a checkpoint is compatible with a given config.

    Args:
        model_state_dict: Model state dictionary
        config: Configuration dictionary

    Returns:
        (is_compatible, message)
    """
    # Check for v2-specific keys in state dict
    has_alphas = any('alphas' in key for key in model_state_dict.keys())
    has_betas = any('betas' in key for key in model_state_dict.keys())

    version = detect_lyra_version(config)

    if version == 'v2' and config.get('fusion_strategy') == 'adaptive_cantor':
        if not (has_alphas and has_betas):
            return False, "Config specifies adaptive_cantor but checkpoint lacks alpha/beta parameters"

    if version == 'v1' and (has_alphas or has_betas):
        return False, "Config is v1 but checkpoint contains v2 parameters (alpha/beta)"

    return True, "Compatible"


# ============================================================================
# BATCH LOADING
# ============================================================================

def load_all_known_models(device: str = "cpu") -> Dict[str, torch.nn.Module]:
    """
    Load all known VAE Lyra models.

    Args:
        device: Device to load models on

    Returns:
        Dictionary of {repo_id: model}

    Warning: This will download and load all models, which may take significant
            time and memory.
    """
    print("🚀 Loading all known VAE Lyra models...")
    print("⚠️  This may take several minutes and require significant memory.\n")

    models = {}

    for repo_id in KNOWN_MODELS.keys():
        try:
            print(f"\nLoading {repo_id}...")
            model = load_vae_lyra(repo_id, device=device)
            models[repo_id] = model
            print(f"✓ Loaded {repo_id}")
        except Exception as e:
            print(f"✗ Failed to load {repo_id}: {e}")

    print(f"\n✓ Successfully loaded {len(models)}/{len(KNOWN_MODELS)} models")
    return models


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("VAE Lyra Loader - Examples with Real Models")
    print("=" * 80)

    # Example 1: List known models
    print("\n[Example 1] List all known models:")
    print("-" * 80)
    list_known_models()

    # Example 2: Get info for each model
    print("\n[Example 2] Inspect each known model:")
    print("-" * 80)

    for repo_id in KNOWN_MODELS.keys():
        try:
            print_model_info(repo_id)
        except Exception as e:
            print(f"⚠️  Could not load info for {repo_id}: {e}\n")

    # Example 3: Load specific models
    print("\n[Example 3] Load specific models:")
    print("-" * 80)

    models_to_try = [
        "AbstractPhil/vae-lyra",
        "AbstractPhil/vae-lyra-sdxl-t5xl",
        "AbstractPhil/vae-lyra-xl-adaptive-cantor"
    ]

    for repo_id in models_to_try:
        try:
            print(f"\nLoading {repo_id}...")
            model = load_vae_lyra(repo_id, device="cpu")
            print(f"✓ Successfully loaded: {type(model).__name__}")

            # Show some model details
            total_params = sum(p.numel() for p in model.parameters())
            print(f"✓ Total parameters: {total_params:,}")

        except Exception as e:
            print(f"✗ Failed: {e}")
            print("   (This is expected if the model hasn't been uploaded yet)")

    # Example 4: Compare models
    print("\n[Example 4] Compare model architectures:")
    print("-" * 80)

    print(f"\n{'Model':<45} {'Version':<8} {'Modalities':<8} {'Latent Dim':<12}")
    print("-" * 80)

    for repo_id, info in KNOWN_MODELS.items():
        short_name = repo_id.split('/')[-1]
        version = info['version']
        num_mods = len(info['modalities'])
        latent = info['latent_dim']
        print(f"{short_name:<45} {version:<8} {num_mods:<8} {latent:<12}")

    print("\n" + "=" * 80)
    print("✨ VAE Lyra Loader ready for use!")
    print("\nQuick start:")
    print("  from geovocab2.train.model.vae.loader import load_vae_lyra, list_known_models")
    print("  ")
    print("  # See what's available")
    print("  list_known_models()")
    print("  ")
    print("  # Load a model")
    print("  model = load_vae_lyra('AbstractPhil/vae-lyra-xl-adaptive-cantor')")
    print("  model.eval()")
    print("=" * 80)