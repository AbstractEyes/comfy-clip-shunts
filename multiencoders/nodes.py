# nodes.py

import random
import torch
from transformers import T5EncoderModel, T5Tokenizer
import json
from pathlib import Path

# Import the smart loader
from .lyra_loader import load_vae_lyra, get_model_info, list_known_models, detect_lyra_version


class TagRandomizer:
    def __init__(self) -> None:
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "tags": ("STRING", {
                    "multiline": True,
                }),
                "delimiter": ("STRING", {"default": ","}),
                "min_power": ("FLOAT", {"default": 0.5, "min": 0.1, "max": 2.0, "step": 0.1}),
                "max_power": ("FLOAT", {"default": 1.5, "min": 0.1, "max": 2.0, "step": 0.1}),
                "num_tags": ("INT", {"default": 5, "min": 1, "max": 200, "step": 1}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "mode": (["weighted", "normal"], {"default": "weighted"}),
            }
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "randomize_tags"
    CATEGORY = "text processing"

    def randomize_tags(self, tags, delimiter, min_power, max_power, num_tags, seed, mode):
        # Parse tags and clean up
        tag_list = [tag.strip() for tag in tags.split(delimiter) if tag.strip()]

        if not tag_list:
            return (",")

        # Set seed for reproducibility
        random.seed(seed)

        # Random selection without replacement
        selected_tags = random.sample(tag_list, min(num_tags, len(tag_list)))

        if mode == "weighted":
            # Assign random weights to each tag
            result_tags = []
            for tag in selected_tags:
                weight = round(random.uniform(min_power, max_power), 2)
                result_tags.append(f"({tag}:{weight})")
            result = ", ".join(result_tags)
        else:
            # Normal mode - just tags
            result = ", ".join(selected_tags)

        return (result,)


class VAELyraLoader:
    """Smart loader that auto-detects VAE Lyra version (v1/v2)"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "lyra_checkpoint": (
                    [
                        "AbstractPhil/vae-lyra",
                        "AbstractPhil/vae-lyra-sdxl-t5xl",
                        "AbstractPhil/vae-lyra-xl-adaptive-cantor",
                        "local"
                    ],
                    {"default": "AbstractPhil/vae-lyra-sdxl-t5xl"}
                ),
                "local_path": ("STRING", {
                    "default": "./checkpoints_lyra/best_model.pt",
                    "multiline": False
                }),
            }
        }

    RETURN_TYPES = ("VAE_LYRA", "STRING")
    RETURN_NAMES = ("lyra_model", "model_info")
    FUNCTION = "load_lyra"
    CATEGORY = "VAE Lyra"

    def load_lyra(self, lyra_checkpoint, local_path):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Local loading
        if lyra_checkpoint.lower() == "local":
            if Path(local_path).exists():
                model = self.load_lyra_from_local(local_path, device)
                info = f"Loaded local checkpoint: {local_path}"
            else:
                print(f"⚠️ Local checkpoint not found at: {local_path}")
                print(f"   Falling back to default HuggingFace repo...")
                model = load_vae_lyra("AbstractPhil/vae-lyra-sdxl-t5xl", device=device)
                info = "Loaded AbstractPhil/vae-lyra-sdxl-t5xl (fallback)"
        else:
            # Use smart loader for HF repos
            print(f"\n{'=' * 70}")
            model = load_vae_lyra(lyra_checkpoint, device=device)

            # Get model info
            config = getattr(model, 'config', None)
            if config:
                modalities = list(config.modality_dims.keys())
                info = f"{lyra_checkpoint} | Modalities: {', '.join(modalities)}"
            else:
                info = f"Loaded {lyra_checkpoint}"

            print(f"{'=' * 70}\n")

        return (model, info)

    def load_lyra_from_local(self, checkpoint_path, device):
        """Load from local checkpoint with version detection"""
        print(f"🎵 Loading VAE Lyra from local: {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location=device)

        if 'config' not in checkpoint:
            raise ValueError("Checkpoint missing config")

        config_dict = checkpoint['config']

        # CRITICAL: Get actual modality_dims from checkpoint
        if 'modality_dims' not in config_dict:
            raise ValueError("Config missing modality_dims")

        modality_dims = config_dict['modality_dims']
        print(f"✓ Modalities from checkpoint: {list(modality_dims.keys())}")

        # Detect version from config
        version = detect_lyra_version(config_dict)
        print(f"✓ Detected version: {version}")

        # Load appropriate version
        if version == "v1":
            from .lyra import MultiModalVAE, MultiModalVAEConfig

            vae_config = MultiModalVAEConfig(
                modality_dims=modality_dims,  # Use actual dims from checkpoint!
                latent_dim=config_dict.get('latent_dim', 768),
                seq_len=config_dict.get('seq_len', 77),
                encoder_layers=config_dict.get('encoder_layers', 3),
                decoder_layers=config_dict.get('decoder_layers', 3),
                hidden_dim=config_dict.get('hidden_dim', 1024),
                dropout=config_dict.get('dropout', 0.1),
                fusion_strategy=config_dict.get('fusion_strategy', 'cantor'),
                fusion_heads=config_dict.get('fusion_heads', 8),
                fusion_dropout=config_dict.get('fusion_dropout', 0.1),
                seed=config_dict.get('seed')
            )

            model = MultiModalVAE(vae_config)

        elif version == "v2":
            from .lyra_v2 import MultiModalVAE, MultiModalVAEConfig

            vae_config = MultiModalVAEConfig(
                modality_dims=modality_dims,  # Use actual dims from checkpoint!
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
                seed=config_dict.get('seed')
            )

            model = MultiModalVAE(vae_config)

        else:
            raise ValueError(f"Unknown version: {version}")

        # Load weights
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint

        # Try to load with strict=True first
        try:
            model.load_state_dict(state_dict, strict=True)
            print(f"✓ Loaded state dict with strict=True")
        except RuntimeError as e:
            print(f"⚠️  Strict loading failed: {str(e)[:200]}")
            print(f"   Attempting flexible loading...")

            # Get missing and unexpected keys
            model_keys = set(model.state_dict().keys())
            checkpoint_keys = set(state_dict.keys())

            missing_keys = model_keys - checkpoint_keys
            unexpected_keys = checkpoint_keys - model_keys

            if missing_keys:
                print(f"   Missing keys ({len(missing_keys)}):")
                for key in sorted(list(missing_keys)[:5]):
                    print(f"     - {key}")
                if len(missing_keys) > 5:
                    print(f"     ... and {len(missing_keys) - 5} more")

            if unexpected_keys:
                print(f"   Unexpected keys ({len(unexpected_keys)}):")
                for key in sorted(list(unexpected_keys)[:5]):
                    print(f"     - {key}")
                if len(unexpected_keys) > 5:
                    print(f"     ... and {len(unexpected_keys) - 5} more")

            # Load with strict=False
            result = model.load_state_dict(state_dict, strict=False)
            print(f"✓ Loaded state dict with strict=False")
            if result.missing_keys:
                print(f"   Note: {len(result.missing_keys)} keys initialized randomly")

        model.to(device).eval()

        total_params = sum(p.numel() for p in model.parameters())
        print(f"✓ VAE Lyra loaded: {total_params:,} parameters")
        print(f"✓ Modalities: {list(modality_dims.keys())}")
        print(f"✓ Latent dim: {vae_config.latent_dim}")
        print(f"✓ Fusion strategy: {vae_config.fusion_strategy}")

        # Show learned parameters for v2
        if version == "v2" and hasattr(model, 'get_fusion_params'):
            fusion_params = model.get_fusion_params()
            if fusion_params and 'alphas' in fusion_params:
                print(f"\n📊 Learned Parameters:")
                for name, alpha in fusion_params['alphas'].items():
                    print(f"   α_{name}: {torch.sigmoid(alpha).item():.4f}")
                for name, beta in fusion_params.get('betas', {}).items():
                    print(f"   β_{name}: {torch.sigmoid(beta).item():.4f}")

        return model


class T5Loader:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "t5_model": (["google/flan-t5-xl", "google/flan-t5-large", "google/flan-t5-base"],
                             {"default": "google/flan-t5-xl"}),
            }
        }

    RETURN_TYPES = ("T5_MODEL", "T5_TOKENIZER")
    RETURN_NAMES = ("t5_model", "t5_tokenizer")
    FUNCTION = "load_t5"
    CATEGORY = "VAE Lyra"

    def load_t5(self, t5_model):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"📝 Loading {t5_model}...")
        tokenizer = T5Tokenizer.from_pretrained(t5_model)
        model = T5EncoderModel.from_pretrained(t5_model).to(device).eval()
        print(f"✓ {t5_model} loaded")
        return (model, tokenizer)


# nodes.py (final fixed version)

class VAELyraEncode:
    """Universal VAE Lyra encoder - works with both v1 and v2 models"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "lyra_model": ("VAE_LYRA",),
                "t5_model": ("T5_MODEL",),
                "t5_tokenizer": ("T5_TOKENIZER",),
                "clip": ("CLIP",),
                "text": ("STRING", {"multiline": True, "default": "a beautiful sunset over mountains"}),
                "use_lyra": ("BOOLEAN", {"default": True, "label_on": "VAE Lyra", "label_off": "Standard"}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            }
        }

    RETURN_TYPES = ("CONDITIONING", "CONDITIONING")
    RETURN_NAMES = ("sd15_conditioning", "sdxl_conditioning")
    FUNCTION = "encode"
    CATEGORY = "VAE Lyra"

    def _get_t5_max_length(self, lyra_model, expected_modalities):
        """
        Determine T5 max_length from model config.

        CRITICAL: v1 fusion modules (Cantor, Geometric) require uniform sequence lengths.
        Only v2 adaptive fusion can support variable lengths.
        """
        model_config = getattr(lyra_model, 'config', None)

        if not model_config:
            return 77  # Safe default

        # Check the actual fusion module type
        fusion_module = getattr(lyra_model, 'fusion', None)
        fusion_class_name = fusion_module.__class__.__name__ if fusion_module else None

        # v1 fusion modules that require uniform sequence lengths
        v1_fusion_types = ['CantorModalityFusion', 'GeometricModalityFusion', 'Sequential']

        # Check if fusion module is v1 type
        is_v1_fusion = fusion_class_name in v1_fusion_types

        if is_v1_fusion:
            # v1 fusion ALWAYS requires uniform seq_len
            seq_len = getattr(model_config, 'seq_len', 77)
            print(f"   v1 fusion ({fusion_class_name}): forcing uniform seq_len = {seq_len}")
            return seq_len

        # Check if this is v2 with adaptive fusion and variable sequence lengths
        has_variable_seq_lens = (
                hasattr(model_config, 'modality_seq_lens') and
                model_config.modality_seq_lens is not None
        )

        if has_variable_seq_lens and fusion_class_name == 'AdaptiveCantorModalityFusion':
            # v2 adaptive model - use modality-specific sequence lengths
            seq_lens = model_config.modality_seq_lens
            # Look for T5 variants in order of preference
            for key in ['t5_xl_l', 't5_xl_g', 't5_xl', 't5']:
                if key in seq_lens:
                    t5_len = seq_lens[key]
                    print(f"   v2 adaptive fusion: using modality_seq_lens['{key}']: {t5_len}")
                    return t5_len

        # Fallback: use uniform seq_len
        seq_len = getattr(model_config, 'seq_len', 77)
        print(f"   Fallback: uniform seq_len = {seq_len}")
        return seq_len

    @torch.no_grad()
    def encode(self, lyra_model, t5_model, t5_tokenizer, clip, text, use_lyra, seed):
        device = next(lyra_model.parameters()).device

        # Set seed
        if seed == 0:
            seed = torch.random.seed()
        torch.manual_seed(seed)
        seed_gen = torch.Generator(device).manual_seed(int(seed))

        # Clean text for T5
        t5_text = text.replace("\n", " ").replace("(", " ").replace(")", " ").replace("[", " ").replace("]", " ")

        # Get CLIP embeddings from ComfyUI (should be SDXL)
        tokens = clip.tokenize(text)
        cond, pooled = clip.encode_from_tokens(tokens, return_pooled=True)
        clip_embed = cond.to(device)
        pooled = pooled.to(device)

        batch_size, seq_len, feat_dim = clip_embed.shape
        print(f"📊 Input CLIP: {clip_embed.shape} (feat_dim={feat_dim})")

        # Verify SDXL input
        if feat_dim != 2048:
            raise ValueError(
                f"VAE Lyra Encode requires SDXL CLIP (2048d). Got {feat_dim}d.\n"
                f"Please use an SDXL checkpoint with this node."
            )

        # Handle sequence length
        LYRA_MAX_TOKENS = 77
        if seq_len > LYRA_MAX_TOKENS:
            print(f"✂️ Slicing from {seq_len} to {LYRA_MAX_TOKENS} tokens")
            clip_embed_lyra = clip_embed[:, :LYRA_MAX_TOKENS, :]
            clip_extra_tokens = clip_embed[:, LYRA_MAX_TOKENS:, :]
        else:
            clip_embed_lyra = clip_embed
            clip_extra_tokens = None

        if not use_lyra:
            # Standard mode
            print("📋 Using standard embeddings (no VAE Lyra)")
            sdxl_cond = [[clip_embed, {"pooled_output": pooled}]]
            clip_l_only = clip_embed[:, :, :768]
            pooled_l = clip_l_only.mean(dim=1)
            sd15_cond = [[clip_l_only, {"pooled_output": pooled_l}]]
            return (sd15_cond, sdxl_cond)

        # VAE Lyra mode
        print("🎨 Processing through VAE Lyra")

        model_config = getattr(lyra_model, 'config', None)
        if model_config:
            expected_modalities = list(model_config.modality_dims.keys())
        else:
            expected_modalities = list(lyra_model.decoders.keys())

        print(f"   Model expects: {expected_modalities}")

        # Split SDXL embedding
        clip_l_embed = clip_embed_lyra[:, :, :768]
        clip_g_embed = clip_embed_lyra[:, :, 768:]

        # Get T5 max_length (checks fusion module type)
        t5_max_length = self._get_t5_max_length(lyra_model, expected_modalities)

        # Get T5 embeddings
        t5_tokens = t5_tokenizer(
            [t5_text],
            max_length=t5_max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        ).to(device)
        t5_embed = t5_model(**t5_tokens).last_hidden_state

        print(f"   T5 embeddings: {t5_embed.shape}")

        # Build inputs
        modality_inputs = {}
        target_modalities = []

        if 'clip' in expected_modalities and 't5' in expected_modalities:
            print(f"   Mode: Simple (clip + t5)")
            modality_inputs['clip'] = clip_l_embed
            modality_inputs['t5'] = t5_embed
            target_modalities = ['clip']

        elif 'clip_l' in expected_modalities and 'clip_g' in expected_modalities:
            print(f"   Mode: SDXL-style")
            modality_inputs['clip_l'] = clip_l_embed
            modality_inputs['clip_g'] = clip_g_embed

            if 't5_xl_l' in expected_modalities and 't5_xl_g' in expected_modalities:
                print(f"   Using decoupled T5 scales (v2 adaptive)")
                modality_inputs['t5_xl_l'] = t5_embed
                modality_inputs['t5_xl_g'] = t5_embed
            elif 't5_xl' in expected_modalities:
                print(f"   Using single T5 scale (v1)")
                modality_inputs['t5_xl'] = t5_embed
            elif 't5' in expected_modalities:
                modality_inputs['t5'] = t5_embed

            target_modalities = ['clip_l', 'clip_g']
        else:
            raise ValueError(f"Unexpected modality configuration: {expected_modalities}")

        print(f"   Inputs prepared: {list(modality_inputs.keys())}")
        print(f"   Input shapes: {[(k, v.shape) for k, v in modality_inputs.items()]}")

        # Forward pass
        output = lyra_model(
            modality_inputs,
            target_modalities=target_modalities,
            generator=seed_gen
        )

        # Handle return format
        if len(output) == 4:
            reconstructions, mu, logvar, per_modality_mus = output
        else:
            reconstructions, mu, logvar = output

        print(f"   Reconstructions: {list(reconstructions.keys())}")

        # Extract reconstructions
        if 'clip' in reconstructions:
            lyra_clip = reconstructions['clip']
            sd15_embed = lyra_clip
            sdxl_embed = torch.cat([lyra_clip, clip_g_embed], dim=-1)
            diff = (lyra_clip - clip_l_embed).abs()
            print(f"✓ CLIP Δ: max={diff.max().item():.4f}, mean={diff.mean().item():.4f}")

        elif 'clip_l' in reconstructions and 'clip_g' in reconstructions:
            lyra_clip_l = reconstructions['clip_l']
            lyra_clip_g = reconstructions['clip_g']
            sd15_embed = lyra_clip_l
            sdxl_embed = torch.cat([lyra_clip_l, lyra_clip_g], dim=-1)
            diff_l = (lyra_clip_l - clip_l_embed).abs()
            diff_g = (lyra_clip_g - clip_g_embed).abs()
            print(f"✓ CLIP-L Δ: max={diff_l.max().item():.4f}, mean={diff_l.mean().item():.4f}")
            print(f"✓ CLIP-G Δ: max={diff_g.max().item():.4f}, mean={diff_g.mean().item():.4f}")
        else:
            raise ValueError(f"Unexpected reconstruction keys: {list(reconstructions.keys())}")

        # swap last token for clip end token if needed
        if clip_extra_tokens is not None:
            # this means there were too many tokens, so the last token isn't the correct end token
            # so we will just replace the final token with the end token from the clip tokenizer
            sdxl_embed[:, -1, :] = torch.cat([clip_extra_tokens[:, -1, :768], clip_extra_tokens[:, -1, 768:]], dim=-1)

        # Pooled outputs
        sd15_pooled = sd15_embed.mean(dim=1)
        sdxl_pooled = pooled

        print(f"   SD1.5 output: {sd15_embed.shape}")
        print(f"   SDXL output: {sdxl_embed.shape}")

        # Show learned parameters
        if hasattr(lyra_model, 'get_fusion_params'):
            fusion_params = lyra_model.get_fusion_params()
            if fusion_params and 'alphas' in fusion_params:
                alpha_str = ', '.join(f'{k}={torch.sigmoid(v).item():.3f}'
                                      for k, v in list(fusion_params['alphas'].items())[:2])
                print(f"   Learned α: {alpha_str}")

        # Create conditioning
        sd15_cond = [[sd15_embed, {"pooled_output": sd15_pooled}]]
        sdxl_cond = [[sdxl_embed, {"pooled_output": sdxl_pooled}]]

        return (sd15_cond, sdxl_cond)

class VAELyraSD15Encode:
    """Specialized node for SD1.5 models only"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "lyra_model": ("VAE_LYRA",),
                "t5_model": ("T5_MODEL",),
                "t5_tokenizer": ("T5_TOKENIZER",),
                "clip": ("CLIP",),
                "text": ("STRING", {"multiline": True, "default": "a beautiful sunset over mountains"}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            }
        }

    RETURN_TYPES = ("CONDITIONING",)
    RETURN_NAMES = ("conditioning",)
    FUNCTION = "encode_sd15"
    CATEGORY = "VAE Lyra"

    def _get_t5_max_length(self, lyra_model, expected_modalities):
        """Determine T5 max_length from model config"""
        model_config = getattr(lyra_model, 'config', None)

        # Check for modality_seq_lens (v2)
        if model_config and hasattr(model_config, 'modality_seq_lens'):
            seq_lens = model_config.modality_seq_lens
            # Look for T5 variants
            for key in ['t5_xl_l', 't5_xl_g', 't5_xl', 't5']:
                if key in seq_lens:
                    return seq_lens[key]

        # Fallback: check if any T5 modality suggests longer sequences
        if any('t5_xl' in mod for mod in expected_modalities):
            return 512  # T5-XL typically uses 512
        elif any('t5' in mod for mod in expected_modalities):
            return 77  # T5-base uses 77

        return 77  # Default

    @torch.no_grad()
    def encode_sd15(self, lyra_model, t5_model, t5_tokenizer, clip, text, seed):
        device = next(lyra_model.parameters()).device

        # Set seed
        if seed == 0:
            seed = torch.random.seed()
        torch.manual_seed(seed)
        seed_gen = torch.Generator(device).manual_seed(int(seed))

        # Get CLIP embeddings
        tokens = clip.tokenize(text)
        cond, pooled = clip.encode_from_tokens(tokens, return_pooled=True)
        clip_embed = cond.to(device)
        pooled = pooled.to(device)

        # Verify this is SD1.5 CLIP
        if clip_embed.shape[-1] != 768:
            raise ValueError(
                f"This node requires SD1.5 CLIP (768d). Got {clip_embed.shape[-1]}d. "
                f"Use VAELyraEncode for SDXL.")

        # Detect model's expected modalities
        model_config = getattr(lyra_model, 'config', None)
        if model_config:
            expected_modalities = list(model_config.modality_dims.keys())
        else:
            expected_modalities = list(lyra_model.decoders.keys())

        # Get T5 max_length
        t5_max_length = self._get_t5_max_length(lyra_model, expected_modalities)
        print(f"🎵 VAE Lyra transforming SD1.5 embeddings...")
        print(f"   T5 max_length: {t5_max_length}")

        # Get T5 embeddings with correct max_length
        t5_tokens = t5_tokenizer(
            [text],
            max_length=t5_max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        ).to(device)
        t5_embed = t5_model(**t5_tokens).last_hidden_state

        # Build inputs
        if 'clip' in expected_modalities:
            modality_inputs = {
                'clip': clip_embed,
                't5': t5_embed
            }
            target_modalities = ['clip']
        else:
            # Might be SDXL model being used for SD1.5
            # Just use CLIP-L portion
            modality_inputs = {
                'clip_l': clip_embed,
                't5': t5_embed
            }
            target_modalities = ['clip_l']

        print(f"   Inputs: {list(modality_inputs.keys())}")

        output = lyra_model(
            modality_inputs,
            target_modalities=target_modalities,
            generator=seed_gen
        )

        # Handle return format
        if len(output) == 4:
            reconstructions, mu, logvar, per_modality_mus = output
        else:
            reconstructions, mu, logvar = output

        # Get the transformed embedding
        if 'clip' in reconstructions:
            transformed_embed = reconstructions['clip']
        elif 'clip_l' in reconstructions:
            transformed_embed = reconstructions['clip_l']
        else:
            raise ValueError(f"Unexpected reconstruction keys: {list(reconstructions.keys())}")

        # Create conditioning
        conditioning = [[transformed_embed, {"pooled_output": pooled}]]

        return (conditioning,)


class VAELyraLatentExplorer:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "lyra_model": ("VAE_LYRA",),
                "t5_model": ("T5_MODEL",),
                "t5_tokenizer": ("T5_TOKENIZER",),
                "clip": ("CLIP",),
                "texts": ("STRING", {"multiline": True, "default": "prompt 1\nprompt 2\nprompt 3"}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("similarity_report",)
    FUNCTION = "explore"
    CATEGORY = "VAE Lyra"

    def _get_t5_max_length(self, lyra_model, expected_modalities):
        """Determine T5 max_length from model config"""
        model_config = getattr(lyra_model, 'config', None)

        # Check for modality_seq_lens (v2)
        if model_config and hasattr(model_config, 'modality_seq_lens'):
            seq_lens = model_config.modality_seq_lens
            # Look for T5 variants
            for key in ['t5_xl_l', 't5_xl_g', 't5_xl', 't5']:
                if key in seq_lens:
                    return seq_lens[key]

        # Fallback: check if any T5 modality suggests longer sequences
        if any('t5_xl' in mod for mod in expected_modalities):
            print("   DEFAULTED to T5-XL modality")
            return 512  # T5-XL typically uses 512
        elif any('t5' in mod for mod in expected_modalities):
            print("   DEFAULTED to T5-base modality")
            return 77  # T5-base uses 77

        return 77  # Default

    @torch.no_grad()
    def explore(self, lyra_model, t5_model, t5_tokenizer, clip, texts):
        device = next(lyra_model.parameters()).device
        prompts = [p.strip() for p in texts.split('\n') if p.strip()]

        latents = []
        report_lines = ["🔍 VAE Lyra Latent Space Analysis", "=" * 50]

        # Detect model's expected modalities
        model_config = getattr(lyra_model, 'config', None)
        if model_config:
            expected_modalities = list(model_config.modality_dims.keys())
        else:
            expected_modalities = list(lyra_model.decoders.keys())

        report_lines.append(f"Model modalities: {', '.join(expected_modalities)}")

        # Get T5 max_length
        t5_max_length = self._get_t5_max_length(lyra_model, expected_modalities)
        report_lines.append(f"T5 max_length: {t5_max_length}")

        for prompt in prompts:
            # Get CLIP and T5 embeddings
            tokens = clip.tokenize(prompt)
            cond, _ = clip.encode_from_tokens(tokens, return_pooled=True)
            clip_embed = cond.to(device)

            # Handle different CLIP dimensions
            clip_dim = clip_embed.shape[-1]

            if clip_dim == 2048:
                # SDXL CLIP
                clip_l = clip_embed[:, :, :768]
                clip_g = clip_embed[:, :, 768:]
            else:
                # SD1.5 CLIP
                clip_l = clip_embed

            # Get T5 with correct max_length
            t5_tokens = t5_tokenizer(
                [prompt],
                max_length=t5_max_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            ).to(device)
            t5_embed = t5_model(**t5_tokens).last_hidden_state

            # Build inputs based on expected modalities
            if 'clip' in expected_modalities:
                modality_inputs = {
                    'clip': clip_l,
                    't5': t5_embed
                }
            elif 'clip_l' in expected_modalities:
                modality_inputs = {
                    'clip_l': clip_l,
                }
                if 'clip_g' in expected_modalities and clip_dim == 2048:
                    modality_inputs['clip_g'] = clip_g
                if 't5_xl' in expected_modalities:
                    modality_inputs['t5_xl'] = t5_embed
                elif 't5' in expected_modalities:
                    modality_inputs['t5'] = t5_embed

            # Encode
            output = lyra_model.encode(modality_inputs)
            if len(output) == 3:
                # v2 format
                mu, logvar, per_modality_mus = output
            else:
                # v1 format
                mu, logvar = output

            latents.append(mu.cpu())

            report_lines.append(f"\n📝 '{prompt[:60]}...'")
            report_lines.append(f"   μ: [{mu.min().item():.3f}, {mu.max().item():.3f}] "
                                f"mean={mu.mean().item():.3f} ± {mu.std().item():.3f}")
            report_lines.append(f"   σ²: logvar mean={logvar.mean().item():.3f}")

        # Compute similarities
        if len(prompts) > 1:
            report_lines.append("\n📊 Cosine Similarity Matrix:")
            for i in range(len(prompts)):
                row = []
                for j in range(len(prompts)):
                    sim = torch.nn.functional.cosine_similarity(
                        latents[i].flatten(),
                        latents[j].flatten(),
                        dim=0
                    )
                    row.append(f"{sim.item():.3f}")
                report_lines.append(f"  [{', '.join(row)}]")

        report = "\n".join(report_lines)
        print(report)

        return (report,)

# Register nodes
LYRA_NODE_CLASS_MAPPINGS = {
    "LyraTagRandomizer": TagRandomizer,
    "LyraVAELyraLoader": VAELyraLoader,
    "LyraT5Loader": T5Loader,
    "LyraVAELyraEncode": VAELyraEncode,
    "LyraVAELyraSD15Encode": VAELyraSD15Encode,
    "LyraVAELyraLatentExplorer": VAELyraLatentExplorer,
}

LYRA_NODE_DISPLAY_NAME_MAPPINGS = {
    "LyraTagRandomizer": "Tag Randomizer",
    "LyraVAELyraLoader": "Load VAE Lyra (Smart)",
    "LyraT5Loader": "Load T5 Model",
    "LyraVAELyraEncode": "VAE Lyra Encode (Universal)",
    "LyraVAELyraSD15Encode": "VAE Lyra Encode (SD1.5 Only)",
    "LyraVAELyraLatentExplorer": "VAE Lyra Latent Explorer",
}