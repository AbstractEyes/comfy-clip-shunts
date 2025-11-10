import random
import comfy
import torch
from transformers import T5EncoderModel, T5Tokenizer
import json


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


import torch
import nodes
import folder_paths
from transformers import T5EncoderModel, T5Tokenizer
import comfy.utils
import comfy.sd
from comfy.model_patcher import ModelPatcher
import numpy as np
from PIL import Image
import json
from pathlib import Path

# Try to import your VAE Lyra model
from .lyra import MultiModalVAE, MultiModalVAEConfig


class VAELyraLoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "lyra_checkpoint": ("STRING", {
                    "default": "AbstractPhil/vae-lyra-sdxl-t5xl",
                    "multiline": False
                }),
                "local_path": ("STRING", {
                    "default": "./checkpoints_lyra/best_model.pt",
                    "multiline": False
                }),
            }
        }

    RETURN_TYPES = ("VAE_LYRA",)
    RETURN_NAMES = ("lyra_model",)
    FUNCTION = "load_lyra"
    CATEGORY = "VAE Lyra"

    def load_lyra(self, lyra_checkpoint, local_path):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # If checkpoint string is "local", use local path
        if lyra_checkpoint.lower() == "local" and Path(local_path).exists():
            model = self.load_lyra_from_local(local_path, device)
        elif lyra_checkpoint.lower() == "local":
            print(f"⚠️ Local checkpoint not found at: {local_path}")
            print(f"   Falling back to default HuggingFace repo...")
            model = self.load_lyra_from_hub("AbstractPhil/vae-lyra-sdxl-t5xl", device)
        else:
            # Use the repo string directly
            model = self.load_lyra_from_hub(lyra_checkpoint, device)

        return (model,)

    def load_lyra_from_local(self, checkpoint_path, device):
        print(f"🎵 Loading VAE Lyra from local: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)

        if 'config' in checkpoint:
            config_dict = checkpoint['config']
        else:
            raise ValueError("Checkpoint missing config")

        vae_config = MultiModalVAEConfig(
            modality_dims=config_dict.get('modality_dims', {"clip_l": 768, "clip_g": 1280, "t5_xl": 2048}),
            latent_dim=config_dict.get('latent_dim', 2048),
            seq_len=config_dict.get('seq_len', 77),
            encoder_layers=config_dict.get('encoder_layers', 3),
            decoder_layers=config_dict.get('decoder_layers', 3),
            hidden_dim=config_dict.get('hidden_dim', 1024),
            dropout=config_dict.get('dropout', 0.1),
            fusion_strategy=config_dict.get('fusion_strategy', 'cantor'),
            fusion_heads=config_dict.get('fusion_heads', 8),
            fusion_dropout=config_dict.get('fusion_dropout', 0.1)
        )

        model = MultiModalVAE(vae_config)

        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)

        model.to(device).eval()
        print(f"✓ VAE Lyra loaded: {sum(p.numel() for p in model.parameters()):,} parameters")
        return model

    def load_lyra_from_hub(self, repo_id, device):
        from huggingface_hub import hf_hub_download

        print(f"🎵 Loading VAE Lyra from Hub: {repo_id}")

        try:
            model_path = hf_hub_download(repo_id=repo_id, filename="model.pt", repo_type="model")
            config_path = hf_hub_download(repo_id=repo_id, filename="config.json", repo_type="model")
        except Exception as e:
            raise ValueError(f"Failed to download VAE Lyra model from '{repo_id}': {e}")

        with open(config_path) as f:
            config_dict = json.load(f)

        vae_config = MultiModalVAEConfig(
            modality_dims=config_dict.get('modality_dims', {"clip_l": 768, "clip_g": 1280, "t5_xl": 2048}),
            latent_dim=config_dict.get('latent_dim', 2048),
            seq_len=config_dict.get('seq_len', 77),
            encoder_layers=config_dict.get('encoder_layers', 3),
            decoder_layers=config_dict.get('decoder_layers', 3),
            hidden_dim=config_dict.get('hidden_dim', 1024),
            dropout=config_dict.get('dropout', 0.1),
            fusion_strategy=config_dict.get('fusion_strategy', 'cantor'),
            fusion_heads=config_dict.get('fusion_heads', 8),
            fusion_dropout=config_dict.get('fusion_dropout', 0.1)
        )

        checkpoint = torch.load(model_path, map_location=device)
        model = MultiModalVAE(vae_config)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device).eval()

        print(f"✓ VAE Lyra loaded from Hub: {sum(p.numel() for p in model.parameters()):,} parameters")
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


class VAELyraEncode:
    """Universal VAE Lyra encoder - requires SDXL input, outputs both SD1.5 and SDXL conditioning"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "lyra_model": ("VAE_LYRA",),
                "t5_model": ("T5_MODEL",),
                "t5_tokenizer": ("T5_TOKENIZER",),
                "clip": ("CLIP",),  # Should be SDXL CLIP
                "text": ("STRING", {"multiline": True, "default": "a beautiful sunset over mountains"}),
                "use_lyra": ("BOOLEAN", {"default": True, "label_on": "VAE Lyra", "label_off": "Standard"}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            }
        }

    RETURN_TYPES = ("CONDITIONING", "CONDITIONING")
    RETURN_NAMES = ("sd15_conditioning", "sdxl_conditioning")
    FUNCTION = "encode"
    CATEGORY = "VAE Lyra"

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
            # Standard mode - return original embeddings for both outputs
            print("📋 Using standard embeddings (no VAE Lyra)")

            # SDXL: Full 2048d
            sdxl_cond = [[clip_embed, {"pooled_output": pooled}]]

            # SD1.5: Just CLIP-L (first 768 dims)
            clip_l_only = clip_embed[:, :, :768]
            pooled_l = clip_l_only.mean(dim=1)
            sd15_cond = [[clip_l_only, {"pooled_output": pooled_l}]]

            return (sd15_cond, sdxl_cond)

        # VAE Lyra mode
        print("🎨 Processing SDXL input through VAE Lyra")

        # Split SDXL embedding into CLIP-L + CLIP-G
        clip_l_embed = clip_embed_lyra[:, :, :768]  # First 768 dims
        clip_g_embed = clip_embed_lyra[:, :, 768:]  # Last 1280 dims

        # Get T5-XL embeddings (2048d)
        t5_tokens = t5_tokenizer(
            [t5_text],
            max_length=77,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        ).to(device)
        t5_embed = t5_model(**t5_tokens).last_hidden_state

        # Verify T5 is 2048d (T5-XL)
        if t5_embed.shape[-1] != 2048:
            raise ValueError(
                f"VAE Lyra requires T5-XL (2048d). Got {t5_embed.shape[-1]}d.\n"
                f"Use 'google/flan-t5-xl' in the T5Loader node."
            )

        print(f"🎵 Transforming with VAE Lyra...")
        print(f"   CLIP-L: {clip_l_embed.shape}, CLIP-G: {clip_g_embed.shape}, T5-XL: {t5_embed.shape}")

        # Process through VAE Lyra with all three modalities
        modality_inputs = {
            'clip_l': clip_l_embed,
            'clip_g': clip_g_embed,
            't5_xl': t5_embed
        }

        reconstructions, mu, logvar = lyra_model(
            modality_inputs,
            target_modalities=['clip_l', 'clip_g'],
            generator=seed_gen
        )

        lyra_clip_l = reconstructions['clip_l']
        lyra_clip_g = reconstructions['clip_g']

        # SD1.5 output: CLIP-L only (768d)
        sd15_embed = lyra_clip_l

        # Reattach extra tokens for SD1.5
        if clip_extra_tokens is not None:
            sd15_extra = clip_extra_tokens[:, :, :768]  # Only CLIP-L portion
            sd15_embed = torch.cat([sd15_embed, sd15_extra], dim=1)

        # Compute pooled from CLIP-L
        sd15_pooled = sd15_embed.mean(dim=1)

        # SDXL output: CLIP-L + CLIP-G concatenated (2048d)
        sdxl_embed = torch.cat([lyra_clip_l, lyra_clip_g], dim=-1)

        # Reattach extra tokens for SDXL
        if clip_extra_tokens is not None:
            print(f"🔗 Reattaching {clip_extra_tokens.shape[1]} extra tokens")
            sdxl_embed = torch.cat([sdxl_embed, clip_extra_tokens], dim=1)

        # Keep original SDXL pooled output
        sdxl_pooled = pooled

        # Stats
        diff_l = (lyra_clip_l - clip_l_embed).abs()
        diff_g = (lyra_clip_g - clip_g_embed).abs()
        print(f"✓ Transformation complete:")
        print(f"   CLIP-L Δ: max={diff_l.max().item():.4f}, mean={diff_l.mean().item():.4f}")
        print(f"   CLIP-G Δ: max={diff_g.max().item():.4f}, mean={diff_g.mean().item():.4f}")
        print(f"   SD1.5 output: {sd15_embed.shape} (768d)")
        print(f"   SDXL output: {sdxl_embed.shape} (2048d)")

        # Create conditioning outputs
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
            }
        }

    RETURN_TYPES = ("CONDITIONING",)
    RETURN_NAMES = ("conditioning",)
    FUNCTION = "encode_sd15"
    CATEGORY = "VAE Lyra"

    def encode_sd15(self, lyra_model, t5_model, t5_tokenizer, clip, text):
        device = next(lyra_model.parameters()).device

        # Get CLIP embeddings
        tokens = clip.tokenize(text)
        cond, pooled = clip.encode_from_tokens(tokens, return_pooled=True)
        clip_embed = cond.to(device)

        # Verify this is SD1.5 CLIP
        if clip_embed.shape[-1] != 768:
            raise ValueError(
                f"This node requires SD1.5 CLIP (768D). Got {clip_embed.shape[-1]}D. Use VAELyraEncode for SDXL.")


        # Get T5 embeddings
        t5_tokens = t5_tokenizer(
            [text],
            max_length=77,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        ).to(device)
        t5_embed = t5_model(**t5_tokens).last_hidden_state

        # Process through VAE Lyra
        modality_inputs = {
            'clip': clip_embed,
            't5': t5_embed
        }

        print(f"🎵 VAE Lyra transforming SD1.5 embeddings...")

        with torch.no_grad():
            reconstructions, mu, logvar = lyra_model(
                modality_inputs,
                target_modalities=['clip']
            )

            transformed_embed = reconstructions['clip']

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

    def explore(self, lyra_model, t5_model, t5_tokenizer, clip, texts):
        device = next(lyra_model.parameters()).device
        prompts = [p.strip() for p in texts.split('\n') if p.strip()]

        latents = []
        report_lines = ["🔍 VAE Lyra Latent Space Analysis", "=" * 50]

        for prompt in prompts:
            # Get CLIP and T5 embeddings
            tokens = clip.tokenize(prompt)
            cond, _ = clip.encode_from_tokens(tokens, return_pooled=True)
            clip_embed = cond.to(device)

            # Handle SDXL if needed
            if clip_embed.shape[-1] == 2048:
                if not hasattr(self, 'projection_matrix'):
                    self.projection_matrix = torch.randn(2048, 768, device=device) * 0.02
                clip_embed = torch.matmul(clip_embed, self.projection_matrix)

            t5_tokens = t5_tokenizer(
                [prompt],
                max_length=77,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            ).to(device)
            t5_embed = t5_model(**t5_tokens).last_hidden_state

            modality_inputs = {
                'clip': clip_embed,
                't5': t5_embed
            }

            with torch.no_grad():
                mu, logvar = lyra_model.encode(modality_inputs)
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
    "LyraVAELyraLoader": "Load VAE Lyra",
    "LyraT5Loader": "Load T5 Model",
    "LyraVAELyraEncode": "VAE Lyra Encode (Universal)",
    "LyraVAELyraSD15Encode": "VAE Lyra Encode (SD1.5 Only)",
    "LyraVAELyraLatentExplorer": "VAE Lyra Latent Explorer",
}