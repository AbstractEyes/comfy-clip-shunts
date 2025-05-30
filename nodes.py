import torch
import logging
from typing import Optional, Dict, Any

from .configs import T5_CONFIGS, T5_SHUNT_REPOS
from .model_manager import get_model_manager, ModelType

logger = logging.getLogger(__name__)


class T5LoaderTest:
    """
    Loads T5 encoder-decoder model and prepares tokenized context for adapters.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_name": (list(T5_CONFIGS.keys()), {"default": "google/flan-t5-base"}),
                "local_path": ("STRING", {"default": "", "tooltip": "Local path override. If empty, use HuggingFace."}),
                "context_window": ("STRING", {"default": "a photo of a robot.", "multiline": True}),
                "use_context_window": ("BOOLEAN", {"default": True}),
                "sliding_window_size": ("INT", {"default": 512, "min": 1, "max": 2048}),
                "sliding_window_stride": ("INT", {"default": 256, "min": 1, "max": 2048}),
                "max_length": ("INT", {"default": 77, "min": 1, "max": 512}),
                "padding": (["max_length", "longest", "do_not_pad"], {"default": "max_length"}),
                "max_slices": ("INT", {"default": 1, "min": 1, "max": 100}),
                "min_slices": ("INT", {"default": 1, "min": 1, "max": 100}),
                "truncate_option": (["fold", "interpolate", "collapse", "zipper"], {"default": "fold"}),
                "device": (["cpu", "cuda", "mps"], {"default": "cuda" if torch.cuda.is_available() else "cpu"})
            }
        }

    RETURN_TYPES = ("T5_PIPE",)
    RETURN_NAMES = ("t5_pipe",)
    FUNCTION = "load"
    CATEGORY = "adapter/testing"

    def load(self, model_name, local_path, context_window, use_context_window,
             sliding_window_size, sliding_window_stride, max_length, padding,
             max_slices, min_slices, truncate_option, device):
        """Load the T5 model and tokenizer, and encode a sample context window."""

        # Get model manager
        model_manager = get_model_manager()

        # Determine model source
        model_config = T5_CONFIGS.get(model_name, {})
        model_source = local_path or model_config.get("repo_name", "")

        if not model_source:
            raise ValueError(f"No path found for model '{model_name}'.")

        # Create unique model ID
        model_id = f"t5_{model_name}_{hash(model_source)}"

        # Load model and tokenizer through manager
        device_obj = torch.device(device)
        result = model_manager.load_t5_model(
            model_id=model_id,
            model_name_or_path=model_source,
            device=device_obj,
            dtype=torch.float32
        )

        if not result:
            raise RuntimeError(f"Failed to load T5 model: {model_name}")

        model, tokenizer = result

        # Tokenize context
        input_ids, attention_mask = None, None
        if use_context_window:
            tokens = tokenizer(
                context_window,
                return_tensors="pt",
                padding=padding if padding != "do_not_pad" else False,
                truncation=True,
                max_length=max_length
            )
            input_ids = tokens["input_ids"].to(device_obj)
            attention_mask = tokens["attention_mask"].to(device_obj)

        return ({
                    "model": model,
                    "tokenizer": tokenizer,
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "prompt": context_window,
                    "max_slices": max_slices,
                    "min_slices": min_slices,
                    "sliding_window_size": sliding_window_size,
                    "sliding_window_stride": sliding_window_stride,
                    "use_context_window": use_context_window,
                    "max_length": max_length,
                    "padding": padding,
                    "truncate_option": truncate_option,
                    "device": str(device),
                    "model_id": model_id  # Include for tracking
                },)


class LoadAdapterShunt:
    """Load a shunt adapter with improved model management."""

    @classmethod
    def INPUT_TYPES(cls):
        # Get all available shunts
        shunt_list = []
        for shunt_type, shunt_info in T5_SHUNT_REPOS.items():
            shunt_list.extend(shunt_info["shunts_available"]["shunt_list"])

        return {
            "required": {
                "adapter_path": ("STRING", {
                    "default": "",
                    "tooltip": "Full path to the adapter .safetensors or .pt file."
                }),
                "shunt_type": (
                    list(T5_SHUNT_REPOS.keys()),
                    {
                        "default": list(T5_SHUNT_REPOS.keys())[0],
                        "tooltip": "The shunt variant (clip_l, clip_g, etc)."
                    }
                ),
                "shunt_name": (
                    shunt_list,
                    {
                        "default": shunt_list[0] if shunt_list else "",
                        "tooltip": "Which preconfigured shunt to load from the repo."
                    }
                ),
                "device": (
                    ["cpu", "cuda", "mps"],
                    {
                        "default": "cuda" if torch.cuda.is_available() else "cpu"
                    }
                )
            }
        }

    RETURN_TYPES = ("ADAPTER",)
    RETURN_NAMES = ("shunt_adapter",)
    FUNCTION = "load_adapter"
    CATEGORY = "adapter/shunt"

    def load_adapter(self, adapter_path, shunt_type, shunt_name, device):
        """Load adapter using the refactored model manager."""

        # Get model manager
        model_manager = get_model_manager()

        # Get configuration
        config_entry = T5_SHUNT_REPOS.get(shunt_type)
        if not config_entry:
            raise ValueError(f"Unknown shunt type: {shunt_type}")

        # Create unique adapter ID
        adapter_id = f"shunt_{shunt_type}_{shunt_name}"

        # Prepare loading parameters
        device_obj = torch.device(device)
        repo_id = config_entry.get("repo")
        config = config_entry.get("config", {})

        # Load adapter
        adapter = model_manager.load_shunt_adapter(
            adapter_id=adapter_id,
            config=config,
            path=adapter_path if adapter_path else None,
            repo_id=repo_id if not adapter_path else None,
            filename=shunt_name if not adapter_path else None,
            device=device_obj,
            dtype=torch.float32
        )

        if not adapter:
            raise RuntimeError(f"Failed to load adapter '{shunt_name}' of type '{shunt_type}'")

        logger.info(f"Successfully loaded adapter: {adapter_id}")

        return ([{
            "adapter": adapter,
            "adapter_id": adapter_id,
            "config": config,
            "timestep_start": 0.0,
            "timestep_end": 1.0,
            "config_overrides": None
        }],)


class ShuntConditioning:
    """Apply adapter to conditioning with automatic slicing for CLIP-L/G"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "conditioning": ("CONDITIONING", {}),
                "t5_pipe": ("T5_PIPE", {}),
                "adapter": ("ADAPTER", {}),
                "strength": ("FLOAT", {"default": 1.0, "min": -50.0, "max": 50.0, "step": 0.1}),
                "delta_mean": ("FLOAT", {"default": 0.0, "min": -10.0, "max": 10.0, "step": 0.1}),
                "log_sigma": ("FLOAT", {"default": 0.0, "min": -10.0, "max": 10.0, "step": 0.1}),
                "gate_probability": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "g_pred": ("FLOAT", {"default": 0.0, "min": -10.0, "max": 10.0})
            }
        }

    RETURN_TYPES = ("CONDITIONING",)
    RETURN_NAMES = ("adapted_conditioning",)
    FUNCTION = "adapt_conditioning"
    CATEGORY = "adapter/shunt"

    def adapt_conditioning(self, conditioning, t5_pipe, adapter, strength,
                           delta_mean, log_sigma, gate_probability, g_pred):

        logger.info(f"Adapting conditioning with {len(adapter)} adapters")

        device = torch.device(t5_pipe["device"])

        # Get T5 embeddings
        with torch.no_grad():
            t5_embeddings = t5_pipe["model"].encoder(
                input_ids=t5_pipe["input_ids"],
                attention_mask=t5_pipe.get("attention_mask")
            ).last_hidden_state

        # Process conditioning
        adapted_conditioning = []
        for cond_idx, (cond_tensor, cond_meta) in enumerate(conditioning):
            cond_tensor = cond_tensor.clone().to(device)

            logger.info(f"Processing conditioning {cond_idx}: shape {cond_tensor.shape}")

            # Track which parts of the conditioning have been modified
            modified_ranges = []

            # Apply each adapter
            for adapter_idx, adapter_info in enumerate(adapter):
                adapter_model = adapter_info["adapter"].to(device)
                adapter_config = adapter_info["config"]

                # Determine adapter type and slice conditioning accordingly
                clip_dim = adapter_config.get("clip", {}).get("hidden_size", 768)
                total_dim = cond_tensor.size(-1)

                if clip_dim == 768:  # CLIP-L
                    # Take first 768 dimensions
                    clip_slice = cond_tensor[:, :, :768]
                    slice_start = 0
                    slice_end = 768
                    adapter_type = "clip_l"
                elif clip_dim == 1280:  # CLIP-G
                    # SDXL conditioning is typically CLIP-L (768) + CLIP-G (1280) = 2048
                    if total_dim >= 2048:
                        # Take from 768 to 2048 (the CLIP-G portion)
                        clip_slice = cond_tensor[:, :, 768:2048]
                        slice_start = 768
                        slice_end = 2048
                    else:
                        # Fallback for non-standard conditioning
                        clip_slice = cond_tensor[:, :, 768:]
                        slice_start = 768
                        slice_end = total_dim
                    adapter_type = "clip_g"
                else:
                    logger.warning(f"Unknown CLIP dimension {clip_dim}, skipping adapter")
                    continue

                logger.info(
                    f"Conditioning total dim: {total_dim}, extracting {adapter_type} [{slice_start}:{slice_end}]")

                # Validate slice dimensions
                if clip_slice.size(-1) != clip_dim:
                    logger.error(
                        f"Expected {clip_dim} dims for {adapter_type}, but slice has {clip_slice.size(-1)}. "
                        f"Conditioning tensor total size: {cond_tensor.size(-1)}"
                    )
                    continue

                logger.info(f"Applying {adapter_type} adapter to dims [{slice_start}:{slice_end}]")

                try:
                    # Forward pass with the sliced conditioning
                    outputs = adapter_model(t5_embeddings.float(), clip_slice.float())

                    # Unpack outputs
                    if isinstance(outputs, tuple) and len(outputs) == 8:
                        anchor, delta_mean_adapter, log_sigma_adapter, _, _, _, g_pred_adapter, gate_adapter = outputs
                    else:
                        raise ValueError(f"Unexpected adapter output format: {type(outputs)}")

                    # Apply modifications
                    gate = gate_adapter * gate_probability
                    delta = (delta_mean_adapter + delta_mean) * strength * gate

                    # Resize if needed
                    if delta.shape[1] != clip_slice.shape[1]:
                        logger.info(f"Resizing delta from {delta.shape} to match slice {clip_slice.shape}")
                        delta = torch.nn.functional.interpolate(
                            delta.transpose(1, 2),
                            size=clip_slice.size(1),
                            mode="nearest"
                        ).transpose(1, 2)

                    # Apply delta only to the appropriate slice
                    cond_tensor[:, :, slice_start:slice_end] = (
                            clip_slice.float() + delta
                    ).type_as(cond_tensor)

                    modified_ranges.append((slice_start, slice_end, adapter_type))
                    logger.info(f"Successfully applied {adapter_type} adapter")

                except Exception as e:
                    logger.error(f"Error applying adapter {adapter_idx}: {e}")
                    continue

            # Log what was modified
            if modified_ranges:
                logger.info(f"Modified conditioning ranges: {modified_ranges}")
            else:
                logger.warning("No adapters were successfully applied to this conditioning")

            adapted_conditioning.append([cond_tensor, cond_meta])

        if not adapted_conditioning:
            raise RuntimeError("No conditioning was successfully adapted")

        return (adapted_conditioning,)


class StackShuntAdapters:
    """Stack multiple adapters together."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "adapter_1": ("ADAPTER", {}),
                "adapter_2": ("ADAPTER", {}),
            }
        }

    RETURN_NAMES = ("adapters",)
    RETURN_TYPES = ("ADAPTER",)
    FUNCTION = "stack_adapters"
    CATEGORY = "adapter/shunt"

    def stack_adapters(self, adapter_1: list, adapter_2: list):
        """Combine two adapter lists."""
        logger.info(f"Stacking {len(adapter_1)} + {len(adapter_2)} adapters")

        # Create new list to avoid modifying inputs
        stacked = adapter_1.copy()
        stacked.extend(adapter_2)

        return (stacked,)


class UnloadShuntModels:
    """Node to unload models from memory."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_id": ("STRING", {
                    "default": "",
                    "tooltip": "Model ID to unload, or 'all' to clear all models"
                }),
            }
        }

    RETURN_TYPES = ()
    FUNCTION = "unload"
    CATEGORY = "adapter/utils"
    OUTPUT_NODE = True

    def unload(self, model_id):
        """Unload specified model or all models."""
        model_manager = get_model_manager()

        if model_id.lower() == "all":
            model_manager.clear_all()
            logger.info("Cleared all models from memory")
        elif model_id:
            success = model_manager.unload_model(model_id)
            if success:
                logger.info(f"Unloaded model: {model_id}")
            else:
                logger.warning(f"Failed to unload model: {model_id}")

        return ()


class ListLoadedShuntModels:
    """Node to list all currently loaded models."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {}}

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("model_list",)
    FUNCTION = "list_models"
    CATEGORY = "adapter/utils"

    def list_models(self):
        """List all loaded models."""
        model_manager = get_model_manager()
        models = model_manager.list_models()

        if not models:
            return ("No models currently loaded",)

        # Format output
        output_lines = ["Loaded Models:"]
        for model_id, info in models.items():
            output_lines.append(
                f"  - {model_id}: {info['type']} on {info['device']} ({info['dtype']})"
            )

        return ("\n".join(output_lines),)


import torch
import torch.nn.functional as F
import numpy as np


class ShuntConditioningAdvanced:
    """Advanced adapter with full capabilities including timestep scheduling and noise injection"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "conditioning": ("CONDITIONING", {}),
                "t5_pipe": ("T5_PIPE", {}),
                "adapter": ("ADAPTER", {}),
                "strength": ("FLOAT", {"default": 1.0, "min": -50.0, "max": 50.0, "step": 0.1}),
                "delta_mean": ("FLOAT", {"default": 0.0, "min": -10.0, "max": 10.0, "step": 0.1}),
                "log_sigma": ("FLOAT", {"default": 0.0, "min": -10.0, "max": 10.0, "step": 0.1}),
                "gate_probability": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "g_pred_scale": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.1}),
                "noise_injection": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "use_anchor": ("BOOLEAN", {"default": False}),
                "timestep_start": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "timestep_end": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("CONDITIONING", "FLOAT")
    RETURN_NAMES = ("adapted_conditioning", "guidance_scale")
    FUNCTION = "adapt_conditioning"
    CATEGORY = "adapter/advanced"

    def adapt_conditioning(self, conditioning, t5_pipe, adapter, strength,
                           delta_mean, log_sigma, gate_probability, g_pred_scale,
                           noise_injection, use_anchor, timestep_start, timestep_end):

        device = torch.device(t5_pipe["device"])

        # Get T5 embeddings
        with torch.no_grad():
            t5_embeddings = t5_pipe["model"].encoder(
                input_ids=t5_pipe["input_ids"],
                attention_mask=t5_pipe.get("attention_mask")
            ).last_hidden_state

        # Track guidance predictions
        all_g_preds = []

        # Process conditioning
        adapted_conditioning = []
        for cond_tensor, cond_meta in conditioning:
            cond_tensor = cond_tensor.clone().to(device)
            total_dim = cond_tensor.size(-1)

            # Apply each adapter
            for adapter_info in adapter:
                adapter_model = adapter_info["adapter"].to(device)
                adapter_config = adapter_info["config"]

                # Check timestep scheduling

                cond_meta["timestep_start"] = timestep_start
                cond_meta["timestep_end"] = timestep_end

                # Determine clip dimensions and slice
                clip_dim = adapter_config.get("clip", {}).get("hidden_size", 768)

                if clip_dim == 768:  # CLIP-L
                    clip_slice = cond_tensor[:, :, :768]
                    slice_start, slice_end = 0, 768
                elif clip_dim == 1280:  # CLIP-G
                    if total_dim >= 2048:
                        clip_slice = cond_tensor[:, :, 768:2048]
                        slice_start, slice_end = 768, 2048
                    else:
                        clip_slice = cond_tensor[:, :, 768:]
                        slice_start, slice_end = 768, total_dim
                else:
                    continue

                # Forward pass
                anchor, delta_mean_out, log_sigma_out, _, _, _, g_pred, gate = \
                    adapter_model(t5_embeddings.float(), clip_slice.float())

                # Apply modifications with timestep scaling
                effective_strength = strength
                gate = gate * gate_probability
                delta = (delta_mean_out + delta_mean) * effective_strength * gate

                # Optionally use anchor
                if use_anchor:
                    # Blend between original and anchor based on gate
                    clip_slice = clip_slice * (1 - gate) + anchor * gate

                # Apply noise injection
                if noise_injection > 0:
                    sigma = torch.exp(log_sigma_out + log_sigma)
                    noise = torch.randn_like(clip_slice) * sigma * noise_injection
                    clip_slice = clip_slice + noise

                # Resize delta if needed
                if delta.shape[1] != clip_slice.shape[1]:
                    delta = F.interpolate(
                        delta.transpose(1, 2),
                        size=clip_slice.size(1),
                        mode="nearest"
                    ).transpose(1, 2)

                # Apply delta
                cond_tensor[:, :, slice_start:slice_end] = (
                        clip_slice.float() + delta
                ).type_as(cond_tensor)

                # Collect guidance predictions
                if g_pred is not None:
                    all_g_preds.append(g_pred.mean().item())

            adapted_conditioning.append([cond_tensor, cond_meta])

        # Calculate average guidance scale
        avg_guidance = np.mean(all_g_preds) * g_pred_scale if all_g_preds else 0.0

        return (adapted_conditioning, float(avg_guidance))


class ShuntScheduler:
    """Schedule adapter strength over timesteps"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "adapter": ("ADAPTER", {}),
                "schedule_type": (["constant", "linear", "cosine", "exponential"], {}),
                "start_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.1}),
                "end_strength": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 2.0, "step": 0.1}),
                "timestep_start": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "timestep_end": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("ADAPTER",)
    RETURN_NAMES = ("scheduled_adapter",)
    FUNCTION = "schedule"
    CATEGORY = "adapter/scheduling"

    def schedule(self, adapter, schedule_type, start_strength, end_strength,
                 timestep_start, timestep_end):
        scheduled_adapters = []

        for adapter_info in adapter:
            scheduled_info = adapter_info.copy()
            scheduled_info.update({
                "timestep_start": timestep_start,
                "timestep_end": timestep_end,
                "schedule_type": schedule_type,
                "start_strength": start_strength,
                "end_strength": end_strength
            })
            scheduled_adapters.append(scheduled_info)

        return (scheduled_adapters,)


class VisualizeShuntEffect:
    """Visualize the effect of adapters on conditioning"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "original_conditioning": ("CONDITIONING", {}),
                "adapted_conditioning": ("CONDITIONING", {}),
                "visualization_type": (["difference", "heatmap", "magnitude"], {}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("visualization",)
    FUNCTION = "visualize"
    CATEGORY = "adapter/debug"

    def visualize(self, original_conditioning, adapted_conditioning, visualization_type):
        import matplotlib.pyplot as plt
        import matplotlib.cm as cm

        # Get first conditioning pair
        orig_cond = original_conditioning[0][0]
        adapt_cond = adapted_conditioning[0][0]

        # clone and cast to the cpu
        orig_cond = orig_cond.clone().cpu()
        adapt_cond = adapt_cond.clone().cpu()

        # Calculate difference
        diff = adapt_cond - orig_cond

        # Create visualization
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # Original
        axes[0].imshow(orig_cond[0].cpu().numpy().T, aspect='auto', cmap='viridis')
        axes[0].set_title("Original Conditioning")

        # Adapted
        axes[1].imshow(adapt_cond[0].cpu().numpy().T, aspect='auto', cmap='viridis')
        axes[1].set_title("Adapted Conditioning")

        # Difference
        if visualization_type == "difference":
            im = axes[2].imshow(diff[0].cpu().numpy().T, aspect='auto', cmap='RdBu', vmin=-1, vmax=1)
        elif visualization_type == "heatmap":
            im = axes[2].imshow(torch.abs(diff[0]).cpu().numpy().T, aspect='auto', cmap='hot')
        else:  # magnitude
            magnitude = torch.norm(diff[0], dim=-1).cpu().numpy()
            im = axes[2].imshow(magnitude[:, None], aspect='auto', cmap='plasma')

        axes[2].set_title(f"{visualization_type.capitalize()}")

        plt.colorbar(im, ax=axes[2])
        plt.tight_layout()

        # Convert to image tensor
        fig.canvas.draw()
        img_array = np.frombuffer(fig.canvas.buffer_rgb(), dtype=np.uint8)
        img_array = img_array.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        plt.close()

        # Convert to ComfyUI format
        img_tensor = torch.from_numpy(img_array).float() / 255.0
        img_tensor = img_tensor.unsqueeze(0)

        return (img_tensor,)


class MergeShunts:
    """Merge multiple adapters with different weights"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "adapter_1": ("ADAPTER", {}),
                "adapter_2": ("ADAPTER", {}),
                "weight_1": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.1}),
                "weight_2": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.1}),
                "merge_type": (["weighted_sum", "max", "min", "multiply"], {}),
            }
        }

    RETURN_TYPES = ("ADAPTER",)
    RETURN_NAMES = ("merged_adapter",)
    FUNCTION = "merge"
    CATEGORY = "adapter/advanced"

    def merge(self, adapter_1, adapter_2, weight_1, weight_2, merge_type):
        # This would require modifying the adapter models themselves
        # For now, just return a list with both adapters and adjusted weights
        merged = []

        for a in adapter_1:
            a_copy = a.copy()
            a_copy["merge_weight"] = weight_1
            merged.append(a_copy)

        for a in adapter_2:
            a_copy = a.copy()
            a_copy["merge_weight"] = weight_2
            merged.append(a_copy)

        return (merged,)


import torch
import logging
from typing import Dict, List, Tuple, Optional


class SimpleShuntSetup:
    """One-click setup for common shunt configurations"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mode": ([
                             "SDXL Standard",
                             "SD 1.5",
                             "Artistic Enhancement",
                             "Photorealistic",
                             "Anime/Illustration"
                         ], {}),
                "strength": (["Low", "Medium", "High", "Very High"], {"default": "Medium"}),
                "prompt": ("STRING", {"default": "a beautiful scene", "multiline": True}),
            }
        }

    RETURN_TYPES = ("SHUNT_CONFIG",)
    RETURN_NAMES = ("shunt_config",)
    FUNCTION = "setup"
    CATEGORY = "adapter/simple"

    def setup(self, mode, strength, prompt):
        # Strength presets
        strength_values = {
            "Low": 0.3,
            "Medium": 0.7,
            "High": 1.0,
            "Very High": 1.5
        }

        # Mode presets
        mode_configs = {
            "SDXL Standard": {
                "shunt_types": ["clip_l", "clip_g"],
                "shunt_names": ["t5-vit-l-14-dual_shunt_caption.safetensors",
                                "t5-flan-vit-bigG-14-dual_shunt_caption.safetensors"],
                "t5_model": "google/flan-t5-base",
                "gate_probability": 0.9,
                "noise_injection": 0.0
            },
            "SD 1.5": {
                "shunt_types": ["clip_l"],
                "shunt_names": ["t5-vit-l-14-dual_shunt_caption.safetensors"],
                "t5_model": "google/flan-t5-base",
                "gate_probability": 0.85,
                "noise_injection": 0.0
            },
            "Artistic Enhancement": {
                "shunt_types": ["clip_l", "clip_g"],
                "shunt_names": ["t5-vit-l-14-dual_shunt_no_caption.safetensors",
                                "t5-flan-vit-bigG-14-dual_shunt_no_caption_e3.safetensors"],
                "t5_model": "google/flan-t5-base",
                "gate_probability": 1.0,
                "noise_injection": 0.05
            },
            "Photorealistic": {
                "shunt_types": ["clip_l", "clip_g"],
                "shunt_names": ["t5-vit-l-14-dual_shunt_caption.safetensors",
                                "t5-flan-vit-bigG-14-dual_shunt_caption.safetensors"],
                "t5_model": "google/flan-t5-base",
                "gate_probability": 0.8,
                "noise_injection": 0.0
            },
            "Anime/Illustration": {
                "shunt_types": ["clip_l", "clip_g"],
                "shunt_names": ["t5-vit-l-14-dual_shunt_summarize.safetensors",
                                "t5-flan-vit-bigG-14-dual_shunt_summarize.safetensors"],
                "t5_model": "google/flan-t5-base",
                "gate_probability": 0.95,
                "noise_injection": 0.02
            }
        }

        config = mode_configs[mode]
        config["strength"] = strength_values[strength]
        config["prompt"] = prompt

        logger.info(f"Simple Shunt Setup: {mode} mode with {strength} strength")

        return (config,)


class EasyShunt:
    """Simple one-node shunt application - just plug and play!"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "conditioning": ("CONDITIONING", {}),
                "shunt_config": ("SHUNT_CONFIG", {}),
            },
            "optional": {
                "custom_strength": ("FLOAT", {
                    "default": -1.0,
                    "min": -1.0,
                    "max": 2.0,
                    "step": 0.1,
                    "tooltip": "Override config strength. -1 = use config default"
                }),
            }
        }

    RETURN_TYPES = ("CONDITIONING",)
    RETURN_NAMES = ("shunted_conditioning",)
    FUNCTION = "apply_shunt"
    CATEGORY = "adapter/simple"

    def apply_shunt(self, conditioning, shunt_config, custom_strength=-1.0):
        """Apply shunt with minimal configuration"""
        from .model_manager import get_model_manager
        from .configs import T5_SHUNT_REPOS

        manager = get_model_manager()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load T5 model
        t5_result = manager.load_t5(
            name=shunt_config["t5_model"],
            model_path=shunt_config["t5_model"]
        )

        if not t5_result:
            raise RuntimeError("Failed to load T5 model")

        t5_model, tokenizer = t5_result

        # Tokenize prompt
        tokens = tokenizer(
            shunt_config["prompt"],
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=77
        )

        input_ids = tokens["input_ids"].to(device)
        attention_mask = tokens["attention_mask"].to(device)

        # Get T5 embeddings
        with torch.no_grad():
            t5_embeddings = t5_model.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask
            ).last_hidden_state

        # Load adapters
        adapters = []
        for shunt_type, shunt_name in zip(shunt_config["shunt_types"], shunt_config["shunt_names"]):
            config_entry = T5_SHUNT_REPOS.get(shunt_type)
            if not config_entry:
                continue

            adapter = manager.load_shunt_adapter(
                name=f"{shunt_type}_{shunt_name}",
                config=config_entry["config"],
                repo=config_entry["repo"],
                filename=shunt_name
            )

            if adapter:
                adapters.append({
                    "adapter": adapter,
                    "config": config_entry["config"]
                })

        # Apply adapters
        strength = custom_strength if custom_strength >= 0 else shunt_config["strength"]
        adapted_conditioning = []

        for cond_tensor, cond_meta in conditioning:
            cond_tensor = cond_tensor.clone().to(device)

            for adapter_info in adapters:
                adapter_model = adapter_info["adapter"].to(device)
                clip_dim = adapter_info["config"]["clip"]["hidden_size"]

                # Determine slice
                if clip_dim == 768:  # CLIP-L
                    clip_slice = cond_tensor[:, :, :768]
                    slice_range = (0, 768)
                elif clip_dim == 1280:  # CLIP-G
                    clip_slice = cond_tensor[:, :, 768:2048]
                    slice_range = (768, 2048)
                else:
                    continue

                # Apply adapter
                outputs = adapter_model(t5_embeddings.float(), clip_slice.float())
                _, delta_mean, log_sigma, _, _, _, _, gate = outputs

                # Simple application
                delta = delta_mean * strength * gate * shunt_config["gate_probability"]

                # Add noise if configured
                if shunt_config["noise_injection"] > 0:
                    sigma = torch.exp(log_sigma)
                    noise = torch.randn_like(clip_slice) * sigma * shunt_config["noise_injection"]
                    delta = delta + noise

                # Resize if needed
                if delta.shape[1] != clip_slice.shape[1]:
                    delta = torch.nn.functional.interpolate(
                        delta.transpose(1, 2),
                        size=clip_slice.size(1),
                        mode="nearest"
                    ).transpose(1, 2)

                # Apply to conditioning
                cond_tensor[:, :, slice_range[0]:slice_range[1]] += delta

            adapted_conditioning.append([cond_tensor, cond_meta])

        return (adapted_conditioning,)


import numpy as np
import matplotlib.pyplot as plt
import torch
from PIL import Image

class QuickShuntPreview:
    """Preview the effect of shunting with a simple comparison"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "original_conditioning": ("CONDITIONING", {}),
                "shunted_conditioning": ("CONDITIONING", {}),
                "preview_type": (["heatmap", "difference", "magnitude"], {}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("preview",)
    FUNCTION = "preview"
    CATEGORY = "adapter/simple"


    def preview(
            self,
            original_conditioning,
            shunted_conditioning,
            preview_type="difference",
            debug=False
    ):
        """
        ComfyUI-compatible preview: returns a single torch image tensor as [ [C,H,W] ] normalized to [0,1 ].
        Guaranteed to produce a color image for the ComfyUI image viewer, matching your save pipeline.
        """

        # 1. Flatten all batch/grid dims to 2D [N, D]
        def flatten(x):
            while isinstance(x, (list, tuple)):
                x = x[0]
            if hasattr(x, "detach"):
                x = x.detach().cpu().float().numpy()
            elif isinstance(x, np.ndarray):
                pass
            else:
                raise TypeError("Input is not a tensor or ndarray.")
            return x.reshape(-1, x.shape[-1])

        orig = flatten(original_conditioning)
        shnt = flatten(shunted_conditioning)

        # 2. Compute matrix for preview
        if preview_type == "difference":
            mat = shnt - orig
            cmap = "jet"  # or your preferred colorful map
        elif preview_type == "heatmap":
            mat = np.abs(shnt - orig)
            cmap = "hot"
        elif preview_type == "magnitude":
            mag = np.linalg.norm(shnt - orig, axis=1, keepdims=True)
            mat = np.tile(mag, (1, min(orig.shape[1], 32)))
            cmap = "plasma"
        elif preview_type == "original":
            mat = orig
            cmap = "viridis"
        else:
            raise ValueError(f"Unknown preview_type: {preview_type}")

        mat = np.nan_to_num(mat)
        if np.ptp(mat) > 0:
            mat = (mat - mat.min()) / (mat.max() - mat.min())
        else:
            mat = mat * 0

        # 3. Render matplotlib to RGBA and convert to RGB PIL image
        fig, ax = plt.subplots(figsize=(8, 4), dpi=100)
        im = ax.imshow(mat, aspect="auto", cmap=cmap, origin="upper")
        ax.axis("off")
        fig.tight_layout(pad=0)
        fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
        plt.draw()
        w, h = fig.canvas.get_width_height()
        buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape(h, w, 4)
        plt.close(fig)

        # Convert to PIL and ensure RGB format (completely decouples from any channel issues)
        img_pil = Image.fromarray(buf[..., :3], mode="RGB")

        # [Optional: save for debugging]
        # img_pil.save("preview_debug.png")

        # Convert to numpy array as HWC for PIL, then to CHW for torch/ComfyUI
        rgb = np.array(img_pil)  # [H, W, 3]
        img = torch.from_numpy(rgb).permute(2, 0, 1).contiguous().float() / 255.0  # [3, H, W]

        return [img]  # SINGLE image in a list as ComfyUI expects


class ShuntStrengthTest:
    """Quick test different shunt strengths"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "conditioning": ("CONDITIONING", {}),
                "shunt_config": ("SHUNT_CONFIG", {}),
                "test_strengths": ("STRING", {
                    "default": "0.0, 0.3, 0.5, 0.7, 1.0",
                    "tooltip": "Comma-separated strength values to test"
                }),
            }
        }

    RETURN_TYPES = ("CONDITIONING",)
    RETURN_NAMES = ("test_batch",)
    FUNCTION = "test_strengths"
    CATEGORY = "adapter/simple"
    OUTPUT_IS_LIST = (True,)

    def test_strengths(self, conditioning, shunt_config, test_strengths):
        """Create a batch of conditioning with different strengths"""

        # Parse strengths
        strengths = [float(s.strip()) for s in test_strengths.split(",")]

        # Get the EasyShunt node
        easy_shunt = EasyShunt()

        # Apply shunt at each strength
        results = []
        for strength in strengths:
            shunted = easy_shunt.apply_shunt(
                conditioning=conditioning,
                shunt_config=shunt_config,
                custom_strength=strength
            )[0]

            # Add strength info to metadata
            for cond, meta in shunted:
                meta["shunt_strength"] = strength

            results.append(shunted)

        logger.info(f"Created {len(results)} test conditions with strengths: {strengths}")

        return (results,)


import numpy as np
import matplotlib.pyplot as plt
import torch
from PIL import Image

# 2D meshgrid with visible per-pixel color
#n, d = 77, 128
#xv, yv = np.meshgrid(np.linspace(0, 1, d), np.linspace(0, 1, n))
#mat = (np.sin(xv * 8 * np.pi) + np.cos(yv * 8 * np.pi)) / 2  # strong color
#
#mat = (mat - mat.min()) / (mat.max() - mat.min())
#
#fig, ax = plt.subplots(figsize=(8, 4), dpi=100)
#im = ax.imshow(mat, aspect="auto", cmap="jet", origin="upper")
#ax.axis("off")
#plt.draw()
#w, h = fig.canvas.get_width_height()
#buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape(h, w, 4)
#rgb = buf[..., :3]
#plt.close(fig)
#
## Save the RGB buffer as an image using PIL
#img_out = Image.fromarray(rgb)
#img_out.save("color_test_output.png")
#
## Convert to tensor for ComfyUI [C, H, W]
#img_tensor = torch.from_numpy(rgb).permute(2, 0, 1).contiguous().float() / 255.0
#print(img_tensor.shape, img_tensor.min().item(), img_tensor.max().item())
#
#print("Image saved as color_test_output.png")
#

import numpy as np
import matplotlib.pyplot as plt
import torch
from PIL import Image
import tempfile
import os

import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import logging

logger = logging.getLogger(__name__)


class SuperiorConditioningPreview:
    """Enhanced conditioning preview with multiple visualization modes"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "original_conditioning": ("CONDITIONING", {}),
                "shunted_conditioning": ("CONDITIONING", {}),
                "preview_type": ([
                                     "difference",
                                     "heatmap",
                                     "magnitude",
                                     "side_by_side",
                                     "histogram",
                                     "clip_analysis"
                                 ], {}),
                "colormap": (["RdBu", "viridis", "hot", "plasma", "jet", "coolwarm", "seismic"], {"default": "RdBu"}),
                "sensitivity": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 10.0, "step": 0.1}),
                "clip_percentile": ("FLOAT", {"default": 99.0, "min": 90.0, "max": 100.0, "step": 0.5}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("preview",)
    FUNCTION = "preview"
    CATEGORY = "adapter/visualization"

    def preview(self, original_conditioning, shunted_conditioning,
                preview_type="difference", colormap="RdBu",
                sensitivity=1.0, clip_percentile=99.0):

        def extract_tensor(x):
            """Recursively extract tensor from nested structures"""
            while isinstance(x, (list, tuple)):
                if len(x) == 0:
                    raise ValueError("Empty conditioning")
                x = x[0]

            if hasattr(x, "detach"):
                return x.detach().cpu().float().numpy()
            elif isinstance(x, np.ndarray):
                return x.astype(np.float32)
            else:
                raise TypeError(f"Cannot extract tensor from type: {type(x)}")

        # Extract tensors
        orig = extract_tensor(original_conditioning)
        shnt = extract_tensor(shunted_conditioning)

        # Reshape to 2D for visualization [tokens, features]
        orig_flat = orig.reshape(-1, orig.shape[-1])
        shnt_flat = shnt.reshape(-1, shnt.shape[-1])

        # Compute statistics
        diff = shnt_flat - orig_flat
        abs_diff = np.abs(diff)

        # Enhanced statistics
        stats = {
            "mean_change": np.mean(diff),
            "std_change": np.std(diff),
            "max_increase": np.max(diff),
            "max_decrease": np.min(diff),
            "percent_changed": np.mean(abs_diff > 1e-6) * 100,
            "rms_change": np.sqrt(np.mean(diff ** 2)),
        }

        # Create figure based on preview type
        if preview_type == "side_by_side":
            fig, axes = plt.subplots(1, 3, figsize=(15, 5), dpi=100)

            # Original
            im0 = axes[0].imshow(orig_flat[:, :256].T, aspect='auto', cmap='viridis')
            axes[0].set_title("Original Conditioning", fontsize=12, fontweight='bold')
            axes[0].set_xlabel("Token Position")
            axes[0].set_ylabel("Feature Dimension")

            # Shunted
            im1 = axes[1].imshow(shnt_flat[:, :256].T, aspect='auto', cmap='viridis')
            axes[1].set_title("Shunted Conditioning", fontsize=12, fontweight='bold')
            axes[1].set_xlabel("Token Position")

            # Difference
            vmax = np.percentile(abs_diff, clip_percentile) * sensitivity
            norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
            im2 = axes[2].imshow(diff[:, :256].T, aspect='auto', cmap='RdBu_r', norm=norm)
            axes[2].set_title("Difference (Red=↑, Blue=↓)", fontsize=12, fontweight='bold')
            axes[2].set_xlabel("Token Position")

            # Add colorbars
            plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
            plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
            plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

        elif preview_type == "histogram":
            fig, axes = plt.subplots(2, 2, figsize=(12, 10), dpi=100)

            # Change distribution
            axes[0, 0].hist(diff.flatten(), bins=100, alpha=0.7, color='blue', edgecolor='black')
            axes[0, 0].axvline(0, color='red', linestyle='--', linewidth=2)
            axes[0, 0].set_title("Distribution of Changes", fontsize=12, fontweight='bold')
            axes[0, 0].set_xlabel("Change Value")
            axes[0, 0].set_ylabel("Frequency")
            axes[0, 0].set_yscale('log')

            # Magnitude distribution
            axes[0, 1].hist(abs_diff.flatten(), bins=100, alpha=0.7, color='green', edgecolor='black')
            axes[0, 1].set_title("Magnitude Distribution", fontsize=12, fontweight='bold')
            axes[0, 1].set_xlabel("Absolute Change")
            axes[0, 1].set_ylabel("Frequency")
            axes[0, 1].set_yscale('log')

            # Per-token magnitude
            token_magnitude = np.linalg.norm(diff, axis=1)
            axes[1, 0].plot(token_magnitude, linewidth=2, color='purple')
            axes[1, 0].fill_between(range(len(token_magnitude)), token_magnitude, alpha=0.3, color='purple')
            axes[1, 0].set_title("Per-Token Change Magnitude", fontsize=12, fontweight='bold')
            axes[1, 0].set_xlabel("Token Position")
            axes[1, 0].set_ylabel("L2 Magnitude")

            # Statistics text
            axes[1, 1].axis('off')
            stats_text = f"Statistics:\n\n"
            stats_text += f"Mean Change: {stats['mean_change']:.6f}\n"
            stats_text += f"Std Change: {stats['std_change']:.6f}\n"
            stats_text += f"Max Increase: {stats['max_increase']:.6f}\n"
            stats_text += f"Max Decrease: {stats['max_decrease']:.6f}\n"
            stats_text += f"Changed Features: {stats['percent_changed']:.2f}%\n"
            stats_text += f"RMS Change: {stats['rms_change']:.6f}"
            axes[1, 1].text(0.1, 0.5, stats_text, fontsize=12, family='monospace',
                            verticalalignment='center', transform=axes[1, 1].transAxes)

        elif preview_type == "clip_analysis":
            fig, axes = plt.subplots(2, 2, figsize=(12, 10), dpi=100)

            # Separate CLIP-L and CLIP-G if possible
            if orig_flat.shape[1] >= 2048:
                clip_l_diff = diff[:, :768]
                clip_g_diff = diff[:, 768:2048]

                # CLIP-L heatmap
                vmax = np.percentile(np.abs(clip_l_diff), clip_percentile) * sensitivity
                im0 = axes[0, 0].imshow(clip_l_diff.T, aspect='auto', cmap=colormap,
                                        vmin=-vmax, vmax=vmax)
                axes[0, 0].set_title("CLIP-L Changes", fontsize=12, fontweight='bold')
                axes[0, 0].set_xlabel("Token Position")
                axes[0, 0].set_ylabel("Feature Dimension")
                plt.colorbar(im0, ax=axes[0, 0])

                # CLIP-G heatmap
                vmax = np.percentile(np.abs(clip_g_diff), clip_percentile) * sensitivity
                im1 = axes[0, 1].imshow(clip_g_diff.T, aspect='auto', cmap=colormap,
                                        vmin=-vmax, vmax=vmax)
                axes[0, 1].set_title("CLIP-G Changes", fontsize=12, fontweight='bold')
                axes[0, 1].set_xlabel("Token Position")
                plt.colorbar(im1, ax=axes[0, 1])

                # Magnitude comparison
                clip_l_mag = np.linalg.norm(clip_l_diff, axis=1)
                clip_g_mag = np.linalg.norm(clip_g_diff, axis=1)

                axes[1, 0].plot(clip_l_mag, label='CLIP-L', linewidth=2)
                axes[1, 0].plot(clip_g_mag, label='CLIP-G', linewidth=2)
                axes[1, 0].set_title("Per-Token Magnitude", fontsize=12, fontweight='bold')
                axes[1, 0].set_xlabel("Token Position")
                axes[1, 0].set_ylabel("L2 Magnitude")
                axes[1, 0].legend()
                axes[1, 0].grid(True, alpha=0.3)

                # Statistics comparison
                axes[1, 1].axis('off')
                stats_text = "CLIP-L vs CLIP-G Statistics:\n\n"
                stats_text += f"CLIP-L RMS: {np.sqrt(np.mean(clip_l_diff ** 2)):.6f}\n"
                stats_text += f"CLIP-G RMS: {np.sqrt(np.mean(clip_g_diff ** 2)):.6f}\n"
                stats_text += f"CLIP-L Changed: {np.mean(np.abs(clip_l_diff) > 1e-6) * 100:.2f}%\n"
                stats_text += f"CLIP-G Changed: {np.mean(np.abs(clip_g_diff) > 1e-6) * 100:.2f}%\n"
                axes[1, 1].text(0.1, 0.5, stats_text, fontsize=12, family='monospace',
                                verticalalignment='center', transform=axes[1, 1].transAxes)
            else:
                # Fallback for non-SDXL
                self._create_single_view(axes, diff, colormap, sensitivity, clip_percentile)

        else:
            # Single view for other types
            fig, ax = plt.subplots(figsize=(10, 6), dpi=100)

            if preview_type == "difference":
                mat = diff[:, :min(diff.shape[1], 512)]
                vmax = np.percentile(np.abs(mat), clip_percentile) * sensitivity
                norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
                im = ax.imshow(mat.T, aspect='auto', cmap=colormap, norm=norm)
                title = "Conditioning Difference"

            elif preview_type == "heatmap":
                mat = abs_diff[:, :min(abs_diff.shape[1], 512)]
                vmax = np.percentile(mat, clip_percentile) * sensitivity
                im = ax.imshow(mat.T, aspect='auto', cmap=colormap, vmin=0, vmax=vmax)
                title = "Absolute Change Heatmap"

            elif preview_type == "magnitude":
                mag = np.linalg.norm(diff, axis=1, keepdims=True)
                mat = np.tile(mag, (1, min(64, diff.shape[1])))
                vmax = np.percentile(mag, clip_percentile) * sensitivity
                im = ax.imshow(mat.T, aspect='auto', cmap=colormap, vmin=0, vmax=vmax)
                title = "Per-Token Change Magnitude"

            ax.set_title(title, fontsize=14, fontweight='bold')
            ax.set_xlabel("Token Position")
            ax.set_ylabel("Feature/Magnitude")
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

            # Add statistics
            stats_str = f"μ={stats['mean_change']:.5f}, σ={stats['std_change']:.5f}, %changed={stats['percent_changed']:.1f}%"
            ax.text(0.02, 0.98, stats_str, transform=ax.transAxes,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
                    verticalalignment='top', fontsize=10)

        plt.tight_layout()

        # Render to array using buffer_rgba
        plt.draw()
        w, h = fig.canvas.get_width_height()
        buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        buf = buf.reshape(h, w, 4)
        plt.close(fig)

        # Convert to RGB and normalize
        rgb = buf[:, :, :3].astype(np.float32) / 255.0

        # Convert to ComfyUI format [B, H, W, C]
        img_tensor = torch.from_numpy(rgb).unsqueeze(0)

        return (img_tensor,)

    def _create_single_view(self, axes, diff, colormap, sensitivity, clip_percentile):
        """Helper for creating single conditioning view"""
        for ax in axes.flat:
            ax.axis('off')

        axes[0, 0].axis('on')
        vmax = np.percentile(np.abs(diff), clip_percentile) * sensitivity
        im = axes[0, 0].imshow(diff.T, aspect='auto', cmap=colormap, vmin=-vmax, vmax=vmax)
        axes[0, 0].set_title("Conditioning Changes", fontsize=12, fontweight='bold')
        plt.colorbar(im, ax=axes[0, 0])
