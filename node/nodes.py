import logging

from ..model.configs import ShuntUtil

logger = logging.getLogger(__name__)

import hashlib
from ..model.model_manager import get_model_manager
from ..model.configs import ENCODER_CONFIGS, ShuntData, EncoderData


class EncoderLoader:
    """
    Loads T5 or BERT encoder model and prepares tokenized context window.
    Returns a complete CONDITIONING_PIPE config block for downstream interpolation, masking, and scheduling.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_name": (
                    list(ENCODER_CONFIGS.keys()),
                    {"default": "bert-base-uncased"}
                ),
                "local_path": ("STRING", {"default": ""}),
                "context_window": ("STRING", {
                    "default": "a photo of a robot.",
                    "multiline": True
                }),
                "use_context_window": ("BOOLEAN", {"default": True}),
                "context_window_size": ("INT", {"default": 1024, "min": 77, "max": 8192}),
                "sliding_window_size": ("INT", {"default": 77, "min": 1, "max": 2048}),
                "sliding_window_stride": ("INT", {"default": 33, "min": 1, "max": 2048}),
                "max_length": ("INT", {"default": 512, "min": 1, "max": 8192}),
                "folding": ([
                    "zeus", "helios", "surge", "surge-fold", "fold", "interpolate",
                    "collapse", "zipper", "concat-flatten", "cascade", "ripple"
                ], {"default": "surge-fold"}),
                "folding_scheduler": ([
                    "none", "tau", "top_k", "top_20k", "top_50k",
                    "cosine", "cascade", "cos", "sine",
                    "shockwave", "pulse", "wave"
                ], {"default": "none"}),
                "pos_embedding": (["none", "cos", "sine", "cosine_sine_product"], {"default": "none"}),
                "padding": (["max_length", "longest", "do_not_pad"], {"default": "max_length"}),
                "dtype": (["default", "float32", "float16", "bfloat16"], {"default": "default"}),
                "device": (["cpu", "cuda", "mps"], {"default": "cuda" if torch.cuda.is_available() else "cpu"}),
                "trust_remote_code": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Allow execution of remote model code. Use only with trusted sources."
                }),
            }
        }


    RETURN_TYPES = ("ENCODER_PIPE",)
    RETURN_NAMES = ("encoder_pipe",)
    FUNCTION = "load"
    CATEGORY = "adapter/testing"
    DEPRECATED = False

    def load(self,
             model_name,
             local_path,
             context_window,
             use_context_window,
             context_window_size,
             sliding_window_size,
             sliding_window_stride,
             max_length,
             folding,
             folding_scheduler,
             pos_embedding,
             padding,
             dtype,
             device,
             trust_remote_code):

        model_manager = get_model_manager()
        device_obj = torch.device(device)

        # Determine source
        model_config = ENCODER_CONFIGS.get(model_name, {})
        model_type = model_config.get("type", "unknown")
        model_source = local_path or model_config.get("repo_name", model_name)
        model_id = f"{model_type}_{model_name}_{hashlib.sha1(model_source.encode()).hexdigest()[:10]}"

        dtype = torch.get_default_dtype() if dtype == "default" else {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16
        }[dtype]

        # Load model/tokenizer
        result = model_manager.load_encoder_model(
            model_type=model_type,
            model_id=model_id,
            model_name_or_path=model_source,
            device=device_obj,
            dtype=dtype,
            force_reload=False,
            trust_remote_code=trust_remote_code
        )
        if not result:
            raise RuntimeError(f"Failed to load encoder model: {model_name}")
        model, tokenizer = result

        # Build config dictionary for downstream control
        config_dict = {
            "model_id": model_id,
            "model_type": model_type,
            "model_name": model_name,
            "source": model_source,
            "device": str(device),
            "trust_remote_code": trust_remote_code,
            "context_window": context_window,
            "config": {
                "use_context_window": use_context_window,
                "context_window_size": context_window_size,
                "sliding_window_size": sliding_window_size,
                "sliding_window_stride": sliding_window_stride,
                "max_length": max_length,
                "folding": folding,
                "folding_scheduler": folding_scheduler,
                "pos_embedding": pos_embedding,
                "padding": padding
            }
        }

        return ({ # this is the encoder_pipe paradigm
            "model": model,
            "tokenizer": tokenizer,
            "config": config_dict,
        },)

class SimpleEncoderLoader:
    """
    Loads a simple encoder model and prepares tokenized context window.
    Returns a complete CONDITIONING_PIPE config block for downstream interpolation, masking, and scheduling.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_name": (list(ENCODER_CONFIGS.keys()), {"default": "beatrix-bert-2048"}),
                "device": (["cpu", "cuda", "mps"], {"default": "cuda" if torch.cuda.is_available() else "cpu"}),
                "trust_remote_code": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Allow execution of remote model code. Use only with trusted sources."
                }),
            }
        }

    RETURN_TYPES = ("ENCODER_PIPE",)
    RETURN_NAMES = ("encoder_pipe",)
    FUNCTION = "load"
    CATEGORY = "adapter/testing"
    DEPRECATED = False

    def load(self, model_name, device, trust_remote_code):
        """Load an encoder with defaulted configuration to allow simplistic loading process.
            This still loads all the more advanced features from the advanced system, but it's all set to default.
        """

        model_manager = get_model_manager()
        device_obj = torch.device(device)
        # Determine source
        model_config:Optional[EncoderData] = ShuntUtil.get_encoder_by_model_name(model_name)
        if not model_config:
            raise ValueError(f"No configuration found for model '{model_name}'.")

        """
        class EncoderData:
            def __init__(self,
                         name: str,
                         file: str,
                         repo: str,
                         config: dict,
                         type: str = "t5"):
                self.name = name
                self.file = file
                self.repo = repo
                self.config = config
                self.type = type

        """

        # load using the model managers load_encoder_model method
        # using the correct model archetype and paradigm we
        model_type = model_config.type
        model_source = model_config.repo
        model_id = f"{model_type}_{model_name}_{hashlib.sha1(model_source.encode()).hexdigest()[:10]}"
        result = model_manager.load_encoder_model(
            model_type=model_type,
            model_id=model_id,
            model_name_or_path=model_source,
            device=device_obj,
            dtype=torch.float32,  # Default dtype
            force_reload=False,
            trust_remote_code=trust_remote_code  # Default to not trusting remote code
        )
        if not result:
            raise RuntimeError(f"Failed to load encoder model: {model_name}")
        model, tokenizer = result
        # Build config dictionary for downstream control
        config_dict = {
            "model_id": model_id,
            "model_type": model_type,
            "model_name": model_name,
            "source": model_source,
            "device": str(device),
            "trust_remote_code": False,  # Default to not trusting remote code
            "config": {
                # attempt to seek the config details from the model config, then default if not found
                "max_length": model_config.config.get("max_length", 77),
                "padding": model_config.config.get("padding", "max_length"),
                "sliding_window_size": model_config.config.get("sliding_window_size", 77),
                "sliding_window_stride": model_config.config.get("sliding_window_stride", 33),
                "use_context_window": model_config.config.get("use_context_window", True),
                "context_window_size": model_config.config.get("context_window_size", 77),
                "folding": model_config.config.get("folding", "surge-fold"),
                "folding_scheduler": model_config.config.get("folding_scheduler", "none"),
                "pos_embedding": model_config.config.get("pos_embedding", "none"),
                "min_slices": model_config.config.get("min_slices", 1),
                "max_slices": model_config.config.get("max_slices", 10),
                "truncate_option": model_config.config.get("truncate_option", "fold"),
                "sliding_window": model_config.config.get("sliding_window", True),
            }
        }
        return ({
            "model": model,
            "tokenizer": tokenizer,
            "config": config_dict,
        },)


class T5LoaderTest:
    """
    Loads T5 encoder-decoder model and prepares tokenized context for adapters.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_name": (list(ENCODER_CONFIGS.keys()), {"default": "google/flan-t5-base"}),
                "local_path": ("STRING", {"default": "", "tooltip": "Local path override. If empty, use HuggingFace."}),
                "context_window": ("STRING", {"default": "a photo of a robot.", "multiline": True}),
                "use_context_window": ("BOOLEAN", {"default": True}),
                "sliding_window_size": ("INT", {"default": 512, "min": 1, "max": 2048}),
                "sliding_window_stride": ("INT", {"default": 256, "min": 1, "max": 2048}),
                "max_length": ("INT", {"default": 77, "min": 1, "max": 512}),
                "padding": (["max_length", "longest", "do_not_pad"], {"default": "max_length"}),
                "max_slices": ("INT", {"default": 10, "min": 1, "max": 100}),
                "min_slices": ("INT", {"default": 1, "min": 1, "max": 100}),
                "truncate_option": (["fold", "interpolate", "collapse", "zipper"], {"default": "fold"}),
                "device": (["cpu", "cuda", "mps"], {"default": "cuda" if torch.cuda.is_available() else "cpu"})
            }
        }

    RETURN_TYPES = ("ENCODER_PIPE",)
    RETURN_NAMES = ("encoder_pipe",)
    FUNCTION = "load"
    CATEGORY = "adapter/testing"
    DEPRECATED = True

    def load(self, model_name, local_path, context_window, use_context_window,
             sliding_window_size, sliding_window_stride, max_length, padding,
             max_slices, min_slices, truncate_option, device):
        """Load the T5 model and tokenizer, and encode a sample context window."""

        # Get model manager
        model_manager = get_model_manager()

        # Determine model source
        model_config = ENCODER_CONFIGS.get(model_name, {})
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

class LoadShuntSimple:
    """Load a shunt adapter with simplified model management."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "shunt_name": ("STRING", {
                    "default": "",
                    "tooltip": "Name of the shunt adapter to load."
                }),
            }
        }

    RETURN_TYPES = ("ADAPTER_PIPE",)
    RETURN_NAMES = ("adapter_pipe",)
    FUNCTION = "load_adapter"
    CATEGORY = "loader/shunt"

    def load_adapter(self, shunt_name):
        """Load adapter using the model manager."""
        model_manager = get_model_manager()
        device_obj = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        shunt_data: Optional[ShuntData] = ShuntUtil.get_shunt_by_name(shunt_name)

        if shunt_data:
            shunt_name = shunt_data.file
            shunt_type = shunt_data.shunt_type_name
            adapter_id = f"shunt_{shunt_type}_{shunt_name}"

            logging.info(f"Loading adapter '{shunt_name}' of type '{shunt_type}' with Name '{adapter_id}'")
            logging.info(f"Config: {shunt_data.config}")

            # Load adapter
            adapter = model_manager.load_shunt_adapter(
                adapter_id=adapter_id,
                config=shunt_data.config,
                repo_id=shunt_data.repo,
                filename=shunt_name,
                device=device_obj,
                dtype=torch.float32,
                force_reload=False
            )

            if not adapter:
                raise RuntimeError(f"Failed to load adapter '{shunt_name}' of type '{shunt_type}'")

            logger.info(f"Successfully loaded adapter: {adapter_id}")

            return ([{
                "adapter": adapter,
                "adapter_id": adapter_id,
                "config": shunt_data.config
            }],)


class LoadAdapterShunt:
    """Load a shunt adapter with improved model management."""

    @classmethod
    def INPUT_TYPES(cls):
        # Get the names from the ShuntsUtil
        shunt_list = ShuntUtil.get_shunt_names()

        return {
            "required": {
                "adapter_path": ("STRING", {
                    "default": "",
                    "tooltip": "Full path to the adapter .safetensors or .pt file."
                }),
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

    RETURN_TYPES = ("ADAPTER_PIPE",)
    RETURN_NAMES = ("adapter_pipe",)
    FUNCTION = "load_adapter"
    CATEGORY = "adapter/shunt"

    def load_adapter(self, adapter_path, shunt_name, device):
        """Load adapter using the refactored model manager."""

        # Get model manager
        model_manager = get_model_manager()

        # Get configuration
        config_entry:Optional[ShuntData] = ShuntUtil.get_shunt_by_name(shunt_name)

        if not config_entry:
            raise ValueError(f"No configuration found for shunt '{shunt_name}'.")
        shunt_name = config_entry.file
        shunt_type = config_entry.shunt_type_name
        adapter_id = f"shunt_{shunt_type}_{shunt_name}"
        logger.info(f"Loading adapter '{shunt_name}' of type '{shunt_type}' with ID '{adapter_id}'")
        # Load adapter
        device_obj = torch.device(device)
        adapter = model_manager.load_shunt_adapter(
            adapter_id=adapter_id,
            config=config_entry.config,
            repo_id=config_entry.repo,
            filename=adapter_path or shunt_name,
            device=device_obj,
            dtype=torch.float32,  # Default dtype
            force_reload=False
        )
        if not adapter:
            raise RuntimeError(f"Failed to load adapter '{shunt_name}' of type '{shunt_type}'")
        logger.info(f"Successfully loaded adapter: {adapter_id}")
        return ([{
            "adapter": adapter,
            "adapter_id": adapter_id,
            "config": config_entry.config
        }],)


class EncodeClipShunted:
    # intercepts the non-encoded clip instead of the conditioning to prepare more complex streamlines with multiple different clips
    # takes in encoder pipelines, clip pipelines, adapter pipelines, and text pipelines.
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip": ("CLIP", {}),
                "adapter_pipe": ("ADAPTER_PIPE", {}),
                "encoder_pipe": ("ENCODER_PIPE", {}),
                "positive_prompt": ("STRING", {"default": "A photo of a robot.", "multiline": True}),
                "negative_prompt": ("STRING", {"default": "watermark, greyscale, monochrome", "multiline": True}),
            },
            "optional": {
                "conditioning_sampler": ("CONDITIONING_SAMPLER", {}),
                "encoder_config": ("ENCODER_CONFIG", {}),
                "encoder_schedule": ("ENCODER_SCHEDULE", {}),
                "clip_config": ("CLIP_CONFIG", {}),
                "adapter_config": ("ADAPTER_CONFIG", {}),
                "positive_prompt_schedule": ("PROMPT_SCHEDULE", {}),
                "negative_prompt_schedule": ("PROMPT_SCHEDULE", {}),
            }
        }

    RETURN_TYPES = ("CONDITIONING", "CONDITIONING")
    RETURN_NAMES = ("positive_conditionings", "negative_conditionings")

    FUNCTION = "encode"
    CATEGORY = "encode/clip_shunt"

    def encode(self,
               clip,
               adapter_pipe,
               encoder_pipe,
               positive_prompt,
               negative_prompt,
               conditioning_sampler=None,
               encoder_config=None,
               encoder_schedule=None,
               clip_config=None,
               adapter_config=None,
               positive_prompt_schedule=None,
               negative_prompt_schedule=None):
        """
        Encode the provided prompts using the specified CLIP and adapter pipelines.
        Returns positive and negative conditionings.
        """
        # stub for the encode function, requires multiple structures before application is possible.

        return (None, None,)#positive_conditioning, negative_conditioning)

class ShuntConfig:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "strength": ("FLOAT", {"default": 1.5, "min": -50.0, "max": 50.0, "step": 0.1}),
                "delta_mean": ("FLOAT", {"default": 0.5, "min": -10.0, "max": 10.0, "step": 0.1}),
                "delta_scale": ("FLOAT", {"default": 1.0, "min": -15.0, "max": 15.0, "step": 0.1}),
                "log_sigma": ("FLOAT", {"default": 0.5, "min": -10.0, "max": 10.0, "step": 0.1}),
                "sigma_scale": ("FLOAT", {"default": 0.1, "min": 0.0, "max": 15.0, "step": 0.1}),
                "gate_probability": ("FLOAT", {"default": 0.27, "min": 0.0, "max": 1.0, "step": 0.01}),
                "g_pred": ("FLOAT", {"default": 2.0, "min": -10.0, "max": 10.0}),
                "gpred_scale": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 20.0, "step": 0.01}),
                "noise_injection": ("FLOAT", {"default": 0.00, "min": 0.0, "max": 1.0, "step": 0.01}),
                "use_anchor": ("BOOLEAN", {"default": True}),
            },
        }

    RETURN_TYPES = ("ADAPTER_CONFIG",)
    RETURN_NAMES = ("config",)
    FUNCTION = "get_config"
    CATEGORY = "adapter/shunt"
    DEPRECATED = False
    def get_config(self, strength, delta_mean, delta_scale, log_sigma,
                     sigma_scale, gate_probability, g_pred, gpred_scale,
                        noise_injection, use_anchor):
        """Return a configuration dictionary for shunt adapters."""
        config = {
            "strength": strength,
            "delta_mean": delta_mean,
            "delta_scale": delta_scale,
            "log_sigma": log_sigma,
            "sigma_scale": sigma_scale,
            "gate_probability": gate_probability,
            "g_pred": g_pred,
            "gpred_scale": gpred_scale,
            "noise_injection": noise_injection,
            "use_anchor": use_anchor
        }
        return ([{
            "config": config,
            "config_id": "shunt_config",
            "description": "Configuration for shunt adapters",
            "version": 1.0
        }],)



class ShuntSampler:
    # meant to house the sampling logic for shunt adapters
    # default scheduler and sampler logic is used if none is provided
    pass


from ..utils.conditioning_shifter import ShiftConfig, ConditioningShifter

class ShuntConditioning:
    """Orchestrates the conditioning modification process"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "conditioning": ("CONDITIONING", {}),
                "encoder_pipe": ("ENCODER_PIPE", {}),
                "adapter_pipe": ("ADAPTER_PIPE", {}),
                "strength": ("FLOAT", {"default": 0.5, "min": -10.0, "max": 10.00, "step": 0.1}),
                "delta_mean": ("FLOAT", {"default": 0.3, "min": -2.0, "max": 2.00, "step": 0.1}),
                "delta_scale": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 5.00, "step": 0.1}),
                "sigma_scale": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 2.00, "step": 0.1}),
                "gate_probability": ("FLOAT", {"default": 0.50, "min": 0.0, "max": 1.00, "step": 0.01}),
                "gate_threshold": ("FLOAT", {"default": 0.27, "min": 0.0, "max": 1.00, "step": 0.01}),
                "noise_injection": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 0.99, "step": 0.01}),
                "use_anchor": ("BOOLEAN", {"default": True}),
                "pool_method": (["sequential", "weighted_average"], {"default": "sequential"}),
                # Top-K parameters
                "use_topk": ("BOOLEAN", {"default": True}),
                "topk_percentage": ("FLOAT", {"default": 50.0, "min": 1.0, "max": 100.0, "step": 1.0}),
                "tau_temperature": ("FLOAT", {"default": 5.0, "min": 0.1, "max": 10.0, "step": 0.1}),
                "topk_mode": (["attention", "gate", "combined", "tau_softmax"], {"default": "attention"}),
            },
            "optional": {
                "guidance_scale": ("FLOAT", {"default": 5.0, "min": 0.0, "max": 20.0, "step": 0.1}),
            }
        }

    RETURN_TYPES = ("CONDITIONING", "STRING")
    RETURN_NAMES = ("adapted_conditioning", "statistics")
    FUNCTION = "adapt_conditioning"
    CATEGORY = "adapter/shunt"

    def adapt_conditioning(self, conditioning, encoder_pipe, adapter_pipe,
                           strength, delta_mean, delta_scale, sigma_scale,
                           gate_probability, gate_threshold, noise_injection,
                           use_anchor, pool_method, use_topk, topk_percentage,
                           tau_temperature, topk_mode, guidance_scale=0.0):

        logger.info(
            f"Adapting conditioning with {len(adapter_pipe)} adapters (pool_method={pool_method}, topk={use_topk})")

        device = torch.device(
            encoder_pipe.get("config", {}).get("device", "cuda" if torch.cuda.is_available() else "cpu")
        )


        # Create unified config with top-k parameters
        config = ShiftConfig(
            strength=strength,
            delta_mean=delta_mean,
            delta_scale=delta_scale,
            sigma_scale=sigma_scale,
            gate_probability=gate_probability,
            gate_threshold=gate_threshold,
            noise_injection=noise_injection,
            use_anchor=use_anchor,
            pool_method=pool_method,
            use_topk=use_topk,
            topk_percentage=topk_percentage,
            tau_temperature=tau_temperature,
            topk_mode=topk_mode
        )

        # Get encoder embeddings (extracted to shifter)
        encoder_embeddings = ConditioningShifter.extract_encoder_embeddings(encoder_pipe, device)

        # Statistics tracking
        all_guidance_predictions = []
        all_modifications = []
        all_topk_stats = []

        # Process each conditioning tensor
        adapted_conditioning = []
        for cond_idx, (cond_tensor, cond_meta) in enumerate(conditioning):
            cond_tensor = cond_tensor.clone().to(device)

            # Collect adapter outputs by type for this conditioning
            outputs_by_type = {'clip_l': [], 'clip_g': []}

            # Run all adapters
            for adapter_info in adapter_pipe:
                try:
                    adapter_model = adapter_info["adapter"].to(device)
                    adapter_config = adapter_info["config"]

                    # Determine slice type and range
                    clip_dim = adapter_config.get("hidden_size", 768)
                    total_dim = cond_tensor.size(-1)

                    if clip_dim == 768:  # CLIP-L
                        slice_start, slice_end = 0, 768
                        adapter_type = "clip_l"
                    elif clip_dim == 1280:  # CLIP-G
                        slice_start = 768
                        slice_end = min(2048, total_dim)
                        adapter_type = "clip_g"
                    else:
                        logger.warning(f"Unknown CLIP dimension {clip_dim}")
                        continue

                    if slice_start >= total_dim:
                        continue

                    # Get slice and run adapter
                    clip_slice = cond_tensor[:, :, slice_start:slice_end]

                    output = ConditioningShifter.run_adapter(
                        adapter_model, encoder_embeddings, clip_slice,
                        guidance_scale, adapter_type, (slice_start, slice_end)
                    )

                    outputs_by_type[adapter_type].append(output)

                    # Collect guidance predictions
                    if output.g_pred is not None:
                        all_guidance_predictions.append(float(output.g_pred.mean().item()))

                    # Collect tau statistics if using top-k
                    if use_topk and output.tau is not None:
                        all_topk_stats.append({
                            'adapter_type': adapter_type,
                            'tau_mean': float(output.tau.mean().item()),
                            'tau_std': float(output.tau.std().item()) if output.tau.numel() > 1 else 0.0,
                        })

                except Exception as e:
                    logger.error(f"Error running adapter: {e}")
                    import traceback
                    traceback.print_exc()
                    continue

            # Apply modifications by type
            for adapter_type, outputs in outputs_by_type.items():
                if not outputs:
                    continue

                slice_start, slice_end = outputs[0].slice_range
                clip_slice = cond_tensor[:, :, slice_start:slice_end]

                # Apply modifications through shifter
                clip_modified = ConditioningShifter.apply_modifications(
                    clip_slice, outputs, config
                )

                # Update conditioning
                cond_tensor[:, :, slice_start:slice_end] = clip_modified.type_as(cond_tensor)

                # Track modifications
                all_modifications.append({
                    'adapter_type': adapter_type,
                    'num_adapters': len(outputs),
                    'slice_range': (slice_start, slice_end),
                    'mean_change': float((clip_modified - clip_slice).abs().mean().item())
                })

            adapted_conditioning.append([cond_tensor, cond_meta])

        if not adapted_conditioning:
            raise RuntimeError("No conditioning was successfully adapted")

        # Format statistics with top-k info
        stats_str = self._format_statistics(
            all_modifications, all_guidance_predictions,
            conditioning, adapted_conditioning,
            config, device, all_topk_stats
        )

        return (adapted_conditioning, stats_str)

    def _format_statistics(self, modifications, guidance_predictions,
                           orig_conditioning, adapted_conditioning,
                           config, device, topk_stats=None):
        """Format statistics output"""
        stats_str = f"Modification Statistics ({config.pool_method}):\n"
        stats_str += f"Total Modifications: {len(modifications)}\n"

        # Add top-k info if enabled
        if config.use_topk:
            stats_str += f"\nTop-K Selection:\n"
            stats_str += f"  Mode: {config.topk_mode}\n"
            stats_str += f"  Keep: {config.topk_percentage}% of tokens\n"
            stats_str += f"  Tau Temperature: {config.tau_temperature}\n"

            if topk_stats:
                avg_tau = np.mean([s['tau_mean'] for s in topk_stats])
                stats_str += f"  Average Tau: {avg_tau:.4f}\n"

        # Group by adapter type
        by_type = {}
        for mod in modifications:
            adapter_type = mod['adapter_type']
            if adapter_type not in by_type:
                by_type[adapter_type] = []
            by_type[adapter_type].append(mod)

        # Format by type
        for adapter_type, mods in by_type.items():
            stats_str += f"\n{adapter_type.upper()}:\n"
            total_adapters = sum(m['num_adapters'] for m in mods)
            avg_change = np.mean([m['mean_change'] for m in mods])
            stats_str += f"  Adapters Applied: {total_adapters}\n"
            stats_str += f"  Average Change: {avg_change:.6f}\n"

        # Guidance predictions
        if guidance_predictions:
            avg_guidance = np.mean(guidance_predictions)
            stats_str += f"\nGuidance Predictions:\n"
            stats_str += f"  Average: {avg_guidance:.2f}\n"

        # Overall change
        if orig_conditioning and adapted_conditioning:
            orig = orig_conditioning[0][0].to(device)
            adapted = adapted_conditioning[0][0].to(device)

            total_change = (adapted - orig).abs().mean().item()
            max_change = (adapted - orig).abs().max().item()
            pct_changed = ((adapted - orig).abs() > 1e-6).float().mean().item() * 100

            stats_str += f"\nOverall Change:\n"
            stats_str += f"  Mean: {total_change:.6f}\n"
            stats_str += f"  Max: {max_change:.6f}\n"
            stats_str += f"  Modified Tokens: {pct_changed:.1f}%\n"

        return stats_str

class StackShuntAdapters:
    """Stack multiple adapters together."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "adapter_1": ("ADAPTER_PIPE", {}),
                "adapter_2": ("ADAPTER_PIPE", {}),
            }
        }

    RETURN_NAMES = ("adapters",)
    RETURN_TYPES = ("ADAPTER_PIPE",)
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


import torch.nn.functional as F


class ShuntConditioningAdvanced:
    """Advanced adapter with full capabilities including timestep scheduling and noise injection"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "conditioning": ("CONDITIONING", {}),
                "encoder_pipe": ("ENCODER_PIPE", {}),
                "adapter_pipe": ("ADAPTER_PIPE", {}),
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

    def adapt_conditioning(self, conditioning, encoder_pipe, adapter_pipe, strength,
                           delta_mean, log_sigma, gate_probability, g_pred_scale,
                           noise_injection, use_anchor, timestep_start, timestep_end):

        device = torch.device(encoder_pipe["device"])

        # Get T5 embeddings
        with torch.no_grad():
            encoder_embeddings = encoder_pipe["model"].encoder(
                input_ids=encoder_pipe["input_ids"],
                attention_mask=encoder_pipe.get("attention_mask")
            ).last_hidden_state

        # Track guidance predictions
        all_g_preds = []

        # Process conditioning
        adapted_conditioning = []
        for cond_tensor, cond_meta in conditioning:
            cond_tensor = cond_tensor.clone().to(device)
            total_dim = cond_tensor.size(-1)

            # Apply each adapter
            for adapter_info in adapter_pipe:
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
                    adapter_model(encoder_embeddings.float(), clip_slice.float())

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
                "adapter_pipe": ("ADAPTER_PIPE", {}),
                "schedule_type": (["constant", "linear", "cosine", "exponential"], {}),
                "start_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.1}),
                "end_strength": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 2.0, "step": 0.1}),
                "timestep_start": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "timestep_end": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("ADAPTER_PIPE",)
    RETURN_NAMES = ("scheduled_adapter",)
    FUNCTION = "schedule"
    CATEGORY = "adapter/scheduling"

    def schedule(self, adapter_pipe, schedule_type, start_strength, end_strength,
                 timestep_start, timestep_end):
        scheduled_adapters = []

        for adapter_info in adapter_pipe:
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
                "adapter_pipe_1": ("ADAPTER_PIPE", {}),
                "adapter_pipe_2": ("ADAPTER_PIPE", {}),
                "weight_1": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.1}),
                "weight_2": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.1}),
                "merge_type": (["weighted_sum", "max", "min", "multiply"], {}),
            }
        }

    RETURN_TYPES = ("ADAPTER_PIPE",)
    RETURN_NAMES = ("merged_adapter_pipes",)
    FUNCTION = "merge"
    CATEGORY = "adapter/advanced"

    def merge(self, adapter_pipe_1, adapter_pipe_2, weight_1, weight_2, merge_type):
        # This would require modifying the adapter models themselves
        # For now, just return a list with both adapters and adjusted weights
        merged = []

        for a in adapter_pipe_1:
            a_copy = a.copy()
            a_copy["merge_weight"] = weight_1
            merged.append(a_copy)

        for a in adapter_pipe_2:
            a_copy = a.copy()
            a_copy["merge_weight"] = weight_2
            merged.append(a_copy)

        return (merged,)


from typing import Optional


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
        from ..model.model_manager import get_model_manager
        from ..model.configs import HARMONIC_SHUNT_REPOS

        manager = get_model_manager()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        #todo: obvious errors need fixing on EasyShunt
        # Load T5 model
        t5_result = manager.load_encoder_model(
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
            config_entry = HARMONIC_SHUNT_REPOS.get(shunt_type)
            if not config_entry:
                continue

            #todo: test and fix this
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

    DEPRECATED = True  # Marked as deprecated, use ComfyUI-compatible preview instead


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

from PIL import Image

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