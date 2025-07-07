"""
    Encoder module for ComfyUI, providing a flexible interface for various text, vision, and multimodal encoders.
    Author: AbstractPhil

    The primary workhorse is NeuralIO, which is a dataclass that houses the expectations for neural models to be used in a pipeline.
    If this paradigm of neural model handling requires additional functionality, it can be easily extended with
    simple torch lambda functions or additional methods.

    I built this as a flexible interface for various text, vision, and multimodal encoders.
    This is built as a solution to the messy and inconsistent handling of encoders in ComfyUI,
    this inconsistency leads to issues with model loading, configuration, and usage across different nodes.
    The entire spectrum of encoders is affected by this sloppy handling, including CLIP, T5, LLAMA, and other text-based.

    This is a centralized and unified interface for handling encoders, allowing for easier management in the
    torch-based ComfyUI environment. It's intended to provide a consistent way to load, configure, and use encoders
    without the need for multiple different rewritten classes and methods - essentially gating the entire encoder system
    from the average user who isn't allowed to touch the internals of ComfyUI in a distributed environment.

    The encoder module provides a series of convenient classes and methods to handle various encoders without altering
    core ComfyUI functionality or workflows.

    The entire structure is built around the concept of fixing problems.

    So if this module is not working as intended, or causes more problems than it solves, it will be deprecated.
"""
import comfy
import torch
import torch.nn as nn
import uuid

from comfy.model_management import intermediate_device
from dataclasses import dataclass
from typing import Optional, Callable, Union, Dict, List

from comfy import supported_models

from ..abs_sd.sd import CLIP # we will be using a modified CLIP pipeline for the encoder, so we import it here


class NeuralIO(nn.Module):
    # houses the target expectations for a transformers-based neural model to be used in a pipeline
    identifier: str = ""                               # required target identifier for a tokenizer, e.g. "clip-vit-large-patch14"
    types: str = ""                                    # required target type for a tokenizer, e.g. "clip", "text", etc
    config: dict = None                                # optional configuration for the tokenizer, e.g. {"clip": {"vision_tower": "clip-vit-large-patch14"}}
    callback_hooks: dict[Callable, Callable] = None    # hooks attached for callbacks based on met conditions
    input_expectations: dict = None                    # optional list of expected tokenizers, e.g. ["clip-vit-large-patch14", "t5-xxl"]
    output_expectations: dict = None                   # optional list of expected outputs, e.g. ["clip", "text", shape=(1, 768), "clip-vit-large-patch14", "t5-xxl"]


class EncoderWrapper(nn.Module):
    """Neural network module wrapper for managing encoders and tokenizers."""

    def __init__(
            self,
            identifier: str = "",
            expectations: Optional[Dict[str, NeuralIO]] = None,
            encoders: Optional[Dict[str, NeuralIO]] = None,
            tokenizers: Optional[Dict[str, NeuralIO]] = None,
            state_dict: Optional[dict] = None,
            config: Optional[dict] = None,
            patcher: Optional[object] = None,
            device: Union[str, torch.device] = "cpu",
            metadata: Optional[dict] = None
    ):
        super().__init__()

        self.identifier = identifier
        self.expectations = expectations or {}
        self.encoders = nn.ModuleDict(encoders or {})  # Use ModuleDict for proper registration
        self.tokenizers = tokenizers or {}  # Tokenizers might not be nn.Modules
        self.state_dict_ref = state_dict  # Renamed to avoid conflict with nn.Module.state_dict()
        self.config = config or {}
        self.patcher = patcher
        self.device = torch.device(device) if isinstance(device, str) else device
        self.metadata = metadata or {}

        # Validate that at least one encoder is provided
        if not self.encoders:
            raise ValueError(f"EncoderWrapper '{identifier}' requires at least one encoder")

        # Move to specified device
        self.to(self.device)

    def forward(self, inputs: Dict[str, any], encoder_name: Optional[str] = None) -> Dict[str, torch.Tensor]:
        """
        Forward pass through specified encoder(s).

        Args:
            inputs: Dictionary of inputs keyed by encoder name
            encoder_name: Optional specific encoder to use. If None, uses all encoders.

        Returns:
            Dictionary of outputs keyed by encoder name
        """
        outputs = {}

        if encoder_name:
            if encoder_name not in self.encoders:
                raise ValueError(f"Encoder '{encoder_name}' not found in wrapper")
            encoder = self.encoders[encoder_name]
            encoder_input = inputs.get(encoder_name, inputs)
            outputs[encoder_name] = encoder(encoder_input)
        else:
            # Process all encoders
            for name, encoder in self.encoders.items():
                if name in inputs:
                    outputs[name] = encoder(inputs[name])

        return outputs

    def encode(self, text: Union[str, List[str]], encoder_names: Optional[List[str]] = None) -> Dict[str, torch.Tensor]:
        """
        Encode text using specified encoders with their associated tokenizers.

        Args:
            text: Input text or list of texts
            encoder_names: Optional list of encoder names to use. If None, uses all.

        Returns:
            Dictionary of encoded outputs keyed by encoder name
        """
        encoder_names = encoder_names or list(self.encoders.keys())
        outputs = {}

        for name in encoder_names:
            if name not in self.encoders:
                continue

            # Get tokenizer for this encoder
            tokenizer = self.tokenizers.get(name)
            encoder = self.encoders[name]

            # Tokenize if tokenizer available
            if tokenizer:
                if hasattr(tokenizer, 'tokenize'):
                    tokens = tokenizer.tokenize(text)
                elif callable(tokenizer):
                    tokens = tokenizer(text)
                else:
                    raise ValueError(f"Tokenizer for '{name}' is not callable")
            else:
                # Assume encoder handles raw text
                tokens = text

            # Encode
            if hasattr(encoder, 'encode'):
                outputs[name] = encoder.encode(tokens)
            else:
                outputs[name] = encoder(tokens)

        return outputs

    def add_encoder(self, name: str, encoder: NeuralIO, tokenizer: Optional[NeuralIO] = None):
        """Add a new encoder to the wrapper."""
        self.encoders[name] = encoder
        if tokenizer:
            self.tokenizers[name] = tokenizer
        encoder.to(self.device)

    def remove_encoder(self, name: str):
        """Remove an encoder from the wrapper."""
        if name in self.encoders:
            del self.encoders[name]
        if name in self.tokenizers:
            del self.tokenizers[name]

    def get_encoder(self, name: str) -> Optional[NeuralIO]:
        """Get a specific encoder by name."""
        return self.encoders.get(name)

    def get_tokenizer(self, name: str) -> Optional[NeuralIO]:
        """Get a specific tokenizer by name."""
        return self.tokenizers.get(name)

    def apply_patcher(self, encoder_name: Optional[str] = None):
        """Apply patcher (e.g., for LoRA) to specified encoder(s)."""
        if not self.patcher:
            return

        if encoder_name:
            encoders_to_patch = [self.encoders.get(encoder_name)]
        else:
            encoders_to_patch = self.encoders.values()

        for encoder in encoders_to_patch:
            if encoder and hasattr(self.patcher, 'patch'):
                self.patcher.patch(encoder)

    def to(self, device: Union[str, torch.device]) -> 'EncoderWrapper':
        """Move all encoders to specified device."""
        self.device = torch.device(device) if isinstance(device, str) else device

        # Move all encoders
        for encoder in self.encoders.values():
            if hasattr(encoder, 'to'):
                encoder.to(self.device)

        # Move tokenizers if they support it
        for tokenizer in self.tokenizers.values():
            if hasattr(tokenizer, 'to'):
                tokenizer.to(self.device)

        return super().to(self.device)

    def offload_to_cpu(self, encoder_name: Optional[str] = None):
        """Offload specified encoder(s) to CPU to save GPU memory."""
        if encoder_name:
            encoder = self.encoders.get(encoder_name)
            if encoder and hasattr(encoder, 'to'):
                encoder.to('cpu')
        else:
            for encoder in self.encoders.values():
                if hasattr(encoder, 'to'):
                    encoder.to('cpu')

    def load_encoder_state(self, encoder_name: str, state_dict: dict, strict: bool = True):
        """Load state dict for a specific encoder."""
        if encoder_name not in self.encoders:
            raise ValueError(f"Encoder '{encoder_name}' not found")

        encoder = self.encoders[encoder_name]
        if hasattr(encoder, 'load_state_dict'):
            encoder.load_state_dict(state_dict, strict=strict)

    def get_encoder_config(self, encoder_name: str) -> Optional[dict]:
        """Get configuration for a specific encoder."""
        return self.config.get(encoder_name, {})

    def __repr__(self) -> str:
        return (f"EncoderWrapper(identifier='{self.identifier}', "
                f"encoders={list(self.encoders.keys())}, "
                f"device={self.device})")

import torch
import torch.nn as nn
from abc import ABC
from typing import Any, Callable, Optional, Union, List, Dict


class AbstractEncoderModel(nn.Module, ABC):
    """
    Abstract encoder interface with modular hooks, symbolic runtime overrides,
    and explicit hook lifecycle structure for tokenizer/encoder pipelines.
    """

    # Hook execution stages (declarative and canonical)
    HOOK_STAGES = [
        "pre_init",  # before any initialization, for setup
        "init",  # during initialization, for model setup
        "post_init",  # after initialization, for adjustments

        "pre_load_dict",  # before loading state dict, for pre-load adjustments
        "load_dict",  # during state dict loading, for model loading
        "post_load_dict",  # after loading state dict, for post-load adjustments

        "pre_load_layer",  # before loading a specific layer, for pre-load adjustments
        "load_layer",  # during layer loading, for model layer loading
        "post_load_layer",  # after loading a specific layer, for offloading at runtime or adjustments

        # raw input processing stages
        "pre_processing",  # before any processing, for initial setup
        "processing",  # main processing stage, e.g. tokenization
        "post_processing",  # after main processing, for cleanup or adjustments
        # tokenization and encoding stages
        "pre_tokenize",  # before tokenizing input
        "tokenize",  # main tokenization stage
        "post_tokenize",  # after tokenization, for adjustments or checks
        # encoding stages
        "pre_encode",  # after tokenization, before encoding
        "encode",  # main encoding stage, where the model processes tokens
        "post_encode",  # after encoder forward pass, for post processing adjustments
        # output transformation stages
        "pre_output_transform",  # before final transformations, e.g. pooling
        "output_transform",  # before final return (projection, slicing, etc)
        "post_output_transform"  # after output transformation, for final adjustments
    ]

    # Standard hook aliases for common behaviors
    HOOK_ALIASES = {
        # Memory management hooks
        "memory_tracker": "track_layer_memory",
        "device_mover": "move_layer_to_device",
        "memory_calculator": "calculate_memory_usage",
        "offloader": "offload_layer_to_cpu",

        # State dict tracking
        "key_tracker": "track_state_dict_keys",
        "key_transformer": "transform_state_dict_keys",

        # ComfyUI compatibility
        "comfy_dtype": "apply_comfy_dtype_policy",
        "comfy_patcher": "apply_comfy_patches",
        "comfy_loader": "comfy_partial_loader",
        "comfy_unloader": "comfy_partial_unloader",

        # Monitoring
        "load_monitor": "monitor_layer_load",
        "memory_monitor": "monitor_memory_usage"
    }

    def __init__(self, identifier: str = "", config: Optional[dict] = None, device: Union[str, torch.device] = "cpu"):
        super().__init__()
        self.identifier = identifier
        self.config = config or {}
        self.device = torch.device(device)

        # Model and tokenizer are expected to be injected or registered
        self.model = None
        self.tokenizer = None

        # External registration slots
        self._tokenizer_fn: Optional[Callable] = None
        self._encoder_fn: Optional[Callable] = None

        # Hook pipeline (visibly and explicitly declared)
        self._hooks: Dict[str, List[tuple[int, Callable, str]]] = {}  # priority, function, alias
        self._hook_registry: Dict[str, Callable] = {}  # Alias -> function mapping

        for stage in self.HOOK_STAGES:
            self._hooks[stage] = []

        # ComfyUI compatibility attributes
        self.loaded_layers = {}  # layer_name -> device
        self.layer_sizes = {}  # layer_name -> size
        self._offload_device = torch.device('cpu')
        self._current_memory_limit = float('inf')
        self._target_device = self.device

        # Register standard hooks
        self._register_standard_hooks()

    def _register_standard_hooks(self):
        """Register all standard ComfyUI compatibility hooks"""

        # Memory tracking hook
        @self.register_hook_fn("memory_tracker")
        def track_layer_memory(data):
            if isinstance(data, tuple) and len(data) == 2:
                layer_name, layer = data
                size = sum(p.nelement() * p.element_size() for p in layer.parameters())
                self.layer_sizes[layer_name] = size
            return data

        # Device movement hook
        @self.register_hook_fn("device_mover")
        def move_layer_to_device(data):
            if isinstance(data, tuple) and len(data) == 2:
                layer_name, layer = data
                target_device = getattr(self, '_target_device', self.device)

                # Check memory budget
                current_loaded = self._calculate_loaded_memory(target_device)
                layer_size = self.layer_sizes.get(layer_name, 0)

                if current_loaded + layer_size <= self._current_memory_limit:
                    layer.to(target_device)
                    self.loaded_layers[layer_name] = target_device
                else:
                    layer.to(self._offload_device)
                    self.loaded_layers[layer_name] = self._offload_device

            return data

        # State dict key tracking
        @self.register_hook_fn("key_tracker")
        def track_state_dict_keys(data):
            if isinstance(data, tuple) and len(data) == 2:
                layer_name, layer = data
                if hasattr(layer, 'state_dict'):
                    keys = list(layer.state_dict().keys())
                    if not hasattr(self, '_key_mappings'):
                        self._key_mappings = {}
                    self._key_mappings[layer_name] = keys
            return data

        # ComfyUI dtype policy
        @self.register_hook_fn("comfy_dtype")
        def apply_comfy_dtype(data):
            if isinstance(data, tuple) and len(data) == 2:
                layer_name, layer = data
                import comfy.model_management as mm
                dtype = mm.text_encoder_dtype(self.device)
                layer.to(dtype=dtype)
            return data

        # Partial loader for ComfyUI
        @self.register_hook_fn("comfy_loader")
        def partial_load_layer(data):
            """Load individual layer respecting memory limits"""
            if isinstance(data, tuple) and len(data) == 2:
                layer_name, layer = data
                if self.loaded_layers.get(layer_name) != self._target_device:
                    # Trigger device movement through hook chain
                    self._run_hooks("pre_load_layer", data)
                    self._run_hooks("load_layer", data)
                    self._run_hooks("post_load_layer", data)
            return data

    def pre_load_layer(self):
        """Run pre-load hooks before loading a specific layer."""
        self._run_hooks("pre_load_layer", None)

    def load_layer(self, layer: Any):
        """Run load hooks for a specific layer."""
        if self._encoder_fn is None:
            raise NotImplementedError("Encoder function has not been registered.")
        self._encoder_fn(layer)
        self._run_hooks("load_layer", layer)

    def post_load_layer(self):
        """Run post-load hooks after loading a specific layer."""
        self._run_hooks("post_load_layer", None)

    def pre_load(self):
        """Run pre-load hooks before loading the model."""
        self._run_hooks("pre_load_dict", None)

    def forward(self, *args, **kwargs) -> torch.Tensor:
        """
        Forward pass that captures all arguments raw and passes them through the pipeline.
        """
        # Keep args/kwargs raw - let hooks and registered functions handle them
        raw_input = (args, kwargs)

        raw_input = self._run_hooks("pre_tokenize", raw_input)
        tokens = self._run_hooks("tokenize", self.tokenize(args, kwargs))

        tokens = self._run_hooks("pre_encode", tokens)
        encoded = self._run_hooks("encode", self.encode(tokens))

        encoded = self._run_hooks("post_encode", encoded)
        encoded = self._run_hooks("output_transform", encoded)

        return encoded

    def tokenize(self, args: tuple, kwargs: dict) -> Any:
        """Tokenize inputs using registered tokenizer function"""
        if self._tokenizer_fn is None:
            raise NotImplementedError("Tokenizer function has not been registered.")
        return self._tokenizer_fn(args, kwargs)

    def encode(self, tokens: Any) -> torch.Tensor:
        """Encode tokens using registered encoder function"""
        if self._encoder_fn is None:
            raise NotImplementedError("Encoder function has not been registered.")
        return self._encoder_fn(tokens)

    def register_tokenizer(self, fn: Callable):
        """Register a tokenizer function."""
        self._tokenizer_fn = fn

    def register_encoder(self, fn: Callable):
        """Register an encoder function."""
        self._encoder_fn = fn

    def register_hook_fn(self, alias: str):
        """Decorator to register a hook function with an alias"""

        def decorator(fn):
            self._hook_registry[alias] = fn
            return fn

        return decorator

    def add_hook(self, stage: str, hook_input: Union[str, Callable], priority: int = 0):
        """Add a callable hook at a specific execution stage."""
        if stage not in self._hooks:
            raise ValueError(f"Unknown hook stage: {stage}")

        # Resolve alias to function
        if isinstance(hook_input, str):
            if hook_input not in self._hook_registry:
                raise ValueError(f"Unknown hook alias: {hook_input}")
            fn = self._hook_registry[hook_input]
            alias = hook_input
        else:
            fn = hook_input
            alias = fn.__name__

        self._hooks[stage].append((priority, fn, alias))
        self._hooks[stage].sort(key=lambda x: x[0], reverse=True)

    def remove_hook(self, stage: str, alias: str):
        """Remove hook by alias"""
        if stage not in self._hooks:
            raise ValueError(f"Unknown hook stage: {stage}")
        self._hooks[stage] = [(p, f, a) for p, f, a in self._hooks[stage] if a != alias]

    def clear_hooks(self, stage: Optional[str] = None):
        """Remove all hooks from one or all stages."""
        if stage:
            if stage not in self._hooks:
                raise ValueError(f"Unknown hook stage: {stage}")
            self._hooks[stage] = []
        else:
            for stage in self.HOOK_STAGES:
                self._hooks[stage] = []

    def _run_hooks(self, stage: str, data: Any) -> Any:
        """Run all hooks for a given stage."""
        for _, hook, _ in self._hooks.get(stage, []):
            data = hook(data)
        return data

    def move_to_device(self, device: Union[str, torch.device]):
        """Move model and tokenizer to specified device."""
        self.device = torch.device(device)
        if self.model:
            self.model.to(self.device)
        if hasattr(self.tokenizer, 'to'):
            self.tokenizer.to(self.device)

    def enable_comfy_compatibility(self):
        """Enable all ComfyUI compatibility hooks"""
        # Memory management
        self.add_hook("pre_load_layer", "memory_tracker", priority=100)
        self.add_hook("pre_load_layer", "key_tracker", priority=95)
        self.add_hook("load_layer", "comfy_dtype", priority=90)
        self.add_hook("load_layer", "device_mover", priority=85)

        # Enable patches if available
        if hasattr(self, 'patcher') and self.patcher:
            self.add_hook("post_load_layer", "comfy_patcher", priority=50)

    def disable_comfy_compatibility(self):
        """Disable ComfyUI hooks for standalone operation"""
        for stage in ["pre_load_layer", "load_layer", "post_load_layer"]:
            self.remove_hook(stage, "memory_tracker")
            self.remove_hook(stage, "device_mover")
            self.remove_hook(stage, "comfy_dtype")
            self.remove_hook(stage, "key_tracker")

    # ComfyUI interface methods
    def partially_load(self, device, extra_memory, force_patch_weights=False):
        """ComfyUI-compatible partial loading"""
        self._target_device = device
        self._current_memory_limit = extra_memory

        # Ensure ComfyUI hooks are enabled
        self.enable_comfy_compatibility()

        # Run loading through hooks
        loaded = 0
        for layer_name, layer in self._get_named_layers():
            if self.loaded_layers.get(layer_name) == device:
                continue

            # Let hooks handle everything
            self._run_hooks("comfy_loader", (layer_name, layer))

            if self.loaded_layers.get(layer_name) == device:
                loaded += self.layer_sizes.get(layer_name, 0)

        return loaded

    def partially_unload(self, device, memory_to_free):
        """Unload layers to free memory"""
        freed = 0

        # Unload in reverse order
        for layer_name, layer in reversed(list(self._get_named_layers())):
            if freed >= memory_to_free:
                break

            if self.loaded_layers.get(layer_name) != device:
                layer.to(device)  # Move to offload device
                self.loaded_layers[layer_name] = device
                freed += self.layer_sizes.get(layer_name, 0)

        return freed

    def model_size(self):
        """Total model size for ComfyUI"""
        if not self.layer_sizes:
            # Calculate on first call
            for layer_name, layer in self._get_named_layers():
                self._run_hooks("memory_tracker", (layer_name, layer))
        return sum(self.layer_sizes.values())

    def loaded_size(self):
        """Currently loaded size for ComfyUI"""
        return self._calculate_loaded_memory(self.device)

    def _calculate_loaded_memory(self, device):
        """Helper to calculate loaded memory on device"""
        return sum(size for name, size in self.layer_sizes.items()
                   if self.loaded_layers.get(name) == device)

    def _get_named_layers(self):
        """Get layers in load priority order"""
        if self.model and hasattr(self.model, 'named_modules'):
            for name, module in self.model.named_modules():
                # Only yield modules with parameters (actual layers)
                if len(list(module.parameters())) > 0 and len(list(module.children())) == 0:
                    yield name, module
        elif self.model:
            # Fallback for non-standard models
            yield "model", self.model

    def configure_hooks(self, hook_config: Dict[str, List[str]]):
        """Configure which hooks are active for each stage"""
        for stage, aliases in hook_config.items():
            # Clear existing hooks for stage
            self.clear_hooks(stage)
            # Add requested hooks
            for alias in aliases:
                self.add_hook(stage, alias)

    def get_active_hooks(self, stage: Optional[str] = None) -> Union[List[str], Dict[str, List[str]]]:
        """Get currently active hooks by stage"""
        if stage:
            return [alias for _, _, alias in self._hooks.get(stage, [])]
        return {s: [alias for _, _, alias in hooks] for s, hooks in self._hooks.items()}

    # ComfyUI compatibility stubs
    def detach(self, unpatch_all=True):
        """Detach model from GPU (ComfyUI compatibility)"""
        for layer_name, layer in self._get_named_layers():
            layer.to(self._offload_device)
            self.loaded_layers[layer_name] = self._offload_device

    def model_patches_to(self, device):
        """Apply patches to device (stub for ComfyUI compatibility)"""
        # This would be implemented by concrete classes that support patching
        pass

    def model_patches_to_dtype(self, dtype):
        """Apply patches to dtype (stub for ComfyUI compatibility)"""
        # This would be implemented by concrete classes that support patching
        pass

    def lowvram_patch_counter(self):
        """Count patches for low VRAM mode (stub)"""
        return 0

    def clone(self):
        """Create a clone of this model"""
        # Basic cloning - concrete implementations might need more
        import copy
        return copy.deepcopy(self)

    def is_clone(self, other):
        """Check if other model is a clone of this one"""
        return self.identifier == getattr(other, 'identifier', None)