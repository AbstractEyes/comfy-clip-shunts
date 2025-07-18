"""
Modular encoder class hierarchy for ComfyUI
Designed for maximum reusability and clean separation of concerns
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, Union, Dict, List, Any, Callable, Tuple
from enum import Enum
import torch
import torch.nn as nn


# ============================================
# Core Interfaces and Base Types
# ============================================

@dataclass
class EncoderMetadata:
    """Metadata for encoder identification and configuration"""
    identifier: str
    encoder_type: str
    version: str = "1.0"
    capabilities: List[str] = field(default_factory=list)
    requirements: Dict[str, Any] = field(default_factory=dict)
    config: Dict[str, Any] = field(default_factory=dict)


class IEncoder(ABC):
    """Core encoder interface - minimal contract all encoders must fulfill"""

    @abstractmethod
    def encode(self, inputs: Any) -> torch.Tensor:
        """Transform inputs into embeddings"""
        pass

    @abstractmethod
    def get_metadata(self) -> EncoderMetadata:
        """Return encoder metadata"""
        pass


class ITokenizer(ABC):
    """Core tokenizer interface"""

    @abstractmethod
    def tokenize(self, inputs: Union[str, List[str]]) -> Any:
        """Transform text into tokens"""
        pass

    @abstractmethod
    def detokenize(self, tokens: Any) -> Union[str, List[str]]:
        """Transform tokens back to text"""
        pass


# ============================================
# Hook Management System
# ============================================

class HookStage(Enum):
    """Simplified hook stages focusing on essential extension points"""
    # Lifecycle hooks
    PRE_INIT = "pre_init"
    POST_INIT = "post_init"

    # Processing hooks
    PRE_PROCESS = "pre_process"
    POST_PROCESS = "post_process"

    # Model operation hooks
    PRE_FORWARD = "pre_forward"
    POST_FORWARD = "post_forward"

    # Memory management hooks
    PRE_LOAD = "pre_load"
    POST_LOAD = "post_load"
    PRE_UNLOAD = "pre_unload"
    POST_UNLOAD = "post_unload"


class HookManager:
    """Centralized hook management"""

    def __init__(self):
        self._hooks: Dict[HookStage, List[Tuple[int, Callable, str]]] = {
            stage: [] for stage in HookStage
        }
        self._hook_registry: Dict[str, Callable] = {}

    def register_hook(self, name: str) -> Callable:
        """Register a named hook function as a decorator"""
        def decorator(hook: Callable) -> Callable:
            self._hook_registry[name] = hook
            return hook
        return decorator

    def add_hook(self, stage: Union[HookStage, str], hook: Union[str, Callable],
                 priority: int = 0) -> None:
        """Add hook to stage with priority (higher = earlier)"""
        # Convert string to HookStage if needed
        if isinstance(stage, str):
            try:
                stage = HookStage(stage)
            except ValueError:
                # Try to find by name
                stage_name = stage.upper().replace("-", "_")
                try:
                    stage = HookStage[stage_name]
                except KeyError:
                    raise ValueError(f"Unknown hook stage: {stage}")

        if isinstance(hook, str):
            if hook not in self._hook_registry:
                raise ValueError(f"Unknown hook: {hook}")
            hook_fn = self._hook_registry[hook]
            hook_name = hook
        else:
            hook_fn = hook
            hook_name = hook.__name__

        self._hooks[stage].append((priority, hook_fn, hook_name))
        self._hooks[stage].sort(key=lambda x: x[0], reverse=True)

    def remove_hook(self, stage: Union[HookStage, str], name: str) -> None:
        """Remove hook by name"""
        # Convert string to HookStage if needed
        if isinstance(stage, str):
            try:
                stage = HookStage(stage)
            except ValueError:
                stage_name = stage.upper().replace("-", "_")
                try:
                    stage = HookStage[stage_name]
                except KeyError:
                    raise ValueError(f"Unknown hook stage: {stage}")

        self._hooks[stage] = [
            (p, f, n) for p, f, n in self._hooks[stage] if n != name
        ]

    def run_hooks(self, stage: Union[HookStage, str], data: Any) -> Any:
        """Execute all hooks for a stage"""
        # Convert string to HookStage if needed
        if isinstance(stage, str):
            try:
                stage = HookStage(stage)
            except ValueError:
                stage_name = stage.upper().replace("-", "_")
                try:
                    stage = HookStage[stage_name]
                except KeyError:
                    raise ValueError(f"Unknown hook stage: {stage}")

        for _, hook, _ in self._hooks[stage]:
            data = hook(data)
        return data

    def clear_stage(self, stage: Union[HookStage, str]) -> None:
        """Clear all hooks from a stage"""
        # Convert string to HookStage if needed
        if isinstance(stage, str):
            try:
                stage = HookStage(stage)
            except ValueError:
                stage_name = stage.upper().replace("-", "_")
                try:
                    stage = HookStage[stage_name]
                except KeyError:
                    raise ValueError(f"Unknown hook stage: {stage}")

        self._hooks[stage] = []

    @staticmethod
    def list_stages() -> List[str]:
        """List all available hook stages"""
        return [stage.value for stage in HookStage]


# ============================================
# Memory and Device Management
# ============================================

@dataclass
class LayerInfo:
    """Information about a model layer"""
    name: str
    size: int
    device: torch.device
    dtype: torch.dtype


class MemoryManager:
    """Manages memory allocation and tracking for models"""

    def __init__(self, device: torch.device = torch.device("cpu")):
        self.device = device
        self.layers: Dict[str, LayerInfo] = {}
        self.memory_limit: float = float('inf')
        self._offload_device = torch.device('cpu')

    def track_layer(self, name: str, module: nn.Module) -> None:
        """Track a layer's memory usage"""
        size = sum(p.nelement() * p.element_size()
                  for p in module.parameters(recurse=True))

        self.layers[name] = LayerInfo(
            name=name,
            size=size,
            device=next(module.parameters()).device,
            dtype=next(module.parameters()).dtype
        )

    def can_load(self, layer_name: str) -> bool:
        """Check if layer can be loaded within memory limit"""
        if layer_name not in self.layers:
            return False

        current_usage = self.get_device_usage(self.device)
        layer_size = self.layers[layer_name].size

        return current_usage + layer_size <= self.memory_limit

    def get_device_usage(self, device: torch.device) -> int:
        """Calculate total memory usage on device"""
        return sum(info.size for info in self.layers.values()
                  if info.device == device)

    def get_total_size(self) -> int:
        """Get total model size"""
        return sum(info.size for info in self.layers.values())


# ============================================
# Base Encoder Implementation
# ============================================

class BaseEncoder(nn.Module, IEncoder):
    """Base encoder with hook and memory management"""

    def __init__(self, metadata: EncoderMetadata,
                 device: Union[str, torch.device] = "cpu"):
        super().__init__()
        self.metadata = metadata
        self.device = torch.device(device) if isinstance(device, str) else device

        # Component managers
        self.hooks = HookManager()
        self.memory = MemoryManager(self.device)

        # Model components
        self.model: Optional[nn.Module] = None
        self.tokenizer: Optional[ITokenizer] = None

        # Run initialization hooks
        self.hooks.run_hooks(HookStage.PRE_INIT, self)
        self._setup_default_hooks()
        self.hooks.run_hooks(HookStage.POST_INIT, self)

    def _setup_default_hooks(self):
        """Register default hooks for common operations"""

        @self.hooks.register_hook("track_memory")
        def track_memory(data):
            if isinstance(data, tuple) and len(data) == 2:
                name, module = data
                self.memory.track_layer(name, module)
            return data

        # Add to appropriate stages
        self.hooks.add_hook(HookStage.POST_LOAD, "track_memory")

    def set_model(self, model: nn.Module) -> None:
        """Set the underlying model"""
        self.model = model
        self.model.to(self.device)

        # Track all layers
        for name, module in model.named_modules():
            if len(list(module.parameters(recurse=False))) > 0:
                self.memory.track_layer(name, module)

    def set_tokenizer(self, tokenizer: ITokenizer) -> None:
        """Set the tokenizer"""
        self.tokenizer = tokenizer

    def encode(self, inputs: Any) -> torch.Tensor:
        """Encode inputs with full hook pipeline"""
        # Pre-process hooks
        inputs = self.hooks.run_hooks(HookStage.PRE_PROCESS, inputs)

        # Tokenize if needed
        if isinstance(inputs, (str, list)) and self.tokenizer:
            inputs = self.tokenizer.tokenize(inputs)

        # Pre-forward hooks
        inputs = self.hooks.run_hooks(HookStage.PRE_FORWARD, inputs)

        # Forward pass
        if self.model is None:
            raise ValueError("No model set")

        # Use new autocast syntax
        device_type = 'cuda' if inputs.is_cuda else 'cpu'
        with torch.amp.autocast(device_type=device_type, enabled=False):  # Let user control this
            output = self.model(inputs)

        # Post-forward hooks
        output = self.hooks.run_hooks(HookStage.POST_FORWARD, output)

        # Post-process hooks
        output = self.hooks.run_hooks(HookStage.POST_PROCESS, output)

        return output

    def get_metadata(self) -> EncoderMetadata:
        """Return encoder metadata"""
        return self.metadata

    def to(self, device: Union[str, torch.device]) -> 'BaseEncoder':
        """Move encoder to device"""
        self.device = torch.device(device) if isinstance(device, str) else device

        if self.model:
            self.model.to(self.device)

        # Update memory tracking
        for layer_info in self.memory.layers.values():
            layer_info.device = self.device

        return self


# ============================================
# ComfyUI Integration Layer
# ============================================

class ComfyUIAdapter:
    """Adapter for ComfyUI-specific operations"""

    def __init__(self, encoder: BaseEncoder):
        self.encoder = encoder
        self._setup_comfy_hooks()

    def _setup_comfy_hooks(self):
        """Setup ComfyUI-specific hooks"""

        @self.encoder.hooks.register_hook("comfy_dtype")
        def apply_comfy_dtype(module):
            # Import only when needed
            try:
                import comfy.model_management as mm
                dtype = mm.text_encoder_dtype(self.encoder.device)
                if hasattr(module, 'to'):
                    module.to(dtype=dtype)
            except ImportError:
                pass
            return module

        self.encoder.hooks.add_hook(HookStage.POST_LOAD, "comfy_dtype")

    def partially_load(self, device: torch.device, memory_limit: float) -> int:
        """ComfyUI-compatible partial loading"""
        self.encoder.memory.memory_limit = memory_limit
        loaded = 0

        if self.encoder.model is None:
            return 0

        for name, module in self.encoder.model.named_modules():
            if name not in self.encoder.memory.layers:
                continue

            if self.encoder.memory.can_load(name):
                module.to(device)
                self.encoder.memory.layers[name].device = device
                loaded += self.encoder.memory.layers[name].size
            else:
                module.to('cpu')
                self.encoder.memory.layers[name].device = torch.device('cpu')

        return loaded

    def model_size(self) -> int:
        """Total model size for ComfyUI"""
        return self.encoder.memory.get_total_size()

    def loaded_size(self) -> int:
        """Currently loaded size"""
        return self.encoder.memory.get_device_usage(self.encoder.device)


# ============================================
# Encoder Registry and Factory
# ============================================

class EncoderRegistry:
    """Central registry for encoder types"""

    _encoders: Dict[str, type] = {}

    @classmethod
    def register(cls, name: str, encoder_class: type) -> None:
        """Register an encoder class"""
        if not issubclass(encoder_class, IEncoder):
            raise TypeError(f"{encoder_class} must implement IEncoder")
        cls._encoders[name] = encoder_class

    @classmethod
    def create(cls, name: str, *args, **kwargs) -> IEncoder:
        """Create an encoder instance"""
        if name not in cls._encoders:
            raise ValueError(f"Unknown encoder type: {name}")
        return cls._encoders[name](*args, **kwargs)

    @classmethod
    def list_encoders(cls) -> List[str]:
        """List available encoder types"""
        return list(cls._encoders.keys())


# ============================================
# Orchestration Layer
# ============================================

class EncoderOrchestrator:
    """Orchestrates multiple encoders"""

    def __init__(self):
        self.encoders: Dict[str, BaseEncoder] = {}
        self.adapters: Dict[str, ComfyUIAdapter] = {}

    def add_encoder(self, name: str, encoder: BaseEncoder,
                   enable_comfy: bool = True) -> None:
        """Add an encoder with optional ComfyUI support"""
        self.encoders[name] = encoder

        if enable_comfy:
            self.adapters[name] = ComfyUIAdapter(encoder)

    def encode(self, inputs: Any, encoder_names: Optional[List[str]] = None) -> Dict[str, torch.Tensor]:
        """Encode using specified encoders"""
        if encoder_names is None:
            encoder_names = list(self.encoders.keys())

        results = {}
        for name in encoder_names:
            if name in self.encoders:
                try:
                    results[name] = self.encoders[name].encode(inputs)
                except Exception as e:
                    # Log error but continue with other encoders
                    print(f"Error encoding with {name}: {e}")

        return results

    def get_encoder(self, name: str) -> Optional[BaseEncoder]:
        """Get encoder by name"""
        return self.encoders.get(name)

    def get_adapter(self, name: str) -> Optional[ComfyUIAdapter]:
        """Get ComfyUI adapter for encoder"""
        return self.adapters.get(name)


# ============================================
# Example Concrete Implementation
# ============================================

class TextEncoder(BaseEncoder):
    """Example text encoder implementation"""

    def __init__(self, model_name: str, device: Union[str, torch.device] = "cpu"):
        metadata = EncoderMetadata(
            identifier=model_name,
            encoder_type="text",
            capabilities=["text_embedding"],
            config={"model": model_name}
        )
        super().__init__(metadata, device)

        # Add text-specific hooks
        self._setup_text_hooks()

    def _setup_text_hooks(self):
        """Setup text processing hooks"""

        @self.hooks.register_hook("normalize_text")
        def normalize_text(inputs):
            if isinstance(inputs, str):
                return inputs.strip().lower()
            elif isinstance(inputs, list):
                return [s.strip().lower() for s in inputs]
            return inputs

        # Add normalization to pre-process
        self.hooks.add_hook(HookStage.PRE_PROCESS, "normalize_text", priority=10)



# Register the text encoder
EncoderRegistry.register("text", TextEncoder)