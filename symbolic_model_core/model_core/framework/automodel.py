"""
Modular encoder class hierarchy for symbolic AI frameworks.
Defines abstract base classes, device-aware model control, and hook systems.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, Union, Dict, List, Any, Callable, Tuple
from enum import Enum
import torch
import torch.nn as nn
import logging
logger = logging.getLogger(__name__)

# ========== Metadata and Interface Definitions ==========

@dataclass
class EncoderMetadata:
    identifier: str
    encoder_type: str
    version: str = "1.0"
    capabilities: List[str] = field(default_factory=list)
    requirements: Dict[str, Any] = field(default_factory=dict)
    config: Dict[str, Any] = field(default_factory=dict)


class IEncoder(ABC):
    @abstractmethod
    def encode(self, inputs: Any) -> torch.Tensor:
        pass

    @abstractmethod
    def get_metadata(self) -> EncoderMetadata:
        pass


class ITokenizer(ABC):
    @abstractmethod
    def tokenize(self, inputs: Union[str, List[str]]) -> Any:
        pass

    @abstractmethod
    def detokenize(self, tokens: Any) -> Union[str, List[str]]:
        pass


# ========== Hook System ==========

class HookStage(Enum):
    PRE_INIT = "pre_init"
    POST_INIT = "post_init"
    PRE_PROCESS = "pre_process"
    POST_PROCESS = "post_process"
    PRE_FORWARD = "pre_forward"
    POST_FORWARD = "post_forward"
    PRE_LOAD = "pre_load"
    POST_LOAD = "post_load"
    PRE_UNLOAD = "pre_unload"
    POST_UNLOAD = "post_unload"


class HookManager:
    def __init__(self):
        self._hooks: Dict[HookStage, List[Tuple[int, Callable, str]]] = {
            stage: [] for stage in HookStage
        }
        self._hook_registry: Dict[str, Callable] = {}

    def register_hook(self, name: str) -> Callable:
        def decorator(fn: Callable) -> Callable:
            self._hook_registry[name] = fn
            return fn
        return decorator

    def add_hook(self, stage: Union[str, HookStage], hook: Union[str, Callable], priority: int = 0) -> None:
        if isinstance(stage, str):
            stage = HookStage[stage.upper().replace("-", "_")]
        if isinstance(hook, str):
            hook_fn = self._hook_registry.get(hook)
            if hook_fn is None:
                raise ValueError(f"Unknown hook name: {hook}")
            hook_name = hook
        else:
            hook_fn = hook
            hook_name = hook.__name__
        self._hooks[stage].append((priority, hook_fn, hook_name))
        self._hooks[stage].sort(key=lambda x: x[0], reverse=True)

    def run_hooks(self, stage: Union[str, HookStage], data: Any) -> Any:
        if isinstance(stage, str):
            stage = HookStage[stage.upper().replace("-", "_")]
        for _, hook_fn, _ in self._hooks[stage]:
            data = hook_fn(data)
        return data

    def list_stages(self) -> List[str]:
        return [stage.value for stage in HookStage]


# ========== Memory Manager ==========

@dataclass
class LayerInfo:
    name: str
    size: int
    device: torch.device
    dtype: torch.dtype


class MemoryManager:
    def __init__(self, device: torch.device = torch.device("cpu")):
        self.device = device
        self.layers: Dict[str, LayerInfo] = {}
        self.memory_limit = float("inf")

    def track_layer(self, name: str, module: nn.Module) -> None:
        size = sum(p.nelement() * p.element_size() for p in module.parameters(recurse=True))
        self.layers[name] = LayerInfo(
            name=name,
            size=size,
            device=next(module.parameters()).device,
            dtype=next(module.parameters()).dtype
        )

    def can_load(self, name: str) -> bool:
        return self.get_device_usage(self.device) + self.layers[name].size <= self.memory_limit

    def get_device_usage(self, device: torch.device) -> int:
        return sum(info.size for info in self.layers.values() if info.device == device)

    def get_total_size(self) -> int:
        return sum(info.size for info in self.layers.values())


# ========== BaseEncoder ==========

class BaseEncoder(nn.Module, IEncoder):
    def __init__(self, metadata: EncoderMetadata, device: Union[str, torch.device] = "cpu"):
        super().__init__()
        self.metadata = metadata
        self.device = torch.device(device)
        self.hooks = HookManager()
        self.memory = MemoryManager(self.device)
        self.model: Optional[nn.Module] = None
        self.tokenizer: Optional[ITokenizer] = None

        self.hooks.run_hooks(HookStage.PRE_INIT, self)
        self._setup_default_hooks()
        self.hooks.run_hooks(HookStage.POST_INIT, self)

    def _setup_default_hooks(self):
        @self.hooks.register_hook("track_memory")
        def track_memory(data):
            if isinstance(data, tuple) and len(data) == 2:
                name, module = data
                self.memory.track_layer(name, module)
            return data

        self.hooks.add_hook(HookStage.POST_LOAD, "track_memory")

    def set_model(self, model: nn.Module) -> None:
        self.model = model.to(self.device)
        for name, module in model.named_modules():
            if any(p.requires_grad for p in module.parameters(recurse=False)):
                self.memory.track_layer(name, module)

    def set_tokenizer(self, tokenizer: ITokenizer) -> None:
        self.tokenizer = tokenizer

    def encode(self, inputs: Any) -> torch.Tensor:
        inputs = self.hooks.run_hooks(HookStage.PRE_PROCESS, inputs)
        if isinstance(inputs, (str, list)) and self.tokenizer:
            inputs = self.tokenizer.tokenize(inputs)
        inputs = self.hooks.run_hooks(HookStage.PRE_FORWARD, inputs)

        if self.model is None:
            raise ValueError("No model set")

        with torch.no_grad():
            output = self.model(inputs)

        output = self.hooks.run_hooks(HookStage.POST_FORWARD, output)
        output = self.hooks.run_hooks(HookStage.POST_PROCESS, output)
        return output

    # alias forward for encode
    def forward(self, inputs: Any) -> torch.Tensor:
        return self.encode(inputs)
    def get_metadata(self) -> EncoderMetadata:
        return self.metadata

    def to(self, device: Union[str, torch.device]) -> 'BaseEncoder':
        self.device = torch.device(device)
        if self.model:
            self.model.to(self.device)
        for layer_info in self.memory.layers.values():
            layer_info.device = self.device
        return self


# ========== Registry ==========

class EncoderRegistry:
    _encoders: Dict[str, type] = {}

    @classmethod
    def register(cls, name: str, encoder_class: type) -> None:
        if not issubclass(encoder_class, IEncoder):
            raise TypeError(f"{encoder_class} must implement IEncoder")
        cls._encoders[name] = encoder_class

    @classmethod
    def create(cls, name: str, *args, **kwargs) -> IEncoder:
        if name not in cls._encoders:
            raise ValueError(f"Unknown encoder: {name}")
        return cls._encoders[name](*args, **kwargs)

    @classmethod
    def list_encoders(cls) -> List[str]:
        return list(cls._encoders.keys())
