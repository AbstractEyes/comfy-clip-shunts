from typing import Optional, Dict, Any, List

import torch

import logging
logger = logging.getLogger(__name__)

from model_core.framework.automodel import BaseEncoder, EncoderMetadata

class EncoderOrchestrator:
    """
    Coordinates multiple encoder instances.
    Handles encoder registration, optional adapter logic, and batch encode dispatch.
    """

    def __init__(self):
        self.encoders: Dict[str, BaseEncoder] = {}
        self.adapters: Dict[str, Any] = {}  # Optional ComfyUI or custom adapters

    def add_encoder(self, name: str, encoder: BaseEncoder, adapter: Optional[Any] = None) -> None:
        """
        Register an encoder instance with optional adapter.
        """
        self.encoders[name] = encoder
        if adapter is not None:
            self.adapters[name] = adapter

    def encode(self, inputs: Any, encoder_names: Optional[List[str]] = None) -> Dict[str, torch.Tensor]:
        """
        Run encode on one or more registered encoders.

        Returns:
            Dict mapping encoder name to encoded output.
        """
        if encoder_names is None:
            encoder_names = list(self.encoders.keys())

        results = {}
        for name in encoder_names:
            encoder = self.encoders.get(name)
            if not encoder:
                continue
            try:
                results[name] = encoder.encode(inputs)
            except Exception as e:
                print(f"[EncoderOrchestrator] Failed to encode with '{name}': {e}")
        return results

    def get_encoder(self, name: str) -> Optional[BaseEncoder]:
        return self.encoders.get(name)

    def get_adapter(self, name: str) -> Optional[Any]:
        return self.adapters.get(name)

    def list_encoders(self) -> List[str]:
        return list(self.encoders.keys())

    def remove_encoder(self, name: str) -> None:
        self.encoders.pop(name, None)
        self.adapters.pop(name, None)
