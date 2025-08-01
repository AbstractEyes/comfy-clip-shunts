# rose_omega.py
# ----------------------------------------------------------------------
# Omega: Universal Role-Based Similarity Interpreter
# Implementation of Rose using OmegaInput structures and routed adapters

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Callable, Any, List
from dataclasses import dataclass

from .rose_config import RoseConfig
from .rose import Rose  # Base Rose class (for seeding, cleanup)


@dataclass
class OmegaInput:
    raw: Any
    domain: str = "tensor"
    representation: str = "auto"  # e.g., mean, cls, flat, normalize, etc.
    tags: Optional[List[str]] = None
    embedding: Optional[torch.Tensor] = None
    meta: Optional[Dict[str, Any]] = None


class OmegaPreprocessor:
    def __init__(self):
        self.registry: Dict[str, Callable[[torch.Tensor], torch.Tensor]] = {}

    def register(self, mode: str, fn: Callable[[torch.Tensor], torch.Tensor]):
        self.registry[mode] = fn

    def apply(self, mode: str, tensor: torch.Tensor) -> torch.Tensor:
        if mode not in self.registry:
            raise ValueError(f"No preprocessor registered for mode '{mode}'")
        return self.registry[mode](tensor)


class RoseOmega(Rose):
    """
    Omega: The final Rose-based interpreter class for symbolic, semantic, and structural similarity.
    Supports arbitrary domains and user-defined preprocessing pipelines.
    """
    def __init__(self, a: OmegaInput, b: OmegaInput, config: Optional[RoseConfig] = None):
        super().__init__({}, config)  # No tensor bank needed
        self.a = a
        self.b = b
        self.adapter_registry: Dict[str, Callable[[OmegaInput], torch.Tensor]] = {}
        self.preprocessor = OmegaPreprocessor()

    def register_adapter(self, domain: str, fn: Callable[[OmegaInput], torch.Tensor]):
        self.adapter_registry[domain] = fn

    def forward(self) -> torch.Tensor:
        self.seed()
        a_vec = self._tensorize(self.a)
        b_vec = self._tensorize(self.b)
        return F.cosine_similarity(a_vec.view(1, -1), b_vec.view(1, -1)).squeeze()

    def _tensorize(self, inp: OmegaInput) -> torch.Tensor:
        if inp.embedding is not None:
            return inp.embedding.to(self.primary_device())

        if inp.domain not in self.adapter_registry:
            raise ValueError(f"No adapter registered for domain '{inp.domain}'")

        base = self.adapter_registry[inp.domain](inp).to(self.primary_device())

        # Apply optional preprocessing
        if inp.representation and inp.representation != "auto":
            return self.preprocessor.apply(inp.representation, base)
        return base

    def primary_device(self) -> torch.device:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
