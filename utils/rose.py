# ----------------------------------------------------------------------
# ROSE Class Architecture: Abstract + Specialized Instantiations
# Base classes only; Omega logic moved to rose_omega.py
# Author: AbstractPhil
# Date: 07/31/2025
# License: Apache-2.0

import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any

from .rose_util import rose_score
from .rose_config import RoseConfig
from .rose_tensor_bank import RoseTensorBank


class Rose(nn.Module, ABC):
    """
    Abstract ROSE module. Inherits RoseConfig, manages RoseTensorBank for input regulation.
    """
    def __init__(self, inputs: Dict[str, Any], config: Optional[RoseConfig] = None):
        super().__init__()
        self.config = config or RoseConfig()
        self.bank = RoseTensorBank(self._process_inputs(inputs), config=self.config)

    def _process_inputs(self, tensors: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        processed = {}
        for k, v in tensors.items():
            if isinstance(v, torch.Tensor):
                t = v
            elif isinstance(v, (list, tuple)):
                t = torch.tensor(v, dtype=torch.float32)
            elif isinstance(v, (float, int, bool)):
                t = torch.tensor([v], dtype=torch.float32)
            else:
                raise TypeError(f"Unsupported input for '{k}': type={type(v)}")
            processed[k] = t
        return processed

    @abstractmethod
    def forward(self, *args, **kwargs) -> torch.Tensor:
        raise NotImplementedError

    def seed(self):
        if self.config and self.config.seed is not None:
            torch.manual_seed(self.config.seed)

    def cleanup(self):
        self.bank.cleanup()

    def __del__(self):
        self.cleanup()


class Rose1D(Rose):
    def __init__(self, x: Any, config: Optional[RoseConfig] = None):
        super().__init__({"x": x}, config)

    def forward(self) -> torch.Tensor:
        self.seed()
        return self.bank.get_tensor("x")


class Rose2D(Rose):
    def __init__(self, x: Any, need: Any, config: Optional[RoseConfig] = None):
        super().__init__({"x": x, "need": need}, config)

    def forward(self) -> torch.Tensor:
        self.seed()
        return rose_score(
            self.bank.get_tensor("x"),
            self.bank.get_tensor("need"),
            self.bank.get_tensor("need"),
            self.bank.get_tensor("need"),
            config=self.config,
        )


class Rose3D(Rose):
    def __init__(self, x: Any, need: Any, relation: Any, config: Optional[RoseConfig] = None):
        super().__init__({"x": x, "need": need, "relation": relation}, config)

    def forward(self) -> torch.Tensor:
        self.seed()
        return rose_score(
            self.bank.get_tensor("x"),
            self.bank.get_tensor("need"),
            self.bank.get_tensor("relation"),
            self.bank.get_tensor("relation"),
            config=self.config,
        )


class Rose4D(Rose):
    def __init__(self, x: Any, need: Any, relation: Any, purpose: Any, config: Optional[RoseConfig] = None):
        super().__init__({"x": x, "need": need, "relation": relation, "purpose": purpose}, config)

    def forward(self) -> torch.Tensor:
        self.seed()
        x = self.bank.get_tensor("x")
        need = self.bank.get_tensor("need")
        relation = self.bank.get_tensor("relation")
        purpose = self.bank.get_tensor("purpose")
        if self.config.clone_inputs:
            x, need, relation, purpose = x.clone(), need.clone(), relation.clone(), purpose.clone()
        return rose_score(x, need, relation, purpose, config=self.config)


class Rose5D(Rose):
    def __init__(self, x: Any, need: Any, relation: Any, purpose: Any, field: Any, config: Optional[RoseConfig] = None):
        super().__init__({
            "x": x,
            "need": need,
            "relation": relation,
            "purpose": purpose,
            "field": field
        }, config)

    def forward(self) -> torch.Tensor:
        self.seed()
        return rose_score(
            self.bank.get_tensor("x"),
            self.bank.get_tensor("need"),
            self.bank.get_tensor("relation"),
            self.bank.get_tensor("purpose"),
            config=self.config,
            external_field=self.bank.get_tensor("field")
        )
