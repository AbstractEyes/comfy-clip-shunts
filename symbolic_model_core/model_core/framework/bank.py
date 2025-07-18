import torch
import logging
import asyncio
from abc import ABC, abstractmethod
from typing import Dict, Union, Any, Callable, List, Tuple
from model_core.framework.automodel import BaseEncoder

logger = logging.getLogger(__name__)

class AbstractVramBankFirmware(ABC):

    @abstractmethod
    def register_encoder(self, name: str, encoder: BaseEncoder):
        pass

    @abstractmethod
    def can_lend(self, encoder: BaseEncoder) -> bool:
        pass

    @abstractmethod
    def get_total_usage(self, device: Union[str, torch.device]) -> float:
        pass

    @abstractmethod
    def report(self) -> Dict[str, Any]:
        pass

    @abstractmethod
    def print_report(self):
        pass

