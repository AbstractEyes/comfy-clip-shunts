import os
import gc
from dataclasses import dataclass
from enum import Enum

import torch


import logging
logger = logging.getLogger(__name__)


class LoaderType(Enum):
    T5 = "t5"
    BERT = "bert"
    CLIP = "clip"
    SHUNT = "shunt"
    TOKENIZER = "tokenizer"
    CUSTOM = "custom"

class LoadMode(Enum):
    FULL = "full"           # Load directly to GPU
    OFFLOAD = "offload"     # CPU/GPU offload (via accelerate)
    CPU_ONLY = "cpu"        # CPU-safe fallback
    LOW_MEM = "low_mem"     # HF-native low-CPU init


@dataclass
class LoadedObject:
    name: str
    type: LoaderType
    model: PreTrainedModel
    tokenizer: PreTrainedTokenizer
    device: torch.device
    dtype: torch.dtype
    load_mode: LoadMode
    size_estimate_mb: int = -1


class UniversalModelLoader:
    def __init__(self, device=None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.cache = {}

    def _log(self, msg):
        logger.info(f"[UniversalModelLoader] {msg}")

    def is_loaded(self, name: str) -> bool:
        return name in self.cache

    def get(self, name: str) -> LoadedObject:
        return self.cache[name]

    def unload(self, name: str) -> bool:
        if name not in self.cache:
            return False
        obj = self.cache[name]
        try:
            del obj.model
            if obj.tokenizer:
                del obj.tokenizer
            del self.cache[name]
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return True
        except Exception as e:
            logger.error(f"Failed to unload {name}: {e}")
            return False
