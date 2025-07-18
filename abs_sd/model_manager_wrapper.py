# model_manager_wrapper.py
import torch
from typing import Dict, Optional
from .multi_clip_registry import MultiClipEntry
from .sd import CLIP as ComfyCLIP
from comfy import model_management
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class ExtendedModelManager:
    def __init__(self):
        self.cache: Dict[str, MultiClipEntry] = {}
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _make_key(self, model_id: str, clip_type: str) -> str:
        return f"{clip_type}::{model_id}"

    def is_cached(self, model_id: str, clip_type: str) -> bool:
        return self._make_key(model_id, clip_type) in self.cache

    def get_cached(self, model_id: str, clip_type: str) -> Optional[MultiClipEntry]:
        return self.cache.get(self._make_key(model_id, clip_type), None)

    def load_abs_clip(
        self,
        model_id: str,
        model_path: str,
        clip_type: str = "clip_l",
        dtype: torch.dtype = torch.float16,
        tokenizer: Optional[any] = None,
    ) -> MultiClipEntry:
        key = self._make_key(model_id, clip_type)
        if key in self.cache:
            logger.info(f"[ExtendedModelManager] Reusing cached model: {key}")
            return self.cache[key]

        logger.info(f"[ExtendedModelManager] Loading model from: {model_path}")
        full_path = Path(model_path).expanduser().absolute()
        if not full_path.exists():
            raise FileNotFoundError(f"Model file does not exist: {full_path}")

        model_sd = torch.load(str(full_path), map_location="cpu")
        model = torch.nn.Module()  # TODO: Replace with real init
        model.load_state_dict(model_sd, strict=False)
        model.to(dtype=dtype, device=self.device)

        entry = MultiClipEntry(
            model=model,
            name=model_id,
            encoder_type="clip",
            clip_type=clip_type,
            dtype=dtype,
            tokenizer=tokenizer,
            source=str(full_path),
            device_info={
                "load_device": self.device,
                "offload_device": model_management.text_encoder_offload_device(),
            },
            config={"expected_dtype": str(dtype)},
        )
        self.cache[key] = entry
        return entry


# Singleton pattern
_GLOBAL_EXTENDED_MANAGER = None


def get_extended_model_manager() -> ExtendedModelManager:
    global _GLOBAL_EXTENDED_MANAGER
    if _GLOBAL_EXTENDED_MANAGER is None:
        _GLOBAL_EXTENDED_MANAGER = ExtendedModelManager()
    return _GLOBAL_EXTENDED_MANAGER
