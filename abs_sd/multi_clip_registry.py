import torch
import logging
from typing import Optional, Dict

logger = logging.getLogger(__name__)


def resolve_clip_type(name: str) -> str:
    """Maps known clip submodule names to canonical clip types."""
    name = name.lower()
    if "clip_l" in name:
        return "clip_l"
    elif "clip_g" in name:
        return "clip_g"
    elif "clip_h" in name:
        return "clip_h"
    elif "t5" in name:
        return "t5"
    elif "llama" in name:
        return "llama"
    elif "base" in name:
        return "clip_base"
    else:
        return f"unknown_{name}"


class MultiClipEntry:
    def __init__(
        self,
        model: torch.nn.Module,
        name: str,
        encoder_type: str = "clip",
        clip_type: str = "clip_l",
        dtype: torch.dtype = torch.float32,
        tokenizer=None,
        source: str = "unknown",
        patcher=None,
        device_info: Optional[dict] = None,
        config: Optional[dict] = None,
    ):
        self.model = model
        self.name = name
        self.encoder_type = encoder_type
        self.clip_type = clip_type
        self.dtype = dtype
        self.tokenizer = tokenizer
        self.source = source
        self.patcher = patcher
        self.device_info = device_info or {}
        self.config = config or {}

    def to_cliprouter_dict(self) -> dict:
        return {self.clip_type: self.model}

    def summary(self):
        return {
            "name": self.name,
            "encoder_type": self.encoder_type,
            "clip_type": self.clip_type,
            "dtype": str(self.dtype),
            "source": self.source
        }


class MultiClipRegistry:
    def __init__(self):
        self.entries: Dict[str, MultiClipEntry] = {}

    def add_entry(self, entry: MultiClipEntry):
        if entry.clip_type in self.to_cliprouter_dict():
            logger.warning(f"[MultiClipRegistry] Duplicate clip type '{entry.clip_type}' detected.")
        self.entries[entry.name] = entry

    def get_by_clip_type(self, clip_type: str) -> Optional[MultiClipEntry]:
        for entry in self.entries.values():
            if entry.clip_type == clip_type:
                return entry
        return None

    def to_cliprouter_dict(self) -> Dict[str, torch.nn.Module]:
        return {entry.clip_type: entry.model for entry in self.entries.values()}

    def summary(self):
        return {name: entry.summary() for name, entry in self.entries.items()}

    @staticmethod
    def extract_from_comfy(comfy_clip) -> Dict[str, MultiClipEntry]:
        registry = {}
        cond_stage = comfy_clip.cond_stage_model
        tokenizer = getattr(comfy_clip, "tokenizer", None)
        patcher = getattr(comfy_clip, "patcher", None)

        for name in dir(cond_stage):
            attr = getattr(cond_stage, name)
            if isinstance(attr, torch.nn.Module) and not name.startswith("_"):
                if any(key in name.lower() for key in ["clip", "t5", "vision", "llm"]):
                    try:
                        dtype = next(attr.parameters(), torch.tensor([])).dtype
                    except StopIteration:
                        dtype = torch.float32

                    entry = MultiClipEntry(
                        model=attr,
                        name=name,
                        encoder_type="clip",  # can be updated externally
                        clip_type=resolve_clip_type(name),
                        dtype=dtype,
                        tokenizer=tokenizer,
                        patcher=patcher,
                        source="comfy_submodule",
                        device_info={
                            "load_device": getattr(patcher, "load_device", None) if patcher else None,
                            "offload_device": getattr(patcher, "offload_device", None) if patcher else None,
                        },
                        config={"has_tokenizer": tokenizer is not None}
                    )
                    registry[name] = entry

        if not registry:
            try:
                fallback_dtype = next(cond_stage.parameters()).dtype
            except StopIteration:
                fallback_dtype = torch.float32

            fallback = MultiClipEntry(
                model=cond_stage,
                name="base_clip",
                encoder_type="clip",
                clip_type="clip_base",
                dtype=fallback_dtype,
                tokenizer=tokenizer,
                patcher=patcher,
                source="fallback"
            )
            registry["base_clip"] = fallback

        return registry


# --- Form a registry from clips ---
def form_clip(model_type, clip_a, clip_b=None, clip_c=None, clip_d=None, clip_e=None, clip_f=None):
    all_clips = [clip_a, clip_b, clip_c, clip_d, clip_e, clip_f]
    registry = MultiClipRegistry()

    for clip in all_clips:
        if clip is None:
            continue
        reg_dict = MultiClipRegistry.extract_from_comfy(clip)
        for entry in reg_dict.values():
            registry.add_entry(entry)

    return (
        registry.entries,
        registry.to_cliprouter_dict(),
        {"model_type": model_type, "registered_clips": registry.summary()}
    )
