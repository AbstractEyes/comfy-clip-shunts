import torch
import torch.nn as nn
from transformers import T5Tokenizer, T5EncoderModel
from typing import Union
import logging

from .automodel import (
    BaseEncoder,
    EncoderMetadata,
    HookStage,
    EncoderRegistry,
)
from .hf_tokenizer import HFTokenizerWrapper

logger = logging.getLogger(__name__)


class T5TextEncoder(BaseEncoder):
    """T5 Encoder with lazy-loading, registered into hook-based execution system"""

    def __init__(self, name: str = "t5-small", device: Union[str, torch.device] = "cpu"):
        metadata = EncoderMetadata(
            identifier=name,
            encoder_type="text/t5",
            capabilities=["text_embedding", "seq2seq"],
            config={"model": name}
        )
        super().__init__(metadata, device)

        self._repo_id = name
        self._loaded = False

        # Defer actual loading
        self._setup_hooks()

    def _load_model_and_tokenizer(self):
        if self._loaded:
            return

        logger.info(f"[T5TextEncoder] Loading {self._repo_id} lazily")
        tokenizer = T5Tokenizer.from_pretrained(self._repo_id)
        model = T5EncoderModel.from_pretrained(self._repo_id)

        self.set_model(model.to(self.device))
        self.set_tokenizer(HFTokenizerWrapper(tokenizer))
        self._loaded = True

    def _setup_hooks(self):
        @self.hooks.register_hook("lazy_load_t5")
        def lazy_loader(data):
            self._load_model_and_tokenizer()
            return data

        @self.hooks.register_hook("reshape_output")
        def reshape_t5(output):
            if hasattr(output, 'last_hidden_state'):
                return output.last_hidden_state.mean(dim=1)
            return output

        # Trigger lazy load in pre-forward or pre-load, depending on need
        self.hooks.add_hook(HookStage.PRE_FORWARD, "lazy_load_t5", priority=100)
        self.hooks.add_hook(HookStage.PRE_LOAD, "lazy_load_t5", priority=100)

        # Add output reshape
        self.hooks.add_hook(HookStage.POST_FORWARD, "reshape_output", priority=10)


# Register into global encoder registry
EncoderRegistry.register("t5", T5TextEncoder)

#from text_encoders.loader import TextEncoder

# Create instance using real model
enc = TextEncoder(
    name="t5-small",
    dummy=False,
    repo_id="google-t5/t5-small",
    device="cuda" if torch.cuda.is_available() else "cpu"
)
