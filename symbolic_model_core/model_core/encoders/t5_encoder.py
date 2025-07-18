import logging
from model_core.framework.automodel import EncoderMetadata, BaseEncoder, EncoderRegistry
from model_core.implementation.hf_loader import load_hf_encoder, HFTokenizerWrapper

logger = logging.getLogger(__name__)

class T5Encoder(BaseEncoder):
    """HuggingFace T5 encoder model integration."""

    def __init__(self,
                 name: str = "t5",
                 dummy: bool = False,
                 repo_id: str = "t5-small",
                 device: str = "cpu",
                 **kwargs):
        metadata = EncoderMetadata(
            identifier=name,
            encoder_type="text",
            capabilities=["embedding", "symbolic"],
            config={"dummy": dummy, "repo_id": repo_id, **kwargs}
        )
        super().__init__(metadata, device=device)

        if dummy:
            logger.warning("T5Encoder is in dummy mode — not yet implemented.")
            raise NotImplementedError("T5 dummy mode not supported yet.")
        else:
            model, tokenizer = load_hf_encoder(repo_id, device)
            self.set_model(model)
            self.set_tokenizer(HFTokenizerWrapper(tokenizer))
            self._setup_output_hook()

    def _setup_output_hook(self):
        @self.hooks.register_hook("reshape_t5_output")
        def reshape(output):
            if hasattr(output, "last_hidden_state"):
                return output.last_hidden_state.mean(dim=1)
            return output

        self.hooks.add_hook("post_forward", "reshape_t5_output", priority=10)

EncoderRegistry.register("t5", T5Encoder)