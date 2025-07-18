import logging
from model_core.framework.encoder_orchestrator import EncoderOrchestrator

from .bert_encoder import BERTEncoder
from .clip_encoder import CLIPEncoder
from .t5_encoder import T5Encoder

logger = logging.getLogger(__name__)

def register_all_encoders(orch: EncoderOrchestrator, device="cpu", **kwargs):
    """Register all implemented encoders into the given orchestrator."""
    try:
        logger.info("Registering BERT encoder")
        orch.add_encoder("bert", BERTEncoder(name="bert", device=device, **kwargs))
    except Exception as e:
        logger.warning(f"BERT encoder failed to load: {e}")

    try:
        logger.info("Registering CLIP encoder")
        orch.add_encoder("clip", CLIPEncoder(name="clip", device=device, **kwargs))
    except Exception as e:
        logger.warning(f"CLIP encoder failed to load: {e}")

    try:
        logger.info("Registering T5 encoder")
        orch.add_encoder("t5", T5Encoder(name="t5", device=device, **kwargs))
    except Exception as e:
        logger.warning(f"T5 encoder failed to load: {e}")