import logging
from typing import Union, List, Any, Dict
import torch
from model_core.framework.automodel import ITokenizer


import logging
logger = logging.getLogger(__name__)



def load_hf_encoder(repo_id: str, device: str = "cpu") -> tuple:
    """Load a HuggingFace model and tokenizer"""
    try:
        from transformers import AutoModel, AutoTokenizer

        logger.info(f"Loading HuggingFace model: {repo_id}")
        model = AutoModel.from_pretrained(
            repo_id,
            device_map={"": device},
            trust_remote_code=True
        )
        tokenizer = AutoTokenizer.from_pretrained(repo_id, trust_remote_code=True)
        return model, tokenizer

    except ImportError:
        logger.error("transformers library not installed")
        raise
    except Exception as e:
        logger.error(f"Failed to load model {repo_id}: {e}")
        raise


class HFTokenizerWrapper(ITokenizer):
    """Wrapper to adapt HuggingFace tokenizer to our interface"""

    def __init__(self, hf_tokenizer):
        self.tokenizer = hf_tokenizer

    def tokenize(self, inputs: Union[str, List[str]]) -> Dict[str, torch.Tensor]:
        return self.tokenizer(
            inputs,
            padding=True,
            truncation=True,
            return_tensors="pt"
        )

    def detokenize(self, tokens: Any) -> Union[str, List[str]]:
        if isinstance(tokens, dict):
            tokens = tokens.get('input_ids', tokens)
        decoded = self.tokenizer.batch_decode(tokens, skip_special_tokens=True)
        return decoded[0] if len(decoded) == 1 else decoded
