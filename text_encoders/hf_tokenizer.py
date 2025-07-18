import torch
from typing import Union, List, Any
from .automodel import ITokenizer  # or adjust import as needed


class HFTokenizerWrapper(ITokenizer):
    """
    Adapter to use HuggingFace tokenizers in the ITokenizer interface.
    Provides tokenize and detokenize operations.
    """

    def __init__(self, tokenizer, max_length: int = 512):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def tokenize(self, inputs: Union[str, List[str]]) -> dict:
        return self.tokenizer(
            inputs,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length
        )

    def detokenize(self, tokens: Any) -> Union[str, List[str]]:
        if isinstance(tokens, dict) and "input_ids" in tokens:
            tokens = tokens["input_ids"]
        return self.tokenizer.batch_decode(tokens, skip_special_tokens=True)
