import torch
import torch.nn as nn
from typing import Union, List, Any
from model_core.framework.automodel import ITokenizer

import logging
logger = logging.getLogger(__name__)


class DummyTokenizer(ITokenizer):
    """Dummy tokenizer that returns tensor based on string length"""

    def tokenize(self, inputs: Union[str, List[str]]) -> torch.Tensor:
        texts = inputs if isinstance(inputs, list) else [inputs]
        return torch.tensor([[len(text)] for text in texts], dtype=torch.long)

    def detokenize(self, tokens: Any) -> Union[str, List[str]]:
        batch_size = tokens.shape[0] if hasattr(tokens, 'shape') else 1
        result = ["<dummy_text>" for _ in range(batch_size)]
        return result[0] if batch_size == 1 else result


class DummyModel(nn.Module):
    """Dummy model that projects input to specified dimension"""

    def __init__(self, out_dim: int = 4):
        super().__init__()
        self.out_dim = out_dim
        self.proj = nn.Linear(1, out_dim, bias=False)
        nn.init.xavier_uniform_(self.proj.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 1:
            x = x.unsqueeze(1)
        return self.proj(x.float())
