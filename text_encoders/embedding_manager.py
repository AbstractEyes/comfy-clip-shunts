

import comfy
import logging
import torch
import huggingface_hub

from transformers import CLIPTokenizer
from clip_model import CLIPEncoder

logger = logging.getLogger(__name__)


class EmbeddingManager:
    """
        This houses the behavioral controllers meant to represent complex embeddings.
        These embeddings are concatenated and preformatted to be used in any conditioning generation.
    """

    def __init__(self):
        self.embeddings = {}
        self.active_embedding = None
        self.embedding_format = "float32"

    def learn_embedding(self,
                        tokenizer: CLIPTokenizer,
                        encoder: CLIPEncoder,
                        prompt_trigger: str,
                        prompt_sequence: list[str]):
        """
        Tokenizes the prompt_sequence, generates embeddings, normalizes them,
        and stores the result under the given prompt_trigger key.

        Args:
            prompt_trigger (str): The alias token name (e.g., "masterpiece")
            prompt_sequence (list[str]): List of token strings to expand (e.g., ["beautiful", "sharp"])
        """
        # Join the sequence into a space-separated string for tokenization
        prompt_str = " ".join(prompt_sequence)
        tokenized = tokenizer(prompt_str, return_tensors="pt", add_special_tokens=False)
        token_ids = tokenized["input_ids"].to(encoder.device)  # Shape: [1, N]

        with torch.no_grad():
            embedding = encoder(token_ids)  # Expected shape: [1, N, D]

        if embedding.dim() == 3 and embedding.shape[0] == 1:
            embedding = embedding.squeeze(0)  # Shape: [N, D]

        embedding = embedding.to(self.embedding_format)
        embedding = self.normalize_embedding(prompt_trigger, embedding)
        self.embeddings[prompt_trigger] = embedding

    def get_embedding(self, prompt_trigger: str):
        """
        Retrieves the embedding corresponding to a given alias trigger.
        """
        embedding = self.embeddings.get(prompt_trigger)
        if embedding is None:
            logger.warning(f"Embedding for '{prompt_trigger}' not found.")
        return embedding

    def impose_embeddings(self, token_sequence, embedding_to_change) -> torch.Tensor:
        # this method applies the embedding to the token sequence
        if token_sequence in self.embeddings:
            embedding = self.embeddings[token_sequence]
            if embedding_to_change is not None:
                embedding = embedding + embedding_to_change
            return self.normalize_embedding(token_sequence, embedding)
        else:
            logger.warning(f"Embedding for '{token_sequence}' not found, returning unchanged.")
            return embedding_to_change if embedding_to_change is not None else None

    def normalize_embedding(self, token_sequence, embedding):
        if embedding.ndim == 2:
            # Per-token normalization: each [D] vector gets unit norm
            return torch.nn.functional.normalize(embedding, p=2, dim=-1)
        elif embedding.ndim == 1:
            return embedding / embedding.norm()
        else:
            logger.warning(f"Embedding for '{token_sequence}' has unsupported shape {embedding.shape}")
            return embedding
