# vocab/vocab_heat_cache.py

import torch
from typing import List, Dict, Optional, Callable, Union

from .weighted_map import WeightedVocabMap


class VocabHeatCache:
    """
    Caches vocab embeddings and computes token-wise similarity masks
    for heat-based alpha substitution blending.
    """

    def __init__(
        self,
        base_vocab: List[str],
        base_embeddings: torch.Tensor,
        sim_fn: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None
    ):
        self.base_vocab = base_vocab
        self.base_embeddings = base_embeddings  # shape: [V, D]
        self.sim_fn = sim_fn or self.default_similarity

        self._index: Dict[str, int] = {tok: i for i, tok in enumerate(base_vocab)}
        self._map_cache: Dict[str, WeightedVocabMap] = {}

    @staticmethod
    def default_similarity(
        a: torch.Tensor,
        b: torch.Tensor
    ) -> torch.Tensor:
        """Default similarity: cosine dot product."""
        a_norm = a / a.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        b_norm = b / b.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        return torch.matmul(a_norm, b_norm.T)  # [V_base, V_other]

    def compare_vocab(
        self,
        other_vocab: List[str],
        other_embeddings: torch.Tensor,
        tau: float = 1.0,
        flat_strength: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None
    ) -> WeightedVocabMap:
        """
        Generate similarity map from other_vocab → base_vocab.
        Output is a WeightedVocabMap usable for alpha-masking.
        """
        sim = self.sim_fn(other_embeddings, self.base_embeddings)  # [V_other, V_base]

        sim = sim / tau
        weights = torch.softmax(sim, dim=-1) * flat_strength  # [V_other, V_base]

        vocab_map = WeightedVocabMap()
        for i, source_token in enumerate(other_vocab):
            tgt_weights = weights[i]
            scored = list(zip(self.base_vocab, tgt_weights.tolist()))

            if top_k is not None:
                scored = sorted(scored, key=lambda x: -x[1])[:top_k]

            if top_p is not None:
                sorted_scored = sorted(scored, key=lambda x: -x[1])
                cumulative, filtered = 0.0, []
                for token, score in sorted_scored:
                    cumulative += score
                    if cumulative <= top_p:
                        filtered.append((token, score))
                    else:
                        break
                scored = filtered

            for target_token, weight in scored:
                vocab_map.add(source_token, target_token, weight)

        return vocab_map

    def get_heat_map(
        self,
        other_vocab: List[str],
        other_embeddings: torch.Tensor,
        cache_key: Optional[str] = None,
        tau: float = 1.0,
        flat_strength: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None
    ) -> WeightedVocabMap:
        """
        Retrieve or compute a heat map for another vocab.
        Caches results if a cache_key is provided.
        """
        if cache_key and cache_key in self._map_cache:
            return self._map_cache[cache_key]

        heat_map = self.compare_vocab(
            other_vocab=other_vocab,
            other_embeddings=other_embeddings,
            tau=tau,
            flat_strength=flat_strength,
            top_k=top_k,
            top_p=top_p
        )

        if cache_key:
            self._map_cache[cache_key] = heat_map

        return heat_map

    def get_token_mask(
        self,
        source_token: str,
        heat_map: WeightedVocabMap
    ) -> Dict[str, float]:
        """Return the alpha mask for substituting a token across the base vocab."""
        return heat_map.get_targets(source_token)
