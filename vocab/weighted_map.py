from typing import Dict, List, Tuple, Optional, Set, Callable
import torch


class WeightedVocabMap:
    """
    Relational weighted mapping between vocabularies.
    Stores directed weights from source to target and exposes bidirectional access.
    Maintains token set for inspection and embedding alignment.
    """

    def __init__(self):
        self._forward: Dict[str, Dict[str, float]] = {}
        self._reverse: Dict[str, Dict[str, float]] = {}
        self._token_set: Set[str] = set()
        self._special_tokens: Set[str] = set()

    def add(self, source: str, target: str, weight: float = 1.0):
        self._forward.setdefault(source, {})[target] = weight
        self._reverse.setdefault(target, {})[source] = weight
        self._token_set.update([source, target])

    def add_special_token(self, token: str):
        self._special_tokens.add(token)
        self._token_set.add(token)

    def add_many(self, links: List[Tuple[str, str, float]]):
        for source, target, weight in links:
            self.add(source, target, weight)

    def normalize(self, direction: str = "forward", power: float = 1.0):
        def _normalize_map(mapping: Dict[str, Dict[str, float]]):
            for k, inner in mapping.items():
                total = sum(v ** power for v in inner.values())
                if total > 0:
                    for subk in inner:
                        inner[subk] /= total

        if direction == "forward":
            _normalize_map(self._forward)
        elif direction == "reverse":
            _normalize_map(self._reverse)
        else:
            raise ValueError("direction must be 'forward' or 'reverse'")

    def targets_for(self, source: str, top_k: Optional[int] = None) -> List[Tuple[str, float]]:
        items = self._forward.get(source, {})
        return sorted(items.items(), key=lambda x: -x[1])[:top_k] if top_k else sorted(items.items(), key=lambda x: -x[1])

    def sources_for(self, target: str, top_k: Optional[int] = None) -> List[Tuple[str, float]]:
        items = self._reverse.get(target, {})
        return sorted(items.items(), key=lambda x: -x[1])[:top_k] if top_k else sorted(items.items(), key=lambda x: -x[1])

    def merge(self, other: "WeightedVocabMap", scale: float = 1.0):
        for source, tgts in other._forward.items():
            for target, weight in tgts.items():
                self.add(source, target, weight * scale)
        self._token_set.update(other._token_set)
        self._special_tokens.update(other._special_tokens)

    def prune(self, threshold: float = 0.01):
        def _prune_map(mapping: Dict[str, Dict[str, float]]) -> Dict[str, Dict[str, float]]:
            return {
                k: {ik: iv for ik, iv in v.items() if iv >= threshold}
                for k, v in mapping.items()
                if any(iv >= threshold for iv in v.values())
            }
        self._forward = _prune_map(self._forward)
        self._reverse = _prune_map(self._reverse)
        self._token_set = set(self._forward.keys()) | set(self._reverse.keys()) | self._special_tokens

    def forward(self) -> Dict[str, Dict[str, float]]:
        return self._forward

    def reverse(self) -> Dict[str, Dict[str, float]]:
        return self._reverse

    def items(self) -> List[Tuple[str, Dict[str, float]]]:
        return list(self._forward.items())

    def reverse_items(self) -> List[Tuple[str, Dict[str, float]]]:
        return list(self._reverse.items())

    def tokens(self) -> List[str]:
        return sorted(self._token_set)

    def special_tokens(self) -> List[str]:
        return sorted(self._special_tokens)

    def __contains__(self, source: str) -> bool:
        return source in self._forward

    def __getitem__(self, source: str) -> Dict[str, float]:
        return self._forward.get(source, {})

    def __len__(self) -> int:
        return len(self._forward)

    def apply_alpha(
        self,
        vectors: torch.Tensor,
        source_token: str,
        token_to_index: Dict[str, int]
    ) -> torch.Tensor:
        """
        Apply alpha-masked token interpolation to a vector or batch of vectors.
        vectors: Tensor of shape [D] or [B, D].
        source_token: the token whose substitutions will be blended.
        token_to_index: maps token string to vector index.
        Returns new vector(s) as weighted sum of substitutes.
        """
        mapping = self._forward.get(source_token, None)
        if not mapping:
            return vectors.detach().clone().contiguous()

        is_batch = vectors.dim() == 2
        total = torch.zeros_like(vectors)

        for token, alpha in mapping.items():
            if token not in token_to_index:
                continue
            idx = token_to_index[token]
            if is_batch:
                ref_vec = vectors[:, idx]
                if ref_vec.dim() == 1:
                    ref_vec = ref_vec.unsqueeze(1)
                ref_vec = ref_vec.expand_as(total)
            else:
                ref_vec = vectors[idx]
            total += float(alpha) * ref_vec.detach().clone().contiguous()

        return total.detach().clone().contiguous()

    @classmethod
    def from_vocab_similarity(
        cls,
        source_vocab: List[str],
        source_embeddings: torch.Tensor,
        target_vocab: List[str],
        target_embeddings: torch.Tensor,
        tau: float = 1.0,
        flat_strength: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        sim_fn: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None
    ) -> "WeightedVocabMap":
        """
        Constructs a WeightedVocabMap from two vocab+embedding sources.
        Precomputes similarity and stores as directional mapping.
        """
        sim_fn = sim_fn or cls._default_cosine
        sim = sim_fn(source_embeddings, target_embeddings) / tau
        weights = torch.softmax(sim, dim=-1) * flat_strength

        inst = cls()
        for i, src_token in enumerate(source_vocab):
            row = weights[i]
            scored = list(zip(target_vocab, row.tolist()))

            if top_k is not None:
                scored = sorted(scored, key=lambda x: -x[1])[:top_k]
            if top_p is not None:
                sorted_scored = sorted(scored, key=lambda x: -x[1])
                cumulative, filtered = 0.0, []
                for tok, score in sorted_scored:
                    cumulative += score
                    if cumulative <= top_p:
                        filtered.append((tok, score))
                    else:
                        break
                scored = filtered

            for tgt_token, score in scored:
                inst.add(src_token, tgt_token, float(score))

        return inst

    @staticmethod
    def _default_cosine(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        a = a / a.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        b = b / b.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        return torch.matmul(a, b.T)
