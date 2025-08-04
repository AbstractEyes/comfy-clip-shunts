# vocab_mapper.py
from typing import Dict, List, Tuple, Optional
import heapq

class WeightedVocabMap:
    """
    A simple bidirectional weighted mapping between vocabularies.
    Supports asymmetric and normalized weights.
    """
    def __init__(self):
        self.forward: Dict[str, Dict[str, float]] = {}
        self.reverse: Dict[str, Dict[str, float]] = {}

    def add(self, source: str, target: str, weight: float = 1.0):
        """Add a weighted connection from source → target."""
        self.forward.setdefault(source, {})[target] = weight
        self.reverse.setdefault(target, {})[source] = weight

    def add_many(self, pairs: List[Tuple[str, str, float]]):
        """Bulk-add multiple (source, target, weight) entries."""
        for source, target, weight in pairs:
            self.add(source, target, weight)

    def normalize(self, *, method: str = "forward", p: float = 1.0):
        """
        Normalize weights so each source maps to targets summing to 1.
        Options:
            - method='forward': normalize source → target
            - method='reverse': normalize target → source
        """
        if method == "forward":
            for src, tgts in self.forward.items():
                total = sum(weight ** p for weight in tgts.values())
                if total > 0:
                    for tgt in tgts:
                        self.forward[src][tgt] /= total
        elif method == "reverse":
            for tgt, srcs in self.reverse.items():
                total = sum(weight ** p for weight in srcs.values())
                if total > 0:
                    for src in srcs:
                        self.reverse[tgt][src] /= total
        else:
            raise ValueError("Normalization method must be 'forward' or 'reverse'.")

    def get_targets(self, source: str, top_k: Optional[int] = None) -> List[Tuple[str, float]]:
        """Return sorted (target, weight) list for a given source."""
        mapping = self.forward.get(source, {})
        sorted_items = sorted(mapping.items(), key=lambda x: -x[1])
        return sorted_items[:top_k] if top_k else sorted_items

    def get_sources(self, target: str, top_k: Optional[int] = None) -> List[Tuple[str, float]]:
        """Return sorted (source, weight) list for a given target."""
        mapping = self.reverse.get(target, {})
        sorted_items = sorted(mapping.items(), key=lambda x: -x[1])
        return sorted_items[:top_k] if top_k else sorted_items

    def merge(self, other: "WeightedVocabMap", weight_scale: float = 1.0):
        """Merge another map into this one with an optional weight scale."""
        for src, tgts in other.forward.items():
            for tgt, weight in tgts.items():
                self.add(src, tgt, weight * weight_scale)

    def prune(self, min_weight: float = 0.01):
        """Remove mappings below a weight threshold."""
        self.forward = {
            src: {tgt: w for tgt, w in tgts.items() if w >= min_weight}
            for src, tgts in self.forward.items()
            if any(w >= min_weight for w in tgts.values())
        }
        self.reverse = {
            tgt: {src: w for src, w in srcs.items() if w >= min_weight}
            for tgt, srcs in self.reverse.items()
            if any(w >= min_weight for w in srcs.values())
        }

    def as_dict(self) -> Dict[str, Dict[str, float]]:
        return self.forward

    def as_reverse_dict(self) -> Dict[str, Dict[str, float]]:
        return self.reverse
