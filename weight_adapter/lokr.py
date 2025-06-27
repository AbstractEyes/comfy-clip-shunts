import logging
from typing import Optional, Tuple
import torch
from .base import WeightAdapterBase, register_adapter, make_kron

logger = logging.getLogger(__name__)


@register_adapter
class LoKrAdapter(WeightAdapterBase):
    """Low-rank Kronecker product Adapter (LoKr)"""
    name = "lokr"

    @classmethod
    def is_applicable(cls, lora_key: str, lora: dict[str, torch.Tensor]) -> bool:
        """Check if this is a LoKr layer"""
        return any(f"{lora_key}.{k}" in lora for k in ["lokr_w1", "lokr_w2", "lokr_w1_a", "lokr_w2_a"])

    @classmethod
    def extract_weight(cls, lora_key: str, lora: dict[str, torch.Tensor],
                       alpha: Optional[float], loaded_keys: set[str]) -> Optional[torch.Tensor]:
        """Extract and reconstruct LoKr weight"""
        # Component names
        lokr_w1_name = f"{lora_key}.lokr_w1"
        lokr_w2_name = f"{lora_key}.lokr_w2"
        lokr_w1_a_name = f"{lora_key}.lokr_w1_a"
        lokr_w1_b_name = f"{lora_key}.lokr_w1_b"
        lokr_w2_a_name = f"{lora_key}.lokr_w2_a"
        lokr_w2_b_name = f"{lora_key}.lokr_w2_b"
        lokr_t2_name = f"{lora_key}.lokr_t2"

        # Extract available components
        w1 = lora.get(lokr_w1_name)
        w2 = lora.get(lokr_w2_name)
        w1_a = lora.get(lokr_w1_a_name)
        w1_b = lora.get(lokr_w1_b_name)
        w2_a = lora.get(lokr_w2_a_name)
        w2_b = lora.get(lokr_w2_b_name)
        t2 = lora.get(lokr_t2_name)

        # Check if we have any LoKr components
        if not any([w1, w2, w1_a, w2_a]):
            return None

        # Track loaded keys
        for name, tensor in [(lokr_w1_name, w1), (lokr_w2_name, w2),
                             (lokr_w1_a_name, w1_a), (lokr_w1_b_name, w1_b),
                             (lokr_w2_a_name, w2_a), (lokr_w2_b_name, w2_b),
                             (lokr_t2_name, t2)]:
            if tensor is not None:
                loaded_keys.add(name)

        dim = None

        try:
            import time
            start_time = time.time()

            # Define computation function
            def compute_lokr(w1, w2, w1_a=None, w1_b=None, w2_a=None, w2_b=None, t2=None):
                # Reconstruct w1 if needed
                if w1 is None and w1_a is not None:
                    w1 = torch.mm(w1_a, w1_b)

                # Reconstruct w2 if needed
                if w2 is None and w2_a is not None:
                    if t2 is None:
                        w2 = torch.mm(w2_a, w2_b)
                    else:
                        # Tucker decomposition
                        w2 = torch.einsum('i j k l, j r, i p -> p r k l', t2, w2_b, w2_a)

                # Compute Kronecker product
                return make_kron(w1, w2, scale)

            # Prepare tensors for CUDA computation
            tensors = []
            args = [w1, w2, w1_a, w1_b, w2_a, w2_b, t2]
            for t in args:
                if t is not None:
                    tensors.append(t)

            lora_diff = cls.compute_on_cuda(tensors, lambda *ts: compute_lokr(*args))

            elapsed = time.time() - start_time
            if elapsed > 0.1:  # Log slow operations
                logger.info(
                    f"LoKr {lora_key} took {elapsed:.2f}s, kron size: {w1.shape if w1 is not None else 'decomposed'}")

            return lora_diff

        except Exception as e:
            logger.error(f"Failed to reconstruct LoKr weights for {lora_key}: {e}")
            return None