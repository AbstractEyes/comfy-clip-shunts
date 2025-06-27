import logging
from typing import Optional
import torch
from .base import WeightAdapterBase, register_adapter

logger = logging.getLogger(__name__)


@register_adapter
class LycorisAdapter(WeightAdapterBase):
    """Generic Lycoris adapter - appears to be LoHA format"""
    name = "lycoris"

    @classmethod
    def is_applicable(cls, lora_key: str, lora: dict[str, torch.Tensor]) -> bool:
        """Check if this is a generic Lycoris layer (currently just LoHA format)"""
        # This is essentially checking for LoHA format
        # Consider if this adapter is needed or if LoHA should handle it
        return (f"{lora_key}.hada_w1_a" in lora and
                f"{lora_key}.hada_w1_b" in lora and
                f"{lora_key}.hada_w2_a" in lora and
                f"{lora_key}.hada_w2_b" in lora)

    @classmethod
    def extract_weight(cls, lora_key: str, lora: dict[str, torch.Tensor],
                       alpha: Optional[float], loaded_keys: set[str]) -> Optional[torch.Tensor]:
        """Extract weight - currently identical to LoHA"""
        # Component names (LoHA format)
        w1a_name = f"{lora_key}.hada_w1_a"
        w1b_name = f"{lora_key}.hada_w1_b"
        w2a_name = f"{lora_key}.hada_w2_a"
        w2b_name = f"{lora_key}.hada_w2_b"
        t1_name = f"{lora_key}.hada_t1"
        t2_name = f"{lora_key}.hada_t2"

        # Check components exist
        if not all(k in lora for k in [w1a_name, w1b_name, w2a_name, w2b_name]):
            return None

        # Extract components
        w1a = lora[w1a_name]
        w1b = lora[w1b_name]
        w2a = lora[w2a_name]
        w2b = lora[w2b_name]
        t1 = lora.get(t1_name)
        t2 = lora.get(t2_name)

        # Track loaded keys
        loaded_keys.update([w1a_name, w1b_name, w2a_name, w2b_name])
        if t1 is not None:
            loaded_keys.update([t1_name, t2_name])

        # Apply alpha scaling
        if alpha is not None:
            alpha = alpha / w1b.shape[0]
        else:
            alpha = 1.0

        try:
            if t1 is not None and t2 is not None:
                # CP decomposition
                m1 = torch.einsum('i j k l, j r, i p -> p r k l', t1, w1b, w1a)
                m2 = torch.einsum('i j k l, j r, i p -> p r k l', t2, w2b, w2a)
            else:
                # Standard
                m1 = torch.mm(w1a, w1b)
                m2 = torch.mm(w2a, w2b)

            # Hadamard product
            lora_diff = (m1 * m2) * alpha

            return lora_diff

        except Exception as e:
            logger.error(f"Failed to reconstruct Lycoris weights for {lora_key}: {e}")
            return None