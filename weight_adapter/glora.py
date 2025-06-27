import logging
from typing import Optional, Tuple
import torch
from .base import WeightAdapterBase, register_adapter

logger = logging.getLogger(__name__)


@register_adapter
class GLoRAAdapter(WeightAdapterBase):
    """Generalized Low-Rank Adaptation (GLoRA)"""
    name = "glora"

    @classmethod
    def is_applicable(cls, lora_key: str, lora: dict[str, torch.Tensor]) -> bool:
        """Check if this is a GLoRA layer"""
        return f"{lora_key}.a1.weight" in lora

    @classmethod
    def extract_weight(cls, lora_key: str, lora: dict[str, torch.Tensor],
                       alpha: Optional[float], loaded_keys: set[str]) -> Optional[torch.Tensor]:
        """Extract GLoRA components - cannot fully pre-calculate as it needs original weight"""
        # Component names
        a1_name = f"{lora_key}.a1.weight"
        a2_name = f"{lora_key}.a2.weight"
        b1_name = f"{lora_key}.b1.weight"
        b2_name = f"{lora_key}.b2.weight"

        if a1_name not in lora:
            return None

        # Extract components
        a1 = lora[a1_name]
        a2 = lora[a2_name]
        b1 = lora[b1_name]
        b2 = lora[b2_name]

        # Track loaded keys
        loaded_keys.update([a1_name, a2_name, b1_name, b2_name])

        # Detect old vs new GLoRA format
        old_glora = False
        if b2.shape[1] == b1.shape[0] == a1.shape[0] == a2.shape[1]:
            rank = a1.shape[0]
            old_glora = True
        elif b2.shape[0] == b1.shape[1] == a1.shape[1] == a2.shape[0]:
            rank = a2.shape[0]
            old_glora = False
        else:
            logger.error(f"Invalid GLoRA shapes for {lora_key}")
            return None

        # Apply alpha scaling
        if alpha is not None:
            alpha = alpha / rank
        else:
            alpha = 1.0

        try:
            # Pre-calculate the b1 @ b2 part
            if old_glora:
                b_product = torch.mm(b2.flatten(start_dim=1), b1.flatten(start_dim=1))
            else:
                b_product = torch.mm(b1.flatten(start_dim=1), b2.flatten(start_dim=1))

            b_product = b_product * alpha

            # GLoRA needs original weight, so we can only return the b_product part
            # The a1/a2 part must be calculated at runtime
            logger.warning(f"GLoRA adapter {lora_key} requires runtime weight access - returning b_product only")
            return b_product

        except Exception as e:
            logger.error(f"Failed to pre-calculate GLoRA weights for {lora_key}: {e}")
            return None