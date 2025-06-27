import logging
from typing import Optional
import torch
from .base import WeightAdapterBase, register_adapter

logger = logging.getLogger(__name__)


@register_adapter
class BLoRAAdapter(WeightAdapterBase):
    """Block-wise LoRA (BLoRA) with optional gating"""
    name = "blora"

    @classmethod
    def is_applicable(cls, lora_key: str, lora: dict[str, torch.Tensor]) -> bool:
        """Check if this is a BLoRA layer"""
        # Must have both up/down and block_gate to distinguish from regular LoRA
        return (f"{lora_key}.lora_up.weight" in lora and
                f"{lora_key}.lora_down.weight" in lora and
                f"{lora_key}.block_gate" in lora)

    @classmethod
    def extract_weight(cls, lora_key: str, lora: dict[str, torch.Tensor],
                       alpha: Optional[float], loaded_keys: set[str]) -> Optional[torch.Tensor]:
        """Extract and reconstruct BLoRA weight"""
        up_name = f"{lora_key}.lora_up.weight"
        down_name = f"{lora_key}.lora_down.weight"
        gate_name = f"{lora_key}.block_gate"

        if up_name not in lora or down_name not in lora:
            return None

        # Extract components
        w_up = lora[up_name]
        w_down = lora[down_name]
        block_gate = lora.get(gate_name)

        # Track loaded keys
        loaded_keys.update([up_name, down_name])
        if block_gate is not None:
            loaded_keys.add(gate_name)

        # Apply alpha scaling
        rank = w_down.shape[0]
        scale = (alpha / rank) if alpha is not None else (1.0 / rank)

        try:
            # Compute LoRA: up @ down
            lora_diff = torch.matmul(w_up.flatten(1), w_down.flatten(1))
            lora_diff = lora_diff * scale

            # Apply block gate if present
            if block_gate is not None:
                lora_diff = lora_diff * block_gate.flatten()

            return lora_diff

        except Exception as e:
            logger.error(f"Failed to reconstruct BLoRA weights for {lora_key}: {e}")
            return None