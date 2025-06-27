import logging
from typing import Optional, Tuple
import torch
from .base import WeightAdapterBase, register_adapter

logger = logging.getLogger(__name__)


@register_adapter
class LoRAAdapter(WeightAdapterBase):
    """Standard Low-Rank Adaptation (LoRA)"""
    name = "lora"

    @classmethod
    def is_applicable(cls, lora_key: str, lora: dict[str, torch.Tensor]) -> bool:
        """Check if this is a LoRA layer"""
        variants = [
            f"{lora_key}.lora_up.weight",
            f"{lora_key}_lora.up.weight",
            f"{lora_key}.lora_B.weight",
            f"{lora_key}.lora.up.weight",
            f"{lora_key}.lora_B",
            f"{lora_key}.lora_linear_layer.up.weight"
        ]
        return any(v in lora for v in variants)

    @classmethod
    def extract_weight(cls, lora_key: str, lora: dict[str, torch.Tensor],
                       alpha: Optional[float], loaded_keys: set[str]) -> Optional[torch.Tensor]:
        """Extract and reconstruct LoRA weight"""
        # Find which LoRA format variant we have
        variants = [
            # (up_key, down_key, mid_key)
            (f"{lora_key}.lora_up.weight", f"{lora_key}.lora_down.weight", f"{lora_key}.lora_mid.weight"),
            (f"{lora_key}_lora.up.weight", f"{lora_key}_lora.down.weight", None),
            (f"{lora_key}.lora_B.weight", f"{lora_key}.lora_A.weight", None),
            (f"{lora_key}.lora.up.weight", f"{lora_key}.lora.down.weight", None),
            (f"{lora_key}.lora_B", f"{lora_key}.lora_A", None),
            (f"{lora_key}.lora_linear_layer.up.weight", f"{lora_key}.lora_linear_layer.down.weight", None),
        ]

        up_key = down_key = mid_key = None
        for up, down, mid in variants:
            if up in lora:
                up_key, down_key, mid_key = up, down, mid
                break

        if not up_key:
            return None

        # Extract matrices
        mat1 = lora[up_key]
        mat2 = lora[down_key]
        mid = lora.get(mid_key) if mid_key else None

        # Track loaded keys
        loaded_keys.update([up_key, down_key])
        if mid is not None:
            loaded_keys.add(mid_key)

        # Check for reshape info
        reshape_key = f"{lora_key}.reshape_weight"
        if reshape_key in lora:
            loaded_keys.add(reshape_key)
            logger.info(f"LoRA layer {lora_key} has reshape info")

        # Apply alpha scaling
        if alpha is not None:
            alpha = alpha / mat2.shape[0]
        else:
            alpha = 1.0

        try:
            # Apply mid transformation if present (LoCon-style)
            if mid is not None:
                final_shape = [mat2.shape[1], mat2.shape[0], mid.shape[2], mid.shape[3]]
                mat2 = (
                    torch.mm(
                        mat2.transpose(0, 1).flatten(start_dim=1),
                        mid.transpose(0, 1).flatten(start_dim=1),
                    )
                    .reshape(final_shape)
                    .transpose(0, 1)
                )

            # Compute LoRA: up @ down
            lora_diff = torch.mm(mat1.flatten(start_dim=1), mat2.flatten(start_dim=1))
            lora_diff = lora_diff * alpha

            return lora_diff

        except Exception as e:
            logger.error(f"Failed to reconstruct LoRA weights for {lora_key}: {e}")
            return None