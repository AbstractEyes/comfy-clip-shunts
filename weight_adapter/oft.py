import logging
from typing import Optional
import torch
from .base import WeightAdapterBase, register_adapter

logger = logging.getLogger(__name__)


@register_adapter
class OFTAdapter(WeightAdapterBase):
    """Orthogonal Fine-Tuning (OFT) Adapter"""
    name = "oft"

    @classmethod
    def is_applicable(cls, lora_key: str, lora: dict[str, torch.Tensor]) -> bool:
        """Check if this is an OFT layer"""
        blocks_name = f"{lora_key}.oft_blocks"
        return blocks_name in lora and lora[blocks_name].ndim == 3

    @classmethod
    def extract_weight(cls, lora_key: str, lora: dict[str, torch.Tensor],
                       alpha: Optional[float], loaded_keys: set[str]) -> Optional[torch.Tensor]:
        """OFT cannot be pre-calculated as it requires the original weight"""
        blocks_name = f"{lora_key}.oft_blocks"
        rescale_name = f"{lora_key}.rescale"

        if blocks_name not in lora or lora[blocks_name].ndim != 3:
            return None

        # Track loaded keys
        loaded_keys.add(blocks_name)
        if rescale_name in lora:
            loaded_keys.add(rescale_name)

        # OFT requires applying orthogonal transformations to the original weight
        logger.warning(f"OFT adapter {lora_key} cannot be pre-calculated - requires runtime weight access")
        return None