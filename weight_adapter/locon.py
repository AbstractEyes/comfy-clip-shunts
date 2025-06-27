import logging
from typing import Optional, List
import torch
from .base import WeightAdapterBase, register_adapter

logger = logging.getLogger(__name__)


@register_adapter
class LoConAdapter(WeightAdapterBase):
    """Low-rank Convolution Adapter (LoCon)"""
    name = "locon"

    @classmethod
    def is_applicable(cls, lora_key: str, lora: dict[str, torch.Tensor]) -> bool:
        """Check if this is a LoCon layer"""
        return f"{lora_key}.lora_up.weight" in lora and f"{lora_key}.lora_down.weight" in lora

    @classmethod
    def extract_weight(cls, lora_key: str, lora: dict[str, torch.Tensor],
                       alpha: Optional[float], loaded_keys: set[str]) -> Optional[torch.Tensor]:
        """Extract and reconstruct LoCon weight"""
        # Component names
        keys = {
            "up": f"{lora_key}.lora_up.weight",
            "down": f"{lora_key}.lora_down.weight",
            "mid": f"{lora_key}.lora_mid.weight",
            "reshape": f"{lora_key}.reshape_weight",
        }

        # Alternative mid key names
        if keys["mid"] not in lora:
            for alt in [f"{lora_key}.hada_t1", f"{lora_key}.hada_mid", f"{lora_key}.mid"]:
                if alt in lora:
                    keys["mid"] = alt
                    break

        # Check required components
        if keys["up"] not in lora or keys["down"] not in lora:
            return None

        # Extract components
        up = lora[keys["up"]]
        down = lora[keys["down"]]
        mid = lora.get(keys["mid"])

        # Track loaded keys
        loaded_keys.update([keys["up"], keys["down"]])
        if mid is not None:
            loaded_keys.add(keys["mid"])

        # Handle reshape if present
        reshape_info = None
        if keys["reshape"] in lora:
            try:
                reshape_info = lora[keys["reshape"]].tolist()
                loaded_keys.add(keys["reshape"])
            except Exception:
                pass

        # Apply alpha scaling
        if alpha is not None:
            alpha = alpha / down.shape[0]
        else:
            alpha = 1.0

        try:
            import time
            start_time = time.time()

            # Define computation function
            def compute_locon(up, down, mid=None):
                # Apply mid transformation if present
                if mid is not None:
                    final_shape = [down.shape[1], down.shape[0], mid.shape[2], mid.shape[3]]
                    down = (
                        torch.mm(
                            down.transpose(0, 1).flatten(start_dim=1),
                            mid.transpose(0, 1).flatten(start_dim=1),
                        )
                        .reshape(final_shape)
                        .transpose(0, 1)
                    )

                # Compute weight difference
                # Compute weight difference
                if up.dim() == 4 and down.dim() == 4:
                    # Conv weights - kernel size from down tensor
                    out_ch, rank = up.shape[0], up.shape[1]
                    in_ch = down.shape[1]
                    kernel_h, kernel_w = down.shape[2], down.shape[3]

                    up_flat = up.reshape(out_ch, rank)
                    down_flat = down.reshape(rank, in_ch * kernel_h * kernel_w)
                    result = torch.mm(up_flat, down_flat)

                    # Reshape to conv shape based on actual dimensions
                    if result.shape[1] == in_ch * 9:  # 3x3 conv
                        result = result.reshape(out_ch, in_ch, 3, 3)
                    elif result.shape[1] == in_ch:  # 1x1 conv
                        result = result.reshape(out_ch, in_ch, 1, 1)
                    else:
                        # Default reshape
                        result = result.reshape(out_ch, in_ch, kernel_h, kernel_w)
                else:
                    # Linear weights
                    result = torch.mm(up.flatten(start_dim=1), down.flatten(start_dim=1))

                result = result * alpha

                return result

            # Use CUDA if available
            tensors = [up, down] if mid is None else [up, down, mid]
            lora_diff = cls.compute_on_cuda(tensors, compute_locon)

            elapsed = time.time() - start_time
            if elapsed > 0.1:  # Log slow operations
                logger.info(
                    f"LoCon {lora_key} took {elapsed:.2f}s, shapes: up={up.shape}, down={down.shape}, mid={'yes' if mid is not None else 'no'}")

            # Note: reshape_info would need to be handled at the patch application level
            if reshape_info:
                logger.info(f"LoCon layer {lora_key} has reshape info: {reshape_info}")

            return lora_diff

        except Exception as e:
            logger.error(f"Failed to reconstruct LoCon weights for {lora_key}: {e}")
            return None