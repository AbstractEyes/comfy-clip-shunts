import logging
from typing import Optional, Tuple
import torch
from .base import WeightAdapterBase, register_adapter

logger = logging.getLogger(__name__)


@register_adapter
class LoHaAdapter(WeightAdapterBase):
    """Low-rank Hadamard product Adapter (LoHA)"""
    name = "loha"

    @classmethod
    def is_applicable(cls, lora_key: str, lora: dict[str, torch.Tensor]) -> bool:
        """Check if this is a LoHA layer"""
        return f"{lora_key}.hada_w1_a" in lora

    @classmethod
    def extract_weight(cls, lora_key: str, lora: dict[str, torch.Tensor],
                       alpha: Optional[float], loaded_keys: set[str]) -> Optional[torch.Tensor]:
        """Extract and reconstruct LoHA weight"""
        # Component names
        hada_w1_a_name = f"{lora_key}.hada_w1_a"
        hada_w1_b_name = f"{lora_key}.hada_w1_b"
        hada_w2_a_name = f"{lora_key}.hada_w2_a"
        hada_w2_b_name = f"{lora_key}.hada_w2_b"
        hada_t1_name = f"{lora_key}.hada_t1"
        hada_t2_name = f"{lora_key}.hada_t2"

        # Check if LoHA components exist
        if hada_w1_a_name not in lora:
            return None

        # Extract components
        w1a = lora[hada_w1_a_name]
        w1b = lora[hada_w1_b_name]
        w2a = lora[hada_w2_a_name]
        w2b = lora[hada_w2_b_name]

        # Track loaded keys
        loaded_keys.update([hada_w1_a_name, hada_w1_b_name, hada_w2_a_name, hada_w2_b_name])

        # Apply alpha scaling
        if alpha is None:
            alpha = 1.0
        else:
            alpha = alpha / w1b.shape[0]

        # Check for CP decomposition components
        t1 = None
        t2 = None
        if hada_t1_name in lora:
            t1 = lora[hada_t1_name]
            t2 = lora[hada_t2_name]
            loaded_keys.update([hada_t1_name, hada_t2_name])

        try:
            import time
            start_time = time.time()

            def compute_loha(w1a, w1b, w2a, w2b, t1=None, t2=None):
                if t1 is not None:
                    # CP decomposition path
                    m1 = torch.einsum('i j k l, j r, i p -> p r k l', t1, w1b, w1a)
                    m2 = torch.einsum('i j k l, j r, i p -> p r k l', t2, w2b, w2a)
                else:
                    # Standard path
                    m1 = torch.mm(w1a, w1b)
                    m2 = torch.mm(w2a, w2b)

                # Hadamard product (element-wise multiplication)
                result = (m1 * m2) * alpha

                # For 2D results, check if it's a transformer linear layer (should stay 2D)
                if result.dim() == 2:
                    # Check if this is a transformer attention/projection/embedding layer
                    if any(x in lora_key for x in
                           ['transformer_blocks', 'attn', 'proj_in', 'proj_out', 'to_q', 'to_k', 'to_v', 'to_out',
                            'emb_layers']):
                        # Keep as 2D for linear layers
                        pass
                    else:
                        # Conv layer reshaping
                        out_ch = result.shape[0]
                        if result.shape[1] % 9 == 0:  # Likely 3x3 conv
                            in_ch = result.shape[1] // 9
                            result = result.reshape(out_ch, in_ch, 3, 3)
                        elif result.shape[1] == out_ch:  # Likely 1x1 conv
                            result = result.reshape(out_ch, out_ch, 1, 1)

                return result

            # Use CUDA computation
            tensors = [w1a, w1b, w2a, w2b]
            if t1 is not None:
                tensors.extend([t1, t2])

            lora_diff = cls.compute_on_cuda(tensors, compute_loha)

            elapsed = time.time() - start_time
            if elapsed > 0.1:  # Log slow operations
                logger.info(f"LoHA {lora_key} took {elapsed:.2f}s")

            return lora_diff

        except Exception as e:
            logger.error(f"Failed to reconstruct LoHA weights for {lora_key}: {e}")
            return None