from typing import Optional, Union, Tuple
import torch
import torch.nn as nn
import logging

logger = logging.getLogger(__name__)


class WeightAdapterBase:
    """Base class for all weight adapters (LoRA, LoHA, LoKr, etc.)"""
    name: str

    @classmethod
    def load(cls, lora_key: str, lora: dict[str, torch.Tensor], alpha: Optional[float] = None,
             dora_scale: Optional[torch.Tensor] = None, loaded_keys: set[str] = None) -> Optional[Tuple[str, Tuple]]:
        """
        Load and pre-calculate weight modifications in ComfyUI-compatible format.

        Args:
            lora_key: The key prefix for this layer's weights in the lora dict
            lora: Dictionary containing all lora weights
            alpha: Optional alpha scaling factor
            dora_scale: Optional DoRA scale tensor
            loaded_keys: Set to track which keys have been loaded

        Returns:
            ComfyUI-compatible patch format: (patch_type, (data,)) or None if not applicable
        """
        if loaded_keys is None:
            loaded_keys = set()

        # Extract the weight modification using adapter-specific logic
        weight_diff = cls.extract_weight(lora_key, lora, alpha, loaded_keys)
        if weight_diff is None:
            return None

        # For now, just return the diff - DoRA needs runtime application
        # TODO: DoRA requires runtime weight access for proper normalization
        if dora_scale is not None:
            logger.warning(f"DoRA scale present for {lora_key} but cannot be pre-applied")

        # Standard diff patch
        return ("diff", (weight_diff,))

    @classmethod
    def extract_weight(cls, lora_key: str, lora: dict[str, torch.Tensor],
                       alpha: Optional[float], loaded_keys: set[str]) -> Optional[torch.Tensor]:
        """
        Extract and reconstruct the weight modification tensor.
        Each adapter subclass implements its specific reconstruction math.

        Args:
            lora_key: The key prefix for this layer's weights
            lora: Dictionary containing all lora weights
            alpha: Optional alpha scaling factor
            loaded_keys: Set to track which keys have been loaded

        Returns:
            Reconstructed weight difference tensor or None if not applicable
        """
        raise NotImplementedError(f"{cls.name} must implement extract_weight")

    @staticmethod
    def compute_on_cuda(tensors: list[torch.Tensor], compute_fn, *args, **kwargs) -> torch.Tensor:
        """
        Move tensors to CUDA for computation if available, then return to CPU.
        """
        if torch.cuda.is_available():
            device = 'cuda'
            orig_devices = [t.device for t in tensors]
            cuda_tensors = [t.to(device) for t in tensors]

            with torch.no_grad():
                result = compute_fn(*cuda_tensors, *args, **kwargs)

            return result.cpu()
        else:
            with torch.no_grad():
                return compute_fn(*tensors, *args, **kwargs)

    @classmethod
    def is_applicable(cls, lora_key: str, lora: dict[str, torch.Tensor]) -> bool:
        """
        Check if this adapter type can handle the given lora weights.

        Args:
            lora_key: The key prefix to check
            lora: Dictionary containing all lora weights

        Returns:
            True if this adapter can process these weights
        """
        raise NotImplementedError(f"{cls.name} must implement is_applicable")


# Utility functions for weight reconstruction

def weight_decompose(dora_scale: torch.Tensor, weight: torch.Tensor, lora_diff: torch.Tensor,
                     alpha: float = 1.0, strength: float = 1.0) -> torch.Tensor:
    """
    Apply DoRA (Weight-Decomposed Low-Rank Adaptation) to a weight tensor.
    This needs to be called during model execution, not during loading.
    """
    # Apply alpha scaling to lora diff
    if alpha != 1.0:
        lora_diff = lora_diff * alpha

    # Calculate modified weight
    weight_calc = weight + lora_diff

    # Determine if decomposition is on output or input axis
    wd_on_output_axis = dora_scale.shape[0] == weight_calc.shape[0]

    if wd_on_output_axis:
        # Normalize along output dimension
        weight_norm = (
            weight_calc.reshape(weight_calc.shape[0], -1)
            .norm(dim=1, keepdim=True)
            .reshape(weight_calc.shape[0], *[1] * (weight_calc.dim() - 1))
        )
    else:
        # Normalize along input dimension
        weight_norm = (
            weight_calc.transpose(0, 1)
            .reshape(weight_calc.shape[1], -1)
            .norm(dim=1, keepdim=True)
            .reshape(weight_calc.shape[1], *[1] * (weight_calc.dim() - 1))
            .transpose(0, 1)
        )

    # Avoid division by zero
    weight_norm = weight_norm + torch.finfo(weight.dtype).eps

    # Apply DoRA scaling
    weight_calc = weight_calc * (dora_scale / weight_norm)

    # Apply strength if not 1.0
    if strength != 1.0:
        weight_calc = weight + strength * (weight_calc - weight)

    return weight_calc


def tucker_weight_from_conv(up: torch.Tensor, down: torch.Tensor, mid: torch.Tensor) -> torch.Tensor:
    """Reconstruct weight from Tucker decomposition (for convolutional layers)"""
    up = up.reshape(up.size(0), up.size(1))
    down = down.reshape(down.size(0), down.size(1))
    return torch.einsum("m n ..., i m, n j -> i j ...", mid, up, down)


def tucker_weight(wa: torch.Tensor, wb: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """Reconstruct weight from Tucker decomposition"""
    temp = torch.einsum("i j ..., j r -> i r ...", t, wb)
    return torch.einsum("i j ..., i r -> r j ...", temp, wa)


def make_kron(w1: torch.Tensor, w2: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    """Compute Kronecker product of two tensors"""
    if len(w2.shape) == 4:
        w1 = w1.unsqueeze(2).unsqueeze(2)
    w2 = w2.contiguous()
    return torch.kron(w1, w2).reshape(w1.size(0) * w2.size(0), w1.size(1) * w2.size(1), *w2.shape[2:]) * scale


# Adapter registry
adapters: list[type[WeightAdapterBase]] = []


def register_adapter(adapter_cls: type[WeightAdapterBase]):
    """Register an adapter class"""
    if adapter_cls not in adapters:
        adapters.append(adapter_cls)
        logger.info(f"Registered adapter: {adapter_cls.name}")
    return adapter_cls