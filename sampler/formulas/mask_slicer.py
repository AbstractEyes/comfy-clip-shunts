from typing import Optional, Tuple
import logging
import torch

logger = logging.getLogger(__name__)

class MaskSlicer:
    def stencil_to_mask(
        self,
        stencil: torch.Tensor,
        target_shape: Optional[torch.Size] = None,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None
    ) -> torch.Tensor:
        """
        Normalize a stencil into a float mask in [0,1], aligned to
        target_shape, dtype, and device.
        - bool → 0/1 float mask
        - float → must already be in [0,1]
        """
        # Convert boolean → float
        if stencil.dtype == torch.bool:
            mask = stencil.to(torch.float32)
        elif torch.is_floating_point(stencil):
            mask = stencil.clone()
        else:
            raise TypeError(f"Unsupported stencil dtype {stencil.dtype}; expected bool or float.")

        dtype = dtype or mask.dtype
        device = device or mask.device

        if mask.dtype != dtype:
            mask = mask.to(dtype=dtype)
        if mask.device != device:
            mask = mask.to(device)

        if target_shape is not None and mask.shape != target_shape:
            mask = mask.expand(target_shape)

        return mask

    def prepare_and_project(
        self,
        raw_tensor: torch.Tensor,
        raw_stencil: Optional[torch.Tensor] = None,
        raw_preserve: Optional[torch.Tensor] = None,
        config: Optional[dict] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Prepare raw_tensor, stencil mask, and preserve mask:
          • Cast to config['dtype'] & device
          • Default missing masks to all-ones
          • Expand masks to tensor.shape if needed

        Returns: (tensor, stencil_mask, preserve_mask)
        """
        cfg = config or {}
        target_dtype = cfg.get("dtype", torch.float32)
        target_device = torch.device(cfg.get("device", "cpu"))

        # Always do dtype AND device conversion
        tensor = raw_tensor.to(device=target_device, dtype=target_dtype)

        # Stencil mask
        if raw_stencil is None:
            stencil = torch.ones_like(tensor, dtype=target_dtype, device=target_device)
        else:
            stencil = raw_stencil.to(device=target_device, dtype=target_dtype)
            if stencil.shape != tensor.shape:
                stencil = stencil.expand_as(tensor)

        # Preserve mask
        if raw_preserve is None:
            preserve = torch.ones_like(tensor, dtype=target_dtype, device=target_device)
        else:
            preserve = raw_preserve.to(device=target_device, dtype=target_dtype)
            if preserve.shape != tensor.shape:
                preserve = preserve.expand_as(tensor)

        return tensor, stencil, preserve

    def slice(
        self,
        tensor: torch.Tensor,
        stencil: Optional[torch.Tensor] = None,
        preserve: Optional[torch.Tensor] = None,
        soft_mask: bool = False,
        mask_value: float = -100.0
    ) -> torch.Tensor:
        """
        Apply stencil and/or preserve masks:
        - soft_mask=False: masked positions get set to mask_value
        - soft_mask=True: multiply by mask (values in [0,1])
        """
        original = tensor
        result = tensor

        if stencil is not None:
            mask_bool = stencil.bool()
            if soft_mask:
                # Zero out where stencil==0
                result = original * stencil
            else:
                # Hard-mask: set outside region to mask_value
                fill = torch.full_like(original, mask_value)
                result = torch.where(mask_bool, original, fill)

        if preserve is not None:
            mask_bool = preserve.bool()
            if soft_mask:
                # Keep result inside preserve, leave original outside
                result = result * preserve + original * (~mask_bool)
            else:
                # Hard-mask outside preserve
                fill = torch.full_like(original, mask_value)
                result = torch.where(mask_bool, result, fill)

        return result
