import math, torch, torch.nn.functional as F

import torch.nn.functional as F


def match_project(tensor: torch.Tensor, reference: torch.Tensor, mode: str = "linear") -> torch.Tensor:
    if tensor.ndim == 4:
        tensor = tensor.squeeze(1)
    if reference.ndim == 4:
        reference = reference.squeeze(1)

    B, T, D = tensor.shape
    target_d = reference.shape[-1]

    if D == target_d:
        return tensor

    # Reshape to 3D: [B*T, 1, D]
    reshaped = tensor.reshape(B * T, 1, D)

    # Interpolate feature dimension
    interpolated = F.interpolate(
        reshaped,
        size=target_d,
        mode=mode,
        align_corners=False if mode in {"linear", "bilinear", "bicubic", "trilinear"} else None
    )

    # Reshape back to [B, T, target_d]
    return interpolated.reshape(B, T, target_d)



def match_feature_dims(x: torch.Tensor, ref: torch.Tensor,
                       mode: str = "linear") -> torch.Tensor:
    """
    Resample last dimension of `x` to match `ref`.
    Works for up‑ and down‑sampling. Keeps gradients.
    """
    if x.shape[-1] == ref.shape[-1]:
        return x
    # reshape to [B*T, D] → [B*T, 1, D] so interpolate operates on last dim
    B, T, D_in  = x.shape
    D_out       = ref.shape[-1]
    x_reshape   = x.reshape(-1, 1, D_in)
    x_resampled = F.interpolate(x_reshape, size=D_out,
                                mode=mode, align_corners=False)
    return x_resampled.reshape(B, T, D_out)


def match_tokens(x: torch.Tensor, target_len: int) -> torch.Tensor:
    """
    Repeat or truncate token dimension until length == target_len.
    Keeps semantic ordering (wrap‑repeat).
    """
    if x.shape[1] == target_len:
        return x
    if x.shape[1] < target_len:                                # repeat
        reps = math.ceil(target_len / x.shape[1])
        x    = x.repeat(1, reps, 1)
        return x[:, :target_len, :]
    else:                                                      # truncate
        return x[:, :target_len, :]
