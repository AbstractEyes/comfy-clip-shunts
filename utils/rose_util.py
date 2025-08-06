from abc import ABC
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Union

import torch
import torch.nn.functional as F
from torch import nn


from dataclasses import dataclass, field
from typing import List, Optional

from .rose_config import RoseConfig



def normalize(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return x / (x.norm(dim=-1, keepdim=True) + eps)

def cosine_similarity(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    a_norm = normalize(a, eps)
    b_norm = normalize(b, eps)
    return (a_norm * b_norm).sum(dim=-1)






def entropy(tensor: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    prob = F.softmax(tensor, dim=-1)
    log_prob = torch.log(prob + eps)
    return -(prob * log_prob).sum(dim=-1)


import torch
import torch.nn.functional as F

def legacy_normalize(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return x / (x.norm(dim=-1, keepdim=True) + eps)

def legacy_cosine_similarity(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    a_norm = legacy_normalize(a, eps)
    b_norm = legacy_normalize(b, eps)
    return (a_norm * b_norm).sum(dim=-1)


def legacy_entropy(tensor: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    prob = F.softmax(tensor, dim=-1)
    log_prob = torch.log(prob + eps)
    return -(prob * log_prob).sum(dim=-1)


def rose_score(
    x: torch.Tensor,
    need: torch.Tensor,
    relation: torch.Tensor,
    purpose: torch.Tensor,
    eps: float = 1e-8,
    full: bool = False,
    magnitude: float = 1.0,
    entropy_weight: float = 1.0,
) -> Union[torch.Tensor, Dict[str, Union[torch.Tensor, List[torch.Tensor]]]]:
    eps = max(eps, 1e-8)
    x =         legacy_normalize(x, eps)
    need =      legacy_normalize(need, eps)
    relation =  legacy_normalize(relation, eps)
    purpose =   legacy_normalize(purpose, eps)

    # Triadic alignments
    a_n = legacy_cosine_similarity(x, need, eps)
    a_r = legacy_cosine_similarity(x, relation, eps)
    a_p = legacy_cosine_similarity(x, purpose, eps)

    # Condensed vectors
    s1 = legacy_normalize(need + relation, eps)
    s2 = legacy_normalize(need + purpose, eps)
    s3 = legacy_normalize(relation + purpose, eps)
    s4 = legacy_normalize(need - relation, eps)
    s5 = legacy_normalize(need - purpose, eps)
    s6 = legacy_normalize(relation - purpose, eps)

    # Resonance angles (cosine scores)
    r1 = legacy_cosine_similarity(x, s1, eps)
    r2 = legacy_cosine_similarity(x, s2, eps)
    r3 = legacy_cosine_similarity(x, s3, eps)
    r4 = legacy_cosine_similarity(x, s4, eps)
    r5 = legacy_cosine_similarity(x, s5, eps)
    r6 = legacy_cosine_similarity(x, s6, eps)

    # Extended resonance values
    r7 = (a_n + a_r + a_p) / 3.0                     # core triadic resonance
    r8 = x.norm(dim=-1)                              # magnitude component
    r9 = legacy_entropy(x, eps)                                  # entropy as signal complexity

    # Final ROSE value: weighted average for now
    rose = (r1 + r2 + r3 + r4 + r5 + r6 + r7 + r8 + r9) / 9.0
    if full:
        return {
            "rose": rose,
            "triadic": r7,
            "magnitude": r8,
            "entropy": r9,
            "components": [r1, r2, r3, r4, r5, r6],
            "alignment_vectors": [s1, s2, s3, s4, s5, s6]
        }
    return rose


import torch
from typing import List, Dict, Union, Optional

def rose_score_magnitude(
    x: torch.Tensor,
    need: torch.Tensor,
    relation: torch.Tensor,
    purpose: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    Computes a magnitude-only Rose similarity score between `x` and `need`,
    modulated by triadic reference vectors `relation` and `purpose`.

    Output: [B, T]
    """
    # Normalize all inputs
    x_n = F.normalize(x, dim=-1, eps=eps)
    n_n = F.normalize(need, dim=-1, eps=eps)
    r_n = F.normalize(relation, dim=-1, eps=eps)
    p_n = F.normalize(purpose, dim=-1, eps=eps)

    # Core directional cosine components
    a_n = torch.cosine_similarity(x_n, n_n, dim=-1)     # similarity to need
    a_r = torch.cosine_similarity(x_n, r_n, dim=-1)     # similarity to relation
    a_p = torch.cosine_similarity(x_n, p_n, dim=-1)     # similarity to purpose

    # Triadic magnitude score (no entropy)
    r7 = (a_n + a_r + a_p) / 3.0                        # resonance magnitude average
    r8 = x.norm(dim=-1)                                 # magnitude of symbolic field

    return r7 * r8                                       # final score [B, T]


def rose_score_flow(
    x: torch.Tensor,
    need: torch.Tensor,
    relation: torch.Tensor,
    purpose: torch.Tensor,
    steps: int = 32,
    base_eps: float = 1e-8,
    max_eps: float = 1.0,
    retain_trace: bool = False,
    grad_scale: float = 1.0,
    detach_between_steps: bool = True
) -> Union[torch.Tensor, List[torch.Tensor]]:
    """
    Evolves a symbolic vector `x` over ROSE-space using Euler steps based on ∇ROSE(x).
    Returns final evolved vector or full trajectory if `retain_trace=True`.

    Args:
        x: token embedding [D] or batch [B, D]
        need, relation, purpose: symbolic pentachoron basis vectors [D] or [B, D]
        steps: number of flow steps
        base_eps: starting eps value
        max_eps: ending eps value (trajectory controls symbolic decay)
        retain_trace: return list of intermediate vectors
        grad_scale: velocity scale factor (Δt)
        detach_between_steps: whether to detach x between updates to prevent gradient chaining
    """
    x = x.detach().clone().requires_grad_(True)
    dt = 1.0 / steps
    trace = []

    for i in range(steps):
        eps = base_eps + i * (max_eps - base_eps) / steps
        rose = rose_score(x, need, relation, purpose, eps=eps)
        if isinstance(rose, dict):
            rose = rose["rose"]

        grad = torch.autograd.grad(rose.sum(), x, create_graph=False)[0]  # ∇ROSE(x)
        x = x + dt * grad_scale * grad

        if detach_between_steps:
            x = x.detach().clone().requires_grad_(True)

        if retain_trace:
            trace.append(x.detach().clone())

    return trace if retain_trace else x.detach()



def rose_score_v2(
    x: torch.Tensor,
    need: torch.Tensor,
    relation: torch.Tensor,
    purpose: torch.Tensor,
    config: Optional[RoseConfig] = None,
    external_field: Optional[torch.Tensor] = None,
) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    Canonical ROSE Score calculation — pentachoron-guided resonance with optional external field.
    This is the flagship accessor for all ROSE variants.
    """
    cfg = config or RoseConfig()

    if cfg.clone_inputs:
        x, need, relation, purpose = x.clone(), need.clone(), relation.clone(), purpose.clone()
    x = normalize(x)
    need = normalize(need)
    relation = normalize(relation)
    purpose = normalize(purpose)

    # --- [External field: observer modulation] ---
    if external_field is not None:
        x = normalize(x + external_field * cfg.weight_external_field)

    # --- [Noise injection] ---
    if cfg.noise_amplification > 0.0:
        noise = torch.randn_like(x) * cfg.noise_amplification
        x = normalize(x + noise)

    # --- [Triadic alignments] ---
    a_n = cosine_similarity(x, need)
    a_r = cosine_similarity(x, relation)
    a_p = cosine_similarity(x, purpose)
    triadic = (a_n + a_r + a_p) / 3.0

    # --- [Condensed vector alignments] ---
    s1 = normalize(need + relation)
    s2 = normalize(need + purpose)
    s3 = normalize(relation + purpose)
    s4 = normalize(need - relation)
    s5 = normalize(need - purpose)
    s6 = normalize(relation - purpose)
    components = [cosine_similarity(x, s) for s in [s1, s2, s3, s4, s5, s6]]
    condensed = sum(components) / 6.0

    # --- [Magnitude and entropy] ---
    magnitude = x.norm(dim=-1)
    ent = entropy(x)

    # --- [Weighted composition] ---
    numer = (
        cfg.weight_condensed * condensed +
        cfg.weight_triads * triadic +
        cfg.weight_entropy * ent +
        cfg.weight_magnitude * magnitude
    )
    denom = cfg.weight_condensed + cfg.weight_triads + cfg.weight_entropy + cfg.weight_magnitude
    rose = numer / (denom + 1e-8)

    # --- [Delta logic / Output projection modes] ---
    delta = None
    if cfg.use_normalized_delta:
        delta = normalize(condensed.unsqueeze(0) - x)  # Vector shift from current x to condensed
        x = normalize(x + delta)

    if cfg.projection_mode == "folding":
        x = normalize(x + (condensed.unsqueeze(0) - x) * rose.unsqueeze(0))
    elif cfg.projection_mode == "projection":
        x = condensed

    if cfg.residual_output:
        return {
            "rose": rose,
            "triadic": triadic,
            "magnitude": magnitude,
            "entropy": ent,
            "components": components,
            "condensed": condensed,
            "delta": delta,
            "output": x,
        }

    return rose


def rose_score_5d(
    x: torch.Tensor,
    need: torch.Tensor,
    relation: torch.Tensor,
    purpose: torch.Tensor,
    config: Optional[RoseConfig] = None,
    external_field: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    x = normalize(x)
    need = normalize(need)
    relation = normalize(relation)
    purpose = normalize(purpose)

    cfg = config or RoseConfig()

    # Core alignments
    a_n = cosine_similarity(x, need)
    a_r = cosine_similarity(x, relation)
    a_p = cosine_similarity(x, purpose)

    # Composite resonance states
    s1 = normalize(need + relation)
    s2 = normalize(need + purpose)
    s3 = normalize(relation + purpose)
    s4 = normalize(need - relation)
    s5 = normalize(need - purpose)
    s6 = normalize(relation - purpose)

    r = [
        cosine_similarity(x, s1),
        cosine_similarity(x, s2),
        cosine_similarity(x, s3),
        cosine_similarity(x, s4),
        cosine_similarity(x, s5),
        cosine_similarity(x, s6),
    ]

    # Modulators
    triadic = (a_n + a_r + a_p) / 3.0
    magnitude = x.norm(dim=-1)
    ent = entropy(x)

    # External Field Adjustment
    if external_field is not None:
        x = normalize(x + external_field)

    # Compute weighted rose resonance
    rose = (
        cfg.w_condensed * sum(r) / 6.0 +
        cfg.w_triads * triadic +
        cfg.w_entropy * ent +
        cfg.w_magnitude * magnitude
    ) / (cfg.w_condensed + cfg.w_triads + cfg.w_entropy + cfg.w_magnitude)

    if cfg.return_full:
        return {
            "rose": rose,
            "triadic": triadic,
            "magnitude": magnitude,
            "entropy": ent,
            "components": r,
            "alignment_vectors": [s1, s2, s3, s4, s5, s6]
        }

    return rose