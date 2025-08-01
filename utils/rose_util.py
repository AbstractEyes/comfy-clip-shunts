from abc import ABC
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Union

import torch
import torch.nn.functional as F
from torch import nn


from dataclasses import dataclass, field
from typing import List, Optional

from .rose_config import RoseConfig



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


def normalize(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return x / (x.norm(dim=-1, keepdim=True) + eps)

def cosine_similarity(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    a_norm = normalize(a, eps)
    b_norm = normalize(b, eps)
    return (a_norm * b_norm).sum(dim=-1)



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


#def entropy(tensor: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
#    prob = F.softmax(tensor, dim=-1)
#    log_prob = torch.log(prob + eps)
#    return -(prob * log_prob).sum(dim=-1)


import torch
import torch.nn.functional as F

#def normalize(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
#    return x / (x.norm(dim=-1, keepdim=True) + eps)
#
#def cosine_similarity(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
#    a_norm = normalize(a, eps)
#    b_norm = normalize(b, eps)
#    return (a_norm * b_norm).sum(dim=-1)


def rose_score(x: torch.Tensor, need: torch.Tensor, relation: torch.Tensor, purpose: torch.Tensor) -> torch.Tensor:
    x = normalize(x)
    need = normalize(need)
    relation = normalize(relation)
    purpose = normalize(purpose)

    # Triadic alignments
    a_n = cosine_similarity(x, need)
    a_r = cosine_similarity(x, relation)
    a_p = cosine_similarity(x, purpose)

    # Condensed vectors
    s1 = normalize(need + relation)
    s2 = normalize(need + purpose)
    s3 = normalize(relation + purpose)
    s4 = normalize(need - relation)
    s5 = normalize(need - purpose)
    s6 = normalize(relation - purpose)

    # Resonance angles (cosine scores)
    r1 = cosine_similarity(x, s1)
    r2 = cosine_similarity(x, s2)
    r3 = cosine_similarity(x, s3)
    r4 = cosine_similarity(x, s4)
    r5 = cosine_similarity(x, s5)
    r6 = cosine_similarity(x, s6)

    # Extended resonance values
    r7 = (a_n + a_r + a_p) / 3.0                     # core triadic resonance
    r8 = x.norm(dim=-1)                              # magnitude component
    r9 = entropy(x)                                  # entropy as signal complexity

    # Final ROSE value: weighted average for now
    rose = (r1 + r2 + r3 + r4 + r5 + r6 + r7 + r8 + r9) / 9.0
    return rose

def entropy(tensor: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    prob = F.softmax(tensor, dim=-1)
    log_prob = torch.log(prob + eps)
    return -(prob * log_prob).sum(dim=-1)