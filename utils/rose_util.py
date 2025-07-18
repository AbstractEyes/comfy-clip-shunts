import torch
import torch.nn.functional as F

def normalize(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return x / (x.norm(dim=-1, keepdim=True) + eps)

def cosine_similarity(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    a_norm = normalize(a, eps)
    b_norm = normalize(b, eps)
    return (a_norm * b_norm).sum(dim=-1)

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