import torch
import torch.nn.functional as F
from typing import Dict, Optional, Union


class PentachoronStabilizer:
    """
    Geometric constraint utilities for a 5-simplex (pentachoron).
    Includes Rose scoring for semantic alignment.
    """

    @staticmethod
    def vertices_to_tensor(vertices):
        """Convert dict to tensor once, reuse everywhere."""
        if isinstance(vertices, dict):
            return torch.stack([
                vertices['anchor'], vertices['need'],
                vertices['relation'], vertices['purpose'],
                vertices['observer']
            ], dim=1)  # [B, 5, D]
        return vertices

    @staticmethod
    def tensor_to_dict(verts):
        """Convert tensor [B, 5, D] back to dict."""
        return {
            'anchor': verts[:, 0],
            'need': verts[:, 1],
            'relation': verts[:, 2],
            'purpose': verts[:, 3],
            'observer': verts[:, 4]
        }

    @staticmethod
    def rose_score_magnitude(
            x: torch.Tensor,
            vertices: Union[Dict[str, torch.Tensor], torch.Tensor],
            eps: float = 1e-6
    ) -> torch.Tensor:
        """
        Compute Rose similarity score between x and pentachoron vertices.

        Args:
            x: Query tensor [B, T, D] or [B, D]
            vertices: Either dict or tensor [B, 5, D]
            eps: Small value for numerical stability

        Returns:
            scores: [B, T] or [B] depending on input shape
        """
        # Handle input shapes
        squeeze_output = False
        if x.dim() == 2:
            x = x.unsqueeze(1)  # [B, 1, D]
            squeeze_output = True

        # Get vertices as dict
        if not isinstance(vertices, dict):
            vertices = PentachoronStabilizer.tensor_to_dict(vertices)

        # Expand vertices to match sequence dimension
        B, T, D = x.shape
        need = vertices['need'].unsqueeze(1).expand(-1, T, -1)
        relation = vertices['relation'].unsqueeze(1).expand(-1, T, -1)
        purpose = vertices['purpose'].unsqueeze(1).expand(-1, T, -1)

        # Normalize all inputs
        x_n = F.normalize(x, dim=-1, eps=eps)
        n_n = F.normalize(need, dim=-1, eps=eps)
        r_n = F.normalize(relation, dim=-1, eps=eps)
        p_n = F.normalize(purpose, dim=-1, eps=eps)

        # Core directional cosine components
        a_n = torch.cosine_similarity(x_n, n_n, dim=-1)
        a_r = torch.cosine_similarity(x_n, r_n, dim=-1)
        a_p = torch.cosine_similarity(x_n, p_n, dim=-1)

        # Triadic magnitude score
        r7 = (a_n + a_r + a_p) / 3.0
        r8 = x.norm(dim=-1)

        score = r7 * r8

        return score.squeeze(1) if squeeze_output else score

    @staticmethod
    def compute_gram_matrix(verts):
        """Compute Gram matrix for batch of vertices."""
        return torch.bmm(verts, verts.transpose(-2, -1))

    @staticmethod
    def cayley_menger_determinant(verts):
        """Compute Cayley-Menger determinant (vectorized)."""
        B = verts.shape[0]

        gram = torch.bmm(verts, verts.transpose(-2, -1))
        diag = gram.diagonal(dim1=-2, dim2=-1).unsqueeze(-1)
        dist_sq = diag + diag.transpose(-2, -1) - 2 * gram

        cm = torch.zeros(B, 6, 6, device=verts.device)
        cm[:, 0, 1:] = 1
        cm[:, 1:, 0] = 1
        cm[:, 1:, 1:] = dist_sq

        return torch.det(cm)

    @staticmethod
    def enforce_regular_simplex(verts):
        """Compute edge length variance (fully vectorized)."""
        diff = verts.unsqueeze(2) - verts.unsqueeze(1)
        dist = torch.norm(diff, dim=-1)

        triu_indices = torch.triu_indices(5, 5, offset=1)
        edges = dist[:, triu_indices[0], triu_indices[1]]

        return torch.var(edges, dim=-1)

    @staticmethod
    def orthoplex_projection(verts):
        """Project to unit hypersphere, centered."""
        verts_norm = F.normalize(verts, dim=-1)
        center = verts_norm.mean(dim=1, keepdim=True)
        verts_centered = verts_norm - center
        return F.normalize(verts_centered, dim=-1)

    @staticmethod
    def apply(
            vertices,
            cayley_target: float = 1.0,
            return_dict: bool = False,
            compute_rose_scores: Optional[torch.Tensor] = None
    ):
        """
        Apply all constraints and return stable vertices + losses.

        Args:
            vertices: Either dict or tensor [B, 5, D]
            cayley_target: Target Cayley-Menger determinant
            return_dict: If True and input was dict, return dict
            compute_rose_scores: Optional tensor to compute Rose scores against

        Returns:
            vertices_stable: Stabilized vertices
            losses: Dict of loss components (includes rose_scores if requested)
        """
        was_dict = isinstance(vertices, dict)
        verts = PentachoronStabilizer.vertices_to_tensor(vertices)

        # Compute geometric losses
        cm_det = PentachoronStabilizer.cayley_menger_determinant(verts)
        validity_loss = torch.abs(cm_det - cayley_target).mean()
        regularity_loss = PentachoronStabilizer.enforce_regular_simplex(verts).mean()

        # Stabilize vertices
        verts_stable = PentachoronStabilizer.orthoplex_projection(verts)

        # Compute Gram entropy
        gram = PentachoronStabilizer.compute_gram_matrix(verts_stable)
        gram_entropy = -torch.sum(gram * torch.log(torch.abs(gram) + 1e-8)) / (verts.shape[0] * 25)

        losses = {
            'validity': validity_loss,
            'regularity': regularity_loss,
            'gram_entropy': gram_entropy
        }

        # Compute Rose scores if requested
        if compute_rose_scores is not None:
            rose_scores = PentachoronStabilizer.rose_score_magnitude(
                compute_rose_scores,
                verts_stable
            )
            losses['rose_scores'] = rose_scores

        # Convert back to dict if requested
        if was_dict and return_dict:
            verts_stable = PentachoronStabilizer.tensor_to_dict(verts_stable)

        return verts_stable, losses