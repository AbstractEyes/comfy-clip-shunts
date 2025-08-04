import torch
import torch.nn.functional as F
from torch import nn


class PentachoronStabilizer(nn.Module):
    """
    Mathematical constraints from 5-simplex geometry
    Based on Cayley-Menger determinants and simplex regularity conditions
    """

    def __init__(self, hidden_dim=256):
        super().__init__()
        self.hidden_dim = hidden_dim

        # Gram matrix normalizer (enforces distance constraints)
        self.gram_proj = nn.Linear(hidden_dim, hidden_dim)

        # Cayley-Menger regularizer (enforces simplex validity)
        self.cayley_scale = nn.Parameter(torch.ones(1))

    def compute_gram_matrix(self, vertices):
        """Compute Gram matrix for 5 vertices"""
        # vertices: dict with 5 tensors of shape [B, D]
        verts = torch.stack([
            vertices['anchor'], vertices['need'],
            vertices['relation'], vertices['purpose'],
            vertices['observer']
        ], dim=1)  # [B, 5, D]

        # Gram matrix G_ij = <v_i, v_j>
        gram = torch.matmul(verts, verts.transpose(-2, -1))  # [B, 5, 5]
        return gram

    def cayley_menger_determinant(self, vertices):
        """
        Compute Cayley-Menger determinant for 5-simplex
        This should be non-zero for a valid pentachoron
        """
        gram = self.compute_gram_matrix(vertices)

        # Build extended Cayley-Menger matrix
        B = gram.shape[0]
        cm_matrix = torch.zeros(B, 6, 6, device=gram.device)

        # Top-left: 0
        cm_matrix[:, 0, 0] = 0

        # First row/column: all 1s except corner
        cm_matrix[:, 0, 1:] = 1
        cm_matrix[:, 1:, 0] = 1

        # Rest: distance matrix (from Gram matrix)
        # d²_ij = ||v_i||² + ||v_j||² - 2<v_i, v_j>
        for i in range(5):
            for j in range(5):
                if i == j:
                    cm_matrix[:, i + 1, j + 1] = 0
                else:
                    # Distance squared between vertices
                    d_sq = (gram[:, i, i] + gram[:, j, j] - 2 * gram[:, i, j])
                    cm_matrix[:, i + 1, j + 1] = d_sq

        # Compute determinant (should be non-zero)
        det = torch.det(cm_matrix)
        return det

    def enforce_regular_simplex(self, vertices):
        """
        Enforce regular 5-simplex constraints:
        1. All edges equal length
        2. All faces are regular tetrahedra
        """
        verts_list = [vertices['anchor'], vertices['need'],
                      vertices['relation'], vertices['purpose'],
                      vertices['observer']]

        # Compute all pairwise distances
        distances = []
        for i in range(5):
            for j in range(i + 1, 5):
                dist = torch.norm(verts_list[i] - verts_list[j], dim=-1)
                distances.append(dist)

        distances = torch.stack(distances, dim=-1)  # [B, 10]

        # Regularization: variance of edge lengths should be minimal
        edge_variance = torch.var(distances, dim=-1)

        return edge_variance

    def orthoplex_projection(self, vertices):
        """
        Project onto 5-orthoplex (cross-polytope) constraints
        Dual of the 5-hypercube, all vertices equidistant from origin
        """
        verts = torch.stack([
            vertices['anchor'], vertices['need'],
            vertices['relation'], vertices['purpose'],
            vertices['observer']
        ], dim=1)

        # Normalize to unit hypersphere
        verts_normalized = F.normalize(verts, dim=-1)

        # Enforce sum-to-zero constraint (centered at origin)
        center = verts_normalized.mean(dim=1, keepdim=True)
        verts_centered = verts_normalized - center

        # Reproject to hypersphere
        verts_final = F.normalize(verts_centered, dim=-1)

        return {
            'anchor': verts_final[:, 0],
            'need': verts_final[:, 1],
            'relation': verts_final[:, 2],
            'purpose': verts_final[:, 3],
            'observer': verts_final[:, 4]
        }

    def forward(self, vertices):
        """Apply all geometric constraints"""

        # 1. Gram matrix normalization
        gram = self.compute_gram_matrix(vertices)
        gram_normalized = F.softmax(gram.view(-1, 25), dim=-1).view(gram.shape)

        # 2. Cayley-Menger validity
        cm_det = self.cayley_menger_determinant(vertices)
        validity_loss = torch.abs(cm_det - self.cayley_scale).mean()

        # 3. Regular simplex constraint
        regularity_loss = self.enforce_regular_simplex(vertices)

        # 4. Orthoplex projection for stability
        vertices_stable = self.orthoplex_projection(vertices)

        losses = {
            'validity': validity_loss,
            'regularity': regularity_loss.mean(),
            'gram_entropy': -(gram_normalized * torch.log(gram_normalized + 1e-8)).sum(-1).mean()
        }

        return vertices_stable, losses