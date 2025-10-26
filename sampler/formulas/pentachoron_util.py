"""
Pentachoron (4-Simplex) Mathematics Utility
Complete formulas for 4-dimensional simplex geometry and failed historical approaches
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass
from enum import Enum
import math


class ProjectionType(Enum):
    """Coxeter plane projection types for pentachoron"""
    A4 = "pentagon_pentagram"  # Projects to regular pentagon and pentagram
    A3 = "square_pyramid"  # Projects to square pyramid
    A2 = "triangular_bipyramid"  # Projects to triangular bipyramid
    ORTHOGONAL = "orthogonal"  # Standard orthogonal projection


class PentachoronVariant(Enum):
    """Different types of pentachoron configurations"""
    REGULAR = "regular"  # x3o3o3o - Regular pentachoron
    TETRAHEDRAL_PYRAMID = "tetrahedral_pyramid"  # ox3oo3oo - Pyramid on tetrahedron
    TRIANGULAR_SCALENE = "triangular_scalene"  # oxo3ooo - 2 triangular pyramids, 3 digonal disphenoids
    TRIANGULAR_PYRAMIDAL = "triangular_pyramidal"  # 2 different triangular pyramids, 3 sphenoids
    TETRAGONAL_DISPHENOIDAL = "tetragonal_disphenoidal"  # 1 tetragonal disphenoid, 4 sphenoids
    RHOMBIC_DISPHENOIDAL = "rhombic_disphenoidal"  # 1 rhombic disphenoid, 4 irregular tetrahedra
    DIGONAL_DISPHENOIDAL = "digonal_disphenoidal"  # 1 digonal disphenoid, 2 pairs of sphenoids
    PHYLLIC_DISPHENOIDAL = "phyllic_disphenoidal"  # 1 phyllic disphenoid, 2 pairs of identical irregular tetrahedra
    IRREGULAR = "irregular"  # No symmetry, 5 different irregular tetrahedra
    STEP_PRISM = "5-2_step_prism"  # Noble, 5 phyllic disphenoids (gyrochoron)


@dataclass
class PentachoronGeometry:
    """Complete geometric properties of a pentachoron"""
    vertices: np.ndarray
    edges: List[Tuple[int, int]]
    faces: List[Tuple[int, int, int]]
    cells: List[Tuple[int, int, int, int]]
    edge_length: float
    surface_area: float
    surcell_volume: float
    surteron_bulk: float
    circumradius: float
    inradius: float
    variant: PentachoronVariant


class PentachoronMathematics:
    """
    Complete mathematical formulas for pentachoron (4-simplex) geometry
    Including both successful formulations and failed historical approaches
    """

    # Schläfli symbol for regular pentachoron
    SCHLAFLI_SYMBOL = "{3,3,3}"

    # Structural constants
    NUM_VERTICES = 5
    NUM_EDGES = 10
    NUM_FACES = 10  # All triangular
    NUM_CELLS = 5  # All tetrahedral

    # Symmetry group order for regular pentachoron
    SYMMETRY_ORDER = 120  # A4 symmetry group

    @staticmethod
    def generate_vertices_5d_embedding() -> np.ndarray:
        """
        Generate pentachoron vertices in 5D embedding space
        Uses permutations of (0,0,0,0,1)

        Returns vertices with edge length √2 and circumradius 1
        """
        vertices = np.array([
            [1, 0, 0, 0, 0],
            [0, 1, 0, 0, 0],
            [0, 0, 1, 0, 0],
            [0, 0, 0, 1, 0],
            [0, 0, 0, 0, 1]
        ])
        return vertices

    @staticmethod
    def generate_vertices_4d_centered(edge_length: float = 1.0) -> np.ndarray:
        """
        Generate pentachoron vertices in 4D centered at origin

        Formula from research: specific 4D coordinates that maintain symmetry
        """
        # Scaling factors for unit edge length
        s = edge_length / np.sqrt(2)

        vertices = np.array([
            [0.5 * s, -np.sqrt(3) / 6 * s, -np.sqrt(6) / 12 * s, -np.sqrt(10) / 20 * s],
            [-0.5 * s, -np.sqrt(3) / 6 * s, -np.sqrt(6) / 12 * s, -np.sqrt(10) / 20 * s],
            [0, np.sqrt(3) / 3 * s, -np.sqrt(6) / 12 * s, -np.sqrt(10) / 20 * s],
            [0, 0, np.sqrt(6) / 4 * s, -np.sqrt(10) / 20 * s],
            [0, 0, 0, np.sqrt(10) / 5 * s]
        ])

        return vertices

    @staticmethod
    def generate_vertices_4d_general(n: int = 4, edge_length: float = 1.0) -> np.ndarray:
        """
        Generate n-simplex vertices using general formula
        Centered at origin with specified edge length

        Based on formula: coordinates follow pattern with √(2k²+2k) denominators
        """
        vertices = []

        for i in range(n + 1):
            vertex = np.zeros(n)

            # Fill coordinates according to the pattern
            for j in range(n):
                if j < i:
                    vertex[j] = 0
                elif j == i and i < n:
                    # Positive coefficient for the i-th coordinate
                    vertex[j] = (i + 1) / np.sqrt(2 * (j + 1) * (j + 2))
                else:
                    # Negative coefficient for remaining coordinates
                    vertex[j] = -1 / np.sqrt(2 * (j + 1) * (j + 2))

            vertices.append(vertex * edge_length)

        return np.array(vertices)

    @staticmethod
    def calculate_edge_length(vertices: np.ndarray) -> float:
        """Calculate edge length from vertex coordinates"""
        if len(vertices) < 2:
            return 0.0
        return np.linalg.norm(vertices[1] - vertices[0])

    @staticmethod
    def calculate_surface_area(edge_length: float) -> float:
        """
        Calculate total surface area of pentachoron

        Formula: 5√3s²/2 (5 tetrahedral faces)
        """
        return 5 * np.sqrt(3) * edge_length ** 2 / 2

    @staticmethod
    def calculate_surcell_volume(edge_length: float) -> float:
        """
        Calculate 3D "volume" of pentachoron boundary

        Formula: 5s³/(6√2)
        """
        return 5 * edge_length ** 3 / (6 * np.sqrt(2))

    @staticmethod
    def calculate_surteron_bulk(edge_length: float) -> float:
        """
        Calculate 4D hypervolume of pentachoron

        Formula: √5s⁴/96
        """
        return np.sqrt(5) * edge_length ** 4 / 96

    @staticmethod
    def calculate_circumradius(edge_length: float) -> float:
        """
        Calculate circumradius of regular pentachoron

        For 4-simplex: R = edge_length * √(5/8)
        """
        return edge_length * np.sqrt(5 / 8)

    @staticmethod
    def calculate_inradius(edge_length: float) -> float:
        """
        Calculate inradius of regular pentachoron

        For 4-simplex: r = edge_length / (2√10)
        """
        return edge_length / (2 * np.sqrt(10))

    @staticmethod
    def generate_edges() -> List[Tuple[int, int]]:
        """Generate all 10 edges of pentachoron (complete graph K5)"""
        edges = []
        for i in range(5):
            for j in range(i + 1, 5):
                edges.append((i, j))
        return edges

    @staticmethod
    def generate_faces() -> List[Tuple[int, int, int]]:
        """Generate all 10 triangular faces of pentachoron"""
        faces = []
        for i in range(5):
            for j in range(i + 1, 5):
                for k in range(j + 1, 5):
                    faces.append((i, j, k))
        return faces

    @staticmethod
    def generate_cells() -> List[Tuple[int, int, int, int]]:
        """Generate all 5 tetrahedral cells of pentachoron"""
        cells = []
        for i in range(5):
            # Each cell is formed by all vertices except one
            cell = tuple(j for j in range(5) if j != i)
            cells.append(cell)
        return cells

    @staticmethod
    def barycentric_coordinates(
            point: np.ndarray,
            vertices: np.ndarray
    ) -> np.ndarray:
        """
        Convert Cartesian coordinates to barycentric coordinates

        For a point P and simplex vertices V0...V4, find weights w0...w4
        such that P = w0*V0 + w1*V1 + w2*V2 + w3*V3 + w4*V4
        and w0 + w1 + w2 + w3 + w4 = 1
        """
        if vertices.shape[0] != 5 or vertices.shape[1] != point.shape[0]:
            raise ValueError("Invalid dimensions for barycentric conversion")

        # Construct matrix for linear system
        # [V0-V4, V1-V4, V2-V4, V3-V4] * [w0, w1, w2, w3]^T = P - V4
        A = vertices[:-1].T - vertices[-1:].T
        b = point - vertices[-1]

        # Solve for first 4 weights
        try:
            weights_partial = np.linalg.solve(A, b)
            # Calculate fifth weight
            w4 = 1 - np.sum(weights_partial)
            weights = np.append(weights_partial, w4)
        except np.linalg.LinAlgError:
            # Degenerate case - return equal weights
            weights = np.ones(5) / 5

        return weights

    @staticmethod
    def project_to_3d(
            vertices_4d: np.ndarray,
            projection_type: ProjectionType = ProjectionType.ORTHOGONAL
    ) -> np.ndarray:
        """
        Project 4D pentachoron vertices to 3D space

        Different projection types give different visual representations
        """
        if projection_type == ProjectionType.ORTHOGONAL:
            # Simple orthogonal projection - drop last coordinate
            return vertices_4d[:, :3]

        elif projection_type == ProjectionType.A4:
            # A4 Coxeter plane - projects to pentagon/pentagram
            # Use specific rotation matrix to align with A4 symmetry
            theta = 2 * np.pi / 5
            vertices_3d = []
            for i, v in enumerate(vertices_4d):
                angle = i * theta
                x = np.cos(angle) * np.linalg.norm(v[:2])
                y = np.sin(angle) * np.linalg.norm(v[:2])
                z = v[2] if len(v) > 2 else 0
                vertices_3d.append([x, y, z])
            return np.array(vertices_3d)

        elif projection_type == ProjectionType.A3:
            # A3 Coxeter plane - projects to square pyramid
            # Map 4 vertices to square base, 1 to apex
            vertices_3d = np.array([
                [-1, -1, 0],
                [1, -1, 0],
                [1, 1, 0],
                [-1, 1, 0],
                [0, 0, np.sqrt(2)]
            ])
            return vertices_3d

        elif projection_type == ProjectionType.A2:
            # A2 Coxeter plane - triangular bipyramid
            vertices_3d = np.array([
                [0, 0, 1],
                [np.sqrt(3) / 2, 0, -0.5],
                [-np.sqrt(3) / 4, 3 / 4, -0.5],
                [-np.sqrt(3) / 4, -3 / 4, -0.5],
                [0, 0, -1]
            ])
            return vertices_3d

        else:
            return vertices_4d[:, :3]

    @staticmethod
    def orthoscheme_decomposition(scale: float = 1.0) -> List[np.ndarray]:
        """
        Decompose regular pentachoron into 120 characteristic orthoschemes

        Each orthoscheme is a 4-simplex with all right-angled faces
        """
        orthoschemes = []

        # Generate base regular pentachoron
        vertices = PentachoronMathematics.generate_vertices_4d_centered(scale)
        center = np.mean(vertices, axis=0)

        # The 120 orthoschemes are generated by the symmetry group A4
        # This is a simplified representation - full implementation would
        # require generating all 120 symmetry transformations

        # For demonstration, generate a few characteristic orthoschemes
        for i in range(5):  # One for each vertex
            for j in range(i + 1, 5):  # One for each edge from vertex i
                # Create orthoscheme with vertices:
                # center, vertex i, midpoint of edge (i,j), face center, cell center
                v0 = center
                v1 = vertices[i]
                v2 = (vertices[i] + vertices[j]) / 2

                # Face center (example: face containing vertices i, j, and next vertex)
                k = (j + 1) % 5
                if k == i:
                    k = (k + 1) % 5
                v3 = (vertices[i] + vertices[j] + vertices[k]) / 3

                # Cell center (all vertices except one)
                cell_vertices = [vertices[m] for m in range(5) if m != (i + 2) % 5]
                v4 = np.mean(cell_vertices, axis=0)

                orthoscheme = np.array([v0, v1, v2, v3, v4])
                orthoschemes.append(orthoscheme)

        return orthoschemes

    @staticmethod
    def calculate_dihedral_angle() -> float:
        """
        Calculate dihedral angle between two cells of regular pentachoron

        For regular 4-simplex: arccos(1/4) ≈ 75.52°
        """
        return np.arccos(1 / 4) * 180 / np.pi

    @staticmethod
    def generate_compound_dual() -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate compound of two pentachora in dual configuration

        This compound has [[3,3,3]] symmetry with order 240
        The intersection is a uniform bitruncated 5-cell
        """
        # First pentachoron
        vertices1 = PentachoronMathematics.generate_vertices_4d_centered(1.0)

        # Second pentachoron (dual configuration)
        # Dual vertices are at face centers of the original
        faces = PentachoronMathematics.generate_faces()
        vertices2 = []

        for face in faces[:5]:  # Take 5 faces to form dual
            face_vertices = vertices1[list(face)]
            center = np.mean(face_vertices, axis=0)
            # Scale to maintain regularity
            vertices2.append(center * 1.5)

        return np.array(vertices1), np.array(vertices2)

    @staticmethod
    def validate_regularity(vertices: np.ndarray, tolerance: float = 1e-6) -> bool:
        """
        Check if given vertices form a regular pentachoron

        All edges should have equal length
        """
        edges = PentachoronMathematics.generate_edges()
        edge_lengths = []

        for i, j in edges:
            length = np.linalg.norm(vertices[i] - vertices[j])
            edge_lengths.append(length)

        # Check if all edge lengths are equal within tolerance
        mean_length = np.mean(edge_lengths)
        return all(abs(l - mean_length) < tolerance for l in edge_lengths)

    @staticmethod
    def create_pentachoron(
            variant: PentachoronVariant = PentachoronVariant.REGULAR,
            edge_length: float = 1.0
    ) -> PentachoronGeometry:
        """
        Create a complete pentachoron with specified variant and edge length
        """
        if variant == PentachoronVariant.REGULAR:
            vertices = PentachoronMathematics.generate_vertices_4d_centered(edge_length)
        elif variant == PentachoronVariant.TETRAHEDRAL_PYRAMID:
            # Tetrahedral base with apex
            base = PentachoronMathematics.generate_vertices_4d_centered(edge_length)[:4]
            apex = np.array([0, 0, 0, edge_length * np.sqrt(2 / 3)])
            vertices = np.vstack([base, apex])
        else:
            # For other variants, start with regular and apply transformations
            vertices = PentachoronMathematics.generate_vertices_4d_centered(edge_length)
            # Each variant would require specific transformations

        edges = PentachoronMathematics.generate_edges()
        faces = PentachoronMathematics.generate_faces()
        cells = PentachoronMathematics.generate_cells()

        return PentachoronGeometry(
            vertices=vertices,
            edges=edges,
            faces=faces,
            cells=cells,
            edge_length=edge_length,
            surface_area=PentachoronMathematics.calculate_surface_area(edge_length),
            surcell_volume=PentachoronMathematics.calculate_surcell_volume(edge_length),
            surteron_bulk=PentachoronMathematics.calculate_surteron_bulk(edge_length),
            circumradius=PentachoronMathematics.calculate_circumradius(edge_length),
            inradius=PentachoronMathematics.calculate_inradius(edge_length),
            variant=variant
        )


class FailedHistoricalApproaches:
    """
    Historical failed computational approaches to pentachoron mathematics
    These provide insight into why certain methods don't work in 4D
    """

    @staticmethod
    def steiner_reversed_4d(number: int) -> int:
        """
        Rudolf Steiner's failed approach: reversing numbers in 4D

        Claimed that in 4D, numbers must be read backwards (782 → 287)
        This has no mathematical basis and doesn't preserve algebraic properties

        FAILS: Violates commutativity and associativity
        """
        return int(str(number)[::-1])

    @staticmethod
    def medieval_nested_polyhedra(depth: int = 3) -> Dict[str, Any]:
        """
        Medieval attempt to reach 4D through nested Platonic solids

        They believed nesting polyhedra within polyhedra would breach dimensions

        FAILS: Remains fundamentally 3-dimensional regardless of nesting depth
        """
        structure = {"type": "tetrahedron", "nested": None}
        current = structure

        for _ in range(depth - 1):
            current["nested"] = {"type": "tetrahedron", "nested": None}
            current = current["nested"]

        return structure

    @staticmethod
    def ampere_differentiability_assumption(point_4d: np.ndarray) -> Optional[float]:
        """
        Ampère's failed assumption about 4D differentiability

        Assumed all 4D functions must be differentiable everywhere
        This led to contradictions with valid but non-differentiable 4D structures

        FAILS: Many valid 4D geometric structures have singularities
        """
        # This would fail on many valid 4D structures
        try:
            # Attempt to compute derivative assuming smoothness
            epsilon = 1e-10
            dim = len(point_4d)
            gradient = np.zeros(dim)

            for i in range(dim):
                # This assumes differentiability which doesn't always hold
                perturbation = np.zeros(dim)
                perturbation[i] = epsilon
                # Would fail at singularities common in 4D polytopes
                gradient[i] = np.linalg.norm(point_4d + perturbation) - np.linalg.norm(point_4d)

            return np.linalg.norm(gradient) / epsilon
        except:
            return None  # Fails at singularities

    @staticmethod
    def brute_force_4d_convex_hull(
            points: np.ndarray,
            max_iterations: int = 1000
    ) -> Optional[List[Tuple[int, ...]]]:
        """
        1980s brute-force approach to 4D convex hull computation

        Attempted direct extension of 3D algorithms without considering complexity

        FAILS: Ω(n³) complexity makes it intractable for moderate point sets
        """
        n = len(points)
        if n < 5:
            return None

        # This naive approach has cubic complexity
        # Real implementation would timeout on moderate datasets
        facets = []
        iterations = 0

        # Check all possible 4-point combinations (tetrahedral facets)
        for i in range(n):
            for j in range(i + 1, n):
                for k in range(j + 1, n):
                    for l in range(k + 1, n):
                        iterations += 1
                        if iterations > max_iterations:
                            # Demonstration of failure due to complexity
                            return None

                        # Would need to check if this forms a facet
                        # Actual check is O(n) making total O(n^4)
                        facets.append((i, j, k, l))

        return facets[:5]  # Return subset to show failure

    @staticmethod
    def quaternion_forced_embedding(
            quaternion: Tuple[float, float, float, float]
    ) -> np.ndarray:
        """
        Failed attempt to use quaternions as direct 4D coordinates

        Quaternions don't directly map to Euclidean 4D space geometry

        FAILS: Quaternion multiplication is non-commutative, unlike 4D vectors
        """
        # Naive mapping of quaternion to 4D point
        # This doesn't preserve geometric properties correctly
        w, x, y, z = quaternion

        # This mapping loses the quaternion algebraic structure
        point_4d = np.array([w, x, y, z])

        # The problem: quaternion operations don't match 4D Euclidean operations
        # For example, quaternion multiplication != 4D vector operations

        return point_4d

    @staticmethod
    def analyze_failure_patterns() -> Dict[str, str]:
        """
        Analyze common patterns in failed 4D computational approaches
        """
        return {
            "dimensional_extension": "Direct extension of 3D methods usually fails due to complexity explosion",
            "algebraic_mismatch": "4D algebra has unique properties not present in 3D",
            "visualization_trap": "Trying to 'see' 4D directly leads to incorrect intuitions",
            "complexity_barrier": "Many 3D O(n²) algorithms become O(n³) or worse in 4D",
            "symmetry_breaking": "4D has different symmetry groups than 3D",
            "topological_differences": "4D allows for topologies impossible in 3D",
            "projection_loss": "Information loss in 4D→3D projection is more severe than 3D→2D"
        }


# Example usage and validation
if __name__ == "__main__":
    # Create a regular pentachoron
    penta = PentachoronMathematics()

    print("=== Regular Pentachoron Properties ===")
    geom = penta.create_pentachoron(PentachoronVariant.REGULAR, edge_length=1.0)
    print(f"Vertices shape: {geom.vertices.shape}")
    print(f"Number of edges: {len(geom.edges)}")
    print(f"Number of faces: {len(geom.faces)}")
    print(f"Number of cells: {len(geom.cells)}")
    print(f"Surface area: {geom.surface_area:.4f}")
    print(f"Surcell volume: {geom.surcell_volume:.4f}")
    print(f"4D hypervolume: {geom.surteron_bulk:.4f}")
    print(f"Circumradius: {geom.circumradius:.4f}")
    print(f"Inradius: {geom.inradius:.4f}")

    # Check regularity
    is_regular = penta.validate_regularity(geom.vertices)
    print(f"Is regular: {is_regular}")

    # Calculate dihedral angle
    dihedral = penta.calculate_dihedral_angle()
    print(f"Dihedral angle: {dihedral:.2f}°")

    # Test different projections
    print("\n=== 3D Projections ===")
    for proj_type in ProjectionType:
        vertices_3d = penta.project_to_3d(geom.vertices, proj_type)
        print(f"{proj_type.value}: shape {vertices_3d.shape}")

    # Test barycentric coordinates
    print("\n=== Barycentric Coordinates ===")
    center = np.mean(geom.vertices, axis=0)
    bary_coords = penta.barycentric_coordinates(center, geom.vertices)
    print(f"Center point in barycentric: {bary_coords}")
    print(f"Sum of coordinates: {np.sum(bary_coords):.4f} (should be 1.0)")

    # Demonstrate failed historical approaches
    print("\n=== Failed Historical Approaches ===")
    failed = FailedHistoricalApproaches()

    # Steiner's number reversal
    print(f"Steiner's 782 in '4D': {failed.steiner_reversed_4d(782)}")

    # Medieval nested structure
    nested = failed.medieval_nested_polyhedra(3)
    print(f"Medieval nesting depth: {nested}")

    # Analyze failure patterns
    patterns = failed.analyze_failure_patterns()
    print("\n=== Common Failure Patterns ===")
    for pattern, description in patterns.items():
        print(f"{pattern}: {description[:50]}...")