"""Protein Flood Complex (PFC) implementation.

This module implements the Protein-Aware Flood Complex, which extends
the standard Flood complex with protein-specific modifications:
- Residue-level landmark selection
- Weighted flooding radii based on atom size, SASA, and chemistry
- Heterogeneous balls for hydrophobic/charged regions
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import torch
import gudhi

from .core import generate_landmarks, flood_complex
from .protein_io import ProteinStructure
from .protein_landmarks import select_residue_landmarks, get_landmark_attributes
from .protein_landmark_sampling import protein_landmark_sampling


def compute_landmark_weights(
    atom_radii: torch.Tensor,
    sasa: torch.Tensor,
    hydrophobicity: torch.Tensor,
    charge: torch.Tensor,
    r0: float = 1.0,
    alpha: float = 1.0,
    beta: float = 0.5,
    gamma_h: float = 0.2,
    gamma_q: float = 0.2,
    r_min: Optional[float] = None,
    r_max: Optional[float] = None,
) -> torch.Tensor:
    """
    Compute protein-aware flooding weights for landmarks.
    
    Weight formula: w(ℓ) = clip((r₀ + α·r_atom + β·SASA) · (1 + γ_h·H - γ_q·Q), R_min, R_max)
    
    Args:
        atom_radii: Van der Waals radius proxies for each landmark
        sasa: Normalized SASA values [0, 1] for each landmark
        hydrophobicity: Normalized hydrophobicity scores [0, 1] for each landmark
        charge: Charge indicators [0, 1] for each landmark
        r0: Global offset (Angstroms)
        alpha: Weight for atom radius term
        beta: Weight for SASA term
        gamma_h: Weight for hydrophobicity modulation
        gamma_q: Weight for charge modulation
        r_min: Minimum weight (if None, uses 0.5 * r0)
        r_max: Maximum weight (if None, uses 2.0 * r0)
        
    Returns:
        Tensor of flooding weights for each landmark
    """
    device = atom_radii.device
    
    # Base radius term
    base_radius = r0 + alpha * atom_radii + beta * sasa
    
    # Chemistry modulation
    chemistry_factor = 1.0 + gamma_h * hydrophobicity - gamma_q * charge
    
    # Combined weight
    weights = base_radius * chemistry_factor
    
    # Clip to bounds
    if r_min is None:
        r_min = 0.5 * r0
    if r_max is None:
        r_max = 2.0 * r0
    
    weights = torch.clamp(weights, min=r_min, max=r_max)
    
    return weights


def compute_simplex_circumball(
    vertices: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute circumcenter and circumradius for a simplex.
    
    Args:
        vertices: (d+1, 3) tensor of vertex coordinates for a d-simplex
        
    Returns:
        Tuple of (circumcenter, circumradius)
    """
    d = vertices.shape[0] - 1  # Dimension of simplex
    
    if d == 0:
        # Point: circumcenter is the point itself
        return vertices[0], torch.tensor(0.0, device=vertices.device)
    
    elif d == 1:
        # Edge: circumcenter is midpoint, radius is half length
        center = (vertices[0] + vertices[1]) / 2.0
        radius = torch.norm(vertices[0] - vertices[1]) / 2.0
        return center, radius
    
    elif d == 2:
        # Triangle: use formula for circumcenter
        a, b, c = vertices[0], vertices[1], vertices[2]
        ab = b - a
        ac = c - a
        
        ab_len_sq = torch.dot(ab, ab)
        ac_len_sq = torch.dot(ac, ac)
        
        # Cross product for area
        cross = torch.cross(ab, ac)
        area_sq = torch.dot(cross, cross)
        
        if area_sq < 1e-10:
            # Degenerate triangle
            center = (a + b + c) / 3.0
            radius = torch.max(torch.norm(center - a), torch.norm(center - b))
            radius = torch.max(radius, torch.norm(center - c))
        else:
            # Circumcenter formula
            u = ab_len_sq * torch.cross(cross, ac) - ac_len_sq * torch.cross(cross, ab)
            u = u / (2.0 * area_sq)
            center = a + u
            radius = torch.norm(center - a)
        
        return center, radius
    
    elif d == 3:
        # Tetrahedron: use formula for circumcenter
        a, b, c, d_vert = vertices[0], vertices[1], vertices[2], vertices[3]
        
        ab = b - a
        ac = c - a
        ad = d_vert - a
        
        # Compute circumcenter using matrix inversion
        # For numerical stability, use a more robust method
        ab_len_sq = torch.dot(ab, ab)
        ac_len_sq = torch.dot(ac, ac)
        ad_len_sq = torch.dot(ad, ad)
        
        # Build system: 2*(b-a)·x = |b|² - |a|², etc.
        # For simplicity, use approximate method
        center = (a + b + c + d_vert) / 4.0
        
        # Iterative refinement (simplified)
        for _ in range(3):
            dists = torch.stack([
                torch.norm(center - a),
                torch.norm(center - b),
                torch.norm(center - c),
                torch.norm(center - d_vert)
            ])
            avg_dist = dists.mean()
            # Simple gradient step
            grads = torch.stack([
                (center - a) / (torch.norm(center - a) + 1e-8),
                (center - b) / (torch.norm(center - b) + 1e-8),
                (center - c) / (torch.norm(center - c) + 1e-8),
                (center - d_vert) / (torch.norm(center - d_vert) + 1e-8),
            ])
            grad = (dists - avg_dist).unsqueeze(1) * grads
            center = center - 0.1 * grad.mean(0)
        
        radius = torch.norm(center - a)
        return center, radius
    
    else:
        # For higher dimensions, use approximate method
        center = vertices.mean(dim=0)
        radius = torch.max(torch.norm(vertices - center, dim=1))
        return center, radius


def weighted_flood_inclusion_test(
    simplex_vertex_indices: List[int],
    landmark_positions: torch.Tensor,
    landmark_weights: torch.Tensor,
    epsilon: float,
    tolerance: float = 1e-6,
) -> bool:
    """
    Test if a simplex is included in the weighted flooded region at filtration value epsilon.
    
    A simplex is included if its circumball is contained in the union of weighted balls:
    U(ε) = ∪ B(ℓ, ε·w(ℓ))
    
    Args:
        simplex_vertex_indices: List of landmark indices forming the simplex
        landmark_positions: (M, 3) tensor of all landmark positions
        landmark_weights: (M,) tensor of flooding weights
        epsilon: Filtration value
        tolerance: Numerical tolerance
        
    Returns:
        True if simplex is included, False otherwise
    """
    # Get vertex coordinates
    simplex_vertices = landmark_positions[simplex_vertex_indices]
    
    # Compute circumcenter and circumradius
    circumcenter, circumradius = compute_simplex_circumball(simplex_vertices)
    
    # Compute effective radii at this epsilon
    effective_radii = epsilon * landmark_weights
    
    # Find minimum distance from circumcenter to any landmark, minus its effective radius
    # The simplex is covered if: min_ℓ(||c - ℓ|| - R(ℓ,ε)) ≤ -ρ
    distances = torch.norm(landmark_positions - circumcenter.unsqueeze(0), dim=1)
    min_coverage = torch.min(distances - effective_radii)
    
    return (min_coverage <= -circumradius + tolerance).item()


def protein_flood_complex(
    protein: ProteinStructure,
    target_landmarks: Optional[int] = None,
    max_dimension: int = 2,
    points_per_edge: Optional[int] = 30,
    num_rand: Optional[int] = None,
    batch_size: int = 64,
    use_triton: bool = True,
    r0: float = 1.0,
    alpha: float = 1.0,
    beta: float = 0.5,
    gamma_h: float = 0.2,
    gamma_q: float = 0.2,
    r_min: Optional[float] = None,
    r_max: Optional[float] = None,
    device: str = "cuda",  # Default to GPU
    return_simplex_tree: bool = True,
    use_pls: bool = False,
    pls_oversampling: float = 4.0,
    pls_lambda_u: float = 0.6,
    pls_lambda_p: float = 0.3,
    pls_lambda_c: float = 0.1,
    pls_alpha: float = 1.0,
    pocket_center: Optional[np.ndarray] = None,
    pocket_probabilities: Optional[np.ndarray] = None,
) -> Union[gudhi.SimplexTree, Dict]:
    """
    Construct a Protein Flood Complex (PFC) from a protein structure.
    
    This function extends the standard flood_complex with protein-aware weighted
    flooding radii based on atom properties, SASA, and chemistry.
    
    Two landmark selection methods are available:
    1. Simplified (default, use_pls=False): Fast residue-based selection with basic curvature bias
    2. Full PLS (use_pls=True): Complete algorithm with voxel downsampling, proper pocket/curvature
       scoring, and weighted farthest-point sampling (better for binding sites)
    
    Args:
        protein: ProteinStructure object
        target_landmarks: Target number of landmarks. If None, uses all residue landmarks.
        max_dimension: Maximum dimension of simplices (default 2 for H0, H1, H2)
        points_per_edge: Resolution for simplex sampling (default 30)
        num_rand: Number of random points per simplex (alternative to points_per_edge)
        batch_size: Batch size for processing simplices
        use_triton: Whether to use Triton kernels for GPU acceleration
        r0: Global radius offset (Angstroms)
        alpha: Weight for atom radius term
        beta: Weight for SASA term
        gamma_h: Weight for hydrophobicity modulation
        gamma_q: Weight for charge modulation
        r_min: Minimum flooding weight (default: 0.5 * r0)
        r_max: Maximum flooding weight (default: 2.0 * r0)
        device: Device for computation ('cpu' or 'cuda')
        return_simplex_tree: If True, return gudhi.SimplexTree, else return dict
        
        Landmark Selection Options:
        use_pls: If True, use full Protein Landmark Sampling (PLS) algorithm.
                 If False (default), use simplified residue-based landmark selection.
                 
                 Simplified (use_pls=False):
                 - Fast residue-level selection (backbone + sidechain centroids)
                 - Basic curvature/pocket bias approximation
                 - Good for general use, faster computation
                 
                 Full PLS (use_pls=True):
                 - Voxel grid downsample for scalability
                 - Proper pocket scoring (Gaussian from ligand center)
                 - PCA-based curvature detection
                 - Weighted farthest-point sampling
                 - Better for binding site detection, more computationally intensive
                 
        PLS Parameters (only used when use_pls=True):
        pls_oversampling: PLS oversampling factor s (default 4.0)
        pls_lambda_u: PLS uniform weight (default 0.6)
        pls_lambda_p: PLS pocket weight (default 0.3)
        pls_lambda_c: PLS curvature weight (default 0.1)
        pls_alpha: PLS importance influence in WFPS (default 1.0)
        pocket_center: Optional (3,) ligand/pocket center for pocket scoring
        pocket_probabilities: Optional (N,) pocket probabilities per atom
        
    Returns:
        Either a gudhi.SimplexTree or a dictionary mapping simplices to filtration values
    """
    device = torch.device(device)
    
    # Step A: Landmark selection
    # Option 1: Full PLS algorithm (better for binding sites, more compute)
    # Option 2: Simplified residue-based (faster, good for general use)
    if use_pls and target_landmarks is not None:
        # Use full PLS algorithm
        landmarks, landmark_to_residue, pls_weights = protein_landmark_sampling(
            protein,
            target_count=target_landmarks,
            oversampling_factor=pls_oversampling,
            lambda_u=pls_lambda_u,
            lambda_p=pls_lambda_p,
            lambda_c=pls_lambda_c,
            alpha=pls_alpha,
            pocket_center=pocket_center,
            pocket_probabilities=pocket_probabilities,
            compute_weights=True,
            device=device,
        )
        landmarks = landmarks.to(device)
        
        # Step B: Compute protein-aware weights
        # Combine PLS weights with PFC weights
        attributes = get_landmark_attributes(protein, landmark_to_residue, landmarks)
        
        # Move attributes to device
        for key in attributes:
            if isinstance(attributes[key], torch.Tensor):
                attributes[key] = attributes[key].to(device)
        
        pfc_weights = compute_landmark_weights(
            attributes["atom_radii"],
            attributes["sasa"],
            attributes["hydrophobicity"],
            attributes["charge"],
            r0=r0,
            alpha=alpha,
            beta=beta,
            gamma_h=gamma_h,
            gamma_q=gamma_q,
            r_min=r_min,
            r_max=r_max,
        )
        
        # Combine PLS and PFC weights (weighted average)
        if pls_weights is not None:
            pls_weights = pls_weights.to(device)
            # Normalize PLS weights to similar scale as PFC weights
            pls_weights_norm = pls_weights / pls_weights.mean() * pfc_weights.mean()
            # Combine: 70% PFC, 30% PLS (can be made configurable)
            landmark_weights = 0.7 * pfc_weights + 0.3 * pls_weights_norm
        else:
            landmark_weights = pfc_weights
    else:
        # Option 2: Use simplified residue-based landmark selection
        # This is faster and works well for general protein analysis
        landmarks, landmark_to_residue = select_residue_landmarks(
            protein,
            target_count=target_landmarks,
            include_sidechains=True,
        )
        landmarks = landmarks.to(device)
        
        # Step B: Compute protein-aware weights
        attributes = get_landmark_attributes(protein, landmark_to_residue, landmarks)
        
        # Move attributes to device
        for key in attributes:
            if isinstance(attributes[key], torch.Tensor):
                attributes[key] = attributes[key].to(device)
        
        landmark_weights = compute_landmark_weights(
            attributes["atom_radii"],
            attributes["sasa"],
            attributes["hydrophobicity"],
            attributes["charge"],
            r0=r0,
            alpha=alpha,
            beta=beta,
            gamma_h=gamma_h,
            gamma_q=gamma_q,
            r_min=r_min,
            r_max=r_max,
        )
    
    # Step C: Use extended flood_complex with weighted radii
    # For PFC, we use landmarks as both the complex vertices and coverage centers
    # The witness points are the landmarks themselves (with weighted radii)
    
    # Enforce GPU/Triton for weighted flooding
    if landmark_weights is not None:
        # Check device
        if device != "cuda":
            raise RuntimeError(
                "Weighted flooding (PFC) requires GPU. "
                f"Current device: {device}. Please use device='cuda'."
            )
        
        # Check landmarks are on CUDA
        if not isinstance(landmarks, torch.Tensor):
            raise RuntimeError(
                "Weighted flooding (PFC) requires landmarks to be a torch.Tensor. "
                f"Got type: {type(landmarks)}"
            )
        
        if not landmarks.is_cuda:
            raise RuntimeError(
                "Weighted flooding (PFC) requires landmarks on CUDA. "
                f"Landmarks device: {landmarks.device}. Please move landmarks to CUDA."
            )
        
        # Check Triton
        if not use_triton:
            raise RuntimeError(
                "Weighted flooding (PFC) requires Triton. "
                "Please set use_triton=True."
            )
    
    pfc_stree = flood_complex(
        points=landmarks,  # Use landmarks as witness points
        landmarks=landmarks,  # Landmarks form the Delaunay complex
        max_dimension=max_dimension,
        points_per_edge=points_per_edge,
        num_rand=num_rand,
        batch_size=batch_size,
        use_triton=use_triton,
        return_simplex_tree=return_simplex_tree,
        landmark_weights=landmark_weights,  # Enable weighted flooding
    )
    
    return pfc_stree

