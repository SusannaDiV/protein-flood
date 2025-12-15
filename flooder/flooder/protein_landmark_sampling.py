"""Full Protein Landmark Sampling (PLS) Algorithm Implementation.

This module implements the complete PLS algorithm for protein point clouds,
including voxel downsampling, pocket/curvature scoring, stratified seeding,
and weighted farthest-point sampling.
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import torch
# Note: scipy imports removed - using torch.cdist for GPU optimization
# from scipy.spatial.distance import cdist
# from scipy.spatial import KDTree
# from sklearn.decomposition import PCA

from .protein_io import ProteinStructure


def voxel_grid_downsample(
    points: np.ndarray,
    target_count: int,
    voxel_size: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Step 0: Fast prefilter using voxel grid downsample.
    
    Args:
        points: (N, 3) array of point coordinates
        target_count: Target number of points K = s*M
        voxel_size: Optional voxel size. If None, computed automatically.
        
    Returns:
        Tuple of (downsampled_points, indices) where indices map to original points
    """
    N, d = points.shape
    assert d == 3, "Points must be 3D"
    
    if N <= target_count:
        return points, np.arange(N)
    
    # Compute voxel size if not provided
    if voxel_size is None:
        # Estimate voxel size to get approximately target_count points
        bbox_size = points.max(axis=0) - points.min(axis=0)
        volume = np.prod(bbox_size)
        voxel_volume = volume / target_count
        voxel_size = np.cbrt(voxel_volume)
    
    # Compute voxel indices for each point
    min_coords = points.min(axis=0)
    voxel_indices = ((points - min_coords) / voxel_size).astype(int)
    
    # Group points by voxel
    voxel_dict = {}
    for i, v_idx in enumerate(voxel_indices):
        v_key = tuple(v_idx)
        if v_key not in voxel_dict:
            voxel_dict[v_key] = []
        voxel_dict[v_key].append(i)
    
    # For each voxel, keep point closest to voxel centroid
    downsampled_indices = []
    downsampled_points = []
    
    for v_key, point_indices in voxel_dict.items():
        voxel_center = min_coords + np.array(v_key) * voxel_size + voxel_size / 2.0
        voxel_points = points[point_indices]
        
        # Find point closest to voxel center
        distances = np.linalg.norm(voxel_points - voxel_center, axis=1)
        closest_idx = point_indices[np.argmin(distances)]
        
        downsampled_indices.append(closest_idx)
        downsampled_points.append(points[closest_idx])
    
    downsampled_points = np.array(downsampled_points)
    downsampled_indices = np.array(downsampled_indices)
    
    # If we still have too many points, randomly subsample
    if len(downsampled_points) > target_count:
        indices = np.random.choice(
            len(downsampled_points),
            size=target_count,
            replace=False
        )
        downsampled_points = downsampled_points[indices]
        downsampled_indices = downsampled_indices[indices]
    
    return downsampled_points, downsampled_indices


def compute_pocket_score(
    points: np.ndarray,
    pocket_center: Optional[np.ndarray] = None,
    pocket_probabilities: Optional[np.ndarray] = None,
    sigma: float = 8.0,
    device: str = "cpu",
) -> np.ndarray:
    """
    Step 1a: Compute pocket scores for candidates.
    
    GPU-optimized version using PyTorch for distance computation.
    
    Args:
        points: (K, 3) candidate points
        pocket_center: Optional (3,) ligand center or pocket center
        pocket_probabilities: Optional (K,) predicted pocket probabilities
        sigma: Gaussian width for distance-based scoring (default 8.0 Å)
        device: Device to use ("cpu" or "cuda")
        
    Returns:
        (K,) array of pocket scores in [0, 1]
    """
    K = len(points)
    
    if pocket_probabilities is not None:
        # Use provided pocket probabilities
        scores = pocket_probabilities.astype(np.float32)
    elif pocket_center is not None:
        # Gaussian from pocket center - GPU optimized
        points_t = torch.from_numpy(points.astype(np.float32))
        center_t = torch.from_numpy(pocket_center.astype(np.float32))
        if device == "cuda" and torch.cuda.is_available():
            points_t = points_t.cuda()
            center_t = center_t.cuda()
        
        # Compute distances
        distances = torch.norm(points_t - center_t.unsqueeze(0), dim=1)  # (K,)
        scores = torch.exp(-(distances ** 2) / (2 * sigma ** 2))
        scores = scores.cpu().numpy().astype(np.float32)
    else:
        # Buriedness proxy: use local density as approximation
        # Points with more neighbors nearby are more "buried"
        k = min(20, K // 10)  # Number of neighbors
        if k < 2:
            return np.ones(K, dtype=np.float32) * 0.5
        
        # GPU-optimized: use torch.cdist instead of KDTree
        points_t = torch.from_numpy(points.astype(np.float32))
        if device == "cuda" and torch.cuda.is_available():
            points_t = points_t.cuda()
        
        # Compute all pairwise distances
        distances = torch.cdist(points_t, points_t)  # (K, K)
        
        # Get k+1 nearest neighbors (including self)
        distances_k, _ = torch.topk(distances, k=k+1, dim=1, largest=False)  # (K, k+1)
        mean_distances = distances_k[:, 1:].mean(dim=1)  # (K,) - exclude self
        
        # Invert: lower mean distance = more buried = higher score
        max_dist = mean_distances.max()
        if max_dist > 0:
            scores = 1.0 - (mean_distances / max_dist)
        else:
            scores = torch.ones(K, dtype=torch.float32, device=points_t.device)
        
        scores = scores.cpu().numpy().astype(np.float32)
    
    # Normalize to [0, 1]
    if scores.max() > scores.min():
        scores = (scores - scores.min()) / (scores.max() - scores.min())
    else:
        scores = np.ones(K, dtype=np.float32) * 0.5
    
    return scores.astype(np.float32)


def compute_curvature_score(
    points: np.ndarray,
    k: int = 30,
    device: str = "cpu",
) -> np.ndarray:
    """
    Step 1b: Compute curvature scores via local PCA.
    
    GPU-optimized version using PyTorch for batched computation.
    
    Args:
        points: (K, 3) candidate points
        k: Number of neighbors for PCA (default 30)
        device: Device to use ("cpu" or "cuda")
        
    Returns:
        (K,) array of curvature scores in [0, 1]
    """
    K = len(points)
    k = min(k, K - 1)
    
    if k < 3:
        return np.ones(K, dtype=np.float32) * 0.5
    
    # Convert to PyTorch tensor
    points_t = torch.from_numpy(points.astype(np.float32))
    if device == "cuda" and torch.cuda.is_available():
        points_t = points_t.cuda()
    
    # Compute all pairwise distances
    # (K, K) distance matrix
    distances = torch.cdist(points_t, points_t)  # (K, K)
    
    # Get k+1 nearest neighbors (including self)
    _, neighbor_indices = torch.topk(distances, k=k+1, dim=1, largest=False)  # (K, k+1)
    neighbor_indices = neighbor_indices[:, 1:]  # (K, k) - exclude self
    
    # Get neighbor points for each point
    # (K, k, 3) - for each of K points, k neighbors with 3D coords
    batch_indices = torch.arange(K, device=points_t.device).unsqueeze(1).expand(-1, k)  # (K, k)
    neighbor_points = points_t[neighbor_indices]  # (K, k, 3)
    center_points = points_t.unsqueeze(1)  # (K, 1, 3)
    
    # Center neighbors around each point
    centered = neighbor_points - center_points  # (K, k, 3)
    
    # Compute covariance matrices for all points in batch
    # For each point: cov = (1/(k-1)) * centered^T @ centered
    # centered: (K, k, 3) -> we want (K, 3, 3) covariance matrices
    k_float = float(k - 1) if k > 1 else 1.0
    cov = torch.bmm(centered.transpose(1, 2), centered) / k_float  # (K, 3, 3)
    
    # Compute eigenvalues using SVD (more stable than eig)
    # For symmetric matrices, SVD gives eigenvalues
    try:
        # Use torch.linalg.eigvalsh for Hermitian matrices (faster and more stable)
        eigenvalues, _ = torch.linalg.eigh(cov)  # (K, 3) - eigenvalues for each point
        eigenvalues = torch.sort(eigenvalues, dim=1, descending=True)[0]  # (K, 3) descending
        
        lambda1 = eigenvalues[:, 0]  # (K,)
        lambda2 = eigenvalues[:, 1]  # (K,)
        lambda3 = eigenvalues[:, 2]  # (K,)
        
        # Curvature proxy: λ₃ / (λ₁ + λ₂ + λ₃)
        denom = lambda1 + lambda2 + lambda3
        curvature_scores = torch.where(
            denom > 1e-10,
            lambda3 / denom,
            torch.full((K,), 0.5, device=points_t.device, dtype=torch.float32)
        )
    except:
        # Fallback: use centroid approximation
        curvature_scores = torch.full((K,), 0.5, device=points_t.device, dtype=torch.float32)
    
    # Normalize to [0, 1]
    scores_min = curvature_scores.min()
    scores_max = curvature_scores.max()
    if scores_max > scores_min:
        curvature_scores = (curvature_scores - scores_min) / (scores_max - scores_min)
    else:
        curvature_scores = torch.full((K,), 0.5, device=points_t.device, dtype=torch.float32)
    
    # Convert back to numpy
    return curvature_scores.cpu().numpy().astype(np.float32)


def compute_importance_scores(
    candidates: np.ndarray,
    lambda_u: float = 0.6,
    lambda_p: float = 0.3,
    lambda_c: float = 0.1,
    pocket_center: Optional[np.ndarray] = None,
    pocket_probabilities: Optional[np.ndarray] = None,
    pocket_sigma: float = 8.0,
    curvature_k: int = 30,
    device: str = "cpu",
) -> np.ndarray:
    """
    Step 1: Compute combined importance scores.
    
    Args:
        candidates: (K, 3) candidate points
        lambda_u: Weight for uniform component
        lambda_p: Weight for pocket component
        lambda_c: Weight for curvature component
        pocket_center: Optional pocket/ligand center
        pocket_probabilities: Optional pocket probabilities
        pocket_sigma: Gaussian width for pocket scoring
        curvature_k: Number of neighbors for curvature computation
        
    Returns:
        (K,) array of importance scores
    """
    assert abs(lambda_u + lambda_p + lambda_c - 1.0) < 1e-6, "Weights must sum to 1"
    
    K = len(candidates)
    
    # Uniform component
    uniform_scores = np.ones(K, dtype=np.float32)
    
    # Pocket scores
    pocket_scores = compute_pocket_score(
        candidates,
        pocket_center=pocket_center,
        pocket_probabilities=pocket_probabilities,
        sigma=pocket_sigma,
        device=device,
    )
    
    # Curvature scores
    curvature_scores = compute_curvature_score(candidates, k=curvature_k, device=device)
    
    # Combined score
    importance_scores = (
        lambda_u * uniform_scores +
        lambda_p * pocket_scores +
        lambda_c * curvature_scores
    )
    
    return importance_scores


def stratified_importance_sample(
    candidates: np.ndarray,
    candidate_indices: np.ndarray,
    importance_scores: np.ndarray,
    target_count: int,
    residue_ids: Optional[np.ndarray] = None,
    q_max: int = 5,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Step 2: Stratified probabilistic seeding with residue awareness.
    
    Args:
        candidates: (K, 3) candidate points
        candidate_indices: (K,) indices mapping to original points
        importance_scores: (K,) importance scores
        target_count: Target number of seeds m₀ ≈ 0.1*M
        residue_ids: Optional (K,) residue IDs for stratification
        q_max: Maximum quota per residue
        
    Returns:
        Tuple of (selected_points, selected_indices)
    """
    K = len(candidates)
    m0 = min(target_count, K)
    
    if residue_ids is not None and len(np.unique(residue_ids)) > 1:
        # Residue-stratified sampling
        unique_residues = np.unique(residue_ids)
        selected_indices = []
        
        # Compute quota for each residue
        residue_counts = {}
        for res_id in unique_residues:
            residue_counts[res_id] = np.sum(residue_ids == res_id)
        
        total_candidates = len(candidates)
        
        for res_id in unique_residues:
            # Quota: proportional to residue size, clipped
            residue_candidates = np.where(residue_ids == res_id)[0]
            q_r = min(
                max(1, int(np.ceil(m0 * len(residue_candidates) / total_candidates))),
                q_max
            )
            q_r = min(q_r, len(residue_candidates))
            
            if q_r == 0:
                continue
            
            # Sample q_r points from this residue with probability ∝ importance
            residue_scores = importance_scores[residue_candidates]
            residue_scores = residue_scores + 1e-6  # Avoid zeros
            probs = residue_scores / residue_scores.sum()
            
            selected_from_residue = np.random.choice(
                len(residue_candidates),
                size=q_r,
                replace=False,
                p=probs
            )
            selected_indices.extend(residue_candidates[selected_from_residue])
        
        # If we need more, fill randomly
        if len(selected_indices) < m0:
            remaining = set(range(K)) - set(selected_indices)
            remaining = list(remaining)
            if remaining:
                additional = min(m0 - len(selected_indices), len(remaining))
                additional_indices = np.random.choice(
                    remaining,
                    size=additional,
                    replace=False
                )
                selected_indices.extend(additional_indices)
        
        selected_indices = np.array(selected_indices[:m0])
    else:
        # No residue info: sample with probability proportional to importance
        probs = importance_scores + 1e-6
        probs = probs / probs.sum()
        selected_indices = np.random.choice(
            K,
            size=m0,
            replace=False,
            p=probs
        )
    
    return candidates[selected_indices], candidate_indices[selected_indices]


def weighted_farthest_point_sampling(
    candidates: np.ndarray,
    initial_landmarks: np.ndarray,
    importance_scores: np.ndarray,
    target_count: int,
    alpha: float = 1.0,
    epsilon: float = 1e-3,
    device: str = "cpu",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Step 3: Weighted farthest-point sampling (WFPS).
    
    Args:
        candidates: (K, 3) candidate points
        initial_landmarks: (m0, 3) initial seed landmarks
        importance_scores: (K,) importance scores
        target_count: Target number of landmarks M
        alpha: Importance influence parameter [0.5, 2]
        epsilon: Small constant to avoid zeroing uniform terms
        device: Computation device
        
    Returns:
        Tuple of (selected_landmarks, selected_indices)
    """
    K = len(candidates)
    M = target_count
    m0 = len(initial_landmarks)
    
    if M <= m0:
        # Find indices of initial landmarks in candidates - GPU optimized
        candidates_t = torch.from_numpy(candidates.astype(np.float32))
        initial_t = torch.from_numpy(initial_landmarks.astype(np.float32))
        if device == "cuda" and torch.cuda.is_available():
            candidates_t = candidates_t.cuda()
            initial_t = initial_t.cuda()
        
        distances = torch.cdist(initial_t, candidates_t)  # (m0, K)
        initial_indices = distances.argmin(dim=1).cpu().numpy()  # (m0,)
        return initial_landmarks, initial_indices
    
    # Convert to torch for GPU acceleration if available
    candidates_t = torch.from_numpy(candidates.astype(np.float32))
    importance_t = torch.from_numpy(importance_scores.astype(np.float32))
    
    if device == "cuda" and torch.cuda.is_available():
        candidates_t = candidates_t.cuda()
        importance_t = importance_t.cuda()
    
    # Initialize landmarks
    landmarks = [initial_landmarks]
    landmark_indices = []
    
    # Find initial landmark indices in candidates - GPU optimized
    candidates_t = torch.from_numpy(candidates.astype(np.float32))
    initial_t = torch.from_numpy(initial_landmarks.astype(np.float32))
    if device == "cuda" and torch.cuda.is_available():
        candidates_t = candidates_t.cuda()
        initial_t = initial_t.cuda()
    
    # Find nearest candidate for each initial landmark
    distances = torch.cdist(initial_t, candidates_t)  # (m0, K)
    init_indices = distances.argmin(dim=1).cpu().numpy()  # (m0,)
    landmark_indices.extend(init_indices.tolist())
    
    # Initialize distances: d(c) = min_{l in L} ||c - l||
    landmarks_t = torch.from_numpy(initial_landmarks.astype(np.float32))
    if device == "cuda" and torch.cuda.is_available():
        landmarks_t = landmarks_t.cuda()
    
    # Compute distances from all candidates to current landmarks
    distances = torch.cdist(candidates_t, landmarks_t)  # (K, m0)
    min_distances = distances.min(dim=1)[0]  # (K,)
    
    # Iterate until we have M landmarks
    while len(landmark_indices) < M:
        # Compute selection key: d(c) * (ε + S(c))^α
        keys = min_distances * (epsilon + importance_t) ** alpha
        
        # Mask out already selected landmarks
        mask = torch.ones(K, dtype=torch.bool, device=keys.device)
        mask[landmark_indices] = False
        
        if mask.sum() == 0:
            break
        
        keys_masked = keys.clone()
        keys_masked[~mask] = -torch.inf
        
        # Select candidate with maximum key
        c_star_idx = keys_masked.argmax().item()
        c_star = candidates[c_star_idx]
        
        # Add to landmarks
        landmarks.append(c_star.reshape(1, -1))
        landmark_indices.append(c_star_idx)
        
        # Update distances: d(c) = min(d(c), ||c - c*||)
        c_star_t = candidates_t[c_star_idx:c_star_idx+1]  # (1, 3)
        new_distances = torch.cdist(candidates_t, c_star_t).squeeze(1)  # (K,)
        min_distances = torch.minimum(min_distances, new_distances)
    
    # Convert back to numpy
    selected_landmarks = np.vstack(landmarks)
    selected_indices = np.array(landmark_indices)
    
    return selected_landmarks, selected_indices


def compute_pls_landmark_weights(
    landmarks: np.ndarray,
    pocket_scores: Optional[np.ndarray] = None,
    curvature_scores: Optional[np.ndarray] = None,
    density_estimate: Optional[np.ndarray] = None,
    eta0: float = 1.0,
    eta1: float = 0.2,
    eta2: float = 0.1,
    eta3: float = 0.1,
    w_min: float = 0.5,
    w_max: float = 2.0,
) -> np.ndarray:
    """
    Step 4: Assign per-landmark flooding weights.
    
    Args:
        landmarks: (M, 3) selected landmarks
        pocket_scores: Optional (M,) pocket scores
        curvature_scores: Optional (M,) curvature scores
        density_estimate: Optional (M,) local density estimates
        eta0: Base weight
        eta1: Pocket weight coefficient
        eta2: Curvature weight coefficient
        eta3: Density weight coefficient
        w_min: Minimum weight
        w_max: Maximum weight
        
    Returns:
        (M,) array of landmark weights
    """
    M = len(landmarks)
    
    # Initialize base weight
    weights = np.ones(M, dtype=np.float32) * eta0
    
    # Add pocket contribution
    if pocket_scores is not None:
        weights += eta1 * pocket_scores
    
    # Add curvature contribution
    if curvature_scores is not None:
        weights += eta2 * curvature_scores
    
    # Add density contribution
    if density_estimate is not None:
        weights += eta3 * density_estimate
    else:
        # Estimate density from kNN distances - GPU optimized
        if M > 1:
            landmarks_t = torch.from_numpy(landmarks.astype(np.float32))
            device = "cuda" if torch.cuda.is_available() else "cpu"
            if device == "cuda":
                landmarks_t = landmarks_t.cuda()
            
            k = min(5, M - 1)
            # Compute all pairwise distances
            distances = torch.cdist(landmarks_t, landmarks_t)  # (M, M)
            # Get k+1 nearest neighbors (including self)
            distances_k, _ = torch.topk(distances, k=k+1, dim=1, largest=False)  # (M, k+1)
            mean_distances = distances_k[:, 1:].mean(dim=1).cpu().numpy()  # (M,) - exclude self
            
            # Normalize density estimate
            if mean_distances.max() > mean_distances.min():
                density_norm = (mean_distances - mean_distances.min()) / (
                    mean_distances.max() - mean_distances.min()
                )
            else:
                density_norm = np.ones(M, dtype=np.float32) * 0.5
            weights += eta3 * density_norm
    
    # Clip to bounds
    weights = np.clip(weights, w_min, w_max)
    
    return weights.astype(np.float32)


def protein_landmark_sampling(
    protein: ProteinStructure,
    target_count: int,
    oversampling_factor: float = 4.0,
    lambda_u: float = 0.6,
    lambda_p: float = 0.3,
    lambda_c: float = 0.1,
    alpha: float = 1.0,
    pocket_center: Optional[np.ndarray] = None,
    pocket_probabilities: Optional[np.ndarray] = None,
    pocket_sigma: float = 8.0,
    curvature_k: int = 30,
    q_max: int = 5,
    compute_weights: bool = True,
    device: str = "cpu",
) -> Tuple[torch.Tensor, Dict[int, int], Optional[torch.Tensor]]:
    """
    Full Protein Landmark Sampling (PLS) Algorithm.
    
    Args:
        protein: ProteinStructure object
        target_count: Target number of landmarks M
        oversampling_factor: Oversampling factor s (default 4.0)
        lambda_u: Uniform weight (default 0.6)
        lambda_p: Pocket weight (default 0.3)
        lambda_c: Curvature weight (default 0.1)
        alpha: Importance influence in WFPS (default 1.0)
        pocket_center: Optional (3,) ligand/pocket center
        pocket_probabilities: Optional (N,) pocket probabilities per atom
        pocket_sigma: Gaussian width for pocket scoring (default 8.0 Å)
        curvature_k: Number of neighbors for curvature (default 30)
        q_max: Maximum quota per residue (default 5)
        compute_weights: Whether to compute PLS-based weights
        device: Computation device
        
    Returns:
        Tuple of (landmarks, landmark_to_residue_map, optional_weights)
    """
    # Get point cloud
    points = protein.atom_coords  # (N, 3)
    N = len(points)
    
    # Get residue IDs if available
    residue_ids = None
    # ProteinStructure stores residue_ids per atom
    if hasattr(protein, 'residue_ids') and protein.residue_ids is not None:
        residue_ids = np.array(protein.residue_ids)
    elif hasattr(protein, 'residue_info'):
        # Build residue IDs from residue_info mapping
        residue_ids = np.zeros(N, dtype=int)
        for res_idx, atom_indices in protein.residue_to_atoms.items():
            for atom_idx in atom_indices:
                if atom_idx < N:
                    residue_ids[atom_idx] = res_idx
    
    # Step 0: Voxel grid downsample
    K = int(oversampling_factor * target_count)
    candidates, candidate_indices = voxel_grid_downsample(points, K)
    
    # Map candidate residue IDs
    candidate_residue_ids = None
    if residue_ids is not None:
        candidate_residue_ids = residue_ids[candidate_indices]
    
    # Map pocket probabilities to candidates if provided
    candidate_pocket_probs = None
    if pocket_probabilities is not None:
        candidate_pocket_probs = pocket_probabilities[candidate_indices]
    
    # Step 1: Compute importance scores
    importance_scores = compute_importance_scores(
        candidates,
        lambda_u=lambda_u,
        lambda_p=lambda_p,
        lambda_c=lambda_c,
        pocket_center=pocket_center,
        pocket_probabilities=candidate_pocket_probs,
        pocket_sigma=pocket_sigma,
        curvature_k=curvature_k,
        device=device,
    )
    
    # Store pocket and curvature scores for weight computation
    pocket_scores = compute_pocket_score(
        candidates,
        pocket_center=pocket_center,
        pocket_probabilities=candidate_pocket_probs,
        sigma=pocket_sigma,
        device=device,
    )
    curvature_scores = compute_curvature_score(candidates, k=curvature_k, device=device)
    
    # Step 2: Stratified probabilistic seeding
    m0 = max(1, int(0.1 * target_count))
    initial_landmarks, initial_indices = stratified_importance_sample(
        candidates,
        candidate_indices,
        importance_scores,
        m0,
        residue_ids=candidate_residue_ids,
        q_max=q_max,
    )
    
    # Step 3: Weighted farthest-point sampling
    selected_landmarks, selected_candidate_indices = weighted_farthest_point_sampling(
        candidates,
        initial_landmarks,
        importance_scores,
        target_count,
        alpha=alpha,
        device=device,
    )
    
    # Map back to original point indices
    selected_original_indices = candidate_indices[selected_candidate_indices]
    
    # Build landmark to residue mapping
    landmark_to_residue = {}
    if residue_ids is not None:
        for i, orig_idx in enumerate(selected_original_indices):
            if orig_idx < len(residue_ids):
                landmark_to_residue[i] = int(residue_ids[orig_idx])
    else:
        # Fallback: use atom index as residue proxy
        for i, orig_idx in enumerate(selected_original_indices):
            landmark_to_residue[i] = orig_idx
    
    # Step 4: Compute landmark weights (optional)
    weights = None
    if compute_weights:
        # Get scores for selected landmarks
        selected_pocket = pocket_scores[selected_candidate_indices]
        selected_curvature = curvature_scores[selected_candidate_indices]
        
        weights = compute_pls_landmark_weights(
            selected_landmarks,
            pocket_scores=selected_pocket,
            curvature_scores=selected_curvature,
        )
        weights = torch.from_numpy(weights)
    
    landmarks_tensor = torch.from_numpy(selected_landmarks.astype(np.float32))
    
    return landmarks_tensor, landmark_to_residue, weights

