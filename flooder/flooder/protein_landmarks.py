"""Residue-level landmark selection for Protein Flood Complex.

This module implements residue-aware landmark selection, including
backbone and sidechain representatives, and optional additional
landmarks using curvature/pocket-biased sampling.
"""

from typing import List, Tuple, Optional, Dict
import numpy as np
import torch

from .protein_io import ProteinStructure, BACKBONE_ATOMS


def select_residue_landmarks(
    protein: ProteinStructure,
    target_count: Optional[int] = None,
    include_sidechains: bool = True,
) -> Tuple[torch.Tensor, Dict[int, int]]:
    """
    Select residue-level landmarks (backbone + optional sidechain centroids).
    
    Args:
        protein: ProteinStructure object
        target_count: Target number of landmarks. If None, uses all residues.
        include_sidechains: If True, include sidechain centroids for residues with sidechains
        
    Returns:
        Tuple of (landmarks_tensor, landmark_to_residue_map)
        - landmarks_tensor: (M, 3) tensor of landmark coordinates
        - landmark_to_residue_map: Dict mapping landmark index to residue index
    """
    landmarks = []
    landmark_to_residue = {}
    landmark_idx = 0
    
    # Collect backbone and sidechain landmarks
    for res_idx in range(protein.num_residues):
        # Backbone landmark
        bb_atoms = protein.get_backbone_atoms(res_idx)
        if bb_atoms:
            bb_centroid = protein.get_residue_centroid(res_idx, bb_atoms)
            landmarks.append(bb_centroid)
            landmark_to_residue[landmark_idx] = res_idx
            landmark_idx += 1
        
        # Sidechain landmark (if requested and sidechain exists)
        if include_sidechains:
            sc_atoms = protein.get_sidechain_atoms(res_idx)
            if sc_atoms:
                sc_centroid = protein.get_residue_centroid(res_idx, sc_atoms)
                landmarks.append(sc_centroid)
                landmark_to_residue[landmark_idx] = res_idx
                landmark_idx += 1
    
    landmarks_array = np.array(landmarks, dtype=np.float32)
    
    # If target_count is specified and we have fewer landmarks, add more
    if target_count is not None and len(landmarks) < target_count:
        additional = target_count - len(landmarks)
        extra_landmarks, extra_mapping = _add_extra_landmarks(
            protein, additional, landmark_to_residue
        )
        landmarks_array = np.vstack([landmarks_array, extra_landmarks])
        landmark_to_residue.update(extra_mapping)
    
    # If we have more landmarks than target, subsample
    elif target_count is not None and len(landmarks) > target_count:
        # Use farthest point sampling to select diverse landmarks
        from .core import generate_landmarks
        landmarks_tensor = torch.from_numpy(landmarks_array)
        selected_landmarks = generate_landmarks(landmarks_tensor, target_count)
        
        # Update mapping for selected landmarks
        # This is approximate - in practice, you'd want to track which landmarks were selected
        selected_indices = _match_landmarks(landmarks_array, selected_landmarks.cpu().numpy())
        landmarks_array = selected_landmarks.cpu().numpy()
        landmark_to_residue = {
            new_idx: landmark_to_residue[old_idx]
            for new_idx, old_idx in enumerate(selected_indices)
        }
    
    return torch.from_numpy(landmarks_array), landmark_to_residue


def _add_extra_landmarks(
    protein: ProteinStructure,
    num_extra: int,
    existing_mapping: Dict[int, int]
) -> Tuple[np.ndarray, Dict[int, int]]:
    """
    Add additional landmarks using residue-stratified sampling with curvature/pocket bias.
    
    This implements a simplified version of Protein Landmark Sampling (PLS).
    """
    extra_landmarks = []
    extra_mapping = {}
    start_idx = len(existing_mapping)
    
    # Compute curvature/pocket scores for each residue
    # Simple approximation: use local density as proxy
    coords = protein.atom_coords
    residue_scores = []
    
    for res_idx in range(protein.num_residues):
        atom_indices = protein.residue_to_atoms.get(res_idx, [])
        if not atom_indices:
            residue_scores.append(0.0)
            continue
        
        # Compute local density around residue
        res_coords = coords[atom_indices]
        res_center = res_coords.mean(axis=0)
        
        # Count nearby atoms (within 5 Angstroms) - GPU optimized
        res_center_t = torch.from_numpy(res_center.astype(np.float32))
        coords_t = torch.from_numpy(coords.astype(np.float32))
        if torch.cuda.is_available():
            res_center_t = res_center_t.cuda()
            coords_t = coords_t.cuda()
        
        distances = torch.cdist(res_center_t.unsqueeze(0), coords_t).squeeze(0)  # (N,)
        nearby_count = (distances < 5.0).sum().cpu().item()
        residue_scores.append(float(nearby_count))
    
    residue_scores = np.array(residue_scores)
    
    # Normalize scores
    if residue_scores.max() > 0:
        residue_scores = residue_scores / residue_scores.max()
    
    # Sample residues with bias toward high scores (pockets/concave regions)
    # Use weighted sampling
    probs = residue_scores + 0.1  # Add small uniform component
    probs = probs / probs.sum()
    
    sampled_residues = np.random.choice(
        len(residue_scores),
        size=min(num_extra, len(residue_scores)),
        replace=False,
        p=probs
    )
    
    for i, res_idx in enumerate(sampled_residues):
        # Sample a random atom from this residue as landmark
        atom_indices = protein.residue_to_atoms.get(res_idx, [])
        if atom_indices:
            # Use centroid of all atoms in residue
            landmark = protein.get_residue_centroid(res_idx, atom_indices)
            extra_landmarks.append(landmark)
            extra_mapping[start_idx + i] = res_idx
    
    return np.array(extra_landmarks, dtype=np.float32), extra_mapping


def _match_landmarks(
    original: np.ndarray,
    selected: np.ndarray,
    threshold: float = 0.1
) -> List[int]:
    """Match selected landmarks to original indices based on proximity."""
    # GPU-optimized: use torch.cdist instead of scipy
    selected_t = torch.from_numpy(selected.astype(np.float32))
    original_t = torch.from_numpy(original.astype(np.float32))
    if torch.cuda.is_available():
        selected_t = selected_t.cuda()
        original_t = original_t.cuda()
    
    distances = torch.cdist(selected_t, original_t)  # (M, L)
    matches = distances.argmin(dim=1).cpu().numpy()
    
    return matches.tolist()


def get_landmark_attributes(
    protein: ProteinStructure,
    landmark_to_residue: Dict[int, int],
    landmarks: torch.Tensor
) -> Dict[str, torch.Tensor]:
    """
    Compute protein attributes for each landmark.
    
    Returns:
        Dictionary with keys:
        - 'atom_radii': van der Waals radius proxies
        - 'sasa': normalized SASA values
        - 'hydrophobicity': normalized hydrophobicity scores
        - 'charge': charge indicators (0 or 1)
        - 'residue_names': residue names for each landmark
    """
    from .protein_io import (
        get_atom_vdw_radius,
        get_residue_hydrophobicity,
        is_charged_residue,
    )
    from .protein_io import compute_simple_sasa
    
    n_landmarks = len(landmarks)
    atom_radii = []
    sasa_values = []
    hydrophobicity = []
    charge = []
    residue_names = []
    
    # Compute SASA for all atoms
    atom_sasa = compute_simple_sasa(protein)
    
    for landmark_idx in range(n_landmarks):
        res_idx = landmark_to_residue.get(landmark_idx, 0)
        
        # Get residue info
        if res_idx < len(protein.residue_info):
            chain_id, res_id, res_name = protein.residue_info[res_idx]
        else:
            res_name = "UNK"
        
        residue_names.append(res_name)
        
        # Get atoms in this residue
        atom_indices = protein.residue_to_atoms.get(res_idx, [])
        
        if atom_indices:
            # Average atom radius for residue
            atom_rads = [
                get_atom_vdw_radius(protein.atom_types[i])
                for i in atom_indices
            ]
            avg_radius = np.mean(atom_rads) if atom_rads else 1.7
            
            # Average SASA for residue
            res_sasa = np.mean(atom_sasa[atom_indices]) if atom_indices else 0.5
            
            # Residue-level properties
            res_hydro = get_residue_hydrophobicity(res_name)
            res_charge = 1.0 if is_charged_residue(res_name) else 0.0
        else:
            avg_radius = 1.7
            res_sasa = 0.5
            res_hydro = 0.5
            res_charge = 0.0
        
        atom_radii.append(avg_radius)
        sasa_values.append(res_sasa)
        hydrophobicity.append(res_hydro)
        charge.append(res_charge)
    
    return {
        "atom_radii": torch.tensor(atom_radii, dtype=torch.float32),
        "sasa": torch.tensor(sasa_values, dtype=torch.float32),
        "hydrophobicity": torch.tensor(hydrophobicity, dtype=torch.float32),
        "charge": torch.tensor(charge, dtype=torch.float32),
        "residue_names": residue_names,
    }

