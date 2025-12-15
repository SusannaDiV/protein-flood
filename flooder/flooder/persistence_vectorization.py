"""Persistence diagram vectorization for machine learning.

This module provides functions to convert persistence diagrams into fixed-size
feature vectors suitable for machine learning models.

Two approaches are available:
1. Persistence Images: Grid-based fixed representation (recommended for baseline)
2. Learnable DeepSets: Neural network-based (requires training, more expressive)
"""

from typing import List, Tuple, Optional, Union
import numpy as np
import torch

try:
    from gudhi.representations import PersistenceImage
    GUDHI_REPRESENTATIONS_AVAILABLE = True
except ImportError:
    GUDHI_REPRESENTATIONS_AVAILABLE = False
    PersistenceImage = None


def compute_persistence_images(
    diagrams: List[np.ndarray],
    bandwidth: float = 1.0,
    resolution: Tuple[int, int] = (20, 20),
    weight_function: Optional[callable] = None,
    normalize: bool = True,
) -> np.ndarray:
    """
    Convert persistence diagrams to persistence images.
    
    Persistence images are fixed-size grid-based representations of persistence
    diagrams, suitable for machine learning models.
    
    Args:
        diagrams: List of persistence diagrams, each as (N, 2) array of (birth, death) pairs.
                 Can be list of lists or list of numpy arrays.
                 Typically [H0_diagram, H1_diagram, H2_diagram]
        bandwidth: Bandwidth for Gaussian kernel (default: 1.0)
        resolution: Grid resolution (width, height) (default: (20, 20))
        weight_function: Optional function to weight points. If None, uses persistence (death - birth).
                        Should take (birth, death) and return weight.
        normalize: If True, normalize each image to [0, 1] (default: True)
        
    Returns:
        Concatenated feature vector of shape (sum(resolution[0] * resolution[1] * len(diagrams)),)
        For default: (20 * 20 * 3,) = (1200,) for H0, H1, H2
        
    Example:
        >>> diagrams = [H0_diag, H1_diag, H2_diag]  # Each is list of (birth, death) tuples
        >>> features = compute_persistence_images(diagrams)
        >>> features.shape  # (1200,)
    """
    if not GUDHI_REPRESENTATIONS_AVAILABLE:
        raise ImportError(
            "gudhi.representations is required for persistence images. "
            "Install with: pip install gudhi"
        )
    
    # Default weight function: persistence (death - birth)
    if weight_function is None:
        weight_function = lambda x: x[1] - x[0] if x[1] != float('inf') else 0.0
    
    # Convert diagrams to numpy arrays if needed
    processed_diagrams = []
    for diag in diagrams:
        if isinstance(diag, list):
            # Convert list of tuples to numpy array
            diag_array = np.array(diag, dtype=np.float32)
        else:
            diag_array = np.array(diag, dtype=np.float32)
        
        # Filter out infinite death times (essential features)
        # For essential features, we can either:
        # 1. Set death to a large value (e.g., max finite death + 1)
        # 2. Remove them
        # Here we'll set them to a large value based on finite deaths
        if len(diag_array) > 0:
            finite_deaths = diag_array[diag_array[:, 1] != float('inf'), 1]
            if len(finite_deaths) > 0:
                max_finite_death = np.max(finite_deaths)
                # Set infinite deaths to max + 10% of range
                diag_array[diag_array[:, 1] == float('inf'), 1] = max_finite_death * 1.1
            else:
                # All are infinite, use birth values
                max_birth = np.max(diag_array[:, 0])
                diag_array[diag_array[:, 1] == float('inf'), 1] = max_birth * 1.1
        
        processed_diagrams.append(diag_array)
    
    # Create persistence image transformer
    pi = PersistenceImage(
        bandwidth=bandwidth,
        weight=weight_function,
        resolution=list(resolution),
    )
    
    # Fit on all diagrams to determine bounds
    all_diagrams = processed_diagrams
    pi.fit(all_diagrams)
    
    # Transform each diagram to image
    images = []
    for diag in processed_diagrams:
        if len(diag) == 0:
            # Empty diagram -> zero image
            img = np.zeros(resolution[0] * resolution[1], dtype=np.float32)
        else:
            img = pi.transform([diag])[0]  # (resolution[0] * resolution[1],)
            
            # Normalize if requested
            if normalize:
                img_max = img.max()
                if img_max > 0:
                    img = img / img_max
        
        images.append(img)
    
    # Concatenate all images
    feature_vector = np.concatenate(images, axis=0)
    
    return feature_vector.astype(np.float32)


def compute_persistence_images_from_simplex_tree(
    simplex_tree,
    max_dimension: int = 2,
    bandwidth: float = 1.0,
    resolution: Tuple[int, int] = (20, 20),
    weight_function: Optional[callable] = None,
    normalize: bool = True,
) -> np.ndarray:
    """
    Compute persistence images directly from a gudhi SimplexTree.
    
    Convenience function that extracts diagrams and converts to images in one step.
    
    Args:
        simplex_tree: gudhi.SimplexTree with computed persistence
        max_dimension: Maximum homology dimension (default: 2 for H0, H1, H2)
        bandwidth: Bandwidth for Gaussian kernel
        resolution: Grid resolution (width, height)
        weight_function: Optional weight function
        normalize: Whether to normalize images
        
    Returns:
        Concatenated feature vector
    """
    # Extract persistence diagrams
    diagrams = []
    for dim in range(max_dimension + 1):
        diag = simplex_tree.persistence_intervals_in_dimension(dim)
        diagrams.append(diag)
    
    # Convert to persistence images
    return compute_persistence_images(
        diagrams,
        bandwidth=bandwidth,
        resolution=resolution,
        weight_function=weight_function,
        normalize=normalize,
    )


def compute_persistence_images_batch(
    diagrams_list: List[List[np.ndarray]],
    bandwidth: float = 1.0,
    resolution: Tuple[int, int] = (20, 20),
    weight_function: Optional[callable] = None,
    normalize: bool = True,
) -> np.ndarray:
    """
    Compute persistence images for a batch of proteins.
    
    Args:
        diagrams_list: List of diagram lists, one per protein.
                      Each element is [H0_diag, H1_diag, H2_diag]
        bandwidth: Bandwidth for Gaussian kernel
        resolution: Grid resolution
        weight_function: Optional weight function
        normalize: Whether to normalize images
        
    Returns:
        Array of shape (batch_size, feature_dim) where feature_dim = resolution[0] * resolution[1] * num_diagrams
    """
    features_list = []
    
    for diagrams in diagrams_list:
        features = compute_persistence_images(
            diagrams,
            bandwidth=bandwidth,
            resolution=resolution,
            weight_function=weight_function,
            normalize=normalize,
        )
        features_list.append(features)
    
    return np.stack(features_list, axis=0)


def get_feature_dimension(
    num_diagrams: int = 3,
    resolution: Tuple[int, int] = (20, 20),
) -> int:
    """
    Get the dimension of the feature vector for given parameters.
    
    Args:
        num_diagrams: Number of persistence diagrams (default: 3 for H0, H1, H2)
        resolution: Grid resolution (width, height)
        
    Returns:
        Total feature dimension
    """
    return resolution[0] * resolution[1] * num_diagrams

