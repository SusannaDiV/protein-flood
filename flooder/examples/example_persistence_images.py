"""Example: Using Persistence Images for Machine Learning.

This script demonstrates how to:
1. Compute Protein Flood Complex
2. Extract persistence diagrams
3. Convert to persistence images (ML-ready features)
4. Use in a simple MLP classifier
"""

from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from flooder.protein_io import load_pdb_file
from flooder.pfc import protein_flood_complex
from flooder.persistence_vectorization import (
    compute_persistence_images_from_simplex_tree,
    get_feature_dimension,
)


def extract_features_from_protein(
    pdb_path: Path,
    target_landmarks: int = 500,
    device: str = "cpu",
) -> np.ndarray:
    """
    Extract persistence image features from a protein.
    
    Returns:
        Feature vector ready for ML input
    """
    # Load protein
    protein = load_pdb_file(pdb_path)
    
    # Construct PFC
    pfc_stree = protein_flood_complex(
        protein,
        target_landmarks=target_landmarks,
        max_dimension=2,
        device=device,
        return_simplex_tree=True,
    )
    
    # Compute persistence
    pfc_stree.compute_persistence()
    
    # Convert to persistence images
    features = compute_persistence_images_from_simplex_tree(
        pfc_stree,
        max_dimension=2,
        bandwidth=1.0,
        resolution=(20, 20),
        normalize=True,
    )
    
    return features


class SimpleMLP(nn.Module):
    """Simple MLP for protein classification from persistence images."""
    
    def __init__(self, input_dim: int, hidden_dims: list = [128, 64], num_classes: int = 2):
        super().__init__()
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.2))
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, num_classes))
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)


def example_classification():
    """
    Example: Train a simple classifier on persistence images.
    
    This is a template - you would need:
    1. Labeled data (e.g., binding site labels, function labels)
    2. Multiple proteins to train on
    """
    print("=" * 60)
    print("Example: Persistence Images for ML")
    print("=" * 60)
    
    # Example: Extract features from a protein
    # In practice, you'd do this for many proteins
    pdb_path = Path("path/to/protein.pdb")
    
    if not pdb_path.exists():
        print(f"Example PDB not found: {pdb_path}")
        print("This is a template - replace with your data")
        return
    
    print(f"Extracting features from: {pdb_path.name}")
    features = extract_features_from_protein(pdb_path, target_landmarks=500)
    
    feature_dim = get_feature_dimension(num_diagrams=3, resolution=(20, 20))
    print(f"Feature dimension: {feature_dim}")
    print(f"Extracted features shape: {features.shape}")
    
    # Example: Create MLP
    model = SimpleMLP(input_dim=feature_dim, num_classes=2)
    print(f"\nMLP model created:")
    print(model)
    
    # Example: Forward pass
    features_tensor = torch.from_numpy(features).float().unsqueeze(0)
    output = model(features_tensor)
    print(f"\nModel output shape: {output.shape}")
    print(f"Predicted logits: {output.detach().numpy()}")
    
    print("\n" + "=" * 60)
    print("To train a classifier:")
    print("1. Extract features from all proteins")
    print("2. Collect labels (binding sites, functions, etc.)")
    print("3. Train MLP with standard PyTorch training loop")
    print("=" * 60)


def example_batch_processing():
    """
    Example: Process multiple proteins and prepare for ML.
    """
    print("=" * 60)
    print("Example: Batch Processing for ML")
    print("=" * 60)
    
    # Example: Process multiple proteins
    pdb_files = [
        Path("path/to/protein1.pdb"),
        Path("path/to/protein2.pdb"),
        Path("path/to/protein3.pdb"),
    ]
    
    all_features = []
    all_labels = []  # Would come from your dataset
    
    for pdb_path in pdb_files:
        if not pdb_path.exists():
            continue
        
        try:
            features = extract_features_from_protein(pdb_path)
            all_features.append(features)
            # all_labels.append(label)  # Add your labels here
        except Exception as e:
            print(f"Error processing {pdb_path.name}: {e}")
    
    if len(all_features) > 0:
        # Stack into batch
        X = np.stack(all_features, axis=0)
        print(f"Feature matrix shape: {X.shape}")  # (num_proteins, feature_dim)
        
        # Normalize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        print(f"Normalized features shape: {X_scaled.shape}")
        
        # Convert to PyTorch tensors
        X_tensor = torch.from_numpy(X_scaled).float()
        # y_tensor = torch.from_numpy(np.array(all_labels)).long()
        
        print(f"PyTorch tensor shape: {X_tensor.shape}")
        print("\nReady for ML training!")


if __name__ == "__main__":
    # Run examples
    example_classification()
    print("\n")
    example_batch_processing()

