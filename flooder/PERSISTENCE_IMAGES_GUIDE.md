# Persistence Images: Usage Guide

## Overview

Persistence images convert variable-size persistence diagrams into fixed-size feature vectors suitable for machine learning models.

## Quick Start

### Basic Usage

```python
from flooder.protein_io import load_pdb_file
from flooder.pfc import protein_flood_complex
from flooder.persistence_vectorization import (
    compute_persistence_images_from_simplex_tree,
    get_feature_dimension,
)

# Load protein and compute PFC
protein = load_pdb_file("protein.pdb")
pfc_stree = protein_flood_complex(protein, target_landmarks=500)
pfc_stree.compute_persistence()

# Convert to persistence images (one line!)
features = compute_persistence_images_from_simplex_tree(
    pfc_stree,
    max_dimension=2,  # H0, H1, H2
    bandwidth=1.0,
    resolution=(20, 20),
    normalize=True,
)

print(f"Feature vector shape: {features.shape}")  # (1200,) for default
```

### From Persistence Diagrams

```python
from flooder.persistence_vectorization import compute_persistence_images

# If you already have diagrams
diagrams = [H0_diag, H1_diag, H2_diag]  # List of (birth, death) pairs

features = compute_persistence_images(
    diagrams,
    bandwidth=1.0,
    resolution=(20, 20),
    normalize=True,
)
```

## Parameters

### `bandwidth` (default: 1.0)
- **Gaussian kernel width** for smoothing
- **Larger** = smoother, less detail
- **Smaller** = sharper, more detail
- **Typical range**: 0.5 - 2.0

### `resolution` (default: (20, 20))
- **Grid size** for the image
- **Larger** = more detail, larger feature vector
- **Smaller** = less detail, smaller feature vector
- **Feature dimension**: `resolution[0] * resolution[1] * num_diagrams`
- **Default**: (20, 20) → 400 per diagram → 1200 total for H0, H1, H2

### `normalize` (default: True)
- **Normalize each image** to [0, 1]
- **True**: Better for ML (standardized scale)
- **False**: Preserves raw intensity values

### `weight_function` (default: persistence)
- **Weight each point** in the diagram
- **Default**: `lambda x: x[1] - x[0]` (persistence = death - birth)
- **Custom**: Can weight by other factors

## Feature Dimensions

```python
from flooder.persistence_vectorization import get_feature_dimension

# Get dimension for given parameters
dim = get_feature_dimension(
    num_diagrams=3,  # H0, H1, H2
    resolution=(20, 20)
)
print(dim)  # 1200
```

## Batch Processing

```python
from flooder.persistence_vectorization import compute_persistence_images_batch

# Process multiple proteins
diagrams_list = [
    [H0_diag_1, H1_diag_1, H2_diag_1],
    [H0_diag_2, H1_diag_2, H2_diag_2],
    ...
]

# Batch compute
features_batch = compute_persistence_images_batch(
    diagrams_list,
    bandwidth=1.0,
    resolution=(20, 20),
)

print(features_batch.shape)  # (num_proteins, 1200)
```

## Integration with Example Script

The `example_protein_pfc.py` script now automatically computes persistence images:

```bash
python example_protein_pfc.py --scpdb_dir "path/to/pdbs" --max_proteins 1
```

Output includes:
- `persistence_diagrams`: Raw diagrams
- `persistence_images`: ML-ready features (NEW!)

## ML Integration

### Simple MLP

```python
import torch
import torch.nn as nn

# Extract features
features = compute_persistence_images_from_simplex_tree(pfc_stree)

# Create MLP
class ProteinClassifier(nn.Module):
    def __init__(self, input_dim=1200):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes),
        )
    
    def forward(self, x):
        return self.network(x)

# Use features
model = ProteinClassifier()
features_tensor = torch.from_numpy(features).float().unsqueeze(0)
output = model(features_tensor)
```

### With scikit-learn

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Extract features for all proteins
X = []  # Features
y = []  # Labels

for protein, label in zip(proteins, labels):
    features = extract_features(protein)
    X.append(features)
    y.append(label)

X = np.array(X)
y = np.array(y)

# Train classifier
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
clf = RandomForestClassifier()
clf.fit(X_train, y_train)
accuracy = clf.score(X_test, y_test)
```

## Example: Complete Pipeline

```python
from pathlib import Path
from flooder.protein_io import load_pdb_file
from flooder.pfc import protein_flood_complex
from flooder.persistence_vectorization import compute_persistence_images_from_simplex_tree

def process_protein_to_features(pdb_path: Path) -> np.ndarray:
    """Complete pipeline: PDB → Features"""
    # 1. Load protein
    protein = load_pdb_file(pdb_path)
    
    # 2. Compute PFC
    pfc_stree = protein_flood_complex(
        protein,
        target_landmarks=500,
        device="cuda",
    )
    
    # 3. Compute persistence
    pfc_stree.compute_persistence()
    
    # 4. Convert to features
    features = compute_persistence_images_from_simplex_tree(
        pfc_stree,
        max_dimension=2,
        resolution=(20, 20),
    )
    
    return features

# Use it
features = process_protein_to_features(Path("protein.pdb"))
print(f"ML-ready features: {features.shape}")  # (1200,)
```

## Tips

1. **Start with default parameters**: `bandwidth=1.0`, `resolution=(20, 20)`
2. **Normalize**: Always use `normalize=True` for ML
3. **Grid size**: Larger grids (30x30, 40x40) for more detail, smaller for speed
4. **Bandwidth**: Tune based on your data - larger for smooth features, smaller for sharp
5. **Batch processing**: Use `compute_persistence_images_batch` for efficiency

## Troubleshooting

### Import Error
```python
ImportError: gudhi.representations is required
```
**Solution**: `pip install gudhi` (should already be installed)

### Empty Diagrams
- Empty diagrams → zero images (handled automatically)
- No error, just zero features

### Infinite Death Times
- Essential features (infinite death) are handled automatically
- Set to `max_finite_death * 1.1` for processing

## Next Steps

1. ✅ **Persistence Images**: Implemented and ready to use
2. ⚠️ **DeepSets**: Optional, requires training (see `VECTORIZATION_COMPARISON.md`)
3. ⚠️ **ML Training**: Need labeled data (binding sites, functions, etc.)

See `examples/example_persistence_images.py` for ML integration examples.

