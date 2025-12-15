# Full Protein Landmark Sampling (PLS) Implementation

## Summary

The complete PLS algorithm has been implemented as specified, providing superior landmark selection for protein structures with explicit pocket and curvature detection.

## Implementation Status

### ✅ Step 0: Voxel Grid Downsample
- **Function**: `voxel_grid_downsample()`
- **Location**: `flooder/flooder/protein_landmark_sampling.py`
- **Features**:
  - Automatic voxel size computation
  - Representative point selection (closest to voxel centroid)
  - Linear time complexity O(N)
  - Handles millions of points efficiently

### ✅ Step 1: Importance Score Computation
- **Functions**: 
  - `compute_pocket_score()`: Three modes (ligand center, pocket probabilities, buriedness proxy)
  - `compute_curvature_score()`: PCA-based eigenvalue ratio
  - `compute_importance_scores()`: Combined scoring
- **Features**:
  - Gaussian pocket scoring from ligand centers
  - PCA-based curvature detection (λ₃/(λ₁+λ₂+λ₃))
  - Buriedness proxy for apo proteins
  - Normalized scores [0, 1]

### ✅ Step 2: Stratified Probabilistic Seeding
- **Function**: `stratified_importance_sample()`
- **Features**:
  - Residue-stratified bins with quotas
  - Prevents landmark collapse into dense regions
  - Quota clipping (q_max parameter)
  - Importance-weighted sampling within each residue

### ✅ Step 3: Weighted Farthest-Point Sampling (WFPS)
- **Function**: `weighted_farthest_point_sampling()`
- **Features**:
  - GPU-accelerated distance computation
  - Selection key: `d(c) · (ε + S(c))^α`
  - Balances uniform coverage with importance
  - Efficient distance updates

### ✅ Step 4: Per-Landmark Weights
- **Function**: `compute_pls_landmark_weights()`
- **Features**:
  - Combines pocket, curvature, and density
  - Bounded weights for stability
  - Optional density estimation from kNN

## Main Function

### `protein_landmark_sampling()`

Complete PLS pipeline with all steps integrated.

**Usage**:
```python
from flooder.protein_landmark_sampling import protein_landmark_sampling

landmarks, landmark_to_residue, weights = protein_landmark_sampling(
    protein,
    target_count=500,
    oversampling_factor=4.0,
    lambda_u=0.6,
    lambda_p=0.3,
    lambda_c=0.1,
    alpha=1.0,
    pocket_center=ligand_center,  # Optional
    device="cuda",
)
```

## Integration with PFC

The full PLS is now integrated into `protein_flood_complex()`:

```python
pfc_stree = protein_flood_complex(
    protein,
    target_landmarks=500,
    use_pls=True,  # Enable full PLS
    pls_lambda_p=0.3,  # Higher pocket weight
    pocket_center=ligand_center,  # If available
    device="cuda",
)
```

When `use_pls=True`:
- Uses full PLS algorithm for landmark selection
- Combines PLS weights with PFC weights (70% PFC, 30% PLS)
- Provides better pocket/channel detection

## Default Parameters

Following the specification:

- `s = 4.0` (oversampling factor)
- `λ = (0.6, 0.3, 0.1)` (uniform, pocket, curvature)
- `α = 1.0` (importance influence)
- `m₀ = 0.1·M` (initial seed count)
- `σ = 8.0` Å (pocket Gaussian width)
- `k = 30` (curvature neighbors)

## Performance

- **Voxel downsample**: O(N) - linear in point cloud size
- **Curvature computation**: O(K·k) where K = s·M, k = 30
- **WFPS**: O(K·M) with GPU acceleration
- **Overall**: Efficient for proteins with millions of atoms

## Advantages Over Simplified Version

1. **Better pocket detection**: Gaussian scoring from ligand centers
2. **Geometric accuracy**: PCA-based curvature captures concave regions
3. **Uniform coverage**: WFPS ensures no region is missed
4. **Scalability**: Voxel downsample handles large proteins
5. **Biochemical awareness**: Residue stratification prevents collapse

## Comparison

| Feature | Simplified | Full PLS |
|---------|-----------|----------|
| Pocket detection | Density proxy | Gaussian/Probability |
| Curvature | Density-based | PCA eigenvalue ratio |
| Coverage | Basic FPS | Weighted FPS |
| Scalability | Limited | Voxel downsample |
| Stratification | Basic | Quota-based |

The full PLS implementation provides significantly better results for protein function and ligand binding site detection.

