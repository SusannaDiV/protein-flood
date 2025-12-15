# Landmark Selection Guide: Simplified vs Full PLS

## Overview

The Protein Flood Complex (PFC) supports two landmark selection methods:

1. **Simplified Residue-Based** (default, `use_pls=False`)
2. **Full Protein Landmark Sampling (PLS)** (`use_pls=True`)

## Quick Comparison

| Feature | Simplified | Full PLS |
|---------|-----------|----------|
| **Speed** | Fast | Slower (more compute) |
| **Pocket Detection** | Basic (density proxy) | Advanced (Gaussian/probability) |
| **Curvature Detection** | Density-based | PCA eigenvalue ratio |
| **Coverage** | Standard FPS | Weighted FPS with importance |
| **Scalability** | Good | Excellent (voxel downsample) |
| **Best For** | General analysis | Binding site detection |

## Simplified Residue-Based Selection (Default)

**When to use:**
- General protein topology analysis
- Fast computation needed
- No specific binding site focus
- Smaller proteins (< 100k atoms)

**How it works:**
- Selects backbone + sidechain centroids per residue
- Basic curvature/pocket bias using local density
- Standard farthest-point sampling for additional landmarks

**Usage:**
```python
pfc_stree = protein_flood_complex(
    protein,
    target_landmarks=500,
    use_pls=False,  # Default
)
```

## Full PLS Algorithm

**When to use:**
- Binding site / ligand binding analysis
- Need better pocket/channel detection
- Large proteins (millions of atoms)
- Have ligand center or pocket predictions

**How it works:**
1. **Voxel grid downsample**: Efficiently reduces large point clouds
2. **Pocket scoring**: Gaussian from ligand center or pocket probabilities
3. **Curvature scoring**: PCA-based eigenvalue ratio (λ₃/(λ₁+λ₂+λ₃))
4. **Stratified seeding**: Residue-stratified with quotas
5. **Weighted FPS**: Balances uniform coverage with importance

**Usage:**
```python
# With ligand center
pfc_stree = protein_flood_complex(
    protein,
    target_landmarks=500,
    use_pls=True,  # Enable full PLS
    pls_lambda_p=0.3,  # Higher pocket weight
    pocket_center=ligand_center,  # (3,) array
    device="cuda",
)

# With pocket probabilities
pfc_stree = protein_flood_complex(
    protein,
    target_landmarks=500,
    use_pls=True,
    pocket_probabilities=pocket_probs,  # (N,) array
    device="cuda",
)

# Apo protein (no ligand)
pfc_stree = protein_flood_complex(
    protein,
    target_landmarks=500,
    use_pls=True,
    pls_lambda_p=0.0,  # No pocket weight
    pls_lambda_c=0.4,  # Higher curvature weight
    pls_lambda_u=0.6,  # Uniform weight
    device="cuda",
)
```

## Performance Comparison

### Simplified Method
- **Time**: ~1-5 seconds for typical proteins
- **Memory**: Low
- **GPU**: Optional (faster with GPU)

### Full PLS Method
- **Time**: ~5-30 seconds depending on protein size
- **Memory**: Moderate (voxel grid + scores)
- **GPU**: Recommended for WFPS step

## Recommendations

### Use Simplified (`use_pls=False`) when:
- ✅ General protein topology analysis
- ✅ Fast screening of many proteins
- ✅ No specific binding site focus
- ✅ Computational resources are limited

### Use Full PLS (`use_pls=True`) when:
- ✅ Binding site detection is critical
- ✅ Ligand-bound structures available
- ✅ Need best pocket/channel detection
- ✅ Large proteins (> 100k atoms)
- ✅ Have pocket predictions available

## Example: Switching Between Methods

```python
# Fast general analysis
pfc_simple = protein_flood_complex(
    protein,
    target_landmarks=500,
    use_pls=False,  # Fast
)

# Detailed binding site analysis
pfc_pls = protein_flood_complex(
    protein,
    target_landmarks=500,
    use_pls=True,  # More accurate
    pocket_center=ligand_center,
)
```

Both methods use the same weighted flooding and circumball coverage - the difference is only in landmark selection quality.

