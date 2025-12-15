# PFC Refactoring: Extending flood_complex with Weighted Radii

## Summary

The Protein Flood Complex (PFC) implementation has been refactored to extend the existing `flood_complex` function with weighted radii support, rather than implementing a separate algorithm. This leverages all existing GPU optimizations and infrastructure.

## Key Changes

### 1. Extended `flood_complex` in `core.py`

- Added `landmark_weights` parameter (optional `torch.Tensor` of shape `(N_l,)`)
- When weights are provided, computes weighted distances: `min_ℓ(||p - ℓ|| / w(ℓ))`
- This gives the minimum epsilon needed to cover each point on a simplex
- Maintains backward compatibility: if `landmark_weights=None`, behaves as before

### 2. Refactored `protein_flood_complex` in `pfc.py`

- Now calls the extended `flood_complex` instead of implementing its own algorithm
- Uses landmarks as both:
  - Delaunay complex vertices (via `landmarks` parameter)
  - Coverage centers (via `points` parameter, set to landmarks)
- Passes computed `landmark_weights` to enable weighted flooding

### 3. Benefits

- **Reuses GPU optimizations**: All Triton kernels and GPU acceleration work automatically
- **Consistent API**: Same interface as standard flood_complex
- **Maintainable**: Single code path for both uniform and weighted flooding
- **Efficient**: Leverages existing batch processing and optimizations

## Algorithm Details

### Weighted Flooding Formula

For a point `p` on a simplex and landmark `ℓ` with weight `w(ℓ)`:
- At filtration value `ε`, landmark `ℓ` has radius `ε·w(ℓ)`
- Point `p` is covered if: `||p - ℓ|| ≤ ε·w(ℓ)` for some `ℓ`
- Equivalently: `||p - ℓ|| / w(ℓ) ≤ ε`
- Minimum epsilon to cover `p`: `min_ℓ(||p - ℓ|| / w(ℓ))`
- Filtration value for simplex: `max_p min_ℓ(||p - ℓ|| / w(ℓ))`

### Implementation

```python
# In flood_complex, when landmark_weights is provided:
dists_to_landmarks = ||points_on_simplex - landmarks||  # (batch, num_points, n_lms)
weighted_dists = dists_to_landmarks / landmark_weights   # Divide by weights
distances = min(weighted_dists, dim=2)                   # Min over landmarks
filtration_value = max(distances, dim=1)                 # Max over points on simplex
```

## Backward Compatibility

- Existing code using `flood_complex` without `landmark_weights` works unchanged
- The weighted flooding path is only activated when `landmark_weights` is provided
- All existing tests and examples continue to work

## Usage

```python
# Standard flood complex (unchanged)
stree = flood_complex(points, landmarks)

# Weighted flood complex (new)
stree = flood_complex(points, landmarks, landmark_weights=weights)

# Protein Flood Complex (now uses extended flood_complex)
pfc_stree = protein_flood_complex(protein, target_landmarks=500)
```

