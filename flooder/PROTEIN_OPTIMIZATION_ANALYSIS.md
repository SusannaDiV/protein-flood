# Protein Code Optimization Analysis

## Current Status

### ✅ Fully Optimized (Triton/GPU)
1. **`flood_complex`** (core.py)
   - Uses Triton kernels: `compute_mask_kernel`, `compute_filtration_kernel`
   - GPU-accelerated witness point sampling

2. **Weighted Flooding Path** (triton_kernels.py)
   - `compute_circumballs_kernel`: Triton kernel for circumball computation
   - `compute_weighted_distances_kernel`: Triton kernel for weighted distance computation
   - Fully GPU-accelerated

3. **`compute_landmark_weights`** (pfc.py)
   - Uses PyTorch tensor operations (GPU-friendly)
   - Already optimized

### ⚠️ Partially Optimized (PyTorch but not Triton)
1. **`weighted_farthest_point_sampling`** (protein_landmark_sampling.py)
   - Uses `torch.cdist` (GPU-accelerated)
   - BUT: Iterative Python loop (not parallelizable)
   - Could benefit from batched distance updates

### ❌ Not Optimized (CPU-bound, NumPy/SciPy)
1. **`compute_curvature_score`** (protein_landmark_sampling.py)
   - **CRITICAL BOTTLENECK**: Sequential Python loop
   - Uses `KDTree` (CPU-only, scipy)
   - Per-point PCA computation (numpy)
   - **Impact**: O(K * k) where K = candidates, k = neighbors

2. **`compute_pocket_score`** (protein_landmark_sampling.py)
   - Uses `KDTree` (CPU-only, scipy)
   - NumPy operations
   - **Impact**: O(K * log K) for KDTree construction + queries

3. **`voxel_grid_downsample`** (protein_landmark_sampling.py)
   - Python loops for voxel grouping
   - NumPy operations
   - **Impact**: O(N) but could be faster on GPU

4. **`_add_extra_landmarks`** (protein_landmarks.py)
   - Uses `scipy.spatial.distance.cdist` (CPU-only)
   - Python loops
   - **Impact**: O(R * N) where R = residues, N = atoms

5. **`_match_landmarks`** (protein_landmarks.py)
   - Uses `scipy.spatial.distance.cdist` (CPU-only)
   - **Impact**: O(M * L) where M = selected, L = original

6. **`compute_pls_landmark_weights`** (protein_landmark_sampling.py)
   - Uses `KDTree` for density estimation (CPU-only)
   - NumPy operations
   - **Impact**: O(M * log M)

## Optimization Priority

### High Priority (Major Bottlenecks)
1. **`compute_curvature_score`**: Sequential loop with PCA - should be batched on GPU
2. **`compute_pocket_score`**: KDTree queries - should use `torch.cdist` on GPU
3. **`weighted_farthest_point_sampling`**: Iterative loop - could batch distance updates

### Medium Priority (Moderate Impact)
4. **`voxel_grid_downsample`**: Python loops - could use PyTorch operations
5. **`_add_extra_landmarks`**: scipy cdist - should use `torch.cdist`
6. **`_match_landmarks`**: scipy cdist - should use `torch.cdist`

### Low Priority (Minor Impact)
7. **`compute_pls_landmark_weights`**: KDTree for density - could use `torch.cdist`

## Recommended Optimizations

### 1. `compute_curvature_score` → GPU Batch PCA
- Replace KDTree with `torch.cdist` for k-nearest neighbors
- Batch PCA computation using PyTorch SVD
- Process all points in parallel

### 2. `compute_pocket_score` → GPU Distance Computation
- Replace KDTree with `torch.cdist` for neighbor queries
- Use PyTorch operations throughout
- Support GPU tensors

### 3. `weighted_farthest_point_sampling` → Batched Updates
- Batch distance updates instead of one-by-one
- Use `torch.cdist` efficiently
- Consider vectorized selection

### 4. Replace all `scipy.spatial.distance.cdist` → `torch.cdist`
- `_add_extra_landmarks`
- `_match_landmarks`
- Ensure GPU support

### 5. `voxel_grid_downsample` → PyTorch Implementation
- Use `torch.bucketize` or similar for voxel assignment
- Vectorized operations for voxel processing

## Implementation Strategy

1. **Phase 1**: Replace CPU-bound distance computations
   - Replace all `cdist` (scipy) → `torch.cdist`
   - Replace all `KDTree` → `torch.cdist` or `torch.topk`

2. **Phase 2**: Batch operations
   - Batch PCA in `compute_curvature_score`
   - Batch distance updates in `weighted_farthest_point_sampling`

3. **Phase 3**: GPU-native implementations
   - Convert voxel downsampling to PyTorch
   - Ensure all operations support GPU tensors

## Expected Performance Gains

- **`compute_curvature_score`**: 10-50x speedup (GPU batching)
- **`compute_pocket_score`**: 5-10x speedup (GPU distance computation)
- **`weighted_farthest_point_sampling`**: 2-5x speedup (batched updates)
- **Overall PLS**: 5-20x speedup for large proteins

