# Triton Optimization for Protein Flood Complex

## Summary

The PFC implementation now uses **Triton-optimized kernels** for weighted flooding with circumball coverage, fully leveraging GPU acceleration.

## Implementation Details

### 1. Triton Kernel: `compute_circumballs_kernel`

**Location**: `flooder/flooder/triton_kernels.py`

**Purpose**: Computes circumcenters and circumradii for multiple simplices in parallel.

**Algorithm**:
- **Edges**: Midpoint and half-length (fully vectorized)
- **Triangles**: Exact circumcenter formula using cross products
- **Tetrahedra**: Centroid approximation with max distance radius
- Handles degenerate cases gracefully

**Key Features**:
- Parallel processing of simplices (BLOCK_S = 64)
- Supports edges, triangles, and tetrahedra
- Handles degenerate simplices
- GPU-optimized memory access

### 2. Triton Kernel: `compute_weighted_distances_kernel`

**Location**: `flooder/flooder/triton_kernels.py`

**Purpose**: Computes weighted filtration values using circumball coverage test.

**Algorithm**:
- For each simplex with circumcenter `c` and circumradius `ρ`
- Computes: `max_ℓ((||c - ℓ|| + ρ) / w(ℓ))` over all landmarks `ℓ`
- This gives the minimum epsilon where the circumball is covered by weighted balls

**Key Features**:
- Parallel processing of simplices (BLOCK_S = 64)
- Tiled processing of landmarks (BLOCK_L = 256)
- Atomic max operations to combine results across landmark tiles
- GPU-optimized memory access patterns

### 3. Integration with `flood_complex`

**Location**: `flooder/flooder/core.py`

**Changes**:
- When `landmark_weights` is provided, uses circumball coverage instead of witness point sampling
- Calls `compute_weighted_filtration_triton()` for GPU-accelerated computation
- Falls back to CPU/PyTorch implementation if Triton fails or on CPU

### 4. Functions: `compute_circumballs_triton` and `compute_weighted_filtration_triton`

**Location**: `flooder/flooder/triton_kernels.py`

**Workflow**:
1. **Circumball Computation**: Uses `compute_circumballs_triton()` with Triton kernel
   - Fully parallel computation on GPU
   - Handles edges, triangles, and tetrahedra
   - Processes all simplices simultaneously

2. **Triton Kernel Launch**: 
   - Grid: `(triton.cdiv(S, BLOCK_S), triton.cdiv(L, BLOCK_L))`
   - Each thread block processes a tile of simplices and landmarks
   - Atomic operations combine results across landmark tiles

3. **Fallback**: If Triton kernel fails or on CPU, uses PyTorch implementation

## Performance Benefits

### Before Optimization:
- Python loops for circumball computation
- Standard PyTorch operations for distance computation
- Sequential processing of simplices

### After Optimization:
- **Triton kernel** for circumball computation (`compute_circumballs_kernel`)
- **Triton kernel** for weighted distance computation (`compute_weighted_distances_kernel`)
- **Fully parallel** processing of multiple simplices and landmarks
- **Optimized memory access** patterns
- **End-to-end GPU acceleration** with Triton

## Usage

The optimization is **automatic** when using `protein_flood_complex`:

```python
from flooder.pfc import protein_flood_complex

# Automatically uses Triton kernels on GPU
pfc_stree = protein_flood_complex(
    protein,
    target_landmarks=500,
    device="cuda",  # GPU required for Triton
    use_triton=True,  # Default
)
```

## Technical Details

### Kernel Parameters

- **BLOCK_S = 64**: Number of simplices processed per thread block
- **BLOCK_L = 256**: Number of landmarks processed per tile
- **Grid dimensions**: `(num_simplex_tiles, num_landmark_tiles)`

### Memory Layout

- **Circumcenters**: `(S, d)` contiguous tensor
- **Circumradii**: `(S,)` contiguous tensor  
- **Landmarks**: `(L, d)` contiguous tensor
- **Weights**: `(L,)` contiguous tensor
- **Output**: `(S,)` tensor with filtration values

### Atomic Operations

The kernel uses `tl.atomic_max` to combine results from different landmark tiles:
- First tile (pid_l == 0): Initializes output
- Subsequent tiles: Atomically updates maximum

## Future Optimizations

1. **Batched Circumball Computation**: Use `torch.vmap` or custom Triton kernel for parallel circumball computation
2. **Fused Kernel**: Combine circumball computation and weighted distance test in a single kernel
3. **Adaptive Block Sizes**: Dynamically adjust BLOCK_S and BLOCK_L based on problem size

## Testing

The implementation includes automatic fallback to CPU/PyTorch if:
- GPU is not available
- Triton kernel fails (memory/compilation errors)
- `use_triton=False` is specified

This ensures correctness and robustness across different hardware configurations.

