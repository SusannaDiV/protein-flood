"""Implementation of the Triton kernels.

Copyright (c) 2025 Paolo Pellizzoni, Florian Graf, Martin Uray, Stefan Huber and Roland Kwitt
SPDX-License-Identifier: MIT
"""

from typing import Tuple
import torch
import triton
import triton.language as tl


@triton.jit
def compute_filtration_kernel(
    x_ptr,  # pointer to x, shape (S, R, d)
    y_ptr,  # pointer to y, shape (W, d)
    s_idx_ptr,
    w_idx_ptr,
    inter_ptr,  # pointer to intermediate output
    R,  # total number of rows per sample in x
    W,  # number of y vectors
    d: tl.constexpr,  # feature dimension
    BLOCK_R: tl.constexpr,  # block size (tile size) for R dimension (must divide R)
    BLOCK_W: tl.constexpr,  # block size for the W dimension per tile
):
    pid_w = tl.program_id(0)  # tile index for W dimension
    pid_r = tl.program_id(1)  # tile index for R dimension
    id_s = tl.load(s_idx_ptr + pid_w)

    r_offset = pid_r * BLOCK_R + tl.arange(0, BLOCK_R)
    r_mask = r_offset < R
    x_idx = id_s * R * d + r_offset * d  # offset into x_ptr for each row
    w_idx = tl.load(w_idx_ptr + pid_w * BLOCK_W + tl.arange(0, BLOCK_W))

    # Initialize the squared-distance accumulator for this (BLOCK_R x BLOCK_W) tile.
    dist2 = tl.zeros((BLOCK_R, BLOCK_W), dtype=tl.float32)
    for i in range(d):
        x_vals = tl.load(x_ptr + x_idx + i, mask=r_mask, other=0.0)
        y_vals = tl.load(y_ptr + w_idx * d + i, mask=(w_idx < W), other=float("inf"))
        diff = x_vals[:, None] - y_vals[None, :]
        dist2 += diff * diff

    # Use tl.min with axis=1 to compute the minimum along the BLOCK_W (tile) dimension.
    tile_min = tl.sqrt(tl.min(dist2, axis=1))
    tl.atomic_min(inter_ptr + id_s * R + r_offset, tile_min, mask=r_mask)


def compute_filtration(
    x: torch.Tensor,
    y: torch.Tensor,
    row_idx: torch.Tensor,
    col_idx: torch.Tensor,
    BLOCK_W,
    BLOCK_R,
) -> torch.Tensor:

    S, R, d = x.shape
    W, d_y = y.shape
    num_valid = col_idx.shape[0]
    assert d == d_y, "Feature dimensions of x and y must match."

    T = num_valid // BLOCK_W  # Number of tiles along the W dimension.
    R_tiles = triton.cdiv(R, BLOCK_R)
    # Number of tiles in the R dimension.

    # Allocate an intermediate tensor of shape (S, R) on the GPU.
    inter = torch.full((S, R), torch.inf, device=x.device, dtype=torch.float32)

    # Bounds check
    assert (
        row_idx.shape == col_idx.shape
    ), f"row_idx.shape ({row_idx.shape}) does not match col_idx.shape ({col_idx.shape}"
    assert (
        col_idx.shape[0] == T * BLOCK_W
    ), f"col_idx.shape[0] {col_idx.shape[0]} does not match T * BLOCK_W ({T} * {BLOCK_W} = {T * BLOCK_W})"

    row_idx = row_idx[
        ::BLOCK_W
    ]  # consecutive row_indices need to be constant in blocks of length BLOCK_W
    # make sure indexing is contiguous and of type int32 for triton
    row_idx = row_idx.to(torch.int32).contiguous()
    col_idx = col_idx.to(torch.int32).contiguous()

    try:
        x = x.contiguous().view(-1)  # Make sure indexing math (later) matches layout
        compute_filtration_kernel[(T, R_tiles)](
            x, y, row_idx, col_idx, inter, R, W, d, BLOCK_R=BLOCK_R, BLOCK_W=BLOCK_W
        )
    except RuntimeError:
        raise RuntimeError(
            "Memory/Grid size error in CUDA, try lowering the batch size or setting disable_kernel=True"
        )
    return inter


@triton.jit
def compute_mask_kernel(
    points_ptr,  # (m, d), row-major
    mask_ptr,  # (n, m), flat index
    counts_ptr,  # (n), Trues per row
    cent_ptr,  # (n, d), center positions
    radi_ptr,  # (n, 1) or (n,), radius
    n_32b,
    m_32b,
    d_32b,
    BLOCK_N: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_W: tl.constexpr,
):

    # According to https://triton-lang.org/main/python-api/triton-semantics.html, any
    # operation with where one tensor is of a dtype of a higher kind, the other tensor
    # is promoted to this dtype
    n = tl.full((), n_32b, tl.int64)
    m = tl.full((), m_32b, tl.int64)
    d = tl.full((), d_32b, tl.int64)

    pid_m = tl.program_id(0)  # points
    pid_n = tl.program_id(1)  # centers

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    mask_n = offs_n < n
    mask_m = offs_m < m

    pt_stride = d
    cent_stride = d

    radi = tl.load(radi_ptr + offs_n, mask=mask_n, other=0.0)
    sq_radi = radi * radi  # [BLOCK_N]

    sq_dist = tl.zeros((BLOCK_N, BLOCK_M), dtype=tl.float32)
    for i in range(d):
        pt_i = tl.load(
            points_ptr + offs_m * pt_stride + i, mask=mask_m, other=0.0
        )  # [BLOCK_M]
        cent_i = tl.load(
            cent_ptr + offs_n * cent_stride + i, mask=mask_n, other=0.0
        )  # [BLOCK_N]
        diff_i = pt_i[None, :] - cent_i[:, None]  # [BLOCK_N, BLOCK_M]
        sq_dist += diff_i * diff_i  # [BLOCK_N, BLOCK_M]

    inside = sq_dist <= sq_radi[:, None]  # [BLOCK_N, BLOCK_M]

    stride_mask = m + BLOCK_W
    out_idx = offs_n[:, None] * stride_mask + offs_m[None, :]
    write_mask = (offs_n[:, None] < n) & (offs_m[None, :] < m)  # [BLOCK_N, BLOCK_M]

    tl.store(mask_ptr + out_idx, inside, mask=write_mask)
    counts_tile = tl.sum((inside * write_mask).to(tl.int32), axis=1)  # [BLOCK_N]

    # Atomically add counts_tile to global counts_ptr at offsets offs_n
    tl.atomic_add(counts_ptr + offs_n, counts_tile, mask=mask_n)


def compute_mask(
    points: torch.Tensor,
    centers: torch.Tensor,
    radii: torch.Tensor,
    BLOCK_N,
    BLOCK_M,
    BLOCK_W,
) -> torch.Tensor:
    """
    Check which points are inside Euclidean balls with given radii.

    Args:
        points (torch.Tensor):
            Tensor of shape (m, d), tensor with points to test.
        centers (torch.Tensor):
            Tensor of shape (n, d), tensor with centers of balls.
        radii (torch.Tensor):
            Tensor of shape (n, d), tensor with radii of balls.
        BLOCK_N (int):
            Block size along balls axis
        BLOCK_M (int):
            Block size along points axis
        BLOCk_W (int):
            Only used for padding

    Returns:
        mask (torch.Tensor):
            Boolean tensor of shape (n, m + BLOCK_W), mask[i,j] = True if points[j] inside simplices[i].
            Last (n, BLOCK_W) block is padded so that number of Trues per row is divisible by BLOCK_W
    """
    n, d = centers.shape
    m = points.shape[0]

    centers_flat = centers.view(n, -1).contiguous()
    radii_flat = radii.view(-1).contiguous()
    mask = torch.zeros((n, m + BLOCK_W), dtype=torch.bool, device=points.device)
    counts = torch.zeros(n, dtype=torch.int32, device=points.device)

    grid = (triton.cdiv(m, BLOCK_M), triton.cdiv(n, BLOCK_N))
    compute_mask_kernel[grid](
        points,
        mask,
        counts,
        centers_flat,
        radii_flat,
        n,
        m,
        d,
        BLOCK_N=BLOCK_N,
        BLOCK_M=BLOCK_M,
        BLOCK_W=BLOCK_W,
    )
    extra = ((-counts) % BLOCK_W).unsqueeze(1)  # [n, 1]
    extra_range = torch.arange(BLOCK_W, device=counts.device).unsqueeze(
        0
    )  # [1, BLOCK_W]
    mask[:, m : m + BLOCK_W] = extra_range < extra  # [n, BLOCK_W]
    return mask


@triton.jit
def compute_circumballs_kernel(
    vertices_ptr,  # (S, d+1, d), flattened - simplex vertices
    centers_ptr,  # (S, d), output circumcenters
    radii_ptr,  # (S,), output circumradii
    S,  # number of simplices
    simplex_dim: tl.constexpr,  # dimension of simplex (d+1)
    d: tl.constexpr,  # spatial dimension (typically 3)
    BLOCK_S: tl.constexpr,  # block size for simplices
):
    """
    Compute circumballs for multiple simplices in parallel.
    
    Supports:
    - Edges (simplex_dim=2): midpoint and half-length
    - Triangles (simplex_dim=3): circumcenter formula
    - Tetrahedra (simplex_dim=4): approximate method
    """
    pid = tl.program_id(0)  # simplex tile index
    
    # Compute global simplex indices for this block
    s_offset = pid * BLOCK_S + tl.arange(0, BLOCK_S)
    s_mask = s_offset < S
    
    # Process each simplex in the block
    for s_idx in range(BLOCK_S):
        s_global = s_offset[s_idx]
        
        # Skip if out of bounds
        if s_global >= S:
            continue
        
        if simplex_dim == 2:  # Edge
            # Load vertices
            v0 = tl.zeros([d], dtype=tl.float32)
            v1 = tl.zeros([d], dtype=tl.float32)
            for dim in range(d):
                v0[dim] = tl.load(vertices_ptr + s_global * simplex_dim * d + 0 * d + dim)
                v1[dim] = tl.load(vertices_ptr + s_global * simplex_dim * d + 1 * d + dim)
            
            # Circumcenter is midpoint
            center = (v0 + v1) / 2.0
            
            # Radius is half length
            sq_dist = 0.0
            for dim in range(d):
                diff = v1[dim] - v0[dim]
                sq_dist += diff * diff
            radius = tl.sqrt(sq_dist) / 2.0
            
            # Store
            for dim in range(d):
                tl.store(centers_ptr + s_global * d + dim, center[dim])
            tl.store(radii_ptr + s_global, radius)
            
        elif simplex_dim == 3:  # Triangle
            # Load vertices
            a = tl.zeros([d], dtype=tl.float32)
            b = tl.zeros([d], dtype=tl.float32)
            c = tl.zeros([d], dtype=tl.float32)
            for dim in range(d):
                a[dim] = tl.load(vertices_ptr + s_global * simplex_dim * d + 0 * d + dim)
                b[dim] = tl.load(vertices_ptr + s_global * simplex_dim * d + 1 * d + dim)
                c[dim] = tl.load(vertices_ptr + s_global * simplex_dim * d + 2 * d + dim)
            
            # Compute ab and ac
            ab = tl.zeros([d], dtype=tl.float32)
            ac = tl.zeros([d], dtype=tl.float32)
            for dim in range(d):
                ab[dim] = b[dim] - a[dim]
                ac[dim] = c[dim] - a[dim]
            
            # ab_len_sq and ac_len_sq
            ab_len_sq = 0.0
            ac_len_sq = 0.0
            for dim in range(d):
                ab_len_sq += ab[dim] * ab[dim]
                ac_len_sq += ac[dim] * ac[dim]
            
            # Cross product (for 3D)
            if d == 3:
                cross = tl.zeros([d], dtype=tl.float32)
                cross[0] = ab[1] * ac[2] - ab[2] * ac[1]
                cross[1] = ab[2] * ac[0] - ab[0] * ac[2]
                cross[2] = ab[0] * ac[1] - ab[1] * ac[0]
                
                area_sq = 0.0
                for dim in range(d):
                    area_sq += cross[dim] * cross[dim]
            else:
                # For 2D, use determinant
                area_sq = ab[0] * ac[1] - ab[1] * ac[0]
                area_sq = area_sq * area_sq
            
            if area_sq > 1e-10:
                # Circumcenter formula: u = (ab_len_sq * cross(cross, ac) - ac_len_sq * cross(cross, ab)) / (2 * area_sq)
                if d == 3:
                    # cross(cross, ac)
                    cross_ac = tl.zeros([d], dtype=tl.float32)
                    cross_ac[0] = cross[1] * ac[2] - cross[2] * ac[1]
                    cross_ac[1] = cross[2] * ac[0] - cross[0] * ac[2]
                    cross_ac[2] = cross[0] * ac[1] - cross[1] * ac[0]
                    
                    # cross(cross, ab)
                    cross_ab = tl.zeros([d], dtype=tl.float32)
                    cross_ab[0] = cross[1] * ab[2] - cross[2] * ab[1]
                    cross_ab[1] = cross[2] * ab[0] - cross[0] * ab[2]
                    cross_ab[2] = cross[0] * ab[1] - cross[1] * ab[0]
                    
                    u = tl.zeros([d], dtype=tl.float32)
                    for dim in range(d):
                        u[dim] = (ab_len_sq * cross_ac[dim] - ac_len_sq * cross_ab[dim]) / (2.0 * area_sq)
                else:
                    # 2D case (simplified)
                    u = tl.zeros([d], dtype=tl.float32)
                    u[0] = (ab_len_sq * ac[1] - ac_len_sq * ab[1]) / (2.0 * area_sq)
                    u[1] = (ac_len_sq * ab[0] - ab_len_sq * ac[0]) / (2.0 * area_sq)
                
                center = tl.zeros([d], dtype=tl.float32)
                for dim in range(d):
                    center[dim] = a[dim] + u[dim]
                
                # Radius
                sq_dist = 0.0
                for dim in range(d):
                    diff = center[dim] - a[dim]
                    sq_dist += diff * diff
                radius = tl.sqrt(sq_dist)
            else:
                # Degenerate triangle: use centroid
                center = tl.zeros([d], dtype=tl.float32)
                for dim in range(d):
                    center[dim] = (a[dim] + b[dim] + c[dim]) / 3.0
                
                # Radius is max distance to vertices
                dist_a = 0.0
                dist_b = 0.0
                dist_c = 0.0
                for dim in range(d):
                    diff_a = center[dim] - a[dim]
                    diff_b = center[dim] - b[dim]
                    diff_c = center[dim] - c[dim]
                    dist_a += diff_a * diff_a
                    dist_b += diff_b * diff_b
                    dist_c += diff_c * diff_c
                radius = tl.sqrt(tl.maximum(dist_a, tl.maximum(dist_b, dist_c)))
            
            # Store
            for dim in range(d):
                tl.store(centers_ptr + s_global * d + dim, center[dim])
            tl.store(radii_ptr + s_global, radius)
            
        elif simplex_dim == 4:  # Tetrahedron
            # Load vertices
            a = tl.zeros([d], dtype=tl.float32)
            b = tl.zeros([d], dtype=tl.float32)
            c_vert = tl.zeros([d], dtype=tl.float32)
            d_vert = tl.zeros([d], dtype=tl.float32)
            for dim in range(d):
                a[dim] = tl.load(vertices_ptr + s_global * simplex_dim * d + 0 * d + dim)
                b[dim] = tl.load(vertices_ptr + s_global * simplex_dim * d + 1 * d + dim)
                c_vert[dim] = tl.load(vertices_ptr + s_global * simplex_dim * d + 2 * d + dim)
                d_vert[dim] = tl.load(vertices_ptr + s_global * simplex_dim * d + 3 * d + dim)
            
            # Approximate: use centroid
            center = tl.zeros([d], dtype=tl.float32)
            for dim in range(d):
                center[dim] = (a[dim] + b[dim] + c_vert[dim] + d_vert[dim]) / 4.0
            
            # Radius is max distance to vertices
            max_sq_dist = 0.0
            for v_idx in range(4):
                sq_dist = 0.0
                for dim in range(d):
                    if v_idx == 0:
                        diff = center[dim] - a[dim]
                    elif v_idx == 1:
                        diff = center[dim] - b[dim]
                    elif v_idx == 2:
                        diff = center[dim] - c_vert[dim]
                    else:
                        diff = center[dim] - d_vert[dim]
                    sq_dist += diff * diff
                max_sq_dist = tl.maximum(max_sq_dist, sq_dist)
            radius = tl.sqrt(max_sq_dist)
            
            # Store
            for dim in range(d):
                tl.store(centers_ptr + s_global * d + dim, center[dim])
            tl.store(radii_ptr + s_global, radius)
        
        else:
            # Higher dimensions: use centroid approximation
            center = tl.zeros([d], dtype=tl.float32)
            for v_idx in range(simplex_dim):
                for dim in range(d):
                    v_val = tl.load(vertices_ptr + s_global * simplex_dim * d + v_idx * d + dim)
                    center[dim] += v_val
            for dim in range(d):
                center[dim] = center[dim] / float(simplex_dim)
            
            # Radius is max distance to vertices
            max_sq_dist = 0.0
            for v_idx in range(simplex_dim):
                sq_dist = 0.0
                for dim in range(d):
                    v_val = tl.load(vertices_ptr + s_global * simplex_dim * d + v_idx * d + dim)
                    diff = center[dim] - v_val
                    sq_dist += diff * diff
                max_sq_dist = tl.maximum(max_sq_dist, sq_dist)
            radius = tl.sqrt(max_sq_dist)
            
            # Store
            for dim in range(d):
                tl.store(centers_ptr + s_global * d + dim, center[dim])
            tl.store(radii_ptr + s_global, radius)


def compute_circumballs_triton(
    simplex_vertices: torch.Tensor,  # (S, d+1, d)
    use_triton: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute circumballs for multiple simplices using Triton kernel.
    
    GPU/Triton ONLY - no CPU fallback.
    
    Args:
        simplex_vertices: (S, d+1, d) tensor of simplex vertices (must be on GPU)
        use_triton: Whether to use Triton kernel (must be True)
        
    Returns:
        Tuple of (circumcenters, circumradii) where:
        - circumcenters: (S, d) tensor
        - circumradii: (S,) tensor
    """
    # Enforce GPU/Triton only
    if not simplex_vertices.is_cuda:
        raise RuntimeError("simplex_vertices must be on GPU (CUDA)")
    if not use_triton:
        raise RuntimeError("use_triton must be True (GPU/Triton only)")
    
    S, simplex_dim, d = simplex_vertices.shape
    
    # Triton kernel implementation
    BLOCK_S = 64
    
    # Flatten vertices: (S * (d+1) * d,)
    vertices_flat = simplex_vertices.contiguous().view(-1)
    
    # Output tensors
    circumcenters = torch.zeros(S, d, device=simplex_vertices.device, dtype=torch.float32)
    circumradii = torch.zeros(S, device=simplex_vertices.device, dtype=torch.float32)
    
    # Launch kernel (GPU/Triton only, no fallback)
    grid = (triton.cdiv(S, BLOCK_S),)
    
    compute_circumballs_kernel[grid](
        vertices_flat,
        circumcenters,
        circumradii,
        S,
        simplex_dim=simplex_dim,
        d=d,
        BLOCK_S=BLOCK_S,
    )
    
    # Validate output
    if torch.isnan(circumcenters).any() or torch.isnan(circumradii).any():
        nan_centers = torch.isnan(circumcenters).sum().item()
        nan_radii = torch.isnan(circumradii).sum().item()
        raise RuntimeError(
            f"NaN detected in circumball computation: {nan_centers} centers, {nan_radii} radii"
        )
    if torch.isinf(circumcenters).any() or torch.isinf(circumradii).any():
        inf_centers = torch.isinf(circumcenters).sum().item()
        inf_radii = torch.isinf(circumradii).sum().item()
        raise RuntimeError(
            f"Inf detected in circumball computation: {inf_centers} centers, {inf_radii} radii"
        )
    
    return circumcenters, circumradii


@triton.jit
def compute_weighted_distances_kernel(
    centers_ptr,  # (S, d), circumcenters of simplices
    radii_ptr,  # (S,), circumradii
    landmarks_ptr,  # (L, d), landmarks
    weights_ptr,  # (L,), landmark weights
    output_ptr,  # (S,), output max weighted distances
    S,  # number of simplices
    L,  # number of landmarks
    d: tl.constexpr,  # dimension (typically 3)
    BLOCK_S: tl.constexpr,  # block size for simplices
    BLOCK_L: tl.constexpr,  # block size for landmarks
):
    """
    Compute max weighted distances: max_ℓ((||c - ℓ|| + ρ) / w(ℓ))
    for each simplex with circumcenter c and circumradius ρ.
    """
    pid_s = tl.program_id(0)  # simplex tile
    pid_l = tl.program_id(1)  # landmark tile
    
    s_offset = pid_s * BLOCK_S + tl.arange(0, BLOCK_S)
    l_offset = pid_l * BLOCK_L + tl.arange(0, BLOCK_L)
    
    s_mask = s_offset < S
    l_mask = l_offset < L
    
    # Process each simplex in this block
    for s_idx in range(BLOCK_S):
        s_global = s_offset[s_idx]
        
        # Skip if out of bounds
        if s_global >= S:
            continue
        
        # Load circumcenter and radius
        center = tl.zeros([d], dtype=tl.float32)
        for dim in range(d):
            center[dim] = tl.load(centers_ptr + s_global * d + dim)
        radius = tl.load(radii_ptr + s_global)
        
        # Compute max weighted distance over landmarks in this tile
        max_weighted = float('-inf')  # Start with -inf so max works correctly
        for l_idx in range(BLOCK_L):
            l_global = l_offset[l_idx]
            
            # Skip if out of bounds
            if l_global >= L:
                continue
            
            # Distance from circumcenter to landmark
            sq_dist = 0.0
            for dim in range(d):
                l_val = tl.load(landmarks_ptr + l_global * d + dim)
                diff = l_val - center[dim]
                sq_dist += diff * diff
            dist = tl.sqrt(sq_dist)
            
            # Weighted distance: (dist + radius) / weight
            weight = tl.load(weights_ptr + l_global)
            # Avoid division by zero (shouldn't happen, but safety check)
            if weight > 0.0:
                weighted_dist = (dist + radius) / weight
                max_weighted = tl.maximum(max_weighted, weighted_dist)
        
        # Atomic max to combine results from different landmark tiles
        if pid_l == 0:
            # First tile: store initial value
            tl.store(output_ptr + s_global, max_weighted)
        else:
            # Subsequent tiles: atomic max
            tl.atomic_max(output_ptr + s_global, max_weighted)


def compute_weighted_filtration_triton(
    simplex_vertices: torch.Tensor,  # (S, d+1, d)
    landmarks: torch.Tensor,  # (L, d)
    landmark_weights: torch.Tensor,  # (L,)
    use_triton: bool = True,
) -> torch.Tensor:
    """
    Compute weighted filtration values using circumball coverage (Triton-optimized).
    
    GPU/Triton ONLY - no CPU fallback.
    
    Args:
        simplex_vertices: (S, d+1, d) tensor of simplex vertices (must be on GPU)
        landmarks: (L, d) tensor of landmark positions (must be on GPU)
        landmark_weights: (L,) tensor of landmark weights (must be on GPU, all > 0)
        use_triton: Whether to use Triton kernel (must be True)
        
    Returns:
        (S,) tensor of filtration values
    """
    # Enforce GPU/Triton only
    if not simplex_vertices.is_cuda:
        raise RuntimeError("simplex_vertices must be on GPU (CUDA)")
    if not landmarks.is_cuda:
        raise RuntimeError("landmarks must be on GPU (CUDA)")
    if not landmark_weights.is_cuda:
        raise RuntimeError("landmark_weights must be on GPU (CUDA)")
    if not use_triton:
        raise RuntimeError("use_triton must be True (GPU/Triton only)")
    
    # Validate weights
    if (landmark_weights <= 0).any():
        raise ValueError("All landmark_weights must be positive")
    
    S, simplex_dim, d = simplex_vertices.shape
    L = len(landmarks)
    
    # Compute circumballs using Triton kernel (GPU only)
    circumcenters, circumradii = compute_circumballs_triton(
        simplex_vertices,
        use_triton=True,  # Force Triton
    )
    
    # Check for NaN/Inf in circumballs
    if torch.isnan(circumcenters).any() or torch.isnan(circumradii).any():
        raise RuntimeError("NaN detected in circumball computation")
    if torch.isinf(circumcenters).any() or torch.isinf(circumradii).any():
        raise RuntimeError("Inf detected in circumball computation")
    
    # Use Triton kernel for weighted distance computation
    BLOCK_S = 64
    BLOCK_L = 256
    
    landmarks_flat = landmarks.contiguous()
    weights_flat = landmark_weights.contiguous()
    centers_flat = circumcenters.contiguous()
    radii_flat = circumradii.contiguous()
    
    # Initialize output with -inf (we'll take max, so start with -inf)
    output = torch.full((S,), float('-inf'), device=simplex_vertices.device, dtype=torch.float32)
    
    grid = (triton.cdiv(S, BLOCK_S), triton.cdiv(L, BLOCK_L))
    
    compute_weighted_distances_kernel[grid](
        centers_flat,
        radii_flat,
        landmarks_flat,
        weights_flat,
        output,
        S,
        L,
        d=d,
        BLOCK_S=BLOCK_S,
        BLOCK_L=BLOCK_L,
    )
    
    # Check for NaN/Inf in output
    if torch.isnan(output).any():
        nan_count = torch.isnan(output).sum().item()
        raise RuntimeError(f"NaN detected in weighted filtration computation: {nan_count}/{S} values")
    if torch.isinf(output).any():
        inf_count = torch.isinf(output).sum().item()
        # Check if it's -inf (initialization) or +inf (actual issue)
        if (output == float('inf')).any():
            inf_count_pos = (output == float('inf')).sum().item()
            raise RuntimeError(f"Inf detected in weighted filtration computation: {inf_count_pos}/{S} values")
    
    return output