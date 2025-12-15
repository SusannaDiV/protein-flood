# Flow: PFC → Persistence Images

## Complete Pipeline

```
PDB File
  ↓
Protein Structure (atoms, residues)
  ↓
Protein Flood Complex (PFC)
  ├─→ Residue-based landmarks
  ├─→ Protein-aware weights (vdW, SASA, hydrophobicity, charge)
  ├─→ Heterogeneous balls: B(ℓ, w(ℓ)·ε)
  └─→ Weighted flooding with circumball coverage
  ↓
SimplexTree (PFC output)
  ↓
Persistent Homology Computation
  ├─→ H₀ diagram: Connected components
  ├─→ H₁ diagram: Tunnels/loops
  └─→ H₂ diagram: Cavities/voids
  ↓
Persistence Images (vectorization)
  ├─→ H₀ image: (20, 20) grid
  ├─→ H₁ image: (20, 20) grid
  └─→ H₂ image: (20, 20) grid
  ↓
Concatenated Feature Vector (1200,)
  ↓
ML Model (MLP, CNN, etc.)
```

## Key Point

**The persistence images are specifically from PFC output**, which means:

✅ **Protein-aware topology**: Features reflect protein biochemistry
✅ **Heterogeneous balls**: Different radii based on atom size, SASA, chemistry
✅ **Residue-level landmarks**: Biochemical coverage, not just geometric
✅ **Weighted flooding**: Filtration values incorporate protein properties

## What Makes These Images Special

Unlike standard persistence images from generic point clouds, these images capture:

1. **Protein-specific features**:
   - Binding pockets (from weighted flooding)
   - Tunnels/channels (from curvature-aware landmarks)
   - Cavities (from heterogeneous balls)

2. **Biochemical information**:
   - Hydrophobic regions expand faster
   - Charged regions expand slower
   - Buried vs exposed differences

3. **Functionally relevant topology**:
   - Features aligned with protein structure
   - Better for binding site detection
   - More relevant for structure-function relationships

## Code Flow

```python
# Step 1: Create PFC (protein-aware)
pfc_stree = protein_flood_complex(
    protein,
    target_landmarks=500,
    # Uses protein attributes: vdW, SASA, hydrophobicity, charge
    # Creates heterogeneous balls: B(ℓ, w(ℓ)·ε)
)

# Step 2: Compute persistence on PFC
pfc_stree.compute_persistence()
# This gives diagrams from the PFC's weighted filtration

# Step 3: Convert PFC diagrams to images
persistence_images = compute_persistence_images_from_simplex_tree(
    pfc_stree,  # ← PFC output!
    max_dimension=2,
)
# These images encode the PFC's protein-aware topology
```

## Comparison

### Standard Persistence Images
- From: Generic point cloud → Standard complex (e.g., Vietoris-Rips, Alpha)
- Features: Pure geometric topology
- No protein awareness

### Our Persistence Images
- From: **Protein → PFC → Diagrams → Images**
- Features: **Protein-aware topology** (biochemistry + geometry)
- Incorporates: Atom size, SASA, hydrophobicity, charge

## Summary

**Yes, persistence images are created from PFC output!**

The images encode:
- ✅ Topological features from the PFC
- ✅ Protein-aware weighted flooding
- ✅ Heterogeneous ball radii
- ✅ Residue-level landmark selection

This makes them **protein-specific topological features** suitable for ML tasks like:
- Binding site prediction
- Function classification
- Structure-function relationships
- Protein comparison

