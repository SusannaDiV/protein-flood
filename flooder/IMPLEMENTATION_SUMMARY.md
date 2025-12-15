# Protein Flood Complex (PFC) Implementation Summary

This document summarizes the implementation of the Protein-Aware Flood Complex (PFC) method for protein structure analysis.

## Files Created

### Core Modules

1. **`flooder/flooder/protein_io.py`**
   - `load_pdb_file()`: Loads PDB/mmCIF files using BioPython
   - `ProteinStructure`: Container class for protein data
   - Helper functions for atom radii, hydrophobicity, charge detection
   - Simple SASA approximation

2. **`flooder/flooder/protein_landmarks.py`**
   - `select_residue_landmarks()`: Residue-level landmark selection
   - `get_landmark_attributes()`: Compute protein attributes per landmark
   - Backbone and sidechain centroid computation
   - Optional curvature/pocket-biased additional landmarks

3. **`flooder/flooder/pfc.py`**
   - `protein_flood_complex()`: Main PFC construction function
   - `compute_landmark_weights()`: Weighted flooding radius computation
   - `weighted_flood_inclusion_test()`: Inclusion test for simplices
   - `compute_simplex_circumball()`: Circumball computation for simplices

### Example Scripts

4. **`flooder/examples/example_protein_pfc.py`**
   - Interactive example for processing single proteins
   - Command-line interface for scPDB processing
   - Detailed progress reporting

5. **`flooder/examples/batch_process_scpdb.py`**
   - Batch processing script for scPDB directory
   - Progress bar support (with tqdm fallback)
   - Saves individual results and summary

### Documentation

6. **`flooder/PFC_README.md`**
   - Comprehensive documentation
   - Usage examples
   - Parameter descriptions
   - Algorithm details

## Key Features Implemented

### 1. Residue-Level Landmark Selection ✓
- Backbone centroids (N, CA, C, O atoms)
- Sidechain centroids (all non-backbone atoms)
- Residue-to-landmark mapping preserved
- Optional additional landmarks with pocket/curvature bias

### 2. Protein-Aware Weighted Flooding Radii ✓
- Base radius: `r₀ + α·r_atom + β·SASA`
- Chemistry modulation: `1 + γ_h·H - γ_q·Q`
- Weighted flooding: `R(ℓ, ε) = ε·w(ℓ)`
- Bounded weights: `clip(w(ℓ), R_min, R_max)`

### 3. Weighted Flooding Inclusion Test ✓
- Circumball computation for simplices
- Weighted union coverage test
- Efficient binary search for filtration values
- Numerical tolerance handling

### 4. Integration with Existing Flood Complex ✓
- Uses existing Delaunay triangulation (gudhi)
- Compatible with gudhi.SimplexTree
- Persistent homology computation
- GPU support (via device parameter)

## Algorithm Implementation

The PFC algorithm follows the specification:

1. **Step A**: Residue-level landmark selection
   - Backbone + sidechain centroids
   - Optional PLS (Protein Landmark Sampling) for extra landmarks

2. **Step B**: Protein-aware weights
   - Compute atom radii, SASA, hydrophobicity, charge
   - Apply weighted formula
   - Clip to bounds

3. **Step C**: Delaunay substrate
   - 3D Delaunay triangulation on landmarks

4. **Step D**: Weighted flooding inclusion
   - For each simplex, find minimum epsilon where included
   - Uses binary search for efficiency

5. **Step E**: Persistent homology
   - Compute persistence diagrams (H₀, H₁, H₂)

## Usage

### Basic Usage
```python
from flooder.protein_io import load_pdb_file
from flooder.pfc import protein_flood_complex

protein = load_pdb_file("protein.pdb")
pfc_stree = protein_flood_complex(protein, target_landmarks=500)
pfc_stree.compute_persistence()
```

### Batch Processing
```bash
python flooder/examples/batch_process_scpdb.py \
    --scpdb_dir "C:\Users\s.divita\Downloads\scPDB\scPDB" \
    --output_dir "./results" \
    --target_landmarks 500
```

## Dependencies

- **BioPython**: Required for PDB file parsing
  ```bash
  pip install biopython
  ```

- **Existing flooder dependencies**: torch, gudhi, numpy, scipy, fpsample

## Default Parameters

- `r0 = 1.0` Å (global offset)
- `alpha = 1.0` (atom radius weight)
- `beta = 0.5` (SASA weight)
- `gamma_h = 0.2` (hydrophobicity modulation)
- `gamma_q = 0.2` (charge modulation)
- `R_min = 0.5·r₀`, `R_max = 2.0·r₀` (weight bounds)

## Testing

To test the implementation:

1. **Single protein test**:
   ```python
   python flooder/examples/example_protein_pfc.py --scpdb_dir <path> --max_proteins 1
   ```

2. **Batch test**:
   ```python
   python flooder/examples/batch_process_scpdb.py --scpdb_dir <path> --output_dir ./test_results --max_proteins 5
   ```

## Notes

- SASA computation uses a simplified approximation. For production use, consider integrating with FreeSASA or BioPython's DSSP.
- The current implementation processes simplices sequentially. For very large proteins, consider further optimization.
- GPU support is available but may require tuning batch sizes for memory constraints.

## Future Enhancements

Potential improvements:
1. More accurate SASA computation (FreeSASA integration)
2. Parallel processing of simplices
3. Caching of landmark attributes
4. Support for ligand-bound structures
5. Integration with machine learning pipelines (persistence images/landscapes)

