# Example Script: `example_protein_pfc.py`

## Overview

This script demonstrates how to use the Protein Flood Complex (PFC) to analyze protein structures from PDB files. It processes proteins, constructs PFCs, and computes persistent homology to extract topological features.

## What It Does

### Main Functionality

1. **Loads protein structures** from PDB files
2. **Constructs Protein Flood Complexes** with protein-aware weighted flooding
3. **Computes persistent homology** (H₀, H₁, H₂) to extract topological features
4. **Saves results** (optional) for further analysis

### Key Features

- ✅ Processes single proteins or batch processes directories
- ✅ Supports both simplified and full PLS landmark selection
- ✅ GPU acceleration support (CUDA)
- ✅ Saves persistence diagrams and metadata
- ✅ Command-line interface for easy use

## Functions

### 1. `process_protein_pdb()`

Processes a **single protein** PDB file:

**Inputs**:
- `pdb_path`: Path to PDB file
- `target_landmarks`: Number of landmarks (default: 500)
- `max_dimension`: Maximum homology dimension (default: 2, for H₀, H₁, H₂)
- `device`: 'cpu' or 'cuda'
- `output_dir`: Optional directory to save results
- `use_pls`: Use full PLS algorithm (False = simplified, faster)

**What it does**:
1. Loads protein structure from PDB
2. Constructs Protein Flood Complex
3. Computes persistent homology
4. Extracts persistence diagrams for H₀, H₁, H₂
5. Returns/saves results dictionary

**Output**:
```python
{
    "pdb_path": str,
    "pdb_name": str,
    "num_atoms": int,
    "num_residues": int,
    "num_landmarks": int,
    "num_simplices": int,
    "persistence_diagrams": [H0_diag, H1_diag, H2_diag],
    "simplex_tree": gudhi.SimplexTree
}
```

### 2. `process_scpdb_directory()`

Processes **multiple proteins** from a directory (e.g., scPDB database):

**Inputs**:
- `scpdb_dir`: Path to directory containing PDB files
- `max_proteins`: Maximum number to process (default: 1, None = all)
- `target_landmarks`: Number of landmarks per protein
- `device`: 'cpu' or 'cuda'
- `output_dir`: Directory to save results
- `use_pls`: Use full PLS algorithm

**What it does**:
1. Finds all `.pdb` and `.ent` files in directory
2. Processes each protein sequentially
3. Collects results and prints summary
4. Returns list of results

### 3. `main()`

Command-line interface with argparse:

**Command-line arguments**:
- `--scpdb_dir`: Path to scPDB directory (default: Windows path)
- `--max_proteins`: Max proteins to process (default: 1)
- `--target_landmarks`: Target landmarks (default: 500)
- `--device`: 'cpu' or 'cuda' (default: 'cpu')
- `--output_dir`: Directory to save results (optional)
- `--use_pls`: Use full PLS algorithm (flag)

## Usage Examples

### Example 1: Process Single Protein (Simplified)

```bash
python example_protein_pfc.py \
    --scpdb_dir "path/to/pdbs" \
    --max_proteins 1 \
    --target_landmarks 500 \
    --device cpu
```

**Output**:
```
============================================================
Processing: protein.pdb
============================================================
Loading protein structure...
  Loaded 1234 atoms
  156 residues
Constructing Protein Flood Complex (Simplified (residue-based))...
  Note: Simplified PFC is ~3-6x faster than full PLS
        (Simplified: ~1.7-4.5s, Full PLS: ~6.5-19s for typical proteins)
        Simplified PFC is often 1.1-1.3x FASTER than standard flooder
        due to more efficient circumball coverage vs witness points
  Complex has 15234 simplices
Computing persistent homology...
  H0: 45 features
  H1: 12 features
  H2: 3 features
  Results saved to: ./results/protein_pfc_results.pt
```

### Example 2: Process with Full PLS (Better for Binding Sites)

```bash
python example_protein_pfc.py \
    --scpdb_dir "path/to/pdbs" \
    --max_proteins 1 \
    --target_landmarks 500 \
    --device cuda \
    --use_pls
```

### Example 3: Batch Process Multiple Proteins

```bash
python example_protein_pfc.py \
    --scpdb_dir "path/to/scPDB" \
    --max_proteins 10 \
    --target_landmarks 500 \
    --device cuda \
    --output_dir "./results"
```

## What the Results Contain

### Persistence Diagrams

Each diagram is a list of `(birth, death)` pairs:
- **H₀**: Connected components (birth = when component appears, death = when merged)
- **H₁**: Loops/holes (birth = when loop forms, death = when filled)
- **H₂**: Voids/cavities (birth = when void forms, death = when filled)

**Example**:
```python
H0_diagram = [(0.0, 1.2), (0.0, 2.5), (0.0, 3.1), ...]  # Components
H1_diagram = [(1.5, 4.2), (2.1, 5.8), ...]              # Loops
H2_diagram = [(3.2, 6.5), ...]                          # Cavities
```

### Metadata

- Protein name, number of atoms/residues
- Number of landmarks and simplices
- Full simplex tree for further analysis

## Use Cases

### 1. Binding Site Detection
- Look for H₂ features (cavities) that persist across filtration
- These often correspond to binding pockets

### 2. Tunnel/Channel Analysis
- H₁ features (loops) can indicate tunnels through proteins
- Long-persisting H₁ features are likely functional channels

### 3. Protein Comparison
- Compare persistence diagrams across proteins
- Use for classification or similarity analysis

### 4. Structure-Function Relationships
- Correlate topological features with known functions
- Identify novel functional sites

## Performance Notes

- **Simplified PFC**: ~1.7-4.5s per protein (typical)
- **Full PLS**: ~6.5-19s per protein (more accurate)
- **GPU acceleration**: 2-4x speedup with CUDA
- **Memory**: ~100-500 MB per protein

## Output Files

If `--output_dir` is specified, results are saved as:
```
{output_dir}/{pdb_name}_pfc_results.pt
```

These are PyTorch `.pt` files containing:
- Persistence diagrams
- Simplex tree
- Metadata

Can be loaded later:
```python
results = torch.load("protein_pfc_results.pt")
diagrams = results["persistence_diagrams"]
```

## Summary

This script is a **complete pipeline** for:
1. Loading protein structures
2. Computing protein-aware topological features
3. Extracting persistence diagrams
4. Saving results for analysis

It's designed to be **easy to use** for both single proteins and batch processing, with options for speed (simplified) or accuracy (full PLS).

