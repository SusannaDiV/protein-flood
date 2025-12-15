# Protein Flood Complex (PFC)

This module implements the **Protein-Aware Flood Complex (PFC)**, an extension of the Flood complex method specifically designed for protein structure analysis. PFC incorporates protein physics and chemistry into the topological analysis, enabling more accurate detection of functional sites like binding pockets, tunnels, and cavities.

## Key Features

### 1. Residue-Level Landmark Selection
- **Backbone representatives**: Centroid of backbone atoms (N, CA, C, O) per residue
- **Sidechain representatives**: Centroid of sidechain atoms per residue
- **Biochemical coverage**: Ensures uniform sampling across all residues
- **Optional additional landmarks**: Curvature/pocket-biased sampling for higher resolution

### 2. Protein-Aware Weighted Flooding Radii
The flooding radius for each landmark is computed as:

```
R(ℓ, ε) = ε · w(ℓ)

where w(ℓ) = clip((r₀ + α·r_atom + β·SASA) · (1 + γ_h·H - γ_q·Q), R_min, R_max)
```

- **r₀**: Global offset (default: 1.0 Å)
- **r_atom**: Van der Waals radius proxy
- **SASA**: Normalized Solvent-Accessible Surface Area [0, 1]
- **H**: Hydrophobicity score (Kyte-Doolittle, normalized)
- **Q**: Charge indicator (1 for charged residues, 0 otherwise)
- **α, β, γ_h, γ_q**: Hyperparameters controlling term weights

### 3. Heterogeneous Balls
- **Larger balls** for hydrophobic, buried regions (common in binding pockets)
- **Smaller balls** for charged, solvent-exposed regions
- Better captures protein geometry and chemistry

## Installation

The PFC module requires BioPython for PDB file parsing:

```bash
pip install biopython
```

## Usage

### Basic Example

```python
from flooder.protein_io import load_pdb_file
from flooder.pfc import protein_flood_complex
import torch

# Load protein structure
protein = load_pdb_file("protein.pdb")

# Construct Protein Flood Complex
pfc_stree = protein_flood_complex(
    protein,
    target_landmarks=500,
    max_dimension=2,
    device="cpu",
    r0=1.0,
    alpha=1.0,
    beta=0.5,
    gamma_h=0.2,
    gamma_q=0.2,
)

# Compute persistent homology
pfc_stree.compute_persistence()

# Extract persistence diagrams
diagrams = [
    pfc_stree.persistence_intervals_in_dimension(dim)
    for dim in range(3)  # H0, H1, H2
]
```

### Batch Processing scPDB

Process all proteins from the scPDB directory:

```bash
python flooder/examples/batch_process_scpdb.py \
    --scpdb_dir "C:\Users\s.divita\Downloads\scPDB\scPDB" \
    --output_dir "./results" \
    --target_landmarks 500 \
    --device cuda \
    --max_proteins 100
```

### Custom Parameters

Adjust PFC hyperparameters for your specific use case:

```python
pfc_stree = protein_flood_complex(
    protein,
    target_landmarks=1000,  # More landmarks for higher resolution
    r0=1.0,                 # Base radius offset
    alpha=1.0,              # Atom radius weight
    beta=0.5,               # SASA weight
    gamma_h=0.3,            # Hydrophobicity modulation (higher = larger balls for hydrophobic)
    gamma_q=0.2,            # Charge modulation (higher = smaller balls for charged)
    r_min=0.5,             # Minimum weight
    r_max=2.0,             # Maximum weight
)
```

## Algorithm Details

### Step 1: Residue-Level Landmark Selection
1. For each residue, compute backbone centroid
2. For residues with sidechains, compute sidechain centroid
3. Optionally add extra landmarks using curvature/pocket-biased sampling

### Step 2: Compute Protein Attributes
For each landmark:
- **Atom radius**: Average vdW radius of atoms in the residue
- **SASA**: Normalized solvent exposure (simplified approximation)
- **Hydrophobicity**: Kyte-Doolittle score normalized to [0, 1]
- **Charge**: Binary indicator for charged residues (ASP, GLU, LYS, ARG, HIS)

### Step 3: Compute Flooding Weights
Apply the weighted formula to compute per-landmark flooding weights.

### Step 4: Delaunay Substrate
Construct 3D Delaunay triangulation on landmarks.

### Step 5: Weighted Flooding Inclusion Test
For each simplex in the Delaunay complex:
- Compute circumcenter and circumradius
- Test if circumball is contained in weighted flooded union
- Include simplex at minimum epsilon where test passes

### Step 6: Persistent Homology
Compute persistence diagrams up to dimension 2 (H₀, H₁, H₂).

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `target_landmarks` | 500 | Target number of landmarks |
| `r0` | 1.0 | Global radius offset (Å) |
| `alpha` | 1.0 | Weight for atom radius term |
| `beta` | 0.5 | Weight for SASA term |
| `gamma_h` | 0.2 | Hydrophobicity modulation weight |
| `gamma_q` | 0.2 | Charge modulation weight |
| `r_min` | 0.5·r₀ | Minimum flooding weight |
| `r_max` | 2.0·r₀ | Maximum flooding weight |

## Output

The PFC returns a `gudhi.SimplexTree` object that can be used for:
- Computing persistent homology
- Extracting persistence diagrams
- Downstream machine learning (persistence images, landscapes, etc.)

## Novelty vs. Standard Flood Complex

PFC differs from the standard Flood complex by:

1. **Residue-aware landmarks**: Biochemical coverage instead of geometric sampling
2. **Heterogeneous balls**: Physics/chemistry-aware instead of uniform radii
3. **Weighted flooding**: Filtration reflects exposure and chemistry, not just Euclidean scale

This creates a filtration that is still sparse (Delaunay on landmarks), scalable, and GPU-friendly, but better aligned with protein structure and function.

## References

- Original Flood Complex: [NeurIPS 2025 paper](https://arxiv.org/abs/2509.22432)
- Protein topology: Persistent homology captures binding sites, tunnels, and cavities
- Kyte-Doolittle hydrophobicity scale: Standard biochemical measure

## Troubleshooting

### BioPython Import Error
If you see `ImportError: BioPython is required`, install it:
```bash
pip install biopython
```

### Memory Issues
For large proteins, reduce `target_landmarks` or use CPU instead of CUDA:
```python
pfc_stree = protein_flood_complex(protein, target_landmarks=300, device="cpu")
```

### SASA Computation
The current implementation uses a simplified SASA approximation. For more accurate results, consider integrating with specialized tools like FreeSASA or BioPython's DSSP.

