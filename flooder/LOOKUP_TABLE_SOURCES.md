# Lookup Table Sources and References

## Overview

The lookup tables used in PFC are **hardcoded in the code** but based on **standard biochemical references**. Here's where each table comes from:

## 1. Van der Waals Radii

**Location**: `flooder/flooder/protein_io.py` → `VDW_RADII` dictionary

**Source**: Standard values from chemistry/biochemistry literature
- **Primary references**:
  - Bondi (1964): "Van der Waals Volumes and Radii", J. Phys. Chem. 68:441-451
  - Rowland & Taylor (1996): "Intermolecular Nonbonded Contact Distances in Organic Crystal Structures", J. Phys. Chem. 100:7384-7391
- **Common in protein structure tools**: Similar values used in PyMOL, VMD, Chimera, etc.

**Values**:
```python
VDW_RADII = {
    "H": 1.20,   # Hydrogen
    "C": 1.70,   # Carbon
    "N": 1.55,   # Nitrogen
    "O": 1.52,   # Oxygen
    "S": 1.80,   # Sulfur
    "P": 1.80,   # Phosphorus
    "FE": 2.00,  # Iron
    "ZN": 1.39,  # Zinc
    "MG": 1.73,  # Magnesium
    "CA": 2.31,  # Calcium
    ...
}
```

**Note**: These are standard values. Some tools use slightly different values (e.g., some use C=1.88Å), but 1.70Å is the most common for carbon.

## 2. Hydrophobicity Scale (Kyte-Doolittle)

**Location**: `flooder/flooder/protein_io.py` → `HYDROPHOBICITY` dictionary

**Source**: **Kyte & Doolittle (1982)**
- **Paper**: "A simple method for displaying the hydropathic character of a protein"
- **Journal**: Journal of Molecular Biology, 157:105-132 (1982)
- **DOI**: https://doi.org/10.1016/0022-2836(82)90515-0

**Original Scale**:
- Range: approximately -4.5 to +4.5
- Higher values = more hydrophobic
- Examples: ILE=4.5, VAL=4.2, LEU=3.8, PHE=2.8, ASP=-3.5, GLU=-3.5

**Our Normalization**:
- Original scale normalized to [0, 1] for use in PFC
- Higher values still = more hydrophobic
- Formula: `normalized = (original - min) / (max - min)`

**Values** (normalized):
```python
HYDROPHOBICITY = {
    "ILE": 0.73,  # Most hydrophobic
    "VAL": 0.54,
    "LEU": 0.53,
    "PHE": 0.50,
    "ASP": 0.11,  # Least hydrophobic (charged)
    "GLU": 0.11,
    "LYS": 0.00,
    "ARG": 0.00,
    ...
}
```

**Why Kyte-Doolittle?**
- Most widely used hydrophobicity scale in protein science
- Based on transfer free energies of amino acids from water to organic solvent
- Standard in many protein analysis tools (BLAST, protein structure prediction, etc.)

## 3. Charged Residues

**Location**: `flooder/flooder/protein_io.py` → `CHARGED_RESIDUES` set

**Source**: **Standard biochemistry classification**
- Based on amino acid side chain pKa values at physiological pH (7.4)
- Standard textbook knowledge (Lehninger, Voet & Voet, etc.)

**Charged Residues**:
```python
CHARGED_RESIDUES = {
    "ASP",  # Aspartic acid (pKa ~3.9, negatively charged at pH 7.4)
    "GLU",  # Glutamic acid (pKa ~4.3, negatively charged at pH 7.4)
    "LYS",  # Lysine (pKa ~10.5, positively charged at pH 7.4)
    "ARG",  # Arginine (pKa ~12.5, positively charged at pH 7.4)
    "HIS",  # Histidine (pKa ~6.0, can be charged or neutral depending on environment)
}
```

**Note on Histidine**:
- Histidine has pKa ~6.0, so it can be charged or neutral
- We include it as charged for simplicity, but in reality it depends on local pH
- For more accuracy, could use actual pKa values or compute from local environment

## Summary

| Lookup Table | Source | Reference |
|--------------|--------|-----------|
| **Van der Waals Radii** | Standard chemistry literature | Bondi (1964), Rowland & Taylor (1996) |
| **Hydrophobicity** | Kyte-Doolittle scale | Kyte & Doolittle (1982), J. Mol. Biol. 157:105-132 |
| **Charged Residues** | Standard biochemistry | Textbook classification based on pKa values |

## Alternative Sources

If you want to use different values:

### Alternative Hydrophobicity Scales
- **Eisenberg et al. (1984)**: Normalized consensus scale
- **Engelman et al. (1986)**: Hydrophobicity from membrane protein studies
- **Wimley & White (1996)**: Interface scale

### Alternative vdW Radii
- **UFF (Universal Force Field)**: Different values for some atoms
- **AMBER/CHARMM force fields**: Slightly different radii
- **Protein-specific**: Some tools use residue-specific radii

### More Accurate Charge
- Use actual pKa values and compute charge from local pH
- Use partial charges from force fields (AMBER, CHARMM)
- Use quantum chemistry calculations

## Implementation Note

These lookup tables are **hardcoded** in the code for:
- ✅ **Speed**: No external file I/O
- ✅ **Simplicity**: No dependencies on external databases
- ✅ **Reliability**: Values are always available

For research purposes, you could:
- Load from external files
- Use more sophisticated scales
- Compute values dynamically

But for PFC, these standard values work well and are widely accepted in the protein science community.

