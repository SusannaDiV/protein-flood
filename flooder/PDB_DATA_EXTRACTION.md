# PDB Data Extraction: What We Get vs What We Compute

## Summary

We extract **basic structural information** from PDB files, then use **lookup tables** for biochemical properties and **compute approximations** for SASA.

## ✅ Extracted from PDB Files (via BioPython)

### Direct PDB Data
1. **Atom Coordinates** (x, y, z in Angstroms)
   - Source: `ATOM` and `HETATM` records
   - Used directly for landmark positions

2. **Atom Types** (element names)
   - Source: Element column in PDB
   - Examples: 'C', 'N', 'O', 'S', 'CA' (calcium)
   - Used to look up van der Waals radii

3. **Residue Names** (3-letter codes)
   - Source: Residue name column
   - Examples: 'ALA', 'GLY', 'ASP', 'LYS', 'ILE'
   - Used to look up hydrophobicity and charge

4. **Residue IDs** (sequence numbers)
   - Source: Residue sequence number
   - Used for residue mapping

5. **Atom Names**
   - Source: Atom name column
   - Examples: 'CA', 'CB', 'N', 'O'
   - Used to identify backbone vs sidechain atoms

6. **Chain IDs**
   - Source: Chain identifier
   - Used for multi-chain proteins

**Code Location**: `flooder/flooder/protein_io.py` → `load_pdb_file()`

## 📚 From Lookup Tables (NOT in PDB)

### 1. Van der Waals Radii
- **Source**: Hardcoded dictionary in `protein_io.py`
- **Values**: Standard vdW radii (e.g., C=1.70Å, N=1.55Å, O=1.52Å)
- **Usage**: Looked up by atom type from PDB
- **Code**: `get_atom_vdw_radius(atom_type)`

```python
VDW_RADII = {
    "H": 1.20, "C": 1.70, "N": 1.55, "O": 1.52,
    "S": 1.80, "P": 1.80, "FE": 2.00, "ZN": 1.39,
    ...
}
```

### 2. Hydrophobicity Scores
- **Source**: Kyte-Doolittle hydrophobicity scale (hardcoded)
- **Values**: Normalized to [0, 1]
- **Usage**: Looked up by residue name from PDB
- **Code**: `get_residue_hydrophobicity(residue_name)`

```python
HYDROPHOBICITY = {
    "ILE": 0.73, "VAL": 0.54, "LEU": 0.53,  # Hydrophobic
    "ASP": 0.11, "GLU": 0.11, "LYS": 0.00,  # Hydrophilic
    ...
}
```

### 3. Charge Information
- **Source**: Hardcoded list of charged residues
- **Values**: Binary (0 or 1)
- **Usage**: Checked by residue name from PDB
- **Code**: `is_charged_residue(residue_name)`

```python
CHARGED_RESIDUES = {"ASP", "GLU", "LYS", "ARG", "HIS"}
```

## 🧮 Computed/Approximated (NOT in PDB)

### Solvent-Accessible Surface Area (SASA)
- **Source**: **NOT in PDB files** - we compute an approximation
- **Method**: Simple nearest-neighbor distance proxy
- **Algorithm**:
  1. Compute pairwise distances between all atoms
  2. Use minimum distance to nearest neighbor as exposure proxy
  3. Normalize to [0, 1] using 95th percentile
- **Code**: `compute_simple_sasa(protein)`

**Note**: This is a **simplified approximation**. For accurate SASA, you would need:
- Full SASA calculation tools (FreeSASA, BioPython DSSP)
- Requires solvent probe radius (1.4Å for water)
- Much more computationally expensive

## Data Flow

```
PDB File
  ↓
BioPython Parser
  ↓
Extract: coordinates, atom_types, residue_names, residue_ids, atom_names, chain_ids
  ↓
ProteinStructure object
  ↓
get_landmark_attributes()
  ├─→ Lookup vdW radii by atom_type
  ├─→ Lookup hydrophobicity by residue_name
  ├─→ Check charge by residue_name
  └─→ Compute SASA approximation from coordinates
  ↓
compute_landmark_weights()
  ↓
Heterogeneous ball radii
```

## What PDB Files DON'T Contain

PDB files typically **do not contain**:
- ❌ Van der Waals radii (we use standard values)
- ❌ Hydrophobicity scores (we use Kyte-Doolittle scale)
- ❌ Charge states (we use residue type, not actual charge)
- ❌ SASA values (we compute approximation)
- ❌ Partial charges (we use binary charged/not-charged)

## Accuracy Considerations

### Current Approach
- ✅ **Fast**: No expensive SASA computation
- ✅ **Works well**: Good enough for PFC weighting
- ⚠️ **Approximate**: SASA is a simple proxy, not true SASA

### For Higher Accuracy
If you need more accurate SASA:
1. Use FreeSASA library: `pip install freesasa`
2. Use BioPython DSSP: Requires DSSP executable
3. Pre-compute SASA and pass as input

**Example** (if you have pre-computed SASA):
```python
# You could modify get_landmark_attributes() to accept
# pre-computed SASA values instead of computing approximation
```

## Summary

| Attribute | Source | Accuracy |
|-----------|--------|----------|
| **Coordinates** | PDB file | ✅ Exact |
| **Atom types** | PDB file | ✅ Exact |
| **Residue names** | PDB file | ✅ Exact |
| **vdW radii** | Lookup table | ✅ Standard values |
| **Hydrophobicity** | Lookup table | ✅ Kyte-Doolittle scale |
| **Charge** | Lookup table | ✅ Binary (residue type) |
| **SASA** | Computed approximation | ⚠️ Simple proxy |

**Bottom line**: We extract structural data from PDB, use biochemical lookup tables, and compute a simple SASA approximation. This is sufficient for PFC's weighted flooding, but SASA could be made more accurate if needed.

