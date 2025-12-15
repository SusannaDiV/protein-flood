"""Protein structure I/O functionality for loading PDB files.

This module provides functions to load protein structures from PDB files
and extract atom coordinates, residue information, and structural attributes.
"""

from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
import torch

try:
    from Bio.PDB import PDBParser, MMCIFParser
    from Bio.PDB.PDBExceptions import PDBConstructionWarning
    import warnings
    warnings.filterwarnings("ignore", category=PDBConstructionWarning)
    BIOPYTHON_AVAILABLE = True
except ImportError:
    BIOPYTHON_AVAILABLE = False

# Standard atom types for backbone
BACKBONE_ATOMS = {"N", "CA", "C", "O"}

# Van der Waals radii (in Angstroms) for common atoms
# Source: Standard values from chemistry/biochemistry literature
# References: Bondi (1964), Rowland & Taylor (1996), and common protein structure tools
VDW_RADII = {
    "H": 1.20,
    "C": 1.70,
    "N": 1.55,
    "O": 1.52,
    "S": 1.80,
    "P": 1.80,
    "F": 1.47,
    "CL": 1.75,
    "BR": 1.85,
    "I": 1.98,
    "FE": 2.00,
    "ZN": 1.39,
    "MG": 1.73,
    "CA": 2.31,
    "DEFAULT": 1.70,  # Default for unknown atoms
}

# Kyte-Doolittle hydrophobicity scale (normalized to [0, 1])
# Source: Kyte & Doolittle (1982) "A simple method for displaying the hydropathic
#         character of a protein", J. Mol. Biol. 157:105-132
# Original scale: ILE=4.5, VAL=4.2, LEU=3.8, PHE=2.8, etc. (range ~-4.5 to +4.5)
# Normalized here to [0, 1] where higher = more hydrophobic
HYDROPHOBICITY = {
    "ILE": 0.73, "VAL": 0.54, "LEU": 0.53, "PHE": 0.50, "CYS": 0.66,
    "MET": 0.26, "ALA": 0.31, "GLY": 0.00, "THR": 0.45, "SER": 0.36,
    "TRP": 0.37, "TYR": 0.69, "PRO": 0.00, "HIS": 0.17, "GLN": 0.00,
    "ASN": 0.43, "ASP": 0.11, "GLU": 0.11, "LYS": 0.00, "ARG": 0.00,
}

# Charged residues (standard biochemistry)
# Source: Standard amino acid classification
# These residues are typically charged at physiological pH (7.4)
# Note: HIS can be charged or neutral depending on pH and local environment
CHARGED_RESIDUES = {"ASP", "GLU", "LYS", "ARG", "HIS"}


class ProteinStructure:
    """Container for protein structure data."""
    
    def __init__(
        self,
        atom_coords: np.ndarray,
        atom_types: List[str],
        residue_ids: List[int],
        residue_names: List[str],
        atom_names: List[str],
        chain_ids: Optional[List[str]] = None,
    ):
        """
        Initialize a ProteinStructure.
        
        Args:
            atom_coords: (N, 3) array of atom coordinates in Angstroms
            atom_types: List of atom element types (e.g., 'C', 'N', 'O')
            residue_ids: List of residue sequence numbers
            residue_names: List of 3-letter residue codes (e.g., 'ALA', 'GLY')
            atom_names: List of atom names (e.g., 'CA', 'CB', 'N')
            chain_ids: Optional list of chain identifiers
        """
        self.atom_coords = atom_coords
        self.atom_types = atom_types
        self.residue_ids = residue_ids
        self.residue_names = residue_names
        self.atom_names = atom_names
        self.chain_ids = chain_ids if chain_ids is not None else ["A"] * len(atom_coords)
        
        # Create residue mapping
        self._build_residue_mapping()
    
    def _build_residue_mapping(self):
        """Build mapping from residue index to atom indices."""
        self.residue_to_atoms = {}
        unique_residues = {}
        
        for i, (res_id, res_name, chain_id) in enumerate(
            zip(self.residue_ids, self.residue_names, self.chain_ids)
        ):
            key = (chain_id, res_id, res_name)
            if key not in unique_residues:
                unique_residues[key] = len(unique_residues)
            res_idx = unique_residues[key]
            
            if res_idx not in self.residue_to_atoms:
                self.residue_to_atoms[res_idx] = []
            self.residue_to_atoms[res_idx].append(i)
        
        self.num_residues = len(unique_residues)
        self.residue_info = list(unique_residues.keys())
    
    def get_backbone_atoms(self, residue_idx: int) -> List[int]:
        """Get atom indices for backbone atoms of a residue."""
        atom_indices = self.residue_to_atoms.get(residue_idx, [])
        return [
            idx for idx in atom_indices
            if self.atom_names[idx] in BACKBONE_ATOMS
        ]
    
    def get_sidechain_atoms(self, residue_idx: int) -> List[int]:
        """Get atom indices for sidechain atoms of a residue."""
        atom_indices = self.residue_to_atoms.get(residue_idx, [])
        return [
            idx for idx in atom_indices
            if self.atom_names[idx] not in BACKBONE_ATOMS
        ]
    
    def get_residue_centroid(self, residue_idx: int, atom_indices: List[int]) -> np.ndarray:
        """Compute centroid of specified atoms in a residue."""
        if not atom_indices:
            return np.array([0.0, 0.0, 0.0])
        coords = self.atom_coords[atom_indices]
        return coords.mean(axis=0)
    
    def to_torch(self, device: str = "cpu") -> torch.Tensor:
        """Convert atom coordinates to torch tensor."""
        return torch.from_numpy(self.atom_coords.astype(np.float32)).to(device)


def load_pdb_file(
    pdb_path: Path,
    use_mmcif: bool = False
) -> ProteinStructure:
    """
    Load a protein structure from a PDB or mmCIF file.
    
    Args:
        pdb_path: Path to PDB or mmCIF file
        use_mmcif: If True, use mmCIF parser instead of PDB parser
        
    Returns:
        ProteinStructure object containing atom coordinates and metadata
        
    Raises:
        ImportError: If BioPython is not installed
        FileNotFoundError: If the file does not exist
        ValueError: If the file cannot be parsed
    """
    if not BIOPYTHON_AVAILABLE:
        raise ImportError(
            "BioPython is required for PDB loading. Install with: pip install biopython"
        )
    
    pdb_path = Path(pdb_path)
    if not pdb_path.exists():
        raise FileNotFoundError(f"PDB file not found: {pdb_path}")
    
    try:
        if use_mmcif or pdb_path.suffix.lower() in [".cif", ".mmcif"]:
            parser = MMCIFParser(QUIET=True)
            structure = parser.get_structure("protein", str(pdb_path))
        else:
            parser = PDBParser(QUIET=True)
            structure = parser.get_structure("protein", str(pdb_path))
    except Exception as e:
        raise ValueError(f"Failed to parse PDB file '{pdb_path}': {e}") from e
    
    # Extract atom information
    atom_coords = []
    atom_types = []
    residue_ids = []
    residue_names = []
    atom_names = []
    chain_ids = []
    
    for model in structure:
        for chain in model:
            chain_id = chain.id
            for residue in chain:
                # Skip heteroatoms (water, ligands) for now
                if residue.id[0] != " ":
                    continue
                
                res_id = residue.id[1]
                res_name = residue.get_resname()
                
                for atom in residue:
                    atom_coords.append(atom.coord)
                    atom_types.append(atom.element)
                    residue_ids.append(res_id)
                    residue_names.append(res_name)
                    atom_names.append(atom.name)
                    chain_ids.append(chain_id)
    
    if len(atom_coords) == 0:
        raise ValueError(f"No atoms found in PDB file: {pdb_path}")
    
    atom_coords = np.array(atom_coords, dtype=np.float32)
    
    return ProteinStructure(
        atom_coords=atom_coords,
        atom_types=atom_types,
        residue_ids=residue_ids,
        residue_names=residue_names,
        atom_names=atom_names,
        chain_ids=chain_ids,
    )


def get_atom_vdw_radius(atom_type: str) -> float:
    """Get van der Waals radius for an atom type."""
    atom_type_upper = atom_type.upper()
    return VDW_RADII.get(atom_type_upper, VDW_RADII["DEFAULT"])


def get_residue_hydrophobicity(residue_name: str) -> float:
    """Get normalized hydrophobicity score for a residue (0-1 scale)."""
    res_name_upper = residue_name.upper()
    return HYDROPHOBICITY.get(res_name_upper, 0.5)


def is_charged_residue(residue_name: str) -> bool:
    """Check if a residue is charged."""
    return residue_name.upper() in CHARGED_RESIDUES


def compute_simple_sasa(
    protein: ProteinStructure,
    probe_radius: float = 1.4
) -> np.ndarray:
    """
    Compute a simple approximation of Solvent-Accessible Surface Area (SASA).
    
    This is a simplified version. For accurate SASA, use specialized tools
    like FreeSASA or BioPython's DSSP.
    
    Args:
        protein: ProteinStructure object
        probe_radius: Radius of water probe (default 1.4 Angstroms)
        
    Returns:
        Array of SASA values per atom (normalized to [0, 1])
    """
    from scipy.spatial.distance import cdist
    
    coords = protein.atom_coords
    n_atoms = len(coords)
    
    # Simple approximation: use distance to nearest neighbor as proxy for exposure
    # More exposed atoms have larger distances to neighbors
    if n_atoms < 2:
        return np.ones(n_atoms, dtype=np.float32)
    
    # Compute pairwise distances
    distances = cdist(coords, coords)
    np.fill_diagonal(distances, np.inf)
    
    # Use minimum distance to neighbor as exposure proxy
    min_distances = np.min(distances, axis=1)
    
    # Normalize: larger min_distance = more exposed
    # Use a reasonable cutoff (e.g., 3.0 Angstroms)
    max_dist = np.percentile(min_distances, 95)  # Use 95th percentile as normalization
    sasa_approx = np.clip(min_distances / max_dist, 0.0, 1.0)
    
    return sasa_approx.astype(np.float32)

