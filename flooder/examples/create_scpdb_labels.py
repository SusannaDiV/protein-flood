"""Helper script to create labels from scPDB dataset.

scPDB contains binding site information. This script creates a labels JSON file
for use with the ablation study.
"""

from pathlib import Path
import json
import argparse


def create_labels_from_scpdb(
    scpdb_dir: Path,
    output_file: Path,
    has_ligand_is_positive: bool = True,
) -> None:
    """
    Create labels JSON from scPDB directory.
    
    scPDB structure:
    scPDB/
      protein1/
        protein1.pdb
        ligand.pdb  <- indicates binding pocket
      protein2/
        protein2.pdb
        (no ligand) <- no binding pocket
    
    Args:
        scpdb_dir: Path to scPDB directory
        output_file: Path to save labels JSON
        has_ligand_is_positive: If True, proteins with ligands = 1 (has pocket)
    """
    scpdb_dir = Path(scpdb_dir)
    labels = {}
    
    # Find all PDB files
    pdb_files = list(scpdb_dir.glob("**/*.pdb"))
    pdb_files.extend(list(scpdb_dir.glob("**/*.ent")))
    
    print(f"Found {len(pdb_files)} PDB files")
    
    for pdb_file in pdb_files:
        protein_name = pdb_file.stem
        
        # Check for ligand file in same directory
        ligand_file = pdb_file.parent / "ligand.pdb"
        if not ligand_file.exists():
            # Try other common names
            ligand_file = pdb_file.parent / "ligand.mol2"
            if not ligand_file.exists():
                ligand_file = pdb_file.parent / f"{protein_name}_ligand.pdb"
        
        # Determine label
        if ligand_file.exists():
            label = 1 if has_ligand_is_positive else 0
        else:
            label = 0 if has_ligand_is_positive else 1
        
        labels[protein_name] = label
    
    # Save labels
    with open(output_file, 'w') as f:
        json.dump(labels, f, indent=2)
    
    # Print summary
    positive = sum(1 for v in labels.values() if v == 1)
    negative = len(labels) - positive
    
    print(f"\nLabels created:")
    print(f"  Total proteins: {len(labels)}")
    print(f"  Positive (has pocket): {positive}")
    print(f"  Negative (no pocket): {negative}")
    print(f"\nSaved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Create labels JSON from scPDB dataset"
    )
    parser.add_argument(
        "--scpdb_dir",
        type=str,
        required=True,
        help="Path to scPDB directory",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="scpdb_labels.json",
        help="Output JSON file for labels",
    )
    parser.add_argument(
        "--has_ligand_is_positive",
        action="store_true",
        default=True,
        help="Proteins with ligands = positive class (has binding pocket)",
    )
    
    args = parser.parse_args()
    
    create_labels_from_scpdb(
        Path(args.scpdb_dir),
        Path(args.output_file),
        has_ligand_is_positive=args.has_ligand_is_positive,
    )


if __name__ == "__main__":
    main()

