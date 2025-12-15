"""Batch process proteins from scPDB directory with Protein Flood Complex.

This script processes all proteins in the scPDB directory structure and
saves persistence diagrams for downstream analysis.

Usage:
    python batch_process_scpdb.py --scpdb_dir <path> --output_dir <path> [options]
"""

import argparse
from pathlib import Path
from typing import List, Optional
import torch
import numpy as np

try:
    from tqdm import tqdm
except ImportError:
    # Fallback if tqdm not available
    def tqdm(iterable, desc=None):
        if desc:
            print(desc)
        return iterable

from flooder.protein_io import load_pdb_file
from flooder.pfc import protein_flood_complex


def find_pdb_files(scpdb_dir: Path) -> List[Path]:
    """Find all PDB files in scPDB directory structure."""
    pdb_files = []
    
    # scPDB typically has structure: scPDB/XXXX_XXXX/pdb_file.pdb
    for pdb_file in scpdb_dir.glob("**/*.pdb"):
        pdb_files.append(pdb_file)
    
    # Also check for .ent files (some PDB files use this extension)
    for ent_file in scpdb_dir.glob("**/*.ent"):
        pdb_files.append(ent_file)
    
    return sorted(pdb_files)


def process_single_protein(
    pdb_path: Path,
    target_landmarks: int = 500,
    max_dimension: int = 2,
    device: str = "cpu",
    r0: float = 1.0,
    alpha: float = 1.0,
    beta: float = 0.5,
    gamma_h: float = 0.2,
    gamma_q: float = 0.2,
) -> Optional[dict]:
    """
    Process a single protein and return results.
    
    Returns:
        Dictionary with persistence diagrams and metadata, or None if failed
    """
    try:
        # Load protein
        protein = load_pdb_file(pdb_path)
        
        # Construct PFC
        pfc_stree = protein_flood_complex(
            protein,
            target_landmarks=target_landmarks,
            max_dimension=max_dimension,
            device=device,
            return_simplex_tree=True,
            r0=r0,
            alpha=alpha,
            beta=beta,
            gamma_h=gamma_h,
            gamma_q=gamma_q,
        )
        
        # Compute persistence
        pfc_stree.compute_persistence()
        
        # Extract diagrams
        diagrams = []
        for dim in range(max_dimension + 1):
            diag = pfc_stree.persistence_intervals_in_dimension(dim)
            diagrams.append(diag)
        
        return {
            "pdb_path": str(pdb_path),
            "pdb_name": pdb_path.stem,
            "num_atoms": len(protein.atom_coords),
            "num_residues": protein.num_residues,
            "num_landmarks": target_landmarks,
            "num_simplices": pfc_stree.num_simplices(),
            "persistence_diagrams": diagrams,
        }
        
    except Exception as e:
        print(f"  Error processing {pdb_path.name}: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(
        description="Batch process scPDB proteins with Protein Flood Complex"
    )
    parser.add_argument(
        "--scpdb_dir",
        type=str,
        default=r"C:\Users\s.divita\Downloads\scPDB\scPDB",
        help="Path to scPDB directory",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory for results",
    )
    parser.add_argument(
        "--max_proteins",
        type=int,
        default=None,
        help="Maximum number of proteins to process",
    )
    parser.add_argument(
        "--target_landmarks",
        type=int,
        default=500,
        help="Target number of landmarks per protein",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="Computation device",
    )
    parser.add_argument(
        "--r0",
        type=float,
        default=1.0,
        help="Global radius offset",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=1.0,
        help="Weight for atom radius term",
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=0.5,
        help="Weight for SASA term",
    )
    parser.add_argument(
        "--gamma_h",
        type=float,
        default=0.2,
        help="Weight for hydrophobicity modulation",
    )
    parser.add_argument(
        "--gamma_q",
        type=float,
        default=0.2,
        help="Weight for charge modulation",
    )
    
    args = parser.parse_args()
    
    # Check device
    if args.device == "cuda" and not torch.cuda.is_available():
        print("WARNING: CUDA not available, using CPU")
        args.device = "cpu"
    
    # Find PDB files
    scpdb_dir = Path(args.scpdb_dir)
    if not scpdb_dir.exists():
        print(f"ERROR: scPDB directory not found: {scpdb_dir}")
        return
    
    pdb_files = find_pdb_files(scpdb_dir)
    print(f"Found {len(pdb_files)} PDB files")
    
    if args.max_proteins is not None:
        pdb_files = pdb_files[:args.max_proteins]
        print(f"Processing first {len(pdb_files)} files")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Process proteins
    results = []
    successful = 0
    failed = 0
    
    for pdb_file in tqdm(pdb_files, desc="Processing proteins"):
        result = process_single_protein(
            pdb_file,
            target_landmarks=args.target_landmarks,
            device=args.device,
            r0=args.r0,
            alpha=args.alpha,
            beta=args.beta,
            gamma_h=args.gamma_h,
            gamma_q=args.gamma_q,
        )
        
        if result is not None:
            results.append(result)
            successful += 1
            
            # Save individual result
            output_file = output_dir / f"{pdb_file.stem}_pfc.pt"
            torch.save(result, output_file)
        else:
            failed += 1
    
    # Save summary
    summary = {
        "total_processed": len(pdb_files),
        "successful": successful,
        "failed": failed,
        "parameters": {
            "target_landmarks": args.target_landmarks,
            "r0": args.r0,
            "alpha": args.alpha,
            "beta": args.beta,
            "gamma_h": args.gamma_h,
            "gamma_q": args.gamma_q,
        },
    }
    
    summary_file = output_dir / "summary.pt"
    torch.save(summary, summary_file)
    
    print(f"\n{'='*60}")
    print(f"Summary:")
    print(f"  Total: {len(pdb_files)}")
    print(f"  Successful: {successful}")
    print(f"  Failed: {failed}")
    print(f"  Results saved to: {output_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()

