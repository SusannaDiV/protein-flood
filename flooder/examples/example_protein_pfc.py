"""Example: Protein Flood Complex (PFC) on protein structures from scPDB.

This script demonstrates how to:
1. Load protein structures from PDB files
2. Construct Protein Flood Complexes with residue-aware landmarks
3. Compute persistent homology for protein topology analysis

Copyright (c) 2025
"""

from pathlib import Path
from typing import Dict, Optional
import torch
import numpy as np
from gudhi import SimplexTree

from flooder.protein_io import load_pdb_file
from flooder.pfc import protein_flood_complex
from flooder.persistence_vectorization import (
    compute_persistence_images,
    compute_persistence_images_from_simplex_tree,
    get_feature_dimension,
)


def process_protein_pdb(
    pdb_path: Path,
    target_landmarks: int = 500,
    max_dimension: int = 2,
    device: str = "cpu",
    output_dir: Optional[Path] = None,
    use_pls: bool = False,
) -> Dict:
    """
    Process a single protein PDB file and compute PFC persistent homology.
    
    Args:
        pdb_path: Path to PDB file
        target_landmarks: Target number of landmarks
        max_dimension: Maximum homology dimension
        device: Computation device ('cpu' or 'cuda')
        output_dir: Optional directory to save results
        use_pls: If True, use full Protein Landmark Sampling (PLS) algorithm.
                 If False (default), use simplified residue-based selection.
        
    Returns:
        Dictionary with persistence diagrams and metadata
    """
    print(f"\n{'='*60}")
    print(f"Processing: {pdb_path.name}")
    print(f"  Full path: {pdb_path}")
    print(f"  Device: {device}")
    print(f"  Target landmarks: {target_landmarks}")
    print(f"  Use PLS: {use_pls}")
    print(f"{'='*60}")
    
    # Load protein structure
    print("Loading protein structure...")
    try:
        protein = load_pdb_file(pdb_path)
        print(f"  ✓ Loaded {len(protein.atom_coords)} atoms")
        print(f"  ✓ {protein.num_residues} residues")
        print(f"  ✓ Protein structure loaded successfully")
    except Exception as e:
        print(f"  ✗ ERROR: Failed to load PDB: {e}")
        import traceback
        traceback.print_exc()
        return None
    
    # Construct Protein Flood Complex
    landmark_method = "Full PLS" if use_pls else "Simplified (residue-based)"
    print(f"\nConstructing Protein Flood Complex ({landmark_method})...")
    if not use_pls:
        print("  Note: Simplified PFC is ~3-6x faster than full PLS")
        print("        (Simplified: ~1.7-4.5s, Full PLS: ~6.5-19s for typical proteins)")
        print("        Simplified PFC is often 1.1-1.3x FASTER than standard flooder")
        print("        due to more efficient circumball coverage vs witness points")
    
    import time
    pfc_start_time = time.time()
    try:
        print("  Starting PFC construction...")
        pfc_stree = protein_flood_complex(
            protein,
            target_landmarks=target_landmarks,
            max_dimension=max_dimension,
            device=device,
            return_simplex_tree=True,
            r0=1.0,
            alpha=1.0,
            beta=0.5,
            gamma_h=0.2,
            gamma_q=0.2,
            use_pls=use_pls,  # Use full PLS algorithm if True
        )
        pfc_time = time.time() - pfc_start_time
        num_simplices = pfc_stree.num_simplices()
        print(f"  ✓ Complex constructed successfully")
        print(f"  ✓ Complex has {num_simplices} simplices")
        print(f"  ✓ PFC construction time: {pfc_time:.2f} seconds")
    except Exception as e:
        print(f"  ✗ ERROR: Failed to construct PFC: {e}")
        import traceback
        traceback.print_exc()
        return None
    
    # Compute persistent homology
    print("\nComputing persistent homology...")
    ph_start_time = time.time()
    try:
        # Validate simplex tree before computing persistence
        print("  Validating simplex tree...")
        num_simplices = pfc_stree.num_simplices()
        print(f"  ✓ Simplex tree has {num_simplices} simplices")
        
        # Check for potential issues
        if num_simplices == 0:
            print("  ✗ ERROR: Simplex tree is empty!")
            return None
        
        # Try to make filtration non-decreasing (safety check)
        print("  Ensuring filtration is non-decreasing...")
        try:
            pfc_stree.make_filtration_non_decreasing()
            print("  ✓ Filtration validated")
        except Exception as e:
            print(f"  ⚠ WARNING: Could not validate filtration: {e}")
        
        print("  Computing persistence...")
        # Use a subprocess or try-catch to handle segfaults better
        import signal
        import sys
        
        def timeout_handler(signum, frame):
            raise TimeoutError("Persistence computation timed out")
        
        # Set a timeout (30 seconds should be enough)
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(30)
        
        try:
            pfc_stree.compute_persistence()
            signal.alarm(0)  # Cancel alarm
        except TimeoutError:
            signal.alarm(0)
            print("  ✗ ERROR: Persistence computation timed out")
            return None
        except Exception as e:
            signal.alarm(0)
            raise e
        
        ph_time = time.time() - ph_start_time
        print(f"  ✓ Persistence computed in {ph_time:.2f} seconds")
        
        # Extract persistence diagrams
        print("  Extracting persistence diagrams...")
        diagrams = []
        for dim in range(max_dimension + 1):
            diag = pfc_stree.persistence_intervals_in_dimension(dim)
            diagrams.append(diag)
            n_features = len([p for p in diag if p[1] != float('inf')])
            n_infinite = len([p for p in diag if p[1] == float('inf')])
            print(f"  ✓ H{dim}: {n_features} finite features, {n_infinite} infinite features")
        
        # Compute persistence images for ML
        print("\nComputing persistence images...")
        images_start_time = time.time()
        try:
            persistence_images = compute_persistence_images_from_simplex_tree(
                pfc_stree,
                max_dimension=max_dimension,
                bandwidth=1.0,
                resolution=(20, 20),
                normalize=True,
            )
            images_time = time.time() - images_start_time
            feature_dim = get_feature_dimension(num_diagrams=max_dimension + 1, resolution=(20, 20))
            print(f"  ✓ Persistence images computed in {images_time:.2f} seconds")
            print(f"  ✓ Feature vector shape: ({feature_dim},)")
        except Exception as e:
            print(f"  ✗ WARNING: Failed to compute persistence images: {e}")
            import traceback
            traceback.print_exc()
            persistence_images = None
        
        results = {
            "pdb_path": str(pdb_path),
            "pdb_name": pdb_path.stem,
            "num_atoms": len(protein.atom_coords),
            "num_residues": protein.num_residues,
            "num_landmarks": target_landmarks,
            "num_simplices": pfc_stree.num_simplices(),
            "persistence_diagrams": diagrams,
            "persistence_images": persistence_images,  # ML-ready features
            "simplex_tree": pfc_stree,
        }
        
        print("\n" + "="*60)
        print("Summary:")
        print(f"  Protein: {pdb_path.stem}")
        print(f"  Atoms: {len(protein.atom_coords)}")
        print(f"  Residues: {protein.num_residues}")
        print(f"  Landmarks: {target_landmarks}")
        print(f"  Simplices: {pfc_stree.num_simplices()}")
        print(f"  PFC time: {pfc_time:.2f}s")
        print(f"  PH time: {ph_time:.2f}s")
        if persistence_images is not None:
            print(f"  Images time: {images_time:.2f}s")
            print(f"  Total time: {pfc_time + ph_time + images_time:.2f}s")
        else:
            print(f"  Total time: {pfc_time + ph_time:.2f}s")
        print("="*60)
        
        # Save results if output directory specified
        if output_dir is not None:
            print(f"\nSaving results...")
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            output_file = output_dir / f"{pdb_path.stem}_pfc_results.pt"
            torch.save(results, output_file)
            print(f"  ✓ Results saved to: {output_file}")
        else:
            print("\nNote: No output directory specified, results not saved")
        
        return results
        
    except Exception as e:
        print(f"  ✗ ERROR: Failed to compute persistence: {e}")
        import traceback
        traceback.print_exc()
        return None


def process_scpdb_directory(
    scpdb_dir: Path,
    max_proteins: Optional[int] = 1,
    target_landmarks: int = 500,
    device: str = "cpu",
    output_dir: Optional[Path] = None,
    use_pls: bool = False,
):
    """
    Process multiple proteins from scPDB directory.
    
    Args:
        scpdb_dir: Path to scPDB directory
        max_proteins: Maximum number of proteins to process (default: 1, None = all)
        target_landmarks: Target number of landmarks per protein
        device: Computation device
        output_dir: Directory to save results
        use_pls: If True, use full PLS algorithm; if False, use simplified
                 Note: Simplified PLS is typically 3-5x faster than full PLS
    """
    print("\n" + "="*60)
    print("BATCH PROCESSING: scPDB Directory")
    print("="*60)
    print(f"  Directory: {scpdb_dir}")
    print(f"  Device: {device}")
    print(f"  Target landmarks: {target_landmarks}")
    print(f"  Use PLS: {use_pls}")
    if max_proteins is not None:
        print(f"  Max proteins: {max_proteins}")
    else:
        print(f"  Max proteins: ALL")
    if output_dir is not None:
        print(f"  Output directory: {output_dir}")
    print("="*60)
    
    scpdb_dir = Path(scpdb_dir)
    
    if not scpdb_dir.exists():
        print(f"\n✗ ERROR: scPDB directory not found: {scpdb_dir}")
        return
    
    # Find all PDB files
    # scPDB structure: files are in subdirectories like 2011-2019/
    print(f"\nSearching for PDB files in {scpdb_dir}...")
    
    # Try common scPDB subdirectory structures
    search_paths = [
        scpdb_dir / "2011-2019",  # Common scPDB structure
        scpdb_dir,  # Also search root
    ]
    
    pdb_files = []
    for search_path in search_paths:
        if search_path.exists():
            print(f"  Searching in: {search_path}")
            found_pdb = list(search_path.glob("**/*.pdb"))
            found_ent = list(search_path.glob("**/*.ent"))
            pdb_files.extend(found_pdb)
            pdb_files.extend(found_ent)
            if found_pdb or found_ent:
                print(f"    Found {len(found_pdb)} .pdb and {len(found_ent)} .ent files")
    
    # Remove duplicates
    pdb_files = list(set(pdb_files))
    
    if not pdb_files:
        print(f"✗ WARNING: No PDB files found in {scpdb_dir}")
        print(f"  Searched in: {[str(p) for p in search_paths if p.exists()]}")
        return
    
    print(f"  ✓ Found {len(pdb_files)} total PDB files")
    
    if max_proteins is not None:
        pdb_files = pdb_files[:max_proteins]
        print(f"  → Processing first {len(pdb_files)} files")
    else:
        print(f"  → Processing all {len(pdb_files)} files")
    
    results_list = []
    successful = 0
    failed = 0
    
    import time
    total_start_time = time.time()
    
    for i, pdb_file in enumerate(pdb_files, 1):
        print(f"\n{'#'*60}")
        print(f"# Protein {i}/{len(pdb_files)}")
        print(f"{'#'*60}")
        result = process_protein_pdb(
            pdb_file,
            target_landmarks=target_landmarks,
            device=device,
            output_dir=output_dir,
            use_pls=use_pls,
        )
        
        if result is not None:
            results_list.append(result)
            successful += 1
            print(f"\n✓ Protein {i}/{len(pdb_files)} completed successfully")
        else:
            failed += 1
            print(f"\n✗ Protein {i}/{len(pdb_files)} failed")
    
    total_time = time.time() - total_start_time
    
    print(f"\n{'='*60}")
    print("BATCH PROCESSING SUMMARY")
    print(f"{'='*60}")
    print(f"  Total proteins processed: {len(pdb_files)}")
    print(f"  ✓ Successful: {successful}")
    print(f"  ✗ Failed: {failed}")
    print(f"  Total time: {total_time:.2f} seconds")
    if successful > 0:
        print(f"  Average time per protein: {total_time/successful:.2f} seconds")
    print(f"{'='*60}")
    
    return results_list


def main():
    """Main function to run PFC on scPDB proteins."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Process proteins with Protein Flood Complex"
    )
    parser.add_argument(
        "--scpdb_dir",
        type=str,
        default=r"C:\Users\s.divita\Downloads\scPDB\scPDB",
        help="Path to scPDB directory",
    )
    parser.add_argument(
        "--max_proteins",
        type=int,
        default=1,
        help="Maximum number of proteins to process (default: 1, use None for all)",
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
        "--output_dir",
        type=str,
        default=None,
        help="Directory to save results",
    )
    parser.add_argument(
        "--use_pls",
        action="store_true",
        help="Use full Protein Landmark Sampling (PLS) algorithm instead of simplified",
    )
    
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("PROTEIN FLOOD COMPLEX (PFC) - Example Script")
    print("="*60)
    print(f"Arguments:")
    print(f"  scpdb_dir: {args.scpdb_dir}")
    print(f"  max_proteins: {args.max_proteins}")
    print(f"  target_landmarks: {args.target_landmarks}")
    print(f"  device: {args.device}")
    print(f"  output_dir: {args.output_dir}")
    print(f"  use_pls: {args.use_pls}")
    print("="*60)
    
    # Check device availability
    if args.device == "cuda":
        if torch.cuda.is_available():
            print(f"\n✓ CUDA available: {torch.cuda.get_device_name(0)}")
            print(f"  GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        else:
            print("\n⚠ WARNING: CUDA requested but not available, using CPU")
            args.device = "cpu"
    else:
        print("\n→ Using CPU")
    
    # Process proteins
    results = process_scpdb_directory(
        scpdb_dir=Path(args.scpdb_dir),
        max_proteins=args.max_proteins,
        target_landmarks=args.target_landmarks,
        device=args.device,
        output_dir=Path(args.output_dir) if args.output_dir else None,
        use_pls=args.use_pls,
    )
    
    print("\n" + "="*60)
    print("SCRIPT COMPLETED")
    print("="*60)
    
    return results


if __name__ == "__main__":
    main()

