"""Ablation Study: Protein-Aware Flooding Components

Experiment C: Protein-aware flooding ablation (physics matters)

This script validates Contribution 1 by comparing:
- Full PFC (vdW + SASA + hydro/charge + residue landmarks)
- Uniform balls (no weighting)
- No SASA (β=0)
- No atom radius (α=0)
- No hydro/charge modulation (γ_h=γ_q=0)
- Landmarks not residue-aware (generic sampling)

Reports AUROC for pocket classification and runtime.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import time
import json

from flooder.protein_io import load_pdb_file
from flooder.pfc import protein_flood_complex
from flooder.persistence_vectorization import (
    compute_persistence_images_from_simplex_tree,
    get_feature_dimension,
)
from flooder.core import flood_complex, generate_landmarks


class SimpleMLP(nn.Module):
    """Simple MLP for binary classification."""
    
    def __init__(self, input_dim: int, hidden_dims: List[int] = [128, 64]):
        super().__init__()
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.2))
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, 1))  # Binary classification
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x).squeeze(-1)


def extract_pfc_features(
    pdb_path: Path,
    variant: str = "full",
    target_landmarks: int = 500,
    device: str = "cpu",
) -> Tuple[np.ndarray, float]:
    """
    Extract features using different PFC variants.
    
    Args:
        pdb_path: Path to PDB file
        variant: Which variant to use:
            - "full": Full PFC (vdW + SASA + hydro/charge + residue landmarks)
            - "uniform": Uniform balls (no weighting)
            - "no_sasa": No SASA (β=0)
            - "no_atom_radius": No atom radius (α=0)
            - "no_chemistry": No hydro/charge (γ_h=γ_q=0)
            - "generic_landmarks": Generic sampling (not residue-aware)
        target_landmarks: Number of landmarks
        device: Computation device
        
    Returns:
        Tuple of (features, runtime_seconds)
    """
    start_time = time.time()
    
    # Load protein
    protein = load_pdb_file(pdb_path)
    
    if variant == "generic_landmarks":
        # Use generic FPS landmarks instead of residue-based
        # BUT still use full PFC weights (to isolate landmark effect)
        coords_tensor = torch.from_numpy(protein.atom_coords.astype(np.float32))
        if device == "cuda" and torch.cuda.is_available():
            coords_tensor = coords_tensor.cuda()
        
        # Generate landmarks using standard FPS (not residue-aware)
        landmarks = generate_landmarks(
            coords_tensor,
            num_landmarks=target_landmarks,
            device=device,
        )
        
        # Compute protein-aware weights for these generic landmarks
        # Map generic landmarks to nearest residues for attribute computation
        from flooder.protein_landmarks import get_landmark_attributes
        from flooder.pfc import compute_landmark_weights
        
        # Find nearest residue for each landmark (simplified mapping)
        landmark_to_residue = {}
        for i, landmark in enumerate(landmarks.cpu().numpy()):
            # Find nearest atom, then get its residue
            distances = np.linalg.norm(protein.atom_coords - landmark, axis=1)
            nearest_atom_idx = np.argmin(distances)
            # Get residue index for this atom
            for res_idx, atom_indices in protein.residue_to_atoms.items():
                if nearest_atom_idx in atom_indices:
                    landmark_to_residue[i] = res_idx
                    break
            else:
                landmark_to_residue[i] = 0  # Default to first residue
        
        # Get attributes and compute weights
        attributes = get_landmark_attributes(protein, landmark_to_residue, landmarks)
        for key in attributes:
            if isinstance(attributes[key], torch.Tensor):
                attributes[key] = attributes[key].to(device)
        
        landmark_weights = compute_landmark_weights(
            attributes["atom_radii"],
            attributes["sasa"],
            attributes["hydrophobicity"],
            attributes["charge"],
            r0=1.0,
            alpha=1.0,  # Full PFC weights
            beta=0.5,
            gamma_h=0.2,
            gamma_q=0.2,
        )
        
        # Build complex with generic landmarks but PFC weights
        stree = flood_complex(
            points=landmarks,
            landmarks=landmarks,
            max_dimension=2,
            device=device,
            return_simplex_tree=True,
            landmark_weights=landmark_weights,  # PFC weights on generic landmarks
        )
        
    else:
        # Use residue-based landmarks
        if variant == "uniform":
            # Full PFC but with uniform weights (all weights = 1.0)
            pfc_stree = protein_flood_complex(
                protein,
                target_landmarks=target_landmarks,
                max_dimension=2,
                device=device,
                return_simplex_tree=True,
                r0=1.0,
                alpha=0.0,  # No atom radius
                beta=0.0,   # No SASA
                gamma_h=0.0,  # No hydrophobicity
                gamma_q=0.0,  # No charge
                use_pls=False,
            )
        elif variant == "no_sasa":
            # No SASA term
            pfc_stree = protein_flood_complex(
                protein,
                target_landmarks=target_landmarks,
                max_dimension=2,
                device=device,
                return_simplex_tree=True,
                r0=1.0,
                alpha=1.0,  # Keep atom radius
                beta=0.0,   # No SASA
                gamma_h=0.2,  # Keep chemistry
                gamma_q=0.2,
                use_pls=False,
            )
        elif variant == "no_atom_radius":
            # No atom radius term
            pfc_stree = protein_flood_complex(
                protein,
                target_landmarks=target_landmarks,
                max_dimension=2,
                device=device,
                return_simplex_tree=True,
                r0=1.0,
                alpha=0.0,  # No atom radius
                beta=0.5,   # Keep SASA
                gamma_h=0.2,  # Keep chemistry
                gamma_q=0.2,
                use_pls=False,
            )
        elif variant == "no_chemistry":
            # No hydro/charge modulation
            pfc_stree = protein_flood_complex(
                protein,
                target_landmarks=target_landmarks,
                max_dimension=2,
                device=device,
                return_simplex_tree=True,
                r0=1.0,
                alpha=1.0,  # Keep atom radius
                beta=0.5,   # Keep SASA
                gamma_h=0.0,  # No hydrophobicity
                gamma_q=0.0,  # No charge
                use_pls=False,
            )
        else:  # "full"
            # Full PFC with all components
            pfc_stree = protein_flood_complex(
                protein,
                target_landmarks=target_landmarks,
                max_dimension=2,
                device=device,
                return_simplex_tree=True,
                r0=1.0,
                alpha=1.0,
                beta=0.5,
                gamma_h=0.2,
                gamma_q=0.2,
                use_pls=False,
            )
        
        stree = pfc_stree
    
    # Compute persistence
    stree.compute_persistence()
    
    # Convert to persistence images
    features = compute_persistence_images_from_simplex_tree(
        stree,
        max_dimension=2,
        bandwidth=1.0,
        resolution=(20, 20),
        normalize=True,
    )
    
    runtime = time.time() - start_time
    
    return features, runtime


def load_protein_labels(
    pdb_files: List[Path],
    labels_file: Optional[Path] = None,
) -> np.ndarray:
    """
    Load labels for proteins.
    
    Args:
        pdb_files: List of PDB file paths
        labels_file: Optional path to JSON file with labels
                    Format: {"protein_name": 0 or 1, ...}
                    If None, creates dummy labels for demonstration
        
    Returns:
        Array of labels (0 = no pocket, 1 = has pocket)
    """
    if labels_file is not None and labels_file.exists():
        with open(labels_file, 'r') as f:
            labels_dict = json.load(f)
        
        labels = []
        for pdb_file in pdb_files:
            protein_name = pdb_file.stem
            label = labels_dict.get(protein_name, 0)
            labels.append(label)
        
        return np.array(labels, dtype=np.int32)
    else:
        # Dummy labels for demonstration
        # In practice, you would load real labels from your dataset
        print("WARNING: Using dummy labels. Provide --labels_file for real labels.")
        return np.random.randint(0, 2, size=len(pdb_files), dtype=np.int32)


def train_and_evaluate(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    device: str = "cpu",
    epochs: int = 50,
) -> Tuple[float, Dict]:
    """
    Train MLP and evaluate on test set.
    
    Returns:
        Tuple of (AUROC, metrics_dict)
    """
    # Normalize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Convert to tensors
    X_train_t = torch.from_numpy(X_train_scaled).float()
    y_train_t = torch.from_numpy(y_train).float()
    X_test_t = torch.from_numpy(X_test_scaled).float()
    y_test_t = torch.from_numpy(y_test).float()
    
    if device == "cuda" and torch.cuda.is_available():
        X_train_t = X_train_t.cuda()
        y_train_t = y_train_t.cuda()
        X_test_t = X_test_t.cuda()
        y_test_t = y_test_t.cuda()
    
    # Create model
    model = SimpleMLP(input_dim=X_train.shape[1])
    if device == "cuda" and torch.cuda.is_available():
        model = model.cuda()
    
    # Training
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.BCEWithLogitsLoss()
    
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(X_train_t)
        loss = criterion(outputs, y_train_t)
        loss.backward()
        optimizer.step()
    
    # Evaluation
    model.eval()
    with torch.no_grad():
        test_outputs = model(X_test_t)
        test_probs = torch.sigmoid(test_outputs).cpu().numpy()
    
    # Compute AUROC
    auroc = roc_auc_score(y_test, test_probs)
    
    # Additional metrics
    fpr, tpr, thresholds = roc_curve(y_test, test_probs)
    
    metrics = {
        "auroc": auroc,
        "fpr": fpr.tolist(),
        "tpr": tpr.tolist(),
        "thresholds": thresholds.tolist(),
    }
    
    return auroc, metrics


def run_ablation_study(
    pdb_files: List[Path],
    labels: np.ndarray,
    variants: List[str],
    device: str = "cpu",
    test_size: float = 0.2,
    random_seed: int = 42,
) -> Dict:
    """
    Run ablation study comparing different PFC variants.
    
    Args:
        pdb_files: List of PDB file paths
        labels: Array of labels (0 or 1)
        variants: List of variants to test
        device: Computation device
        test_size: Fraction for test set
        random_seed: Random seed for reproducibility
        
    Returns:
        Dictionary with results for each variant
    """
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    
    results = {}
    
    for variant in variants:
        print(f"\n{'='*60}")
        print(f"Testing variant: {variant}")
        print(f"{'='*60}")
        
        # Extract features for all proteins
        all_features = []
        all_runtimes = []
        failed = 0
        
        for i, pdb_file in enumerate(pdb_files):
            print(f"  [{i+1}/{len(pdb_files)}] Processing {pdb_file.name}...", end=" ")
            try:
                features, runtime = extract_pfc_features(
                    pdb_file,
                    variant=variant,
                    device=device,
                )
                all_features.append(features)
                all_runtimes.append(runtime)
                print(f"✓ ({runtime:.2f}s)")
            except Exception as e:
                print(f"✗ Error: {e}")
                failed += 1
                continue
        
        if len(all_features) == 0:
            print(f"  ERROR: No features extracted for variant {variant}")
            continue
        
        # Stack features
        X = np.stack(all_features, axis=0)
        y = labels[:len(all_features)]  # Match length
        
        print(f"\n  Extracted {len(all_features)} proteins")
        print(f"  Feature shape: {X.shape}")
        print(f"  Average runtime: {np.mean(all_runtimes):.2f}s ± {np.std(all_runtimes):.2f}s")
        
        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_seed, stratify=y
        )
        
        print(f"  Train: {len(X_train)}, Test: {len(X_test)}")
        
        # Train and evaluate
        print("  Training classifier...")
        auroc, metrics = train_and_evaluate(
            X_train, y_train, X_test, y_test, device=device
        )
        
        results[variant] = {
            "auroc": auroc,
            "runtime_mean": float(np.mean(all_runtimes)),
            "runtime_std": float(np.std(all_runtimes)),
            "n_proteins": len(all_features),
            "failed": failed,
            "metrics": metrics,
        }
        
        print(f"  AUROC: {auroc:.4f}")
    
    return results


def print_results_table(results: Dict):
    """Print results in a formatted table."""
    print("\n" + "="*80)
    print("ABLATION STUDY RESULTS")
    print("="*80)
    print(f"{'Variant':<25} {'AUROC':<10} {'Runtime (s)':<15} {'N':<5}")
    print("-"*80)
    
    # Sort by AUROC
    sorted_results = sorted(results.items(), key=lambda x: x[1]["auroc"], reverse=True)
    
    for variant, res in sorted_results:
        print(f"{variant:<25} {res['auroc']:<10.4f} {res['runtime_mean']:.2f}±{res['runtime_std']:.2f}  {res['n_proteins']:<5}")
    
    print("="*80)


def save_results(results: Dict, output_file: Path):
    """Save results to JSON file."""
    # Convert numpy types to native Python types for JSON
    json_results = {}
    for variant, res in results.items():
        json_results[variant] = {
            "auroc": float(res["auroc"]),
            "runtime_mean": float(res["runtime_mean"]),
            "runtime_std": float(res["runtime_std"]),
            "n_proteins": int(res["n_proteins"]),
            "failed": int(res["failed"]),
        }
    
    with open(output_file, 'w') as f:
        json.dump(json_results, f, indent=2)
    
    print(f"\nResults saved to: {output_file}")


def find_qualitative_examples(
    pdb_files: List[Path],
    labels: np.ndarray,
    device: str = "cpu",
) -> List[Dict]:
    """
    Find qualitative examples where Full PFC fixes failures.
    
    Compares Full PFC vs Uniform Balls to find cases where:
    - Full PFC correctly predicts binding pocket
    - Uniform Balls fails (misses pocket)
    
    Returns:
        List of example dictionaries with protein info and predictions
    """
    examples = []
    
    for pdb_file, label in zip(pdb_files, labels):
        if label != 1:  # Only check positive examples (has pocket)
            continue
        
        try:
            # Extract features with both variants
            features_full, _ = extract_pfc_features(
                pdb_file, variant="full", device=device
            )
            features_uniform, _ = extract_pfc_features(
                pdb_file, variant="uniform", device=device
            )
            
            # Simple heuristic: Compare H2 features (cavities)
            # In practice, you'd use a trained model
            # Here we use a simple threshold on feature magnitude
            
            # H2 features are in the last 400 dimensions (20x20 grid)
            h2_full = features_full[-400:]
            h2_uniform = features_uniform[-400:]
            
            # Sum of H2 image (proxy for cavity strength)
            h2_sum_full = np.sum(h2_full)
            h2_sum_uniform = np.sum(h2_uniform)
            
            # Full PFC should have stronger H2 signal for binding pockets
            if h2_sum_full > h2_sum_uniform * 1.2:  # 20% stronger
                examples.append({
                    "protein": pdb_file.stem,
                    "label": int(label),
                    "h2_sum_full": float(h2_sum_full),
                    "h2_sum_uniform": float(h2_sum_uniform),
                    "improvement": float(h2_sum_full / h2_sum_uniform),
                })
        
        except Exception as e:
            continue
    
    # Sort by improvement
    examples.sort(key=lambda x: x["improvement"], reverse=True)
    
    return examples


def main():
    """Main function for ablation study."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Ablation study: Protein-aware flooding components"
    )
    parser.add_argument(
        "--pdb_dir",
        type=str,
        required=True,
        help="Directory containing PDB files",
    )
    parser.add_argument(
        "--labels_file",
        type=str,
        default=None,
        help="JSON file with labels: {\"protein_name\": 0 or 1, ...}",
    )
    parser.add_argument(
        "--max_proteins",
        type=int,
        default=None,
        help="Maximum number of proteins to process",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="Computation device",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="ablation_results.json",
        help="Output file for results",
    )
    parser.add_argument(
        "--variants",
        type=str,
        nargs="+",
        default=["full", "uniform", "no_sasa", "no_atom_radius", "no_chemistry", "generic_landmarks"],
        help="Variants to test",
    )
    
    args = parser.parse_args()
    
    # Find PDB files
    pdb_dir = Path(args.pdb_dir)
    pdb_files = list(pdb_dir.glob("*.pdb"))
    pdb_files.extend(list(pdb_dir.glob("*.ent")))
    
    if not pdb_files:
        print(f"ERROR: No PDB files found in {pdb_dir}")
        return
    
    if args.max_proteins is not None:
        pdb_files = pdb_files[:args.max_proteins]
    
    print(f"Found {len(pdb_files)} PDB files")
    
    # Load labels
    labels_file = Path(args.labels_file) if args.labels_file else None
    labels = load_protein_labels(pdb_files, labels_file)
    
    print(f"Labels: {np.sum(labels)} positive, {len(labels) - np.sum(labels)} negative")
    
    # Run ablation study
    results = run_ablation_study(
        pdb_files,
        labels,
        variants=args.variants,
        device=args.device,
    )
    
    # Print results
    print_results_table(results)
    
    # Save results
    save_results(results, Path(args.output_file))
    
    # Find best variant
    best_variant = max(results.items(), key=lambda x: x[1]["auroc"])
    print(f"\nBest variant: {best_variant[0]} (AUROC: {best_variant[1]['auroc']:.4f})")
    
    # Compare to full PFC
    if "full" in results:
        full_auroc = results["full"]["auroc"]
        print(f"\nFull PFC AUROC: {full_auroc:.4f}")
        print("\nImprovement over ablations:")
        for variant, res in results.items():
            if variant != "full":
                improvement = full_auroc - res["auroc"]
                print(f"  {variant}: {improvement:+.4f} ({improvement/full_auroc*100:+.1f}%)")
    
    # Find qualitative examples
    if "full" in results and "uniform" in results:
        print("\n" + "="*60)
        print("Finding qualitative examples...")
        print("="*60)
        examples = find_qualitative_examples(pdb_files, labels, device=args.device)
        
        if examples:
            print(f"\nFound {len(examples)} examples where Full PFC improves over Uniform:")
            print("\nTop 5 examples:")
            for i, ex in enumerate(examples[:5], 1):
                print(f"  {i}. {ex['protein']}:")
                print(f"     Full PFC H₂ sum: {ex['h2_sum_full']:.2f}")
                print(f"     Uniform H₂ sum: {ex['h2_sum_uniform']:.2f}")
                print(f"     Improvement: {ex['improvement']:.2f}x")
        else:
            print("No clear qualitative examples found (may need trained model)")


if __name__ == "__main__":
    main()

