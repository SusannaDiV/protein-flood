"""Task 1: Binding Pocket Classification (MANDATORY)

Dataset: scPDB or PDBBind
Baselines:
  - PointNet++
  - DGCNN
  - GVP-GNN
  - AlphaFold embeddings
  - Standard PH (Alpha complex)

Metrics: AUROC / accuracy
Expected result: Flood-PH beats PH baselines, competitive or better than neural models
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score, accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import json
import time

from flooder.protein_io import load_pdb_file
from flooder.pfc import protein_flood_complex
from flooder.persistence_vectorization import (
    compute_persistence_images_from_simplex_tree,
    get_feature_dimension,
)
from flooder.core import flood_complex, generate_landmarks


# ============================================================================
# Baseline Implementations
# ============================================================================

def extract_alphafold_embeddings(
    pdb_path: Path,
    embeddings_dir: Optional[Path] = None,
) -> Optional[np.ndarray]:
    """
    Extract AlphaFold embeddings if available.
    
    Args:
        pdb_path: Path to PDB file
        embeddings_dir: Directory containing AlphaFold embeddings
        
    Returns:
        Embedding vector or None if not available
    """
    if embeddings_dir is None:
        return None
    
    # Try to find AlphaFold embedding file
    protein_name = pdb_path.stem
    embedding_file = embeddings_dir / f"{protein_name}.npy"
    
    if embedding_file.exists():
        return np.load(embedding_file)
    
    # Try other common formats
    embedding_file = embeddings_dir / f"{protein_name}_embeddings.npy"
    if embedding_file.exists():
        return np.load(embedding_file)
    
    return None


def extract_alpha_complex_ph(
    pdb_path: Path,
    max_dimension: int = 2,
    device: str = "cpu",
) -> Tuple[np.ndarray, float]:
    """
    Extract persistence images from Alpha complex (standard PH baseline).
    
    Args:
        pdb_path: Path to PDB file
        max_dimension: Maximum homology dimension
        device: Computation device
        
    Returns:
        Tuple of (features, runtime)
    """
    start_time = time.time()
    
    try:
        import gudhi
        
        # Load protein
        protein = load_pdb_file(pdb_path)
        coords = protein.atom_coords.astype(np.float64)  # AlphaComplex needs float64
        
        # Build Alpha complex (same as example_01_cheese_3d.py)
        alpha_complex = gudhi.AlphaComplex(points=coords)
        stree = alpha_complex.create_simplex_tree(output_squared_values=False)
        
        # Compute persistence
        stree.compute_persistence()
        
        # Extract diagrams
        diagrams = []
        for dim in range(max_dimension + 1):
            diag = stree.persistence_intervals_in_dimension(dim)
            diagrams.append(diag)
        
        # Convert to persistence images
        from flooder.persistence_vectorization import compute_persistence_images
        features = compute_persistence_images(
            diagrams,
            bandwidth=1.0,
            resolution=(20, 20),
            normalize=True,
        )
        
        runtime = time.time() - start_time
        return features, runtime
    
    except Exception as e:
        print(f"  Error with Alpha complex: {e}")
        # Return zero features
        feature_dim = get_feature_dimension(num_diagrams=max_dimension + 1, resolution=(20, 20))
        return np.zeros(feature_dim, dtype=np.float32), 0.0


def extract_pointnet_features(
    pdb_path: Path,
    num_points: int = 1024,
    device: str = "cpu",
) -> Tuple[np.ndarray, float]:
    """
    Extract features using PointNet++ style (simplified - requires model).
    
    This is a placeholder - in practice you would:
    1. Load a pre-trained PointNet++ model
    2. Extract point cloud features
    3. Return global feature vector
    
    For now, returns a simple geometric feature vector.
    """
    start_time = time.time()
    
    try:
        protein = load_pdb_file(pdb_path)
        coords = protein.atom_coords
        
        # Simplified: Use geometric features as proxy
        # In practice, use actual PointNet++ model
        features = np.concatenate([
            coords.mean(axis=0),  # Centroid
            coords.std(axis=0),   # Spread
            coords.max(axis=0),   # Bounds
            coords.min(axis=0),   # Bounds
            [coords.shape[0]],    # Number of atoms
        ])
        
        runtime = time.time() - start_time
        return features.astype(np.float32), runtime
    
    except Exception as e:
        print(f"  Error with PointNet++: {e}")
        return np.zeros(13, dtype=np.float32), 0.0


def extract_dgcnn_features(
    pdb_path: Path,
    k: int = 20,
    device: str = "cpu",
) -> Tuple[np.ndarray, float]:
    """
    Extract features using DGCNN style (simplified - requires model).
    
    This is a placeholder - in practice you would:
    1. Build k-NN graph
    2. Use DGCNN model to extract features
    3. Return global feature vector
    
    For now, returns graph-based geometric features.
    """
    start_time = time.time()
    
    try:
        protein = load_pdb_file(pdb_path)
        coords = protein.atom_coords
        
        # Simplified: Use graph statistics as proxy
        # In practice, use actual DGCNN model
        from scipy.spatial.distance import cdist
        
        # Compute k-NN graph statistics
        distances = cdist(coords, coords)
        np.fill_diagonal(distances, np.inf)
        
        k_nearest = np.partition(distances, k, axis=1)[:, :k]
        mean_knn_dist = k_nearest.mean(axis=1).mean()
        std_knn_dist = k_nearest.std(axis=1).mean()
        
        features = np.array([
            coords.mean(axis=0).mean(),  # Mean position
            coords.std(axis=0).mean(),    # Spread
            mean_knn_dist,                # Mean k-NN distance
            std_knn_dist,                 # Std k-NN distance
            coords.shape[0],              # Number of atoms
        ])
        
        runtime = time.time() - start_time
        return features.astype(np.float32), runtime
    
    except Exception as e:
        print(f"  Error with DGCNN: {e}")
        return np.zeros(5, dtype=np.float32), 0.0


def extract_gvp_gnn_features(
    pdb_path: Path,
    device: str = "cpu",
) -> Tuple[np.ndarray, float]:
    """
    Extract features using GVP-GNN style (simplified - requires model).
    
    This is a placeholder - in practice you would:
    1. Use GVP-GNN library (https://github.com/drorlab/gvp-pytorch)
    2. Extract node and edge features
    3. Return graph-level embedding
    
    For now, returns residue-based features.
    """
    start_time = time.time()
    
    try:
        protein = load_pdb_file(pdb_path)
        
        # Simplified: Use residue-level statistics
        # In practice, use actual GVP-GNN model
        residue_features = []
        
        for res_idx in range(protein.num_residues):
            atom_indices = protein.residue_to_atoms.get(res_idx, [])
            if atom_indices:
                res_coords = protein.atom_coords[atom_indices]
                residue_features.append([
                    res_coords.mean(axis=0).mean(),  # Mean position
                    res_coords.std(axis=0).mean(),   # Spread
                    len(atom_indices),               # Number of atoms
                ])
        
        if residue_features:
            features = np.array(residue_features).mean(axis=0)  # Average over residues
        else:
            features = np.zeros(3, dtype=np.float32)
        
        runtime = time.time() - start_time
        return features.astype(np.float32), runtime
    
    except Exception as e:
        print(f"  Error with GVP-GNN: {e}")
        return np.zeros(3, dtype=np.float32), 0.0


# ============================================================================
# PFC Feature Extraction
# ============================================================================

def extract_pfc_features(
    pdb_path: Path,
    target_landmarks: int = 500,
    device: str = "cpu",
    use_pls: bool = False,
) -> Tuple[np.ndarray, float]:
    """
    Extract features using Protein Flood Complex (PFC).
    
    Returns:
        Tuple of (features, runtime)
    """
    start_time = time.time()
    
    protein = load_pdb_file(pdb_path)
    
    # Construct PFC
    pfc_stree = protein_flood_complex(
        protein,
        target_landmarks=target_landmarks,
        max_dimension=2,
        device=device,
        return_simplex_tree=True,
        use_pls=use_pls,
    )
    
    # Compute persistence
    pfc_stree.compute_persistence()
    
    # Convert to persistence images
    features = compute_persistence_images_from_simplex_tree(
        pfc_stree,
        max_dimension=2,
        bandwidth=1.0,
        resolution=(20, 20),
        normalize=True,
    )
    
    runtime = time.time() - start_time
    return features, runtime


# ============================================================================
# Classification Models
# ============================================================================

class SimpleMLP(nn.Module):
    """Simple MLP for binary classification."""
    
    def __init__(self, input_dim: int, hidden_dims: List[int] = [128, 64], dropout: float = 0.2):
        super().__init__()
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, 1))  # Binary classification
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x).squeeze(-1)


def train_classifier(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    input_dim: int,
    device: str = "cpu",
    epochs: int = 100,
    lr: float = 0.001,
    patience: int = 10,
) -> Tuple[nn.Module, StandardScaler, Dict]:
    """
    Train MLP classifier with early stopping.
    
    Returns:
        Tuple of (trained_model, scaler, metrics_dict)
    """
    # Normalize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    # Convert to tensors
    X_train_t = torch.from_numpy(X_train_scaled).float()
    y_train_t = torch.from_numpy(y_train).float()
    X_val_t = torch.from_numpy(X_val_scaled).float()
    y_val_t = torch.from_numpy(y_val).float()
    
    if device == "cuda" and torch.cuda.is_available():
        X_train_t = X_train_t.cuda()
        y_train_t = y_train_t.cuda()
        X_val_t = X_val_t.cuda()
        y_val_t = y_val_t.cuda()
    
    # Create model
    model = SimpleMLP(input_dim=input_dim)
    if device == "cuda" and torch.cuda.is_available():
        model = model.cuda()
    
    # Training setup
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()
    
    best_val_auroc = 0.0
    best_model_state = None
    patience_counter = 0
    
    for epoch in range(epochs):
        # Training
        model.train()
        optimizer.zero_grad()
        train_outputs = model(X_train_t)
        train_loss = criterion(train_outputs, y_train_t)
        train_loss.backward()
        optimizer.step()
        
        # Validation
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val_t)
            val_probs = torch.sigmoid(val_outputs).cpu().numpy()
            val_auroc = roc_auc_score(y_val, val_probs)
        
        # Early stopping
        if val_auroc > best_val_auroc:
            best_val_auroc = val_auroc
            best_model_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break
    
    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    # Final evaluation
    model.eval()
    with torch.no_grad():
        val_outputs = model(X_val_t)
        val_probs = torch.sigmoid(val_outputs).cpu().numpy()
        val_preds = (val_probs > 0.5).astype(int)
    
    val_auroc = roc_auc_score(y_val, val_probs)
    val_acc = accuracy_score(y_val, val_preds)
    
    metrics = {
        "best_val_auroc": float(best_val_auroc),
        "final_val_auroc": float(val_auroc),
        "final_val_acc": float(val_acc),
        "epochs_trained": epoch + 1,
    }
    
    return model, scaler, metrics


def evaluate_classifier(
    model: nn.Module,
    X_test: np.ndarray,
    y_test: np.ndarray,
    scaler: StandardScaler,
    device: str = "cpu",
) -> Dict:
    """
    Evaluate trained classifier on test set.
    
    Returns:
        Dictionary with metrics
    """
    # Normalize
    X_test_scaled = scaler.transform(X_test)
    X_test_t = torch.from_numpy(X_test_scaled).float()
    y_test_t = torch.from_numpy(y_test).float()
    
    if device == "cuda" and torch.cuda.is_available():
        X_test_t = X_test_t.cuda()
        y_test_t = y_test_t.cuda()
    
    # Evaluate
    model.eval()
    with torch.no_grad():
        outputs = model(X_test_t)
        probs = torch.sigmoid(outputs).cpu().numpy()
        preds = (probs > 0.5).astype(int)
    
    auroc = roc_auc_score(y_test, probs)
    acc = accuracy_score(y_test, preds)
    
    return {
        "auroc": float(auroc),
        "accuracy": float(acc),
        "predictions": preds.tolist(),
        "probabilities": probs.tolist(),
    }


# ============================================================================
# Main Experiment
# ============================================================================

def run_binding_pocket_classification(
    pdb_files: List[Path],
    labels: np.ndarray,
    methods: List[str],
    device: str = "cpu",
    test_size: float = 0.2,
    val_size: float = 0.1,
    random_seed: int = 42,
    alphafold_dir: Optional[Path] = None,
) -> Dict:
    """
    Run binding pocket classification experiment comparing multiple methods.
    
    Args:
        pdb_files: List of PDB file paths
        labels: Array of labels (0 = no pocket, 1 = has pocket)
        methods: List of methods to test:
            - "pfc": Protein Flood Complex
            - "pfc_pls": PFC with full PLS
            - "alpha_complex": Standard Alpha complex PH
            - "alphafold": AlphaFold embeddings
            - "pointnet": PointNet++ (simplified)
            - "dgcnn": DGCNN (simplified)
            - "gvp_gnn": GVP-GNN (simplified)
        device: Computation device
        test_size: Fraction for test set
        val_size: Fraction for validation set (from train)
        random_seed: Random seed
        alphafold_dir: Directory with AlphaFold embeddings
        
    Returns:
        Dictionary with results for each method
    """
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    
    results = {}
    
    for method in methods:
        print(f"\n{'='*80}")
        print(f"Method: {method.upper()}")
        print(f"{'='*80}")
        
        # Extract features
        all_features = []
        all_runtimes = []
        failed = 0
        
        for i, pdb_file in enumerate(pdb_files):
            print(f"  [{i+1}/{len(pdb_files)}] {pdb_file.name}...", end=" ", flush=True)
            
            try:
                if method == "pfc":
                    features, runtime = extract_pfc_features(
                        pdb_file, device=device, use_pls=False
                    )
                elif method == "pfc_pls":
                    features, runtime = extract_pfc_features(
                        pdb_file, device=device, use_pls=True
                    )
                elif method == "alpha_complex":
                    features, runtime = extract_alpha_complex_ph(
                        pdb_file, device=device
                    )
                elif method == "alphafold":
                    features = extract_alphafold_embeddings(pdb_file, alphafold_dir)
                    if features is None:
                        print("✗ (no embeddings)", end="")
                        failed += 1
                        continue
                    runtime = 0.0  # Embeddings are pre-computed
                elif method == "pointnet":
                    features, runtime = extract_pointnet_features(
                        pdb_file, device=device
                    )
                elif method == "dgcnn":
                    features, runtime = extract_dgcnn_features(
                        pdb_file, device=device
                    )
                elif method == "gvp_gnn":
                    features, runtime = extract_gvp_gnn_features(
                        pdb_file, device=device
                    )
                else:
                    print(f"✗ Unknown method: {method}")
                    failed += 1
                    continue
                
                all_features.append(features)
                all_runtimes.append(runtime)
                print(f"✓ ({runtime:.2f}s)")
            
            except Exception as e:
                print(f"✗ Error: {e}")
                failed += 1
                continue
        
        if len(all_features) == 0:
            print(f"  ERROR: No features extracted for method {method}")
            continue
        
        # Stack features
        X = np.stack(all_features, axis=0)
        y = labels[:len(all_features)]
        
        print(f"\n  Extracted {len(all_features)} proteins")
        print(f"  Feature shape: {X.shape}")
        print(f"  Average runtime: {np.mean(all_runtimes):.2f}s ± {np.std(all_runtimes):.2f}s")
        
        # Train/val/test split
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_seed, stratify=y
        )
        
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size/(1-test_size), random_state=random_seed, stratify=y_temp
        )
        
        print(f"  Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
        
        # Train classifier
        print("  Training classifier...")
        model, scaler, train_metrics = train_classifier(
            X_train, y_train, X_val, y_val,
            input_dim=X.shape[1],
            device=device,
        )
        
        # Evaluate on test set
        print("  Evaluating on test set...")
        test_metrics = evaluate_classifier(model, X_test, y_test, scaler, device=device)
        
        results[method] = {
            "test_auroc": test_metrics["auroc"],
            "test_accuracy": test_metrics["accuracy"],
            "val_auroc": train_metrics["final_val_auroc"],
            "val_accuracy": train_metrics["final_val_acc"],
            "runtime_mean": float(np.mean(all_runtimes)),
            "runtime_std": float(np.std(all_runtimes)),
            "n_proteins": len(all_features),
            "failed": failed,
            "feature_dim": X.shape[1],
        }
        
        print(f"  Test AUROC: {test_metrics['auroc']:.4f}")
        print(f"  Test Accuracy: {test_metrics['accuracy']:.4f}")
    
    return results


def print_results_table(results: Dict):
    """Print results in a formatted table."""
    print("\n" + "="*100)
    print("BINDING POCKET CLASSIFICATION RESULTS")
    print("="*100)
    print(f"{'Method':<20} {'Test AUROC':<12} {'Test Acc':<12} {'Runtime (s)':<15} {'N':<5} {'Dim':<6}")
    print("-"*100)
    
    # Sort by Test AUROC
    sorted_results = sorted(results.items(), key=lambda x: x[1]["test_auroc"], reverse=True)
    
    for method, res in sorted_results:
        print(f"{method:<20} {res['test_auroc']:<12.4f} {res['test_accuracy']:<12.4f} "
              f"{res['runtime_mean']:.2f}±{res['runtime_std']:.2f}  {res['n_proteins']:<5} {res['feature_dim']:<6}")
    
    print("="*100)
    
    # Compare to baselines
    if "pfc" in results:
        pfc_auroc = results["pfc"]["test_auroc"]
        print(f"\nPFC (Flood-PH) Test AUROC: {pfc_auroc:.4f}")
        print("\nComparison to baselines:")
        
        if "alpha_complex" in results:
            alpha_auroc = results["alpha_complex"]["test_auroc"]
            improvement = pfc_auroc - alpha_auroc
            print(f"  vs Alpha Complex PH: {improvement:+.4f} ({improvement/alpha_auroc*100:+.1f}%)")
        
        if "alphafold" in results:
            af_auroc = results["alphafold"]["test_auroc"]
            improvement = pfc_auroc - af_auroc
            print(f"  vs AlphaFold: {improvement:+.4f} ({improvement/af_auroc*100:+.1f}%)")
        
        if "pointnet" in results:
            pn_auroc = results["pointnet"]["test_auroc"]
            improvement = pfc_auroc - pn_auroc
            print(f"  vs PointNet++: {improvement:+.4f} ({improvement/pn_auroc*100:+.1f}%)")
        
        if "dgcnn" in results:
            dgcnn_auroc = results["dgcnn"]["test_auroc"]
            improvement = pfc_auroc - dgcnn_auroc
            print(f"  vs DGCNN: {improvement:+.4f} ({improvement/dgcnn_auroc*100:+.1f}%)")
        
        if "gvp_gnn" in results:
            gvp_auroc = results["gvp_gnn"]["test_auroc"]
            improvement = pfc_auroc - gvp_auroc
            print(f"  vs GVP-GNN: {improvement:+.4f} ({improvement/gvp_auroc*100:+.1f}%)")


def save_results(results: Dict, output_file: Path):
    """Save results to JSON file."""
    json_results = {}
    for method, res in results.items():
        json_results[method] = {
            "test_auroc": float(res["test_auroc"]),
            "test_accuracy": float(res["test_accuracy"]),
            "val_auroc": float(res["val_auroc"]),
            "val_accuracy": float(res["val_accuracy"]),
            "runtime_mean": float(res["runtime_mean"]),
            "runtime_std": float(res["runtime_std"]),
            "n_proteins": int(res["n_proteins"]),
            "failed": int(res["failed"]),
            "feature_dim": int(res["feature_dim"]),
        }
    
    with open(output_file, 'w') as f:
        json.dump(json_results, f, indent=2)
    
    print(f"\nResults saved to: {output_file}")


def load_protein_labels(
    pdb_files: List[Path],
    labels_file: Optional[Path] = None,
) -> np.ndarray:
    """Load labels for proteins."""
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
        print("WARNING: Using dummy labels. Provide --labels_file for real labels.")
        return np.random.randint(0, 2, size=len(pdb_files), dtype=np.int32)


def main():
    """Main function for binding pocket classification."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Task 1: Binding Pocket Classification"
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
        default="classification_results.json",
        help="Output file for results",
    )
    parser.add_argument(
        "--methods",
        type=str,
        nargs="+",
        default=["pfc", "alpha_complex"],
        help="Methods to test: pfc, pfc_pls, alpha_complex, alphafold, pointnet, dgcnn, gvp_gnn",
    )
    parser.add_argument(
        "--alphafold_dir",
        type=str,
        default=None,
        help="Directory containing AlphaFold embeddings (.npy files)",
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
    
    # Run experiment
    alphafold_dir = Path(args.alphafold_dir) if args.alphafold_dir else None
    results = run_binding_pocket_classification(
        pdb_files,
        labels,
        methods=args.methods,
        device=args.device,
        alphafold_dir=alphafold_dir,
    )
    
    # Print results
    print_results_table(results)
    
    # Save results
    save_results(results, Path(args.output_file))
    
    # Summary
    if "pfc" in results:
        pfc_auroc = results["pfc"]["test_auroc"]
        print(f"\n{'='*80}")
        print(f"SUMMARY: PFC (Flood-PH) achieves {pfc_auroc:.4f} AUROC")
        
        if "alpha_complex" in results:
            alpha_auroc = results["alpha_complex"]["test_auroc"]
            if pfc_auroc > alpha_auroc:
                print(f"✅ PFC beats Alpha Complex PH baseline (+{pfc_auroc - alpha_auroc:.4f})")
            else:
                print(f"⚠️  PFC below Alpha Complex PH baseline ({pfc_auroc - alpha_auroc:.4f})")
        
        print(f"{'='*80}")


if __name__ == "__main__":
    main()

