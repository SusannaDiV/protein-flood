from .io import save_to_disk
from .core import flood_complex, generate_landmarks
from .synthetic_data_generators import (
    generate_swiss_cheese_points,
    generate_annulus_points_2d,
    generate_noisy_torus_points_3d,
    generate_figure_eight_points_2d,
)

# Protein Flood Complex functionality
try:
    from .protein_io import (
        load_pdb_file,
        ProteinStructure,
        get_atom_vdw_radius,
        get_residue_hydrophobicity,
        is_charged_residue,
        compute_simple_sasa,
    )
    from .protein_landmarks import (
        select_residue_landmarks,
        get_landmark_attributes,
    )
    from .protein_landmark_sampling import (
        protein_landmark_sampling,
        voxel_grid_downsample,
        compute_pocket_score,
        compute_curvature_score,
        compute_importance_scores,
    )
    from .pfc import (
        protein_flood_complex,
        compute_landmark_weights,
    )
    from .persistence_vectorization import (
        compute_persistence_images,
        compute_persistence_images_from_simplex_tree,
        compute_persistence_images_batch,
        get_feature_dimension,
    )
    PROTEIN_FEATURES_AVAILABLE = True
except ImportError:
    PROTEIN_FEATURES_AVAILABLE = False

__version__ = "1.0rc6"

__all__ = [
    "flood_complex",
    "generate_landmarks",
    "save_to_disk",
    "generate_swiss_cheese_points",
    "generate_annulus_points_2d",
    "generate_noisy_torus_points_3d",
    "generate_figure_eight_points_2d",
]

if PROTEIN_FEATURES_AVAILABLE:
    __all__.extend([
        "load_pdb_file",
        "ProteinStructure",
        "select_residue_landmarks",
        "get_landmark_attributes",
        "protein_landmark_sampling",
        "voxel_grid_downsample",
        "compute_pocket_score",
        "compute_curvature_score",
        "compute_importance_scores",
        "protein_flood_complex",
        "compute_landmark_weights",
        "get_atom_vdw_radius",
        "get_residue_hydrophobicity",
        "is_charged_residue",
        "compute_simple_sasa",
        "compute_persistence_images",
        "compute_persistence_images_from_simplex_tree",
        "compute_persistence_images_batch",
        "get_feature_dimension",
    ])
