"""
Protein Featurizer Module

A comprehensive toolkit for extracting structural features from protein PDB files.
"""

from .pdb_standardizer import PDBStandardizer, standardize_pdb
from .residue_featurizer import ResidueFeaturizer
from .protein_featurizer import ProteinFeaturizer
from .atom_featurizer import AtomFeaturizer, get_protein_atom_features, get_atom_features_with_sasa
from .hierarchical_featurizer import (
    HierarchicalFeaturizer,
    HierarchicalProteinData,
    extract_hierarchical_features,
)
from .pdb_utils import (
    PDBParser,
    ParsedAtom,
    ParsedResidue,
    is_atom_record,
    is_hetatm_record,
    is_hydrogen,
    parse_pdb_line,
    normalize_residue_name,
    calculate_sidechain_centroid,
)

# ESM featurizer (lazy import to avoid esm dependency)
def get_esm_featurizer():
    """Get ESMFeaturizer class (lazy import)."""
    from .esm_featurizer import ESMFeaturizer
    return ESMFeaturizer

def get_dual_esm_featurizer():
    """Get DualESMFeaturizer class (lazy import)."""
    from .esm_featurizer import DualESMFeaturizer
    return DualESMFeaturizer

__version__ = "0.1.0"
__author__ = "Jaemin Sim"

__all__ = [
    # Featurizers
    "PDBStandardizer",
    "ResidueFeaturizer",
    "AtomFeaturizer",
    "HierarchicalFeaturizer",
    "HierarchicalProteinData",
    "ProteinFeaturizer",
    # PDB utilities
    "PDBParser",
    "ParsedAtom",
    "ParsedResidue",
    "is_atom_record",
    "is_hetatm_record",
    "is_hydrogen",
    "parse_pdb_line",
    "normalize_residue_name",
    "calculate_sidechain_centroid",
    # Convenience functions
    "standardize_pdb",
    "get_protein_atom_features",
    "get_atom_features_with_sasa",
    "extract_hierarchical_features",
    "get_esm_featurizer",
    "get_dual_esm_featurizer",
]
