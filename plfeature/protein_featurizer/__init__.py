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
    "PDBStandardizer",
    "ResidueFeaturizer",
    "AtomFeaturizer",
    "HierarchicalFeaturizer",
    "HierarchicalProteinData",
    "standardize_pdb",
    "get_protein_atom_features",
    "get_atom_features_with_sasa",
    "extract_hierarchical_features",
    "ProteinFeaturizer",
    "get_esm_featurizer",
    "get_dual_esm_featurizer",
]
