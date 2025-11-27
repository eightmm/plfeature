"""
Molecule Featurizer Module

A comprehensive toolkit for extracting molecular features from SMILES and RDKit mol objects.

Modules:
    - molecule_feature: Main MoleculeFeaturizer class for molecular descriptors and graph features
    - graph_featurizer: MoleculeGraphFeaturizer for GNN-ready graph representations
    - constants: Physical and chemical constants for featurization
"""

from .molecule_feature import MoleculeFeaturizer
from .graph_featurizer import MoleculeGraphFeaturizer
from ..constants import (
    ATOM_TYPES,
    BOND_TYPES,
    HYBRIDIZATION_TYPES as HYBRIDIZATIONS,
    PERIODIC_TABLE,
    ELECTRONEGATIVITY,
)

__version__ = "0.1.0"
__author__ = "Jaemin Sim"

__all__ = [
    "MoleculeFeaturizer",
    "MoleculeGraphFeaturizer",
    "ATOM_TYPES",
    "BOND_TYPES",
    "HYBRIDIZATIONS",
    "PERIODIC_TABLE",
    "ELECTRONEGATIVITY",
]
