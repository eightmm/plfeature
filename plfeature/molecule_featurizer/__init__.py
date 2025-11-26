"""
Molecule Featurizer Module

A comprehensive toolkit for extracting molecular features from SMILES and RDKit mol objects.

Modules:
    - molecule_feature: Main MoleculeFeaturizer class for molecular descriptors and graph features
    - graph_featurizer: MoleculeGraphFeaturizer for GNN-ready graph representations
    - constants: Physical and chemical constants for featurization
"""

from .molecule_feature import MoleculeFeaturizer as MoleculeFeaturizerCore
from .molecule_featurizer_efficient import MoleculeFeaturizer as EfficientMoleculeFeaturizer
from .graph_featurizer import MoleculeGraphFeaturizer
from .constants import (
    ATOM_TYPES,
    BOND_TYPES,
    HYBRIDIZATIONS,
    PERIODIC_TABLE,
    ELECTRONEGATIVITY,
)

__version__ = "0.1.0"
__author__ = "Jaemin Sim"

__all__ = [
    "MoleculeFeaturizer",  # Main API class
    "MoleculeFeaturizerCore",  # Core implementation
    "MoleculeGraphFeaturizer",  # Graph featurizer for GNNs
    "ATOM_TYPES",
    "BOND_TYPES",
    "HYBRIDIZATIONS",
    "PERIODIC_TABLE",
    "ELECTRONEGATIVITY",
]


# Use efficient implementation by default
MoleculeFeaturizer = EfficientMoleculeFeaturizer


class MoleculeFeaturizerWrapper:
    """
    High-level API for molecule feature extraction.

    This class provides a simple, unified interface for extracting
    features from small molecules (drugs/ligands).

    Examples:
        >>> # Basic usage with SMILES
        >>> from plfeature import MoleculeFeaturizer
        >>> featurizer = MoleculeFeaturizer()
        >>> features = featurizer.extract("CCO")

        >>> # With RDKit mol object
        >>> from rdkit import Chem
        >>> mol = Chem.MolFromSmiles("CCO")
        >>> features = featurizer.extract(mol)

        >>> # Extract specific feature types
        >>> features = featurizer.extract("CCO", feature_types=["descriptors", "fingerprints"])
    """

    def __init__(self, add_hs: bool = True, compute_3d: bool = False):
        """
        Initialize the MoleculeFeaturizer.

        Args:
            add_hs: Whether to add hydrogens to molecules
            compute_3d: Whether to compute 3D conformers (for 3D descriptors)
        """
        self.add_hs = add_hs
        self.compute_3d = compute_3d
        self._extractor = MoleculeFeaturizerCore()

    def extract(self, mol_or_smiles, feature_types=None, save_to=None):
        """
        Extract features from a molecule.

        Args:
            mol_or_smiles: RDKit mol object or SMILES string
            feature_types: List of feature types to extract
                          Options: ["descriptors", "fingerprints", "graph"]
                          Default: ["descriptors", "fingerprints"]
            save_to: Optional path to save the extracted features

        Returns:
            Dictionary containing requested features

        Raises:
            ValueError: If molecule cannot be parsed
        """
        from rdkit import Chem
        import torch

        # Parse molecule
        if isinstance(mol_or_smiles, str):
            mol = Chem.MolFromSmiles(mol_or_smiles)
            if mol is None:
                raise ValueError(f"Invalid SMILES: {mol_or_smiles}")
            input_type = "smiles"
            input_value = mol_or_smiles
        else:
            mol = mol_or_smiles
            input_type = "mol"
            input_value = Chem.MolToSmiles(mol) if mol else None

        # Default feature types
        if feature_types is None:
            feature_types = ["descriptors", "fingerprints"]

        features = {
            'metadata': {
                'input_type': input_type,
                'input': input_value,
                'hydrogens_added': self.add_hs
            }
        }

        # Extract requested features
        if "descriptors" in feature_types or "fingerprints" in feature_types:
            smiles = input_value if input_type == "smiles" else Chem.MolToSmiles(mol)
            mol_features = self._extractor.get_feature(smiles)

            if "descriptors" in feature_types:
                features["descriptors"] = mol_features["descriptor"]

            if "fingerprints" in feature_types:
                features["fingerprints"] = {
                    k: v for k, v in mol_features.items() if k != "descriptor"
                }

        if "graph" in feature_types:
            # Get graph format features
            smiles = input_value if input_type == "smiles" else Chem.MolToSmiles(mol)
            graph = self._extractor.get_graph(smiles)
            features["graph"] = graph

        # Save if requested
        if save_to:
            torch.save(features, save_to)

        return features

    def extract_batch(self, molecules, feature_types=None, output_dir=None,
                     skip_existing=True, verbose=True):
        """
        Extract features from multiple molecules.

        Args:
            molecules: List of SMILES strings or RDKit mol objects
            feature_types: Feature types to extract
            output_dir: Directory to save feature files (optional)
            skip_existing: Whether to skip molecules with existing features
            verbose: Whether to print progress

        Returns:
            Dictionary mapping molecule index to features or output paths
        """
        import os
        from pathlib import Path

        results = {}

        for i, mol in enumerate(molecules):
            # Determine output path if saving
            output_path = None
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
                mol_id = f"mol_{i}"
                output_path = os.path.join(output_dir, f"{mol_id}_features.pt")

                # Skip if exists
                if skip_existing and os.path.exists(output_path):
                    if verbose:
                        print(f"[{i+1}/{len(molecules)}] Skipping {mol_id} (already exists)")
                    results[i] = output_path
                    continue

            try:
                if verbose:
                    print(f"[{i+1}/{len(molecules)}] Processing molecule {i}...")

                features = self.extract(mol, feature_types=feature_types, save_to=output_path)
                results[i] = output_path if output_path else features

            except Exception as e:
                if verbose:
                    print(f"[{i+1}/{len(molecules)}] Failed molecule {i}: {str(e)}")
                results[i] = None

        return results

    @classmethod
    def from_smiles(cls, smiles: str) -> dict:
        """
        Quick feature extraction from SMILES string.

        Args:
            smiles: SMILES string

        Returns:
            Dictionary containing features
        """
        featurizer = cls()
        return featurizer.extract(smiles)

    @staticmethod
    def get_available_features():
        """
        Get list of available feature types.

        Returns:
            List of feature type names
        """
        return ["descriptors", "fingerprints", "graph"]