"""
Efficient Molecule Featurizer with caching.

This module provides an enhanced API for molecule feature extraction
with caching for repeated feature access, similar to ProteinFeaturizer.
"""

from typing import Union, Dict, Any, Tuple, Optional
import torch
from rdkit import Chem
from .molecule_feature import MoleculeFeaturizer as CoreFeaturizer


class MoleculeFeaturizer:
    """
    Enhanced molecule featurizer with efficient caching.

    Similar to ProteinFeaturizer, initialize with a molecule and then call methods
    to extract different features efficiently.

    Examples:
        >>> # From SMILES
        >>> featurizer = MoleculeFeaturizer("CCO")
        >>> features = featurizer.get_feature()
        >>> node, edge = featurizer.get_graph()
        >>>
        >>> # From RDKit mol
        >>> mol = Chem.MolFromSmiles("CCO")
        >>> featurizer = MoleculeFeaturizer(mol)
        >>>
        >>> # From SDF file
        >>> suppl = Chem.SDMolSupplier('molecules.sdf')
        >>> for mol in suppl:
        >>>     featurizer = MoleculeFeaturizer(mol)
        >>>     features = featurizer.get_feature()
    """

    def __init__(self, mol_or_smiles: Union[str, Chem.Mol], hydrogen: bool = True,
                 custom_smarts: Optional[Dict[str, str]] = None):
        """
        Initialize featurizer with a molecule.

        Args:
            mol_or_smiles: Molecule (RDKit mol or SMILES string)
            hydrogen: Whether to add hydrogens to molecules (default: True)
            custom_smarts: Optional dictionary of custom SMARTS patterns
                          e.g., {'aromatic_nitrogen': 'n', 'carboxyl': 'C(=O)O'}

        Raises:
            ValueError: If molecule cannot be parsed
        """
        self._core = CoreFeaturizer()
        self.hydrogen = hydrogen
        self.custom_smarts = custom_smarts or {}
        self._cache = {}

        # Store input for reference
        if isinstance(mol_or_smiles, str):
            self.input_smiles = mol_or_smiles
            self.input_mol = Chem.MolFromSmiles(mol_or_smiles)
            if self.input_mol is None:
                raise ValueError(f"Invalid SMILES: {mol_or_smiles}")
        else:
            self.input_mol = mol_or_smiles
            self.input_smiles = Chem.MolToSmiles(mol_or_smiles) if mol_or_smiles else None

        # Prepare molecule (add hydrogens if requested)
        self._mol = self._core._prepare_mol(self.input_mol, self.hydrogen)
        if self._mol is None:
            raise ValueError(f"Failed to prepare molecule: {mol_or_smiles}")

        self._parse_molecule()

    def _parse_molecule(self):
        """Parse and cache basic molecular data."""
        # Cache basic info
        self.num_atoms = self._mol.GetNumAtoms()
        self.num_bonds = self._mol.GetNumBonds()
        self.num_rings = self._mol.GetRingInfo().NumRings()

        # Check 3D status
        self.has_3d = self._mol.GetNumConformers() > 0

    def get_feature(self) -> Dict[str, torch.Tensor]:
        """
        Get all molecular features (descriptors and fingerprints).

        Returns:
            Dictionary containing:
            - descriptor: 40 molecular descriptors
            - morgan, maccs, rdkit, etc.: 9 fingerprint types
        """
        if 'features' not in self._cache:
            # Pass the molecule without additional hydrogen processing since it's already prepared
            self._cache['features'] = self._core.get_feature(self._mol, add_hs=False)
        return self._cache['features']

    def get_descriptors(self) -> torch.Tensor:
        """
        Get only molecular descriptors.

        Returns:
            torch.Tensor: 40 normalized molecular descriptors
        """
        features = self.get_feature()
        return features['descriptor']

    def get_fingerprints(self) -> Dict[str, torch.Tensor]:
        """
        Get only molecular fingerprints.

        Returns:
            Dictionary of 9 fingerprint types
        """
        features = self.get_feature()
        return {k: v for k, v in features.items() if k != 'descriptor'}

    def _get_custom_smarts_features(self) -> Optional[torch.Tensor]:
        """
        Compute custom SMARTS pattern matches for each atom.

        Returns:
            torch.Tensor or None: Binary features [n_atoms, n_patterns] if patterns provided
        """
        if not self.custom_smarts:
            return None

        cache_key = 'custom_smarts_features'
        if cache_key in self._cache:
            return self._cache[cache_key]

        num_atoms = self._mol.GetNumAtoms()
        num_patterns = len(self.custom_smarts)
        features = torch.zeros(num_atoms, num_patterns)

        for idx, (name, pattern) in enumerate(self.custom_smarts.items()):
            try:
                # Parse SMARTS pattern
                smart_mol = Chem.MolFromSmarts(pattern)
                if smart_mol is None:
                    print(f"Warning: Invalid SMARTS pattern for '{name}': {pattern}")
                    continue

                # Find matches
                matches = self._mol.GetSubstructMatches(smart_mol)

                # Mark matched atoms
                for match in matches:
                    for atom_idx in match:
                        if atom_idx < num_atoms:
                            features[atom_idx, idx] = 1.0

            except Exception as e:
                print(f"Warning: Error processing SMARTS pattern '{name}': {e}")

        self._cache[cache_key] = features
        return features

    def get_graph(self, distance_cutoff: Optional[float] = None,
                  include_custom_smarts: bool = True) -> Tuple[Dict, Dict, torch.Tensor]:
        """
        Get graph representation with node and edge features.

        Args:
            distance_cutoff: Optional distance cutoff for edges (if 3D available)
                           If None, uses bond connectivity
            include_custom_smarts: Whether to include custom SMARTS features in node features

        Returns:
            Tuple of (node, edge, adj):
            - node: {'node_feats', 'coords', 'custom_smarts_feats' (if available)}
            - edge: {'edges', 'edge_feats'}
            - adj: adjacency matrix with bond features [num_atoms, num_atoms, num_bond_features]
        """
        cache_key = f'graph_{distance_cutoff}_{include_custom_smarts}'

        if cache_key not in self._cache:
            # Get basic graph structure
            # Pass the molecule without additional hydrogen processing since it's already prepared
            node, edge, adj = self._core.get_graph(self._mol, add_hs=False)

            # Add custom SMARTS features if requested
            if include_custom_smarts and self.custom_smarts:
                custom_features = self._get_custom_smarts_features()
                if custom_features is not None:
                    # Concatenate custom features to node features
                    original_feats = node['node_feats']
                    node['node_feats'] = torch.cat([original_feats, custom_features], dim=1)

            # If distance cutoff specified and 3D coords available, filter edges
            if distance_cutoff is not None and self.has_3d and 'coords' in node:
                import numpy as np
                from scipy.spatial import distance_matrix

                coords = node['coords'].numpy()
                dist_matrix = distance_matrix(coords, coords)

                # Create edges based on distance cutoff
                edges_array = np.where((dist_matrix < distance_cutoff) & (dist_matrix > 0))

                # Update edge information
                edge['edges'] = torch.tensor([edges_array[0], edges_array[1]])
                edge['distance_cutoff'] = distance_cutoff

            self._cache[cache_key] = (node, edge, adj)

        return self._cache[cache_key]

    def get_morgan_fingerprint(self, radius: int = 2, n_bits: int = 2048) -> torch.Tensor:
        """
        Get Morgan fingerprint with custom parameters.

        Args:
            radius: Radius for Morgan fingerprint
            n_bits: Number of bits

        Returns:
            torch.Tensor: Morgan fingerprint
        """
        cache_key = f'morgan_{radius}_{n_bits}'

        if cache_key not in self._cache:
            # For now, return the default Morgan from get_feature
            features = self.get_feature()
            self._cache[cache_key] = features['morgan']

        return self._cache[cache_key]

    def get_custom_smarts_features(self) -> Optional[Dict[str, Any]]:
        """
        Get custom SMARTS pattern matches for each atom.

        Returns:
            Dictionary with 'features' tensor and 'names' list, or None if no patterns
        """
        if not self.custom_smarts:
            return None

        features = self._get_custom_smarts_features()
        if features is None:
            return None

        return {
            'features': features,  # [n_atoms, n_patterns] binary tensor
            'names': list(self.custom_smarts.keys()),
            'patterns': self.custom_smarts
        }

    def get_3d_coordinates(self) -> Optional[torch.Tensor]:
        """
        Get 3D coordinates if available.

        Returns:
            torch.Tensor or None: 3D coordinates [n_atoms, 3]
        """
        if not self.has_3d:
            return None

        node, _, _ = self.get_graph()
        return node.get('coords', None)

    def get_all_features(self, save_to: Optional[str] = None) -> Dict[str, Any]:
        """
        Get all features at once.

        Args:
            save_to: Optional path to save features

        Returns:
            Dictionary containing all features and metadata
        """
        features = self.get_feature()
        node, edge, adj = self.get_graph()

        all_features = {
            'descriptors': features['descriptor'],
            'fingerprints': {k: v for k, v in features.items() if k != 'descriptor'},
            'graph': {'node': node, 'edge': edge, 'adj': adj},
            'metadata': {
                'input_smiles': self.input_smiles,
                'num_atoms': self.num_atoms,
                'num_bonds': self.num_bonds,
                'num_rings': self.num_rings,
                'has_3d': self.has_3d,
                'hydrogens_added': self.hydrogen
            }
        }

        if save_to:
            torch.save(all_features, save_to)

        return all_features

    # Aliases for consistency
    extract = get_all_features
    get_features = get_feature  # Plural alias

    def __repr__(self):
        """String representation."""
        return (f"MoleculeFeaturizer(smiles='{self.input_smiles}', "
                f"atoms={self.num_atoms}, bonds={self.num_bonds})")