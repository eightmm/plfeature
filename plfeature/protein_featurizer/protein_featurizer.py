"""
Efficient Protein Featurizer with one-time parsing.

This module provides a high-level API for protein feature extraction
with efficient caching of parsed PDB data.
"""

import os
import tempfile
from typing import Optional, Dict, Any, Tuple
import torch
import numpy as np

from .pdb_standardizer import PDBStandardizer
from .residue_featurizer import ResidueFeaturizer


class ProteinFeaturizer:
    """
    Efficient protein featurizer that parses PDB once and caches results.

    Examples:
        >>> # Parse once, extract multiple features efficiently
        >>> featurizer = ProteinFeaturizer("protein.pdb")
        >>> sequence = featurizer.get_sequence_features()
        >>> geometry = featurizer.get_geometric_features()
        >>> sasa = featurizer.get_sasa_features()
    """

    def __init__(self, pdb_file: str, standardize: bool = True,
                 keep_hydrogens: bool = False):
        """
        Initialize and parse PDB file once.

        Args:
            pdb_file: Path to PDB file
            standardize: Whether to standardize the PDB first
            keep_hydrogens: Whether to keep hydrogens during standardization
        """
        self.input_file = pdb_file
        self.standardize = standardize
        self.keep_hydrogens = keep_hydrogens

        # Check if file exists
        if not os.path.exists(pdb_file):
            raise FileNotFoundError(f"PDB file not found: {pdb_file}")

        # Standardize if requested
        if standardize:
            with tempfile.NamedTemporaryFile(suffix='.pdb', delete=False) as tmp_file:
                self.tmp_pdb = tmp_file.name

            standardizer = PDBStandardizer(remove_hydrogens=not keep_hydrogens)
            standardizer.standardize(pdb_file, self.tmp_pdb)
            pdb_to_process = self.tmp_pdb
        else:
            self.tmp_pdb = None
            pdb_to_process = pdb_file

        # Parse PDB once
        self._featurizer = ResidueFeaturizer(pdb_to_process)
        self._parse_structure()

        # Cache for computed features
        self._cache = {}

    def _parse_structure(self):
        """Parse structure and cache basic data."""
        # Get residues
        self.residues = self._featurizer.get_residues()
        self.num_residues = len(self.residues)

        # Build coordinate tensor
        self.coords = torch.zeros(self.num_residues, 15, 3)
        self.residue_types = torch.from_numpy(
            np.array(self.residues)[:, 2].astype(int)
        )

        for idx, residue in enumerate(self.residues):
            residue_coord_series = self._featurizer.get_residue_coordinates(residue)

            # For unknown residues (type 20), only use backbone + CB atoms
            res_type = residue[2]
            if res_type == 20:  # UNK residue
                # Standard backbone atoms + CB
                standard_unk_atoms = ['N', 'CA', 'C', 'O', 'CB']
                filtered_coords = []
                for atom_name in standard_unk_atoms:
                    if atom_name in residue_coord_series.index:
                        filtered_coords.append(residue_coord_series[atom_name])
                    else:
                        # Atom missing, use zero coordinates
                        filtered_coords.append([0.0, 0.0, 0.0])
                residue_coord = torch.as_tensor(filtered_coords)
            else:
                residue_coord = torch.as_tensor(residue_coord_series.tolist())

            self.coords[idx, :residue_coord.shape[0], :] = residue_coord
            # Sidechain centroid
            if residue_coord.shape[0] > 4:
                self.coords[idx, -1, :] = residue_coord[4:, :].mean(0)
            else:
                # No sidechain atoms, use CA position as fallback
                self.coords[idx, -1, :] = residue_coord[1, :] if residue_coord.shape[0] > 1 else torch.zeros(3)

        # Extract CA and SC coordinates
        self.coords_CA = self.coords[:, 1:2, :]
        self.coords_SC = self.coords[:, -1:, :]
        self.coord = torch.cat([self.coords_CA, self.coords_SC], dim=1)

    def __del__(self):
        """Clean up temporary files."""
        if hasattr(self, 'tmp_pdb') and self.tmp_pdb and os.path.exists(self.tmp_pdb):
            os.unlink(self.tmp_pdb)

    def get_sequence_features(self) -> Dict[str, Any]:
        """
        Get amino acid sequence and position features.

        Returns:
            Dictionary with residue types and one-hot encoding
        """
        if 'sequence' not in self._cache:
            residue_one_hot = torch.nn.functional.one_hot(
                self.residue_types, num_classes=21
            )

            self._cache['sequence'] = {
                'residue_types': self.residue_types,
                'residue_one_hot': residue_one_hot,
                'num_residues': self.num_residues
            }

        return self._cache['sequence']

    def get_geometric_features(self) -> Dict[str, Any]:
        """
        Get geometric features including distances, angles, and dihedrals.

        Returns:
            Dictionary with geometric measurements
        """
        if 'geometric' not in self._cache:
            # Get geometric features
            dihedrals, has_chi = self._featurizer.get_dihedral_angles(
                self.coords, self.residue_types
            )
            terminal_flags = self.get_terminal_flags()
            curvature = self._featurizer._calculate_backbone_curvature(
                self.coords, (terminal_flags['n_terminal'], terminal_flags['c_terminal'])
            )
            torsion = self._featurizer._calculate_backbone_torsion(
                self.coords, (terminal_flags['n_terminal'], terminal_flags['c_terminal'])
            )
            self_distance, self_vector = self._featurizer._calculate_self_distances_vectors(
                self.coords
            )

            self._cache['geometric'] = {
                'dihedrals': dihedrals,
                'has_chi_angles': has_chi,
                'backbone_curvature': curvature,
                'backbone_torsion': torsion,
                'self_distances': self_distance,
                'self_vectors': self_vector,
                'coordinates': self.coords
            }

        return self._cache['geometric']

    def get_sasa_features(self) -> torch.Tensor:
        """
        Get Solvent Accessible Surface Area features.

        Returns:
            SASA tensor with multiple components per residue
        """
        if 'sasa' not in self._cache:
            self._cache['sasa'] = self._featurizer.calculate_sasa()

        return self._cache['sasa']

    def get_contact_map(self, cutoff: float = 8.0) -> Dict[str, Any]:
        """
        Get residue-residue contact map and distances.

        Args:
            cutoff: Distance cutoff for contacts (default: 8.0 Å)

        Returns:
            Dictionary with contact information
        """
        cache_key = f'contact_map_{cutoff}'

        if cache_key not in self._cache:
            distance_adj, adj, vectors = self._featurizer._calculate_interaction_features(
                self.coords, cutoff=cutoff
            )

            # Get sparse representation
            sparse = distance_adj.to_sparse(sparse_dim=2)
            src, dst = sparse.indices()
            distances = sparse.values()

            self._cache[cache_key] = {
                'adjacency_matrix': adj,
                'distance_matrix': distance_adj,
                'edges': (src, dst),
                'edge_distances': distances,
                'interaction_vectors': vectors
            }

        return self._cache[cache_key]

    def get_relative_position(self, cutoff: int = 32) -> torch.Tensor:
        """
        Get relative position encoding between residues.

        Args:
            cutoff: Maximum relative position to consider

        Returns:
            One-hot encoded relative position tensor
        """
        cache_key = f'relative_position_{cutoff}'

        if cache_key not in self._cache:
            self._cache[cache_key] = self._featurizer.get_relative_position(
                cutoff=cutoff, onehot=True
            )

        return self._cache[cache_key]

    def get_node_features(self) -> Dict[str, Any]:
        """
        Get all node (residue-level) features.

        Returns:
            Dictionary with scalar and vector node features
        """
        if 'node_features' not in self._cache:
            scalar_features, vector_features = self._featurizer._extract_residue_features(
                self.coords, self.residue_types
            )

            self._cache['node_features'] = {
                'coordinates': self.coord,
                'scalar_features': scalar_features,
                'vector_features': vector_features
            }

        return self._cache['node_features']

    def get_edge_features(self, distance_cutoff: float = 8.0) -> Dict[str, Any]:
        """
        Get all edge (interaction) features.

        Args:
            distance_cutoff: Distance cutoff for interactions

        Returns:
            Dictionary with edge indices and features
        """
        cache_key = f'edge_features_{distance_cutoff}'

        if cache_key not in self._cache:
            edges, scalar_features, vector_features = \
                self._featurizer._extract_interaction_features(
                    self.coords, distance_cutoff=distance_cutoff
                )

            self._cache[cache_key] = {
                'edges': edges,
                'scalar_features': scalar_features,
                'vector_features': vector_features
            }

        return self._cache[cache_key]

    def get_terminal_flags(self) -> Dict[str, torch.Tensor]:
        """
        Get N-terminal and C-terminal residue flags.

        Returns:
            Dictionary with terminal flags
        """
        if 'terminal_flags' not in self._cache:
            n_terminal, c_terminal = self._featurizer.get_terminal_flags()
            self._cache['terminal_flags'] = {
                'n_terminal': n_terminal,
                'c_terminal': c_terminal
            }

        return self._cache['terminal_flags']

    def get_features(self, distance_cutoff: float = 8.0) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Get node and edge features in standard format.

        Args:
            distance_cutoff: Distance cutoff for residue-residue edges (default: 8.0 Å)

        Returns:
            Tuple of (node, edge) dictionaries with:
            - node: {'coord', 'node_scalar_features', 'node_vector_features'}
            - edge: {'edges', 'edge_scalar_features', 'edge_vector_features'}
        """
        cache_key = f'features_{distance_cutoff}'
        if cache_key not in self._cache:
            # Get edges with the specified cutoff
            edges, edge_scalar_features, edge_vector_features = \
                self._featurizer._extract_interaction_features(
                    self.coords, distance_cutoff=distance_cutoff
                )

            # Get node features
            node_scalar_features, node_vector_features = \
                self._featurizer._extract_residue_features(
                    self.coords, self.residue_types
                )

            node = {
                'coord': self.coord,
                'node_scalar_features': node_scalar_features,
                'node_vector_features': node_vector_features
            }

            edge = {
                'edges': edges,
                'edge_scalar_features': edge_scalar_features,
                'edge_vector_features': edge_vector_features
            }

            self._cache[cache_key] = (node, edge)
        return self._cache[cache_key]

    def get_all_features(self, save_to: Optional[str] = None) -> Dict[str, Any]:
        """
        Get all features at once.

        Args:
            save_to: Optional path to save features

        Returns:
            Dictionary containing all features
        """
        node_features = self.get_node_features()
        edge_features = self.get_edge_features()

        features = {
            'node': node_features,
            'edge': edge_features,
            'metadata': {
                'input_file': self.input_file,
                'standardized': self.standardize,
                'hydrogens_removed': not self.keep_hydrogens if self.standardize else None,
                'num_residues': self.num_residues
            }
        }

        if save_to:
            torch.save(features, save_to)

        return features

    # Alias for backward compatibility
    extract = get_all_features

    # ============== ATOM-LEVEL FEATURES ==============

    def get_atom_graph(self, distance_cutoff: float = 4.0) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Get atom-level graph representation with node and edge features.

        Args:
            distance_cutoff: Distance cutoff for atom-atom edges (default: 4.0 Å)

        Returns:
            Tuple of (node, edge) dictionaries:
                - node: {'coord', 'node_features', 'atom_tokens', 'sasa'}
                - edge: {'edges', 'edge_distances'}
        """
        cache_key = f'atom_graph_{distance_cutoff}'

        if cache_key not in self._cache:
            from .atom_featurizer import AtomFeaturizer
            import numpy as np
            from scipy.spatial import distance_matrix

            atom_featurizer = AtomFeaturizer()
            pdb_to_use = self.tmp_pdb if self.tmp_pdb else self.input_file

            # Get atom features with SASA
            atom_features = atom_featurizer.get_all_atom_features(pdb_to_use)

            # Build distance matrix
            coords = atom_features['coord'].numpy()
            dist_matrix = distance_matrix(coords, coords)

            # Create edges based on distance cutoff
            edges_array = np.where((dist_matrix < distance_cutoff) & (dist_matrix > 0))
            edges = (torch.tensor(edges_array[0]), torch.tensor(edges_array[1]))
            edge_distances = torch.tensor(dist_matrix[edges_array])

            # Package as node and edge dictionaries
            # Convert residue_numbers to tensor if it's not already
            residue_nums = atom_features['metadata']['residue_numbers']
            if isinstance(residue_nums, torch.Tensor):
                residue_number_tensor = residue_nums.clone()
            else:
                residue_number_tensor = torch.tensor(residue_nums, dtype=torch.long)

            # Create residue_count: sequential index starting from 0, incrementing when residue changes
            # Example: residue_number [1,1,1,5,5,7,7,7] -> residue_count [0,0,0,1,1,2,2,2]
            residue_count = torch.zeros_like(residue_number_tensor)
            if len(residue_number_tensor) > 0:
                current_count = 0
                residue_count[0] = current_count
                for i in range(1, len(residue_number_tensor)):
                    # Check if residue changed (consider both residue number and chain)
                    chain_labels = atom_features['metadata']['chain_labels']
                    residue_changed = (residue_number_tensor[i] != residue_number_tensor[i-1]) or \
                                    (chain_labels[i] != chain_labels[i-1])
                    if residue_changed:
                        current_count += 1
                    residue_count[i] = current_count

            node = {
                'coord': atom_features['coord'],
                'node_features': atom_features['token'],  # Token as main feature
                'atom_tokens': atom_features['token'],
                'sasa': atom_features['sasa'],
                'residue_token': atom_features['residue_token'],
                'atom_element': atom_features['atom_element'],
                'residue_number': residue_number_tensor,
                'residue_count': residue_count,
                'atom_name': atom_features['metadata']['atom_names'],
                'chain_label': atom_features['metadata']['chain_labels']
            }

            edge = {
                'edges': edges,
                'edge_distances': edge_distances,
                'distance_cutoff': distance_cutoff
            }

            self._cache[cache_key] = (node, edge)

        return self._cache[cache_key]

    # Alias for consistency
    get_atom_features = get_atom_graph
    get_atom_level_graph = get_atom_graph

    def get_atom_tokens_and_coords(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get atom-level tokenized features and coordinates.

        Returns:
            Tuple of (token, coord):
                - token: Atom type tokens (175 types)
                - coord: 3D coordinates
        """
        if 'atom_tokens_coords' not in self._cache:
            from .atom_featurizer import AtomFeaturizer
            atom_featurizer = AtomFeaturizer()

            # Use the standardized PDB if available
            pdb_to_use = self.tmp_pdb if self.tmp_pdb else self.input_file
            token, coord = atom_featurizer.get_protein_atom_features(pdb_to_use)
            self._cache['atom_tokens_coords'] = (token, coord)
        return self._cache['atom_tokens_coords']

    # Aliases for clarity
    get_atom_tokens = get_atom_tokens_and_coords

    def get_atom_features_with_sasa(self) -> Dict[str, Any]:
        """
        Get comprehensive atom features including SASA.

        Returns:
            Dictionary with atom features and SASA
        """
        if 'atom_features_sasa' not in self._cache:
            from .atom_featurizer import AtomFeaturizer
            atom_featurizer = AtomFeaturizer()

            # Use the standardized PDB if available
            pdb_to_use = self.tmp_pdb if self.tmp_pdb else self.input_file
            features = atom_featurizer.get_all_atom_features(pdb_to_use)
            self._cache['atom_features_sasa'] = features
        return self._cache['atom_features_sasa']

    # Aliases for clarity
    get_atom_sasa = get_atom_features_with_sasa
    get_atom_level_features_with_sasa = get_atom_features_with_sasa
    get_atom_level_sasa = get_atom_features_with_sasa

    def get_atom_coordinates(self) -> torch.Tensor:
        """
        Get only atom-level 3D coordinates.

        Returns:
            torch.Tensor: [n_atoms, 3] coordinates
        """
        token, coord = self.get_atom_tokens_and_coords()
        return coord

    def get_atom_tokens_only(self) -> torch.Tensor:
        """
        Get only atom-level tokens without coordinates.

        Returns:
            torch.Tensor: [n_atoms] token IDs (0-174)
        """
        token, coord = self.get_atom_tokens_and_coords()
        return token

    # ============== RESIDUE-LEVEL FEATURES ==============
    # Adding clearer aliases for residue-level methods

    # Sequence features
    get_residue_sequence = get_sequence_features
    get_residue_types = get_sequence_features
    get_residue_level_sequence = get_sequence_features

    # Geometric features
    get_residue_geometry = get_geometric_features
    get_residue_dihedrals = get_geometric_features
    get_residue_level_geometry = get_geometric_features

    # SASA features
    get_residue_sasa = get_sasa_features
    get_residue_level_sasa = get_sasa_features

    # Contact map
    get_residue_contacts = get_contact_map
    get_residue_contact_map = get_contact_map
    get_residue_level_contacts = get_contact_map

    # Node features
    get_residue_node_features = get_node_features
    get_residue_level_node_features = get_node_features

    # Edge features
    get_residue_edge_features = get_edge_features
    get_residue_level_edge_features = get_edge_features

    # Standard features (residue-level by default)
    get_residue_features = get_features
    get_residue_level_features = get_features

    # All features
    get_all_residue_features = get_all_features
    get_residue_level_all_features = get_all_features

    # ============== SEQUENCE FEATURES ==============

    def get_sequence_by_chain(self) -> Dict[str, str]:
        """
        Get amino acid sequences in one-letter code separated by chain.

        Returns:
            Dictionary mapping chain IDs to one-letter amino acid sequences
        """
        return self._featurizer.get_sequence_by_chain()