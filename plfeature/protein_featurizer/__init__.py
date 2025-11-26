"""
Protein Featurizer Module

A comprehensive toolkit for extracting structural features from protein PDB files.
"""

import torch
from .pdb_standardizer import PDBStandardizer, standardize_pdb
from .residue_featurizer import ResidueFeaturizer
from .protein_featurizer import ProteinFeaturizer as EfficientProteinFeaturizer
from .atom_featurizer import AtomFeaturizer, get_protein_atom_features, get_atom_features_with_sasa

__version__ = "0.1.0"
__author__ = "Jaemin Sim"

__all__ = [
    "PDBStandardizer",
    "ResidueFeaturizer",
    "AtomFeaturizer",
    "standardize_pdb",
    "get_protein_atom_features",
    "get_atom_features_with_sasa",
    "ProteinFeaturizer",  # Main API class
]


# Use the efficient implementation
ProteinFeaturizer = EfficientProteinFeaturizer


class ProteinFeaturizerOld:
    """
    High-level API for protein feature extraction.

    This class provides a simple, unified interface for the complete
    feature extraction pipeline.

    Examples:
        >>> # Basic usage
        >>> from plfeature import ProteinFeaturizer
        >>> featurizer = ProteinFeaturizer()
        >>> features = featurizer.extract("protein.pdb")

        >>> # Without standardization
        >>> featurizer = ProteinFeaturizer(standardize=False)
        >>> features = featurizer.extract("clean.pdb")

        >>> # Custom options
        >>> featurizer = ProteinFeaturizer(keep_hydrogens=True)
        >>> features = featurizer.extract("protein.pdb", save_to="features.pt")
    """

    def __init__(self, standardize: bool = True, keep_hydrogens: bool = False):
        """
        Initialize the Featurizer.

        Args:
            standardize: Whether to standardize PDB files before feature extraction
            keep_hydrogens: Whether to keep hydrogen atoms during standardization
        """
        self.standardize = standardize
        self.keep_hydrogens = keep_hydrogens
        self._standardizer = None
        self._featurizer = None

        if self.standardize:
            self._standardizer = PDBStandardizer(remove_hydrogens=not keep_hydrogens)

    def get_sequence_features(self, pdb_file: str) -> dict:
        """
        Get amino acid sequence and position features.

        Returns:
            Dictionary with residue types and one-hot encoding
        """
        import tempfile
        import os
        import torch

        # Prepare PDB file
        if self.standardize:
            with tempfile.NamedTemporaryFile(suffix='.pdb', delete=False) as tmp_file:
                tmp_pdb = tmp_file.name
            self._standardizer.standardize(pdb_file, tmp_pdb)
            pdb_to_process = tmp_pdb
        else:
            pdb_to_process = pdb_file

        try:
            featurizer = ResidueFeaturizer(pdb_to_process)
            residues = featurizer.get_residues()
            residue_types = torch.tensor([res[2] for res in residues])
            residue_one_hot = torch.nn.functional.one_hot(residue_types, num_classes=21)

            return {
                'residue_types': residue_types,
                'residue_one_hot': residue_one_hot,
                'num_residues': len(residues)
            }
        finally:
            if self.standardize:
                os.unlink(pdb_to_process)

    def get_geometric_features(self, pdb_file: str) -> dict:
        """
        Get geometric features including distances, angles, and dihedrals.

        Returns:
            Dictionary with geometric measurements
        """
        import tempfile
        import os
        import torch
        import numpy as np

        # Prepare PDB file
        if self.standardize:
            with tempfile.NamedTemporaryFile(suffix='.pdb', delete=False) as tmp_file:
                tmp_pdb = tmp_file.name
            self._standardizer.standardize(pdb_file, tmp_pdb)
            pdb_to_process = tmp_pdb
        else:
            pdb_to_process = pdb_file

        try:
            featurizer = ResidueFeaturizer(pdb_to_process)
            residues = featurizer.get_residues()

            # Build coordinate tensor
            coords = torch.zeros(len(residues), 15, 3)
            residue_types = torch.from_numpy(np.array(residues)[:, 2].astype(int))

            for idx, residue in enumerate(residues):
                residue_coord = torch.as_tensor(featurizer.get_residue_coordinates(residue).tolist())
                coords[idx, :residue_coord.shape[0], :] = residue_coord
                coords[idx, -1, :] = residue_coord[4:, :].mean(0)  # Sidechain centroid

            # Get geometric features
            dihedrals, has_chi = featurizer.get_dihedral_angles(coords, residue_types)
            terminal_flags = featurizer.get_terminal_flags()
            curvature = featurizer._calculate_backbone_curvature(coords, terminal_flags)
            torsion = featurizer._calculate_backbone_torsion(coords, terminal_flags)
            self_distance, self_vector = featurizer._calculate_self_distances_vectors(coords)

            return {
                'dihedrals': dihedrals,
                'has_chi_angles': has_chi,
                'backbone_curvature': curvature,
                'backbone_torsion': torsion,
                'self_distances': self_distance,
                'self_vectors': self_vector,
                'coordinates': coords
            }
        finally:
            if self.standardize:
                os.unlink(pdb_to_process)

    def get_sasa_features(self, pdb_file: str) -> torch.Tensor:
        """
        Get Solvent Accessible Surface Area features.

        Returns:
            SASA tensor with multiple components per residue
        """
        import tempfile
        import os

        # Prepare PDB file
        if self.standardize:
            with tempfile.NamedTemporaryFile(suffix='.pdb', delete=False) as tmp_file:
                tmp_pdb = tmp_file.name
            self._standardizer.standardize(pdb_file, tmp_pdb)
            pdb_to_process = tmp_pdb
        else:
            pdb_to_process = pdb_file

        try:
            featurizer = ResidueFeaturizer(pdb_to_process)
            return featurizer.calculate_sasa()
        finally:
            if self.standardize:
                os.unlink(pdb_to_process)

    def get_contact_map(self, pdb_file: str, cutoff: float = 8.0) -> dict:
        """
        Get residue-residue contact map and distances.

        Args:
            cutoff: Distance cutoff for contacts (default: 8.0 Å)

        Returns:
            Dictionary with contact information
        """
        import tempfile
        import os
        import torch
        import numpy as np

        # Prepare PDB file
        if self.standardize:
            with tempfile.NamedTemporaryFile(suffix='.pdb', delete=False) as tmp_file:
                tmp_pdb = tmp_file.name
            self._standardizer.standardize(pdb_file, tmp_pdb)
            pdb_to_process = tmp_pdb
        else:
            pdb_to_process = pdb_file

        try:
            featurizer = ResidueFeaturizer(pdb_to_process)
            residues = featurizer.get_residues()

            # Build coordinate tensor
            coords = torch.zeros(len(residues), 15, 3)
            for idx, residue in enumerate(residues):
                residue_coord = torch.as_tensor(featurizer.get_residue_coordinates(residue).tolist())
                coords[idx, :residue_coord.shape[0], :] = residue_coord
                coords[idx, -1, :] = residue_coord[4:, :].mean(0)

            # Get interaction features
            distance_adj, adj, vectors = featurizer._calculate_interaction_features(coords, cutoff=cutoff)

            # Get sparse representation
            sparse = distance_adj.to_sparse(sparse_dim=2)
            src, dst = sparse.indices()
            distances = sparse.values()

            return {
                'adjacency_matrix': adj,
                'distance_matrix': distance_adj,
                'edges': (src, dst),
                'edge_distances': distances,
                'interaction_vectors': vectors
            }
        finally:
            if self.standardize:
                os.unlink(pdb_to_process)

    def get_relative_position(self, pdb_file: str, cutoff: int = 32) -> torch.Tensor:
        """
        Get relative position encoding between residues.

        Args:
            cutoff: Maximum relative position to consider

        Returns:
            One-hot encoded relative position tensor
        """
        import tempfile
        import os

        # Prepare PDB file
        if self.standardize:
            with tempfile.NamedTemporaryFile(suffix='.pdb', delete=False) as tmp_file:
                tmp_pdb = tmp_file.name
            self._standardizer.standardize(pdb_file, tmp_pdb)
            pdb_to_process = tmp_pdb
        else:
            pdb_to_process = pdb_file

        try:
            featurizer = ResidueFeaturizer(pdb_to_process)
            return featurizer.get_relative_position(cutoff=cutoff, onehot=True)
        finally:
            if self.standardize:
                os.unlink(pdb_to_process)

    def get_node_features(self, pdb_file: str) -> dict:
        """
        Get all node (residue-level) features.

        Returns:
            Dictionary with scalar and vector node features
        """
        import tempfile
        import os
        import torch
        import numpy as np

        # Prepare PDB file
        if self.standardize:
            with tempfile.NamedTemporaryFile(suffix='.pdb', delete=False) as tmp_file:
                tmp_pdb = tmp_file.name
            self._standardizer.standardize(pdb_file, tmp_pdb)
            pdb_to_process = tmp_pdb
        else:
            pdb_to_process = pdb_file

        try:
            featurizer = ResidueFeaturizer(pdb_to_process)
            residues = featurizer.get_residues()

            # Build coordinate tensor
            coords = torch.zeros(len(residues), 15, 3)
            residue_types = torch.from_numpy(np.array(residues)[:, 2].astype(int))

            for idx, residue in enumerate(residues):
                residue_coord = torch.as_tensor(featurizer.get_residue_coordinates(residue).tolist())
                coords[idx, :residue_coord.shape[0], :] = residue_coord
                coords[idx, -1, :] = residue_coord[4:, :].mean(0)

            # Extract node features
            scalar_features, vector_features = featurizer._extract_residue_features(coords, residue_types)

            # Get CA and SC coordinates
            coords_CA = coords[:, 1:2, :]
            coords_SC = coords[:, -1:, :]
            coord = torch.cat([coords_CA, coords_SC], dim=1)

            return {
                'coordinates': coord,
                'scalar_features': scalar_features,
                'vector_features': vector_features
            }
        finally:
            if self.standardize:
                os.unlink(pdb_to_process)

    def get_edge_features(self, pdb_file: str, distance_cutoff: float = 8.0) -> dict:
        """
        Get all edge (interaction) features.

        Args:
            distance_cutoff: Distance cutoff for interactions

        Returns:
            Dictionary with edge indices and features
        """
        import tempfile
        import os
        import torch
        import numpy as np

        # Prepare PDB file
        if self.standardize:
            with tempfile.NamedTemporaryFile(suffix='.pdb', delete=False) as tmp_file:
                tmp_pdb = tmp_file.name
            self._standardizer.standardize(pdb_file, tmp_pdb)
            pdb_to_process = tmp_pdb
        else:
            pdb_to_process = pdb_file

        try:
            featurizer = ResidueFeaturizer(pdb_to_process)
            residues = featurizer.get_residues()

            # Build coordinate tensor
            coords = torch.zeros(len(residues), 15, 3)
            for idx, residue in enumerate(residues):
                residue_coord = torch.as_tensor(featurizer.get_residue_coordinates(residue).tolist())
                coords[idx, :residue_coord.shape[0], :] = residue_coord
                coords[idx, -1, :] = residue_coord[4:, :].mean(0)

            # Extract edge features
            edges, scalar_features, vector_features = featurizer._extract_interaction_features(
                coords, distance_cutoff=distance_cutoff
            )

            return {
                'edges': edges,
                'scalar_features': scalar_features,
                'vector_features': vector_features
            }
        finally:
            if self.standardize:
                os.unlink(pdb_to_process)

    def get_terminal_flags(self, pdb_file: str) -> dict:
        """
        Get N-terminal and C-terminal residue flags.

        Returns:
            Dictionary with terminal flags
        """
        import tempfile
        import os

        # Prepare PDB file
        if self.standardize:
            with tempfile.NamedTemporaryFile(suffix='.pdb', delete=False) as tmp_file:
                tmp_pdb = tmp_file.name
            self._standardizer.standardize(pdb_file, tmp_pdb)
            pdb_to_process = tmp_pdb
        else:
            pdb_to_process = pdb_file

        try:
            featurizer = ResidueFeaturizer(pdb_to_process)
            n_terminal, c_terminal = featurizer.get_terminal_flags()
            return {
                'n_terminal': n_terminal,
                'c_terminal': c_terminal
            }
        finally:
            if self.standardize:
                os.unlink(pdb_to_process)

    def get_features(self, pdb_file: str) -> tuple:
        """
        Get node and edge features in standard format.

        Args:
            pdb_file: Path to the PDB file

        Returns:
            Tuple of (node, edge) dictionaries with:
            - node: {'coord', 'node_scalar_features', 'node_vector_features'}
            - edge: {'edges', 'edge_scalar_features', 'edge_vector_features'}
        """
        import tempfile
        import os

        # Prepare PDB file
        if self.standardize:
            with tempfile.NamedTemporaryFile(suffix='.pdb', delete=False) as tmp_file:
                tmp_pdb = tmp_file.name
            self._standardizer.standardize(pdb_file, tmp_pdb)
            pdb_to_process = tmp_pdb
        else:
            pdb_to_process = pdb_file

        try:
            featurizer = ResidueFeaturizer(pdb_to_process)
            node, edge = featurizer.get_features()
            return node, edge
        finally:
            if self.standardize:
                os.unlink(pdb_to_process)

    def get_atom_features(self, pdb_file: str) -> tuple:
        """
        Get atom-level features from PDB file.

        Args:
            pdb_file: Path to the PDB file

        Returns:
            Tuple of (token, coord):
                - token: torch.Tensor with atom type tokens
                - coord: torch.Tensor with 3D coordinates
        """
        from .atom_featurizer import AtomFeaturizer
        import tempfile
        import os

        # Prepare PDB file
        if self.standardize:
            with tempfile.NamedTemporaryFile(suffix='.pdb', delete=False) as tmp_file:
                tmp_pdb = tmp_file.name
            self._standardizer.standardize(pdb_file, tmp_pdb)
            pdb_to_process = tmp_pdb
        else:
            pdb_to_process = pdb_file

        try:
            atom_featurizer = AtomFeaturizer()
            token, coord = atom_featurizer.get_protein_atom_features(pdb_to_process)
            return token, coord
        finally:
            if self.standardize:
                os.unlink(pdb_to_process)

    def get_atom_features_with_sasa(self, pdb_file: str) -> dict:
        """
        Get all atom-level features including SASA.

        Args:
            pdb_file: Path to the PDB file

        Returns:
            Dictionary with atom features including SASA
        """
        from .atom_featurizer import AtomFeaturizer
        import tempfile
        import os

        # Prepare PDB file
        if self.standardize:
            with tempfile.NamedTemporaryFile(suffix='.pdb', delete=False) as tmp_file:
                tmp_pdb = tmp_file.name
            self._standardizer.standardize(pdb_file, tmp_pdb)
            pdb_to_process = tmp_pdb
        else:
            pdb_to_process = pdb_file

        try:
            atom_featurizer = AtomFeaturizer()
            features = atom_featurizer.get_all_atom_features(pdb_to_process)
            return features
        finally:
            if self.standardize:
                os.unlink(pdb_to_process)

    def extract(self, pdb_file: str, save_to: str = None) -> dict:
        """
        Extract features from a PDB file.

        Args:
            pdb_file: Path to the PDB file
            save_to: Optional path to save the extracted features

        Returns:
            Dictionary containing node and edge features

        Raises:
            FileNotFoundError: If PDB file doesn't exist
            ValueError: If feature extraction fails
        """
        import os
        import tempfile
        import torch

        # Check if file exists
        if not os.path.exists(pdb_file):
            raise FileNotFoundError(f"PDB file not found: {pdb_file}")

        try:
            # Standardize if requested
            if self.standardize:
                with tempfile.NamedTemporaryFile(suffix='.pdb', delete=False) as tmp_file:
                    tmp_pdb = tmp_file.name

                self._standardizer.standardize(pdb_file, tmp_pdb)
                pdb_to_process = tmp_pdb
            else:
                pdb_to_process = pdb_file

            # Extract features
            self._featurizer = ResidueFeaturizer(pdb_to_process)
            node_features, edge_features = self._featurizer.get_features()

            # Package features
            features = {
                'node': node_features,
                'edge': edge_features,
                'metadata': {
                    'input_file': pdb_file,
                    'standardized': self.standardize,
                    'hydrogens_removed': not self.keep_hydrogens if self.standardize else None
                }
            }

            # Save if requested
            if save_to:
                torch.save(features, save_to)

            # Cleanup
            if self.standardize:
                os.unlink(pdb_to_process)

            return features

        except Exception as e:
            raise ValueError(f"Failed to extract features from {pdb_file}: {str(e)}")

    def extract_batch(self, pdb_files: list, output_dir: str = None,
                     skip_existing: bool = True, verbose: bool = True) -> dict:
        """
        Extract features from multiple PDB files.

        Args:
            pdb_files: List of PDB file paths
            output_dir: Directory to save feature files (optional)
            skip_existing: Whether to skip files that already have features
            verbose: Whether to print progress

        Returns:
            Dictionary mapping file names to features or output paths
        """
        import os
        from pathlib import Path

        results = {}

        for i, pdb_file in enumerate(pdb_files):
            file_name = Path(pdb_file).stem

            # Determine output path if saving
            output_path = None
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
                output_path = os.path.join(output_dir, f"{file_name}_features.pt")

                # Skip if exists
                if skip_existing and os.path.exists(output_path):
                    if verbose:
                        print(f"[{i+1}/{len(pdb_files)}] Skipping {file_name} (already exists)")
                    results[file_name] = output_path
                    continue

            try:
                if verbose:
                    print(f"[{i+1}/{len(pdb_files)}] Processing {file_name}...")

                features = self.extract(pdb_file, save_to=output_path)
                results[file_name] = output_path if output_path else features

            except Exception as e:
                if verbose:
                    print(f"[{i+1}/{len(pdb_files)}] Failed {file_name}: {str(e)}")
                results[file_name] = None

        return results

    @classmethod
    def from_clean_pdb(cls, pdb_file: str) -> dict:
        """
        Extract features from an already clean/standardized PDB file.

        Args:
            pdb_file: Path to clean PDB file

        Returns:
            Dictionary containing features
        """
        featurizer = cls(standardize=False)
        return featurizer.extract(pdb_file)

    @staticmethod
    def standardize_only(input_pdb: str, output_pdb: str,
                        keep_hydrogens: bool = False) -> str:
        """
        Only standardize a PDB file without feature extraction.

        Args:
            input_pdb: Input PDB file path
            output_pdb: Output PDB file path
            keep_hydrogens: Whether to keep hydrogen atoms

        Returns:
            Path to standardized PDB file
        """
        return standardize_pdb(input_pdb, output_pdb,
                              remove_hydrogens=not keep_hydrogens)