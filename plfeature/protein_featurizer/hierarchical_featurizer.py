"""
Hierarchical Protein Featurizer for Atom-Residue attention models.

This module provides feature extraction with atom-residue mapping
for hierarchical attention mechanisms:
  - Atom attention → Residue pooling → Residue attention → Atom broadcast

Uses AtomFeaturizer for protein-specific atom tokens, ResidueFeaturizer for
residue features, and DualESMFeaturizer for ESMC + ESM3 embeddings.

Usage:
    from plfeature.protein_featurizer import HierarchicalFeaturizer

    featurizer = HierarchicalFeaturizer()
    data = featurizer.featurize("protein.pdb")

    # Access features
    atom_tokens = data.atom_tokens           # [num_atoms] - token IDs (187 classes)
    residue_features = data.residue_features # [num_residues, 76]
    esmc_embeddings = data.esmc_embeddings   # [num_residues, 1152]
    esm3_embeddings = data.esm3_embeddings   # [num_residues, 1536]

    # For pooling: atom → residue
    atom_to_residue = data.atom_to_residue  # [num_atoms] index
"""

from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Union
from collections import defaultdict

import torch
import numpy as np

from .residue_featurizer import ResidueFeaturizer
from .atom_featurizer import AtomFeaturizer
from ..constants import (
    RESIDUE_ATOM_TOKEN,
    UNK_TOKEN,
    RESIDUE_TOKEN,
)


# Number of unique atom tokens
NUM_ATOM_TOKENS = UNK_TOKEN + 1  # 187

# Simplified element types (8 classes)
ELEMENT_TYPES = {
    'C': 0,      # Carbon
    'N': 1,      # Nitrogen
    'O': 2,      # Oxygen
    'S': 3,      # Sulfur
    'P': 4,      # Phosphorus
    'SE': 5,     # Selenium
    'METAL': 6,  # All metals (CA, MG, ZN, FE, MN, CU, CO, NI, NA, K, etc.)
    'UNK': 7,    # Unknown
}
NUM_ELEMENT_TYPES = len(ELEMENT_TYPES)

# Metal elements for detection
METAL_ELEMENTS = {'CA', 'MG', 'ZN', 'FE', 'MN', 'CU', 'CO', 'NI', 'NA', 'K'}


@dataclass
class HierarchicalProteinData:
    """
    Hierarchical protein data for atom-residue attention models.

    Atom-level tensors (one-hot encoded):
        atom_tokens: [num_atoms, 187] - Atom token one-hot (187 classes)
        atom_coords: [num_atoms, 3] - 3D coordinates (raw, not normalized)
        atom_sasa: [num_atoms] - SASA values (normalized by /100)
        atom_elements: [num_atoms, 8] - Element type one-hot (8 classes)
        atom_residue_types: [num_atoms, 22] - Residue type one-hot (22 classes)
        atom_names: List[str] - PDB atom names (CA, CB, etc.)

    Residue-level tensors:
        residue_features: [num_residues, 76] - Residue feature vectors
        residue_ca_coords: [num_residues, 3] - CA coordinates
        residue_sc_coords: [num_residues, 3] - Sidechain centroid coordinates
        residue_names: List[str] - Residue names (ALA, GLY, etc.)
        residue_ids: List[Tuple[str, int]] - (chain, resnum)

    Mapping tensors (for pooling/broadcast):
        atom_to_residue: [num_atoms] - Residue index for each atom
        residue_atom_indices: [num_residues, max_atoms] - Atom indices per residue
        residue_atom_mask: [num_residues, max_atoms] - Valid atom mask
        num_atoms_per_residue: [num_residues] - Atom count per residue

    ESM embeddings (6 tensors total):
        esmc_embeddings: [num_residues, 1152] - ESMC per-residue embeddings
        esmc_bos: [1152] - ESMC BOS token
        esmc_eos: [1152] - ESMC EOS token
        esm3_embeddings: [num_residues, 1536] - ESM3 per-residue embeddings
        esm3_bos: [1536] - ESM3 BOS token
        esm3_eos: [1536] - ESM3 EOS token
    """
    # Atom-level (one-hot encoded)
    atom_tokens: torch.Tensor           # [N_atom, 187] one-hot
    atom_coords: torch.Tensor           # [N_atom, 3] raw coordinates
    atom_sasa: torch.Tensor             # [N_atom] normalized SASA
    atom_elements: torch.Tensor         # [N_atom, 8] one-hot
    atom_residue_types: torch.Tensor    # [N_atom, 22] one-hot
    atom_names: List[str]               # atom names (CA, CB, etc.)

    # Residue-level
    residue_features: torch.Tensor      # [N_res, 76]
    residue_ca_coords: torch.Tensor     # [N_res, 3]
    residue_sc_coords: torch.Tensor     # [N_res, 3]
    residue_names: List[str]
    residue_ids: List[Tuple[str, int]]

    # Residue vector features (optional) [N_res, 31, 3]
    residue_vector_features: Optional[torch.Tensor] = None

    # ESMC embeddings (3 tensors)
    esmc_embeddings: Optional[torch.Tensor] = None   # [N_res, 1152]
    esmc_bos: Optional[torch.Tensor] = None          # [1152]
    esmc_eos: Optional[torch.Tensor] = None          # [1152]

    # ESM3 embeddings (3 tensors)
    esm3_embeddings: Optional[torch.Tensor] = None   # [N_res, 1536]
    esm3_bos: Optional[torch.Tensor] = None          # [1536]
    esm3_eos: Optional[torch.Tensor] = None          # [1536]

    # Mapping
    atom_to_residue: torch.Tensor = None
    residue_atom_indices: torch.Tensor = None
    residue_atom_mask: torch.Tensor = None
    num_atoms_per_residue: torch.Tensor = None

    @property
    def num_atoms(self) -> int:
        return self.atom_tokens.shape[0]

    @property
    def num_residues(self) -> int:
        return self.residue_features.shape[0]

    @property
    def max_atoms_per_residue(self) -> int:
        return self.residue_atom_indices.shape[1] if self.residue_atom_indices is not None else 0

    @property
    def residue_dim(self) -> int:
        return self.residue_features.shape[1]

    @property
    def num_atom_classes(self) -> int:
        return self.atom_tokens.shape[1]  # 187

    @property
    def num_element_classes(self) -> int:
        return self.atom_elements.shape[1]  # 8

    @property
    def num_residue_classes(self) -> int:
        return self.atom_residue_types.shape[1]  # 22

    @property
    def esmc_dim(self) -> Optional[int]:
        """Get ESMC embedding dimension if available."""
        return self.esmc_embeddings.shape[-1] if self.esmc_embeddings is not None else None

    @property
    def esm3_dim(self) -> Optional[int]:
        """Get ESM3 embedding dimension if available."""
        return self.esm3_embeddings.shape[-1] if self.esm3_embeddings is not None else None

    @property
    def has_esm(self) -> bool:
        """Check if ESM embeddings are available."""
        return self.esmc_embeddings is not None or self.esm3_embeddings is not None

    def to(self, device: torch.device) -> 'HierarchicalProteinData':
        """Move all tensors to device."""
        return HierarchicalProteinData(
            atom_tokens=self.atom_tokens.to(device),
            atom_coords=self.atom_coords.to(device),
            atom_sasa=self.atom_sasa.to(device),
            atom_elements=self.atom_elements.to(device),
            atom_residue_types=self.atom_residue_types.to(device),
            atom_names=self.atom_names,
            residue_features=self.residue_features.to(device),
            residue_ca_coords=self.residue_ca_coords.to(device),
            residue_sc_coords=self.residue_sc_coords.to(device),
            residue_names=self.residue_names,
            residue_ids=self.residue_ids,
            residue_vector_features=self.residue_vector_features.to(device) if self.residue_vector_features is not None else None,
            esmc_embeddings=self.esmc_embeddings.to(device) if self.esmc_embeddings is not None else None,
            esmc_bos=self.esmc_bos.to(device) if self.esmc_bos is not None else None,
            esmc_eos=self.esmc_eos.to(device) if self.esmc_eos is not None else None,
            esm3_embeddings=self.esm3_embeddings.to(device) if self.esm3_embeddings is not None else None,
            esm3_bos=self.esm3_bos.to(device) if self.esm3_bos is not None else None,
            esm3_eos=self.esm3_eos.to(device) if self.esm3_eos is not None else None,
            atom_to_residue=self.atom_to_residue.to(device),
            residue_atom_indices=self.residue_atom_indices.to(device),
            residue_atom_mask=self.residue_atom_mask.to(device),
            num_atoms_per_residue=self.num_atoms_per_residue.to(device),
        )

    def get_feature_dims(self) -> Dict[str, int]:
        """Get feature dimensions."""
        dims = {
            'num_atoms': self.num_atoms,
            'num_residues': self.num_residues,
            'residue_dim': self.residue_dim,
            'max_atoms_per_residue': self.max_atoms_per_residue,
            'num_atom_classes': self.num_atom_classes,
            'num_element_classes': self.num_element_classes,
            'num_residue_classes': self.num_residue_classes,
        }
        if self.esmc_dim is not None:
            dims['esmc_dim'] = self.esmc_dim
        if self.esm3_dim is not None:
            dims['esm3_dim'] = self.esm3_dim
        return dims

    def select_residues(
        self,
        residue_indices: Union[List[int], torch.Tensor],
    ) -> 'HierarchicalProteinData':
        """
        Select specific residues and their corresponding atoms.

        Args:
            residue_indices: Indices of residues to select

        Returns:
            New HierarchicalProteinData with only selected residues/atoms
        """
        if not isinstance(residue_indices, torch.Tensor):
            residue_indices = torch.tensor(residue_indices, dtype=torch.long)

        # Get atom mask for selected residues
        atom_mask = torch.isin(self.atom_to_residue, residue_indices)
        atom_indices = atom_mask.nonzero(as_tuple=True)[0]

        # Remap residue indices to 0, 1, 2, ...
        old_to_new = {old.item(): new for new, old in enumerate(residue_indices)}
        new_atom_to_residue = torch.tensor([
            old_to_new[self.atom_to_residue[i].item()]
            for i in atom_indices
        ], dtype=torch.long)

        # Extract atom-level data
        new_atom_tokens = self.atom_tokens[atom_mask]
        new_atom_coords = self.atom_coords[atom_mask]
        new_atom_sasa = self.atom_sasa[atom_mask]
        new_atom_elements = self.atom_elements[atom_mask]
        new_atom_residue_types = self.atom_residue_types[atom_mask]
        new_atom_names = [self.atom_names[i] for i in atom_indices.tolist()]

        # Extract residue-level data
        new_residue_features = self.residue_features[residue_indices]
        new_residue_ca_coords = self.residue_ca_coords[residue_indices]
        new_residue_sc_coords = self.residue_sc_coords[residue_indices]
        new_residue_names = [self.residue_names[i] for i in residue_indices.tolist()]
        new_residue_ids = [self.residue_ids[i] for i in residue_indices.tolist()]

        # Residue vector features (optional)
        new_residue_vector_features = None
        if self.residue_vector_features is not None:
            new_residue_vector_features = self.residue_vector_features[residue_indices]

        # Build new residue_atom_indices and mask
        num_new_residues = len(residue_indices)
        atoms_per_residue = []
        for new_idx in range(num_new_residues):
            atoms_per_residue.append((new_atom_to_residue == new_idx).sum().item())

        max_atoms = max(atoms_per_residue) if atoms_per_residue else 1
        new_residue_atom_indices = torch.full((num_new_residues, max_atoms), -1, dtype=torch.long)
        new_residue_atom_mask = torch.zeros(num_new_residues, max_atoms, dtype=torch.bool)
        new_num_atoms_per_residue = torch.zeros(num_new_residues, dtype=torch.long)

        for new_res_idx in range(num_new_residues):
            res_atom_mask = (new_atom_to_residue == new_res_idx)
            res_atom_indices = res_atom_mask.nonzero(as_tuple=True)[0]
            n_atoms = len(res_atom_indices)
            new_num_atoms_per_residue[new_res_idx] = n_atoms
            for j, atom_idx in enumerate(res_atom_indices.tolist()):
                if j < max_atoms:
                    new_residue_atom_indices[new_res_idx, j] = atom_idx
                    new_residue_atom_mask[new_res_idx, j] = True

        # ESM embeddings: select residue embeddings, keep original BOS/EOS
        new_esmc_embeddings = None
        new_esm3_embeddings = None
        if self.esmc_embeddings is not None:
            new_esmc_embeddings = self.esmc_embeddings[residue_indices]
        if self.esm3_embeddings is not None:
            new_esm3_embeddings = self.esm3_embeddings[residue_indices]

        return HierarchicalProteinData(
            atom_tokens=new_atom_tokens,
            atom_coords=new_atom_coords,
            atom_sasa=new_atom_sasa,
            atom_elements=new_atom_elements,
            atom_residue_types=new_atom_residue_types,
            atom_names=new_atom_names,
            residue_features=new_residue_features,
            residue_ca_coords=new_residue_ca_coords,
            residue_sc_coords=new_residue_sc_coords,
            residue_names=new_residue_names,
            residue_ids=new_residue_ids,
            residue_vector_features=new_residue_vector_features,
            esmc_embeddings=new_esmc_embeddings,
            esmc_bos=self.esmc_bos,  # Keep original BOS/EOS
            esmc_eos=self.esmc_eos,
            esm3_embeddings=new_esm3_embeddings,
            esm3_bos=self.esm3_bos,
            esm3_eos=self.esm3_eos,
            atom_to_residue=new_atom_to_residue,
            residue_atom_indices=new_residue_atom_indices,
            residue_atom_mask=new_residue_atom_mask,
            num_atoms_per_residue=new_num_atoms_per_residue,
        )


class HierarchicalFeaturizer:
    """
    Extract hierarchical features from protein for atom-residue attention.

    Combines:
        - AtomFeaturizer for protein-specific atom tokens (187 classes)
        - ResidueFeaturizer for residue-level features (76 dim + 31x3 vectors)
        - DualESMFeaturizer for ESMC + ESM3 embeddings

    Args:
        esmc_model: ESMC model name (default: "esmc_600m", 1152-dim)
        esm3_model: ESM3 model name (default: "esm3-open", 1536-dim)
        esm_device: Device for ESM models ("cuda" or "cpu")

    Feature dimensions:
        Atom features (one-hot encoded):
            - atom_tokens: [N_atom, 187] - Atom token one-hot
            - atom_coords: [N_atom, 3] - 3D coordinates
            - atom_sasa: [N_atom] - SASA values (normalized /100)
            - atom_elements: [N_atom, 8] - Element type one-hot
            - atom_residue_types: [N_atom, 22] - Residue type one-hot

        Residue features (from ResidueFeaturizer):
            - residue_features: [N_res, 76] scalar features
            - residue_vector_features: [N_res, 31, 3] vector features

        ESM embeddings (6 tensors):
            - esmc_embeddings: [N_res, 1152] - ESMC per-residue
            - esmc_bos: [1152] - ESMC BOS token
            - esmc_eos: [1152] - ESMC EOS token
            - esm3_embeddings: [N_res, 1536] - ESM3 per-residue
            - esm3_bos: [1536] - ESM3 BOS token
            - esm3_eos: [1536] - ESM3 EOS token
    """

    def __init__(
        self,
        esmc_model: str = "esmc_600m",
        esm3_model: str = "esm3-open",
        esm_device: str = "cuda",
    ):
        self._atom_featurizer = AtomFeaturizer()

        # Initialize Dual ESM featurizer (ESMC + ESM3)
        from .esm_featurizer import DualESMFeaturizer
        self._esm_featurizer = DualESMFeaturizer(
            esmc_model=esmc_model,
            esm3_model=esm3_model,
            device=esm_device,
        )

    def featurize(self, pdb_path: str) -> HierarchicalProteinData:
        """
        Extract hierarchical features from PDB file.

        Args:
            pdb_path: Path to PDB file

        Returns:
            HierarchicalProteinData with all features and mappings
        """
        # Step 1: Get atom features from AtomFeaturizer
        atom_tokens, atom_coords = self._atom_featurizer.get_protein_atom_features(pdb_path)

        # Step 2: Get SASA from AtomFeaturizer
        atom_sasa, _ = self._atom_featurizer.get_atom_sasa(pdb_path)
        # Ensure same length (SASA might have different atom count due to different parsing)
        min_len = min(len(atom_tokens), len(atom_sasa))
        atom_tokens = atom_tokens[:min_len]
        atom_coords = atom_coords[:min_len]
        atom_sasa = atom_sasa[:min_len]

        # Step 3: Get additional atom info (elements, residue types)
        atom_info = self._parse_atom_info(pdb_path)
        atom_elements = atom_info['elements'][:min_len]
        atom_residue_types = atom_info['residue_tokens'][:min_len]
        atom_names = atom_info['atom_names'][:min_len]
        atom_residue_keys = atom_info['residue_keys'][:min_len]

        # Step 4: Normalize SASA (typical range 0-150 Å²)
        atom_sasa = atom_sasa / 100.0

        # Step 5: Get residue features from ResidueFeaturizer
        residue_featurizer = ResidueFeaturizer(pdb_path)
        residues = residue_featurizer.get_residues()

        # Build residue coordinate tensor
        num_residues = len(residues)
        residue_coords_full = torch.zeros(num_residues, 15, 3)
        residue_types = torch.from_numpy(np.array(residues)[:, 2].astype(int))

        for idx, residue in enumerate(residues):
            residue_coord = torch.as_tensor(residue_featurizer.get_residue_coordinates(residue).tolist())
            residue_coords_full[idx, :residue_coord.shape[0], :] = residue_coord
            residue_coords_full[idx, -1, :] = residue_coord[4:, :].mean(0) if residue_coord.shape[0] > 4 else residue_coord.mean(0)

        # Extract residue features
        scalar_features, vector_features = residue_featurizer._extract_residue_features(
            residue_coords_full, residue_types
        )

        # Concatenate scalar features
        residue_one_hot, terminal_flags, self_distance, degree_feature, has_chi, sasa, rf_distance = scalar_features
        residue_features = torch.cat([
            residue_one_hot.float(),      # 21
            terminal_flags.float(),       # 2
            self_distance,                # 10
            degree_feature,               # 20
            has_chi.float(),              # 5
            sasa,                         # 10
            rf_distance,                  # 8
        ], dim=-1)  # Total: 76

        # Residue coordinates: CA and sidechain centroid
        residue_ca_coords = residue_coords_full[:, 1, :]   # CA atom (index 1)
        residue_sc_coords = residue_coords_full[:, -1, :]  # Sidechain centroid (index 14)

        # Step 6: Build atom-residue mapping
        atom_to_residue, residue_atom_indices, residue_atom_mask, num_atoms_per_residue = \
            self._build_mapping(atom_residue_keys, residues)

        # Residue info
        residue_names = []
        residue_ids = []
        INT_TO_3LETTER = {
            0: 'ALA', 1: 'ARG', 2: 'ASN', 3: 'ASP', 4: 'CYS',
            5: 'GLN', 6: 'GLU', 7: 'GLY', 8: 'HIS', 9: 'ILE',
            10: 'LEU', 11: 'LYS', 12: 'MET', 13: 'PHE', 14: 'PRO',
            15: 'SER', 16: 'THR', 17: 'TRP', 18: 'TYR', 19: 'VAL', 20: 'UNK'
        }
        for chain, resnum, restype in residues:
            residue_names.append(INT_TO_3LETTER.get(restype, 'UNK'))
            residue_ids.append((chain, resnum))

        # Residue vector features [N_res, 31, 3]
        self_vector, rf_vector, local_frames = vector_features
        residue_vector_features = torch.cat([
            self_vector,      # [N_res, 20, 3]
            rf_vector,        # [N_res, 8, 3]
            local_frames,     # [N_res, 3, 3]
        ], dim=1)

        # Step 7: Convert categorical features to one-hot encoding
        num_atoms = len(atom_tokens)
        atom_tokens_onehot = torch.zeros(num_atoms, NUM_ATOM_TOKENS, dtype=torch.float32)
        atom_tokens_onehot.scatter_(1, atom_tokens.unsqueeze(1), 1.0)

        atom_elements_onehot = torch.zeros(num_atoms, NUM_ELEMENT_TYPES, dtype=torch.float32)
        atom_elements_onehot.scatter_(1, atom_elements.unsqueeze(1), 1.0)

        atom_residue_types_onehot = torch.zeros(num_atoms, 22, dtype=torch.float32)
        atom_residue_types_onehot.scatter_(1, atom_residue_types.unsqueeze(1), 1.0)

        # Step 8: Extract dual ESM embeddings (ESMC + ESM3) - 6 tensors total
        esm_result = self._esm_featurizer.extract_from_pdb(pdb_path)

        # ESMC: embeddings [N_res, 1152], bos [1152], eos [1152]
        esmc_embeddings = esm_result['esmc_embeddings']
        esmc_bos = esm_result['esmc_bos_token']
        esmc_eos = esm_result['esmc_eos_token']

        # ESM3: embeddings [N_res, 1536], bos [1536], eos [1536]
        esm3_embeddings = esm_result['esm3_embeddings']
        esm3_bos = esm_result['esm3_bos_token']
        esm3_eos = esm_result['esm3_eos_token']

        # Verify length matches and truncate/pad if needed
        def _adjust_length(emb, target_len, name):
            if emb.shape[0] != target_len:
                import logging
                logging.warning(
                    f"{name} length ({emb.shape[0]}) != residue count ({target_len}). Adjusting."
                )
                if emb.shape[0] > target_len:
                    return emb[:target_len]
                else:
                    pad = torch.zeros(target_len - emb.shape[0], emb.shape[1])
                    return torch.cat([emb, pad], dim=0)
            return emb

        esmc_embeddings = _adjust_length(esmc_embeddings, num_residues, "ESMC")
        esm3_embeddings = _adjust_length(esm3_embeddings, num_residues, "ESM3")

        return HierarchicalProteinData(
            atom_tokens=atom_tokens_onehot,
            atom_coords=atom_coords,
            atom_sasa=atom_sasa,
            atom_elements=atom_elements_onehot,
            atom_residue_types=atom_residue_types_onehot,
            atom_names=atom_names,
            residue_features=residue_features,
            residue_ca_coords=residue_ca_coords,
            residue_sc_coords=residue_sc_coords,
            residue_names=residue_names,
            residue_ids=residue_ids,
            residue_vector_features=residue_vector_features,
            esmc_embeddings=esmc_embeddings,
            esmc_bos=esmc_bos,
            esmc_eos=esmc_eos,
            esm3_embeddings=esm3_embeddings,
            esm3_bos=esm3_bos,
            esm3_eos=esm3_eos,
            atom_to_residue=atom_to_residue,
            residue_atom_indices=residue_atom_indices,
            residue_atom_mask=residue_atom_mask,
            num_atoms_per_residue=num_atoms_per_residue,
        )

    def featurize_pocket(
        self,
        protein_pdb_path: str,
        ligand,  # Chem.Mol
        cutoff: float = 6.0,
    ) -> HierarchicalProteinData:
        """
        Extract hierarchical features from binding pocket.

        Args:
            protein_pdb_path: Path to protein PDB file
            ligand: RDKit Mol object of ligand (with 3D coords)
            cutoff: Distance cutoff for pocket extraction (default 6.0 Å)

        Returns:
            HierarchicalProteinData with all features and mappings
        """
        import tempfile
        import os
        from rdkit import Chem
        from ..interaction_featurizer import extract_pocket

        # Extract pocket
        pocket_info = extract_pocket(protein_pdb_path, ligand, cutoff=cutoff)
        pocket_mol = pocket_info.pocket_mol

        # Save pocket to temp PDB
        with tempfile.NamedTemporaryFile(mode='w', suffix='.pdb', delete=False) as f:
            f.write(Chem.MolToPDBBlock(pocket_mol))
            pocket_pdb = f.name

        try:
            return self.featurize(pocket_pdb)
        finally:
            os.unlink(pocket_pdb)

    def _parse_atom_info(self, pdb_path: str) -> Dict:
        """
        Parse additional atom information from PDB file.

        Returns dict with:
            - elements: [N_atom] element type indices
            - residue_tokens: [N_atom] residue type for each atom
            - atom_names: List[str] atom names
            - residue_keys: List[(chain, resnum)] for mapping
        """
        from .atom_featurizer import (
            is_atom_record, is_hetatm_record, is_hydrogen, parse_pdb_atom_line
        )
        from ..constants import HISTIDINE_VARIANTS, CYSTEINE_VARIANTS

        elements = []
        residue_tokens = []
        atom_names = []
        residue_keys = []

        with open(pdb_path, 'r') as f:
            lines = f.readlines()

        for line in lines:
            if not (is_atom_record(line) or is_hetatm_record(line)):
                continue
            if is_hydrogen(line):
                continue

            record_type, atom_name, res_name, res_num, chain_id, xyz, element = parse_pdb_atom_line(line)

            # Skip water and special residues
            if res_name == 'HOH':
                continue
            if atom_name == 'OXT' or res_name in ['LLP', 'PTR']:
                continue

            # Normalize residue name
            res_name_clean = res_name.strip()

            # Handle metal ions
            if len(atom_name) >= 2 and len(res_name_clean) >= 2 and atom_name[:2] == res_name_clean[:2]:
                res_name_clean = 'METAL'
            elif res_name_clean in HISTIDINE_VARIANTS:
                res_name_clean = 'HIS'
            elif res_name_clean in CYSTEINE_VARIANTS:
                res_name_clean = 'CYS'
            elif res_name_clean not in self._atom_featurizer.aa_letter:
                res_name_clean = 'UNK'

            # Element type (simplified: C, N, O, S, P, Se, Metal, UNK)
            element_upper = element.upper() if element else ''
            if element_upper in ELEMENT_TYPES:
                elem_idx = ELEMENT_TYPES[element_upper]
            elif element_upper in METAL_ELEMENTS:
                elem_idx = ELEMENT_TYPES['METAL']
            else:
                elem_idx = ELEMENT_TYPES['UNK']

            # Residue token
            res_tok = RESIDUE_TOKEN.get(res_name_clean, RESIDUE_TOKEN['UNK'])

            elements.append(elem_idx)
            residue_tokens.append(res_tok)
            atom_names.append(atom_name)
            residue_keys.append((chain_id, res_num))

        return {
            'elements': torch.tensor(elements, dtype=torch.long),
            'residue_tokens': torch.tensor(residue_tokens, dtype=torch.long),
            'atom_names': atom_names,
            'residue_keys': residue_keys,
        }

    def _build_mapping(
        self,
        atom_residue_keys: List[Tuple[str, int]],
        residues: List[Tuple],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Build atom-residue mapping tensors.

        Args:
            atom_residue_keys: List of (chain, resnum) for each atom
            residues: List of (chain, resnum, restype) tuples

        Returns:
            Tuple of (atom_to_residue, residue_atom_indices, residue_atom_mask, num_atoms_per_residue)
        """
        # Build residue key to index mapping
        residue_to_idx = {(chain, resnum): i for i, (chain, resnum, _) in enumerate(residues)}

        num_atoms = len(atom_residue_keys)
        num_residues = len(residues)

        # Map atoms to residues
        atom_to_residue_list = []
        residue_to_atoms = defaultdict(list)

        for atom_idx, (chain, resnum) in enumerate(atom_residue_keys):
            key = (chain, resnum)
            res_idx = residue_to_idx.get(key, 0)  # default to first residue if not found
            atom_to_residue_list.append(res_idx)
            residue_to_atoms[res_idx].append(atom_idx)

        atom_to_residue = torch.tensor(atom_to_residue_list, dtype=torch.long)

        # Max atoms per residue
        max_atoms_per_res = max(len(atoms) for atoms in residue_to_atoms.values()) if residue_to_atoms else 1

        # Build residue -> atoms mapping
        residue_atom_indices = torch.full((num_residues, max_atoms_per_res), -1, dtype=torch.long)
        residue_atom_mask = torch.zeros(num_residues, max_atoms_per_res, dtype=torch.bool)
        num_atoms_per_residue = torch.zeros(num_residues, dtype=torch.long)

        for res_idx in range(num_residues):
            atom_indices = residue_to_atoms.get(res_idx, [])
            n_atoms = len(atom_indices)
            num_atoms_per_residue[res_idx] = n_atoms
            for j, atom_idx in enumerate(atom_indices):
                if j < max_atoms_per_res:
                    residue_atom_indices[res_idx, j] = atom_idx
                    residue_atom_mask[res_idx, j] = True

        return atom_to_residue, residue_atom_indices, residue_atom_mask, num_atoms_per_residue


def extract_hierarchical_features(
    pdb_path: str,
    include_residue_vectors: bool = False,
    esmc_model: str = "esmc_600m",
    esm3_model: str = "esm3-open",
    esm_device: str = "cuda",
) -> HierarchicalProteinData:
    """
    Convenience function to extract hierarchical features.

    Args:
        pdb_path: Path to PDB file
        include_residue_vectors: Include residue vector features
        esmc_model: ESMC model name
        esm3_model: ESM3 model name
        esm_device: Device for ESM models

    Returns:
        HierarchicalProteinData with all features including ESM embeddings
    """
    featurizer = HierarchicalFeaturizer(
        include_residue_vectors=include_residue_vectors,
        esmc_model=esmc_model,
        esm3_model=esm3_model,
        esm_device=esm_device,
    )
    return featurizer.featurize(pdb_path)
