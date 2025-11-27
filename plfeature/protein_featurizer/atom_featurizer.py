"""
Atom-level protein featurizer for extracting atomic features and SASA.
"""

import torch
import numpy as np
from typing import Dict, Tuple, Optional, List
import freesasa

from ..constants import (
    # Amino acid mappings
    AMINO_ACID_3TO1,
    AMINO_ACID_1TO3,
    AMINO_ACID_3_TO_INT,
    AMINO_ACID_1_TO_INT,
    AMINO_ACID_LETTERS,
    # Residue tokens
    RESIDUE_TOKEN,
    RESIDUE_ATOM_TOKEN,
    UNK_TOKEN,
    # Histidine/Cysteine variants
    HISTIDINE_VARIANTS,
    CYSTEINE_VARIANTS,
    # Element mappings
    PROTEIN_ELEMENT_TYPES,
    ATOM_NAME_TO_ELEMENT,
)


####################################################################################################
################################   UTILITY FUNCTIONS   #############################################
####################################################################################################

def is_atom_record(line: str) -> bool:
    """
    Check if a PDB line is an ATOM record.

    Args:
        line: PDB format line

    Returns:
        True if the line is an ATOM record
    """
    if len(line) < 6:
        return False
    record_type = line[:6].strip()
    return record_type == 'ATOM'


def is_hetatm_record(line: str) -> bool:
    """
    Check if a PDB line is a HETATM record.

    Args:
        line: PDB format line

    Returns:
        True if the line is a HETATM record
    """
    if len(line) < 6:
        return False
    record_type = line[:6].strip()
    return record_type == 'HETATM'


def is_hydrogen(line: str) -> bool:
    """
    Check if atom is hydrogen based on PDB line.

    Args:
        line: PDB format line

    Returns:
        True if the atom is hydrogen
    """
    if len(line) < 14:
        return False
    # Check element column (77-78) first
    if len(line) > 77:
        element = line[76:78].strip()
        if element and element.upper() == 'H':
            return True
    # Fallback: check atom name (column 13-16)
    if len(line) > 13:
        atom_name = line[12:16].strip()
        if atom_name and atom_name[0] == 'H':
            return True
    return False


def parse_pdb_atom_line(line: str) -> Tuple[str, str, str, int, str, Tuple[float, float, float], str]:
    """
    Parse a PDB ATOM/HETATM line into components.

    Args:
        line: PDB format line

    Returns:
        Tuple of (record_type, atom_name, res_name, res_num, chain_id, coordinates, element)
    """
    record_type = line[:6].strip()
    atom_name = line[12:16].strip()
    res_name = line[17:20].strip()
    chain_id = line[21] if len(line) > 21 else ' '
    res_num = int(line[22:26]) if len(line) > 26 else 0

    # Parse element symbol (columns 77-78 in PDB format)
    element = ''
    if len(line) > 77:
        element = line[76:78].strip().upper()

    # Fallback: infer from atom name if element not present
    if not element and atom_name:
        # Remove digits and special characters
        element = ''.join(c for c in atom_name if c.isalpha())
        if element:
            # Take first 1-2 characters
            if len(element) >= 2 and element[:2] in ['CA', 'MG', 'ZN', 'FE', 'MN', 'CU', 'CO', 'NI', 'NA', 'CL', 'BR', 'SE']:
                element = element[:2].upper()
            else:
                element = element[0].upper()

    # Parse coordinates
    try:
        x = float(line[30:38]) if len(line) > 38 else 0.0
        y = float(line[38:46]) if len(line) > 46 else 0.0
        z = float(line[46:54]) if len(line) > 54 else 0.0
        coordinates = (x, y, z)
    except (ValueError, IndexError):
        coordinates = (0.0, 0.0, 0.0)

    return record_type, atom_name, res_name, res_num, chain_id, coordinates, element

####################################################################################################
################################       PROTEIN      ################################################
####################################################################################################


class AtomFeaturizer:
    """
    Atom-level featurizer for protein structures.
    Extracts atomic features including tokens, coordinates, and SASA.
    """

    def __init__(self):
        """Initialize the atom featurizer."""
        self.res_atm_token = RESIDUE_ATOM_TOKEN
        self.res_token = RESIDUE_TOKEN
        self.aa_letter = AMINO_ACID_LETTERS

    def get_protein_atom_features(self, pdb_file: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extract atom-level features from PDB file.

        Args:
            pdb_file: Path to PDB file

        Returns:
            Tuple of (token, coord):
                - token: torch.Tensor of shape [n_atoms] with atom type tokens
                - coord: torch.Tensor of shape [n_atoms, 3] with 3D coordinates
        """
        token, coord = [], []

        with open(pdb_file, 'r') as file:
            lines = file.readlines()

        for line in lines:
            # Use unified parsing functions
            if not (is_atom_record(line) or is_hetatm_record(line)):
                continue

            # Skip hydrogens
            if is_hydrogen(line):
                continue

            # Parse line components (now includes element)
            record_type, atom_type, res_type, res_num, chain_id, xyz, element = parse_pdb_atom_line(line)

            # Skip water molecules
            if res_type == 'HOH':
                continue

            # Skip terminal oxygen and modified residues
            if atom_type == 'OXT' or res_type in ['LLP', 'PTR']:
                continue

            # Handle metal ions - check if atom name matches first 2 chars of residue
            if len(atom_type) >= 2 and len(res_type) >= 2 and atom_type[:2] == res_type[:2]:
                # Metal ion detected - keep specific metal type
                res_type = 'METAL'
                # atom_type stays as specific metal (CA, MG, ZN, FE, etc.)
            # Handle histidine variants
            elif res_type in HISTIDINE_VARIANTS:
                res_type = 'HIS'
            # Handle cysteine variants
            elif res_type in CYSTEINE_VARIANTS:
                res_type = 'CYS'
            # Handle unknown residues
            elif res_type not in self.aa_letter:
                res_type = 'XXX'
                # For non-standard residues, try to preserve key atoms
                if atom_type not in ['N', 'CA', 'C', 'O', 'CB', 'P', 'S', 'SE']:
                    # Use first character as generic atom type
                    atom_type = atom_type[0] if atom_type else 'C'

            # Get token ID
            tok = self.res_atm_token.get((res_type, atom_type), UNK_TOKEN)

            token.append(tok)
            coord.append(xyz)

        token = torch.tensor(token, dtype=torch.long)
        coord = torch.tensor(coord, dtype=torch.float32)

        return token, coord

    def get_atom_sasa(self, pdb_file: str) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Calculate atom-level SASA using FreeSASA.

        Args:
            pdb_file: Path to PDB file

        Returns:
            Tuple of (atom_sasa, atom_info):
                - atom_sasa: torch.Tensor of shape [n_atoms] with SASA values
                - atom_info: Dictionary containing:
                    - 'residue_name': Residue names for each atom
                    - 'residue_number': Residue numbers
                    - 'atom_name': Atom names
                    - 'chain_label': Chain labels
                    - 'radius': Atomic radii
        """
        # Calculate SASA using FreeSASA
        structure = freesasa.Structure(pdb_file)
        result = freesasa.calc(structure)

        n_atoms = result.nAtoms()

        atom_sasa = []
        residue_names = []
        residue_numbers = []
        atom_names = []
        chain_labels = []
        radii = []

        for i in range(n_atoms):
            # Get SASA value
            sasa = result.atomArea(i)
            atom_sasa.append(sasa)

            # Get atom information
            residue_names.append(structure.residueName(i))
            residue_numbers.append(int(structure.residueNumber(i)))
            atom_names.append(structure.atomName(i).strip())
            chain_labels.append(structure.chainLabel(i))
            radii.append(structure.radius(i))

        # Convert to tensors
        atom_sasa = torch.tensor(atom_sasa, dtype=torch.float32)

        atom_info = {
            'residue_name': residue_names,
            'residue_number': torch.tensor(residue_numbers, dtype=torch.long),
            'atom_name': atom_names,
            'chain_label': chain_labels,
            'radius': torch.tensor(radii, dtype=torch.float32)
        }

        return atom_sasa, atom_info

    def get_all_atom_features(self, pdb_file: str) -> Dict[str, torch.Tensor]:
        """
        Get all atom-level features including tokens, coordinates, and SASA.

        Args:
            pdb_file: Path to PDB file

        Returns:
            Dictionary containing:
                - 'token': Atom type tokens [n_atoms]
                - 'coord': 3D coordinates [n_atoms, 3]
                - 'sasa': SASA values [n_atoms]
                - 'residue_token': Residue type for each atom [n_atoms]
                - 'atom_element': Element type for each atom [n_atoms]
                - 'radius': Atomic radii [n_atoms]
        """
        # Get basic atom features
        token, coord = self.get_protein_atom_features(pdb_file)

        # Get SASA features
        atom_sasa, atom_info = self.get_atom_sasa(pdb_file)

        # Parse PDB again to get element information directly
        residue_tokens = []
        atom_elements = []

        with open(pdb_file, 'r') as file:
            lines = file.readlines()

        for line in lines:
            # Use same filtering as get_protein_atom_features
            if not (is_atom_record(line) or is_hetatm_record(line)):
                continue
            if is_hydrogen(line):
                continue

            # Parse with element
            record_type, atom_name, res_name, res_num, chain_id, xyz, element = parse_pdb_atom_line(line)

            # Skip water
            if res_name == 'HOH':
                continue

            # Skip same atoms as get_protein_atom_features
            if atom_name == 'OXT' or res_name in ['LLP', 'PTR']:
                continue

            # Handle residue name normalization
            res_name_clean = res_name.strip()
            if res_name_clean in HISTIDINE_VARIANTS:
                res_name_clean = 'HIS'
            elif res_name_clean in CYSTEINE_VARIANTS:
                res_name_clean = 'CYS'
            elif res_name_clean not in self.aa_letter:
                res_name_clean = 'XXX'

            res_tok = self.res_token.get(res_name_clean, RESIDUE_TOKEN['UNK'])
            residue_tokens.append(res_tok)

            # Map element symbol to element type integer
            # Handle special cases for metals and 2-letter elements
            if element in PROTEIN_ELEMENT_TYPES:
                element_type = PROTEIN_ELEMENT_TYPES[element]
            elif element in ['CA', 'MG', 'ZN', 'FE', 'MN', 'CU', 'CO', 'NI', 'NA', 'K']:
                # Metal ions
                element_type = PROTEIN_ELEMENT_TYPES.get(element, PROTEIN_ELEMENT_TYPES['METAL'])
            elif len(element) == 1 and element in ['C', 'N', 'O', 'S', 'P', 'H']:
                # Single letter elements
                element_type = PROTEIN_ELEMENT_TYPES[element]
            else:
                # Unknown element - try to infer from atom name
                atom_name_clean = atom_name.strip()
                fallback_element = ATOM_NAME_TO_ELEMENT.get(atom_name_clean, None)
                if fallback_element:
                    element_type = PROTEIN_ELEMENT_TYPES.get(fallback_element, PROTEIN_ELEMENT_TYPES['UNK'])
                else:
                    element_type = PROTEIN_ELEMENT_TYPES['UNK']

            atom_elements.append(element_type)

        # Ensure all tensors have the same length
        min_len = min(len(token), len(atom_sasa))

        features = {
            'token': token[:min_len],
            'coord': coord[:min_len],
            'sasa': atom_sasa[:min_len],
            'residue_token': torch.tensor(residue_tokens[:min_len], dtype=torch.long),
            'atom_element': torch.tensor(atom_elements[:min_len], dtype=torch.long),
            'radius': atom_info['radius'][:min_len] if len(atom_info['radius']) > min_len else atom_info['radius'],
            'metadata': {
                'n_atoms': min_len,
                'residue_names': atom_info['residue_name'][:min_len],
                'residue_numbers': atom_info['residue_number'][:min_len],
                'atom_names': atom_info['atom_name'][:min_len],
                'chain_labels': atom_info['chain_label'][:min_len]
            }
        }

        return features

    def get_residue_aggregated_features(self, pdb_file: str) -> Dict[str, torch.Tensor]:
        """
        Get residue-level features by aggregating atom features.

        Args:
            pdb_file: Path to PDB file

        Returns:
            Dictionary with residue-aggregated features
        """
        # Get all atom features
        atom_features = self.get_all_atom_features(pdb_file)

        # Group by residue
        residue_numbers = atom_features['metadata']['residue_numbers']
        unique_residues = torch.unique(residue_numbers)

        residue_features = {
            'residue_token': [],
            'center_of_mass': [],
            'total_sasa': [],
            'mean_sasa': [],
            'n_atoms': []
        }

        for res_num in unique_residues:
            mask = residue_numbers == res_num

            # Get residue token (should be same for all atoms in residue)
            res_tokens = atom_features['residue_token'][mask]
            residue_features['residue_token'].append(res_tokens[0])

            # Calculate center of mass
            coords = atom_features['coord'][mask]
            center_of_mass = coords.mean(dim=0)
            residue_features['center_of_mass'].append(center_of_mass)

            # Aggregate SASA
            sasa = atom_features['sasa'][mask]
            residue_features['total_sasa'].append(sasa.sum())
            residue_features['mean_sasa'].append(sasa.mean())

            # Count atoms
            residue_features['n_atoms'].append(mask.sum())

        # Convert to tensors
        for key in residue_features:
            residue_features[key] = torch.stack(residue_features[key]) if key == 'center_of_mass' else torch.tensor(residue_features[key])

        return residue_features


# Convenience function for direct use
def get_protein_atom_features(pdb_file: str) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Extract atom-level features from PDB file.

    Args:
        pdb_file: Path to PDB file

    Returns:
        Tuple of (token, coord)
    """
    featurizer = AtomFeaturizer()
    return featurizer.get_protein_atom_features(pdb_file)


def get_atom_features_with_sasa(pdb_file: str) -> Dict[str, torch.Tensor]:
    """
    Get all atom-level features including SASA.

    Args:
        pdb_file: Path to PDB file

    Returns:
        Dictionary with all atom features
    """
    featurizer = AtomFeaturizer()
    return featurizer.get_all_atom_features(pdb_file)


