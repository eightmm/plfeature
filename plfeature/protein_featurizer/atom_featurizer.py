"""
Atom-level protein featurizer for extracting atomic features and SASA.
"""

import torch
import numpy as np
from typing import Dict, Tuple, Optional, List
import freesasa


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

amino_acid_mapping = {'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C',
                      'GLN': 'Q', 'GLU': 'E', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I',
                      'LEU': 'L', 'LYS': 'K', 'MET': 'M', 'PHE': 'F', 'PRO': 'P',
                      'SER': 'S', 'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V'}

amino_acid_mapping_reverse = {v: k for k, v in amino_acid_mapping.items()}
amino_acid_3_to_int = {amino_acid_mapping_reverse[k]: i for i, k in enumerate(sorted(amino_acid_mapping_reverse.keys()))}
amino_acid_1_to_int = {k: i for i, k in enumerate(sorted(amino_acid_mapping_reverse.keys()))}

aa_letter = list(amino_acid_mapping.keys())

res_token = {
    'ALA': 0,  'ARG': 1,  'ASN': 2,  'ASP': 3,  'CYS': 4,
    'GLN': 5,  'GLU': 6,  'GLY': 7,  'HIS': 8,  'ILE': 9,
    'LEU': 10, 'LYS': 11, 'MET': 12, 'PHE': 13, 'PRO': 14,
    'SER': 15, 'THR': 16, 'TRP': 17, 'TYR': 18, 'VAL': 19,
    'UNK': 20, 'METAL': 21,  # UNK: Unknown residue
}

# Element type mapping (for atom_element feature)
ELEMENT_TYPES = {
    'H': 0,   # Hydrogen (if kept)
    'C': 1,   # Carbon
    'N': 2,   # Nitrogen
    'O': 3,   # Oxygen
    'S': 4,   # Sulfur
    'P': 5,   # Phosphorus
    'SE': 6,  # Selenium
    # Metals
    'CA': 7,  # Calcium
    'MG': 8,  # Magnesium
    'ZN': 9,  # Zinc
    'FE': 10, # Iron
    'MN': 11, # Manganese
    'CU': 12, # Copper
    'CO': 13, # Cobalt
    'NI': 14, # Nickel
    'NA': 15, # Sodium
    'K': 16,  # Potassium
    'METAL': 17,  # Generic metal
    'UNK': 18,    # Unknown
}

# Atom name to element mapping for standard amino acids
ATOM_NAME_TO_ELEMENT = {
    'N': 'N', 'CA': 'C', 'C': 'C', 'O': 'O', 'CB': 'C',
    'CG': 'C', 'CG1': 'C', 'CG2': 'C',
    'CD': 'C', 'CD1': 'C', 'CD2': 'C',
    'CE': 'C', 'CE1': 'C', 'CE2': 'C', 'CE3': 'C',
    'CZ': 'C', 'CZ2': 'C', 'CZ3': 'C',
    'CH2': 'C',
    'ND1': 'N', 'ND2': 'N', 'NE': 'N', 'NE1': 'N', 'NE2': 'N',
    'NH1': 'N', 'NH2': 'N', 'NZ': 'N',
    'OD1': 'O', 'OD2': 'O', 'OE1': 'O', 'OE2': 'O',
    'OG': 'O', 'OG1': 'O', 'OH': 'O',
    'SG': 'S', 'SD': 'S',
    'P': 'P',
    'SE': 'SE',
}

res_atm_token = {
    # ALA: N, CA, C, O, CB
    ('ALA', 'N'): 0, ('ALA', 'CA'): 1, ('ALA', 'C'): 2, ('ALA', 'O'): 3, ('ALA', 'CB'): 4,
    # ARG: N, CA, C, O, CB, CG, CD, NE, CZ, NH1, NH2
    ('ARG', 'N'): 5, ('ARG', 'CA'): 6, ('ARG', 'C'): 7, ('ARG', 'O'): 8, ('ARG', 'CB'): 9, ('ARG', 'CG'): 10, ('ARG', 'CD'): 11, ('ARG', 'NE'): 12, ('ARG', 'CZ'): 13, ('ARG', 'NH1'): 14, ('ARG', 'NH2'): 15,
    # ASN: N, CA, C, O, CB, CG, OD1, ND2
    ('ASN', 'N'): 16, ('ASN', 'CA'): 17, ('ASN', 'C'): 18, ('ASN', 'O'): 19, ('ASN', 'CB'): 20, ('ASN', 'CG'): 21, ('ASN', 'OD1'): 22, ('ASN', 'ND2'): 23,
    # ASP: N, CA, C, O, CB, CG, OD1, OD2
    ('ASP', 'N'): 24, ('ASP', 'CA'): 25, ('ASP', 'C'): 26, ('ASP', 'O'): 27, ('ASP', 'CB'): 28, ('ASP', 'CG'): 29, ('ASP', 'OD1'): 30, ('ASP', 'OD2'): 31,
    # CYS: N, CA, C, O, CB, SG
    ('CYS', 'N'): 32, ('CYS', 'CA'): 33, ('CYS', 'C'): 34, ('CYS', 'O'): 35, ('CYS', 'CB'): 36, ('CYS', 'SG'): 37,
    # GLN: N, CA, C, O, CB, CG, CD, OE1, NE2
    ('GLN', 'N'): 38, ('GLN', 'CA'): 39, ('GLN', 'C'): 40, ('GLN', 'O'): 41, ('GLN', 'CB'): 42, ('GLN', 'CG'): 43, ('GLN', 'CD'): 44, ('GLN', 'OE1'): 45, ('GLN', 'NE2'): 46,
    # GLU: N, CA, C, O, CB, CG, CD, OE1, OE2
    ('GLU', 'N'): 47, ('GLU', 'CA'): 48, ('GLU', 'C'): 49, ('GLU', 'O'): 50, ('GLU', 'CB'): 51, ('GLU', 'CG'): 52, ('GLU', 'CD'): 53, ('GLU', 'OE1'): 54, ('GLU', 'OE2'): 55,
    # GLY: N, CA, C, O
    ('GLY', 'N'): 56, ('GLY', 'CA'): 57, ('GLY', 'C'): 58, ('GLY', 'O'): 59,
    # HIS: N, CA, C, O, CB, CG, ND1, CD2, CE1, NE2
    ('HIS', 'N'): 60, ('HIS', 'CA'): 61, ('HIS', 'C'): 62, ('HIS', 'O'): 63, ('HIS', 'CB'): 64, ('HIS', 'CG'): 65, ('HIS', 'ND1'): 66, ('HIS', 'CD2'): 67, ('HIS', 'CE1'): 68, ('HIS', 'NE2'): 69,
    # ILE: N, CA, C, O, CB, CG1, CG2, CD1
    ('ILE', 'N'): 70, ('ILE', 'CA'): 71, ('ILE', 'C'): 72, ('ILE', 'O'): 73, ('ILE', 'CB'): 74, ('ILE', 'CG1'): 75, ('ILE', 'CG2'): 76, ('ILE', 'CD1'): 77,
    # LEU: N, CA, C, O, CB, CG, CD1, CD2
    ('LEU', 'N'): 78, ('LEU', 'CA'): 79, ('LEU', 'C'): 80, ('LEU', 'O'): 81, ('LEU', 'CB'): 82, ('LEU', 'CG'): 83, ('LEU', 'CD1'): 84, ('LEU', 'CD2'): 85,
    # LYS: N, CA, C, O, CB, CG, CD, CE, NZ
    ('LYS', 'N'): 86, ('LYS', 'CA'): 87, ('LYS', 'C'): 88, ('LYS', 'O'): 89, ('LYS', 'CB'): 90, ('LYS', 'CG'): 91, ('LYS', 'CD'): 92, ('LYS', 'CE'): 93, ('LYS', 'NZ'): 94,
    # MET: N, CA, C, O, CB, CG, SD, CE
    ('MET', 'N'): 95, ('MET', 'CA'): 96, ('MET', 'C'): 97, ('MET', 'O'): 98, ('MET', 'CB'): 99, ('MET', 'CG'): 100, ('MET', 'SD'): 101, ('MET', 'CE'): 102,
    # PHE: N, CA, C, O, CB, CG, CD1, CD2, CE1, CE2, CZ
    ('PHE', 'N'): 103, ('PHE', 'CA'): 104, ('PHE', 'C'): 105, ('PHE', 'O'): 106, ('PHE', 'CB'): 107, ('PHE', 'CG'): 108, ('PHE', 'CD1'): 109, ('PHE', 'CD2'): 110, ('PHE', 'CE1'): 111, ('PHE', 'CE2'): 112, ('PHE', 'CZ'): 113,
    # PRO: N, CA, C, O, CB, CG, CD
    ('PRO', 'N'): 114, ('PRO', 'CA'): 115, ('PRO', 'C'): 116, ('PRO', 'O'): 117, ('PRO', 'CB'): 118, ('PRO', 'CG'): 119, ('PRO', 'CD'): 120,
    # SER: N, CA, C, O, CB, OG
    ('SER', 'N'): 121, ('SER', 'CA'): 122, ('SER', 'C'): 123, ('SER', 'O'): 124, ('SER', 'CB'): 125, ('SER', 'OG'): 126,
    # THR: N, CA, C, O, CB, OG1, CG2
    ('THR', 'N'): 127, ('THR', 'CA'): 128, ('THR', 'C'): 129, ('THR', 'O'): 130, ('THR', 'CB'): 131, ('THR', 'OG1'): 132, ('THR', 'CG2'): 133,
    # TRP: N, CA, C, O, CB, CG, CD1, CD2, NE1, CE2, CE3, CZ2, CZ3, CH2
    ('TRP', 'N'): 134, ('TRP', 'CA'): 135, ('TRP', 'C'): 136, ('TRP', 'O'): 137, ('TRP', 'CB'): 138, ('TRP', 'CG'): 139, ('TRP', 'CD1'): 140, ('TRP', 'CD2'): 141, ('TRP', 'NE1'): 142, ('TRP', 'CE2'): 143, ('TRP', 'CE3'): 144, ('TRP', 'CZ2'): 145, ('TRP', 'CZ3'): 146, ('TRP', 'CH2'): 147,
    # TYR: N, CA, C, O, CB, CG, CD1, CD2, CE1, CE2, CZ, OH
    ('TYR', 'N'): 148, ('TYR', 'CA'): 149, ('TYR', 'C'): 150, ('TYR', 'O'): 151, ('TYR', 'CB'): 152, ('TYR', 'CG'): 153, ('TYR', 'CD1'): 154, ('TYR', 'CD2'): 155, ('TYR', 'CE1'): 156, ('TYR', 'CE2'): 157, ('TYR', 'CZ'): 158, ('TYR', 'OH'): 159,
    # VAL: N, CA, C, O, CB, CG1, CG2
    ('VAL', 'N'): 160, ('VAL', 'CA'): 161, ('VAL', 'C'): 162, ('VAL', 'O'): 163, ('VAL', 'CB'): 164, ('VAL', 'CG1'): 165, ('VAL', 'CG2'): 166,
    # UNK: N, CA, C, O, CB (unknown residue, backbone + CB only)
    ('UNK', 'N'): 167, ('UNK', 'CA'): 168, ('UNK', 'C'): 169, ('UNK', 'O'): 170, ('UNK', 'CB'): 171,
    # Metal ions (biologically important metals with distinct roles)
    ('METAL', 'CA'): 175,  # Calcium - signaling, structural
    ('METAL', 'MG'): 176,  # Magnesium - enzymatic cofactor, ATP binding
    ('METAL', 'ZN'): 177,  # Zinc - structural (zinc fingers), catalytic
    ('METAL', 'FE'): 178,  # Iron - electron transfer, oxygen binding
    ('METAL', 'MN'): 179,  # Manganese - photosynthesis, oxidoreductases
    ('METAL', 'CU'): 180,  # Copper - electron transfer, oxidases
    ('METAL', 'CO'): 181,  # Cobalt - vitamin B12, some enzymes
    ('METAL', 'NI'): 182,  # Nickel - urease, hydrogenases
    ('METAL', 'NA'): 183,  # Sodium - ion channels, osmotic balance
    ('METAL', 'K'): 184,   # Potassium - ion channels, protein stability
    ('METAL', 'METAL'): 185,  # Generic/unspecified metal
    # Special tokens
    ('UNK', 'UNK'): 186
}


class AtomFeaturizer:
    """
    Atom-level featurizer for protein structures.
    Extracts atomic features including tokens, coordinates, and SASA.
    """

    def __init__(self):
        """Initialize the atom featurizer."""
        self.res_atm_token = res_atm_token
        self.res_token = res_token
        self.aa_letter = aa_letter

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
            elif res_type in ['HIS', 'HID', 'HIE', 'HIP']:
                res_type = 'HIS'
            # Handle cysteine variants
            elif res_type in ['CYS', 'CYX', 'CYM']:
                res_type = 'CYS'
            # Handle unknown residues
            elif res_type not in self.aa_letter:
                res_type = 'XXX'
                # For non-standard residues, try to preserve key atoms
                if atom_type not in ['N', 'CA', 'C', 'O', 'CB', 'P', 'S', 'SE']:
                    # Use first character as generic atom type
                    atom_type = atom_type[0] if atom_type else 'C'

            # Get token ID
            tok = self.res_atm_token.get((res_type, atom_type), 186)  # 186 is UNK token

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
            if res_name_clean in ['HIS', 'HID', 'HIE', 'HIP']:
                res_name_clean = 'HIS'
            elif res_name_clean in ['CYS', 'CYX', 'CYM']:
                res_name_clean = 'CYS'
            elif res_name_clean not in self.aa_letter:
                res_name_clean = 'XXX'

            res_tok = self.res_token.get(res_name_clean, 20)  # 20 is XXX token
            residue_tokens.append(res_tok)

            # Map element symbol to element type integer
            # Handle special cases for metals and 2-letter elements
            if element in ELEMENT_TYPES:
                element_type = ELEMENT_TYPES[element]
            elif element in ['CA', 'MG', 'ZN', 'FE', 'MN', 'CU', 'CO', 'NI', 'NA', 'K']:
                # Metal ions
                element_type = ELEMENT_TYPES.get(element, ELEMENT_TYPES['METAL'])
            elif len(element) == 1 and element in ['C', 'N', 'O', 'S', 'P', 'H']:
                # Single letter elements
                element_type = ELEMENT_TYPES[element]
            else:
                # Unknown element - try to infer from atom name
                atom_name_clean = atom_name.strip()
                fallback_element = ATOM_NAME_TO_ELEMENT.get(atom_name_clean, None)
                if fallback_element:
                    element_type = ELEMENT_TYPES.get(fallback_element, ELEMENT_TYPES['UNK'])
                else:
                    element_type = ELEMENT_TYPES['UNK']

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


