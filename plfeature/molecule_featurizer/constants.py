"""
Molecular constants and lookup tables for featurization.

This module contains physical constants, periodic table data, and chemical
property lookup tables used for molecular feature extraction.
"""

from rdkit import Chem

# =============================================================================
# One-hot Encoding Categories
# =============================================================================

ATOM_TYPES = ['H', 'C', 'N', 'O', 'S', 'P', 'F', 'Cl', 'Br', 'I', 'UNK']
PERIODS = list(range(5))
GROUPS = list(range(18))
DEGREES = list(range(7))
HEAVY_DEGREES = list(range(7))
VALENCES = list(range(8))
TOTAL_HS = list(range(5))

HYBRIDIZATIONS = [
    Chem.rdchem.HybridizationType.SP,
    Chem.rdchem.HybridizationType.SP2,
    Chem.rdchem.HybridizationType.SP3,
    Chem.rdchem.HybridizationType.SP3D,
    Chem.rdchem.HybridizationType.SP3D2,
    Chem.rdchem.HybridizationType.UNSPECIFIED
]

BOND_TYPES = [
    Chem.rdchem.BondType.SINGLE,
    Chem.rdchem.BondType.DOUBLE,
    Chem.rdchem.BondType.TRIPLE,
    Chem.rdchem.BondType.AROMATIC
]

BOND_STEREOS = [
    Chem.rdchem.BondStereo.STEREOANY,
    Chem.rdchem.BondStereo.STEREOCIS,
    Chem.rdchem.BondStereo.STEREOE,
    Chem.rdchem.BondStereo.STEREONONE,
    Chem.rdchem.BondStereo.STEREOTRANS,
    Chem.rdchem.BondStereo.STEREOZ
]

# =============================================================================
# Periodic Table Data
# =============================================================================

# Element symbol -> (period, group)
PERIODIC_TABLE = {
    'H': (0, 0), 'He': (0, 17),
    'Li': (1, 0), 'Be': (1, 1), 'B': (1, 12), 'C': (1, 13),
    'N': (1, 14), 'O': (1, 15), 'F': (1, 16), 'Ne': (1, 17),
    'Na': (2, 0), 'Mg': (2, 1), 'Al': (2, 12), 'Si': (2, 13),
    'P': (2, 14), 'S': (2, 15), 'Cl': (2, 16), 'Ar': (2, 17),
    'K': (3, 0), 'Ca': (3, 1), 'Sc': (3, 2), 'Ti': (3, 3),
    'V': (3, 4), 'Cr': (3, 5), 'Mn': (3, 6), 'Fe': (3, 7),
    'Co': (3, 8), 'Ni': (3, 9), 'Cu': (3, 10), 'Zn': (3, 11),
    'Ga': (3, 12), 'Ge': (3, 13), 'As': (3, 14), 'Se': (3, 15),
    'Br': (3, 16), 'Kr': (3, 17),
    'Rb': (4, 0), 'Sr': (4, 1), 'Y': (4, 2), 'Zr': (4, 3),
    'Nb': (4, 4), 'Mo': (4, 5), 'Tc': (4, 6), 'Ru': (4, 7),
    'Rh': (4, 8), 'Pd': (4, 9), 'Ag': (4, 10), 'Cd': (4, 11),
    'In': (4, 12), 'Sn': (4, 13), 'Sb': (4, 14), 'Te': (4, 15),
    'I': (4, 16), 'Xe': (4, 17)
}

# (period, group) -> Pauling electronegativity
ELECTRONEGATIVITY = {
    (0, 0): 2.20,  # H
    (1, 0): 0.98, (1, 1): 1.57, (1, 12): 2.04, (1, 13): 2.55,
    (1, 14): 3.04, (1, 15): 3.44, (1, 16): 3.98,
    (2, 0): 0.93, (2, 1): 1.31, (2, 12): 1.61, (2, 13): 1.90,
    (2, 14): 2.19, (2, 15): 2.58, (2, 16): 3.16,
    (3, 0): 0.82, (3, 1): 1.00, (3, 2): 1.36, (3, 3): 1.54,
    (3, 4): 1.63, (3, 5): 1.66, (3, 6): 1.55, (3, 7): 1.83,
    (3, 8): 1.88, (3, 9): 1.91, (3, 10): 1.90, (3, 11): 1.65,
    (3, 12): 1.81, (3, 13): 2.01, (3, 14): 2.18, (3, 15): 2.55,
    (3, 16): 2.96, (3, 17): 3.00,
    (4, 0): 0.82, (4, 1): 0.95, (4, 2): 1.22, (4, 3): 1.33,
    (4, 4): 1.60, (4, 5): 2.16, (4, 6): 1.90, (4, 7): 2.20,
    (4, 8): 2.28, (4, 9): 2.20, (4, 10): 1.93, (4, 11): 1.69,
    (4, 12): 1.78, (4, 13): 1.96, (4, 14): 2.05, (4, 15): 2.10,
    (4, 16): 2.66, (4, 17): 2.60
}

# =============================================================================
# Physical Properties by Atomic Number
# =============================================================================

# Van der Waals radii in Angstrom (Bondi radii)
VDW_RADIUS = {
    1: 1.20,   # H
    5: 1.92,   # B
    6: 1.70,   # C
    7: 1.55,   # N
    8: 1.52,   # O
    9: 1.47,   # F
    11: 2.27,  # Na
    12: 1.73,  # Mg
    14: 2.10,  # Si
    15: 1.80,  # P
    16: 1.80,  # S
    17: 1.75,  # Cl
    19: 2.75,  # K
    20: 2.31,  # Ca
    26: 2.04,  # Fe
    29: 1.40,  # Cu
    30: 1.39,  # Zn
    33: 1.85,  # As
    34: 1.90,  # Se
    35: 1.85,  # Br
    50: 2.17,  # Sn
    51: 2.06,  # Sb
    52: 2.06,  # Te
    53: 1.98,  # I
}

# Covalent radii in Angstrom
COVALENT_RADIUS = {
    1: 0.31,   # H
    5: 0.84,   # B
    6: 0.76,   # C
    7: 0.71,   # N
    8: 0.66,   # O
    9: 0.57,   # F
    11: 1.66,  # Na
    12: 1.41,  # Mg
    14: 1.11,  # Si
    15: 1.07,  # P
    16: 1.05,  # S
    17: 1.02,  # Cl
    19: 2.03,  # K
    20: 1.76,  # Ca
    26: 1.32,  # Fe
    29: 1.32,  # Cu
    30: 1.22,  # Zn
    33: 1.19,  # As
    34: 1.20,  # Se
    35: 1.20,  # Br
    50: 1.39,  # Sn
    51: 1.39,  # Sb
    52: 1.38,  # Te
    53: 1.39,  # I
}

# First ionization energy in eV
IONIZATION_ENERGY = {
    1: 13.60,  # H
    5: 8.30,   # B
    6: 11.26,  # C
    7: 14.53,  # N
    8: 13.62,  # O
    9: 17.42,  # F
    11: 5.14,  # Na
    12: 7.65,  # Mg
    14: 8.15,  # Si
    15: 10.49, # P
    16: 10.36, # S
    17: 12.97, # Cl
    19: 4.34,  # K
    20: 6.11,  # Ca
    26: 7.90,  # Fe
    29: 7.73,  # Cu
    30: 9.39,  # Zn
    33: 9.79,  # As
    34: 9.75,  # Se
    35: 11.81, # Br
    50: 7.34,  # Sn
    51: 8.61,  # Sb
    52: 9.01,  # Te
    53: 10.45, # I
}

# Atomic polarizability in Angstrom^3
POLARIZABILITY = {
    1: 0.67,   # H
    5: 3.03,   # B
    6: 1.76,   # C
    7: 1.10,   # N
    8: 0.80,   # O
    9: 0.56,   # F
    11: 24.11, # Na
    12: 10.60, # Mg
    14: 5.38,  # Si
    15: 3.63,  # P
    16: 2.90,  # S
    17: 2.18,  # Cl
    19: 43.40, # K
    20: 22.80, # Ca
    26: 8.40,  # Fe
    29: 6.20,  # Cu
    30: 5.75,  # Zn
    33: 4.31,  # As
    34: 3.77,  # Se
    35: 3.05,  # Br
    50: 7.70,  # Sn
    51: 6.60,  # Sb
    52: 5.50,  # Te
    53: 5.35,  # I
}

# Valence electrons for lone pair calculation
VALENCE_ELECTRONS = {
    1: 1,   # H
    5: 3,   # B
    6: 4,   # C
    7: 5,   # N
    8: 6,   # O
    9: 7,   # F
    11: 1,  # Na
    12: 2,  # Mg
    14: 4,  # Si
    15: 5,  # P
    16: 6,  # S
    17: 7,  # Cl
    19: 1,  # K
    20: 2,  # Ca
    26: 8,  # Fe
    29: 11, # Cu
    30: 12, # Zn
    33: 5,  # As
    34: 6,  # Se
    35: 7,  # Br
    50: 4,  # Sn
    51: 5,  # Sb
    52: 6,  # Te
    53: 7,  # I
}

# =============================================================================
# SMARTS Patterns
# =============================================================================

CHEMICAL_SMARTS = {
    "h_acceptor": "[$([O,S;H1;v2;!$(*-*=[O,N,P,S])]),$([O,S;H0;v2]),$([O,S;-]),"
                  "$([N;v3;!$(N-*=[O,N,P,S])]),n&H0&+0,"
                  "$([o,s;+0;!$([o,s]:n);!$([o,s]:c:n)])]",
    "h_donor": "[$([N;!H0;v3,v4&+1]),$([O,S;H1;+0]),n&H1&+0]",
    "e_acceptor": "[$([C,S](=[O,S,P])-[O;H1,-1])]",
    "e_donor": "[#7;+,$([N;H2&+0][$([C,a]);!$([C,a](=O))]),"
               "$([N;H1&+0]([$([C,a]);!$([C,a](=O))])[$([C,a]);!$([C,a](=O))]),"
               "$([N;H0&+0]([C;!$(C(=O))])([C;!$(C(=O))])[C;!$(C(=O))])]",
    "hydrophobic": "[C,c,S&H0&v2,F,Cl,Br,I&!$(C=[O,N,P,S])&!$(C#N);!$(C=O)]"
}

# SMARTS for rotatable bonds
ROTATABLE_BOND_SMARTS = "[!$(*#*)&!D1]-!@[!$(*#*)&!D1]"

# Bond direction types
BOND_DIRS = [
    Chem.rdchem.BondDir.NONE,
    Chem.rdchem.BondDir.BEGINWEDGE,
    Chem.rdchem.BondDir.BEGINDASH,
    Chem.rdchem.BondDir.ENDDOWNRIGHT,
    Chem.rdchem.BondDir.ENDUPRIGHT,
]

# =============================================================================
# Typical Bond Lengths (Angstrom) - for normalization reference
# =============================================================================

# Bond type -> typical length range (min, max)
TYPICAL_BOND_LENGTHS = {
    'C-C': (1.20, 1.54),   # triple to single
    'C-N': (1.16, 1.47),
    'C-O': (1.13, 1.43),
    'C-S': (1.55, 1.82),
    'C-H': (1.06, 1.12),
    'N-H': (1.00, 1.04),
    'O-H': (0.94, 0.98),
    'N-N': (1.10, 1.45),
    'N-O': (1.15, 1.40),
    'O-O': (1.21, 1.48),
    'C-F': (1.27, 1.35),
    'C-Cl': (1.60, 1.79),
    'C-Br': (1.79, 1.97),
    'C-I': (1.99, 2.16),
    'default': (1.0, 2.5),
}

# =============================================================================
# Default Values
# =============================================================================

DEFAULT_VDW_RADIUS = 1.70
DEFAULT_COVALENT_RADIUS = 0.76
DEFAULT_IONIZATION_ENERGY = 10.0
DEFAULT_POLARIZABILITY = 1.76
DEFAULT_VALENCE_ELECTRONS = 4
DEFAULT_ELECTRONEGATIVITY = 2.5

# =============================================================================
# Normalization Constants
# =============================================================================

NORM_CONSTANTS = {
    # Node feature normalization
    'atomic_mass': 200.0,
    'vdw_radius_min': 1.0,
    'vdw_radius_range': 2.0,
    'covalent_radius': 2.0,
    'ionization_energy_min': 4.0,
    'ionization_energy_range': 14.0,
    'polarizability_log_scale': 4.0,
    'lone_pairs': 3.0,
    'neighbor_en_sum': 16.0,
    'neighbor_en_diff': 3.2,
    'neighbor_mass_sum': 600.0,
    'neighbor_charge_shift': 4,
    'neighbor_charge_range': 8.0,
    'eccentricity': 20.0,
    'dist_to_special': 10.0,
    'logp_shift': 2.0,
    'logp_range': 4.0,
    'mr_max': 10.0,
    'tpsa_max': 30.0,
    'asa_max': 20.0,
    # Edge feature normalization
    'bond_length_min': 0.9,
    'bond_length_range': 1.6,
    'en_diff_max': 3.2,
    'mass_diff_max': 100.0,
    'mass_sum_max': 250.0,
    'charge_diff_max': 4.0,
    'path_length_max': 20.0,
}
