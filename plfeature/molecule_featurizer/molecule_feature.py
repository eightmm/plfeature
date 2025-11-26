"""
Unified molecule featurizer for molecular feature extraction.

This module provides molecular-level features (descriptors and fingerprints)
and graph-level features (node and edge features) from RDKit mol objects or SMILES.
"""

import warnings
from typing import Dict, Tuple, Union

import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import (
    AllChem,
    Descriptors,
    MACCSkeys,
    QED,
    rdFingerprintGenerator,
    rdMolDescriptors,
)
from rdkit.Chem.Pharm2D import Generate, Gobbi_Pharm2D

from .graph_featurizer import MoleculeGraphFeaturizer


class MoleculeFeaturizer:
    """
    Unified molecule featurizer for extracting molecular features and graph representations.

    This class provides methods to extract both molecular-level features (descriptors
    and fingerprints) and graph-level features (node and edge features) from RDKit
    mol objects or SMILES strings.

    Example:
        >>> featurizer = MoleculeFeaturizer()
        >>> features = featurizer.get_feature("CCO")  # Molecular descriptors
        >>> node, edge, adj = featurizer.get_graph("CCO")  # Graph features
    """

    def __init__(self):
        """Initialize the molecule featurizer."""
        self._graph_featurizer = MoleculeGraphFeaturizer()

    # =========================================================================
    # Molecule Preparation
    # =========================================================================

    @staticmethod
    def _prepare_mol(mol_or_smiles: Union[str, Chem.Mol], add_hs: bool = True) -> Chem.Mol:
        """
        Prepare molecule from SMILES string or RDKit mol object.

        Preserves 3D coordinates when adding hydrogens if the molecule has a conformer.

        Args:
            mol_or_smiles: RDKit mol object or SMILES string
            add_hs: Whether to add hydrogens

        Returns:
            RDKit mol object with optional hydrogens

        Raises:
            ValueError: If SMILES string is invalid
        """
        if isinstance(mol_or_smiles, str):
            mol = Chem.MolFromSmiles(mol_or_smiles)
            if mol is None:
                raise ValueError(f"Invalid SMILES: {mol_or_smiles}")
        else:
            mol = mol_or_smiles

        if add_hs and mol is not None:
            has_3d_coords = mol.GetNumConformers() > 0
            if has_3d_coords:
                mol = Chem.AddHs(mol, addCoords=True)
            else:
                mol = Chem.AddHs(mol)

        return mol

    # =========================================================================
    # Molecular Descriptor Features
    # =========================================================================

    def get_physicochemical_features(self, mol: Chem.Mol) -> Dict[str, float]:
        """
        Extract physicochemical features from molecule.

        Returns:
            Dictionary of normalized physicochemical descriptors
        """
        features = {}

        # Basic properties
        features['mw'] = min(Descriptors.MolWt(mol) / 1000.0, 1.0)
        features['logp'] = (Descriptors.MolLogP(mol) + 5) / 10.0
        features['tpsa'] = min(Descriptors.TPSA(mol) / 200.0, 1.0)

        # Flexibility
        n_bonds = mol.GetNumBonds()
        features['n_rotatable_bonds'] = min(rdMolDescriptors.CalcNumRotatableBonds(mol) / 20.0, 1.0)
        features['flexibility'] = min(
            rdMolDescriptors.CalcNumRotatableBonds(mol) / n_bonds if n_bonds > 0 else 0, 1.0
        )

        # H-bonding
        features['hbd'] = min(rdMolDescriptors.CalcNumHBD(mol) / 10.0, 1.0)
        features['hba'] = min(rdMolDescriptors.CalcNumHBA(mol) / 15.0, 1.0)

        # Atom/bond counts
        features['n_atoms'] = min(mol.GetNumAtoms() / 100.0, 1.0)
        features['n_bonds'] = min(n_bonds / 120.0, 1.0)
        features['n_rings'] = min(rdMolDescriptors.CalcNumRings(mol) / 10.0, 1.0)
        features['n_aromatic_rings'] = min(rdMolDescriptors.CalcNumAromaticRings(mol) / 8.0, 1.0)

        # Heteroatom ratio
        n_heteroatoms = sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() not in [1, 6])
        features['heteroatom_ratio'] = n_heteroatoms / mol.GetNumAtoms()

        # Topological indices
        features['balaban_j'] = min(Descriptors.BalabanJ(mol) / 5.0, 1.0)
        features['bertz_ct'] = min(Descriptors.BertzCT(mol) / 2000.0, 1.0)
        features['chi0'] = min(Descriptors.Chi0(mol) / 50.0, 1.0)
        features['chi1'] = min(Descriptors.Chi1(mol) / 30.0, 1.0)
        features['chi0n'] = min(Descriptors.Chi0n(mol) / 50.0, 1.0)

        hka = Descriptors.HallKierAlpha(mol)
        features['hall_kier_alpha'] = min(abs(hka) / 5.0, 1.0) if hka != -1 else 0.0

        features['kappa1'] = min(Descriptors.Kappa1(mol) / 50.0, 1.0)
        features['kappa2'] = min(Descriptors.Kappa2(mol) / 20.0, 1.0)
        features['kappa3'] = min(Descriptors.Kappa3(mol) / 10.0, 1.0)

        # Electronic properties
        features['mol_mr'] = min(Descriptors.MolMR(mol) / 200.0, 1.0)
        features['labute_asa'] = min(Descriptors.LabuteASA(mol) / 500.0, 1.0)
        features['num_radical_electrons'] = min(Descriptors.NumRadicalElectrons(mol) / 5.0, 1.0)
        features['num_valence_electrons'] = min(Descriptors.NumValenceElectrons(mol) / 500.0, 1.0)

        # Ring complexity
        features['num_saturated_rings'] = min(rdMolDescriptors.CalcNumSaturatedRings(mol) / 10.0, 1.0)
        features['num_aliphatic_rings'] = min(rdMolDescriptors.CalcNumAliphaticRings(mol) / 10.0, 1.0)
        features['num_saturated_heterocycles'] = min(
            rdMolDescriptors.CalcNumSaturatedHeterocycles(mol) / 8.0, 1.0
        )
        features['num_aliphatic_heterocycles'] = min(
            rdMolDescriptors.CalcNumAliphaticHeterocycles(mol) / 8.0, 1.0
        )
        features['num_aromatic_heterocycles'] = min(
            rdMolDescriptors.CalcNumAromaticHeterocycles(mol) / 8.0, 1.0
        )

        # Atom counts
        features['num_heteroatoms'] = min(rdMolDescriptors.CalcNumHeteroatoms(mol) / 30.0, 1.0)
        total_formal_charge = sum(atom.GetFormalCharge() for atom in mol.GetAtoms())
        features['formal_charge'] = (total_formal_charge + 5) / 10.0

        return features

    def get_druglike_features(self, mol: Chem.Mol) -> Dict[str, float]:
        """
        Extract drug-likeness features from molecule.

        Returns:
            Dictionary of drug-likeness descriptors
        """
        features = {}

        # Lipinski's Rule of Five
        mw = Descriptors.MolWt(mol)
        logp = Descriptors.MolLogP(mol)
        hbd = rdMolDescriptors.CalcNumHBD(mol)
        hba = rdMolDescriptors.CalcNumHBA(mol)

        violations = sum([mw > 500, logp > 5, hbd > 5, hba > 10])
        features['lipinski_violations'] = violations / 4.0
        features['passes_lipinski'] = 1.0 if violations == 0 else 0.0

        # QED score
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            features['qed'] = QED.qed(mol)

        # Other drug-like properties
        features['num_heavy_atoms'] = min(mol.GetNumHeavyAtoms() / 50.0, 1.0)

        csp3_count = sum(
            1 for atom in mol.GetAtoms()
            if atom.GetHybridization() == Chem.HybridizationType.SP3 and atom.GetAtomicNum() == 6
        )
        total_carbons = sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() == 6)
        features['frac_csp3'] = csp3_count / total_carbons if total_carbons > 0 else 0.0

        return features

    def get_structural_features(self, mol: Chem.Mol) -> Dict[str, float]:
        """
        Extract structural features from molecule.

        Returns:
            Dictionary of structural descriptors
        """
        features = {}

        ring_info = mol.GetRingInfo()
        atom_rings = ring_info.AtomRings()

        features['n_ring_systems'] = min(len(atom_rings) / 8.0, 1.0)

        ring_sizes = [len(ring) for ring in atom_rings]
        if ring_sizes:
            features['max_ring_size'] = min(max(ring_sizes) / 12.0, 1.0)
            features['avg_ring_size'] = min(np.mean(ring_sizes) / 8.0, 1.0)
        else:
            features['max_ring_size'] = 0.0
            features['avg_ring_size'] = 0.0

        return features

    # =========================================================================
    # Fingerprint Features
    # =========================================================================

    def get_fingerprints(self, mol: Chem.Mol) -> Dict[str, torch.Tensor]:
        """
        Extract various molecular fingerprints.

        Returns:
            Dictionary of fingerprint tensors
        """
        fingerprints = {}

        # MACCS keys
        fingerprints['maccs'] = torch.tensor(
            MACCSkeys.GenMACCSKeys(mol).ToList(), dtype=torch.float32
        )

        # Morgan fingerprints
        morgan_gen = rdFingerprintGenerator.GetMorganGenerator(
            radius=2, fpSize=2048, countSimulation=True, includeChirality=True
        )
        fingerprints['morgan'] = torch.from_numpy(morgan_gen.GetFingerprintAsNumPy(mol)).float()
        fingerprints['morgan_count'] = torch.from_numpy(
            morgan_gen.GetCountFingerprintAsNumPy(mol)
        ).float()

        # Feature Morgan
        feature_morgan_gen = rdFingerprintGenerator.GetMorganGenerator(
            radius=2,
            fpSize=2048,
            atomInvariantsGenerator=rdFingerprintGenerator.GetMorganFeatureAtomInvGen(),
            countSimulation=True,
        )
        fingerprints['feature_morgan'] = torch.from_numpy(
            feature_morgan_gen.GetFingerprintAsNumPy(mol)
        ).float()

        # RDKit fingerprint
        rdkit_gen = rdFingerprintGenerator.GetRDKitFPGenerator(
            minPath=1, maxPath=7, fpSize=2048, countSimulation=True,
            branchedPaths=True, useBondOrder=True,
        )
        fingerprints['rdkit'] = torch.from_numpy(rdkit_gen.GetFingerprintAsNumPy(mol)).float()

        # Atom pair fingerprint
        ap_gen = rdFingerprintGenerator.GetAtomPairGenerator(
            minDistance=1, maxDistance=8, fpSize=2048, countSimulation=True
        )
        fingerprints['atom_pair'] = torch.from_numpy(ap_gen.GetFingerprintAsNumPy(mol)).float()

        # Topological torsion
        tt_gen = rdFingerprintGenerator.GetTopologicalTorsionGenerator(
            torsionAtomCount=4, fpSize=2048, countSimulation=True
        )
        fingerprints['topological_torsion'] = torch.from_numpy(
            tt_gen.GetFingerprintAsNumPy(mol)
        ).float()

        # Pharmacophore 2D
        pharm_fp = Generate.Gen2DFingerprint(mol, Gobbi_Pharm2D.factory)
        bit_vector = torch.zeros(1024)
        for bit_id in pharm_fp.GetOnBits():
            if bit_id < 1024:
                bit_vector[bit_id] = 1.0
        fingerprints['pharmacophore2d'] = bit_vector.float()

        return fingerprints

    # =========================================================================
    # Main Feature Extraction Methods
    # =========================================================================

    def get_feature(self, mol_or_smiles: Union[str, Chem.Mol], add_hs: bool = True) -> Dict:
        """
        Extract all molecular-level features including descriptors and fingerprints.

        Args:
            mol_or_smiles: RDKit mol object or SMILES string
            add_hs: Whether to add hydrogens

        Returns:
            Dictionary containing:
                - descriptor: Tensor of molecular descriptors [40]
                - maccs: MACCS fingerprint [167]
                - morgan: Morgan fingerprint [2048]
                - morgan_count: Morgan count fingerprint [2048]
                - feature_morgan: Feature Morgan fingerprint [2048]
                - rdkit: RDKit fingerprint [2048]
                - atom_pair: Atom pair fingerprint [2048]
                - topological_torsion: Topological torsion fingerprint [2048]
                - pharmacophore2d: 2D pharmacophore fingerprint [1024]
        """
        mol = self._prepare_mol(mol_or_smiles, add_hs)

        physicochemical = self.get_physicochemical_features(mol)
        druglike = self.get_druglike_features(mol)
        structural = self.get_structural_features(mol)

        # Build descriptor tensor
        descriptor_keys = [
            'mw', 'logp', 'tpsa', 'n_rotatable_bonds', 'flexibility',
            'hbd', 'hba', 'n_atoms', 'n_bonds', 'n_rings', 'n_aromatic_rings',
            'heteroatom_ratio', 'balaban_j', 'bertz_ct', 'chi0', 'chi1',
            'hall_kier_alpha', 'kappa1', 'kappa2', 'kappa3', 'mol_mr',
            'labute_asa', 'num_radical_electrons', 'num_valence_electrons',
            'num_saturated_rings', 'num_aliphatic_rings', 'num_saturated_heterocycles',
            'num_aliphatic_heterocycles', 'num_aromatic_heterocycles',
            'num_heteroatoms', 'formal_charge', 'chi0n',
        ]
        druglike_keys = [
            'lipinski_violations', 'passes_lipinski', 'qed', 'num_heavy_atoms', 'frac_csp3'
        ]
        structural_keys = ['n_ring_systems', 'max_ring_size', 'avg_ring_size']

        descriptors = []
        for key in descriptor_keys:
            descriptors.append(float(physicochemical[key]))
        for key in druglike_keys:
            descriptors.append(float(druglike[key]))
        for key in structural_keys:
            descriptors.append(float(structural[key]))

        fingerprints = self.get_fingerprints(mol)

        return {
            'descriptor': torch.tensor(descriptors, dtype=torch.float32),
            'maccs': fingerprints['maccs'],
            'morgan': fingerprints['morgan'],
            'morgan_count': fingerprints['morgan_count'],
            'feature_morgan': fingerprints['feature_morgan'],
            'rdkit': fingerprints['rdkit'],
            'atom_pair': fingerprints['atom_pair'],
            'topological_torsion': fingerprints['topological_torsion'],
            'pharmacophore2d': fingerprints['pharmacophore2d'],
        }

    def get_graph(
        self, mol_or_smiles: Union[str, Chem.Mol], add_hs: bool = True
    ) -> Tuple[Dict, Dict, torch.Tensor]:
        """
        Create molecular graph with node and edge features.

        Args:
            mol_or_smiles: RDKit mol object or SMILES string
            add_hs: Whether to add hydrogens

        Returns:
            Tuple of (node_dict, edge_dict, adjacency_matrix):
                - node_dict: {'node_feats': [N, 147], 'coords': [N, 3]}
                - edge_dict: {'edges': [2, E], 'edge_feats': [E, 66]}
                - adjacency_matrix: [N, N, 66]
        """
        mol = self._prepare_mol(mol_or_smiles, add_hs)
        return self._graph_featurizer.featurize(mol)
