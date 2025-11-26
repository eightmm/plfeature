# Molecule Features

## Molecular Descriptors

Extracts 40 normalized molecular descriptors.

### API
```python
from plfeature import MoleculeFeaturizer

featurizer = MoleculeFeaturizer("CCO")
features = featurizer.get_feature()
descriptors = features['descriptor']  # torch.Tensor [40]
```

### Descriptor List

#### Physicochemical (12)
- `mw`: Molecular weight (/1000.0)
- `logp`: Octanol-water partition coefficient ((+5)/10.0)
- `tpsa`: Topological polar surface area (/200.0)
- `n_rotatable_bonds`: Rotatable bonds (/20.0)
- `flexibility`: Rotatable bonds ratio
- `hbd`: Hydrogen bond donors (/10.0)
- `hba`: Hydrogen bond acceptors (/15.0)
- `n_atoms`: Total atoms (/100.0)
- `n_bonds`: Total bonds (/120.0)
- `n_rings`: Number of rings (/10.0)
- `n_aromatic_rings`: Aromatic rings (/8.0)
- `heteroatom_ratio`: Heteroatoms ratio

#### Topological (9)
- `balaban_j`: Balaban's J index (/5.0)
- `bertz_ct`: Bertz complexity (/2000.0)
- `chi0`, `chi1`, `chi0n`: Connectivity indices
- `hall_kier_alpha`: Hall-Kier alpha (/5.0)
- `kappa1`, `kappa2`, `kappa3`: Kappa shape indices

#### Electronic (4)
- `mol_mr`: Molar refractivity (/200.0)
- `labute_asa`: Accessible surface area (/500.0)
- `num_radical_electrons`: Radical electrons (/5.0)
- `num_valence_electrons`: Valence electrons (/500.0)

#### Structural (10)
- Ring types and counts
- `num_heteroatoms`: Total heteroatoms (/30.0)
- `formal_charge`: Sum of formal charges ((+5)/10.0)
- `n_ring_systems`: Ring systems (/8.0)
- `max_ring_size`: Maximum ring size (/12.0)
- `avg_ring_size`: Average ring size (/8.0)

#### Drug-likeness (5)
- `lipinski_violations`: Rule of 5 violations
- `passes_lipinski`: Binary flag
- `qed`: Quantitative Estimate of Drug-likeness
- `num_heavy_atoms`: Heavy atoms (/50.0)
- `frac_csp3`: Fraction of sp3 carbons

## Molecular Fingerprints

Nine fingerprint types for similarity and machine learning.

### Available Fingerprints
```python
features = featurizer.get_feature()
# Returns dictionary with:
{
    'descriptor': torch.Tensor,  # [40]
    'morgan': torch.Tensor,      # [2048]
    'maccs': torch.Tensor,       # [167]
    'rdkit': torch.Tensor,       # [2048]
    'atompair': torch.Tensor,    # [2048]
    'torsion': torch.Tensor,     # [2048]
    'avalon': torch.Tensor,      # [1024]
    'pattern': torch.Tensor,     # [2048]
    'ecfp4': torch.Tensor,       # [2048]
    'fcfp4': torch.Tensor        # [2048]
}
```

### Fingerprint Types
- **Morgan**: Circular fingerprint, radius=2
- **MACCS**: 166 predefined structural keys + padding
- **RDKit**: Path-based fingerprint
- **Atom Pair**: Atom pairs and topological distances
- **Torsion**: Four-atom linear paths
- **Avalon**: Feature-class based (requires AvalonTools)
- **Pattern**: Substructure pattern encoding
- **ECFP4**: Extended connectivity, radius=2
- **FCFP4**: Feature-based circular fingerprint

## Input Formats

```python
# From SMILES
featurizer = MoleculeFeaturizer("CCO")
featurizer = MoleculeFeaturizer("CCO", hydrogen=False)

# From RDKit mol
mol = Chem.MolFromSmiles("CCO")
featurizer = MoleculeFeaturizer(mol, hydrogen=True)
```