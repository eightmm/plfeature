# plfeature Examples

This directory contains example files and usage demonstrations for the plfeature package.

## Files

- **10gs_protein.pdb**: Example protein structure (Glutathione S-transferase, 416 residues, 2 chains)
- **10gs_ligand.sdf**: Example small molecule ligand
- **usage_example.ipynb**: Interactive Jupyter notebook demonstrating the API

## Quick Start

### Interactive Tutorial (Recommended)

Open the Jupyter notebook for a complete, interactive tutorial:

```bash
cd example
jupyter notebook usage_example.ipynb
```

The notebook demonstrates:
1. **PDB Standardization** - Clean and standardize protein structures with PTM handling
2. **Protein Featurization** - Extract atom-level and residue-level features
3. **Molecule Featurization** - Extract features from small molecules (ligands)
4. **Working with Features** - Access and use the extracted PyTorch tensors

### Python Script Usage

```python
from plfeature import PDBStandardizer, ProteinFeaturizer, MoleculeFeaturizer

# 1. Standardize PDB file
standardizer = PDBStandardizer(
    remove_hydrogens=True,
    ptm_handling='base_aa'  # 'base_aa', 'preserve', or 'remove'
)
standardizer.standardize('10gs_protein.pdb', '10gs_standardized.pdb')

# 2. Extract protein features
# Residue-level
residue_featurizer = ProteinFeaturizer.ResidueFeaturizer('10gs_standardized.pdb')
sequences = residue_featurizer.get_sequence_by_chain()
res_features = residue_featurizer.get_features()

# Atom-level
atom_featurizer = ProteinFeaturizer.AtomFeaturizer('10gs_standardized.pdb')
atom_features = atom_featurizer.get_features()

# 3. Extract ligand features
mol_featurizer = MoleculeFeaturizer('10gs_ligand.sdf')
mol_features = mol_featurizer.get_features()
```

## Feature Details

### Residue-Level Features

**Scalar Features:**
- One-hot encoded amino acid types (21 classes: 20 standard + UNK)
- N/C-terminal flags
- Intra-residue distances
- Dihedral angles (φ, ψ, ω, χ1-5)
- SASA (Solvent Accessible Surface Area)
- Forward/reverse connection distances

**Vector Features:**
- Intra-residue vectors
- Forward/reverse connection vectors
- Local coordinate frames

**Edge Features:**
- Inter-residue distances (CA-CA, SC-SC, CA-SC, SC-CA)
- Relative position encoding
- Interaction vectors

### PTM Handling

Post-translational modifications (PTMs) are handled during featurization:
- PTM residues are encoded as **UNK (unknown, index 20)**
- Only **backbone atoms (N, CA, C, O) + CB** are kept
- PTMs appear as **'X'** in sequence strings

The standardizer offers three PTM handling modes:
- `'base_aa'`: Convert to base amino acids (e.g., SEP → SER)
- `'preserve'`: Keep all PTM atoms intact
- `'remove'`: Remove PTM residues entirely

## Using Your Own Files

Replace the example files with your own:

```python
# Your protein structure
standardizer.standardize('your_protein.pdb', 'standardized.pdb')
featurizer = ProteinFeaturizer.ResidueFeaturizer('standardized.pdb')

# Your ligand
mol_featurizer = MoleculeFeaturizer('your_ligand.sdf')
```

## Troubleshooting

### Missing Dependencies

```bash
pip install rdkit-pypi torch numpy pandas freesasa
```

### FreeSASA Warnings

Warnings about unknown atoms from FreeSASA are normal and handled internally.

## More Information

- Main [README.md](../README.md) for installation and API overview
- [usage_example.ipynb](usage_example.ipynb) for interactive tutorial
