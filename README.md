# plfeature

A comprehensive Python package for extracting features from both **molecules** and **proteins** for machine learning applications, with special support for graph neural networks and protein-ligand modeling.


## 📦 Installation

```bash
pip install git+https://github.com/eightmm/plfeature.git
```

## 🚀 Quick Start

### Molecule Features
```python
from plfeature import MoleculeFeaturizer
from rdkit import Chem

# From SDF file
suppl = Chem.SDMolSupplier('molecules.sdf')
for mol in suppl:
    featurizer = MoleculeFeaturizer(mol)  # Initialize with molecule
    features = featurizer.get_feature()
    node, edge, adj = featurizer.get_graph()

# From SMILES
featurizer = MoleculeFeaturizer("CC(=O)Oc1ccccc1C(=O)O")
features = featurizer.get_feature()  # All descriptors and fingerprints
node, edge, adj = featurizer.get_graph()  # Graph representation

# With custom SMARTS patterns
custom_patterns = {
    'aromatic_nitrogen': 'n',
    'carboxyl': 'C(=O)O',
    'hydroxyl': '[OH]'
}
featurizer = MoleculeFeaturizer("c1ccncc1CCO", custom_smarts=custom_patterns)
node, edge, adj = featurizer.get_graph()
# node['node_feats'] now has 122 + 3 dimensions (base features + custom patterns)

# Without hydrogens
featurizer = MoleculeFeaturizer("c1ccncc1CCO", hydrogen=False)
features = featurizer.get_feature()  # Features without H atoms
```

### Protein Features
```python
from plfeature import ProteinFeaturizer

featurizer = ProteinFeaturizer("protein.pdb")

# Sequence extraction by chain
sequences_by_chain = featurizer.get_sequence_by_chain()  # {'A': "ACDEF...", 'B': "GHIKL..."}

# Atom-level features (node/edge format with SASA included)
atom_node, atom_edge = featurizer.get_atom_features(distance_cutoff=4.0)

# Residue-level features (node/edge format)
res_node, res_edge = featurizer.get_residue_features(distance_cutoff=8.0)
```

### PDB Standardization
```python
from plfeature import PDBStandardizer

# Default: Convert PTMs to base amino acids (for ESM models, MD simulation)
standardizer = PDBStandardizer()  # ptm_handling='base_aa' (default)
standardizer.standardize("messy.pdb", "clean.pdb")

# For protein-ligand modeling: Preserve PTM atoms
standardizer = PDBStandardizer(ptm_handling='preserve')
standardizer.standardize("protein.pdb", "protein_with_ptms.pdb")

# Remove PTM residues entirely (standard AA only)
standardizer = PDBStandardizer(ptm_handling='remove')
standardizer.standardize("protein.pdb", "protein_standard_only.pdb")

# Keep hydrogens
standardizer = PDBStandardizer(remove_hydrogens=False)
standardizer.standardize("messy.pdb", "clean.pdb")

# Or use convenience function
from plfeature import standardize_pdb
standardize_pdb("messy.pdb", "clean.pdb", ptm_handling='base_aa')

# Standardization automatically:
# - Removes water molecules
# - Removes DNA/RNA residues
# - Handles PTMs based on ptm_handling mode (see below)
# - Normalizes protonation states (HID/HIE/HIP→HIS)
# - Reorders atoms by standard definitions
# - Renumbers residues sequentially (insertion codes removed)
# - Removes hydrogens (optional)
```

## 📊 Feature Overview

### Molecules
- **Descriptors**: 40 normalized molecular properties → [Details](docs/molecular_descriptors.md)
- **Fingerprints**: 9 types including Morgan, MACCS, RDKit → [Details](docs/molecule_feature.md)
- **Graph Features**: 122D atom features, 44D bond features → [Details](docs/molecule_graph.md)

### Proteins
- **Atom Features**: 175 token types with atomic SASA → [Details](docs/protein_atom_feature.md)
- **Residue Features**: Geometry, SASA, contacts, secondary structure → [Details](docs/protein_residue_feature.md)
- **Graph Representations**: Both atom and residue-level networks
- **Standardization**: Flexible PTM handling with 3 modes (base_aa, preserve, remove)
  - Supports 15+ PTM types: phosphorylation (SEP, TPO, PTR), selenomethionine (MSE), methylation (MLY, M3L), acetylation (ALY), hydroxylation (HYP), and more
  - Compatible with ESM models, protein-ligand modeling, and MD simulations

## 🔧 Advanced Examples

### Custom SMARTS Patterns for Molecules
```python
from plfeature import MoleculeFeaturizer

# Define custom SMARTS patterns
custom_patterns = {
    'aromatic_nitrogen': 'n',
    'carboxyl': 'C(=O)O',
    'hydroxyl': '[OH]',
    'amine': '[NX3;H2,H1;!$(NC=O)]'
}

# Initialize with custom patterns (and optionally control hydrogen addition)
featurizer = MoleculeFeaturizer("c1ccncc1CCO", hydrogen=False, custom_smarts=custom_patterns)

# Get graph with custom features automatically included
node, edge, adj = featurizer.get_graph()
# node['node_feats'] is now 122 + n_patterns dimensions

# If you need to check patterns separately
custom_feats = featurizer.get_custom_smarts_features()
# Returns: {'features': tensor, 'names': [...], 'patterns': {...}}
```

### PTM (Post-Translational Modification) Handling
```python
from plfeature import PDBStandardizer

# Mode 1: 'base_aa' - Convert PTMs to base amino acids (DEFAULT)
# Use for: ESM models, AlphaFold, MD simulations with standard force fields
standardizer = PDBStandardizer(ptm_handling='base_aa')
standardizer.standardize("protein.pdb", "protein_esm.pdb")
# SEP → SER (phosphate removed)
# PTR → TYR (phosphate removed)
# MSE → MET (selenium → sulfur)
# MLY → LYS (methyl groups removed)

# Mode 2: 'preserve' - Keep PTM atoms intact
# Use for: Protein-ligand binding analysis, structural studies with PTMs
standardizer = PDBStandardizer(ptm_handling='preserve')
standardizer.standardize("protein.pdb", "protein_structure.pdb")
# SEP stays as SEP with all phosphate atoms
# PTR stays as PTR with all phosphate atoms
# MSE stays as MSE with selenium atom

# Mode 3: 'remove' - Remove PTM residues entirely
# Use for: Standard-AA-only analysis
standardizer = PDBStandardizer(ptm_handling='remove')
standardizer.standardize("protein.pdb", "protein_standard.pdb")
# SEP residues removed
# MSE residues removed
# Only 20 standard amino acids remain

# Supported PTMs (15+ types):
# - Phosphorylation: SEP, TPO, PTR
# - Selenomethionine: MSE
# - Methylation/Acetylation: MLY, M3L, ALY
# - Hydroxylation: HYP
# - Cysteine modifications: CSO, CSS, CME, OCS
# - Others: MEN, FME

# Batch processing with different modes
import glob
import os

os.makedirs("esm_ready", exist_ok=True)
os.makedirs("ligand_ready", exist_ok=True)

for pdb_file in glob.glob("pdbs/*.pdb"):
    basename = os.path.basename(pdb_file)

    # For ESM: base_aa mode
    esm_standardizer = PDBStandardizer(ptm_handling='base_aa')
    esm_standardizer.standardize(pdb_file, f"esm_ready/{basename}")

    # For protein-ligand: preserve mode
    pl_standardizer = PDBStandardizer(ptm_handling='preserve')
    pl_standardizer.standardize(pdb_file, f"ligand_ready/{basename}")

    print(f"✓ Processed: {pdb_file}")
```

### Protein Sequence Extraction
```python
from plfeature import ProteinFeaturizer

featurizer = ProteinFeaturizer("protein.pdb")

# Get sequences separated by chain
sequences_by_chain = featurizer.get_sequence_by_chain()
for chain_id, sequence in sequences_by_chain.items():
    print(f"Chain {chain_id}: {sequence} (length: {len(sequence)})")

# If you need the full sequence, concatenate:
full_sequence = ''.join(sequences_by_chain.values())
print(f"Full sequence: {full_sequence}")

# Example output:
# Chain A: ACDEFGHIKLMNPQRSTVWY (length: 20)
# Chain B: GHIKLMNPQR (length: 10)
```

### Contact Maps with Different Thresholds
```python
from plfeature import ProteinFeaturizer

featurizer = ProteinFeaturizer("protein.pdb")

# Different thresholds for different analyses
close_contacts = featurizer.get_contact_map(cutoff=4.5)   # Close contacts only
standard_contacts = featurizer.get_contact_map(cutoff=8.0) # Standard threshold
extended_contacts = featurizer.get_contact_map(cutoff=12.0) # Extended interactions

# Access contact information
edges = standard_contacts['edges']
distances = standard_contacts['distances']
adjacency = standard_contacts['adjacency_matrix']
```

### Batch Processing
```python
from plfeature import MoleculeFeaturizer
import torch

smiles_list = ["CCO", "CC(=O)O", "c1ccccc1"]
all_features = []

for smiles in smiles_list:
    featurizer = MoleculeFeaturizer(smiles)
    features = featurizer.get_feature()
    all_features.append(features['descriptor'])

descriptors = torch.stack(all_features)
```


## 🧪 Examples

Check out the [example/](example/) directory for:
- **10gs_protein.pdb** and **10gs_ligand.sdf**: Example input files
- **test_featurizer.py**: Comprehensive test script demonstrating all features
- **test_ptm_handling.py**: PTM handling modes test suite

```bash
cd example
python test_featurizer.py       # Test all featurizer functions
python test_ptm_handling.py     # Test PTM handling modes (base_aa, preserve, remove)
```

## 📖 Documentation

- **[Feature Types Overview](docs/feature_types.md)** - Quick overview of all features
- **[Molecular Descriptors & Fingerprints](docs/molecule_feature.md)** - Molecular features guide
- **[Molecule Graph Features](docs/molecule_graph.md)** - Graph representations for molecules
- **[Protein Residue Features](docs/protein_residue_feature.md)** - Residue-level features guide
- **[Protein Atom Features](docs/protein_atom_feature.md)** - Atom-level features guide
- **[Molecular Descriptors Reference](docs/molecular_descriptors.md)** - Complete descriptor reference

## 📄 License

MIT License - see [LICENSE](LICENSE) file

## 📖 Citation

```bibtex
@software{plfeature2025,
  title = {plfeature: Unified molecular and protein feature extraction for protein-ligand modeling},
  author = {Jaemin Sim},
  year = {2025},
  url = {https://github.com/eightmm/plfeature}
}
```

