# Hierarchical Protein Featurizer

Hierarchical feature extraction for atom-residue attention models with ESM embeddings.

## Overview

The `HierarchicalFeaturizer` extracts multi-level protein features designed for hierarchical attention mechanisms:
- **Atom-level**: Token-based atom features with one-hot encoding
- **Residue-level**: Scalar and vector features from local geometry
- **ESM embeddings**: Pre-trained language model embeddings (ESMC + ESM3)

## Quick Start

```python
from plfeature.protein_featurizer import HierarchicalFeaturizer

# Initialize (loads ESM models ~2-3s)
featurizer = HierarchicalFeaturizer()

# Extract features
data = featurizer.featurize("protein.pdb")

# Access features
print(data.atom_tokens.shape)        # [N_atom, 187] one-hot
print(data.residue_features.shape)   # [N_res, 76]
print(data.esmc_embeddings.shape)    # [N_res, 1152]
```

## Feature Dimensions

### Atom-level Features (One-hot Encoded)

| Feature | Shape | Description |
|---------|-------|-------------|
| `atom_tokens` | [N_atom, 187] | Atom token one-hot (residue-atom combinations) |
| `atom_coords` | [N_atom, 3] | 3D coordinates (raw) |
| `atom_sasa` | [N_atom] | Solvent accessible surface area (normalized /100) |
| `atom_elements` | [N_atom, 8] | Element type one-hot (C, N, O, S, P, Se, Metal, UNK) |
| `atom_residue_types` | [N_atom, 22] | Residue type one-hot for each atom |

### Residue-level Features

| Feature | Shape | Description |
|---------|-------|-------------|
| `residue_features` | [N_res, 76] | Scalar features (see breakdown below) |
| `residue_vector_features` | [N_res, 31, 3] | Vector features for SE(3) equivariance |
| `residue_ca_coords` | [N_res, 3] | CA atom coordinates |
| `residue_sc_coords` | [N_res, 3] | Sidechain centroid coordinates |

**Scalar Features (76-dim) Breakdown:**
- Residue type one-hot: 21
- Terminal flags (N/C): 2
- Self distances: 10
- Dihedral features (phi/psi/omega): 20
- Chi angle flags: 5
- SASA features: 10
- Forward/reverse distances: 8

**Vector Features (31x3) Breakdown:**
- Self vectors: [20, 3]
- Reference frame vectors: [8, 3]
- Local coordinate frames: [3, 3]

### ESM Embeddings (6 Tensors)

| Feature | Shape | Description |
|---------|-------|-------------|
| `esmc_embeddings` | [N_res, 1152] | ESMC per-residue embeddings |
| `esmc_bos` | [1152] | ESMC BOS (beginning of sequence) token |
| `esmc_eos` | [1152] | ESMC EOS (end of sequence) token |
| `esm3_embeddings` | [N_res, 1536] | ESM3 per-residue embeddings |
| `esm3_bos` | [1536] | ESM3 BOS token |
| `esm3_eos` | [1536] | ESM3 EOS token |

### Atom-Residue Mapping

| Feature | Shape | Description |
|---------|-------|-------------|
| `atom_to_residue` | [N_atom] | Residue index for each atom |
| `residue_atom_indices` | [N_res, max_atoms] | Atom indices per residue |
| `residue_atom_mask` | [N_res, max_atoms] | Valid atom mask |
| `num_atoms_per_residue` | [N_res] | Atom count per residue |

## Usage Examples

### Basic Feature Extraction

```python
from plfeature.protein_featurizer import HierarchicalFeaturizer

featurizer = HierarchicalFeaturizer()
data = featurizer.featurize("protein.pdb")

# Atom features
atom_feats = data.atom_tokens           # [N_atom, 187]
atom_coords = data.atom_coords          # [N_atom, 3]

# Residue features
res_feats = data.residue_features       # [N_res, 76]
res_vectors = data.residue_vector_features  # [N_res, 31, 3]

# ESM embeddings
esmc = data.esmc_embeddings             # [N_res, 1152]
esm3 = data.esm3_embeddings             # [N_res, 1536]

# Special tokens for sequence-level representation
esmc_seq = torch.cat([data.esmc_bos.unsqueeze(0),
                      data.esmc_embeddings,
                      data.esmc_eos.unsqueeze(0)])  # [N_res+2, 1152]
```

### Pocket Extraction

```python
from rdkit import Chem

# Load ligand
ligand = Chem.MolFromMolFile("ligand.sdf")

# Extract pocket features (6.0 Å cutoff)
pocket_data = featurizer.featurize_pocket("protein.pdb", ligand, cutoff=6.0)
```

### Residue Subset Selection

```python
# Select specific residues (e.g., binding site)
binding_residues = [10, 11, 12, 45, 46, 47, 100, 101]
subset = data.select_residues(binding_residues)

# Subset maintains all features
print(subset.atom_tokens.shape)        # Atoms from selected residues only
print(subset.residue_features.shape)   # [8, 76]
print(subset.esmc_embeddings.shape)    # [8, 1152]
print(subset.esmc_bos.shape)           # [1152] - original BOS preserved
```

### Move to GPU

```python
# Move all tensors to GPU
data_gpu = data.to(torch.device('cuda'))
```

### Batch Processing

```python
from plfeature.protein_featurizer import HierarchicalFeaturizer
import torch

featurizer = HierarchicalFeaturizer()

# Process multiple PDB files
pdb_files = ["protein1.pdb", "protein2.pdb", "protein3.pdb"]
all_data = [featurizer.featurize(pdb) for pdb in pdb_files]

# Access feature dimensions
for data in all_data:
    dims = data.get_feature_dims()
    print(f"Atoms: {dims['num_atoms']}, Residues: {dims['num_residues']}")
```

## Batch Processing Script

For large-scale feature extraction:

```bash
# Extract features for all proteins in a directory
python scripts/batch_featurize.py \
    --input_dir /path/to/pdb/files \
    --output_dir /path/to/output \
    --resume  # Skip already processed files
```

**Output format (.pt files):**
```python
data = torch.load("protein.pt")
# Keys: atom_tokens, atom_coords, atom_sasa, atom_elements, atom_residue_types,
#       residue_features, residue_vector_features, residue_ca_coords, residue_sc_coords,
#       esmc_embeddings, esmc_bos, esmc_eos, esm3_embeddings, esm3_bos, esm3_eos,
#       atom_to_residue, residue_atom_indices, residue_atom_mask, num_atoms_per_residue,
#       num_atoms, num_residues, pdb_id, source_path
```

## Model Configuration

```python
# Custom ESM models
featurizer = HierarchicalFeaturizer(
    esmc_model="esmc_600m",    # or "esmc_300m"
    esm3_model="esm3-open",
    esm_device="cuda",         # or "cpu"
)
```

**Available ESM Models:**

| Model | Embedding Dim | Parameters |
|-------|---------------|------------|
| esmc_300m | 960 | 300M |
| esmc_600m | 1152 | 600M |
| esm3-open | 1536 | Open weights |

## Performance

| Operation | Time |
|-----------|------|
| Model initialization | ~2-3s |
| Feature extraction (per protein) | ~0.9-1.0s |
| Throughput | ~1.2 proteins/sec |

**Memory estimate per protein (~400 residues):**
- Atom features: ~2.5 MB
- Residue features: ~0.5 MB
- ESM embeddings: ~2.0 MB
- Total: ~5 MB per protein

## Integration with PyTorch

```python
import torch
from torch.utils.data import Dataset

class ProteinDataset(Dataset):
    def __init__(self, pdb_files):
        self.featurizer = HierarchicalFeaturizer()
        self.pdb_files = pdb_files

    def __len__(self):
        return len(self.pdb_files)

    def __getitem__(self, idx):
        data = self.featurizer.featurize(self.pdb_files[idx])
        return {
            'atom_tokens': data.atom_tokens,
            'atom_coords': data.atom_coords,
            'residue_features': data.residue_features,
            'esmc_embeddings': data.esmc_embeddings,
            'esm3_embeddings': data.esm3_embeddings,
            'atom_to_residue': data.atom_to_residue,
        }
```

## Dependencies

- `torch`: PyTorch for tensor operations
- `freesasa`: SASA calculation
- `esm`: ESM model library (`pip install esm`)

## See Also

- [Protein Atom Features](protein_atom_feature.md) - Atom token definitions
- [Protein Residue Features](protein_residue_feature.md) - Residue feature details
