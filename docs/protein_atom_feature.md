# Protein Atom-Level Features

## Overview
Atom-level feature extraction from protein structures with 175 unique token types and atomic SASA calculation.

## Atom Tokenization System

### Token Mapping
The atom featurizer encodes each atom based on its residue type and atom name, creating 175 unique tokens.

**Token Range:**
- 0-173: Standard residue-atom combinations
- 174: Unknown/non-standard atoms

**Example Tokens:**
```python
# Some example mappings
('ALA', 'N'): 0     # Alanine backbone nitrogen
('ALA', 'CA'): 1    # Alanine alpha carbon
('ALA', 'C'): 2     # Alanine backbone carbon
('ALA', 'O'): 3     # Alanine backbone oxygen
('ALA', 'CB'): 4    # Alanine beta carbon
('CYS', 'SG'): 21   # Cysteine sulfur
('TRP', 'CZ2'): 163 # Tryptophan ring carbon
```

## Feature Extraction Methods

### Method Naming Convention
All atom-level methods have clear naming to distinguish from residue-level:
- `get_atom_features(distance_cutoff)` → Returns (node, edge) graph format
- `get_atom_graph(distance_cutoff)` → Alias for get_atom_features()
- `get_atom_tokens_and_coords()` → Returns (token, coord) tuple
- `get_atom_features_with_sasa()` → Aliases: `get_atom_sasa()`, `get_atom_level_sasa()`
- `get_atom_coordinates()` → Get only 3D coordinates
- `get_atom_tokens_only()` → Get only token IDs

### 1. Atom Graph Features (`get_atom_features()` / `get_atom_graph()`)

Get atom-level graph representation with node and edge features, similar to residue-level format.

```python
from plfeature import ProteinFeaturizer

featurizer = ProteinFeaturizer("protein.pdb")

# Get atom graph with customizable distance cutoff
node, edge = featurizer.get_atom_features(distance_cutoff=4.0)  # Default: 4.0 Å
# Or use explicit alias
node, edge = featurizer.get_atom_graph(distance_cutoff=4.0)

# Node dictionary contains (SASA is automatically calculated and included):
# - 'coord': 3D coordinates [n_atoms, 3]
# - 'node_features': Atom tokens [n_atoms]
# - 'atom_tokens': Same as node_features
# - 'sasa': Solvent accessible surface area per atom [n_atoms] - INCLUDED BY DEFAULT
# - 'residue_token': Residue type for each atom [n_atoms]
# - 'atom_element': Element type for each atom (list)

# Edge dictionary contains:
# - 'edges': (src, dst) tuple of edge indices
# - 'edge_distances': Distances between connected atoms
# - 'distance_cutoff': The cutoff used
```

### 2. Basic Token and Coordinates (`get_atom_tokens_and_coords()`)

Get just tokens and coordinates without graph structure.

```python
token, coord = featurizer.get_atom_tokens_and_coords()
# Or use alias
token, coord = featurizer.get_atom_tokens()

# Returns:
# token: torch.Tensor [n_atoms] - Atom type tokens (0-174)
# coord: torch.Tensor [n_atoms, 3] - 3D coordinates
```

### 3. Atom Features with SASA (`get_atom_features_with_sasa()`)

Atom features including solvent accessible surface area.

```python
features = featurizer.get_atom_features_with_sasa()  # or get_atom_sasa()

# Returns dictionary:
{
    'token': torch.Tensor,         # [n_atoms] Atom type tokens
    'coord': torch.Tensor,         # [n_atoms, 3] 3D coordinates
    'sasa': torch.Tensor,          # [n_atoms] SASA per atom (Ų)
    'residue_token': torch.Tensor, # [n_atoms] Residue type (0-20)
    'atom_element': list,          # Element symbols ['C', 'N', 'O', ...]
    'radius': torch.Tensor,        # [n_atoms] Atomic radii
    'metadata': {
        'n_atoms': int,
        'n_residues': int,
        'pdb_file': str
    }
}
```

### 4. Standalone Function Usage

Can also be used without class instantiation:

```python
from plfeature.protein_featurizer import get_protein_atom_features

# Basic features
token, coord = get_protein_atom_features("protein.pdb")

# With SASA
from plfeature.protein_featurizer import get_protein_atom_features_with_sasa
features = get_protein_atom_features_with_sasa("protein.pdb")
```

## SASA Calculation

Uses FreeSASA library for solvent accessible surface area calculation.

**Parameters:**
- Algorithm: Lee & Richards
- Probe radius: 1.4 Å (water molecule)
- Resolution: Default FreeSASA parameters

**SASA Values:**
```python
features = featurizer.get_atom_features_with_sasa()  # or get_atom_sasa()
sasa_per_atom = features['sasa']  # Ų per atom

# Statistics
total_sasa = sasa_per_atom.sum()
buried_atoms = (sasa_per_atom < 0.01).sum()
exposed_atoms = (sasa_per_atom > 20.0).sum()
```

## Token Distribution

### Residue-Specific Tokens
Each amino acid has specific atom tokens:

```python
# Glycine (smallest): 5 tokens
GLY_tokens = [('GLY', atom) for atom in ['N', 'CA', 'C', 'O', 'H']]

# Tryptophan (largest): 24 tokens
TRP_tokens = [('TRP', atom) for atom in [
    'N', 'CA', 'C', 'O', 'CB', 'CG', 'CD1', 'CD2',
    'NE1', 'CE2', 'CE3', 'CZ2', 'CZ3', 'CH2', ...
]]
```

### Token Categories
- **Backbone atoms** (N, CA, C, O): Present in all residues
- **Beta carbon** (CB): Present in all except glycine
- **Sidechain atoms**: Residue-specific
- **Hydrogen atoms**: Included when present in PDB

## Usage Examples

### Basic Atom-Level Analysis
```python
from plfeature import ProteinFeaturizer
import torch

featurizer = ProteinFeaturizer("protein.pdb")

# Get atom graph with SASA included
node, edge = featurizer.get_atom_features(distance_cutoff=4.0)

# Access all atom-level data from node dictionary
coords = node['coord']
tokens = node['atom_tokens']
sasa = node['sasa']  # SASA is automatically included!
elements = node['atom_element']

print(f"Number of atoms: {len(tokens)}")
print(f"Unique atom types: {torch.unique(tokens).shape[0]}")
print(f"Total SASA: {sasa.sum():.2f} Ų")
print(f"Number of edges: {edge['edges'][0].shape[0]}")
```

### SASA-Based Analysis
```python
# SASA is already included in get_atom_features!
node, edge = featurizer.get_atom_features(distance_cutoff=4.0)

# Access SASA directly from node dictionary
sasa = node['sasa']
tokens = node['atom_tokens']
residue_tokens = node['residue_token']

# Find exposed atoms
exposed_mask = sasa > 20.0  # Ų threshold
exposed_tokens = tokens[exposed_mask]

print(f"Exposed atoms: {exposed_mask.sum()}/{len(tokens)}")
print(f"Total SASA: {sasa.sum():.2f} Ų")

# Per-residue SASA
for res_type in torch.unique(residue_tokens):
    mask = residue_tokens == res_type
    res_sasa = sasa[mask].sum()
    print(f"Residue type {res_type}: {res_sasa:.2f} Ų")
```

### Atom-Level Graph Construction
```python
import torch
from scipy.spatial import distance_matrix

features = featurizer.get_atom_features_with_sasa()  # or get_atom_sasa()

# Build atom-level contact graph
coords = features['coord'].numpy()
dist_matrix = distance_matrix(coords, coords)

# Define contacts (e.g., < 5 Å)
contact_threshold = 5.0
contacts = dist_matrix < contact_threshold

# Create edge list
edges = torch.tensor(contacts).nonzero().T

print(f"Atom-level edges: {edges.shape[1]}")
```

### Integration with Deep Learning

```python
import torch
import torch.nn as nn

class AtomLevelModel(nn.Module):
    def __init__(self, n_tokens=175, hidden_dim=128):
        super().__init__()
        self.embedding = nn.Embedding(n_tokens, hidden_dim)
        self.sasa_proj = nn.Linear(1, hidden_dim)
        self.coord_proj = nn.Linear(3, hidden_dim)

    def forward(self, tokens, coords, sasa):
        token_emb = self.embedding(tokens)
        coord_emb = self.coord_proj(coords)
        sasa_emb = self.sasa_proj(sasa.unsqueeze(-1))

        # Combine features
        atom_features = token_emb + coord_emb + sasa_emb
        return atom_features

# Use with extracted features
featurizer = ProteinFeaturizer("protein.pdb")
features = featurizer.get_atom_features_with_sasa()  # or get_atom_sasa()

model = AtomLevelModel()
atom_features = model(
    features['token'],
    features['coord'],
    features['sasa']
)
```

### Batch Processing
```python
import glob

pdb_files = glob.glob("pdbs/*.pdb")
all_atom_features = []

for pdb_file in pdb_files:
    featurizer = ProteinFeaturizer(pdb_file)
    features = featurizer.get_atom_features_with_sasa()  # or get_atom_sasa()
    all_atom_features.append(features)

# Statistics
total_atoms = sum(f['token'].shape[0] for f in all_atom_features)
avg_sasa = sum(f['sasa'].mean() for f in all_atom_features) / len(all_atom_features)

print(f"Total atoms: {total_atoms}")
print(f"Average SASA per atom: {avg_sasa:.2f} Ų")
```

## Atom Selection and Filtering

### By Element Type
```python
features = featurizer.get_atom_features_with_sasa()  # or get_atom_sasa()
elements = features['atom_element']

# Select only carbon atoms
carbon_mask = torch.tensor([e == 'C' for e in elements])
carbon_tokens = features['token'][carbon_mask]
carbon_coords = features['coord'][carbon_mask]
```

### By Residue Type
```python
# Select atoms from aromatic residues
aromatic_residues = [5, 19, 9]  # PHE, TYR, TRP
res_tokens = features['residue_token']
aromatic_mask = torch.isin(res_tokens, torch.tensor(aromatic_residues))

aromatic_atoms = features['token'][aromatic_mask]
```

### By SASA Exposure
```python
# Categorize by exposure
sasa = features['sasa']

buried = sasa < 0.01       # Completely buried
partially_exposed = (sasa >= 0.01) & (sasa < 20.0)
exposed = sasa >= 20.0      # Highly exposed

print(f"Buried: {buried.sum()}")
print(f"Partially exposed: {partially_exposed.sum()}")
print(f"Exposed: {exposed.sum()}")
```


## Token Reference Table

Complete mapping available in the source code:

```python
# Access the full token dictionary
from plfeature.protein_featurizer.atom_featurizer import AtomFeaturizer

featurizer = AtomFeaturizer()
token_dict = featurizer.res_atm_token

# Print all mappings
for (res, atom), token in sorted(token_dict.items(), key=lambda x: x[1]):
    print(f"{res:3s} {atom:4s} -> {token:3d}")
```

