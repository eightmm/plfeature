# Molecule Graph Representation

## Overview

Graph representations of molecules for Graph Neural Networks (GNNs). This module converts molecular structures into graph format with comprehensive node (atom) and edge (bond) features.

**Feature Dimensions:**
- Node Features: **157 dimensions**
- Edge Features: **66 dimensions**

## Quick Start

```python
from plfeature import MoleculeFeaturizer

# Initialize featurizer
featurizer = MoleculeFeaturizer()

# Extract graph features
node, edge, adj = featurizer.get_graph("CCO")

# Access features
node_features = node['node_feats']  # [n_atoms, 157]
edge_features = edge['edge_feats']  # [n_edges, 66]
coordinates = node['coords']        # [n_atoms, 3]
edge_indices = edge['edges']        # [2, n_edges]

# Without hydrogens for lighter graphs
node, edge, adj = featurizer.get_graph("CCO", add_hs=False)
```

---

# Node Features (157 dimensions)

Total 157 dimensions organized into 13 categories.

## Summary Table

| Category | Dimensions | Description |
|----------|------------|-------------|
| Basic Features | 39 | Atom type, period, group, basic properties |
| Degree Features | 41 | Connectivity and neighbor statistics |
| Ring Features | 21 | Ring membership and topology |
| SMARTS Features | 5 | Chemical pattern matching |
| Stereochemistry | 8 | Chirality and geometry |
| Partial Charges | 2 | Gasteiger charges |
| Extended Neighborhood | 16 | 1-hop and 2-hop neighbor statistics (PLI-focused) |
| Physical Properties | 6 | Atomic radii, IE, polarizability |
| Crippen Contributions | 2 | logP and MR per atom |
| TPSA Contribution | 1 | Polar surface area |
| Labute ASA | 1 | Accessible surface area |
| Topological Features | 5 | Graph centrality measures |
| Extended Neighbor Stats | 6 | Neighbor property statistics |
| Extended Ring Features | 4 | Advanced ring topology |
| **Total** | **157** | |

---

## 1. Basic Features (39 dimensions)

Fundamental atomic properties and periodic table information.

### Atom Type One-hot (11 dimensions)
| Index | Feature | Description |
|-------|---------|-------------|
| 0 | H | Hydrogen |
| 1 | C | Carbon |
| 2 | N | Nitrogen |
| 3 | O | Oxygen |
| 4 | S | Sulfur |
| 5 | P | Phosphorus |
| 6 | F | Fluorine |
| 7 | Cl | Chlorine |
| 8 | Br | Bromine |
| 9 | I | Iodine |
| 10 | UNK | Unknown/Other elements |

### Period One-hot (5 dimensions)
Periodic table row (0-4). Example: C, N, O are period 1; S, Cl are period 2.

### Group One-hot (18 dimensions)
Periodic table column (0-17). Example: C is group 13; N is group 14; O is group 15.

### Basic Properties (5 dimensions)
| Feature | Description | Normalization |
|---------|-------------|---------------|
| Is Aromatic | Part of aromatic ring | Binary (0/1) |
| Is In Ring | Part of any ring | Binary (0/1) |
| Radical Electrons | Unpaired electrons | / 3.0 |
| Formal Charge | Ionic charge | (charge + 3) / 6.0 |
| Electronegativity | Pauling scale | (EN - 0.8) / 3.2 |

**Example - Formal Charge:**
```
NH4+ : charge = +1 → (1 + 3) / 6 = 0.67
COO- : O charge = -1 → (-1 + 3) / 6 = 0.33
Neutral: charge = 0 → (0 + 3) / 6 = 0.50
```

---

## 2. Degree Features (41 dimensions)

Atom connectivity and neighbor relationship statistics.

### Degree One-hot Encodings (27 dimensions)
| Feature | Dimensions | Description |
|---------|------------|-------------|
| Total Degree | 7 | Number of all neighbors (0-6) |
| Heavy Degree | 7 | Number of non-H neighbors (0-6) |
| Total Valence | 8 | Sum of bond orders (0-7) |
| Total Hs | 5 | Number of attached hydrogens (0-4) |

### Hybridization One-hot (6 dimensions)
| Index | Hybridization | Example |
|-------|---------------|---------|
| 0 | SP | Acetylene C≡C |
| 1 | SP2 | Ethylene C=C, Benzene |
| 2 | SP3 | Methane, Ethane |
| 3 | SP3D | PCl5 |
| 4 | SP3D2 | SF6 |
| 5 | UNSPECIFIED | Metals, unknown |

### Neighbor Statistics (8 dimensions)
| Feature | Description | Normalization |
|---------|-------------|---------------|
| Min Neighbor Degree | Minimum degree among neighbors | / 6 |
| Max Neighbor Degree | Maximum degree among neighbors | / 6 |
| Mean Neighbor Degree | Average degree of neighbors | / 6 |
| Min Neighbor Heavy | Min heavy-atom degree of neighbors | / 6 |
| Max Neighbor Heavy | Max heavy-atom degree of neighbors | / 6 |
| Mean Neighbor Heavy | Average heavy-atom degree | / 6 |
| Degree Centrality | degree / (n_atoms - 1) | [0, 1] |
| Degree Variance | Variance of neighbor degrees | / 10 |

**Example - Isopropanol:**
```
      OH
      |
CH3 - CH - CH3

Central C: degree=4, heavy_degree=3
Terminal C: degree=1, heavy_degree=1
O: degree=2 (with H), heavy_degree=1
```

---

## 3. Ring Features (21 dimensions)

Ring membership and topology information.

| Feature | Dimensions | Description |
|---------|------------|-------------|
| Is In Ring | 1 | Atom is part of any ring |
| Is Aromatic | 1 | Atom is in aromatic ring |
| Num Rings | 1 | Number of rings containing atom (/ 4.0) |
| Ring Size Flags | 6 | Binary flags for ring sizes 3, 4, 5, 6, 7, 8+ |
| Num Rings One-hot | 5 | One-hot for 0, 1, 2, 3, 4 rings |
| Smallest Ring One-hot | 7 | One-hot for smallest ring size (0, 3-8) |

**Example - Naphthalene:**
```
Fusion carbons (shared by 2 rings): num_rings=2, smallest_ring=6
Edge carbons (in 1 ring): num_rings=1, smallest_ring=6
```

---

## 4. SMARTS Features (5 dimensions)

Chemical functionality detection using SMARTS patterns.

| Feature | SMARTS Pattern | Description |
|---------|----------------|-------------|
| H-bond Acceptor | `[O,S,N;...` | Can accept hydrogen bond (lone pairs) |
| H-bond Donor | `[N,O,S;H1...]` | Can donate hydrogen bond (N-H, O-H) |
| Electron Acceptor | `[C,S](=[O,S,P])...` | Electrophilic center (carbonyl, etc.) |
| Electron Donor | `[#7;+,...]` | Nucleophilic nitrogen |
| Hydrophobic | `[C,c,F,Cl,Br,I...]` | Hydrophobic/lipophilic region |

**Example - Aspirin:**
```
-COOH: O is H-bond acceptor, OH is H-bond donor
-OCOCH3: Ester O is H-bond acceptor
Benzene ring: Hydrophobic
```

---

## 5. Stereochemistry Features (8 dimensions)

Chirality and geometric isomerism information.

| Feature | Description |
|---------|-------------|
| Chiral CW | Clockwise tetrahedral (R configuration) |
| Chiral CCW | Counter-clockwise tetrahedral (S configuration) |
| Chiral Unspecified | Unassigned chirality |
| Potential Chiral | SP3 carbon with 4 different neighbors |
| Has Stereo Bond | Connected to E/Z double bond |
| Is Aromatic | Aromatic atom (planar) |
| Is SP2 | SP2 hybridized (non-aromatic, planar) |
| Is SP | SP hybridized (linear) |

**Example - Alanine:**
```
     COOH
      |
H3C - C* - NH2    (* = chiral center)
      |
      H

C* has CW or CCW depending on R/S configuration
```

---

## 6. Partial Charges (2 dimensions)

Gasteiger charge distribution.

| Feature | Description | Normalization |
|---------|-------------|---------------|
| Gasteiger Charge | Partial atomic charge | (charge + 1) / 2 → [0, 1] |
| Abs Charge | Absolute charge magnitude | |charge| → [0, 1] |

**Example - Carbonyl:**
```
C=O: C is δ+ (positive), O is δ- (negative)
Gasteiger assigns partial charges based on electronegativity
```

---

## 7. Extended Neighborhood (16 dimensions)

Statistics about 1-hop and 2-hop neighbors, designed for protein-ligand interaction (PLI) prediction.

### 1-hop Features (8 dimensions)

| Index | Feature | Description | Normalization |
|-------|---------|-------------|---------------|
| 0 | Count | Number of direct neighbors | / 6.0, clipped to 1.0 |
| 1 | Aromatic Ratio | Fraction of aromatic neighbors | [0, 1] |
| 2 | Hetero Ratio | Fraction of N/O/S neighbors | [0, 1] |
| 3 | H-Donor Ratio | Fraction of H-bond donors (N-H, O-H) | [0, 1] |
| 4 | H-Acceptor Ratio | Fraction of H-bond acceptors (N, O) | [0, 1] |
| 5 | Mean Partial Charge | Average Gasteiger charge | (charge + 1) / 2 |
| 6 | Ring Atom Ratio | Fraction of ring atoms | [0, 1] |
| 7 | Halogen Ratio | Fraction of F/Cl/Br/I neighbors | [0, 1] |

### 2-hop Features (8 dimensions)

| Index | Feature | Description | Normalization |
|-------|---------|-------------|---------------|
| 8 | Count | Number of 2-hop neighbors | / 20.0, clipped to 1.0 |
| 9 | Aromatic Ratio | Fraction aromatic in 2-hop shell | [0, 1] |
| 10 | Hetero Ratio | Fraction N/O/S in 2-hop shell | [0, 1] |
| 11 | H-Donor Ratio | Fraction of H-bond donors | [0, 1] |
| 12 | H-Acceptor Ratio | Fraction of H-bond acceptors | [0, 1] |
| 13 | Mean Partial Charge | Average charge in 2-hop shell | (charge + 1) / 2 |
| 14 | Ring Atom Ratio | Fraction of ring atoms | [0, 1] |
| 15 | Halogen Ratio | Fraction of halogens | [0, 1] |

### PLI Relevance

These features are specifically designed for protein-ligand binding prediction:

| Feature | PLI Significance |
|---------|-----------------|
| H-Donor/Acceptor | Critical for H-bond interactions with protein backbone/sidechains |
| Mean Partial Charge | Electrostatic complementarity with binding site |
| Ring Atom Ratio | π-stacking and hydrophobic interactions |
| Halogen Ratio | Halogen bonding (emerging importance in drug design) |

**Example - Aspirin (CC(=O)Oc1ccccc1C(=O)O):**
```
Carbonyl O (C=O):
  1-hop: C neighbor
    h_acceptor_ratio = 0.0 (C is not acceptor)
    ring_ratio = 0.0 or 1.0 (depends on which carbonyl)
  2-hop: More diverse environment
    aromatic_ratio = high (benzene ring nearby)

Carboxylic OH:
  1-hop: C neighbor
    h_donor_ratio = 0.0 (neighbor C has no H)
  Note: The O itself would be counted in its neighbors' features
```

**Example - Halogenated compound (FC(F)(F)c1ccc(Cl)cc1Br):**
```
Central C (CF3):
  1-hop: 3 F atoms → halogen_ratio = 0.75
  2-hop: benzene C atoms → halogen_ratio ≈ 0.33 (Cl, Br in 2-hop)
```

---

## 8. Physical Properties (6 dimensions)

Atomic physical constants from lookup tables.

| Feature | Description | Source | Normalization |
|---------|-------------|--------|---------------|
| Atomic Mass | Atomic weight | Periodic table | / 200.0 |
| Van der Waals Radius | Non-bonded radius | Bondi radii | (r - 1.0) / 2.0 |
| Covalent Radius | Bonding radius | Literature | / 2.0 |
| Ionization Energy | First IE (eV) | NIST | (IE - 4.0) / 14.0 |
| Polarizability | Atomic polarizability (Å³) | Literature | log(1+p) / 4.0 |
| Lone Pairs | Non-bonding electron pairs | Calculated | / 3.0 |

**Lone Pairs Calculation:**
```
Lone pairs = (valence_electrons - bonds - hydrogens) / 2

O in H2O: (6 - 2 - 0) / 2 = 2 lone pairs
N in NH3: (5 - 3 - 0) / 2 = 1 lone pair
```

---

## 9. Crippen Contributions (2 dimensions)

Per-atom contribution to molecular logP and molar refractivity.

| Feature | Description | Normalization |
|---------|-------------|---------------|
| logP Contribution | Wildman-Crippen logP per atom | (logP + 2) / 4 |
| MR Contribution | Molar refractivity per atom | / 10.0 |

**Usage:** Sum of atomic contributions ≈ molecular logP/MR

---

## 10. TPSA Contribution (1 dimension)

Per-atom contribution to Topological Polar Surface Area.

| Feature | Description | Normalization |
|---------|-------------|---------------|
| TPSA | Polar surface area contribution | / 30.0 |

**Note:** Only N and O atoms contribute significantly to TPSA.

---

## 11. Labute ASA (1 dimension)

Per-atom Accessible Surface Area contribution.

| Feature | Description | Normalization |
|---------|-------------|---------------|
| ASA | Labute ASA contribution | / 20.0 |

---

## 12. Topological Features (5 dimensions)

Graph-theoretic centrality and distance measures.

| Feature | Description | Normalization |
|---------|-------------|---------------|
| Eccentricity | Max shortest path to any atom | / 20.0 |
| Closeness Centrality | 1 / (sum of distances) | [0, 1] |
| Betweenness Centrality | Fraction of shortest paths through atom | [0, 1] |
| Distance to Heteroatom | Shortest path to nearest N/O/S | / 10.0 |
| Distance to Ring | Shortest path to nearest ring atom | / 10.0 |

**Example - Betweenness:**
```
A - B - C - D - E
        |
        F

B and C have high betweenness (many paths pass through them)
A, E, F have low betweenness (endpoints)
```

---

## 13. Extended Neighbor Statistics (6 dimensions)

Aggregate properties of neighboring atoms.

| Feature | Description | Normalization |
|---------|-------------|---------------|
| Neighbor EN Sum | Sum of neighbor electronegativities | / 16.0 |
| Neighbor EN Diff | Max - min electronegativity | / 3.2 |
| Neighbor Mass Sum | Sum of neighbor atomic masses | / 600.0 |
| Neighbor Charge Sum | Sum of neighbor formal charges | (sum + 4) / 8.0 |
| Aromatic Neighbor Ratio | Fraction of aromatic neighbors | [0, 1] |
| Ring Neighbor Ratio | Fraction of ring neighbors | [0, 1] |

---

## 14. Extended Ring Features (4 dimensions)

Advanced ring topology features.

| Feature | Description | Normalization |
|---------|-------------|---------------|
| Aromatic Bonds | Number of aromatic bonds on atom | / 3.0 |
| Ring Fusion Degree | Number of rings atom belongs to | / 3.0 |
| Is Bridgehead | Bridgehead in polycyclic system | Binary |
| Is Spiro | Spiro center (connects 2 rings at single atom) | Binary |

**Example - Bridgehead vs Spiro:**
```
Bridgehead (Norbornane):     Spiro:
    C                          C
   /|\                        /|\
  C | C                      C | C
  | C |                       \|/
  C   C                        C
   \ /                        /|\
    C                        C | C
                              \|/
                               C
```

---

# Edge Features (66 dimensions)

Total 66 dimensions organized into 8 categories.

## Summary Table

| Category | Dimensions | Description |
|----------|------------|-------------|
| Bond Type One-hot | 4 | Single, double, triple, aromatic |
| Bond Stereo One-hot | 6 | E/Z, cis/trans configuration |
| Bond Direction One-hot | 5 | Wedge/dash for chirality |
| Basic Bond Properties | 5 | Aromaticity, conjugation, ring, rotation |
| Atom Pair Properties | 8 | Properties comparing bonded atoms |
| Ring Features | 21 | Bond ring membership |
| Topological Features | 6 | Bond centrality and graph position |
| Degree-based Features | 11 | Connectivity statistics |
| **Total** | **66** | |

---

## 1. Bond Type One-hot (4 dimensions)

Fundamental bond classification.

| Index | Type | Bond Order | Description |
|-------|------|------------|-------------|
| 0 | SINGLE | 1.0 | σ bond only |
| 1 | DOUBLE | 2.0 | σ + π bond |
| 2 | TRIPLE | 3.0 | σ + 2π bonds |
| 3 | AROMATIC | 1.5 | Delocalized (benzene) |

**Example:**
```
Ethane (C-C): SINGLE
Ethene (C=C): DOUBLE
Ethyne (C≡C): TRIPLE
Benzene: AROMATIC
```

---

## 2. Bond Stereo One-hot (6 dimensions)

Geometric isomerism around double bonds.

| Index | Type | Description |
|-------|------|-------------|
| 0 | STEREOANY | Unspecified stereochemistry |
| 1 | STEREOCIS | Cis configuration (same side) |
| 2 | STEREOE | E configuration (opposite side, by priority) |
| 3 | STEREONONE | No stereochemistry (single bond, symmetric) |
| 4 | STEREOTRANS | Trans configuration (opposite side) |
| 5 | STEREOZ | Z configuration (same side, by priority) |

**Example - 2-Butene:**
```
Cis (Z):           Trans (E):
  CH3    CH3         CH3    H
    \   /              \   /
     C=C                C=C
    /   \              /   \
   H     H            H    CH3
```

---

## 3. Bond Direction One-hot (5 dimensions)

3D bond orientation for chirality representation in 2D.

| Index | Type | Description |
|-------|------|-------------|
| 0 | NONE | No special direction |
| 1 | BEGINWEDGE | Solid wedge (toward viewer) |
| 2 | BEGINDASH | Dashed wedge (away from viewer) |
| 3 | ENDDOWNRIGHT | End pointing down-right |
| 4 | ENDUPRIGHT | End pointing up-right |

**Example - Chiral Center:**
```
         H (dash - behind plane)
         |
    Br---C---Cl (wedge - in front)
         |
        CH3

Wedge bond: C-Cl comes toward viewer
Dash bond: C-H goes away from viewer
```

---

## 4. Basic Bond Properties (5 dimensions)

Fundamental chemical properties of the bond.

| Feature | Description | Value |
|---------|-------------|-------|
| Is Aromatic | Part of aromatic system | Binary |
| Is Conjugated | Part of conjugated π system | Binary |
| Is In Ring | Bond is within a ring | Binary |
| Is Rotatable | Single bond allowing free rotation | Binary |
| Bond Order | Normalized bond order | order / 3.0 |

**Conjugated System Example:**
```
CH2=CH-CH=CH-CH=CH2  (1,3,5-hexatriene)
All C-C bonds are conjugated (alternating single/double)
```

**Rotatable Bond:**
```
CH3-CH2-OH: C-C and C-O are rotatable (flexibility)
Benzene C-C: Not rotatable (ring constraint)
C=C: Not rotatable (π bond prevents rotation)
```

---

## 5. Atom Pair Properties (8 dimensions)

Properties comparing the two atoms connected by the bond.

| Feature | Description | Normalization | Chemical Meaning |
|---------|-------------|---------------|------------------|
| EN Difference | \|EN₁ - EN₂\| | / 3.2 | Bond polarity |
| Mass Difference | \|mass₁ - mass₂\| | / 100.0 | Bond asymmetry |
| Mass Sum | mass₁ + mass₂ | / 250.0 | Bond "weight" |
| Charge Difference | \|charge₁ - charge₂\| | / 4.0 | Ionic character |
| Same Hybridization | hybrid₁ == hybrid₂ | Binary | Orbital compatibility |
| Both Aromatic | aromatic₁ AND aromatic₂ | Binary | Within aromatic system |
| Both In Ring | ring₁ AND ring₂ | Binary | Intra-ring bond |
| Hetero Bond | non-C/H atom involved | Binary | Functional group bond |

**EN Difference (Bond Polarity) Examples:**
```
C-C:  |2.55 - 2.55| = 0.00  → Nonpolar
C-N:  |2.55 - 3.04| = 0.49  → Slightly polar
C-O:  |2.55 - 3.44| = 0.89  → Polar
C-F:  |2.55 - 3.98| = 1.43  → Highly polar
```

**Hetero Bond Examples:**
```
C-C: Hetero = 0 (both carbon)
C-O: Hetero = 1 (oxygen is heteroatom)
C-N: Hetero = 1 (nitrogen is heteroatom)
N-O: Hetero = 1 (both are heteroatoms)
```

---

## 6. Ring Features (21 dimensions)

Ring membership information for the bond (same encoding as node ring features).

| Feature | Dimensions | Description |
|---------|------------|-------------|
| Is In Ring | 1 | Bond is part of a ring |
| Is Aromatic | 1 | Bond is in aromatic ring |
| Num Rings | 1 | Number of rings containing bond (/ 4.0) |
| Ring Size Flags | 6 | Binary flags for sizes 3, 4, 5, 6, 7, 8+ |
| Num Rings One-hot | 5 | One-hot for 0, 1, 2, 3, 4 rings |
| Smallest Ring One-hot | 7 | One-hot for smallest ring size |

**Example - Naphthalene Fusion Bond:**
```
    ___     ___
   /   \___/   \
   \___/   \___/
        ^
    Fusion bond: in 2 rings, smallest=6
```

---

## 7. Topological Features (6 dimensions)

Graph-theoretic properties of the bond.

| Feature | Description | Normalization | Meaning |
|---------|-------------|---------------|---------|
| Bond Betweenness | Fraction of shortest paths using this bond | [0, 1] | "Traffic" through bond |
| Is Bridge | Removal disconnects graph | Binary | Critical connection |
| Ring Fusion | Number of rings sharing this bond | / 3.0 | Polycyclic junction |
| Distance to Hetero | Min distance to nearest heteroatom | / 20.0 | Proximity to functional groups |
| Distance to Ring | Min distance to nearest ring | / 20.0 | Proximity to ring systems |
| Graph Distance | Average position in molecular graph | / 20.0 | Central vs peripheral |

**Bond Betweenness Example:**
```
A - B - C - D - E
        |
        F

B-C bond: High betweenness (many paths: A→D, A→E, A→F, etc.)
D-E bond: Low betweenness (only paths ending at E)
```

**Bridge Bond Example:**
```
CH3-CH2-CH2-OH
    ^   ^   ^
  All bonds are bridges (removal disconnects)

Benzene ring:
  No bridges (can remove any bond, still connected)
```

**Ring Fusion Example:**
```
Naphthalene:
  Fusion bond connects both 6-membered rings
  ring_fusion = 2 → normalized = 2/3 = 0.67
```

---

## 8. Degree-based Features (11 dimensions)

Connectivity statistics of atoms connected by the bond.

| Feature | Description | Normalization |
|---------|-------------|---------------|
| Degree Diff | \|degree₁ - degree₂\| | / 6 |
| Heavy Degree Diff | \|heavy_deg₁ - heavy_deg₂\| | / 6 |
| Valence Diff | \|valence₁ - valence₂\| | / 8 |
| Degree Sum | degree₁ + degree₂ | / 12 |
| Heavy Degree Sum | heavy_deg₁ + heavy_deg₂ | / 12 |
| Valence Sum | valence₁ + valence₂ | / 16 |
| Centrality Diff | \|centrality₁ - centrality₂\| | [0, 1] |
| Centrality Sum | (centrality₁ + centrality₂) / 2 | [0, 1] |
| Min Degree | min(degree₁, degree₂) | / 6 |
| Max Degree | max(degree₁, degree₂) | / 6 |
| Variance Diff | \|var₁ - var₂\| | / 10 |

**Example - Isopropanol:**
```
      OH (degree=1)
      |
CH3 - CH - CH3
(1)   (4)   (1)

CH-O bond: degree_diff=|4-1|=3, degree_sum=5
CH-CH3 bond: degree_diff=|4-1|=3, degree_sum=5
```

---

# Usage Examples

## Integration with PyTorch Geometric

```python
from torch_geometric.data import Data, Batch
from plfeature import MoleculeFeaturizer

featurizer = MoleculeFeaturizer()

# Single molecule
node, edge, adj = featurizer.get_graph("CCO")
data = Data(
    x=node['node_feats'],        # [N, 157]
    edge_index=edge['edges'],    # [2, E]
    edge_attr=edge['edge_feats'], # [E, 66]
    pos=node['coords']           # [N, 3]
)

# Batch processing
smiles_list = ["CCO", "c1ccccc1", "CC(=O)O"]
data_list = []
for smi in smiles_list:
    node, edge, adj = featurizer.get_graph(smi)
    data_list.append(Data(
        x=node['node_feats'],
        edge_index=edge['edges'],
        edge_attr=edge['edge_feats']
    ))
batch = Batch.from_data_list(data_list)
```

## With 3D Coordinates

```python
from rdkit import Chem
from rdkit.Chem import AllChem

# Generate 3D conformer
mol = Chem.MolFromSmiles("CCO")
mol = Chem.AddHs(mol)
AllChem.EmbedMolecule(mol, randomSeed=42)
AllChem.UFFOptimizeMolecule(mol)

# Extract with 3D coordinates
featurizer = MoleculeFeaturizer()
node, edge, adj = featurizer.get_graph(mol)
coords = node['coords']  # Real 3D coordinates
```

---

# Feature Categories by Application

## Drug Discovery
| Task | Recommended Features |
|------|---------------------|
| ADMET Prediction | Crippen logP/MR, TPSA, ASA, H-bond features |
| Binding Affinity | Partial charges, topological centrality, physical properties |
| Metabolic Stability | Lone pairs, ring features, heteroatom distances |

## Molecular Property Prediction
| Task | Recommended Features |
|------|---------------------|
| Electronic Properties | Gasteiger charges, electronegativity, ionization energy |
| Structural Properties | Ring features, topological features, degree statistics |
| Size/Shape | Atomic mass, VDW radius, ASA contributions |

## Reaction Prediction
| Task | Recommended Features |
|------|---------------------|
| Reactivity Sites | Betweenness centrality, lone pairs, formal charge |
| Leaving Groups | Distance to heteroatom, ring features, bond polarity |
| Electrophilicity | Partial charges, neighbor electronegativity, EN difference |

---

# Normalization

All features are normalized to approximately [0, 1]:

| Type | Method |
|------|--------|
| One-hot | Binary (0 or 1) |
| Counts | Divided by maximum expected value |
| Charges | Shifted and scaled: (x + offset) / range |
| Distances | Divided by maximum path length |
| Physical properties | Scaled by typical ranges |
| Ratios | Already in [0, 1] |
