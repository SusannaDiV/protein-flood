# Why PFC Should Beat Neural Methods for Binding Pocket Classification

## Core Argument

**Binding pockets are topological features (H₂ cavities), and PFC is explicitly designed to capture them.**

## 1. Binding Pockets = Topological Features

### What is a Binding Pocket?

A binding pocket is:
- A **cavity** in the protein structure
- Surrounded by protein atoms
- Accessible to ligands
- Often in **hydrophobic regions**

### Topological Representation

In persistent homology:
- **H₂ features** = cavities/voids
- **Birth time** = when cavity appears (filtration value)
- **Death time** = when cavity fills (filtration value)
- **Persistence** = death - birth (how long cavity exists)

**Binding pockets are H₂ features with long persistence!**

## 2. PFC is Designed for This

### Protein-Aware Topology

PFC's design choices directly target binding pockets:

1. **Heterogeneous Balls**:
   - Hydrophobic regions expand **faster** (larger balls)
   - Charged regions expand **slower** (smaller balls)
   - **Result**: Binding pockets (often hydrophobic) are detected earlier and persist longer

2. **Residue-Level Landmarks**:
   - Ensures **biochemical coverage**
   - Prevents landmark collapse into dense regions
   - **Result**: Better sampling of pocket regions

3. **Weighted Flooding**:
   - Incorporates vdW radii, SASA, hydrophobicity, charge
   - **Result**: Filtration values reflect protein physics, not just geometry

### Explicit Cavity Detection

PFC's persistence diagrams:
- **H₂ diagram** directly encodes cavities
- Each point = (birth, death) of a cavity
- Long persistence = stable cavity = likely binding pocket

## 3. Neural Methods May Miss Topology

### PointNet++ / DGCNN / GVP-GNN

These methods:
- Learn **geometric features** (distances, angles, local neighborhoods)
- Use **learned representations** (embeddings)
- Focus on **structure**, not topology

**Problem**: They don't explicitly model cavities!

### What They Learn

- **PointNet++**: Point cloud features (centroid, spread, local patterns)
- **DGCNN**: Graph features (k-NN distances, edge features)
- **GVP-GNN**: Residue-level features (geometric vectors, angles)

**Missing**: Explicit cavity detection, topological persistence

### Why This Matters

Binding pockets are:
- **Topological** (cavities), not just geometric
- **Global** (entire cavity), not just local
- **Persistent** (stable across filtration), not transient

Neural methods may learn geometric patterns that correlate with pockets, but they don't explicitly model the topology.

## 4. AlphaFold Embeddings

### What They Are

- Pre-computed structure embeddings
- Trained on massive dataset (AlphaFold DB)
- Capture structural patterns

### Why PFC Should Still Win

1. **Structure vs Topology**:
   - AlphaFold = structure-focused
   - PFC = topology-focused
   - Binding pockets need topology!

2. **Explicit vs Implicit**:
   - AlphaFold embeddings may implicitly encode cavities
   - PFC explicitly captures them (H₂ features)

3. **Protein-Aware**:
   - AlphaFold embeddings are generic
   - PFC weights incorporate protein physics (hydrophobicity, SASA)

## 5. Evidence from Design

### PFC's Design Choices

Every design choice in PFC targets binding pockets:

1. **Heterogeneous balls for hydrophobic regions**:
   - Binding pockets are often hydrophobic
   - Larger balls → earlier detection → longer persistence

2. **Residue-level landmarks**:
   - Ensures coverage of pocket regions
   - Prevents missing pockets in sparse regions

3. **Weighted flooding**:
   - Incorporates SASA (exposure)
   - Incorporates hydrophobicity (pocket preference)
   - Incorporates charge (pocket avoidance)

4. **H₂ persistence diagrams**:
   - Directly encode cavities
   - Long persistence = stable pocket

### Neural Methods' Design

Neural methods are:
- **Generic** (not protein-specific)
- **Geometric** (not topological)
- **Learned** (not physics-based)

## 6. Expected Results

### Quantitative Expectations

| Method | Expected AUROC | Why |
|--------|---------------|-----|
| **PFC** | **~0.85** | Explicit H₂ cavity detection + protein-aware weighting |
| AlphaFold | ~0.80 | Strong structure embeddings, but not topology-focused |
| PointNet++ | ~0.75 | Geometric features, may miss topology |
| DGCNN | ~0.72 | Graph features, geometric focus |
| GVP-GNN | ~0.73 | Protein structure, but not topology |
| Alpha Complex | ~0.71 | Standard PH, no protein awareness |

### Key Differences

- **PFC**: Explicit H₂ features + protein-aware weighting = **~0.85**
- **Neural**: Learned geometric features = **~0.72-0.75**
- **Gap**: **+10-13% AUROC** from explicit topology

## 7. When Neural Methods Might Win

### Scenarios Where Neural > PFC

1. **Very large datasets**:
   - Neural methods can learn complex patterns from data
   - PFC relies on physics-based heuristics

2. **Non-topological features**:
   - If binding depends on sequence/evolution
   - Neural methods can learn these patterns

3. **Pre-trained models**:
   - AlphaFold embeddings are pre-trained on massive data
   - May capture patterns PFC misses

### But for Binding Pockets...

Binding pockets are **primarily topological**:
- Cavities (H₂)
- Accessibility (SASA)
- Hydrophobicity (chemistry)

PFC is designed for this!

## 8. Conclusion

**PFC should beat neural methods because:**

1. ✅ **Binding pockets are topological** (H₂ cavities)
2. ✅ **PFC explicitly captures topology** (persistence diagrams)
3. ✅ **PFC is protein-aware** (heterogeneous balls, residue landmarks)
4. ✅ **Neural methods miss explicit topology** (geometric focus)
5. ✅ **PFC's design targets binding pockets** (every choice helps)

**Expected result**: PFC AUROC ~0.85, Neural methods ~0.72-0.75

**Gap**: +10-13% from explicit topology + protein awareness

