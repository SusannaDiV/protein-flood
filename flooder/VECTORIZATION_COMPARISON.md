# Vectorization: Persistence Images vs Learnable DeepSets

## Quick Answer

**Persistence Images** are recommended as the **starting point** because they're:
- ✅ **Simpler**: No training required, works immediately
- ✅ **Interpretable**: Easy to visualize and understand
- ✅ **Proven**: Widely used in TDA literature
- ✅ **Safe choice**: Easy to justify in papers

**Learnable DeepSets** are **stronger** but:
- ⚠️ **Require training**: Need labeled data
- ⚠️ **More complex**: Hyperparameter tuning, architecture design
- ⚠️ **Less interpretable**: Black box representation
- ✅ **More expressive**: Can learn optimal features
- ✅ **Potentially better**: May outperform fixed representations

## Detailed Comparison

### Persistence Images

#### How They Work
1. Place a grid over the persistence diagram
2. Apply Gaussian kernel to each `(birth, death)` point
3. Sum contributions to create a fixed-size image
4. Flatten image to feature vector

#### Advantages
- ✅ **No training needed**: Works immediately with any dataset
- ✅ **Interpretable**: Can visualize the image, see where features are
- ✅ **Fixed size**: Same dimension for all proteins (good for MLPs)
- ✅ **Proven**: Used in many TDA papers, easy to cite
- ✅ **Fast**: Simple computation, no neural network overhead
- ✅ **Stable**: Deterministic, reproducible

#### Disadvantages
- ❌ **Fixed representation**: Can't adapt to data
- ❌ **Information loss**: Grid discretization may lose fine details
- ❌ **Hyperparameters**: Need to choose grid size, bandwidth
- ❌ **Less expressive**: May not capture complex patterns

#### Implementation
```python
from gudhi.representations import PersistenceImage

# Create persistence image
pi = PersistenceImage(
    bandwidth=1.0,  # Gaussian width
    weight=lambda x: x[1] - x[0],  # Persistence (death - birth)
    resolution=[20, 20]  # Grid size
)

# Convert diagrams to images
H0_img = pi.fit_transform([H0_diagram])  # (1, 400) for 20x20
H1_img = pi.fit_transform([H1_diagram])
H2_img = pi.fit_transform([H2_diagram])

# Concatenate for ML input
features = np.concatenate([H0_img, H1_img, H2_img], axis=1)  # (1, 1200)
```

### Learnable DeepSets

#### How They Work
1. Transform each `(birth, death)` point with MLP `φ`
2. Aggregate (sum/mean/max) all transformed points
3. Final MLP `ρ` produces feature vector
4. **Learnable**: Both `φ` and `ρ` are trained end-to-end

#### Advantages
- ✅ **More expressive**: Can learn optimal feature extraction
- ✅ **Adaptive**: Learns from data, not fixed grid
- ✅ **Potentially better**: May outperform fixed representations
- ✅ **Flexible**: Can incorporate additional features (e.g., persistence, location)
- ✅ **Novel**: More interesting for research papers

#### Disadvantages
- ❌ **Requires training**: Need labeled data (binding sites, functions, etc.)
- ❌ **More complex**: Architecture design, hyperparameter tuning
- ❌ **Less interpretable**: Hard to understand what it learned
- ❌ **Variable input size**: Need to handle variable number of points
- ❌ **Computational cost**: Training time, GPU memory

#### Implementation
```python
import torch
import torch.nn as nn

class PersistenceDeepSets(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=64, output_dim=128):
        super().__init__()
        # Point-wise transformation: (birth, death) -> feature
        self.phi = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        # Aggregation: sum of all features
        # Final transformation: aggregated -> output
        self.rho = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )
    
    def forward(self, diagram):
        # diagram: (N, 2) tensor of (birth, death) pairs
        if len(diagram) == 0:
            return torch.zeros(self.rho[-1].out_features)
        
        # Transform each point
        features = self.phi(diagram)  # (N, hidden_dim)
        
        # Aggregate (sum)
        aggregated = features.sum(dim=0)  # (hidden_dim,)
        
        # Final transformation
        output = self.rho(aggregated)  # (output_dim,)
        return output

# Usage
model = PersistenceDeepSets()
H0_features = model(torch.tensor(H0_diagram))  # (128,)
H1_features = model(torch.tensor(H1_diagram))  # (128,)
H2_features = model(torch.tensor(H2_diagram))  # (128,)

# Concatenate
features = torch.cat([H0_features, H1_features, H2_features])  # (384,)
```

## When to Use Each

### Use Persistence Images When:
- ✅ **No labeled data**: Can't train a model
- ✅ **Quick prototyping**: Want to test ideas fast
- ✅ **Interpretability matters**: Need to understand features
- ✅ **Baseline needed**: Want a standard comparison
- ✅ **Small dataset**: Not enough data to train DeepSets
- ✅ **Paper justification**: Easy to cite standard approach

### Use Learnable DeepSets When:
- ✅ **Have labeled data**: Binding sites, functions, properties
- ✅ **Performance matters**: Need best possible accuracy
- ✅ **Large dataset**: Enough data to train effectively
- ✅ **Research focus**: Want to show novel contribution
- ✅ **Complex patterns**: Fixed representations insufficient
- ✅ **End-to-end learning**: Want to learn from raw diagrams

## Hybrid Approach

You can also use **both**:

1. **Start with persistence images** for baseline
2. **Add DeepSets** for comparison
3. **Combine features** from both
4. **Show improvement** from learnable approach

```python
# Combine both representations
pi_features = persistence_image_features  # (1200,)
deepset_features = deepsets_features      # (384,)
combined = torch.cat([pi_features, deepset_features])  # (1584,)
```

## Recommendation for Your Paper

### Option 1: Start Simple (Recommended)
1. **Implement persistence images first**
   - Quick to implement
   - Works immediately
   - Good baseline
   - Easy to justify

2. **Then add DeepSets** (if you have data)
   - Show improvement over baseline
   - Demonstrate learning capability
   - More interesting contribution

### Option 2: Go Straight to DeepSets
- If you have labeled data (binding sites, functions)
- If performance is critical
- If you want to emphasize learning aspect

### Option 3: Both (Best for Paper)
- Show persistence images as baseline
- Show DeepSets as improvement
- Compare both approaches
- Demonstrate that learning helps

## Implementation Priority

### Phase 1: Persistence Images (Quick Win)
- ✅ Easy to implement
- ✅ Works immediately
- ✅ Good baseline
- ✅ Can publish with this

### Phase 2: DeepSets (If You Have Data)
- ⚠️ Requires labeled data
- ⚠️ More implementation work
- ✅ Better performance potential
- ✅ More novel contribution

## Code Complexity

| Aspect | Persistence Images | DeepSets |
|--------|-------------------|----------|
| **Lines of code** | ~10-20 | ~50-100 |
| **Dependencies** | gudhi | torch, training loop |
| **Time to implement** | 1-2 hours | 1-2 days |
| **Training needed** | No | Yes |
| **Hyperparameters** | Grid size, bandwidth | Architecture, learning rate, etc. |

## Conclusion

**For your paper**:
- **Start with persistence images**: Quick, proven, easy to justify
- **Add DeepSets if you have data**: Shows learning capability, potentially better
- **Compare both**: Demonstrates that learning helps

**Persistence images are NOT worse** - they're just **simpler and safer**. DeepSets are **potentially better** but require more work and data.

The choice depends on:
1. **Do you have labeled data?** → DeepSets possible
2. **How much time?** → Images faster
3. **What's the focus?** → Images for baseline, DeepSets for learning

I'd recommend: **Implement persistence images first**, then add DeepSets as an improvement if you have the data and time.

