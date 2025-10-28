# Clustering Algorithms

ClusterTK supports multiple clustering algorithms, each suited for different data patterns and use cases.

## Overview

Available algorithms:
- **K-Means**: Fast, works well with spherical clusters
- **GMM** (Gaussian Mixture Model): Probabilistic, handles elliptical clusters
- **Hierarchical**: Creates dendrograms, no need to specify k
- **DBSCAN**: Density-based, finds arbitrary shapes and noise

## K-Means Clustering

Best for: Spherical clusters, large datasets, when you know the number of clusters.

```python
from clustertk import ClusterAnalysisPipeline

pipeline = ClusterAnalysisPipeline(
    clustering_algorithm='kmeans',
    n_clusters=5,
    random_state=42
)

pipeline.fit(df, feature_columns=features)
```

**Pros:**
- Fast and scalable
- Works well when clusters are spherical
- Easy to interpret

**Cons:**
- Assumes clusters are spherical and similar size
- Sensitive to initialization (use random_state)
- Must specify number of clusters

## Gaussian Mixture Model (GMM)

Best for: Elliptical clusters, probabilistic assignments, overlapping clusters.

```python
pipeline = ClusterAnalysisPipeline(
    clustering_algorithm='gmm',
    n_clusters=4,
    random_state=42
)

pipeline.fit(df, feature_columns=features)
```

**Pros:**
- Handles elliptical cluster shapes
- Provides probability of membership
- More flexible than K-Means

**Cons:**
- Slower than K-Means
- Can overfit with too many components
- Sensitive to initialization

## Hierarchical Clustering

Best for: Exploring cluster hierarchies, when number of clusters is unknown.

```python
pipeline = ClusterAnalysisPipeline(
    clustering_algorithm='hierarchical',
    n_clusters=3
)

pipeline.fit(df, feature_columns=features)
```

**Linkage methods:**
- `ward` (default): Minimizes variance
- `complete`: Maximum distance
- `average`: Average distance

**Pros:**
- No need to specify k initially
- Creates dendrogram for visualization
- Deterministic (no random initialization)

**Cons:**
- Slower than K-Means
- Can't undo merges
- Sensitive to outliers

## DBSCAN

Best for: Finding clusters of arbitrary shape, handling noise/outliers.

```python
pipeline = ClusterAnalysisPipeline(
    clustering_algorithm='dbscan'
    # n_clusters not needed - auto-detected
)

pipeline.fit(df, feature_columns=features)

# Noise points labeled as -1
noise_count = (pipeline.labels_ == -1).sum()
print(f"Noise points: {noise_count}")
```

**Parameters:**
- `eps`: Maximum distance between points (auto-estimated)
- `min_samples`: Minimum points to form cluster (auto-estimated)

**Pros:**
- Finds arbitrary-shaped clusters
- Automatically detects outliers
- No need to specify number of clusters

**Cons:**
- Struggles with varying density clusters
- Sensitive to parameters
- Slower on large datasets

## Automatic Algorithm Selection

Let ClusterTK find the optimal number of clusters:

```python
pipeline = ClusterAnalysisPipeline(
    clustering_algorithm='kmeans',
    n_clusters=None,              # Auto-detect
    n_clusters_range=(2, 10)      # Search range
)

pipeline.fit(df, feature_columns=features)

print(f"Optimal clusters: {pipeline.n_clusters_}")
```

The pipeline uses voting between multiple metrics:
- Silhouette score
- Calinski-Harabasz index
- Davies-Bouldin index

## Custom Clustering Algorithm

Use any scikit-learn compatible clusterer:

```python
from sklearn.cluster import SpectralClustering

custom_clusterer = SpectralClustering(n_clusters=4, random_state=42)

pipeline = ClusterAnalysisPipeline(
    clustering_algorithm=custom_clusterer
)

pipeline.fit(df, feature_columns=features)
```

## Comparing Algorithms

```python
algorithms = ['kmeans', 'gmm', 'hierarchical']
results = {}

for algo in algorithms:
    pipeline = ClusterAnalysisPipeline(
        clustering_algorithm=algo,
        n_clusters=5
    )
    pipeline.fit(df, feature_columns=features)

    results[algo] = {
        'silhouette': pipeline.metrics_['silhouette'],
        'calinski_harabasz': pipeline.metrics_['calinski_harabasz'],
        'davies_bouldin': pipeline.metrics_['davies_bouldin']
    }

import pandas as pd
print(pd.DataFrame(results).T)
```

## Best Practices

### 1. Choose Algorithm Based on Data

```python
# Spherical, well-separated clusters
clustering_algorithm='kmeans'

# Elliptical or overlapping clusters
clustering_algorithm='gmm'

# Arbitrary shapes, outliers present
clustering_algorithm='dbscan'

# Want to explore hierarchy
clustering_algorithm='hierarchical'
```

### 2. Always Use random_state

```python
# ✅ Reproducible
pipeline = ClusterAnalysisPipeline(
    clustering_algorithm='kmeans',
    random_state=42
)

# ❌ Different results each run
pipeline = ClusterAnalysisPipeline(
    clustering_algorithm='kmeans'
)
```

### 3. Validate Results

```python
# Check silhouette score (higher is better, 0.5+ is good)
print(f"Silhouette: {pipeline.metrics_['silhouette']:.3f}")

# Check cluster sizes (too small/large clusters may indicate issues)
sizes = pd.Series(pipeline.labels_).value_counts()
print(sizes)

# Visualize clusters
pipeline.plot_clusters_2d()
```

## Next Steps

- [Evaluation](evaluation.md) - Assess cluster quality
- [Interpretation](interpretation.md) - Understand clusters
- [Visualization](visualization.md) - Visualize results
