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


## Algorithm Comparison (v0.7.0+)

Not sure which algorithm to use? Let ClusterTK compare them for you!

### Basic Comparison

```python
# Compare all algorithms
results = pipeline.compare_algorithms(
    X=df,
    feature_columns=['feature1', 'feature2', 'feature3'],
    algorithms=['kmeans', 'gmm', 'hierarchical', 'dbscan'],
    n_clusters_range=(2, 8)
)

# View comparison table
print(results['comparison'])
#    algorithm  n_clusters  silhouette  calinski_harabasz  davies_bouldin
# 0     kmeans           4    0.650394        1076.898364        0.512246
# 1        gmm           4    0.650394        1076.898364        0.512246
# 2 hierarchical          4    0.650394        1076.898364        0.512246
# 3     dbscan           4    0.623707         735.818803        1.578299

# Get recommendation
print(f"Best algorithm: {results['best_algorithm']}")
print(f"Optimal clusters: {results['best_n_clusters']}")
print(f"Silhouette score: {results['best_score']:.3f}")
```

### Visualize Comparison

```python
# Create comparison visualization
pipeline.plot_algorithm_comparison(
    comparison_results=results,
    title='Algorithm Performance Comparison'
)
```

This creates a two-panel figure:
- **Left panel**: Metrics comparison (Silhouette, Calinski-Harabasz, Davies-Bouldin)
- **Right panel**: Optimal cluster counts for each algorithm

### How It Works

1. **Same Preprocessing**: All algorithms use identical preprocessing settings from the pipeline
2. **Range Testing**: Each algorithm is tested across the specified n_clusters_range
3. **Optimal Selection**: For each algorithm, the k with the best silhouette score is selected
4. **Weighted Scoring**: Algorithms are ranked using weighted metrics:
   - Silhouette: 40% (higher is better)
   - Calinski-Harabasz: 30% (higher is better)
   - Davies-Bouldin: 30% (lower is better)

### Customize Comparison

```python
# Test specific algorithms only
results = pipeline.compare_algorithms(
    X=df,
    feature_columns=features,
    algorithms=['kmeans', 'gmm'],  # Only these two
    n_clusters_range=(3, 6)  # Narrower range
)

# Use specific metrics
results = pipeline.compare_algorithms(
    X=df,
    feature_columns=features,
    algorithms=['kmeans', 'gmm', 'hierarchical'],
    metrics=['silhouette', 'calinski_harabasz']  # Skip davies_bouldin
)

# Access detailed results
for algo, details in results['detailed_results'].items():
    if 'error' not in details:
        print(f"{algo}: {details['n_clusters']} clusters")
        print(f"  Metrics: {details['metrics']}")
```

### Use Case: Auto-Select Best Algorithm

```python
# Step 1: Compare algorithms
results = pipeline.compare_algorithms(
    X=df,
    feature_columns=features
)

# Step 2: Create new pipeline with best settings
best_pipeline = ClusterAnalysisPipeline(
    clustering_algorithm=results['best_algorithm'],
    n_clusters=results['best_n_clusters'],
    # ... other settings ...
)

# Step 3: Fit with optimal configuration
best_pipeline.fit(df, feature_columns=features)
```

### Tips

- **Large datasets**: Use `algorithms=['kmeans']` first for speed, then compare others on a sample
- **Unknown k**: Use wide range like `(2, 15)` to explore thoroughly
- **Similar scores**: If algorithms have similar scores, choose based on interpretability (K-Means > GMM > Hierarchical > DBSCAN)
- **DBSCAN**: May fail on some datasets (no clusters found). Check `'error'` in detailed_results

## Next Steps

- [Evaluation](evaluation.md) - Assess cluster quality
- [Interpretation](interpretation.md) - Understand clusters
- [Visualization](visualization.md) - Visualize results
