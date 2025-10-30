# Clustering Algorithms

ClusterTK supports multiple clustering algorithms, each suited for different data patterns and use cases.

## Overview

Available algorithms:
- **K-Means**: Fast, works well with spherical clusters
- **GMM** (Gaussian Mixture Model): Probabilistic, handles elliptical clusters
- **Hierarchical**: Creates dendrograms, no need to specify k
- **DBSCAN**: Density-based, finds arbitrary shapes and noise
- **HDBSCAN** (v0.8.0+): Advanced density-based, handles varying densities

## Customizing Algorithm Parameters (v0.12.0+)

All clustering algorithms can be customized using the `clustering_params` parameter. This allows you to pass any algorithm-specific parameters:

```python
from clustertk import ClusterAnalysisPipeline

# K-Means with custom parameters
pipeline = ClusterAnalysisPipeline(
    clustering_algorithm='kmeans',
    n_clusters=5,
    clustering_params={
        'n_init': 20,           # More initializations for stability
        'max_iter': 500,        # More iterations if needed
        'algorithm': 'elkan'    # Faster for well-defined clusters
    },
    random_state=42
)

# HDBSCAN with custom parameters
pipeline = ClusterAnalysisPipeline(
    clustering_algorithm='hdbscan',
    clustering_params={
        'min_cluster_size': 50,              # Minimum cluster size
        'min_samples': 10,                    # Core points threshold
        'cluster_selection_method': 'eom',    # Excess of mass
        'metric': 'manhattan'                 # Distance metric
    }
)

# DBSCAN with custom parameters
pipeline = ClusterAnalysisPipeline(
    clustering_algorithm='dbscan',
    clustering_params={
        'eps': 0.5,           # Neighborhood radius
        'min_samples': 5,     # Minimum points per cluster
        'metric': 'euclidean' # Distance metric
    }
)

pipeline.fit(df, feature_columns=features)
```

**How it works:**
- Parameters in `clustering_params` are passed directly to the algorithm
- User parameters override default values
- See each algorithm's section for commonly customized parameters

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

**Customizable parameters:**
```python
clustering_params={
    'n_init': 10,          # Number of initializations (default: 10)
    'max_iter': 300,       # Maximum iterations (default: 300)
    'tol': 1e-4,           # Convergence tolerance (default: 1e-4)
    'algorithm': 'lloyd'   # 'lloyd', 'elkan', or 'auto' (default: 'lloyd')
}
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

**Customizable parameters:**
```python
clustering_params={
    'covariance_type': 'full',  # 'full', 'tied', 'diag', 'spherical' (default: 'full')
    'n_init': 1,                # Number of initializations (default: 1)
    'max_iter': 100,            # Maximum iterations (default: 100)
    'tol': 1e-3,                # Convergence tolerance (default: 1e-3)
    'reg_covar': 1e-6           # Regularization (default: 1e-6)
}
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

**Customizable parameters:**
```python
clustering_params={
    'linkage': 'ward',        # 'ward', 'complete', 'average', 'single' (default: 'ward')
    'metric': 'euclidean',    # Distance metric (default: 'euclidean')
    'memory': None,           # Path to cache dendrogram (default: None)
    'connectivity': None      # Connectivity constraints (default: None)
}
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

# Check noise statistics (v0.12.0+)
print(f"Found {pipeline.n_clusters_} clusters")
print(f"Noise points: {pipeline.metrics_['n_noise']}")
print(f"Noise ratio: {pipeline.metrics_['noise_ratio']:.2%}")
```

**Customizable parameters:**
```python
clustering_params={
    'eps': 0.5,              # Neighborhood radius (default: auto-estimated)
    'min_samples': 5,        # Minimum points per cluster (default: auto-estimated)
    'metric': 'euclidean',   # Distance metric (default: 'euclidean')
    'algorithm': 'auto',     # 'auto', 'ball_tree', 'kd_tree', 'brute' (default: 'auto')
    'leaf_size': 30,         # Leaf size for tree algorithms (default: 30)
    'p': None                # Power for Minkowski metric (default: None)
}
```

**Parameters:**
- `eps`: Maximum distance between points (auto-estimated if not provided)
- `min_samples`: Minimum points to form cluster (auto-estimated if not provided)

**Pros:**
- Finds arbitrary-shaped clusters
- Automatically detects outliers
- No need to specify number of clusters

**Cons:**
- Struggles with varying density clusters
- Sensitive to parameters
- Slower on large datasets


## HDBSCAN (v0.8.0+)

Best for: Varying density clusters, automatic cluster detection, outlier identification.

HDBSCAN (Hierarchical DBSCAN) is an advanced density-based algorithm that improves upon DBSCAN by building a cluster hierarchy and automatically selecting the most stable clusters.

```python
pipeline = ClusterAnalysisPipeline(
    clustering_algorithm='hdbscan'
    # n_clusters not needed - auto-detected
)

pipeline.fit(df, feature_columns=features)

# HDBSCAN provides cluster membership probabilities
probs = pipeline.model_.probabilities_
weak_members = probs < 0.5  # Points weakly assigned

# Check noise statistics (v0.12.0+)
print(f"Found {pipeline.n_clusters_} clusters")
print(f"Noise points: {pipeline.metrics_['n_noise']}")
print(f"Noise ratio: {pipeline.metrics_['noise_ratio']:.2%}")
print(f"Weak cluster members: {weak_members.sum()}")
```

**Customizable parameters (v0.12.0+):**
```python
clustering_params={
    'min_cluster_size': 50,              # Minimum cluster size (default: auto = sqrt(n_samples))
    'min_samples': 10,                    # Core points threshold (default: auto = min_cluster_size)
    'cluster_selection_method': 'eom',    # 'eom' or 'leaf' (default: 'eom')
    'metric': 'euclidean',                # Distance metric (default: 'euclidean')
    'alpha': 1.0,                         # Distance scaling (default: 1.0)
    'cluster_selection_epsilon': 0.0,     # DBSCAN-like epsilon (default: 0.0)
    'algorithm': 'best',                  # 'best', 'generic', 'prims_kdtree', etc. (default: 'best')
    'leaf_size': 40,                      # Leaf size for tree algorithms (default: 40)
    'approx_min_span_tree': True          # Approximate MST (default: True)
}
```

**Example: Custom min_cluster_size for large datasets**
```python
# For large datasets, increase min_cluster_size to avoid too many small clusters
pipeline = ClusterAnalysisPipeline(
    clustering_algorithm='hdbscan',
    clustering_params={
        'min_cluster_size': 100,  # Require at least 100 samples per cluster
        'min_samples': 20         # Higher threshold for core points
    }
)
pipeline.fit(df, feature_columns=features)
```

**Key Features:**
- Handles clusters of varying densities
- More robust to parameter choices than DBSCAN
- Provides cluster membership probabilities
- Automatic cluster stability assessment

**Parameters:**
- `min_cluster_size`: Minimum points for a cluster (auto: sqrt(n_samples), or custom via clustering_params)
- `min_samples`: Minimum points in neighborhood (auto: equals min_cluster_size, or custom via clustering_params)

**Pros:**
- Handles varying density clusters (DBSCAN limitation)
- Fewer parameters to tune
- Provides cluster quality metrics (persistence)
- More robust than DBSCAN

**Cons:**
- Requires hdbscan library: `pip install clustertk[extras]`
- Slower than DBSCAN
- More complex algorithm

**When to use HDBSCAN vs DBSCAN:**
- Use HDBSCAN when clusters have varying densities
- Use DBSCAN when all clusters have similar density
- HDBSCAN is generally more reliable but slower

**Example with probability threshold:**

```python
from clustertk import ClusterAnalysisPipeline

pipeline = ClusterAnalysisPipeline(clustering_algorithm='hdbscan')
pipeline.fit(df, feature_columns=features)

# Identify confident vs uncertain assignments
probs = pipeline.model_.probabilities_
confident = probs > 0.7  # High confidence
uncertain = (probs > 0.3) & (probs <= 0.7)  # Medium confidence
noise = probs <= 0.3  # Low confidence (likely noise)

print(f"Confident: {confident.sum()} samples")
print(f"Uncertain: {uncertain.sum()} samples")
print(f"Noise: {noise.sum()} samples")
```

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
