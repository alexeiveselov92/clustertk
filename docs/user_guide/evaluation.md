# Evaluation

Assess clustering quality using multiple metrics.

## Metrics

### Silhouette Score
Range: [-1, 1]. Higher is better. 0.5+ indicates good clustering.

### Calinski-Harabasz Index  
Higher is better. Measures cluster separation vs compactness.

### Davies-Bouldin Index
Lower is better. Measures average similarity between clusters.

## Auto-detect Optimal K

```python
pipeline = ClusterAnalysisPipeline(
    n_clusters=None,           # Auto-detect
    n_clusters_range=(2, 10)   # Search range
)
pipeline.fit(df, feature_columns=features)
print(f"Optimal k: {pipeline.n_clusters_}")
```

Uses voting between metrics to select optimal number of clusters.

## Access Metrics

```python
metrics = pipeline.metrics_
print(f"Silhouette: {metrics['silhouette']:.3f}")
print(f"Calinski-Harabasz: {metrics['calinski_harabasz']:.1f}")
print(f"Davies-Bouldin: {metrics['davies_bouldin']:.3f}")
```

## Stability Analysis

Assess how reliable your clustering results are using bootstrap resampling.

### Quick Stability Check

```python
# Analyze clustering stability
results = pipeline.analyze_stability(
    n_iterations=100,      # Number of bootstrap iterations
    sample_fraction=0.8,   # 80% of data per iteration
    random_state=42
)

print(f"Overall stability: {results['overall_stability']:.3f}")
print(f"Mean ARI: {results['mean_ari']:.3f}")

# View per-cluster stability
print(results['cluster_stability'])
# Output:
#    cluster  stability  size
# 0        2   0.956234    98
# 1        1   0.943567    99
# 2        0   0.921456   103
```

### Understanding Stability Metrics

**Overall Stability** (0-1, higher = better)
- Measured via pairwise Adjusted Rand Index (ARI)
- Shows consistency of clustering across bootstrap samples
- **> 0.8**: Very stable clustering
- **0.6-0.8**: Reasonably stable
- **< 0.6**: Unstable, results may not be reliable

**Per-Cluster Stability**
- Proportion of sample pairs that stay together across iterations
- Identifies which clusters are well-defined vs fuzzy
- **> 0.7**: Stable cluster
- **< 0.5**: Unstable cluster (consider merging or removing)

**Sample Confidence Scores**
- How consistently each point is assigned to same cluster
- Useful for identifying boundary points

```python
# Get sample confidence scores
confidence = results['sample_confidence']

# Find unstable samples
unstable_mask = confidence < 0.5
unstable_samples = df[unstable_mask]
print(f"Unstable samples: {unstable_mask.sum()} ({unstable_mask.mean()*100:.1f}%)")

# Identify stable vs unstable clusters
print(f"Stable clusters: {results['stable_clusters']}")
print(f"Unstable clusters: {results['unstable_clusters']}")
```

### When to Use Stability Analysis

**Use when:**
- Presenting results to stakeholders (builds confidence)
- Results will drive important decisions
- Data has noise or outliers
- Cluster boundaries seem unclear
- You need to choose between similar clusterings

**Interpretation:**
- High overall stability (>0.8): Trust the results
- Medium stability (0.6-0.8): Results reasonable but validate
- Low stability (<0.6): Try different algorithms/parameters
- Some unstable clusters: Consider removing or merging them
- Many unstable samples: May need better features or more data

### Advanced Usage

```python
# Standalone usage
from clustertk.evaluation import ClusterStabilityAnalyzer
from clustertk.clustering import KMeansClustering

analyzer = ClusterStabilityAnalyzer(
    n_iterations=200,
    sample_fraction=0.75,
    verbose=True
)

clusterer = KMeansClustering(n_clusters=4)
results = analyzer.analyze(X, clusterer)

# Get stable and unstable samples
stable_samples = analyzer.get_stable_samples(threshold=0.7)
unstable_samples = analyzer.get_unstable_samples(threshold=0.5)

print(f"Stable: {len(stable_samples)}, Unstable: {len(unstable_samples)}")
```

### Quick Function

```python
from clustertk.evaluation import quick_stability_analysis
from clustertk.clustering import GMMClustering

results = quick_stability_analysis(
    X=df,
    clusterer=GMMClustering(n_clusters=3),
    n_iterations=50,
    random_state=42
)

print(f"Stability: {results['overall_stability']:.3f}")
```

### Performance Considerations

- Default 100 iterations is good for most cases
- Use 50 iterations for quick checks
- Use 200+ iterations for critical analyses
- Reduce `sample_fraction` if data is large (0.6-0.7)
- Analysis time scales linearly with iterations
