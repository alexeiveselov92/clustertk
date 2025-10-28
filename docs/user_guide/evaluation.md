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
