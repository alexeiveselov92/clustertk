# Frequently Asked Questions

## Installation & Setup

### Q: How do I install ClusterTK?

```bash
# Basic installation
pip install clustertk

# With visualization
pip install clustertk[viz]
```

### Q: Do I need visualization dependencies?

No, they're optional. Install only if you want to use `.plot_*()` methods.

### Q: What Python versions are supported?

Python 3.8, 3.9, 3.10, and 3.11.

## Usage

### Q: How do I choose the number of clusters?

Set `n_clusters=None` for automatic detection:

```python
pipeline = ClusterAnalysisPipeline(n_clusters=None, n_clusters_range=(2, 10))
```

### Q: Which clustering algorithm should I use?

- **K-Means**: Fast, spherical clusters, known number of clusters
- **GMM**: Elliptical clusters, probabilistic assignments
- **Hierarchical**: Explore cluster hierarchy
- **DBSCAN**: Arbitrary shapes, outliers present

### Q: How do I handle missing values?

```python
pipeline = ClusterAnalysisPipeline(
    handle_missing='median'  # or 'mean', 'drop', custom function
)
```

### Q: Should I scale my data?

Yes! Always scale data before clustering:

```python
pipeline = ClusterAnalysisPipeline(
    scaling='robust'  # or 'standard', 'minmax'
)
```

## Results

### Q: How do I access cluster assignments?

```python
labels = pipeline.labels_  # Cluster assignments for each sample
```

### Q: How do I get cluster profiles?

```python
profiles = pipeline.cluster_profiles_  # Mean feature values per cluster
```

### Q: What's a good silhouette score?

- 0.7-1.0: Strong structure
- 0.5-0.7: Reasonable structure
- 0.25-0.5: Weak structure
- <0.25: No meaningful structure

## Visualization

### Q: Why do plots display twice in Jupyter?

This was fixed in v0.4.1. Update to latest version:

```bash
pip install --upgrade clustertk
```

### Q: How do I show multiple plots?

Use `display()`:

```python
from IPython.display import display

display(pipeline.plot_clusters_2d())
display(pipeline.plot_cluster_heatmap())
```

### Q: Can I customize plots?

Yes, most plots return matplotlib Figure objects:

```python
fig = pipeline.plot_clusters_2d()
fig.suptitle('My Custom Title')
fig.savefig('my_plot.png', dpi=300)
```

## Export

### Q: How do I export results?

```python
# CSV with data + labels
pipeline.export_results('results.csv')

# JSON with metadata
pipeline.export_results('results.json', format='json')

# HTML report
pipeline.export_report('report.html')
```

### Q: How do I save a fitted pipeline?

```python
# Save
pipeline.save_pipeline('my_pipeline.joblib')

# Load
from clustertk import ClusterAnalysisPipeline
loaded = ClusterAnalysisPipeline.load_pipeline('my_pipeline.joblib')
```

## Performance

### Q: How large datasets can ClusterTK handle?

Depends on algorithm:
- K-Means: 100k+ samples easily
- GMM: 10k-50k samples
- Hierarchical: <10k samples (memory intensive)
- DBSCAN: 10k-50k samples

### Q: How can I speed up clustering?

1. Reduce dimensionality with PCA
2. Use K-Means instead of GMM/Hierarchical
3. Sample data for large datasets
4. Reduce `n_clusters_range` when auto-detecting

## Errors & Troubleshooting

### Q: "Visualization dependencies not installed"

Install viz extras:

```bash
pip install clustertk[viz]
```

### Q: "No clustering results available"

Run `.fit()` before accessing results:

```python
pipeline.fit(df, feature_columns=features)
labels = pipeline.labels_  # Now available
```

### Q: Poor clustering quality (low silhouette)

Try:
1. Check data quality (outliers, missing values)
2. Remove irrelevant features
3. Try different algorithms
4. Adjust number of clusters
5. Scale data properly

### Q: DBSCAN finds only one cluster or all noise

DBSCAN parameters need tuning. ClusterTK auto-estimates them, but you may need custom values:

```python
from clustertk.clustering import DBSCANClustering

custom_dbscan = DBSCANClustering(eps=0.5, min_samples=5)
pipeline = ClusterAnalysisPipeline(clustering_algorithm=custom_dbscan)
```

## Advanced

### Q: Can I use custom preprocessing?

Yes:

```python
def my_preprocessor(df):
    # Your custom logic
    return processed_df

pipeline = ClusterAnalysisPipeline(handle_missing=my_preprocessor)
```

### Q: Can I use custom clustering algorithms?

Yes, any scikit-learn compatible clusterer:

```python
from sklearn.cluster import SpectralClustering

custom_algo = SpectralClustering(n_clusters=5)
pipeline = ClusterAnalysisPipeline(clustering_algorithm=custom_algo)
```

### Q: How do I understand which features are most important?

Use feature importance analysis:

```python
# Analyze feature importance
results = pipeline.analyze_feature_importance(method='all')

# View permutation importance
print(results['permutation'].head(10))

# View feature contribution (variance ratio)
print(results['contribution'].head(10))

# SHAP values (if shap is installed)
if 'shap' in results:
    print(results['shap']['importance'].head(10))
```

See [Feature Importance](user_guide/interpretation.md#feature-importance-analysis) for details.

### Q: Which feature importance method should I use?

**Quick answer:** Start with `method='contribution'` (fast), then use `method='permutation'` for reliable ranking.

| Method | Best For | Speed | Requires |
|--------|----------|-------|----------|
| **contribution** | Quick statistical insights | Fast | - |
| **permutation** | Reliable feature ranking | Medium | sklearn |
| **shap** | Detailed analysis, interactions | Slow | pip install shap |

**Detailed comparison:**
- **Contribution** (variance ratio): Fast, shows how well each feature separates clusters statistically
- **Permutation**: Measures impact on clustering quality, more robust but slower
- **SHAP**: Most detailed, shows feature interactions, requires extra package

```python
# Use all methods and compare
results = pipeline.analyze_feature_importance(method='all')
```

### Q: Can I reduce features based on importance?

Yes! Use feature importance to identify key features:

```python
# Get feature importance
results = pipeline.analyze_feature_importance(method='permutation')
top_features = results['permutation'].head(5)['feature'].tolist()

# Re-run analysis with only top features
pipeline_focused = ClusterAnalysisPipeline(...)
pipeline_focused.fit(df, feature_columns=top_features)

# Compare quality
print(f"Original: {pipeline.metrics_['silhouette']:.3f}")
print(f"Focused: {pipeline_focused.metrics_['silhouette']:.3f}")
```

### Q: How do I cluster new data with a fitted pipeline?

```python
# Save pipeline
pipeline.save_pipeline('model.joblib')

# Later: load and use on new data
loaded = ClusterAnalysisPipeline.load_pipeline('model.joblib')

# Transform new data through same preprocessing steps
# Note: clustering assigns to nearest existing cluster
new_labels = loaded._clusterer.predict(loaded._pca_reducer.transform(
    loaded._scaler.transform(new_data)
))
```

## Contributing

### Q: How can I contribute?

1. Report bugs on [GitHub Issues](https://github.com/alexeiveselov92/clustertk/issues)
2. Suggest features on [GitHub Discussions](https://github.com/alexeiveselov92/clustertk/discussions)
3. Submit pull requests

### Q: Where can I get help?

- GitHub Issues: Bug reports
- GitHub Discussions: Questions and discussions
- Email: alexei.veselov92@gmail.com

## More Questions?

Check the [User Guide](user_guide/README.md) for detailed documentation or ask on [GitHub Discussions](https://github.com/alexeiveselov92/clustertk/discussions).

### How do I choose the best clustering algorithm?

Use the `compare_algorithms()` method to automatically compare multiple algorithms:

```python
results = pipeline.compare_algorithms(
    X=df,
    feature_columns=features,
    algorithms=['kmeans', 'gmm', 'hierarchical', 'dbscan']
)

print(f"Best algorithm: {results['best_algorithm']}")
pipeline.plot_algorithm_comparison(results)
```

The method tests each algorithm across different cluster counts and recommends the best one based on weighted scoring (40% Silhouette, 30% Calinski-Harabasz, 30% Davies-Bouldin).

**Quick guidelines:**
- **K-Means**: Fast, spherical clusters, known k
- **GMM**: Elliptical clusters, probabilistic
- **Hierarchical**: Dendrogram, no need to specify k
- **DBSCAN**: Arbitrary shapes, handles noise
