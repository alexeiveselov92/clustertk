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
- **HDBSCAN**: Varying density clusters, automatic detection

See [Algorithm Comparison](user_guide/clustering.md#algorithm-comparison) for detailed comparison.

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

## Feature Selection (v0.16.0+)

### Q: I have 30 features - should I use all of them for clustering?

**Short answer:** Not necessarily! More features ≠ better clustering.

**Problem:** Irrelevant features add noise and dilute clustering signal (curse of dimensionality).

**Solution:** Use iterative feature selection:

```python
# 1. Fit on all features
pipeline.fit(df)  # 30 features

# 2. Try refitting with top 10 features
comparison = pipeline.refit_with_top_features(
    n_features=10,
    importance_method='permutation'
)

# 3. If improved, update pipeline
if comparison['metrics_improved']:
    pipeline.refit_with_top_features(n_features=10, update_pipeline=True)
```

See [Feature Selection Guide](user_guide/interpretation.md#feature-selection-for-better-clustering-v0160) for details.

### Q: Which importance method should I use for feature selection?

**Quick answer:** Use `'permutation'` (default and recommended).

**Three methods:**
- **`'permutation'`** - Measures impact on clustering quality (silhouette). Best for finding features that improve clustering. ✅ **Recommended**
- **`'contribution'`** - Variance ratio (fast statistical measure). Good for quick insights.
- **`'pca'`** - PCA loadings (only if `dim_reduction='pca'`). Good for PCA interpretation.

```python
comparison = pipeline.refit_with_top_features(
    n_features=10,
    importance_method='permutation'  # Best choice
)
```

### Q: How many features should I keep?

**It depends on your data**, but here's a guideline:

1. **Try multiple values**: Test [15, 10, 8, 5] and compare metrics
2. **Don't over-reduce**: Keep at least 5-8 features to retain information
3. **Look for improvement**: If metrics don't improve, keep all features

```python
# Try different feature counts
for n in [15, 10, 8, 5]:
    comparison = pipeline.refit_with_top_features(
        n_features=n,
        compare_metrics=False,
        update_pipeline=False
    )
    print(f"Top {n}: Silhouette={comparison['refitted_metrics']['silhouette']:.3f}")

# Use best performing count
```

**Real example:** Dataset with 20 features (3 meaningful + 7 derived + 10 noise):
- All 20 features → Silhouette: 0.106
- Top 8 features (permutation) → Silhouette: 0.223 (**+105% improvement!** 🚀)

### Q: When should I NOT use feature selection?

Skip feature selection if:
- **Small feature count** (<10 features) - already manageable
- **All features important** - domain knowledge says all features matter
- **Metrics don't improve** - feature selection hurts quality
- **After correlation filter** - SmartCorrelationFilter already removed redundant features

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

DBSCAN parameters need tuning. ClusterTK auto-estimates them, but you may need custom values (v0.12.0+):

```python
# Use clustering_params to customize
pipeline = ClusterAnalysisPipeline(
    clustering_algorithm='dbscan',
    clustering_params={
        'eps': 0.5,           # Try different neighborhood radius
        'min_samples': 5      # Try different minimum points
    }
)
```

**Alternative: Try HDBSCAN**

HDBSCAN is more robust to parameter choices:

```python
pipeline = ClusterAnalysisPipeline(
    clustering_algorithm='hdbscan',
    clustering_params={
        'min_cluster_size': 50,  # Adjust based on expected cluster size
        'min_samples': 10
    }
)
```

### Q: What are noise points and how do I check them? (v0.12.0+)

DBSCAN and HDBSCAN label outliers as "noise points" (cluster -1). Check noise statistics:

```python
pipeline = ClusterAnalysisPipeline(clustering_algorithm='hdbscan')
pipeline.fit(df, feature_columns=features)

# Check noise statistics
print(f"Found {pipeline.n_clusters_} clusters")
print(f"Noise points: {pipeline.metrics_['n_noise']}")
print(f"Noise ratio: {pipeline.metrics_['noise_ratio']:.2%}")

# Extract noise points
noise_mask = pipeline.labels_ == -1
noise_points = df[noise_mask]
```

**Interpretation:**
- **<5% noise**: Excellent, very few outliers
- **5-10% noise**: Good, reasonable outlier detection
- **10-20% noise**: Moderate, many outliers
- **>20% noise**: High noise, consider adjusting parameters or using different algorithm

## Advanced

### Q: How do I customize algorithm parameters? (v0.12.0+)

Use the `clustering_params` parameter to pass custom parameters to any algorithm:

```python
# HDBSCAN with custom parameters
pipeline = ClusterAnalysisPipeline(
    clustering_algorithm='hdbscan',
    clustering_params={
        'min_cluster_size': 50,
        'min_samples': 10,
        'cluster_selection_method': 'eom',
        'metric': 'manhattan'
    }
)

# K-Means with custom parameters
pipeline = ClusterAnalysisPipeline(
    clustering_algorithm='kmeans',
    n_clusters=5,
    clustering_params={
        'n_init': 20,
        'max_iter': 500,
        'algorithm': 'elkan'
    }
)

# DBSCAN with custom parameters
pipeline = ClusterAnalysisPipeline(
    clustering_algorithm='dbscan',
    clustering_params={
        'eps': 0.5,
        'min_samples': 5,
        'metric': 'manhattan'
    }
)
```

**How it works:**
- Parameters in `clustering_params` are passed directly to the underlying sklearn/hdbscan algorithm
- User parameters override default values
- Works with any algorithm: kmeans, gmm, hierarchical, dbscan, hdbscan

See [Clustering Algorithms](user_guide/clustering.md#customizing-algorithm-parameters-v0120) for all available parameters.

### Q: When should I use clustering_params?

Use `clustering_params` when:

1. **HDBSCAN min_cluster_size**: Control minimum cluster size for your use case
   ```python
   clustering_params={'min_cluster_size': 100}  # Require at least 100 samples per cluster
   ```

2. **DBSCAN eps/min_samples**: Fine-tune density thresholds
   ```python
   clustering_params={'eps': 0.3, 'min_samples': 10}
   ```

3. **K-Means stability**: Increase initializations for more stable results
   ```python
   clustering_params={'n_init': 50}  # More initializations
   ```

4. **Distance metrics**: Use non-Euclidean distances
   ```python
   clustering_params={'metric': 'manhattan'}  # L1 distance
   ```

5. **Performance tuning**: Adjust algorithm-specific optimizations
   ```python
   clustering_params={'algorithm': 'elkan'}  # Faster K-Means variant
   ```

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
