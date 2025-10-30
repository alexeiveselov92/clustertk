# Interpretation

Understand what each cluster represents.

## Cluster Profiles

```python
# Create profiles
pipeline.create_profiles()

# View profiles
print(pipeline.cluster_profiles_)

# Top distinguishing features per cluster
print(pipeline._profiler.top_features_)
```

## Feature Importance Analysis

Understand which features are most important for your clustering results using three different methods:

### Quick Feature Importance

```python
# Run all methods and get top features
results = pipeline.analyze_feature_importance(
    method='all',  # or 'permutation', 'contribution', 'shap'
    n_repeats=10,  # for permutation importance
    random_state=42
)

# View permutation importance results
print("\nPermutation Importance:")
print(results['permutation'].head(10))
# Output:
#        feature  importance       std
# 0    feature_1    0.204123  0.015234
# 1    feature_3    0.193456  0.012876
# 2    feature_2    0.177234  0.010987

# View feature contribution (variance ratio)
print("\nFeature Contribution:")
print(results['contribution'].head(10))
# Output:
#        feature  contribution
# 0    feature_1     12.145678
# 1    feature_3     11.123456
# 2    feature_0      6.067890

# SHAP values (if shap is installed)
if 'shap' in results:
    print("\nSHAP Importance:")
    print(results['shap']['importance'].head(10))
```

### Understanding the Methods

**1. Permutation Importance**
- Measures how much shuffling each feature **decreases clustering quality**
- Uses silhouette score as the quality metric
- Higher values = more important for cluster separation
- Best for: Understanding which features drive the clustering algorithm

```python
# Run only permutation importance
results = pipeline.analyze_feature_importance(method='permutation')
print(results['permutation'])
```

**2. Feature Contribution (Variance Ratio)**
- Measures **between-cluster variance / within-cluster variance**
- Statistical measure of how well each feature separates clusters
- Higher values = better cluster separation
- Best for: Statistical understanding of cluster structure

```python
# Run only contribution analysis
results = pipeline.analyze_feature_importance(method='contribution')
print(results['contribution'])
```

**3. SHAP Values** (requires `pip install shap`)
- Uses a Random Forest classifier to predict cluster labels
- Computes SHAP values to explain predictions
- Provides feature importance with interaction effects
- Best for: Understanding complex feature interactions

```python
# Run only SHAP analysis (requires shap package)
results = pipeline.analyze_feature_importance(method='shap')
print(results['shap']['importance'])

# Access raw SHAP values for custom visualization
shap_values = results['shap']['shap_values']
explainer = results['shap']['explainer']
```

### Standalone Usage

You can also use the feature importance analyzer independently:

```python
from clustertk.interpretation import FeatureImportanceAnalyzer

# Create analyzer
analyzer = FeatureImportanceAnalyzer(verbose=True)

# Analyze features
results = analyzer.analyze(
    X=df[feature_columns],
    labels=labels,
    method='all',
    n_repeats=10,
    random_state=42
)

# Get top N features
top_features = analyzer.get_top_features(method='permutation', n=5)
print(top_features)
```

### Quick Function

For rapid analysis:

```python
from clustertk.interpretation import quick_feature_importance

# Quick analysis with all methods
top_features = quick_feature_importance(
    X=df[feature_columns],
    labels=labels,
    method='all',
    n_top=10
)
print(top_features)
```

### When to Use Each Method

| Method | Best For | Speed | Requires |
|--------|----------|-------|----------|
| **Permutation** | General purpose, impact on quality | Medium | sklearn |
| **Contribution** | Statistical understanding | Fast | - |
| **SHAP** | Complex interactions, detailed analysis | Slow | shap package |

**Recommendations:**
- Start with **contribution** for quick insights
- Use **permutation** for reliable feature ranking
- Use **SHAP** when you need detailed explanations and have time

## Feature Selection for Better Clustering (v0.16.0+)

Beyond understanding which features are important, you can use this information to **improve clustering quality** by refitting with only the most relevant features.

### Why Feature Selection Matters

**Problem:** More features ≠ better clustering
- Irrelevant features add noise and dilute clustering signal
- PCA can't fix bad features, only compress them
- 10 good features often outperform 30 mixed features (curse of dimensionality)

**Solution:** Iterative feature selection finds optimal subset

### Get PCA Feature Importance

Shows which original features contribute most to PCA components:

```python
# After fitting with PCA
pipeline = ClusterAnalysisPipeline(dim_reduction='pca')
pipeline.fit(df)  # 30 features

# Get PCA feature importance
importance = pipeline.get_pca_feature_importance()
print(importance.head(10))

# Output:
#          feature  total_loading  relative_importance
# 0      feature_1       4.567890             0.145234
# 1      feature_5       4.123456             0.131234
# 2      feature_3       3.876543             0.123456
# ...

# Use top features for focused analysis
top_features = importance.head(5)['feature'].tolist()
```

**When to use:**
- You're using `dim_reduction='pca'`
- Want to understand which original features drive PCA structure
- Need quick interpretation of PCA-based clustering

### Refit with Top Features

Automatically refit clustering using only top N most important features:

```python
# Step 1: Fit on all features
pipeline = ClusterAnalysisPipeline(dim_reduction='pca')
pipeline.fit(df)  # 30 features → Silhouette: 0.42

# Step 2: Try refitting with top 10 features
comparison = pipeline.refit_with_top_features(
    n_features=10,
    importance_method='permutation',  # or 'contribution', 'pca'
    compare_metrics=True,
    update_pipeline=False  # Just compare first
)

# Output:
# ================================================================================
# REFITTING WITH TOP 10 FEATURES (method=permutation)
# ================================================================================
#
# Step 1: Selected top 10 features:
#   1. feature_5
#   2. feature_12
#   ...
#
# ================================================================================
# METRICS COMPARISON
# ================================================================================
# Metric                    Original        Refitted        Change
# --------------------------------------------------------------------------------
# Silhouette                0.4200          0.5800          +38.1% ↑
# Calinski Harabasz         245.67          387.45          +57.7% ↑
# Davies Bouldin            1.85            1.42            +23.2% ↓
# --------------------------------------------------------------------------------
# Overall improvement: +38.6%
#
# ✓ Metrics IMPROVED with top features!

# Step 3: If metrics improved, update pipeline
if comparison['metrics_improved']:
    print(f"Improvement: {comparison['weighted_improvement']:+.1%}")
    pipeline.refit_with_top_features(
        n_features=10,
        importance_method='permutation',
        update_pipeline=True  # Now update pipeline
    )
    # Pipeline now uses only top 10 features!
```

### Three Importance Methods

Choose the method that best fits your needs:

**1. Permutation (Recommended)**
- Measures impact on clustering **quality** (silhouette score)
- Most reliable for finding features that improve clustering
- Example: Feature that separates clusters well vs creates noise

```python
comparison = pipeline.refit_with_top_features(
    n_features=10,
    importance_method='permutation'  # Best for clustering quality
)
```

**2. Contribution**
- Uses variance ratio (between/within cluster variance)
- Fast statistical measure
- Example: Feature with high variance between clusters

```python
comparison = pipeline.refit_with_top_features(
    n_features=10,
    importance_method='contribution'  # Fast statistical measure
)
```

**3. PCA Loadings**
- Uses PCA component contributions
- Only works when `dim_reduction='pca'`
- Example: Feature with high loading on top PCA components

```python
comparison = pipeline.refit_with_top_features(
    n_features=10,
    importance_method='pca'  # Only if using PCA
)
```

### Comparison Results

The `refit_with_top_features()` method returns detailed comparison:

```python
comparison = pipeline.refit_with_top_features(n_features=10)

# Access results
print(comparison['top_features'])          # List of selected features
print(comparison['metrics_improved'])      # True if improved
print(comparison['weighted_improvement'])  # Overall improvement %
print(comparison['original_metrics'])      # Original clustering metrics
print(comparison['refitted_metrics'])      # New clustering metrics
print(comparison['improvement_summary'])   # Per-metric improvements
print(comparison['pipeline_updated'])      # Whether pipeline was updated

# Refitted clustering results (even if not updated)
refitted_labels = comparison['refitted_labels']
refitted_profiles = comparison['refitted_profiles']
```

### Complete Workflow Example

```python
# 1. Initial clustering on all features
pipeline = ClusterAnalysisPipeline(
    dim_reduction='pca',
    clustering_algorithm='kmeans',
    n_clusters=4
)
pipeline.fit(df)  # 25 features

print(f"Initial: {pipeline.n_clusters_} clusters")
print(f"Features: {len(pipeline.selected_features_)}")
print(f"Silhouette: {pipeline.metrics_['silhouette']:.3f}")

# 2. Analyze feature importance
importance_results = pipeline.analyze_feature_importance(method='all')
print("\nTop 10 features (permutation):")
print(importance_results['permutation'].head(10))

# 3. Try different feature counts
for n in [15, 10, 8, 5]:
    comparison = pipeline.refit_with_top_features(
        n_features=n,
        importance_method='permutation',
        compare_metrics=False,  # Silent mode
        update_pipeline=False
    )

    improvement = comparison['weighted_improvement']
    silhouette = comparison['refitted_metrics']['silhouette']

    print(f"\nTop {n} features: Silhouette={silhouette:.3f} ({improvement:+.1%})")

# 4. Use best performing feature count
best_n = 10  # Based on above results
pipeline.refit_with_top_features(
    n_features=best_n,
    importance_method='permutation',
    update_pipeline=True
)

print(f"\n✓ Pipeline updated with top {best_n} features")
print(f"New silhouette: {pipeline.metrics_['silhouette']:.3f}")

# 5. Continue with refined clustering
pipeline.export_report('refined_clustering.html')
```

### Tips & Best Practices

1. **Start with all features**
   - Fit pipeline on all available features first
   - Then use feature selection to refine

2. **Use permutation method**
   - Most reliable for clustering quality improvement
   - Directly measures impact on silhouette score

3. **Try different feature counts**
   - Test multiple values: e.g., [15, 10, 8, 5]
   - Find sweet spot between simplicity and quality

4. **Don't over-reduce**
   - Too few features (<5) may lose important information
   - Balance between noise reduction and information retention

5. **Verify improvement**
   - Always check `metrics_improved` before updating
   - Not all datasets benefit from feature selection

6. **Test stability**
   - Run `analyze_stability()` after refitting
   - Ensure reduced feature set produces stable clusters

### When NOT to Use Feature Selection

- **Small feature count** (<10 features) - already manageable
- **High correlation already handled** - SmartCorrelationFilter removes redundant features
- **All features important** - domain knowledge says all features matter
- **Metrics don't improve** - feature selection hurts quality

## Automatic Naming

```python
pipeline = ClusterAnalysisPipeline(
    auto_name_clusters=True,
    naming_max_features=2
)

pipeline.fit(df, feature_columns=features,
            category_mapping={
                'demographics': ['age', 'income'],
                'behavior': ['purchases', 'visits']
            })

# Get names
for cluster_id in range(pipeline.n_clusters_):
    name = pipeline.get_cluster_name(cluster_id)
    print(f"Cluster {cluster_id}: {name}")
```

## Summary

```python
# Print comprehensive summary
pipeline.print_cluster_summary()
```
