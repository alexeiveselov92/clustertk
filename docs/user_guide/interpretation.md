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
