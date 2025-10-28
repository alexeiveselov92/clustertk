# Feature Selection

Feature selection removes irrelevant or redundant features to improve clustering quality and reduce computational cost.

## Correlation Filtering

Remove highly correlated features:

```python
from clustertk import ClusterAnalysisPipeline

pipeline = ClusterAnalysisPipeline(
    correlation_threshold=0.85  # Remove features with |correlation| > 0.85
)
```

When two features have high correlation, one is kept and the other is removed.

## Variance Filtering

Remove low-variance features:

```python
pipeline = ClusterAnalysisPipeline(
    variance_threshold=0.01  # Remove features with variance < 0.01
)
```

Low-variance features provide little information for clustering.

## Complete Example

```python
pipeline = ClusterAnalysisPipeline(
    correlation_threshold=0.85,
    variance_threshold=0.01,
    verbose=True  # See which features are removed
)

pipeline.fit(df, feature_columns=all_features)

# Check selected features
print(f"Original features: {len(all_features)}")
print(f"Selected features: {len(pipeline.selected_features_)}")
print(pipeline.selected_features_)
```

## Best Practices

1. Use correlation filtering to reduce multicollinearity
2. Set variance_threshold > 0 to remove constants
3. Check selected_features_ after fitting
