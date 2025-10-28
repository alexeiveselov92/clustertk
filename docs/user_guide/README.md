# User Guide

This guide provides detailed documentation for each component of ClusterTK.

## Contents

1. **[Preprocessing](preprocessing.md)** - Data cleaning and transformation
   - Missing value handling
   - Outlier detection and treatment
   - Scaling and normalization
   - Skewness transformation

2. **[Feature Selection](feature_selection.md)** - Selecting relevant features
   - Correlation filtering
   - Variance filtering
   - Custom selection strategies

3. **[Dimensionality Reduction](dimensionality.md)** - Reducing feature space
   - PCA with automatic component selection
   - t-SNE and UMAP for visualization
   - Interpreting components

4. **[Clustering](clustering.md)** - Applying clustering algorithms
   - K-Means clustering
   - Gaussian Mixture Models (GMM)
   - Hierarchical clustering
   - DBSCAN (density-based)
   - Custom clustering algorithms

5. **[Evaluation](evaluation.md)** - Assessing cluster quality
   - Silhouette score
   - Calinski-Harabasz index
   - Davies-Bouldin index
   - Optimal cluster number selection

6. **[Interpretation](interpretation.md)** - Understanding clusters
   - Cluster profiling
   - Top distinguishing features
   - Automatic cluster naming
   - Category-based analysis

7. **[Visualization](visualization.md)** - Visualizing results
   - 2D cluster plots
   - Cluster profiles (heatmaps, radar charts)
   - Feature importance plots
   - PCA variance plots
   - Correlation analysis

8. **[Export](export.md)** - Saving and sharing results
   - CSV export
   - JSON export
   - HTML reports
   - Pipeline serialization

## Quick Links

- [Installation](../installation.md)
- [Quick Start](../quickstart.md)
- [API Reference](../api_reference.md)
- [Examples](../examples.md)
- [FAQ](../faq.md)

## Pipeline Workflow

ClusterTK follows a standard workflow:

```
Raw Data
   ↓
Preprocessing (handle missing, outliers, scaling)
   ↓
Feature Selection (correlation, variance)
   ↓
Dimensionality Reduction (PCA)
   ↓
Optimal K Finding (if n_clusters=None)
   ↓
Clustering (K-Means/GMM/Hierarchical/DBSCAN)
   ↓
Evaluation (compute metrics)
   ↓
Interpretation (create profiles, naming)
   ↓
Visualization & Export
```

Each step can be configured through pipeline parameters or run independently for maximum flexibility.
