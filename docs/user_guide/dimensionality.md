# Dimensionality Reduction

Reduce feature space while preserving information.

## PCA

```python
pipeline = ClusterAnalysisPipeline(
    pca_variance=0.9,          # Keep 90% of variance
    pca_min_components=2       # Minimum for visualization
)
```

Automatically selects number of components based on variance threshold.

## Visualization Methods

t-SNE and UMAP for 2D visualization (not used for clustering):

```python
# t-SNE
pipeline.plot_clusters_2d(method='tsne')

# UMAP
pipeline.plot_clusters_2d(method='umap')

# PCA
pipeline.plot_clusters_2d(method='pca')
```

## Access Results

```python
# PCA components
components = pipeline.data_reduced_

# Explained variance
variance = pipeline._pca_reducer.explained_variance_

# Visualize
pipeline.plot_pca_variance()
```
