# Visualization

Visualize clustering results (requires `pip install clustertk[viz]`).

## 2D Cluster Plot

```python
from IPython.display import display

# t-SNE (default)
display(pipeline.plot_clusters_2d(method='tsne'))

# UMAP
display(pipeline.plot_clusters_2d(method='umap'))
```

## Cluster Profiles

```python
# Heatmap
display(pipeline.plot_cluster_heatmap())

# Radar chart
display(pipeline.plot_cluster_radar())
```

## Other Plots

```python
# Cluster sizes
display(pipeline.plot_cluster_sizes())

# Feature importance for a cluster
display(pipeline.plot_feature_importance(cluster_id=0, n_features=10))

# PCA variance
display(pipeline.plot_pca_variance())

# Correlation matrix
display(pipeline.plot_correlation_matrix())
```

## Saving Plots

```python
fig = pipeline.plot_clusters_2d()
fig.savefig('clusters.png', dpi=300, bbox_inches='tight')
```

## Multiple Plots in Jupyter

Use `display()` to show multiple plots in one cell:

```python
from IPython.display import display

display(pipeline.plot_clusters_2d())
display(pipeline.plot_cluster_heatmap())
display(pipeline.plot_cluster_sizes())
```
