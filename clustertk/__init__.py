"""
ClusterTK - A comprehensive toolkit for cluster analysis.

ClusterTK provides a complete pipeline for cluster analysis including:
- Data preprocessing (missing values, outliers, scaling)
- Feature selection (correlation filtering, variance filtering)
- Dimensionality reduction (PCA, t-SNE, UMAP)
- Clustering algorithms (K-Means, GMM, Hierarchical, DBSCAN)
- Evaluation metrics
- Cluster interpretation and profiling
- Visualization (optional)

Example:
    >>> from clustertk import ClusterAnalysisPipeline
    >>> pipeline = ClusterAnalysisPipeline(
    ...     handle_missing='median',
    ...     correlation_threshold=0.85,
    ...     pca_variance=0.9,
    ...     clustering_algorithm='kmeans'
    ... )
    >>> pipeline.fit(df, feature_columns=['col1', 'col2', 'col3'])
    >>> labels = pipeline.labels_
    >>> profiles = pipeline.cluster_profiles_
"""

__version__ = '0.7.0'

from clustertk.pipeline import ClusterAnalysisPipeline

__all__ = [
    'ClusterAnalysisPipeline',
    '__version__',
]
