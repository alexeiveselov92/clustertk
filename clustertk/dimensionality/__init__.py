"""
Dimensionality reduction module.

This module provides classes for:
- PCA with automatic component selection
- Manifold methods (t-SNE, UMAP) for visualization
"""

from clustertk.dimensionality.pca import PCAReducer, quick_pca
from clustertk.dimensionality.manifold import ManifoldReducer, quick_tsne, quick_umap

__all__ = [
    'PCAReducer',
    'ManifoldReducer',
    'quick_pca',
    'quick_tsne',
    'quick_umap',
]
