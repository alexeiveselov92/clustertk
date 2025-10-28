"""
Visualization module for clustering results.

This module provides optional visualization functions for:
- Cluster scatter plots (2D)
- Cluster profiles (heatmaps, radar charts)
- Dimensionality reduction (PCA variance, loadings)
- Correlation matrices and networks
- Feature distributions

Install visualization dependencies with: pip install clustertk[viz]
"""

import warnings


def check_viz_available() -> bool:
    """
    Check if visualization dependencies are installed.

    Returns
    -------
    bool
        True if matplotlib and seaborn are available, False otherwise.
    """
    try:
        import matplotlib  # noqa: F401
        import seaborn  # noqa: F401
        return True
    except ImportError:
        return False


# Conditional imports - only if viz dependencies are available
if check_viz_available():
    # Cluster visualizations
    from clustertk.visualization.clusters import (
        plot_clusters_2d,
        plot_cluster_sizes,
        plot_algorithm_comparison
    )

    # Profile visualizations
    from clustertk.visualization.profiles import (
        plot_cluster_heatmap,
        plot_cluster_radar,
        plot_feature_importance
    )

    # Dimensionality visualizations
    from clustertk.visualization.dimensionality import (
        plot_pca_variance,
        plot_pca_loadings,
        plot_elbow
    )

    # Correlation visualizations
    from clustertk.visualization.correlation import (
        plot_correlation_matrix,
        plot_correlation_network,
        plot_feature_distributions
    )

    __all__ = [
        'check_viz_available',
        # Clusters
        'plot_clusters_2d',
        'plot_cluster_sizes',
        'plot_algorithm_comparison',
        # Profiles
        'plot_cluster_heatmap',
        'plot_cluster_radar',
        'plot_feature_importance',
        # Dimensionality
        'plot_pca_variance',
        'plot_pca_loadings',
        'plot_elbow',
        # Correlation
        'plot_correlation_matrix',
        'plot_correlation_network',
        'plot_feature_distributions',
    ]
else:
    warnings.warn(
        "Visualization dependencies (matplotlib, seaborn) not found. "
        "Visualization functions will not be available. "
        "Install with: pip install clustertk[viz]",
        ImportWarning
    )

    __all__ = ['check_viz_available']
