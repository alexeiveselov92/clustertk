"""
Visualization module for plotting clustering results.

This module is OPTIONAL and requires viz dependencies:
    pip install clustertk[viz]

Provides plotting functions for:
- Correlation matrices
- Distributions and boxplots
- PCA variance and biplots
- Cluster scatter plots (2D)
- Cluster profiles (heatmaps, radar charts)
"""

import warnings


def check_viz_available() -> bool:
    """
    Check if visualization dependencies are installed.

    Returns:
        bool: True if matplotlib and seaborn are available, False otherwise
    """
    try:
        import matplotlib  # noqa: F401
        import seaborn  # noqa: F401
        return True
    except ImportError:
        return False


def _raise_viz_error() -> None:
    """Raise ImportError with instructions for installing viz dependencies."""
    raise ImportError(
        "Visualization dependencies not installed. "
        "Install with: pip install clustertk[viz]"
    )


# Conditional imports - only if viz dependencies are available
if check_viz_available():
    # These imports will be added as we implement the visualization functions
    # from clustertk.visualization.correlation import plot_correlation_matrix
    # from clustertk.visualization.distributions import plot_distributions
    # from clustertk.visualization.dimensionality import plot_pca_variance, plot_pca_biplot
    # from clustertk.visualization.clusters import plot_clusters_2d
    # from clustertk.visualization.profiles import plot_cluster_heatmap, plot_cluster_radar
    pass
else:
    warnings.warn(
        "Visualization dependencies (matplotlib, seaborn) not found. "
        "Visualization functions will not be available. "
        "Install with: pip install clustertk[viz]",
        ImportWarning
    )

__all__ = [
    'check_viz_available',
    # Plotting functions will be added here as we implement them
]
