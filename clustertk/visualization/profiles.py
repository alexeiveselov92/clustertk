"""
Cluster profile visualization functions.

This module provides functions for visualizing cluster profiles
using heatmaps and radar charts.
"""

from typing import Optional, Tuple, List
import pandas as pd
import numpy as np

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    VISUALIZATION_AVAILABLE = True
except ImportError:
    VISUALIZATION_AVAILABLE = False


def _check_viz_available():
    """Check if visualization dependencies are installed."""
    if not VISUALIZATION_AVAILABLE:
        raise ImportError(
            "Visualization dependencies are not installed. "
            "Install them with: pip install clustertk[viz]"
        )


def plot_cluster_heatmap(
    profiles: pd.DataFrame,
    title: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 8),
    cmap: str = 'RdYlGn',
    normalize: bool = True,
    ax: Optional[plt.Axes] = None
) -> plt.Figure:
    """
    Plot cluster profiles as a heatmap.

    Parameters
    ----------
    profiles : pd.DataFrame
        Cluster profiles with clusters as rows and features as columns.

    title : str, optional
        Plot title.

    figsize : tuple, default=(12, 8)
        Figure size (width, height).

    cmap : str, default='RdYlGn'
        Colormap name.

    normalize : bool, default=True
        Whether to normalize each feature to 0-1 scale for visualization.

    ax : matplotlib.axes.Axes, optional
        Axes to plot on.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object.

    Examples
    --------
    >>> from clustertk.visualization import plot_cluster_heatmap
    >>>
    >>> # After creating profiles
    >>> fig = plot_cluster_heatmap(
    ...     profiles=pipeline.cluster_profiles_,
    ...     title='Cluster Profiles Heatmap'
    ... )
    >>> plt.show()
    """
    _check_viz_available()

    # Create figure if ax not provided
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    # Normalize if requested
    if normalize:
        data = profiles.copy()
        for col in data.columns:
            col_min = data[col].min()
            col_max = data[col].max()
            if col_max != col_min:
                data[col] = (data[col] - col_min) / (col_max - col_min)
            else:
                data[col] = 0.5
    else:
        data = profiles

    # Create heatmap
    sns.heatmap(
        data,
        annot=True,
        fmt='.2f',
        cmap=cmap,
        cbar_kws={'label': 'Normalized Value' if normalize else 'Value'},
        linewidths=0.5,
        linecolor='gray',
        ax=ax,
        vmin=0 if normalize else None,
        vmax=1 if normalize else None
    )

    # Customize
    ax.set_title(title or 'Cluster Profiles Heatmap', fontsize=14, fontweight='bold', pad=20)
    ax.set_xlabel('Features', fontsize=12)
    ax.set_ylabel('Clusters', fontsize=12)

    # Rotate labels
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)

    fig.tight_layout()

    return fig


def plot_cluster_radar(
    profiles: pd.DataFrame,
    cluster_ids: Optional[List[int]] = None,
    title: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 10),
    normalize: bool = True
) -> plt.Figure:
    """
    Plot cluster profiles as radar charts.

    Parameters
    ----------
    profiles : pd.DataFrame
        Cluster profiles with clusters as rows and features as columns.

    cluster_ids : list of int, optional
        Specific cluster IDs to plot. If None, plots all clusters.

    title : str, optional
        Plot title.

    figsize : tuple, default=(10, 10)
        Figure size (width, height).

    normalize : bool, default=True
        Whether to normalize each feature to 0-1 scale.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object.

    Examples
    --------
    >>> from clustertk.visualization import plot_cluster_radar
    >>>
    >>> # Plot specific clusters
    >>> fig = plot_cluster_radar(
    ...     profiles=pipeline.cluster_profiles_,
    ...     cluster_ids=[0, 1, 2],
    ...     title='Cluster Comparison'
    ... )
    >>> plt.show()
    """
    _check_viz_available()

    # Select clusters to plot
    if cluster_ids is not None:
        data = profiles.loc[cluster_ids]
    else:
        data = profiles

    # Normalize if requested
    if normalize:
        normalized = data.copy()
        for col in normalized.columns:
            col_min = normalized[col].min()
            col_max = normalized[col].max()
            if col_max != col_min:
                normalized[col] = (normalized[col] - col_min) / (col_max - col_min)
            else:
                normalized[col] = 0.5
        data = normalized

    # Setup radar chart
    categories = list(data.columns)
    N = len(categories)

    # Compute angle for each axis
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]  # Complete the circle

    # Create figure
    fig, ax = plt.subplots(figsize=figsize, subplot_kw=dict(projection='polar'))

    # Plot each cluster
    colors = sns.color_palette("husl", len(data))

    for idx, (cluster_id, row) in enumerate(data.iterrows()):
        values = row.tolist()
        values += values[:1]  # Complete the circle

        ax.plot(angles, values, 'o-', linewidth=2, label=f'Cluster {cluster_id}',
                color=colors[idx])
        ax.fill(angles, values, alpha=0.15, color=colors[idx])

    # Fix axis to go from 0 to 1 if normalized
    if normalize:
        ax.set_ylim(0, 1)
    else:
        ax.set_ylim(data.values.min(), data.values.max())

    # Set category labels
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, size=10)

    # Add grid
    ax.grid(True, linestyle='--', alpha=0.7)

    # Add title and legend
    ax.set_title(title or 'Cluster Profiles Radar Chart', size=14,
                 fontweight='bold', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))

    fig.tight_layout()

    return fig


def plot_feature_importance(
    top_features: dict,
    cluster_id: int,
    n_features: int = 10,
    title: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 6),
    ax: Optional[plt.Axes] = None
) -> plt.Figure:
    """
    Plot top distinguishing features for a cluster.

    Parameters
    ----------
    top_features : dict
        Dictionary from ClusterProfiler.get_top_features().

    cluster_id : int
        Cluster ID to visualize.

    n_features : int, default=10
        Number of top features to show.

    title : str, optional
        Plot title.

    figsize : tuple, default=(10, 6)
        Figure size.

    ax : matplotlib.axes.Axes, optional
        Axes to plot on.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object.
    """
    _check_viz_available()

    if cluster_id not in top_features:
        raise ValueError(f"Cluster {cluster_id} not found in top_features")

    # Create figure if ax not provided
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    # Get high and low features
    high_features = top_features[cluster_id]['high'][:n_features//2]
    low_features = top_features[cluster_id]['low'][:n_features//2]

    # Combine and sort by absolute value
    all_features = high_features + low_features
    features, values = zip(*all_features)

    # Sort by value
    sorted_indices = np.argsort(values)
    features = [features[i] for i in sorted_indices]
    values = [values[i] for i in sorted_indices]

    # Create colors (red for negative, green for positive)
    colors = ['#d32f2f' if v < 0 else '#388e3c' for v in values]

    # Create horizontal bar chart
    y_pos = np.arange(len(features))
    ax.barh(y_pos, values, color=colors, edgecolor='black', linewidth=0.5)

    # Customize
    ax.set_yticks(y_pos)
    ax.set_yticklabels(features)
    ax.set_xlabel('Deviation from Mean', fontsize=12)
    ax.set_title(title or f'Top Features for Cluster {cluster_id}',
                 fontsize=14, fontweight='bold')
    ax.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
    ax.grid(axis='x', alpha=0.3, linestyle='--')

    # Add value labels
    for i, v in enumerate(values):
        ax.text(v, i, f' {v:.3f}', va='center',
                ha='left' if v > 0 else 'right', fontsize=9)

    fig.tight_layout()

    return fig
