"""
Dimensionality reduction visualization functions.

This module provides functions for visualizing PCA results,
explained variance, and component loadings.
"""

from typing import Optional, Tuple
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


def plot_pca_variance(
    explained_variance_ratio: np.ndarray,
    cumulative_variance: Optional[np.ndarray] = None,
    threshold: Optional[float] = None,
    title: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 6),
    ax: Optional[plt.Axes] = None
) -> plt.Figure:
    """
    Plot PCA explained variance (scree plot).

    Parameters
    ----------
    explained_variance_ratio : np.ndarray
        Explained variance ratio for each component.

    cumulative_variance : np.ndarray, optional
        Cumulative explained variance. If None, computed from explained_variance_ratio.

    threshold : float, optional
        Variance threshold line to show (e.g., 0.9 for 90%).

    title : str, optional
        Plot title.

    figsize : tuple, default=(12, 6)
        Figure size (width, height).

    ax : matplotlib.axes.Axes, optional
        Axes to plot on.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object.

    Examples
    --------
    >>> from clustertk.visualization import plot_pca_variance
    >>>
    >>> # After PCA
    >>> fig = plot_pca_variance(
    ...     explained_variance_ratio=pca_reducer.explained_variance_,
    ...     threshold=0.9
    ... )
    >>> plt.show()
    """
    _check_viz_available()

    # Create figure if ax not provided
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    # Compute cumulative variance if not provided
    if cumulative_variance is None:
        cumulative_variance = np.cumsum(explained_variance_ratio)

    n_components = len(explained_variance_ratio)
    x = np.arange(1, n_components + 1)

    # Plot individual variance
    ax.bar(x, explained_variance_ratio, alpha=0.6, color='steelblue',
           label='Individual', edgecolor='black', linewidth=0.5)

    # Plot cumulative variance
    ax2 = ax.twinx()
    ax2.plot(x, cumulative_variance, color='red', marker='o', linestyle='-',
             linewidth=2, markersize=5, label='Cumulative')

    # Add threshold line if provided
    if threshold is not None:
        # Find component where threshold is reached
        threshold_idx = np.where(cumulative_variance >= threshold)[0]
        if len(threshold_idx) > 0:
            threshold_comp = threshold_idx[0] + 1
            ax2.axhline(y=threshold, color='green', linestyle='--', linewidth=2,
                       label=f'Threshold ({threshold:.0%})')
            ax2.axvline(x=threshold_comp, color='green', linestyle='--',
                       linewidth=1, alpha=0.5)
            ax.text(threshold_comp, 0, f'PC{threshold_comp}',
                   ha='center', va='bottom', fontsize=10,
                   bbox=dict(boxstyle='round', facecolor='green', alpha=0.3))

    # Customize primary axis
    ax.set_xlabel('Principal Component', fontsize=12)
    ax.set_ylabel('Explained Variance Ratio', fontsize=12)
    ax.set_title(title or 'PCA Explained Variance', fontsize=14, fontweight='bold')
    ax.set_xticks(x[::max(1, n_components//20)])  # Show max 20 ticks
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    # Customize secondary axis
    ax2.set_ylabel('Cumulative Explained Variance', fontsize=12)
    ax2.set_ylim(0, 1.05)

    # Legends
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc='center right')

    fig.tight_layout()

    return fig


def plot_pca_loadings(
    loadings: pd.DataFrame,
    components: Optional[list] = None,
    n_features: int = 10,
    title: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 8)
) -> plt.Figure:
    """
    Plot PCA component loadings.

    Parameters
    ----------
    loadings : pd.DataFrame
        Loadings matrix with components as columns and features as rows.

    components : list, optional
        Which components to plot (e.g., [0, 1, 2] for first 3 PCs).
        If None, plots all components.

    n_features : int, default=10
        Number of top features to show per component.

    title : str, optional
        Plot title.

    figsize : tuple, default=(12, 8)
        Figure size.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object.
    """
    _check_viz_available()

    # Select components to plot
    if components is not None:
        loadings = loadings.iloc[:, components]

    n_components = len(loadings.columns)

    # Create subplots
    n_cols = min(3, n_components)
    n_rows = (n_components + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    if n_components == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    # Plot each component
    for idx, col in enumerate(loadings.columns):
        ax = axes[idx]

        # Get top features by absolute loading
        component_loadings = loadings[col].abs().sort_values(ascending=False)
        top_features = component_loadings.head(n_features).index
        values = loadings.loc[top_features, col].sort_values()

        # Plot
        colors = ['#d32f2f' if v < 0 else '#388e3c' for v in values]
        y_pos = np.arange(len(values))

        ax.barh(y_pos, values, color=colors, edgecolor='black', linewidth=0.5)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(values.index, fontsize=9)
        ax.set_xlabel('Loading', fontsize=10)
        ax.set_title(f'{col}', fontsize=11, fontweight='bold')
        ax.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
        ax.grid(axis='x', alpha=0.3, linestyle='--')

    # Hide unused subplots
    for idx in range(n_components, len(axes)):
        axes[idx].axis('off')

    fig.suptitle(title or 'PCA Component Loadings', fontsize=14, fontweight='bold')
    fig.tight_layout()

    return fig


def plot_elbow(
    k_values: np.ndarray,
    metric_values: np.ndarray,
    metric_name: str = 'Inertia',
    optimal_k: Optional[int] = None,
    title: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 6),
    ax: Optional[plt.Axes] = None
) -> plt.Figure:
    """
    Plot elbow curve for optimal k selection.

    Parameters
    ----------
    k_values : np.ndarray
        Number of clusters tested.

    metric_values : np.ndarray
        Metric values for each k.

    metric_name : str, default='Inertia'
        Name of the metric being plotted.

    optimal_k : int, optional
        Optimal k value to highlight.

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

    # Create figure if ax not provided
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    # Plot the curve
    ax.plot(k_values, metric_values, marker='o', linestyle='-',
            linewidth=2, markersize=8, color='steelblue')

    # Highlight optimal k if provided
    if optimal_k is not None:
        opt_idx = np.where(k_values == optimal_k)[0]
        if len(opt_idx) > 0:
            ax.plot(optimal_k, metric_values[opt_idx[0]], marker='*',
                   markersize=20, color='red', label=f'Optimal k={optimal_k}')
            ax.axvline(x=optimal_k, color='red', linestyle='--', alpha=0.5)

    # Customize
    ax.set_xlabel('Number of Clusters (k)', fontsize=12)
    ax.set_ylabel(metric_name, fontsize=12)
    ax.set_title(title or f'Elbow Method ({metric_name})', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_xticks(k_values)

    if optimal_k is not None:
        ax.legend()

    fig.tight_layout()

    return fig
