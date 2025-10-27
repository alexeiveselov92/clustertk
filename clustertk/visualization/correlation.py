"""
Correlation visualization functions.

This module provides functions for visualizing correlation matrices
and feature relationships.
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


def plot_correlation_matrix(
    data: pd.DataFrame,
    method: str = 'pearson',
    title: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 10),
    cmap: str = 'coolwarm',
    annot: bool = False,
    threshold: Optional[float] = None,
    ax: Optional[plt.Axes] = None
) -> plt.Figure:
    """
    Plot correlation matrix as a heatmap.

    Parameters
    ----------
    data : pd.DataFrame
        Input data.

    method : str, default='pearson'
        Correlation method: 'pearson', 'spearman', or 'kendall'.

    title : str, optional
        Plot title.

    figsize : tuple, default=(12, 10)
        Figure size (width, height).

    cmap : str, default='coolwarm'
        Colormap name.

    annot : bool, default=False
        Whether to annotate cells with correlation values.

    threshold : float, optional
        If provided, only shows correlations with |r| > threshold.

    ax : matplotlib.axes.Axes, optional
        Axes to plot on.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object.

    Examples
    --------
    >>> from clustertk.visualization import plot_correlation_matrix
    >>>
    >>> fig = plot_correlation_matrix(
    ...     data=df,
    ...     method='pearson',
    ...     threshold=0.5,
    ...     title='Feature Correlations'
    ... )
    >>> plt.show()
    """
    _check_viz_available()

    # Compute correlation matrix
    corr_matrix = data.corr(method=method)

    # Apply threshold if provided
    if threshold is not None:
        mask = np.abs(corr_matrix) < threshold
        corr_matrix = corr_matrix.mask(mask)

    # Create figure if ax not provided
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    # Create heatmap
    sns.heatmap(
        corr_matrix,
        annot=annot,
        fmt='.2f' if annot else '',
        cmap=cmap,
        center=0,
        vmin=-1,
        vmax=1,
        square=True,
        linewidths=0.5,
        linecolor='gray',
        cbar_kws={'label': 'Correlation Coefficient', 'shrink': 0.8},
        ax=ax
    )

    # Customize
    method_name = method.capitalize()
    threshold_str = f' (|r| > {threshold})' if threshold else ''
    ax.set_title(title or f'{method_name} Correlation Matrix{threshold_str}',
                 fontsize=14, fontweight='bold', pad=20)

    # Rotate x labels - use tick_params to avoid triggering extra rendering
    ax.tick_params(axis='x', rotation=45)
    # Set horizontal alignment after tick_params
    for label in ax.get_xticklabels():
        label.set_ha('right')

    fig.tight_layout()


    return fig


def plot_correlation_network(
    data: pd.DataFrame,
    threshold: float = 0.5,
    method: str = 'pearson',
    title: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 12),
    node_size: int = 1000,
    font_size: int = 10
) -> plt.Figure:
    """
    Plot correlation network graph.

    Shows features as nodes and correlations as edges.
    Only edges with |correlation| > threshold are shown.

    Parameters
    ----------
    data : pd.DataFrame
        Input data.

    threshold : float, default=0.5
        Minimum absolute correlation to show.

    method : str, default='pearson'
        Correlation method.

    title : str, optional
        Plot title.

    figsize : tuple, default=(12, 12)
        Figure size.

    node_size : int, default=1000
        Size of nodes.

    font_size : int, default=10
        Font size for labels.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object.

    Notes
    -----
    Requires networkx package: pip install networkx
    """
    _check_viz_available()

    try:
        import networkx as nx
    except ImportError:
        raise ImportError(
            "networkx is required for correlation network plots. "
            "Install it with: pip install networkx"
        )

    # Compute correlation matrix
    corr_matrix = data.corr(method=method)

    # Create network graph
    G = nx.Graph()

    # Add nodes
    for feature in corr_matrix.columns:
        G.add_node(feature)

    # Add edges for correlations above threshold
    for i, feature1 in enumerate(corr_matrix.columns):
        for j, feature2 in enumerate(corr_matrix.columns):
            if i < j:  # Avoid duplicates
                corr = corr_matrix.loc[feature1, feature2]
                if abs(corr) >= threshold:
                    G.add_edge(feature1, feature2, weight=abs(corr), corr=corr)

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Layout
    pos = nx.spring_layout(G, seed=42, k=0.5, iterations=50)

    # Draw nodes
    nx.draw_networkx_nodes(
        G, pos,
        node_color='lightblue',
        node_size=node_size,
        edgecolors='black',
        linewidths=2,
        ax=ax
    )

    # Draw edges with colors based on correlation
    edges = G.edges()
    weights = [G[u][v]['weight'] for u, v in edges]
    corrs = [G[u][v]['corr'] for u, v in edges]

    # Color edges: red for negative, blue for positive
    edge_colors = ['red' if c < 0 else 'blue' for c in corrs]

    nx.draw_networkx_edges(
        G, pos,
        width=[w * 3 for w in weights],  # Scale edge width by correlation
        alpha=0.6,
        edge_color=edge_colors,
        ax=ax
    )

    # Draw labels
    nx.draw_networkx_labels(
        G, pos,
        font_size=font_size,
        font_weight='bold',
        ax=ax
    )

    # Customize
    ax.set_title(title or f'Correlation Network (|r| > {threshold})',
                 fontsize=14, fontweight='bold')
    ax.axis('off')

    # Add legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='blue', lw=4, label='Positive correlation'),
        Line2D([0], [0], color='red', lw=4, label='Negative correlation')
    ]
    ax.legend(handles=legend_elements, loc='upper right')

    fig.tight_layout()


    return fig


def plot_feature_distributions(
    data: pd.DataFrame,
    features: Optional[List[str]] = None,
    n_cols: int = 3,
    figsize: Optional[Tuple[int, int]] = None,
    bins: int = 30
) -> plt.Figure:
    """
    Plot distribution histograms for features.

    Parameters
    ----------
    data : pd.DataFrame
        Input data.

    features : list of str, optional
        Features to plot. If None, plots all numeric columns.

    n_cols : int, default=3
        Number of columns in subplot grid.

    figsize : tuple, optional
        Figure size. If None, auto-calculated.

    bins : int, default=30
        Number of histogram bins.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object.
    """
    _check_viz_available()

    # Select features
    if features is None:
        features = data.select_dtypes(include=[np.number]).columns.tolist()

    n_features = len(features)
    n_rows = (n_features + n_cols - 1) // n_cols

    # Auto-calculate figsize if not provided
    if figsize is None:
        figsize = (n_cols * 5, n_rows * 4)

    # Create subplots
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    if n_features == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    # Plot each feature
    for idx, feature in enumerate(features):
        ax = axes[idx]

        # Histogram
        ax.hist(data[feature].dropna(), bins=bins, color='steelblue',
                edgecolor='black', alpha=0.7)

        # Add mean and median lines
        mean_val = data[feature].mean()
        median_val = data[feature].median()

        ax.axvline(mean_val, color='red', linestyle='--', linewidth=2,
                  label=f'Mean: {mean_val:.2f}')
        ax.axvline(median_val, color='green', linestyle='--', linewidth=2,
                  label=f'Median: {median_val:.2f}')

        # Customize
        ax.set_xlabel(feature, fontsize=10)
        ax.set_ylabel('Frequency', fontsize=10)
        ax.set_title(f'{feature} Distribution', fontsize=11, fontweight='bold')
        ax.legend(fontsize=8)
        ax.grid(axis='y', alpha=0.3, linestyle='--')

    # Hide unused subplots
    for idx in range(n_features, len(axes)):
        axes[idx].axis('off')

    fig.suptitle('Feature Distributions', fontsize=14, fontweight='bold')
    fig.tight_layout()


    return fig
