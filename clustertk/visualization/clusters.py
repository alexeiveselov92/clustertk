"""
Cluster visualization functions.

This module provides functions for visualizing clusters in 2D space
using dimensionality reduction techniques like t-SNE and UMAP.
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


def plot_clusters_2d(
    X: pd.DataFrame,
    labels: np.ndarray,
    method: str = 'tsne',
    title: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 8),
    alpha: float = 0.6,
    s: int = 50,
    show_centers: bool = False,
    centers: Optional[np.ndarray] = None,
    ax: Optional[plt.Axes] = None
) -> plt.Figure:
    """
    Plot clusters in 2D space.

    Uses t-SNE or UMAP to reduce data to 2D for visualization.

    Parameters
    ----------
    X : pd.DataFrame
        Feature data (can be high-dimensional).

    labels : np.ndarray
        Cluster labels for each sample.

    method : str, default='tsne'
        Dimensionality reduction method:
        - 'tsne': t-SNE (requires sklearn)
        - 'umap': UMAP (requires umap-learn)
        - 'pca': PCA (first 2 components)

    title : str, optional
        Plot title. If None, auto-generated based on method.

    figsize : tuple, default=(10, 8)
        Figure size (width, height) in inches.

    alpha : float, default=0.6
        Point transparency (0-1).

    s : int, default=50
        Point size.

    show_centers : bool, default=False
        Whether to show cluster centers (if available).

    centers : np.ndarray, optional
        Cluster centers to plot (must be same dimensionality as X).

    ax : matplotlib.axes.Axes, optional
        Axes to plot on. If None, creates new figure.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object.

    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> from clustertk.visualization import plot_clusters_2d
    >>>
    >>> # After clustering
    >>> fig = plot_clusters_2d(
    ...     X=data_reduced,
    ...     labels=labels,
    ...     method='tsne',
    ...     title='Cluster Visualization'
    ... )
    >>> plt.show()
    """
    _check_viz_available()

    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE

    # Reduce to 2D if needed
    if X.shape[1] > 2:
        if method == 'tsne':
            reducer = TSNE(n_components=2, random_state=42, perplexity=min(30, len(X) - 1))
            X_2d = reducer.fit_transform(X)
        elif method == 'umap':
            try:
                from umap import UMAP
                reducer = UMAP(n_components=2, random_state=42, n_neighbors=min(15, len(X) - 1))
                X_2d = reducer.fit_transform(X)
            except ImportError:
                raise ImportError(
                    "UMAP is not installed. Install it with: pip install umap-learn"
                )
        elif method == 'pca':
            reducer = PCA(n_components=2, random_state=42)
            X_2d = reducer.fit_transform(X)
        else:
            raise ValueError(f"Unknown method '{method}'. Use 'tsne', 'umap', or 'pca'.")
    elif X.shape[1] == 2:
        X_2d = X.values if isinstance(X, pd.DataFrame) else X
    else:
        raise ValueError("X must have at least 2 dimensions")

    # Create figure if ax not provided
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    # Get unique labels
    unique_labels = np.unique(labels)
    n_clusters = len(unique_labels[unique_labels >= 0])

    # Create color palette
    if -1 in unique_labels:
        # DBSCAN with noise
        colors = sns.color_palette("husl", n_clusters)
        color_map = {label: colors[i] if label >= 0 else 'gray'
                     for i, label in enumerate(unique_labels)}
    else:
        colors = sns.color_palette("husl", n_clusters)
        color_map = {label: colors[i] for i, label in enumerate(unique_labels)}

    # Plot each cluster
    for label in unique_labels:
        mask = labels == label
        label_name = f'Noise' if label == -1 else f'Cluster {label}'

        ax.scatter(
            X_2d[mask, 0],
            X_2d[mask, 1],
            c=[color_map[label]],
            label=label_name,
            alpha=alpha,
            s=s,
            edgecolors='white',
            linewidth=0.5
        )

    # Plot centers if provided
    if show_centers and centers is not None:
        if centers.shape[1] > 2:
            # Reduce centers to 2D using same method
            if method == 'pca':
                centers_2d = reducer.transform(centers)
            else:
                # For t-SNE/UMAP, fit on combined data
                combined = np.vstack([X, centers])
                if method == 'tsne':
                    combined_2d = TSNE(n_components=2, random_state=42).fit_transform(combined)
                else:
                    from umap import UMAP
                    combined_2d = UMAP(n_components=2, random_state=42).fit_transform(combined)
                centers_2d = combined_2d[-len(centers):]
        else:
            centers_2d = centers

        ax.scatter(
            centers_2d[:, 0],
            centers_2d[:, 1],
            c='black',
            marker='X',
            s=200,
            edgecolors='white',
            linewidth=2,
            label='Centers',
            zorder=10
        )

    # Set labels and title
    method_name = method.upper() if method != 'umap' else 'UMAP'
    ax.set_xlabel(f'{method_name} Component 1', fontsize=12)
    ax.set_ylabel(f'{method_name} Component 2', fontsize=12)

    if title:
        ax.set_title(title, fontsize=14, fontweight='bold')
    else:
        ax.set_title(f'Cluster Visualization ({method_name})', fontsize=14, fontweight='bold')

    # Legend
    ax.legend(loc='best', frameon=True, fancybox=True, shadow=True)
    ax.grid(True, alpha=0.3, linestyle='--')

    fig.tight_layout()

    return fig


def plot_cluster_sizes(
    labels: np.ndarray,
    title: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 6),
    ax: Optional[plt.Axes] = None
) -> plt.Figure:
    """
    Plot cluster size distribution as a bar chart.

    Parameters
    ----------
    labels : np.ndarray
        Cluster labels.

    title : str, optional
        Plot title.

    figsize : tuple, default=(10, 6)
        Figure size (width, height).

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

    # Count samples per cluster
    unique, counts = np.unique(labels, return_counts=True)

    # Separate noise if present
    noise_count = 0
    if -1 in unique:
        noise_idx = np.where(unique == -1)[0][0]
        noise_count = counts[noise_idx]
        unique = np.delete(unique, noise_idx)
        counts = np.delete(counts, noise_idx)

    # Create bars
    colors = sns.color_palette("husl", len(unique))
    bars = ax.bar(range(len(unique)), counts, color=colors, edgecolor='white', linewidth=1.5)

    # Add noise bar if present
    if noise_count > 0:
        noise_bar = ax.bar(len(unique), noise_count, color='gray', edgecolor='white', linewidth=1.5)
        labels_list = [f'Cluster {i}' for i in unique] + ['Noise']
    else:
        labels_list = [f'Cluster {i}' for i in unique]

    # Customize plot
    ax.set_xlabel('Cluster', fontsize=12)
    ax.set_ylabel('Number of Samples', fontsize=12)
    ax.set_title(title or 'Cluster Size Distribution', fontsize=14, fontweight='bold')
    ax.set_xticks(range(len(labels_list)))
    ax.set_xticklabels(labels_list)
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    # Add value labels on bars
    for i, (bar, count) in enumerate(zip(bars, counts)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(count)}\n({count/len(labels)*100:.1f}%)',
                ha='center', va='bottom', fontsize=10)

    if noise_count > 0:
        ax.text(len(unique), noise_count,
                f'{int(noise_count)}\n({noise_count/len(labels)*100:.1f}%)',
                ha='center', va='bottom', fontsize=10)

    fig.tight_layout()

    return fig
