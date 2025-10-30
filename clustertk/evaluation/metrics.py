"""
Clustering evaluation metrics.

This module provides functions for evaluating clustering quality using
various metrics.
"""

from typing import Dict, Any
import pandas as pd
import numpy as np
from sklearn.metrics import (
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score
)


def cluster_balance_score(labels: np.ndarray) -> float:
    """
    Measure how balanced cluster sizes are.

    Uses normalized Shannon entropy to measure cluster size distribution.
    A perfectly balanced clustering (all clusters equal size) gets score 1.0.
    A completely unbalanced clustering (99% in one cluster) gets score close to 0.

    This metric is useful to detect when clustering produces one dominant cluster
    with all other clusters being very small, which is often undesirable in practice.

    Parameters
    ----------
    labels : np.ndarray
        Cluster labels for each sample.

    Returns
    -------
    balance : float
        Balance score in range [0, 1]:
        - 1.0 = perfectly balanced (all clusters equal size)
        - >0.8 = well balanced
        - 0.5-0.8 = moderately balanced
        - <0.5 = imbalanced (some clusters much larger than others)
        - ~0.0 = highly imbalanced (e.g., 99% in one cluster)

    Examples
    --------
    >>> import numpy as np
    >>> from clustertk.evaluation import cluster_balance_score
    >>>
    >>> # Perfectly balanced (3 clusters, 100 samples each)
    >>> labels = np.repeat([0, 1, 2], 100)
    >>> print(cluster_balance_score(labels))  # ~1.0
    >>>
    >>> # Imbalanced (cluster 0 has 280 samples, others 10 each)
    >>> labels = np.array([0]*280 + [1]*10 + [2]*10)
    >>> print(cluster_balance_score(labels))  # ~0.46
    >>>
    >>> # Highly imbalanced (99% in one cluster)
    >>> labels = np.array([0]*990 + [1]*5 + [2]*5)
    >>> print(cluster_balance_score(labels))  # ~0.15
    """
    # Filter out noise points if using DBSCAN/HDBSCAN
    mask = labels != -1
    labels_filtered = labels[mask]

    if len(labels_filtered) == 0:
        return 0.0

    # Get cluster sizes
    unique_labels, counts = np.unique(labels_filtered, return_counts=True)

    if len(unique_labels) < 2:
        # Only one cluster - perfectly balanced by definition
        return 1.0

    # Calculate cluster proportions
    proportions = counts / counts.sum()

    # Shannon entropy (measures disorder/uniformity)
    # Higher entropy = more uniform distribution = better balance
    entropy = -np.sum(proportions * np.log(proportions + 1e-10))

    # Maximum possible entropy (when all clusters are equal size)
    max_entropy = np.log(len(unique_labels))

    # Normalize to [0, 1]
    if max_entropy > 0:
        balance = entropy / max_entropy
    else:
        balance = 0.0

    return balance


def compute_clustering_metrics(
    X: pd.DataFrame,
    labels: np.ndarray,
    metric_names: list = None,
    include_balance: bool = True
) -> Dict[str, float]:
    """
    Compute multiple clustering evaluation metrics.

    Parameters
    ----------
    X : pd.DataFrame
        Input data.

    labels : np.ndarray
        Cluster labels for each sample.

    metric_names : list, optional
        List of metrics to compute. If None, computes all metrics.
        Available: 'silhouette', 'calinski_harabasz', 'davies_bouldin', 'cluster_balance'

    include_balance : bool, default=True
        Whether to include cluster balance score in the results.

    Returns
    -------
    metrics : dict
        Dictionary mapping metric names to values.

    Notes
    -----
    Metric interpretations:
    - Silhouette: [-1, 1], higher is better. >0.5 is good, >0.7 is excellent
    - Calinski-Harabasz: [0, inf), higher is better
    - Davies-Bouldin: [0, inf), lower is better
    - Cluster Balance: [0, 1], higher is better. >0.8 is good, 1.0 is perfect balance

    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> from clustertk.evaluation import compute_clustering_metrics
    >>>
    >>> X = pd.DataFrame(np.random.rand(100, 5))
    >>> labels = np.random.randint(0, 3, 100)
    >>> metrics = compute_clustering_metrics(X, labels)
    >>> print(f"Silhouette: {metrics['silhouette']:.3f}")
    >>> print(f"Balance: {metrics['cluster_balance']:.3f}")
    """
    if metric_names is None:
        metric_names = ['silhouette', 'calinski_harabasz', 'davies_bouldin']
        if include_balance:
            metric_names.append('cluster_balance')

    # Check if we have at least 2 clusters
    n_clusters = len(np.unique(labels))
    if n_clusters < 2:
        raise ValueError("Need at least 2 clusters to compute metrics")

    metrics = {}

    if 'silhouette' in metric_names:
        try:
            metrics['silhouette'] = silhouette_score(X, labels)
        except Exception as e:
            metrics['silhouette'] = np.nan
            import warnings
            warnings.warn(f"Could not compute silhouette score: {e}", UserWarning)

    if 'calinski_harabasz' in metric_names:
        try:
            metrics['calinski_harabasz'] = calinski_harabasz_score(X, labels)
        except Exception as e:
            metrics['calinski_harabasz'] = np.nan
            import warnings
            warnings.warn(f"Could not compute Calinski-Harabasz score: {e}", UserWarning)

    if 'davies_bouldin' in metric_names:
        try:
            metrics['davies_bouldin'] = davies_bouldin_score(X, labels)
        except Exception as e:
            metrics['davies_bouldin'] = np.nan
            import warnings
            warnings.warn(f"Could not compute Davies-Bouldin score: {e}", UserWarning)

    if 'cluster_balance' in metric_names:
        try:
            metrics['cluster_balance'] = cluster_balance_score(labels)
        except Exception as e:
            metrics['cluster_balance'] = np.nan
            import warnings
            warnings.warn(f"Could not compute cluster balance score: {e}", UserWarning)

    return metrics


def interpret_silhouette(score: float) -> str:
    """
    Interpret silhouette score.

    Parameters
    ----------
    score : float
        Silhouette score.

    Returns
    -------
    interpretation : str
        Human-readable interpretation.
    """
    if score < 0:
        return "Poor (samples may be in wrong clusters)"
    elif score < 0.25:
        return "Weak (structure is weak)"
    elif score < 0.5:
        return "Fair (reasonable structure)"
    elif score < 0.7:
        return "Good (clear structure)"
    else:
        return "Excellent (strong, distinct clusters)"


def get_metrics_summary(metrics: Dict[str, float]) -> pd.DataFrame:
    """
    Get a formatted summary of clustering metrics.

    Parameters
    ----------
    metrics : dict
        Dictionary of metric name -> value.

    Returns
    -------
    summary : pd.DataFrame
        Formatted summary with interpretation.
    """
    summary_data = []

    if 'silhouette' in metrics:
        summary_data.append({
            'metric': 'Silhouette Score',
            'value': metrics['silhouette'],
            'range': '[-1, 1]',
            'interpretation': 'Higher is better',
            'quality': interpret_silhouette(metrics['silhouette'])
        })

    if 'calinski_harabasz' in metrics:
        summary_data.append({
            'metric': 'Calinski-Harabasz Index',
            'value': metrics['calinski_harabasz'],
            'range': '[0, ∞)',
            'interpretation': 'Higher is better',
            'quality': 'Good' if metrics['calinski_harabasz'] > 100 else 'Fair'
        })

    if 'davies_bouldin' in metrics:
        db_score = metrics['davies_bouldin']
        if db_score < 0.5:
            quality = 'Excellent'
        elif db_score < 1.0:
            quality = 'Good'
        elif db_score < 1.5:
            quality = 'Fair'
        else:
            quality = 'Poor'

        summary_data.append({
            'metric': 'Davies-Bouldin Index',
            'value': db_score,
            'range': '[0, ∞)',
            'interpretation': 'Lower is better',
            'quality': quality
        })

    if 'cluster_balance' in metrics:
        balance_score = metrics['cluster_balance']
        if balance_score > 0.8:
            quality = 'Excellent'
        elif balance_score > 0.6:
            quality = 'Good'
        elif balance_score > 0.4:
            quality = 'Fair'
        else:
            quality = 'Poor'

        summary_data.append({
            'metric': 'Cluster Balance',
            'value': balance_score,
            'range': '[0, 1]',
            'interpretation': 'Higher is better',
            'quality': quality
        })

    return pd.DataFrame(summary_data)


def compare_clusterings(
    X: pd.DataFrame,
    labels_dict: Dict[str, np.ndarray]
) -> pd.DataFrame:
    """
    Compare multiple clustering results using metrics.

    Parameters
    ----------
    X : pd.DataFrame
        Input data.

    labels_dict : dict
        Dictionary mapping method names to label arrays.

    Returns
    -------
    comparison : pd.DataFrame
        DataFrame comparing all methods across all metrics.

    Examples
    --------
    >>> labels_dict = {
    ...     'KMeans_k3': kmeans_labels,
    ...     'GMM_k3': gmm_labels,
    ...     'KMeans_k4': kmeans_labels_k4
    ... }
    >>> comparison = compare_clusterings(X, labels_dict)
    """
    results = []

    for method_name, labels in labels_dict.items():
        metrics = compute_clustering_metrics(X, labels)
        metrics['method'] = method_name
        metrics['n_clusters'] = len(np.unique(labels))
        results.append(metrics)

    df = pd.DataFrame(results)

    # Reorder columns
    cols = ['method', 'n_clusters'] + [c for c in df.columns if c not in ['method', 'n_clusters']]
    return df[cols]
