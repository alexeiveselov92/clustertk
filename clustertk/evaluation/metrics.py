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


def compute_clustering_metrics(
    X: pd.DataFrame,
    labels: np.ndarray,
    metric_names: list = None
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
        Available: 'silhouette', 'calinski_harabasz', 'davies_bouldin'

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
    """
    if metric_names is None:
        metric_names = ['silhouette', 'calinski_harabasz', 'davies_bouldin']

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
