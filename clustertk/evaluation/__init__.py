"""
Evaluation module for clustering quality assessment.

This module provides:
- Clustering metrics (silhouette, calinski-harabasz, davies-bouldin)
- Optimal k finder
- Stability analysis via bootstrap resampling
"""

from clustertk.evaluation.metrics import (
    compute_clustering_metrics,
    interpret_silhouette,
    get_metrics_summary,
    compare_clusterings
)
from clustertk.evaluation.optimal_k import OptimalKFinder, quick_optimal_k
from clustertk.evaluation.stability import (
    ClusterStabilityAnalyzer,
    quick_stability_analysis
)

__all__ = [
    'compute_clustering_metrics',
    'interpret_silhouette',
    'get_metrics_summary',
    'compare_clusterings',
    'OptimalKFinder',
    'quick_optimal_k',
    'ClusterStabilityAnalyzer',
    'quick_stability_analysis',
]
