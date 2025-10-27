"""
Optimal cluster number selection.

This module provides functionality for determining the optimal number of clusters
using various metrics and methods.
"""

from typing import Tuple, Dict, Any, Optional
import pandas as pd
import numpy as np
from clustertk.evaluation.metrics import compute_clustering_metrics


class OptimalKFinder:
    """
    Find optimal number of clusters using multiple metrics.

    This class tests different numbers of clusters and uses various metrics
    to recommend the optimal k.

    Parameters
    ----------
    k_range : tuple, default=(2, 10)
        Range of k values to test (min_k, max_k).

    method : str, default='voting'
        Method for selecting optimal k:
        - 'voting': Majority vote across metrics
        - 'silhouette': Based on silhouette score only
        - 'calinski_harabasz': Based on Calinski-Harabasz only
        - 'davies_bouldin': Based on Davies-Bouldin only

    random_state : int, default=42
        Random state for reproducibility.

    Attributes
    ----------
    results_ : pd.DataFrame
        Results for all tested k values with all metrics.

    optimal_k_ : int
        Recommended optimal number of clusters.

    metric_votes_ : dict
        Vote from each metric for optimal k.

    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> from clustertk.evaluation import OptimalKFinder
    >>> from clustertk.clustering import KMeansClustering
    >>>
    >>> X = pd.DataFrame(np.random.rand(100, 5))
    >>>
    >>> finder = OptimalKFinder(k_range=(2, 6))
    >>> optimal_k = finder.find_optimal_k(X, KMeansClustering)
    >>> print(f"Optimal k: {optimal_k}")
    """

    def __init__(
        self,
        k_range: Tuple[int, int] = (2, 10),
        method: str = 'voting',
        random_state: int = 42
    ):
        valid_methods = ['voting', 'silhouette', 'calinski_harabasz', 'davies_bouldin']

        if method not in valid_methods:
            raise ValueError(f"Invalid method '{method}'. Must be one of {valid_methods}.")

        if k_range[0] < 2:
            raise ValueError("Minimum k must be at least 2")

        if k_range[0] >= k_range[1]:
            raise ValueError("max_k must be greater than min_k")

        self.k_range = k_range
        self.method = method
        self.random_state = random_state
        self.results_: Optional[pd.DataFrame] = None
        self.optimal_k_: Optional[int] = None
        self.metric_votes_: Optional[Dict[str, int]] = None

    def find_optimal_k(self, X: pd.DataFrame, clusterer_class: type) -> int:
        """
        Find optimal number of clusters.

        Parameters
        ----------
        X : pd.DataFrame
            Input data.

        clusterer_class : type
            Clustering algorithm class (e.g., KMeansClustering).
            Must have __init__(n_clusters, random_state) signature.

        Returns
        -------
        optimal_k : int
            Recommended number of clusters.
        """
        if not isinstance(X, pd.DataFrame):
            raise TypeError("X must be a pandas DataFrame")

        results = []
        k_values = range(self.k_range[0], self.k_range[1] + 1)

        # Test each k value
        for k in k_values:
            # Fit clusterer (some algorithms don't support random_state)
            # HierarchicalClustering is deterministic and doesn't use random_state
            if clusterer_class.__name__ == 'HierarchicalClustering':
                clusterer = clusterer_class(n_clusters=k)
            else:
                clusterer = clusterer_class(n_clusters=k, random_state=self.random_state)
            labels = clusterer.fit_predict(X)

            # Compute metrics
            metrics = compute_clustering_metrics(X, labels)
            metrics['k'] = k

            # Add inertia if available (for K-Means)
            if hasattr(clusterer, 'inertia_'):
                metrics['inertia'] = clusterer.inertia_

            # Add BIC/AIC if available (for GMM)
            if hasattr(clusterer, 'bic_'):
                metrics['bic'] = clusterer.bic_
                metrics['aic'] = clusterer.aic_

            results.append(metrics)

        self.results_ = pd.DataFrame(results)

        # Determine optimal k based on method
        if self.method == 'voting':
            self.optimal_k_ = self._voting_method()
        elif self.method == 'silhouette':
            self.optimal_k_ = self.results_.loc[
                self.results_['silhouette'].idxmax(), 'k'
            ]
        elif self.method == 'calinski_harabasz':
            self.optimal_k_ = self.results_.loc[
                self.results_['calinski_harabasz'].idxmax(), 'k'
            ]
        elif self.method == 'davies_bouldin':
            self.optimal_k_ = self.results_.loc[
                self.results_['davies_bouldin'].idxmin(), 'k'
            ]

        return int(self.optimal_k_)

    def _voting_method(self) -> int:
        """Use voting across metrics to determine optimal k."""
        votes = {}

        # Silhouette: higher is better
        k_silhouette = self.results_.loc[
            self.results_['silhouette'].idxmax(), 'k'
        ]
        votes['silhouette'] = int(k_silhouette)

        # Calinski-Harabasz: higher is better
        k_ch = self.results_.loc[
            self.results_['calinski_harabasz'].idxmax(), 'k'
        ]
        votes['calinski_harabasz'] = int(k_ch)

        # Davies-Bouldin: lower is better
        k_db = self.results_.loc[
            self.results_['davies_bouldin'].idxmin(), 'k'
        ]
        votes['davies_bouldin'] = int(k_db)

        self.metric_votes_ = votes

        # Count votes
        from collections import Counter
        vote_counts = Counter(votes.values())

        # Return most common vote, or the one from silhouette in case of tie
        optimal_k = vote_counts.most_common(1)[0][0]

        return optimal_k

    def get_results_summary(self) -> pd.DataFrame:
        """
        Get summary of results for all tested k values.

        Returns
        -------
        summary : pd.DataFrame
            Results with all metrics for each k.
        """
        if self.results_ is None:
            raise ValueError("Must run find_optimal_k first")

        return self.results_.copy()

    def get_recommendation_report(self) -> str:
        """
        Get a detailed report explaining the recommendation.

        Returns
        -------
        report : str
            Human-readable report.
        """
        if self.results_ is None or self.optimal_k_ is None:
            raise ValueError("Must run find_optimal_k first")

        report = []
        report.append("=" * 60)
        report.append("OPTIMAL CLUSTER NUMBER RECOMMENDATION")
        report.append("=" * 60)
        report.append(f"\nRecommended k: {self.optimal_k_}")
        report.append(f"Method: {self.method}")

        if self.method == 'voting' and self.metric_votes_:
            report.append("\nMetric votes:")
            for metric, k in self.metric_votes_.items():
                indicator = " ✓" if k == self.optimal_k_ else ""
                report.append(f"  - {metric}: k={k}{indicator}")

        # Show metrics for optimal k
        optimal_row = self.results_[self.results_['k'] == self.optimal_k_].iloc[0]
        report.append(f"\nMetrics for k={self.optimal_k_}:")
        report.append(f"  - Silhouette: {optimal_row['silhouette']:.3f}")
        report.append(f"  - Calinski-Harabasz: {optimal_row['calinski_harabasz']:.1f}")
        report.append(f"  - Davies-Bouldin: {optimal_row['davies_bouldin']:.3f}")

        return "\n".join(report)

    def __repr__(self) -> str:
        """String representation."""
        if self.optimal_k_ is not None:
            return f"OptimalKFinder(k_range={self.k_range}, optimal_k={self.optimal_k_})"
        return f"OptimalKFinder(k_range={self.k_range})"


def quick_optimal_k(
    X: pd.DataFrame,
    clusterer_class: type,
    k_range: Tuple[int, int] = (2, 10),
    random_state: int = 42
) -> int:
    """
    Quick function to find optimal k.

    Parameters
    ----------
    X : pd.DataFrame
        Input data.

    clusterer_class : type
        Clustering algorithm class.

    k_range : tuple, default=(2, 10)
        Range of k values to test.

    random_state : int, default=42
        Random state.

    Returns
    -------
    optimal_k : int
        Recommended number of clusters.
    """
    finder = OptimalKFinder(k_range=k_range, random_state=random_state)
    return finder.find_optimal_k(X, clusterer_class)
