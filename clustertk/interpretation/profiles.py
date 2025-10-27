"""
Cluster profiling and interpretation.

This module provides functionality for creating and analyzing cluster profiles,
identifying distinguishing features, and interpreting cluster characteristics.
"""

from typing import Optional, Dict, List
import pandas as pd
import numpy as np


class ClusterProfiler:
    """
    Create and analyze cluster profiles.

    This class generates statistical profiles of clusters, identifies
    distinguishing features, and provides category-based analysis.

    Parameters
    ----------
    normalize_per_feature : bool, default=True
        Whether to normalize each feature separately for visualization.
        This makes features comparable on the same scale.

    Attributes
    ----------
    profiles_ : pd.DataFrame
        Mean values of features for each cluster.

    profiles_normalized_ : pd.DataFrame
        Normalized profiles (0-1 scale per feature).

    top_features_ : dict
        Top distinguishing features for each cluster.

    category_scores_ : pd.DataFrame
        Average scores by category for each cluster.

    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> from clustertk.interpretation import ClusterProfiler
    >>>
    >>> # Create sample data with labels
    >>> X = pd.DataFrame(np.random.rand(100, 5), columns=[f'f{i}' for i in range(5)])
    >>> labels = np.random.randint(0, 3, 100)
    >>>
    >>> # Create profiles
    >>> profiler = ClusterProfiler()
    >>> profiles = profiler.create_profiles(X, labels)
    >>>
    >>> # Get top features
    >>> top_features = profiler.get_top_features(n=3)
    """

    def __init__(self, normalize_per_feature: bool = True):
        self.normalize_per_feature = normalize_per_feature
        self.profiles_: Optional[pd.DataFrame] = None
        self.profiles_normalized_: Optional[pd.DataFrame] = None
        self.top_features_: Optional[Dict[int, Dict[str, List]]] = None
        self.category_scores_: Optional[pd.DataFrame] = None
        self._feature_names: Optional[List[str]] = None

    def create_profiles(
        self,
        X: pd.DataFrame,
        labels: np.ndarray,
        feature_names: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Create cluster profiles with mean values.

        Parameters
        ----------
        X : pd.DataFrame
            Feature data.

        labels : np.ndarray
            Cluster labels.

        feature_names : list, optional
            Feature names to use. If None, uses X.columns.

        Returns
        -------
        profiles : pd.DataFrame
            Mean feature values for each cluster.
        """
        if not isinstance(X, pd.DataFrame):
            raise TypeError("X must be a pandas DataFrame")

        if len(labels) != len(X):
            raise ValueError("labels must have same length as X")

        self._feature_names = feature_names or X.columns.tolist()

        # Create DataFrame with labels
        data_with_labels = X.copy()
        data_with_labels['cluster'] = labels

        # Compute mean profiles
        self.profiles_ = data_with_labels.groupby('cluster')[self._feature_names].mean()

        # Normalize profiles per feature (for visualization)
        if self.normalize_per_feature:
            self.profiles_normalized_ = self.profiles_.copy()
            for col in self.profiles_normalized_.columns:
                col_min = self.profiles_normalized_[col].min()
                col_max = self.profiles_normalized_[col].max()
                if col_max != col_min:
                    self.profiles_normalized_[col] = (
                        (self.profiles_normalized_[col] - col_min) / (col_max - col_min)
                    )
                else:
                    self.profiles_normalized_[col] = 0.5

        return self.profiles_

    def get_top_features(self, n: int = 5) -> Dict[int, Dict[str, List]]:
        """
        Get top distinguishing features for each cluster.

        Identifies features where each cluster deviates most from the global mean.

        Parameters
        ----------
        n : int, default=5
            Number of top features to return.

        Returns
        -------
        top_features : dict
            Dictionary mapping cluster ID to dict with 'high' and 'low' feature lists.
            Each list contains (feature_name, deviation_score) tuples.
        """
        if self.profiles_ is None:
            raise ValueError("Must call create_profiles first")

        # Compute deviation from global mean
        global_mean = self.profiles_.mean(axis=0)

        self.top_features_ = {}

        for cluster in self.profiles_.index:
            deviation = self.profiles_.loc[cluster] - global_mean

            # Top positive deviations
            top_positive = deviation.nlargest(n)
            # Top negative deviations
            top_negative = deviation.nsmallest(n)

            self.top_features_[cluster] = {
                'high': [(feat, val) for feat, val in top_positive.items()],
                'low': [(feat, val) for feat, val in top_negative.items()]
            }

        return self.top_features_

    def analyze_by_categories(
        self,
        X: pd.DataFrame,
        labels: np.ndarray,
        category_mapping: Dict[str, List[str]]
    ) -> pd.DataFrame:
        """
        Analyze clusters by feature categories.

        Parameters
        ----------
        X : pd.DataFrame
            Feature data.

        labels : np.ndarray
            Cluster labels.

        category_mapping : dict
            Dictionary mapping category names to lists of feature names.
            Example: {'behavioral': ['feature1', 'feature2'], 'social': ['feature3']}

        Returns
        -------
        category_scores : pd.DataFrame
            Average scores by category for each cluster (normalized 0-1).
        """
        if not isinstance(X, pd.DataFrame):
            raise TypeError("X must be a pandas DataFrame")

        # Create DataFrame with labels
        data_with_labels = X.copy()
        data_with_labels['cluster'] = labels

        category_scores = {}

        for category, features in category_mapping.items():
            # Filter features that exist in data
            available_features = [f for f in features if f in X.columns]

            if available_features:
                # Compute mean of category features for each cluster
                category_means = data_with_labels.groupby('cluster')[available_features].mean()
                category_avg = category_means.mean(axis=1)

                # Normalize to 0-1 scale
                cat_min = category_avg.min()
                cat_max = category_avg.max()
                if cat_max != cat_min:
                    category_scores[category] = (category_avg - cat_min) / (cat_max - cat_min)
                else:
                    category_scores[category] = pd.Series(0.5, index=category_avg.index)

        self.category_scores_ = pd.DataFrame(category_scores)
        return self.category_scores_

    def get_cluster_summary(self, cluster_id: int, n_top: int = 5) -> str:
        """
        Get a text summary of a cluster.

        Parameters
        ----------
        cluster_id : int
            Cluster ID to summarize.

        n_top : int, default=5
            Number of top features to include.

        Returns
        -------
        summary : str
            Human-readable cluster summary.
        """
        if self.profiles_ is None:
            raise ValueError("Must call create_profiles first")

        if cluster_id not in self.profiles_.index:
            raise ValueError(f"Cluster {cluster_id} not found")

        if self.top_features_ is None:
            self.get_top_features(n=n_top)

        summary = []
        summary.append(f"Cluster {cluster_id} Profile")
        summary.append("=" * 40)

        summary.append("\nHighest Features:")
        for feat, val in self.top_features_[cluster_id]['high'][:n_top]:
            summary.append(f"  ↑ {feat}: {val:+.3f} (vs mean)")

        summary.append("\nLowest Features:")
        for feat, val in self.top_features_[cluster_id]['low'][:n_top]:
            summary.append(f"  ↓ {feat}: {val:+.3f} (vs mean)")

        if self.category_scores_ is not None and cluster_id in self.category_scores_.index:
            summary.append("\nCategory Scores (0-1 scale):")
            for category, score in self.category_scores_.loc[cluster_id].items():
                bar = '█' * int(score * 20)
                summary.append(f"  {category}: {bar} {score:.2f}")

        return "\n".join(summary)

    def compare_clusters(self, cluster1: int, cluster2: int, n_features: int = 10) -> pd.DataFrame:
        """
        Compare two clusters by their most different features.

        Parameters
        ----------
        cluster1 : int
            First cluster ID.

        cluster2 : int
            Second cluster ID.

        n_features : int, default=10
            Number of most different features to show.

        Returns
        -------
        comparison : pd.DataFrame
            DataFrame with features sorted by difference.
        """
        if self.profiles_ is None:
            raise ValueError("Must call create_profiles first")

        if cluster1 not in self.profiles_.index or cluster2 not in self.profiles_.index:
            raise ValueError("One or both cluster IDs not found")

        # Compute differences
        diff = self.profiles_.loc[cluster1] - self.profiles_.loc[cluster2]

        # Sort by absolute difference
        diff_abs = diff.abs().sort_values(ascending=False)

        # Create comparison DataFrame
        comparison = pd.DataFrame({
            'feature': diff_abs.index,
            f'cluster_{cluster1}': [self.profiles_.loc[cluster1, f] for f in diff_abs.index],
            f'cluster_{cluster2}': [self.profiles_.loc[cluster2, f] for f in diff_abs.index],
            'difference': [diff[f] for f in diff_abs.index],
            'abs_difference': diff_abs.values
        })

        return comparison.head(n_features).reset_index(drop=True)

    def __repr__(self) -> str:
        """String representation."""
        if self.profiles_ is not None:
            n_clusters = len(self.profiles_)
            n_features = len(self.profiles_.columns)
            return f"ClusterProfiler(n_clusters={n_clusters}, n_features={n_features})"
        return "ClusterProfiler()"


def quick_profile(
    X: pd.DataFrame,
    labels: np.ndarray,
    n_top: int = 5
) -> pd.DataFrame:
    """
    Quick function to create cluster profiles.

    Parameters
    ----------
    X : pd.DataFrame
        Feature data.

    labels : np.ndarray
        Cluster labels.

    n_top : int, default=5
        Number of top features.

    Returns
    -------
    profiles : pd.DataFrame
        Cluster profiles.
    """
    profiler = ClusterProfiler()
    profiles = profiler.create_profiles(X, labels)
    profiler.get_top_features(n=n_top)

    return profiles
