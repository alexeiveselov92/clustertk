"""
Feature importance analysis for cluster interpretation.

This module provides advanced feature importance analysis:
- Permutation importance (sklearn-based)
- SHAP values integration (optional, requires shap package)
- Feature contribution to cluster separation
"""

from typing import Dict, List, Optional, Tuple, Union, Literal
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import silhouette_score


class FeatureImportanceAnalyzer:
    """
    Analyze feature importance for clustering results.

    Provides multiple methods for understanding which features
    are most important for cluster formation and separation.

    Parameters
    ----------
    verbose : bool, default=False
        Whether to print progress messages.

    Attributes
    ----------
    permutation_importance_ : pd.DataFrame
        Results of permutation importance analysis.
    feature_contribution_ : pd.DataFrame
        Feature contribution to cluster separation.
    shap_values_ : np.ndarray or None
        SHAP values if SHAP analysis was performed.

    Examples
    --------
    >>> from clustertk.interpretation import FeatureImportanceAnalyzer
    >>> analyzer = FeatureImportanceAnalyzer()
    >>> results = analyzer.analyze(X, labels, method='permutation')
    >>> print(results.sort_values('importance', ascending=False))
    """

    def __init__(self, verbose: bool = False):
        self.verbose = verbose

        # Results storage
        self.permutation_importance_: Optional[pd.DataFrame] = None
        self.feature_contribution_: Optional[pd.DataFrame] = None
        self.shap_values_: Optional[np.ndarray] = None

    def analyze(
        self,
        X: pd.DataFrame,
        labels: np.ndarray,
        method: Literal['permutation', 'shap', 'contribution', 'all'] = 'all',
        n_repeats: int = 10,
        random_state: Optional[int] = 42
    ) -> Dict[str, pd.DataFrame]:
        """
        Analyze feature importance using specified method(s).

        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix (preprocessed data used for clustering).
        labels : np.ndarray
            Cluster labels.
        method : {'permutation', 'shap', 'contribution', 'all'}, default='all'
            Which analysis method(s) to use:
            - 'permutation': Permutation importance (how much each feature affects clustering quality)
            - 'shap': SHAP values (requires shap package)
            - 'contribution': Feature contribution to cluster separation
            - 'all': Run all available methods
        n_repeats : int, default=10
            Number of permutation repeats (for permutation method).
        random_state : int or None, default=42
            Random state for reproducibility.

        Returns
        -------
        results : dict
            Dictionary with analysis results:
            - 'permutation': DataFrame with permutation importance
            - 'contribution': DataFrame with feature contribution
            - 'shap': SHAP values (if method='shap' or 'all')

        Examples
        --------
        >>> results = analyzer.analyze(X, labels, method='permutation')
        >>> print(results['permutation'].head())
        """
        if self.verbose:
            print(f"Starting feature importance analysis (method={method})...")

        results = {}

        # Run requested analysis
        if method in ['permutation', 'all']:
            if self.verbose:
                print("Computing permutation importance...")
            results['permutation'] = self._permutation_importance(
                X, labels, n_repeats, random_state
            )

        if method in ['contribution', 'all']:
            if self.verbose:
                print("Computing feature contribution...")
            results['contribution'] = self._feature_contribution(X, labels)

        if method in ['shap', 'all']:
            try:
                if self.verbose:
                    print("Computing SHAP values...")
                results['shap'] = self._shap_importance(X, labels)
            except ImportError:
                if self.verbose:
                    print("SHAP not available (install with: pip install shap)")
                if method == 'shap':
                    raise ImportError(
                        "SHAP analysis requested but shap package not installed. "
                        "Install with: pip install shap"
                    )

        if self.verbose:
            print("Feature importance analysis complete!")

        return results

    def _permutation_importance(
        self,
        X: pd.DataFrame,
        labels: np.ndarray,
        n_repeats: int,
        random_state: Optional[int]
    ) -> pd.DataFrame:
        """
        Compute permutation importance.

        Measures how much shuffling each feature decreases clustering quality
        (measured by silhouette score).

        For large datasets (>10k samples), uses sampling to avoid memory issues
        with silhouette score computation.
        """
        from sklearn.base import BaseEstimator

        # For large datasets, use sampling to avoid OOM with silhouette score
        # Silhouette computes pairwise distances: O(n²) memory
        max_samples_for_silhouette = 10000
        use_sampling = len(X) > max_samples_for_silhouette

        if use_sampling:
            # Sample for silhouette computation
            rng = np.random.RandomState(random_state)
            n_samples = min(max_samples_for_silhouette, len(X))
            sample_indices = rng.choice(len(X), size=n_samples, replace=False)

            if self.verbose:
                print(f"  Large dataset detected ({len(X):,} samples)")
                print(f"  Using {n_samples:,} samples for silhouette computation to avoid OOM")

        # Create a dummy estimator that just returns the labels
        class ClusteringEstimator(BaseEstimator):
            def __init__(self, labels, use_sampling=False, sample_indices=None):
                self.labels_ = labels
                self.use_sampling = use_sampling
                self.sample_indices = sample_indices

            def fit(self, X, y=None):
                return self

            def predict(self, X):
                return self.labels_

            def score(self, X, y=None):
                # Use silhouette score as the metric
                if len(np.unique(self.labels_)) < 2:
                    return 0.0

                # For large datasets, compute silhouette on sample
                if self.use_sampling:
                    X_sample = X.iloc[self.sample_indices] if hasattr(X, 'iloc') else X[self.sample_indices]
                    labels_sample = self.labels_[self.sample_indices]

                    # Check if sample has at least 2 clusters
                    if len(np.unique(labels_sample)) < 2:
                        return 0.0

                    return silhouette_score(X_sample, labels_sample)
                else:
                    return silhouette_score(X, self.labels_)

        estimator = ClusteringEstimator(
            labels,
            use_sampling=use_sampling,
            sample_indices=sample_indices if use_sampling else None
        )
        estimator.fit(X)

        # Compute permutation importance
        # Note: we pass dummy y (same as labels) since permutation_importance requires it
        result = permutation_importance(
            estimator,
            X,
            y=labels,  # Dummy y - not actually used, but required by API
            scoring=None,  # Will use estimator.score()
            n_repeats=n_repeats,
            random_state=random_state,
            n_jobs=-1
        )

        # Create DataFrame with results
        importance_df = pd.DataFrame({
            'feature': X.columns,
            'importance': result.importances_mean,
            'std': result.importances_std
        })

        # Sort by importance
        importance_df = importance_df.sort_values('importance', ascending=False)
        importance_df = importance_df.reset_index(drop=True)

        self.permutation_importance_ = importance_df
        return importance_df

    def _feature_contribution(
        self,
        X: pd.DataFrame,
        labels: np.ndarray
    ) -> pd.DataFrame:
        """
        Compute feature contribution to cluster separation.

        Measures how much each feature contributes to separating clusters
        using variance ratio (between-cluster variance / within-cluster variance).

        Optimized version using vectorized operations.
        """
        features = []
        contributions = []

        # Filter out noise points once
        mask = labels != -1
        labels_filtered = labels[mask]
        unique_labels = np.unique(labels_filtered)

        if len(unique_labels) < 2:
            # Not enough clusters for contribution analysis
            return pd.DataFrame({
                'feature': X.columns,
                'contribution': [0.0] * len(X.columns)
            })

        # Relabel clusters to 0, 1, 2, ... for efficient np.bincount
        label_map = {old: new for new, old in enumerate(unique_labels)}
        labels_remapped = np.array([label_map[l] for l in labels_filtered])
        n_clusters = len(unique_labels)

        for col in X.columns:
            feature_data = X[col].values[mask]

            # Fully vectorized NumPy computation using bincount (no loops!)
            # This is true vectorization - much faster than list comprehensions
            cluster_sums = np.bincount(labels_remapped, weights=feature_data, minlength=n_clusters)
            cluster_counts = np.bincount(labels_remapped, minlength=n_clusters)
            cluster_means = cluster_sums / cluster_counts

            # Compute variances vectorized
            squared_diffs = (feature_data - cluster_means[labels_remapped])**2
            cluster_squared_sums = np.bincount(labels_remapped, weights=squared_diffs, minlength=n_clusters)
            cluster_vars = cluster_squared_sums / (cluster_counts - 1)

            # Overall mean (vectorized)
            overall_mean = feature_data.mean()

            # Between-cluster variance (fully vectorized)
            between_var = np.sum(
                cluster_counts * (cluster_means - overall_mean)**2
            ) / cluster_counts.sum()

            # Within-cluster variance (fully vectorized)
            within_var = np.sum(cluster_counts * cluster_vars) / cluster_counts.sum()

            # Variance ratio (higher = better separation)
            if within_var > 0:
                contribution = between_var / within_var
            else:
                contribution = 0.0

            features.append(col)
            contributions.append(contribution)

        contribution_df = pd.DataFrame({
            'feature': features,
            'contribution': contributions
        })

        # Sort by contribution
        contribution_df = contribution_df.sort_values('contribution', ascending=False)
        contribution_df = contribution_df.reset_index(drop=True)

        self.feature_contribution_ = contribution_df
        return contribution_df

    def _shap_importance(
        self,
        X: pd.DataFrame,
        labels: np.ndarray
    ) -> Dict[str, Union[np.ndarray, pd.DataFrame]]:
        """
        Compute SHAP values for feature importance.

        Requires the shap package to be installed.
        """
        try:
            import shap
        except ImportError:
            raise ImportError(
                "SHAP analysis requires the 'shap' package. "
                "Install with: pip install shap"
            )

        # For clustering, we'll use a simple approach:
        # Train a classifier to predict cluster labels, then compute SHAP values
        from sklearn.ensemble import RandomForestClassifier

        # Filter out noise points if present
        mask = labels != -1
        X_filtered = X[mask]
        labels_filtered = labels[mask]

        if len(np.unique(labels_filtered)) < 2:
            raise ValueError("Need at least 2 clusters for SHAP analysis")

        # Train classifier
        clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        clf.fit(X_filtered, labels_filtered)

        # Compute SHAP values
        explainer = shap.TreeExplainer(clf)
        shap_values = explainer.shap_values(X_filtered)

        # If multiclass, shap_values is a list (one array per class)
        # Take mean absolute SHAP value across all classes
        if isinstance(shap_values, list):
            # Each element: (n_samples, n_features), aggregate across classes and samples
            mean_shap = np.mean([np.abs(sv).mean(axis=0) for sv in shap_values], axis=0)
        else:
            # Binary or regression: (n_samples, n_features), aggregate across samples
            mean_shap = np.abs(shap_values).mean(axis=0)

        # Ensure mean_shap is 1D (flatten if needed)
        mean_shap = np.asarray(mean_shap).flatten()

        # Create DataFrame
        shap_df = pd.DataFrame({
            'feature': X.columns,
            'shap_importance': mean_shap
        })

        shap_df = shap_df.sort_values('shap_importance', ascending=False)
        shap_df = shap_df.reset_index(drop=True)

        self.shap_values_ = shap_values

        return {
            'shap_values': shap_values,
            'importance': shap_df,
            'explainer': explainer
        }

    def get_top_features(
        self,
        method: Literal['permutation', 'contribution', 'shap'] = 'permutation',
        n: int = 10
    ) -> pd.DataFrame:
        """
        Get top N most important features from specified method.

        Parameters
        ----------
        method : {'permutation', 'contribution', 'shap'}, default='permutation'
            Which importance method to use.
        n : int, default=10
            Number of top features to return.

        Returns
        -------
        top_features : pd.DataFrame
            DataFrame with top N features.
        """
        if method == 'permutation':
            if self.permutation_importance_ is None:
                raise ValueError("Run analyze() with method='permutation' first")
            return self.permutation_importance_.head(n)

        elif method == 'contribution':
            if self.feature_contribution_ is None:
                raise ValueError("Run analyze() with method='contribution' first")
            return self.feature_contribution_.head(n)

        elif method == 'shap':
            if self.shap_values_ is None:
                raise ValueError("Run analyze() with method='shap' first")
            # Return from stored SHAP results
            # This would need to be extracted from the shap results dict
            raise NotImplementedError("SHAP top features - use results['shap']['importance']")

        else:
            raise ValueError(f"Unknown method: {method}")


def quick_feature_importance(
    X: pd.DataFrame,
    labels: np.ndarray,
    method: Literal['permutation', 'contribution', 'all'] = 'all',
    n_top: int = 10
) -> pd.DataFrame:
    """
    Quick feature importance analysis (convenience function).

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix.
    labels : np.ndarray
        Cluster labels.
    method : {'permutation', 'contribution', 'all'}, default='all'
        Analysis method.
    n_top : int, default=10
        Number of top features to return.

    Returns
    -------
    results : pd.DataFrame
        Top N most important features.

    Examples
    --------
    >>> top_features = quick_feature_importance(X, labels, method='permutation')
    >>> print(top_features)
    """
    analyzer = FeatureImportanceAnalyzer(verbose=False)
    results = analyzer.analyze(X, labels, method=method)

    if method == 'all':
        # Combine results from both methods
        perm = results['permutation'][['feature', 'importance']].rename(
            columns={'importance': 'permutation_importance'}
        )
        contrib = results['contribution'][['feature', 'contribution']].rename(
            columns={'contribution': 'feature_contribution'}
        )

        combined = perm.merge(contrib, on='feature')
        combined['combined_score'] = (
            combined['permutation_importance'] + combined['feature_contribution']
        )
        combined = combined.sort_values('combined_score', ascending=False)
        return combined.head(n_top)

    elif method == 'permutation':
        return results['permutation'].head(n_top)

    elif method == 'contribution':
        return results['contribution'].head(n_top)
