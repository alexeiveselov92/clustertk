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
        """
        from sklearn.base import BaseEstimator

        # Create a dummy estimator that just returns the labels
        class ClusteringEstimator(BaseEstimator):
            def __init__(self, labels):
                self.labels_ = labels

            def fit(self, X, y=None):
                return self

            def predict(self, X):
                return self.labels_

            def score(self, X, y=None):
                # Use silhouette score as the metric
                if len(np.unique(self.labels_)) < 2:
                    return 0.0
                return silhouette_score(X, self.labels_)

        estimator = ClusteringEstimator(labels)
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
        """
        features = []
        contributions = []

        for col in X.columns:
            feature_data = X[col].values

            # Calculate between-cluster variance
            cluster_means = []
            cluster_sizes = []
            for label in np.unique(labels):
                if label == -1:  # Skip noise points
                    continue
                cluster_data = feature_data[labels == label]
                cluster_means.append(np.mean(cluster_data))
                cluster_sizes.append(len(cluster_data))

            if len(cluster_means) < 2:
                contributions.append(0.0)
                features.append(col)
                continue

            overall_mean = np.mean(feature_data[labels != -1])
            between_var = np.sum(
                [size * (mean - overall_mean)**2
                 for size, mean in zip(cluster_sizes, cluster_means)]
            ) / sum(cluster_sizes)

            # Calculate within-cluster variance
            within_var = 0.0
            for label in np.unique(labels):
                if label == -1:
                    continue
                cluster_data = feature_data[labels == label]
                within_var += np.sum((cluster_data - np.mean(cluster_data))**2)

            within_var /= sum(cluster_sizes)

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
            mean_shap = np.mean([np.abs(sv).mean(axis=0) for sv in shap_values], axis=0)
        else:
            mean_shap = np.abs(shap_values).mean(axis=0)

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
