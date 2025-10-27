"""
Correlation-based feature selection.

This module provides functionality for identifying and removing
highly correlated features to reduce redundancy.
"""

from typing import Optional, List, Tuple
import pandas as pd
import numpy as np


class CorrelationFilter:
    """
    Filter features based on correlation analysis.

    This class identifies pairs of highly correlated features and removes
    one feature from each pair to reduce redundancy. When multiple features
    are correlated with each other, the feature with the highest mean absolute
    correlation with other features is removed.

    Parameters
    ----------
    threshold : float, default=0.85
        Absolute correlation threshold. Feature pairs with |correlation| > threshold
        will be considered for removal.

    method : str, default='pearson'
        Correlation method to use:
        - 'pearson': Standard correlation coefficient
        - 'spearman': Spearman rank correlation
        - 'kendall': Kendall Tau correlation

    Attributes
    ----------
    correlation_matrix_ : pd.DataFrame
        Full correlation matrix of features.

    high_correlation_pairs_ : list
        List of feature pairs with high correlation.

    features_to_drop_ : list
        List of features identified for removal.

    selected_features_ : list
        List of features to keep after filtering.

    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> from clustertk.feature_selection import CorrelationFilter
    >>>
    >>> # Create data with correlated features
    >>> df = pd.DataFrame({
    ...     'a': np.random.rand(100),
    ...     'b': np.random.rand(100),
    ... })
    >>> df['c'] = df['a'] * 0.9 + np.random.rand(100) * 0.1  # highly correlated with 'a'
    >>>
    >>> # Filter correlated features
    >>> filter = CorrelationFilter(threshold=0.85)
    >>> filtered_df = filter.fit_transform(df)
    >>> print(filter.features_to_drop_)  # ['c'] - removed because correlated with 'a'
    """

    def __init__(
        self,
        threshold: float = 0.85,
        method: str = 'pearson'
    ):
        valid_methods = ['pearson', 'spearman', 'kendall']

        if method not in valid_methods:
            raise ValueError(
                f"Invalid method '{method}'. Must be one of {valid_methods}."
            )

        if not 0 <= threshold <= 1:
            raise ValueError("Threshold must be between 0 and 1.")

        self.threshold = threshold
        self.method = method
        self.correlation_matrix_: Optional[pd.DataFrame] = None
        self.high_correlation_pairs_: Optional[List[dict]] = None
        self.features_to_drop_: Optional[List[str]] = None
        self.selected_features_: Optional[List[str]] = None

    def fit(self, X: pd.DataFrame) -> 'CorrelationFilter':
        """
        Identify features to remove based on correlation analysis.

        Parameters
        ----------
        X : pd.DataFrame
            Input data.

        Returns
        -------
        self : CorrelationFilter
            Fitted filter.
        """
        if not isinstance(X, pd.DataFrame):
            raise TypeError("X must be a pandas DataFrame")

        # Compute correlation matrix
        self.correlation_matrix_ = X.corr(method=self.method).abs()

        # Find high correlation pairs
        self.high_correlation_pairs_ = self._find_high_correlation_pairs()

        # Identify features to drop
        self.features_to_drop_ = self._identify_features_to_drop()

        # Determine selected features
        self.selected_features_ = [
            col for col in X.columns if col not in self.features_to_drop_
        ]

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Remove highly correlated features.

        Parameters
        ----------
        X : pd.DataFrame
            Input data.

        Returns
        -------
        X_filtered : pd.DataFrame
            Data with correlated features removed.
        """
        if not isinstance(X, pd.DataFrame):
            raise TypeError("X must be a pandas DataFrame")

        if self.selected_features_ is None:
            raise ValueError("Filter must be fitted before transform")

        # Keep only selected features
        available_features = [f for f in self.selected_features_ if f in X.columns]

        if len(available_features) < len(self.selected_features_):
            missing = set(self.selected_features_) - set(X.columns)
            import warnings
            warnings.warn(
                f"Some selected features not found in data: {missing}",
                UserWarning
            )

        return X[available_features]

    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Fit and transform in one step.

        Parameters
        ----------
        X : pd.DataFrame
            Input data.

        Returns
        -------
        X_filtered : pd.DataFrame
            Data with correlated features removed.
        """
        return self.fit(X).transform(X)

    def _find_high_correlation_pairs(self) -> List[dict]:
        """Find pairs of features with high correlation."""
        high_corr_pairs = []

        # Get upper triangle indices (to avoid duplicates and self-correlations)
        for i in range(len(self.correlation_matrix_.columns)):
            for j in range(i + 1, len(self.correlation_matrix_.columns)):
                corr_value = self.correlation_matrix_.iloc[i, j]

                if corr_value > self.threshold:
                    high_corr_pairs.append({
                        'feature1': self.correlation_matrix_.columns[i],
                        'feature2': self.correlation_matrix_.columns[j],
                        'correlation': corr_value
                    })

        return high_corr_pairs

    def _identify_features_to_drop(self) -> List[str]:
        """
        Identify which features to drop from correlated pairs.

        Strategy: For each pair of correlated features, drop the one that has
        higher mean absolute correlation with all other features.
        """
        if not self.high_correlation_pairs_:
            return []

        # Create a graph of correlated features
        features_to_check = set()
        for pair in self.high_correlation_pairs_:
            features_to_check.add(pair['feature1'])
            features_to_check.add(pair['feature2'])

        # Compute mean absolute correlation for each feature
        mean_corr = {}
        for feature in features_to_check:
            # Exclude self-correlation
            other_corrs = self.correlation_matrix_[feature].drop(feature)
            mean_corr[feature] = other_corrs.mean()

        # Use a greedy approach: iteratively remove features with highest mean correlation
        # until no high correlation pairs remain
        features_to_drop = set()
        remaining_pairs = self.high_correlation_pairs_.copy()

        while remaining_pairs:
            # Find features involved in remaining pairs
            involved_features = set()
            for pair in remaining_pairs:
                involved_features.add(pair['feature1'])
                involved_features.add(pair['feature2'])

            # Remove features already marked for dropping
            involved_features -= features_to_drop

            if not involved_features:
                break

            # Find feature with highest mean correlation among involved features
            feature_to_drop = max(
                involved_features,
                key=lambda f: mean_corr.get(f, 0)
            )
            features_to_drop.add(feature_to_drop)

            # Remove pairs involving the dropped feature
            remaining_pairs = [
                pair for pair in remaining_pairs
                if pair['feature1'] != feature_to_drop and pair['feature2'] != feature_to_drop
            ]

        return list(features_to_drop)

    def get_correlation_summary(self) -> pd.DataFrame:
        """
        Get summary of high correlation pairs.

        Returns
        -------
        summary : pd.DataFrame
            DataFrame with high correlation pairs and which feature was dropped.
        """
        if self.high_correlation_pairs_ is None:
            raise ValueError("Filter must be fitted first")

        if not self.high_correlation_pairs_:
            return pd.DataFrame(columns=['feature1', 'feature2', 'correlation', 'dropped'])

        summary = pd.DataFrame(self.high_correlation_pairs_)

        # Add column indicating which feature was dropped
        summary['dropped'] = summary.apply(
            lambda row: (
                row['feature1'] if row['feature1'] in self.features_to_drop_
                else (row['feature2'] if row['feature2'] in self.features_to_drop_ else None)
            ),
            axis=1
        )

        return summary.sort_values('correlation', ascending=False).reset_index(drop=True)

    def get_correlation_matrix(self) -> pd.DataFrame:
        """
        Get the full correlation matrix.

        Returns
        -------
        correlation_matrix : pd.DataFrame
            Correlation matrix of all features.
        """
        if self.correlation_matrix_ is None:
            raise ValueError("Filter must be fitted first")

        return self.correlation_matrix_

    def __repr__(self) -> str:
        """String representation."""
        return f"CorrelationFilter(threshold={self.threshold}, method='{self.method}')"


def find_highly_correlated_features(
    X: pd.DataFrame,
    threshold: float = 0.85,
    method: str = 'pearson'
) -> pd.DataFrame:
    """
    Quick function to find highly correlated feature pairs.

    Parameters
    ----------
    X : pd.DataFrame
        Input data.

    threshold : float, default=0.85
        Correlation threshold.

    method : str, default='pearson'
        Correlation method.

    Returns
    -------
    high_corr_pairs : pd.DataFrame
        DataFrame with highly correlated feature pairs.
    """
    filter = CorrelationFilter(threshold=threshold, method=method)
    filter.fit(X)
    return filter.get_correlation_summary()
