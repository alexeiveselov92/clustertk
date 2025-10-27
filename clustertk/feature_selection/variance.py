"""
Variance-based feature selection.

This module provides functionality for removing features with low variance,
which typically don't contribute much information to clustering.
"""

from typing import Optional, List
import pandas as pd
import numpy as np


class VarianceFilter:
    """
    Filter features based on variance analysis.

    Features with variance below the threshold are removed, as they provide
    little information for clustering. This is particularly useful after scaling,
    where features with near-constant values will have very low variance.

    Parameters
    ----------
    threshold : float, default=0.01
        Minimum variance threshold. Features with variance < threshold will be removed.

    Attributes
    ----------
    variances_ : pd.Series
        Variance of each feature.

    features_to_drop_ : list
        List of features with variance below threshold.

    selected_features_ : list
        List of features to keep after filtering.

    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> from clustertk.feature_selection import VarianceFilter
    >>>
    >>> # Create data with some low-variance features
    >>> df = pd.DataFrame({
    ...     'high_var': np.random.rand(100),
    ...     'low_var': np.ones(100) + np.random.rand(100) * 0.01,
    ...     'constant': np.ones(100)
    ... })
    >>>
    >>> # Filter low-variance features
    >>> filter = VarianceFilter(threshold=0.01)
    >>> filtered_df = filter.fit_transform(df)
    >>> print(filter.features_to_drop_)  # ['low_var', 'constant']
    """

    def __init__(self, threshold: float = 0.01):
        if threshold < 0:
            raise ValueError("Threshold must be non-negative.")

        self.threshold = threshold
        self.variances_: Optional[pd.Series] = None
        self.features_to_drop_: Optional[List[str]] = None
        self.selected_features_: Optional[List[str]] = None

    def fit(self, X: pd.DataFrame) -> 'VarianceFilter':
        """
        Identify features with low variance.

        Parameters
        ----------
        X : pd.DataFrame
            Input data (should be scaled for meaningful variance comparison).

        Returns
        -------
        self : VarianceFilter
            Fitted filter.
        """
        if not isinstance(X, pd.DataFrame):
            raise TypeError("X must be a pandas DataFrame")

        # Compute variance for each feature
        self.variances_ = X.var()

        # Identify features with variance below threshold
        self.features_to_drop_ = self.variances_[
            self.variances_ < self.threshold
        ].index.tolist()

        # Determine selected features
        self.selected_features_ = [
            col for col in X.columns if col not in self.features_to_drop_
        ]

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Remove low-variance features.

        Parameters
        ----------
        X : pd.DataFrame
            Input data.

        Returns
        -------
        X_filtered : pd.DataFrame
            Data with low-variance features removed.
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
            Data with low-variance features removed.
        """
        return self.fit(X).transform(X)

    def get_variance_summary(self) -> pd.DataFrame:
        """
        Get summary of feature variances.

        Returns
        -------
        summary : pd.DataFrame
            DataFrame with variance statistics for all features.
        """
        if self.variances_ is None:
            raise ValueError("Filter must be fitted first")

        summary = pd.DataFrame({
            'feature': self.variances_.index,
            'variance': self.variances_.values,
            'below_threshold': self.variances_.values < self.threshold
        })

        return summary.sort_values('variance', ascending=True).reset_index(drop=True)

    def get_removed_features(self) -> List[str]:
        """
        Get list of features that were removed.

        Returns
        -------
        removed : list
            List of removed feature names.
        """
        if self.features_to_drop_ is None:
            raise ValueError("Filter must be fitted first")

        return self.features_to_drop_

    def __repr__(self) -> str:
        """String representation."""
        return f"VarianceFilter(threshold={self.threshold})"


def find_low_variance_features(
    X: pd.DataFrame,
    threshold: float = 0.01
) -> pd.DataFrame:
    """
    Quick function to find features with low variance.

    Parameters
    ----------
    X : pd.DataFrame
        Input data.

    threshold : float, default=0.01
        Variance threshold.

    Returns
    -------
    low_var_features : pd.DataFrame
        DataFrame with features below the variance threshold.
    """
    filter = VarianceFilter(threshold=threshold)
    filter.fit(X)

    summary = filter.get_variance_summary()
    return summary[summary['below_threshold']].reset_index(drop=True)
