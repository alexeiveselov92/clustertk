"""
Missing value handling for preprocessing pipeline.

This module provides functionality for detecting and imputing missing values
in datasets using various strategies.
"""

from typing import Union, Callable, Dict, Any
import pandas as pd
import numpy as np


class MissingValueHandler:
    """
    Handle missing values in a DataFrame using various strategies.

    Parameters
    ----------
    strategy : str or callable, default='median'
        Strategy for handling missing values:
        - 'median': Fill with median value (robust to outliers)
        - 'mean': Fill with mean value
        - 'mode': Fill with most frequent value
        - 'drop': Drop rows with any missing values
        - 'forward_fill': Forward fill (use previous value)
        - 'backward_fill': Backward fill (use next value)
        - callable: Custom function that takes a DataFrame and returns imputed DataFrame

    fill_value : float or dict, optional
        Specific value(s) to use for filling. If dict, maps column names to fill values.
        Only used when strategy is 'constant'.

    Attributes
    ----------
    fill_values_ : pd.Series
        The computed fill values for each column (for median/mean strategies).

    missing_counts_ : pd.Series
        Count of missing values per column before imputation.

    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> from clustertk.preprocessing import MissingValueHandler
    >>>
    >>> df = pd.DataFrame({'a': [1, 2, np.nan, 4], 'b': [5, np.nan, 7, 8]})
    >>>
    >>> # Median imputation
    >>> handler = MissingValueHandler(strategy='median')
    >>> result = handler.fit_transform(df)
    >>>
    >>> # Custom imputation
    >>> def custom_imputer(df):
    ...     return df.fillna(0)
    >>> handler = MissingValueHandler(strategy=custom_imputer)
    >>> result = handler.fit_transform(df)
    """

    def __init__(
        self,
        strategy: Union[str, Callable] = 'median',
        fill_value: Union[float, Dict[str, float], None] = None
    ):
        valid_strategies = ['median', 'mean', 'mode', 'drop', 'forward_fill', 'backward_fill', 'constant']

        if isinstance(strategy, str) and strategy not in valid_strategies:
            raise ValueError(
                f"Invalid strategy '{strategy}'. "
                f"Must be one of {valid_strategies} or a callable."
            )

        self.strategy = strategy
        self.fill_value = fill_value
        self.fill_values_: Union[pd.Series, None] = None
        self.missing_counts_: Union[pd.Series, None] = None

    def fit(self, X: pd.DataFrame) -> 'MissingValueHandler':
        """
        Compute the fill values based on the strategy.

        Parameters
        ----------
        X : pd.DataFrame
            Input data.

        Returns
        -------
        self : MissingValueHandler
            Fitted handler.
        """
        if not isinstance(X, pd.DataFrame):
            raise TypeError("X must be a pandas DataFrame")

        # Store missing value counts
        self.missing_counts_ = X.isnull().sum()

        # Compute fill values based on strategy
        if self.strategy == 'median':
            self.fill_values_ = X.median()
        elif self.strategy == 'mean':
            self.fill_values_ = X.mean()
        elif self.strategy == 'mode':
            self.fill_values_ = X.mode().iloc[0] if len(X.mode()) > 0 else X.iloc[0]
        elif self.strategy == 'constant':
            if self.fill_value is not None:
                if isinstance(self.fill_value, dict):
                    self.fill_values_ = pd.Series(self.fill_value)
                else:
                    self.fill_values_ = pd.Series(self.fill_value, index=X.columns)
            else:
                raise ValueError("fill_value must be provided when strategy='constant'")
        elif self.strategy in ['drop', 'forward_fill', 'backward_fill']:
            # These strategies don't need fitted values
            self.fill_values_ = None
        elif callable(self.strategy):
            # Custom function doesn't need fitted values
            self.fill_values_ = None

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Apply the missing value imputation to data.

        Parameters
        ----------
        X : pd.DataFrame
            Input data.

        Returns
        -------
        X_transformed : pd.DataFrame
            Data with missing values handled.
        """
        if not isinstance(X, pd.DataFrame):
            raise TypeError("X must be a pandas DataFrame")

        X_transformed = X.copy()

        if self.strategy == 'drop':
            # Drop rows with any missing values
            X_transformed = X_transformed.dropna()

        elif self.strategy == 'forward_fill':
            X_transformed = X_transformed.fillna(method='ffill')

        elif self.strategy == 'backward_fill':
            X_transformed = X_transformed.fillna(method='bfill')

        elif self.strategy in ['median', 'mean', 'mode', 'constant']:
            if self.fill_values_ is None:
                raise ValueError("Handler must be fitted before transform")

            # Fill missing values using computed fill_values
            for col in X_transformed.columns:
                if col in self.fill_values_.index:
                    X_transformed[col] = X_transformed[col].fillna(self.fill_values_[col])

        elif callable(self.strategy):
            # Apply custom function
            X_transformed = self.strategy(X_transformed)

        return X_transformed

    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Fit and transform in one step.

        Parameters
        ----------
        X : pd.DataFrame
            Input data.

        Returns
        -------
        X_transformed : pd.DataFrame
            Data with missing values handled.
        """
        return self.fit(X).transform(X)

    def get_missing_summary(self) -> pd.DataFrame:
        """
        Get a summary of missing values.

        Returns
        -------
        summary : pd.DataFrame
            DataFrame with columns: feature, missing_count, missing_percentage
        """
        if self.missing_counts_ is None:
            raise ValueError("Handler must be fitted first")

        total = self.missing_counts_.sum()
        summary = pd.DataFrame({
            'feature': self.missing_counts_.index,
            'missing_count': self.missing_counts_.values,
            'missing_percentage': (self.missing_counts_.values / len(self.missing_counts_) * 100).round(2)
        })

        return summary[summary['missing_count'] > 0].sort_values(
            'missing_count', ascending=False
        ).reset_index(drop=True)

    def __repr__(self) -> str:
        """String representation."""
        return f"MissingValueHandler(strategy='{self.strategy}')"


def detect_missing_patterns(X: pd.DataFrame, threshold: float = 0.5) -> Dict[str, Any]:
    """
    Analyze missing value patterns in the data.

    Parameters
    ----------
    X : pd.DataFrame
        Input data.

    threshold : float, default=0.5
        Threshold for identifying features with high missing rates.

    Returns
    -------
    patterns : dict
        Dictionary containing:
        - 'total_missing': Total number of missing values
        - 'features_with_missing': List of features with any missing values
        - 'high_missing_features': Features with missing rate > threshold
        - 'missing_by_feature': Series with missing counts per feature
        - 'rows_with_missing': Number of rows with any missing values
    """
    missing_counts = X.isnull().sum()
    missing_rates = missing_counts / len(X)

    patterns = {
        'total_missing': missing_counts.sum(),
        'features_with_missing': missing_counts[missing_counts > 0].index.tolist(),
        'high_missing_features': missing_rates[missing_rates > threshold].index.tolist(),
        'missing_by_feature': missing_counts[missing_counts > 0].sort_values(ascending=False),
        'rows_with_missing': X.isnull().any(axis=1).sum(),
        'rows_with_missing_pct': (X.isnull().any(axis=1).sum() / len(X) * 100).round(2)
    }

    return patterns
