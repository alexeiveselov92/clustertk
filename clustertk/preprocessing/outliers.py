"""
Outlier detection and handling for preprocessing pipeline.

This module provides functionality for detecting and handling outliers
using various statistical methods.
"""

from typing import Union, Dict, Any, Tuple
import pandas as pd
import numpy as np
from scipy import stats


class OutlierHandler:
    """
    Detect and handle outliers in a DataFrame using various methods.

    Parameters
    ----------
    method : str, default='iqr'
        Method for outlier detection:
        - 'iqr': Interquartile Range method (Q1 - 1.5*IQR, Q3 + 1.5*IQR)
        - 'zscore': Z-score method (|z| > threshold)
        - 'modified_zscore': Modified Z-score using median absolute deviation

    action : str, default='clip'
        Action to take for outliers:
        - 'clip': Clip outliers to the boundary values
        - 'remove': Remove rows containing outliers
        - 'nan': Replace outliers with NaN (can be imputed later)
        - None: Only detect, don't modify data

    threshold : float, default=1.5
        Threshold for outlier detection:
        - For 'iqr': multiplier for IQR (typical: 1.5 for outliers, 3 for extreme outliers)
        - For 'zscore': z-score threshold (typical: 3.0)
        - For 'modified_zscore': modified z-score threshold (typical: 3.5)

    Attributes
    ----------
    outlier_bounds_ : pd.DataFrame
        Lower and upper bounds for each feature.

    outlier_mask_ : pd.DataFrame
        Boolean mask indicating outliers in the fitted data.

    outlier_counts_ : pd.Series
        Count of outliers per feature.

    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> from clustertk.preprocessing import OutlierHandler
    >>>
    >>> df = pd.DataFrame({'a': [1, 2, 3, 100], 'b': [5, 6, 7, 8]})
    >>>
    >>> # IQR method with clipping
    >>> handler = OutlierHandler(method='iqr', action='clip', threshold=1.5)
    >>> result = handler.fit_transform(df)
    >>>
    >>> # Z-score method with removal
    >>> handler = OutlierHandler(method='zscore', action='remove', threshold=3.0)
    >>> result = handler.fit_transform(df)
    """

    def __init__(
        self,
        method: str = 'iqr',
        action: Union[str, None] = 'clip',
        threshold: float = 1.5
    ):
        valid_methods = ['iqr', 'zscore', 'modified_zscore']
        valid_actions = ['clip', 'remove', 'nan', None]

        if method not in valid_methods:
            raise ValueError(
                f"Invalid method '{method}'. Must be one of {valid_methods}."
            )

        if action not in valid_actions:
            raise ValueError(
                f"Invalid action '{action}'. Must be one of {valid_actions}."
            )

        self.method = method
        self.action = action
        self.threshold = threshold
        self.outlier_bounds_: Union[pd.DataFrame, None] = None
        self.outlier_mask_: Union[pd.DataFrame, None] = None
        self.outlier_counts_: Union[pd.Series, None] = None

    def fit(self, X: pd.DataFrame) -> 'OutlierHandler':
        """
        Compute outlier bounds based on the detection method.

        Parameters
        ----------
        X : pd.DataFrame
            Input data.

        Returns
        -------
        self : OutlierHandler
            Fitted handler.
        """
        if not isinstance(X, pd.DataFrame):
            raise TypeError("X must be a pandas DataFrame")

        bounds_data = []

        for col in X.columns:
            if not np.issubdtype(X[col].dtype, np.number):
                continue  # Skip non-numeric columns

            if self.method == 'iqr':
                lower, upper = self._compute_iqr_bounds(X[col])
            elif self.method == 'zscore':
                lower, upper = self._compute_zscore_bounds(X[col])
            elif self.method == 'modified_zscore':
                lower, upper = self._compute_modified_zscore_bounds(X[col])

            bounds_data.append({
                'feature': col,
                'lower_bound': lower,
                'upper_bound': upper
            })

        self.outlier_bounds_ = pd.DataFrame(bounds_data)

        # Detect outliers in the fitted data
        self.outlier_mask_ = self._detect_outliers(X)
        self.outlier_counts_ = self.outlier_mask_.sum()

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Apply outlier handling to data.

        Parameters
        ----------
        X : pd.DataFrame
            Input data.

        Returns
        -------
        X_transformed : pd.DataFrame
            Data with outliers handled according to the specified action.
        """
        if not isinstance(X, pd.DataFrame):
            raise TypeError("X must be a pandas DataFrame")

        if self.outlier_bounds_ is None:
            raise ValueError("Handler must be fitted before transform")

        if self.action is None:
            # Only detect, don't modify
            return X.copy()

        X_transformed = X.copy()
        outlier_mask = self._detect_outliers(X_transformed)

        if self.action == 'clip':
            # Clip to bounds
            for _, row in self.outlier_bounds_.iterrows():
                col = row['feature']
                if col in X_transformed.columns:
                    X_transformed[col] = X_transformed[col].clip(
                        lower=row['lower_bound'],
                        upper=row['upper_bound']
                    )

        elif self.action == 'remove':
            # Remove rows with any outliers
            rows_with_outliers = outlier_mask.any(axis=1)
            X_transformed = X_transformed[~rows_with_outliers]

        elif self.action == 'nan':
            # Replace outliers with NaN
            X_transformed[outlier_mask] = np.nan

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
            Data with outliers handled.
        """
        return self.fit(X).transform(X)

    def _compute_iqr_bounds(self, series: pd.Series) -> Tuple[float, float]:
        """Compute outlier bounds using IQR method."""
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - self.threshold * IQR
        upper_bound = Q3 + self.threshold * IQR
        return lower_bound, upper_bound

    def _compute_zscore_bounds(self, series: pd.Series) -> Tuple[float, float]:
        """Compute outlier bounds using Z-score method."""
        mean = series.mean()
        std = series.std()
        lower_bound = mean - self.threshold * std
        upper_bound = mean + self.threshold * std
        return lower_bound, upper_bound

    def _compute_modified_zscore_bounds(self, series: pd.Series) -> Tuple[float, float]:
        """Compute outlier bounds using modified Z-score (MAD) method."""
        median = series.median()
        mad = np.median(np.abs(series - median))

        # Avoid division by zero
        if mad == 0:
            mad = 1e-10

        # Modified z-score = 0.6745 * (x - median) / MAD
        modified_z = 0.6745 * (series - median) / mad

        # Find values where |modified_z| > threshold
        outlier_mask = np.abs(modified_z) > self.threshold

        if outlier_mask.any():
            # Use the min/max of non-outliers as bounds
            non_outliers = series[~outlier_mask]
            lower_bound = non_outliers.min()
            upper_bound = non_outliers.max()
        else:
            lower_bound = series.min()
            upper_bound = series.max()

        return lower_bound, upper_bound

    def _detect_outliers(self, X: pd.DataFrame) -> pd.DataFrame:
        """Detect outliers based on fitted bounds."""
        outlier_mask = pd.DataFrame(False, index=X.index, columns=X.columns)

        for _, row in self.outlier_bounds_.iterrows():
            col = row['feature']
            if col in X.columns:
                outlier_mask[col] = (
                    (X[col] < row['lower_bound']) |
                    (X[col] > row['upper_bound'])
                )

        return outlier_mask

    def get_outlier_summary(self) -> pd.DataFrame:
        """
        Get a summary of detected outliers.

        Returns
        -------
        summary : pd.DataFrame
            DataFrame with outlier statistics per feature.
        """
        if self.outlier_counts_ is None:
            raise ValueError("Handler must be fitted first")

        summary = pd.merge(
            self.outlier_bounds_,
            pd.DataFrame({
                'feature': self.outlier_counts_.index,
                'outlier_count': self.outlier_counts_.values,
                'outlier_percentage': (self.outlier_counts_.values /
                                       len(self.outlier_mask_) * 100).round(2)
            }),
            on='feature'
        )

        return summary[summary['outlier_count'] > 0].sort_values(
            'outlier_count', ascending=False
        ).reset_index(drop=True)

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"OutlierHandler(method='{self.method}', "
            f"action='{self.action}', threshold={self.threshold})"
        )


def detect_outliers_percentage(X: pd.DataFrame, method: str = 'iqr', threshold: float = 1.5) -> Dict[str, float]:
    """
    Quick function to get outlier percentages for each feature.

    Parameters
    ----------
    X : pd.DataFrame
        Input data.

    method : str, default='iqr'
        Outlier detection method.

    threshold : float, default=1.5
        Detection threshold.

    Returns
    -------
    outlier_pcts : dict
        Dictionary mapping feature names to outlier percentages.
    """
    handler = OutlierHandler(method=method, action=None, threshold=threshold)
    handler.fit(X)

    return (handler.outlier_counts_ / len(X) * 100).to_dict()
