"""
Scaling and normalization for preprocessing pipeline.

This module provides automatic scaler selection and application
based on data characteristics.
"""

from typing import Union, Optional
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
import warnings


class ScalerSelector:
    """
    Automatically select and apply the appropriate scaler based on data characteristics.

    The selector chooses between StandardScaler and RobustScaler based on
    the presence of outliers in the data. If outliers are detected (>5% by IQR method),
    RobustScaler is used; otherwise, StandardScaler is used.

    Parameters
    ----------
    scaler_type : str, default='auto'
        Type of scaler to use:
        - 'auto': Automatically select based on outlier analysis
        - 'standard': StandardScaler (z-score normalization)
        - 'robust': RobustScaler (uses median and IQR, robust to outliers)
        - 'minmax': MinMaxScaler (scales to [0, 1] range)

    outlier_threshold : float, default=0.05
        Percentage threshold for considering data to have outliers (only used with 'auto').
        If more than this percentage of values are outliers, RobustScaler is used.

    Attributes
    ----------
    scaler_ : object
        The fitted sklearn scaler object (StandardScaler, RobustScaler, or MinMaxScaler).

    selected_scaler_type_ : str
        The type of scaler that was selected (when using 'auto').

    outlier_percentages_ : pd.Series
        Percentage of outliers per feature (only computed with 'auto').

    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> from clustertk.preprocessing import ScalerSelector
    >>>
    >>> df = pd.DataFrame({'a': [1, 2, 3, 4, 100], 'b': [5, 6, 7, 8, 9]})
    >>>
    >>> # Auto-select scaler
    >>> selector = ScalerSelector(scaler_type='auto')
    >>> scaled = selector.fit_transform(df)
    >>> print(selector.selected_scaler_type_)  # Will be 'robust' due to outlier in 'a'
    >>>
    >>> # Force specific scaler
    >>> selector = ScalerSelector(scaler_type='standard')
    >>> scaled = selector.fit_transform(df)
    """

    def __init__(
        self,
        scaler_type: str = 'auto',
        outlier_threshold: float = 0.05
    ):
        valid_types = ['auto', 'standard', 'robust', 'minmax']

        if scaler_type not in valid_types:
            raise ValueError(
                f"Invalid scaler_type '{scaler_type}'. Must be one of {valid_types}."
            )

        self.scaler_type = scaler_type
        self.outlier_threshold = outlier_threshold
        self.scaler_: Optional[object] = None
        self.selected_scaler_type_: Optional[str] = None
        self.outlier_percentages_: Optional[pd.Series] = None

    def fit(self, X: pd.DataFrame) -> 'ScalerSelector':
        """
        Fit the scaler to the data.

        Parameters
        ----------
        X : pd.DataFrame
            Input data.

        Returns
        -------
        self : ScalerSelector
            Fitted selector.
        """
        if not isinstance(X, pd.DataFrame):
            raise TypeError("X must be a pandas DataFrame")

        # Select scaler type
        if self.scaler_type == 'auto':
            self.selected_scaler_type_ = self._select_scaler(X)
        else:
            self.selected_scaler_type_ = self.scaler_type

        # Initialize and fit the appropriate scaler
        if self.selected_scaler_type_ == 'standard':
            self.scaler_ = StandardScaler()
        elif self.selected_scaler_type_ == 'robust':
            self.scaler_ = RobustScaler()
        elif self.selected_scaler_type_ == 'minmax':
            self.scaler_ = MinMaxScaler()

        self.scaler_.fit(X)

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform the data using the fitted scaler.

        Parameters
        ----------
        X : pd.DataFrame
            Input data.

        Returns
        -------
        X_scaled : pd.DataFrame
            Scaled data with same column names and index.
        """
        if not isinstance(X, pd.DataFrame):
            raise TypeError("X must be a pandas DataFrame")

        if self.scaler_ is None:
            raise ValueError("Selector must be fitted before transform")

        X_scaled = self.scaler_.transform(X)

        # Return as DataFrame with original column names and index
        return pd.DataFrame(X_scaled, columns=X.columns, index=X.index)

    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Fit and transform in one step.

        Parameters
        ----------
        X : pd.DataFrame
            Input data.

        Returns
        -------
        X_scaled : pd.DataFrame
            Scaled data.
        """
        return self.fit(X).transform(X)

    def inverse_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Inverse transform scaled data back to original scale.

        Parameters
        ----------
        X : pd.DataFrame
            Scaled data.

        Returns
        -------
        X_original : pd.DataFrame
            Data in original scale.
        """
        if self.scaler_ is None:
            raise ValueError("Selector must be fitted before inverse_transform")

        X_original = self.scaler_.inverse_transform(X)
        return pd.DataFrame(X_original, columns=X.columns, index=X.index)

    def _select_scaler(self, X: pd.DataFrame) -> str:
        """
        Automatically select scaler based on outlier analysis.

        Parameters
        ----------
        X : pd.DataFrame
            Input data.

        Returns
        -------
        scaler_type : str
            Selected scaler type ('standard' or 'robust').
        """
        from clustertk.preprocessing.outliers import detect_outliers_percentage

        # Detect outlier percentages
        outlier_pcts = detect_outliers_percentage(X, method='iqr', threshold=1.5)
        self.outlier_percentages_ = pd.Series(outlier_pcts)

        # Check if any feature has significant outliers
        max_outlier_pct = max(outlier_pcts.values()) if outlier_pcts else 0

        if max_outlier_pct > self.outlier_threshold * 100:
            return 'robust'
        else:
            return 'standard'

    def get_scaling_summary(self) -> dict:
        """
        Get summary of scaling parameters.

        Returns
        -------
        summary : dict
            Dictionary containing scaling information.
        """
        if self.scaler_ is None:
            raise ValueError("Selector must be fitted first")

        summary = {
            'scaler_type': self.selected_scaler_type_,
            'requested_type': self.scaler_type,
        }

        # Add scaler-specific parameters
        if self.selected_scaler_type_ == 'standard':
            summary['mean'] = self.scaler_.mean_.tolist()
            summary['std'] = self.scaler_.scale_.tolist()

        elif self.selected_scaler_type_ == 'robust':
            summary['median'] = self.scaler_.center_.tolist()
            summary['iqr'] = self.scaler_.scale_.tolist()

        elif self.selected_scaler_type_ == 'minmax':
            summary['min'] = self.scaler_.data_min_.tolist()
            summary['max'] = self.scaler_.data_max_.tolist()
            summary['feature_range'] = self.scaler_.feature_range

        if self.outlier_percentages_ is not None:
            summary['outlier_percentages'] = self.outlier_percentages_.to_dict()

        return summary

    def __repr__(self) -> str:
        """String representation."""
        if self.selected_scaler_type_ is not None:
            return (
                f"ScalerSelector(scaler_type='{self.scaler_type}', "
                f"selected='{self.selected_scaler_type_}')"
            )
        return f"ScalerSelector(scaler_type='{self.scaler_type}')"


def compare_scalers(X: pd.DataFrame) -> pd.DataFrame:
    """
    Compare the effect of different scalers on the data.

    Parameters
    ----------
    X : pd.DataFrame
        Input data.

    Returns
    -------
    comparison : pd.DataFrame
        DataFrame showing statistics for each scaler type.
    """
    results = []

    for scaler_type in ['standard', 'robust', 'minmax']:
        selector = ScalerSelector(scaler_type=scaler_type)
        X_scaled = selector.fit_transform(X)

        results.append({
            'scaler': scaler_type,
            'mean': X_scaled.mean().mean(),
            'std': X_scaled.std().mean(),
            'min': X_scaled.min().min(),
            'max': X_scaled.max().max(),
            'median': X_scaled.median().median()
        })

    return pd.DataFrame(results)
