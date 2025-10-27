"""
Data transformations for handling skewed distributions.

This module provides transformations for normalizing skewed feature distributions.
"""

from typing import Optional, List, Dict
import pandas as pd
import numpy as np
from scipy import stats
import warnings


class SkewnessTransformer:
    """
    Transform skewed features to approximate normal distributions.

    This class automatically detects skewed features and applies
    log transformations to reduce skewness.

    Parameters
    ----------
    threshold : float, default=2.0
        Absolute skewness threshold for detecting skewed features.
        Features with |skewness| > threshold will be transformed.

    method : str, default='log1p'
        Transformation method to use:
        - 'log1p': np.log1p(x + shift) - handles zeros and negative values
        - 'log': np.log(x + shift) - standard log transform
        - 'sqrt': np.sqrt(x + shift) - square root transform (milder)
        - 'boxcox': Box-Cox transformation (only for positive values)

    auto_shift : bool, default=True
        Whether to automatically shift negative/zero values before transformation.

    Attributes
    ----------
    skewed_features_ : list
        List of feature names that were detected as skewed.

    skewness_before_ : pd.Series
        Skewness values before transformation.

    skewness_after_ : pd.Series
        Skewness values after transformation.

    shift_values_ : dict
        Shift values applied to each feature to handle negative/zero values.

    lambdas_ : dict
        Lambda parameters for Box-Cox transformation (if method='boxcox').

    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> from clustertk.preprocessing import SkewnessTransformer
    >>>
    >>> # Create skewed data
    >>> df = pd.DataFrame({
    ...     'normal': np.random.normal(0, 1, 1000),
    ...     'skewed': np.random.exponential(2, 1000)
    ... })
    >>>
    >>> # Transform skewed features
    >>> transformer = SkewnessTransformer(threshold=2.0, method='log1p')
    >>> transformed = transformer.fit_transform(df)
    >>> print(transformer.skewed_features_)  # ['skewed']
    """

    def __init__(
        self,
        threshold: float = 2.0,
        method: str = 'log1p',
        auto_shift: bool = True
    ):
        valid_methods = ['log1p', 'log', 'sqrt', 'boxcox']

        if method not in valid_methods:
            raise ValueError(
                f"Invalid method '{method}'. Must be one of {valid_methods}."
            )

        self.threshold = threshold
        self.method = method
        self.auto_shift = auto_shift
        self.skewed_features_: Optional[List[str]] = None
        self.skewness_before_: Optional[pd.Series] = None
        self.skewness_after_: Optional[pd.Series] = None
        self.shift_values_: Optional[Dict[str, float]] = None
        self.lambdas_: Optional[Dict[str, float]] = None

    def fit(self, X: pd.DataFrame) -> 'SkewnessTransformer':
        """
        Detect skewed features and compute transformation parameters.

        Parameters
        ----------
        X : pd.DataFrame
            Input data.

        Returns
        -------
        self : SkewnessTransformer
            Fitted transformer.
        """
        if not isinstance(X, pd.DataFrame):
            raise TypeError("X must be a pandas DataFrame")

        # Compute skewness for all numeric columns
        self.skewness_before_ = X.skew()

        # Identify skewed features
        self.skewed_features_ = self.skewness_before_[
            abs(self.skewness_before_) > self.threshold
        ].index.tolist()

        # Compute shift values for skewed features
        self.shift_values_ = {}
        self.lambdas_ = {}

        for col in self.skewed_features_:
            if self.auto_shift:
                min_val = X[col].min()

                if self.method in ['log', 'log1p', 'sqrt']:
                    # Shift negative/zero values
                    if min_val <= 0:
                        self.shift_values_[col] = abs(min_val) + 1
                    else:
                        self.shift_values_[col] = 0

                elif self.method == 'boxcox':
                    # Box-Cox requires strictly positive values
                    if min_val <= 0:
                        self.shift_values_[col] = abs(min_val) + 1
                    else:
                        self.shift_values_[col] = 0

                    # Fit Box-Cox lambda
                    shifted_data = X[col] + self.shift_values_[col]
                    try:
                        _, lambda_param = stats.boxcox(shifted_data)
                        self.lambdas_[col] = lambda_param
                    except:
                        warnings.warn(
                            f"Box-Cox fitting failed for column '{col}', "
                            f"falling back to log1p",
                            UserWarning
                        )
                        self.lambdas_[col] = 0  # lambda=0 is equivalent to log

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Apply transformations to skewed features.

        Parameters
        ----------
        X : pd.DataFrame
            Input data.

        Returns
        -------
        X_transformed : pd.DataFrame
            Transformed data.
        """
        if not isinstance(X, pd.DataFrame):
            raise TypeError("X must be a pandas DataFrame")

        if self.skewed_features_ is None:
            raise ValueError("Transformer must be fitted before transform")

        X_transformed = X.copy()

        for col in self.skewed_features_:
            if col not in X_transformed.columns:
                warnings.warn(
                    f"Feature '{col}' not found in transform data, skipping",
                    UserWarning
                )
                continue

            shift = self.shift_values_.get(col, 0)
            shifted_data = X_transformed[col] + shift

            # Apply transformation
            if self.method == 'log1p':
                X_transformed[col] = np.log1p(shifted_data)

            elif self.method == 'log':
                # Handle potential zeros after shifting
                X_transformed[col] = np.log(shifted_data.clip(lower=1e-10))

            elif self.method == 'sqrt':
                # Handle potential negatives
                X_transformed[col] = np.sqrt(shifted_data.clip(lower=0))

            elif self.method == 'boxcox':
                lambda_param = self.lambdas_.get(col, 0)
                try:
                    if lambda_param == 0:
                        X_transformed[col] = np.log(shifted_data.clip(lower=1e-10))
                    else:
                        X_transformed[col] = (
                            (shifted_data ** lambda_param - 1) / lambda_param
                        )
                except:
                    warnings.warn(
                        f"Box-Cox transform failed for column '{col}', "
                        f"using log1p instead",
                        UserWarning
                    )
                    X_transformed[col] = np.log1p(shifted_data)

            # Handle inf/nan values
            if np.isinf(X_transformed[col]).any():
                warnings.warn(
                    f"Infinite values detected in column '{col}' after transformation, "
                    f"replacing with finite max/min",
                    UserWarning
                )
                finite_values = X_transformed[col][np.isfinite(X_transformed[col])]
                if len(finite_values) > 0:
                    max_finite = finite_values.max()
                    min_finite = finite_values.min()
                    X_transformed[col] = X_transformed[col].replace(
                        [np.inf, -np.inf], [max_finite, min_finite]
                    )

            if X_transformed[col].isna().any():
                warnings.warn(
                    f"NaN values detected in column '{col}' after transformation, "
                    f"filling with median",
                    UserWarning
                )
                X_transformed[col] = X_transformed[col].fillna(
                    X_transformed[col].median()
                )

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
            Transformed data.
        """
        self.fit(X)
        X_transformed = self.transform(X)

        # Compute skewness after transformation
        self.skewness_after_ = X_transformed.skew()

        return X_transformed

    def get_transformation_summary(self) -> pd.DataFrame:
        """
        Get summary of transformations applied.

        Returns
        -------
        summary : pd.DataFrame
            DataFrame with transformation details for each feature.
        """
        if self.skewed_features_ is None:
            raise ValueError("Transformer must be fitted first")

        summary_data = []

        for col in self.skewed_features_:
            row = {
                'feature': col,
                'skewness_before': self.skewness_before_[col],
                'shift_applied': self.shift_values_.get(col, 0),
                'method': self.method
            }

            if self.method == 'boxcox':
                row['lambda'] = self.lambdas_.get(col, None)

            if self.skewness_after_ is not None and col in self.skewness_after_.index:
                row['skewness_after'] = self.skewness_after_[col]
                row['skewness_reduction'] = abs(row['skewness_before']) - abs(row['skewness_after'])

            summary_data.append(row)

        return pd.DataFrame(summary_data)

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"SkewnessTransformer(threshold={self.threshold}, "
            f"method='{self.method}', auto_shift={self.auto_shift})"
        )


def detect_skewness(X: pd.DataFrame, threshold: float = 2.0) -> pd.Series:
    """
    Quick function to detect skewed features.

    Parameters
    ----------
    X : pd.DataFrame
        Input data.

    threshold : float, default=2.0
        Absolute skewness threshold.

    Returns
    -------
    skewed : pd.Series
        Series of skewness values for features exceeding the threshold.
    """
    skewness = X.skew()
    return skewness[abs(skewness) > threshold].sort_values(ascending=False, key=abs)
