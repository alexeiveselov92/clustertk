"""
Multivariate outlier detection.

This module provides detection of outliers in the full feature space, not just
per-feature. Multivariate outliers are points that are far from other points
when considering all features together, even if they look normal per-feature.
"""

from typing import Optional, Literal
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.covariance import EllipticEnvelope


class MultivariateOutlierDetector:
    """
    Detect outliers in multivariate space.

    Unlike univariate outlier detection (per-feature), this detector identifies
    points that are outliers when considering all features together. This is
    critical for clustering because multivariate outliers can create small,
    meaningless clusters even after univariate outlier handling and scaling.

    Parameters
    ----------
    method : {'auto', 'isolation_forest', 'lof', 'elliptic_envelope'}, default='auto'
        Outlier detection method:
        - 'auto': Automatically select best method based on data characteristics
        - 'isolation_forest': IsolationForest (best for high-dimensional data)
        - 'lof': Local Outlier Factor (best for low-dimensional varying density)
        - 'elliptic_envelope': Robust covariance (best for Gaussian data)

    contamination : float, default=0.05
        Expected proportion of outliers in the dataset (0.0 to 0.5).
        Typical values: 0.01-0.05 for clean data, 0.05-0.10 for noisy data.

    action : {'remove', 'flag', None}, default='remove'
        How to handle detected outliers:
        - 'remove': Remove outlier rows from data
        - 'flag': Add '_is_outlier' column (True for outliers)
        - None: Only detect, don't modify data (use get_outlier_mask())

    random_state : int, optional
        Random seed for reproducibility.

    **kwargs : dict
        Additional parameters passed to the underlying detector.

    Attributes
    ----------
    outlier_mask_ : np.ndarray
        Boolean mask where True indicates outliers (set after fit).

    n_outliers_ : int
        Number of outliers detected (set after fit).

    outlier_ratio_ : float
        Proportion of outliers detected (set after fit).

    method_used_ : str
        Actual method used (resolved from 'auto' if applicable).

    Notes
    -----
    **When to use multivariate vs univariate outlier detection:**

    - Univariate (Winsorize, IQR): Handles extreme values per feature
    - Multivariate (this class): Handles points far in combined feature space

    **Example problem solved:**
    ```
    Feature1: [100, 110, 120, ..., 150]  # No outliers
    Feature2: [50, 55, 60, ..., 80]      # No outliers

    But point (150, 80) might be an outlier in 2D space if all other points
    cluster around (110, 60). Univariate methods miss this!
    ```

    **Execution order in Pipeline:**
    1. Missing values
    2. Log transform (optional)
    3. Univariate outliers (Winsorize) → per-feature extremes
    4. Scaling
    5. **Multivariate outliers (this class)** → full-space outliers
    6. PCA
    7. Clustering

    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> from clustertk.preprocessing import MultivariateOutlierDetector
    >>>
    >>> # Create data with multivariate outliers
    >>> np.random.seed(42)
    >>> normal = np.random.multivariate_normal([0, 0], [[1, 0], [0, 1]], 100)
    >>> outliers = np.array([[10, 10], [-10, 10]])
    >>> data = pd.DataFrame(np.vstack([normal, outliers]), columns=['x', 'y'])
    >>>
    >>> # Detect and remove multivariate outliers
    >>> detector = MultivariateOutlierDetector(method='auto', contamination=0.05)
    >>> data_clean = detector.fit_transform(data)
    >>> print(f"Removed {detector.n_outliers_} outliers")
    >>>
    >>> # Just flag outliers without removing
    >>> detector = MultivariateOutlierDetector(action='flag')
    >>> data_flagged = detector.fit_transform(data)
    >>> print(data_flagged['_is_outlier'].value_counts())
    """

    def __init__(
        self,
        method: Literal['auto', 'isolation_forest', 'lof', 'elliptic_envelope'] = 'auto',
        contamination: float = 0.05,
        action: Optional[Literal['remove', 'flag']] = 'remove',
        random_state: Optional[int] = None,
        **kwargs
    ):
        self.method = method
        self.contamination = contamination
        self.action = action
        self.random_state = random_state
        self.kwargs = kwargs

        # Set after fitting
        self.outlier_mask_: Optional[np.ndarray] = None
        self.n_outliers_: Optional[int] = None
        self.outlier_ratio_: Optional[float] = None
        self.method_used_: Optional[str] = None
        self._detector = None

    def _select_method(self, X: pd.DataFrame) -> str:
        """
        Automatically select best method based on data characteristics.

        Parameters
        ----------
        X : pd.DataFrame
            Input data.

        Returns
        -------
        method : str
            Selected method name.
        """
        n_samples, n_features = X.shape

        # For small datasets, use LOF (better for small samples)
        if n_samples < 100:
            return 'lof'

        # For low-dimensional data, use LOF (handles varying density better)
        if n_features < 5:
            return 'lof'

        # For high-dimensional data, use IsolationForest (faster, scales better)
        if n_features >= 10:
            return 'isolation_forest'

        # For medium dimensions, check if data is Gaussian
        # Use Shapiro-Wilk test approximation (check skewness and kurtosis)
        try:
            from scipy import stats
            # Check if most features are normally distributed
            normal_features = 0
            for col in X.columns[:min(5, len(X.columns))]:  # Sample first 5 features
                _, p_value = stats.shapiro(X[col].sample(min(100, len(X))))
                if p_value > 0.05:
                    normal_features += 1

            # If >60% features are normal, use EllipticEnvelope
            if normal_features / min(5, len(X.columns)) > 0.6:
                return 'elliptic_envelope'
        except ImportError:
            pass  # scipy not available, skip Gaussian check

        # Default to IsolationForest (most robust)
        return 'isolation_forest'

    def _create_detector(self, X: pd.DataFrame):
        """
        Create the appropriate detector based on method.

        Parameters
        ----------
        X : pd.DataFrame
            Input data.

        Returns
        -------
        detector : object
            Configured detector object.
        """
        # Resolve 'auto' to actual method
        if self.method == 'auto':
            self.method_used_ = self._select_method(X)
        else:
            self.method_used_ = self.method

        # Create detector
        if self.method_used_ == 'isolation_forest':
            return IsolationForest(
                contamination=self.contamination,
                random_state=self.random_state,
                **self.kwargs
            )
        elif self.method_used_ == 'lof':
            return LocalOutlierFactor(
                contamination=self.contamination,
                **self.kwargs
            )
        elif self.method_used_ == 'elliptic_envelope':
            return EllipticEnvelope(
                contamination=self.contamination,
                random_state=self.random_state,
                **self.kwargs
            )
        else:
            raise ValueError(
                f"Unknown method: {self.method_used_}. "
                f"Choose from: 'auto', 'isolation_forest', 'lof', 'elliptic_envelope'"
            )

    def fit(self, X: pd.DataFrame) -> 'MultivariateOutlierDetector':
        """
        Fit the outlier detector.

        Parameters
        ----------
        X : pd.DataFrame
            Input data (should be scaled for best results).

        Returns
        -------
        self : MultivariateOutlierDetector
            Fitted detector.
        """
        if not isinstance(X, pd.DataFrame):
            raise TypeError("X must be a pandas DataFrame")

        # Validate contamination
        if not 0.0 < self.contamination <= 0.5:
            raise ValueError("contamination must be in (0.0, 0.5]")

        # Create and fit detector
        self._detector = self._create_detector(X)
        predictions = self._detector.fit_predict(X)

        # Convert predictions to outlier mask
        # sklearn detectors return: 1 for inliers, -1 for outliers
        self.outlier_mask_ = predictions == -1

        # Store statistics
        self.n_outliers_ = int(np.sum(self.outlier_mask_))
        self.outlier_ratio_ = float(self.n_outliers_ / len(X))

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Apply outlier detection to data.

        Parameters
        ----------
        X : pd.DataFrame
            Input data.

        Returns
        -------
        X_transformed : pd.DataFrame
            Transformed data based on action parameter:
            - 'remove': Data with outliers removed
            - 'flag': Data with '_is_outlier' column added
            - None: Original data unchanged
        """
        if self.outlier_mask_ is None:
            raise ValueError("Must call fit() before transform()")

        if not isinstance(X, pd.DataFrame):
            raise TypeError("X must be a pandas DataFrame")

        if len(X) != len(self.outlier_mask_):
            raise ValueError(
                f"X has {len(X)} rows, but detector was fitted on "
                f"{len(self.outlier_mask_)} rows"
            )

        # Apply action
        if self.action == 'remove':
            return X[~self.outlier_mask_].reset_index(drop=True)
        elif self.action == 'flag':
            X_copy = X.copy()
            X_copy['_is_outlier'] = self.outlier_mask_
            return X_copy
        else:
            # action=None, return unchanged
            return X.copy()

    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Fit detector and transform data in one step.

        Parameters
        ----------
        X : pd.DataFrame
            Input data.

        Returns
        -------
        X_transformed : pd.DataFrame
            Transformed data.
        """
        return self.fit(X).transform(X)

    def get_outlier_mask(self) -> np.ndarray:
        """
        Get boolean mask of outliers.

        Returns
        -------
        outlier_mask : np.ndarray
            Boolean array where True indicates outliers.
        """
        if self.outlier_mask_ is None:
            raise ValueError("Must call fit() before get_outlier_mask()")

        return self.outlier_mask_

    def get_outlier_indices(self) -> np.ndarray:
        """
        Get indices of detected outliers.

        Returns
        -------
        outlier_indices : np.ndarray
            Array of integer indices where outliers were detected.
        """
        if self.outlier_mask_ is None:
            raise ValueError("Must call fit() before get_outlier_indices()")

        return np.where(self.outlier_mask_)[0]

    def __repr__(self) -> str:
        """String representation."""
        if self.n_outliers_ is not None:
            return (
                f"MultivariateOutlierDetector("
                f"method={self.method_used_}, "
                f"contamination={self.contamination}, "
                f"n_outliers={self.n_outliers_}, "
                f"outlier_ratio={self.outlier_ratio_:.3f})"
            )
        return (
            f"MultivariateOutlierDetector("
            f"method={self.method}, "
            f"contamination={self.contamination})"
        )
