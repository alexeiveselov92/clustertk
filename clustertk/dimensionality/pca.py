"""
PCA-based dimensionality reduction.

This module provides Principal Component Analysis with automatic
component selection based on variance explained.
"""

from typing import Optional
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA as SklearnPCA


class PCAReducer:
    """
    Principal Component Analysis with automatic component selection.

    This class performs PCA and automatically selects the number of components
    based on the variance threshold. It ensures a minimum number of components
    for visualization purposes.

    Parameters
    ----------
    variance_threshold : float, default=0.9
        Minimum proportion of total variance to explain.
        The number of components will be selected to explain at least this
        much variance.

    min_components : int, default=2
        Minimum number of components to keep (useful for 2D visualization).

    max_components : int or None, default=None
        Maximum number of components to keep. If None, no maximum limit.

    random_state : int, default=42
        Random state for reproducibility.

    Attributes
    ----------
    n_components_ : int
        Number of components selected.

    pca_ : sklearn.decomposition.PCA
        Fitted PCA object.

    explained_variance_ : np.ndarray
        Variance explained by each component.

    explained_variance_ratio_ : np.ndarray
        Percentage of variance explained by each component.

    cumulative_variance_ : np.ndarray
        Cumulative variance explained.

    loadings_ : pd.DataFrame
        Component loadings (how each original feature contributes to each PC).

    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> from clustertk.dimensionality import PCAReducer
    >>>
    >>> # Create sample data
    >>> df = pd.DataFrame(np.random.rand(100, 10), columns=[f'f{i}' for i in range(10)])
    >>>
    >>> # Reduce dimensions
    >>> reducer = PCAReducer(variance_threshold=0.9, min_components=2)
    >>> reduced_df = reducer.fit_transform(df)
    >>> print(f"Original: {df.shape[1]} features, Reduced: {reduced_df.shape[1]} components")
    >>> print(f"Variance explained: {reducer.cumulative_variance_[reducer.n_components_-1]:.2%}")
    """

    def __init__(
        self,
        variance_threshold: float = 0.9,
        min_components: int = 2,
        max_components: Optional[int] = None,
        random_state: int = 42
    ):
        if not 0 < variance_threshold <= 1:
            raise ValueError("variance_threshold must be between 0 and 1")

        if min_components < 1:
            raise ValueError("min_components must be at least 1")

        if max_components is not None and max_components < min_components:
            raise ValueError("max_components must be >= min_components")

        self.variance_threshold = variance_threshold
        self.min_components = min_components
        self.max_components = max_components
        self.random_state = random_state

        self.n_components_: Optional[int] = None
        self.pca_: Optional[SklearnPCA] = None
        self.explained_variance_: Optional[np.ndarray] = None
        self.explained_variance_ratio_: Optional[np.ndarray] = None
        self.cumulative_variance_: Optional[np.ndarray] = None
        self.loadings_: Optional[pd.DataFrame] = None
        self._feature_names: Optional[list] = None

    def fit(self, X: pd.DataFrame) -> 'PCAReducer':
        """
        Fit PCA and determine optimal number of components.

        Parameters
        ----------
        X : pd.DataFrame
            Input data (should be scaled).

        Returns
        -------
        self : PCAReducer
            Fitted reducer.
        """
        if not isinstance(X, pd.DataFrame):
            raise TypeError("X must be a pandas DataFrame")

        self._feature_names = X.columns.tolist()

        # First, fit PCA with all components to analyze variance
        pca_full = SklearnPCA(random_state=self.random_state)
        pca_full.fit(X)

        # Calculate cumulative variance
        cumsum_variance = np.cumsum(pca_full.explained_variance_ratio_)

        # Determine number of components
        # Find first index where cumsum >= variance_threshold
        n_components = np.argmax(cumsum_variance >= self.variance_threshold) + 1

        # Apply min/max constraints
        n_components = max(self.min_components, n_components)

        if self.max_components is not None:
            n_components = min(self.max_components, n_components)

        # Make sure we don't exceed the maximum possible components
        n_components = min(n_components, min(X.shape))

        self.n_components_ = n_components

        # Fit PCA with selected number of components
        self.pca_ = SklearnPCA(n_components=self.n_components_, random_state=self.random_state)
        self.pca_.fit(X)

        # Store variance information
        self.explained_variance_ = self.pca_.explained_variance_
        self.explained_variance_ratio_ = self.pca_.explained_variance_ratio_
        self.cumulative_variance_ = np.cumsum(self.explained_variance_ratio_)

        # Compute loadings
        self.loadings_ = pd.DataFrame(
            self.pca_.components_.T,
            columns=[f'PC{i+1}' for i in range(self.n_components_)],
            index=self._feature_names
        )

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform data to principal component space.

        Parameters
        ----------
        X : pd.DataFrame
            Input data.

        Returns
        -------
        X_reduced : pd.DataFrame
            Transformed data with principal components.
        """
        if not isinstance(X, pd.DataFrame):
            raise TypeError("X must be a pandas DataFrame")

        if self.pca_ is None:
            raise ValueError("Reducer must be fitted before transform")

        # Transform to PC space
        X_pca = self.pca_.transform(X)

        # Return as DataFrame
        return pd.DataFrame(
            X_pca,
            columns=[f'PC{i+1}' for i in range(self.n_components_)],
            index=X.index
        )

    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Fit and transform in one step.

        Parameters
        ----------
        X : pd.DataFrame
            Input data.

        Returns
        -------
        X_reduced : pd.DataFrame
            Transformed data.
        """
        return self.fit(X).transform(X)

    def inverse_transform(self, X_reduced: pd.DataFrame) -> pd.DataFrame:
        """
        Transform data back to original feature space.

        Parameters
        ----------
        X_reduced : pd.DataFrame
            Data in principal component space.

        Returns
        -------
        X_original : pd.DataFrame
            Data transformed back to original space (approximate reconstruction).
        """
        if self.pca_ is None:
            raise ValueError("Reducer must be fitted before inverse_transform")

        X_reconstructed = self.pca_.inverse_transform(X_reduced)

        return pd.DataFrame(
            X_reconstructed,
            columns=self._feature_names,
            index=X_reduced.index
        )

    def get_loadings(self, n_top: int = 5) -> dict:
        """
        Get top contributing features for each principal component.

        Parameters
        ----------
        n_top : int, default=5
            Number of top features to return for each component.

        Returns
        -------
        top_loadings : dict
            Dictionary mapping PC names to list of (feature, loading) tuples.
        """
        if self.loadings_ is None:
            raise ValueError("Reducer must be fitted first")

        top_loadings = {}

        for pc in self.loadings_.columns:
            # Get absolute loadings and sort
            abs_loadings = self.loadings_[pc].abs().sort_values(ascending=False)
            top_features = abs_loadings.head(n_top)

            # Get actual loadings (with sign)
            top_loadings[pc] = [
                (feature, self.loadings_.loc[feature, pc])
                for feature in top_features.index
            ]

        return top_loadings

    def get_variance_summary(self) -> pd.DataFrame:
        """
        Get summary of variance explained by each component.

        Returns
        -------
        summary : pd.DataFrame
            DataFrame with variance statistics for each component.
        """
        if self.pca_ is None:
            raise ValueError("Reducer must be fitted first")

        summary = pd.DataFrame({
            'component': [f'PC{i+1}' for i in range(self.n_components_)],
            'variance': self.explained_variance_,
            'variance_ratio': self.explained_variance_ratio_,
            'cumulative_variance': self.cumulative_variance_
        })

        return summary

    def get_reconstruction_error(self, X: pd.DataFrame) -> float:
        """
        Calculate reconstruction error after dimensionality reduction.

        Parameters
        ----------
        X : pd.DataFrame
            Original data.

        Returns
        -------
        error : float
            Mean squared error of reconstruction.
        """
        if self.pca_ is None:
            raise ValueError("Reducer must be fitted first")

        X_reduced = self.transform(X)
        X_reconstructed = self.inverse_transform(X_reduced)

        mse = ((X - X_reconstructed) ** 2).mean().mean()
        return mse

    def __repr__(self) -> str:
        """String representation."""
        if self.n_components_ is not None:
            return (
                f"PCAReducer(variance_threshold={self.variance_threshold}, "
                f"n_components={self.n_components_}, "
                f"variance_explained={self.cumulative_variance_[self.n_components_-1]:.3f})"
            )
        return f"PCAReducer(variance_threshold={self.variance_threshold})"


def quick_pca(
    X: pd.DataFrame,
    n_components: int = 2,
    random_state: int = 42
) -> pd.DataFrame:
    """
    Quick PCA transformation with fixed number of components.

    Parameters
    ----------
    X : pd.DataFrame
        Input data.

    n_components : int, default=2
        Number of components.

    random_state : int, default=42
        Random state.

    Returns
    -------
    X_reduced : pd.DataFrame
        Transformed data.
    """
    pca = SklearnPCA(n_components=n_components, random_state=random_state)
    X_pca = pca.fit_transform(X)

    return pd.DataFrame(
        X_pca,
        columns=[f'PC{i+1}' for i in range(n_components)],
        index=X.index
    )
