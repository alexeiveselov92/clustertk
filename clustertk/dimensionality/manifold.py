"""
Manifold learning methods for visualization.

This module provides t-SNE and UMAP for reducing high-dimensional data
to 2D or 3D for visualization purposes.

IMPORTANT: These methods are for VISUALIZATION ONLY, not for clustering!
"""

from typing import Optional, Dict, Any
import pandas as pd
import numpy as np
from sklearn.manifold import TSNE
import warnings


class ManifoldReducer:
    """
    Reduce dimensionality using manifold learning methods (t-SNE or UMAP).

    This class is designed for VISUALIZATION purposes only. The resulting
    low-dimensional representation should NOT be used for clustering, as
    manifold methods can distort distances and relationships in ways that
    make clustering unreliable.

    Parameters
    ----------
    method : str, default='tsne'
        Manifold learning method:
        - 'tsne': t-distributed Stochastic Neighbor Embedding
        - 'umap': Uniform Manifold Approximation and Projection (requires umap-learn)

    n_components : int, default=2
        Number of dimensions for the embedding (typically 2 for visualization).

    random_state : int, default=42
        Random state for reproducibility.

    **kwargs : dict
        Additional parameters specific to the chosen method:

        For t-SNE:
        - perplexity : float, default=30.0
        - learning_rate : float, default=200.0
        - max_iter : int, default=1000
        - metric : str, default='euclidean'

        For UMAP:
        - n_neighbors : int, default=15
        - min_dist : float, default=0.1
        - metric : str, default='euclidean'

    Attributes
    ----------
    reducer_ : object
        Fitted reducer object (TSNE or UMAP).

    embedding_ : np.ndarray
        The learned embedding (only available after fit).

    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> from clustertk.dimensionality import ManifoldReducer
    >>>
    >>> # Create high-dimensional data
    >>> df = pd.DataFrame(np.random.rand(100, 50))
    >>>
    >>> # Reduce to 2D with t-SNE
    >>> reducer = ManifoldReducer(method='tsne', n_components=2)
    >>> embedding_2d = reducer.fit_transform(df)
    >>>
    >>> # Reduce to 2D with UMAP (if installed)
    >>> reducer = ManifoldReducer(method='umap', n_components=2)
    >>> embedding_2d = reducer.fit_transform(df)
    """

    def __init__(
        self,
        method: str = 'tsne',
        n_components: int = 2,
        random_state: int = 42,
        **kwargs
    ):
        valid_methods = ['tsne', 'umap']

        if method not in valid_methods:
            raise ValueError(
                f"Invalid method '{method}'. Must be one of {valid_methods}."
            )

        if method == 'umap':
            try:
                import umap  # noqa: F401
            except ImportError:
                raise ImportError(
                    "UMAP requires the umap-learn package. "
                    "Install with: pip install clustertk[extras]"
                )

        self.method = method
        self.n_components = n_components
        self.random_state = random_state
        self.kwargs = kwargs
        self.reducer_: Optional[object] = None
        self.embedding_: Optional[np.ndarray] = None

    def fit(self, X: pd.DataFrame) -> 'ManifoldReducer':
        """
        Fit the manifold learner (not typically needed, use fit_transform).

        Parameters
        ----------
        X : pd.DataFrame
            Input data.

        Returns
        -------
        self : ManifoldReducer
            Fitted reducer.
        """
        warnings.warn(
            "Manifold methods typically don't support separate fit/transform. "
            "Use fit_transform instead.",
            UserWarning
        )
        return self.fit_transform(X)

    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Fit the manifold learner and transform the data.

        Parameters
        ----------
        X : pd.DataFrame
            Input data.

        Returns
        -------
        X_embedded : pd.DataFrame
            Low-dimensional embedding.
        """
        if not isinstance(X, pd.DataFrame):
            raise TypeError("X must be a pandas DataFrame")

        # Adjust parameters based on data size
        n_samples = len(X)

        if self.method == 'tsne':
            # Adjust perplexity if needed
            perplexity = self.kwargs.get('perplexity', 30.0)
            # Perplexity must be less than n_samples
            perplexity = min(perplexity, n_samples // 4) if n_samples < 120 else perplexity

            params = {
                'n_components': self.n_components,
                'random_state': self.random_state,
                'perplexity': perplexity,
                'learning_rate': self.kwargs.get('learning_rate', 200.0),
                'max_iter': self.kwargs.get('max_iter', 1000),
                'metric': self.kwargs.get('metric', 'euclidean'),
                'init': 'pca',  # Use PCA initialization for better results
                'verbose': 0
            }

            self.reducer_ = TSNE(**params)
            self.embedding_ = self.reducer_.fit_transform(X)

        elif self.method == 'umap':
            try:
                from umap import UMAP
            except ImportError:
                raise ImportError(
                    "UMAP requires the umap-learn package. "
                    "Install with: pip install clustertk[extras]"
                )

            params = {
                'n_components': self.n_components,
                'random_state': self.random_state,
                'n_neighbors': self.kwargs.get('n_neighbors', 15),
                'min_dist': self.kwargs.get('min_dist', 0.1),
                'metric': self.kwargs.get('metric', 'euclidean'),
                'verbose': False
            }

            self.reducer_ = UMAP(**params)
            self.embedding_ = self.reducer_.fit_transform(X)

        # Return as DataFrame
        if self.n_components == 2:
            columns = [f'{self.method.upper()}1', f'{self.method.upper()}2']
        elif self.n_components == 3:
            columns = [f'{self.method.upper()}1', f'{self.method.upper()}2', f'{self.method.upper()}3']
        else:
            columns = [f'{self.method.upper()}{i+1}' for i in range(self.n_components)]

        return pd.DataFrame(
            self.embedding_,
            columns=columns,
            index=X.index
        )

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform new data (only supported by UMAP, not t-SNE).

        Parameters
        ----------
        X : pd.DataFrame
            Input data.

        Returns
        -------
        X_embedded : pd.DataFrame
            Low-dimensional embedding.
        """
        if self.reducer_ is None:
            raise ValueError("Reducer must be fitted before transform")

        if self.method == 'tsne':
            raise ValueError(
                "t-SNE does not support transforming new data. "
                "Use fit_transform on the combined dataset."
            )

        # UMAP supports transform
        embedding = self.reducer_.transform(X)

        if self.n_components == 2:
            columns = [f'{self.method.upper()}1', f'{self.method.upper()}2']
        elif self.n_components == 3:
            columns = [f'{self.method.upper()}1', f'{self.method.upper()}2', f'{self.method.upper()}3']
        else:
            columns = [f'{self.method.upper()}{i+1}' for i in range(self.n_components)]

        return pd.DataFrame(embedding, columns=columns, index=X.index)

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"ManifoldReducer(method='{self.method}', "
            f"n_components={self.n_components})"
        )


def quick_tsne(
    X: pd.DataFrame,
    n_components: int = 2,
    perplexity: float = 30.0,
    random_state: int = 42
) -> pd.DataFrame:
    """
    Quick t-SNE transformation.

    Parameters
    ----------
    X : pd.DataFrame
        Input data.

    n_components : int, default=2
        Number of components.

    perplexity : float, default=30.0
        t-SNE perplexity parameter.

    random_state : int, default=42
        Random state.

    Returns
    -------
    X_embedded : pd.DataFrame
        t-SNE embedding.
    """
    reducer = ManifoldReducer(
        method='tsne',
        n_components=n_components,
        random_state=random_state,
        perplexity=perplexity
    )
    return reducer.fit_transform(X)


def quick_umap(
    X: pd.DataFrame,
    n_components: int = 2,
    n_neighbors: int = 15,
    random_state: int = 42
) -> pd.DataFrame:
    """
    Quick UMAP transformation.

    Parameters
    ----------
    X : pd.DataFrame
        Input data.

    n_components : int, default=2
        Number of components.

    n_neighbors : int, default=15
        UMAP n_neighbors parameter.

    random_state : int, default=42
        Random state.

    Returns
    -------
    X_embedded : pd.DataFrame
        UMAP embedding.
    """
    reducer = ManifoldReducer(
        method='umap',
        n_components=n_components,
        random_state=random_state,
        n_neighbors=n_neighbors
    )
    return reducer.fit_transform(X)
