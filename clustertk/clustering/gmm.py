"""
Gaussian Mixture Model clustering implementation.

GMM is a probabilistic model that assumes data points are generated from a
mixture of Gaussian distributions. It provides soft clustering (probability
of belonging to each cluster).
"""

from typing import Optional
import pandas as pd
import numpy as np
from sklearn.mixture import GaussianMixture

from clustertk.clustering.base import BaseClusterer


class GMMClustering(BaseClusterer):
    """
    Gaussian Mixture Model clustering wrapper.

    GMM represents the data as a mixture of multiple Gaussian distributions.
    Unlike K-Means, it provides soft clustering where each point has a probability
    of belonging to each cluster.

    Parameters
    ----------
    n_clusters : int, default=3
        Number of mixture components (clusters).

    random_state : int, default=42
        Random state for reproducibility.

    covariance_type : str, default='full'
        Type of covariance matrix:
        - 'full': Each component has its own general covariance matrix
        - 'tied': All components share the same covariance matrix
        - 'diag': Diagonal covariance matrices
        - 'spherical': Spherical covariance matrices

    n_init : int, default=10
        Number of initializations to perform. Best result is kept.

    max_iter : int, default=100
        Maximum number of EM iterations.

    **kwargs : dict
        Additional parameters to pass to sklearn's GaussianMixture.

    Attributes
    ----------
    bic_ : float
        Bayesian Information Criterion for the fitted model.

    aic_ : float
        Akaike Information Criterion for the fitted model.

    converged_ : bool
        Whether the EM algorithm converged.

    n_iter_ : int
        Number of EM iterations performed.

    probabilities_ : np.ndarray
        Probability of each sample belonging to each cluster.

    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> from clustertk.clustering import GMMClustering
    >>>
    >>> # Create sample data
    >>> df = pd.DataFrame(np.random.rand(100, 5))
    >>>
    >>> # Fit GMM
    >>> gmm = GMMClustering(n_clusters=3)
    >>> labels = gmm.fit_predict(df)
    >>> print(f"BIC: {gmm.bic_:.2f}, AIC: {gmm.aic_:.2f}")
    >>>
    >>> # Get probabilities
    >>> probs = gmm.predict_proba(df)
    """

    def __init__(
        self,
        n_clusters: int = 3,
        random_state: int = 42,
        covariance_type: str = 'full',
        n_init: int = 10,
        max_iter: int = 100,
        **kwargs
    ):
        super().__init__(n_clusters=n_clusters, random_state=random_state)
        self.covariance_type = covariance_type
        self.n_init = n_init
        self.max_iter = max_iter
        self.kwargs = kwargs
        self.bic_: Optional[float] = None
        self.aic_: Optional[float] = None
        self.converged_: Optional[bool] = None
        self.n_iter_: Optional[int] = None
        self.probabilities_: Optional[np.ndarray] = None

    def fit(self, X: pd.DataFrame) -> 'GMMClustering':
        """
        Fit Gaussian Mixture Model.

        Parameters
        ----------
        X : pd.DataFrame
            Input data.

        Returns
        -------
        self : GMMClustering
            Fitted clusterer.
        """
        if not isinstance(X, pd.DataFrame):
            raise TypeError("X must be a pandas DataFrame")

        # Initialize sklearn GaussianMixture
        self.model_ = GaussianMixture(
            n_components=self.n_clusters,
            random_state=self.random_state,
            covariance_type=self.covariance_type,
            n_init=self.n_init,
            max_iter=self.max_iter,
            **self.kwargs
        )

        # Fit the model
        self.model_.fit(X)

        # Store results
        self.labels_ = self.model_.predict(X)
        self.probabilities_ = self.model_.predict_proba(X)
        self.n_clusters_ = self.n_clusters
        self.bic_ = self.model_.bic(X)
        self.aic_ = self.model_.aic(X)
        self.converged_ = self.model_.converged_
        self.n_iter_ = self.model_.n_iter_

        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict cluster labels for new data.

        Parameters
        ----------
        X : pd.DataFrame
            Input data.

        Returns
        -------
        labels : np.ndarray
            Predicted cluster labels (hard assignment).
        """
        if self.model_ is None:
            raise ValueError("Model must be fitted before predict")

        if not isinstance(X, pd.DataFrame):
            raise TypeError("X must be a pandas DataFrame")

        return self.model_.predict(X)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict posterior probabilities for each cluster.

        Parameters
        ----------
        X : pd.DataFrame
            Input data.

        Returns
        -------
        probabilities : np.ndarray
            Probability of each sample belonging to each cluster.
            Shape: (n_samples, n_clusters)
        """
        if self.model_ is None:
            raise ValueError("Model must be fitted before predict_proba")

        return self.model_.predict_proba(X)

    def get_probabilities_df(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Get cluster probabilities as a DataFrame.

        Parameters
        ----------
        X : pd.DataFrame
            Input data.

        Returns
        -------
        probabilities : pd.DataFrame
            DataFrame with probability for each cluster.
        """
        probs = self.predict_proba(X)

        return pd.DataFrame(
            probs,
            columns=[f'prob_cluster_{i}' for i in range(self.n_clusters)],
            index=X.index
        )

    def get_cluster_centers(self) -> np.ndarray:
        """
        Get GMM cluster centers (means of Gaussian components).

        Returns
        -------
        centers : np.ndarray
            Mean of each Gaussian component.
        """
        if self.model_ is None:
            raise ValueError("Model must be fitted first")

        return self.model_.means_

    def score(self, X: pd.DataFrame) -> float:
        """
        Compute the log-likelihood of data under the model.

        Parameters
        ----------
        X : pd.DataFrame
            Input data.

        Returns
        -------
        score : float
            Log-likelihood of data.
        """
        if self.model_ is None:
            raise ValueError("Model must be fitted first")

        return self.model_.score(X)

    def get_model_selection_criteria(self, X: pd.DataFrame) -> dict:
        """
        Get model selection criteria (BIC, AIC, log-likelihood).

        Parameters
        ----------
        X : pd.DataFrame
            Input data.

        Returns
        -------
        criteria : dict
            Dictionary with BIC, AIC, and log-likelihood.
        """
        return {
            'bic': self.bic_,
            'aic': self.aic_,
            'log_likelihood': self.score(X),
            'n_parameters': self.model_._n_parameters()
        }

    def __repr__(self) -> str:
        """String representation."""
        if self.bic_ is not None:
            return (
                f"GMMClustering(n_clusters={self.n_clusters}, "
                f"bic={self.bic_:.2f}, aic={self.aic_:.2f})"
            )
        return f"GMMClustering(n_clusters={self.n_clusters})"
