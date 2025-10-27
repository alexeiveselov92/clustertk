"""
Base class for clustering algorithms.

This module defines the interface that all clustering algorithms must implement.
"""

from abc import ABC, abstractmethod
from typing import Optional
import pandas as pd
import numpy as np


class BaseClusterer(ABC):
    """
    Abstract base class for all clustering algorithms.

    All clustering algorithms in clustertk should inherit from this class
    and implement the required methods.

    Parameters
    ----------
    n_clusters : int
        Number of clusters to form.

    random_state : int, default=42
        Random state for reproducibility.

    Attributes
    ----------
    labels_ : np.ndarray
        Cluster labels for each sample after fitting.

    model_ : object
        Underlying sklearn (or similar) clustering model.

    n_clusters_ : int
        Actual number of clusters found (may differ from n_clusters for some algorithms).
    """

    def __init__(self, n_clusters: int, random_state: int = 42):
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.labels_: Optional[np.ndarray] = None
        self.model_: Optional[object] = None
        self.n_clusters_: Optional[int] = None

    @abstractmethod
    def fit(self, X: pd.DataFrame) -> 'BaseClusterer':
        """
        Fit the clustering model.

        Parameters
        ----------
        X : pd.DataFrame
            Input data.

        Returns
        -------
        self : BaseClusterer
            Fitted clusterer.
        """
        pass

    @abstractmethod
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
            Cluster labels.
        """
        pass

    def fit_predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Fit the model and predict cluster labels.

        Parameters
        ----------
        X : pd.DataFrame
            Input data.

        Returns
        -------
        labels : np.ndarray
            Cluster labels.
        """
        self.fit(X)
        return self.labels_

    def get_cluster_centers(self) -> Optional[np.ndarray]:
        """
        Get cluster centers (if available for the algorithm).

        Returns
        -------
        centers : np.ndarray or None
            Cluster centers, or None if not applicable.
        """
        if hasattr(self.model_, 'cluster_centers_'):
            return self.model_.cluster_centers_
        return None

    def get_cluster_sizes(self) -> pd.Series:
        """
        Get the size of each cluster.

        Returns
        -------
        sizes : pd.Series
            Number of samples in each cluster.
        """
        if self.labels_ is None:
            raise ValueError("Model must be fitted first")

        return pd.Series(self.labels_).value_counts().sort_index()

    def __repr__(self) -> str:
        """String representation."""
        return f"{self.__class__.__name__}(n_clusters={self.n_clusters})"
