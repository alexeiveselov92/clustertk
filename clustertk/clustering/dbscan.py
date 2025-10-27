"""
DBSCAN clustering algorithm implementation.

DBSCAN (Density-Based Spatial Clustering of Applications with Noise) is a
density-based clustering algorithm that can discover clusters of arbitrary shape
and identify outliers.
"""

from typing import Optional
import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors

from clustertk.clustering.base import BaseClusterer


class DBSCANClustering(BaseClusterer):
    """
    DBSCAN (Density-Based Spatial Clustering) wrapper.

    DBSCAN groups together points that are closely packed together, marking
    points in low-density regions as outliers. Unlike K-Means, it doesn't
    require specifying the number of clusters beforehand.

    Parameters
    ----------
    eps : float or 'auto', default='auto'
        The maximum distance between two samples for them to be considered
        as in the same neighborhood. If 'auto', will be estimated using
        the k-distance method.

    min_samples : int or 'auto', default='auto'
        The number of samples in a neighborhood for a point to be considered
        as a core point. If 'auto', will be set to 2 * n_features.

    metric : str, default='euclidean'
        The metric to use for distance computation:
        - 'euclidean': Euclidean distance
        - 'manhattan': Manhattan distance
        - 'cosine': Cosine similarity
        - Any metric from sklearn.metrics.pairwise

    n_jobs : int, optional
        Number of parallel jobs to run. -1 means using all processors.

    **kwargs : dict
        Additional parameters to pass to sklearn's DBSCAN.

    Attributes
    ----------
    core_sample_indices_ : np.ndarray
        Indices of core samples.

    components_ : np.ndarray
        Copy of each core sample.

    n_noise_points_ : int
        Number of noise points (outliers) detected.

    Notes
    -----
    - Points with label -1 are considered noise/outliers
    - The number of clusters found may vary based on eps and min_samples
    - n_clusters parameter from BaseClusterer is ignored for DBSCAN

    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> from clustertk.clustering import DBSCANClustering
    >>>
    >>> # Create sample data
    >>> df = pd.DataFrame(np.random.rand(100, 5))
    >>>
    >>> # Fit DBSCAN with auto parameters
    >>> dbscan = DBSCANClustering(eps='auto', min_samples='auto')
    >>> labels = dbscan.fit_predict(df)
    >>> print(f"Found {dbscan.n_clusters_} clusters")
    >>> print(f"Noise points: {dbscan.n_noise_points_}")
    >>>
    >>> # Fit with manual parameters
    >>> dbscan = DBSCANClustering(eps=0.5, min_samples=5)
    >>> labels = dbscan.fit_predict(df)
    """

    def __init__(
        self,
        eps: float | str = 'auto',
        min_samples: int | str = 'auto',
        metric: str = 'euclidean',
        n_jobs: Optional[int] = None,
        **kwargs
    ):
        # DBSCAN doesn't use n_clusters, but we inherit from BaseClusterer
        # Set it to None initially, will be determined after fitting
        super().__init__(n_clusters=None, random_state=42)
        self.eps = eps
        self.min_samples = min_samples
        self.metric = metric
        self.n_jobs = n_jobs
        self.kwargs = kwargs
        self.core_sample_indices_: Optional[np.ndarray] = None
        self.components_: Optional[np.ndarray] = None
        self.n_noise_points_: Optional[int] = None
        self._eps_computed: Optional[float] = None
        self._min_samples_computed: Optional[int] = None

    def _estimate_eps(self, X: pd.DataFrame) -> float:
        """
        Estimate optimal eps parameter using k-distance method.

        Uses the "elbow" in the k-distance graph where k = min_samples.

        Parameters
        ----------
        X : pd.DataFrame
            Input data.

        Returns
        -------
        eps : float
            Estimated eps value.
        """
        n_samples = len(X)
        k = self._min_samples_computed if self._min_samples_computed else max(4, int(np.log(n_samples)))

        # Compute k-nearest neighbors
        neighbors = NearestNeighbors(n_neighbors=k, metric=self.metric)
        neighbors.fit(X)
        distances, _ = neighbors.kneighbors(X)

        # Take the distance to the kth nearest neighbor
        k_distances = np.sort(distances[:, -1])

        # Use the 90th percentile as eps (heuristic)
        # This works well for datasets where ~10% are outliers
        eps = np.percentile(k_distances, 90)

        return eps

    def _estimate_min_samples(self, X: pd.DataFrame) -> int:
        """
        Estimate min_samples parameter.

        A common heuristic is to use 2 * n_features.

        Parameters
        ----------
        X : pd.DataFrame
            Input data.

        Returns
        -------
        min_samples : int
            Estimated min_samples value.
        """
        n_features = X.shape[1]
        # Common heuristic: 2 * n_features, with minimum of 3
        return max(3, 2 * n_features)

    def fit(self, X: pd.DataFrame) -> 'DBSCANClustering':
        """
        Fit DBSCAN clustering model.

        Parameters
        ----------
        X : pd.DataFrame
            Input data.

        Returns
        -------
        self : DBSCANClustering
            Fitted clusterer.
        """
        if not isinstance(X, pd.DataFrame):
            raise TypeError("X must be a pandas DataFrame")

        # Estimate min_samples if auto
        if self.min_samples == 'auto':
            self._min_samples_computed = self._estimate_min_samples(X)
        else:
            self._min_samples_computed = self.min_samples

        # Estimate eps if auto
        if self.eps == 'auto':
            self._eps_computed = self._estimate_eps(X)
        else:
            self._eps_computed = self.eps

        # Initialize sklearn DBSCAN
        self.model_ = DBSCAN(
            eps=self._eps_computed,
            min_samples=self._min_samples_computed,
            metric=self.metric,
            n_jobs=self.n_jobs,
            **self.kwargs
        )

        # Fit the model
        self.model_.fit(X)

        # Store results
        self.labels_ = self.model_.labels_
        self.core_sample_indices_ = self.model_.core_sample_indices_
        self.components_ = self.model_.components_

        # Count actual clusters (excluding noise with label -1)
        unique_labels = np.unique(self.labels_)
        self.n_clusters_ = len(unique_labels[unique_labels >= 0])

        # Count noise points
        self.n_noise_points_ = np.sum(self.labels_ == -1)

        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict cluster labels for new data.

        Note: DBSCAN doesn't naturally support prediction on new data.
        This method assigns new points to the nearest core point's cluster,
        or labels them as noise (-1) if too far from any core point.

        Parameters
        ----------
        X : pd.DataFrame
            Input data.

        Returns
        -------
        labels : np.ndarray
            Predicted cluster labels (-1 for noise).
        """
        if self.model_ is None or self.components_ is None:
            raise ValueError("Model must be fitted before predict")

        if not isinstance(X, pd.DataFrame):
            raise TypeError("X must be a pandas DataFrame")

        # Find nearest core sample for each new point
        neighbors = NearestNeighbors(n_neighbors=1, metric=self.metric)
        neighbors.fit(self.components_)
        distances, indices = neighbors.kneighbors(X)

        # Get labels of nearest core samples
        core_labels = self.labels_[self.core_sample_indices_]
        predicted_labels = core_labels[indices.flatten()]

        # Mark as noise if distance > eps
        predicted_labels[distances.flatten() > self._eps_computed] = -1

        return predicted_labels

    def get_core_samples(self) -> pd.DataFrame:
        """
        Get the core samples (dense regions).

        Returns
        -------
        core_samples : pd.DataFrame
            DataFrame containing core sample points.
        """
        if self.components_ is None:
            raise ValueError("Model must be fitted first")

        return pd.DataFrame(
            self.components_,
            columns=[f'feature_{i}' for i in range(self.components_.shape[1])]
        )

    def get_noise_mask(self) -> np.ndarray:
        """
        Get boolean mask for noise points.

        Returns
        -------
        noise_mask : np.ndarray
            Boolean array where True indicates noise points.
        """
        if self.labels_ is None:
            raise ValueError("Model must be fitted first")

        return self.labels_ == -1

    def __repr__(self) -> str:
        """String representation."""
        if self.n_clusters_ is not None:
            return (
                f"DBSCANClustering(n_clusters={self.n_clusters_}, "
                f"eps={self._eps_computed:.3f}, "
                f"min_samples={self._min_samples_computed}, "
                f"noise_points={self.n_noise_points_})"
            )
        return f"DBSCANClustering(eps={self.eps}, min_samples={self.min_samples})"
