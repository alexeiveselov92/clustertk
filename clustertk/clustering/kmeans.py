"""
K-Means clustering algorithm implementation.

K-Means is one of the most popular clustering algorithms, using an iterative
approach to partition data into k clusters based on distance to centroids.
"""

from typing import Optional
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans as SklearnKMeans

from clustertk.clustering.base import BaseClusterer


class KMeansClustering(BaseClusterer):
    """
    K-Means clustering wrapper.

    K-Means attempts to minimize the within-cluster sum of squares by iteratively
    assigning points to the nearest cluster center and updating the centers.

    Parameters
    ----------
    n_clusters : int, default=3
        Number of clusters to form.

    random_state : int, default=42
        Random state for reproducibility.

    n_init : int, default=10
        Number of times the algorithm will be run with different centroid seeds.
        The best result in terms of inertia will be kept.

    max_iter : int, default=300
        Maximum number of iterations for a single run.

    init : str, default='k-means++'
        Method for initialization:
        - 'k-means++': Smart initialization (recommended)
        - 'random': Random initialization

    **kwargs : dict
        Additional parameters to pass to sklearn's KMeans.

    Attributes
    ----------
    inertia_ : float
        Sum of squared distances of samples to their closest cluster center.

    n_iter_ : int
        Number of iterations run.

    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> from clustertk.clustering import KMeansClustering
    >>>
    >>> # Create sample data
    >>> df = pd.DataFrame(np.random.rand(100, 5))
    >>>
    >>> # Fit K-Means
    >>> kmeans = KMeansClustering(n_clusters=3)
    >>> labels = kmeans.fit_predict(df)
    >>> print(f"Inertia: {kmeans.inertia_:.2f}")
    """

    def __init__(
        self,
        n_clusters: int = 3,
        random_state: int = 42,
        n_init: int = 10,
        max_iter: int = 300,
        init: str = 'k-means++',
        **kwargs
    ):
        super().__init__(n_clusters=n_clusters, random_state=random_state)
        self.n_init = n_init
        self.max_iter = max_iter
        self.init = init
        self.kwargs = kwargs
        self.inertia_: Optional[float] = None
        self.n_iter_: Optional[int] = None

    def fit(self, X: pd.DataFrame) -> 'KMeansClustering':
        """
        Fit K-Means clustering model.

        Parameters
        ----------
        X : pd.DataFrame
            Input data.

        Returns
        -------
        self : KMeansClustering
            Fitted clusterer.
        """
        if not isinstance(X, pd.DataFrame):
            raise TypeError("X must be a pandas DataFrame")

        # Initialize sklearn KMeans
        self.model_ = SklearnKMeans(
            n_clusters=self.n_clusters,
            random_state=self.random_state,
            n_init=self.n_init,
            max_iter=self.max_iter,
            init=self.init,
            **self.kwargs
        )

        # Fit the model
        self.model_.fit(X)

        # Store results
        self.labels_ = self.model_.labels_
        self.n_clusters_ = self.n_clusters
        self.inertia_ = self.model_.inertia_
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
            Predicted cluster labels.
        """
        if self.model_ is None:
            raise ValueError("Model must be fitted before predict")

        if not isinstance(X, pd.DataFrame):
            raise TypeError("X must be a pandas DataFrame")

        return self.model_.predict(X)

    def get_cluster_centers(self) -> np.ndarray:
        """
        Get K-Means cluster centers.

        Returns
        -------
        centers : np.ndarray
            Cluster centers.
        """
        if self.model_ is None:
            raise ValueError("Model must be fitted first")

        return self.model_.cluster_centers_

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """
        Transform X to a cluster-distance space.

        Returns distances to each cluster center.

        Parameters
        ----------
        X : pd.DataFrame
            Input data.

        Returns
        -------
        distances : np.ndarray
            Distances to cluster centers.
        """
        if self.model_ is None:
            raise ValueError("Model must be fitted first")

        return self.model_.transform(X)

    def get_distances_to_centers(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Get distances from each point to each cluster center.

        Parameters
        ----------
        X : pd.DataFrame
            Input data.

        Returns
        -------
        distances : pd.DataFrame
            DataFrame with distances to each cluster.
        """
        distances = self.transform(X)

        return pd.DataFrame(
            distances,
            columns=[f'dist_to_cluster_{i}' for i in range(self.n_clusters)],
            index=X.index
        )

    def __repr__(self) -> str:
        """String representation."""
        if self.inertia_ is not None:
            return (
                f"KMeansClustering(n_clusters={self.n_clusters}, "
                f"inertia={self.inertia_:.2f})"
            )
        return f"KMeansClustering(n_clusters={self.n_clusters})"
