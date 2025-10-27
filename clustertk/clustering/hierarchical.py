"""
Hierarchical clustering algorithm implementation.

Hierarchical clustering builds a hierarchy of clusters using a bottom-up approach,
merging clusters based on a linkage criterion.
"""

from typing import Optional
import pandas as pd
import numpy as np
from sklearn.cluster import AgglomerativeClustering

from clustertk.clustering.base import BaseClusterer


class HierarchicalClustering(BaseClusterer):
    """
    Hierarchical (Agglomerative) clustering wrapper.

    Hierarchical clustering builds clusters by recursively merging the closest
    pairs of clusters based on a linkage criterion.

    Parameters
    ----------
    n_clusters : int, default=3
        Number of clusters to form.

    linkage : str, default='ward'
        Linkage criterion to use:
        - 'ward': Minimizes the variance within clusters (requires affinity='euclidean')
        - 'complete': Uses maximum distances between all observations of pairs
        - 'average': Uses average distances between all observations of pairs
        - 'single': Uses minimum distances between all observations of pairs

    metric : str, default='euclidean'
        Metric used to compute the linkage:
        - 'euclidean': Euclidean distance
        - 'manhattan': Manhattan distance
        - 'cosine': Cosine similarity
        - 'l1', 'l2': Alternative names for manhattan and euclidean

    distance_threshold : float, optional
        Distance threshold above which clusters will not be merged.
        If not None, n_clusters must be None.

    **kwargs : dict
        Additional parameters to pass to sklearn's AgglomerativeClustering.

    Attributes
    ----------
    children_ : np.ndarray
        The children of each non-leaf node.

    n_leaves_ : int
        Number of leaves in the hierarchical tree.

    n_connected_components_ : int
        Number of connected components in the graph.

    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> from clustertk.clustering import HierarchicalClustering
    >>>
    >>> # Create sample data
    >>> df = pd.DataFrame(np.random.rand(100, 5))
    >>>
    >>> # Fit Hierarchical clustering with ward linkage
    >>> hc = HierarchicalClustering(n_clusters=3, linkage='ward')
    >>> labels = hc.fit_predict(df)
    >>>
    >>> # Try complete linkage
    >>> hc_complete = HierarchicalClustering(n_clusters=3, linkage='complete')
    >>> labels = hc_complete.fit_predict(df)
    """

    def __init__(
        self,
        n_clusters: int = 3,
        linkage: str = 'ward',
        metric: str = 'euclidean',
        distance_threshold: Optional[float] = None,
        **kwargs
    ):
        # Note: AgglomerativeClustering is deterministic and doesn't use random_state
        super().__init__(n_clusters=n_clusters, random_state=None)
        self.linkage = linkage
        self.metric = metric
        self.distance_threshold = distance_threshold
        self.kwargs = kwargs
        self.children_: Optional[np.ndarray] = None
        self.n_leaves_: Optional[int] = None
        self.n_connected_components_: Optional[int] = None

        # Validate linkage and metric combination
        if linkage == 'ward' and metric != 'euclidean':
            raise ValueError("Ward linkage requires metric='euclidean'")

    def fit(self, X: pd.DataFrame) -> 'HierarchicalClustering':
        """
        Fit Hierarchical clustering model.

        Parameters
        ----------
        X : pd.DataFrame
            Input data.

        Returns
        -------
        self : HierarchicalClustering
            Fitted clusterer.
        """
        if not isinstance(X, pd.DataFrame):
            raise TypeError("X must be a pandas DataFrame")

        # Determine n_clusters parameter
        n_clusters_param = None if self.distance_threshold is not None else self.n_clusters

        # Initialize sklearn AgglomerativeClustering
        self.model_ = AgglomerativeClustering(
            n_clusters=n_clusters_param,
            linkage=self.linkage,
            metric=self.metric,
            distance_threshold=self.distance_threshold,
            **self.kwargs
        )

        # Fit the model
        self.model_.fit(X)

        # Store results
        self.labels_ = self.model_.labels_
        self.n_clusters_ = len(np.unique(self.labels_))
        self.children_ = self.model_.children_
        self.n_leaves_ = self.model_.n_leaves_
        self.n_connected_components_ = self.model_.n_connected_components_

        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict cluster labels for new data.

        Note: Hierarchical clustering is not naturally suited for predicting
        on new data. This method finds the nearest cluster center (computed
        as the mean of cluster members) for each new point.

        Parameters
        ----------
        X : pd.DataFrame
            Input data.

        Returns
        -------
        labels : np.ndarray
            Predicted cluster labels.
        """
        if self.model_ is None or self.labels_ is None:
            raise ValueError("Model must be fitted before predict")

        if not isinstance(X, pd.DataFrame):
            raise TypeError("X must be a pandas DataFrame")

        # Since hierarchical clustering doesn't have a native predict,
        # we compute cluster centroids and assign new points to nearest centroid
        # This requires storing the training data
        raise NotImplementedError(
            "Hierarchical clustering does not support prediction on new data. "
            "Use fit_predict() on the full dataset instead."
        )

    def get_dendrogram_data(self) -> dict:
        """
        Get data needed to plot a dendrogram.

        Returns
        -------
        dendrogram_data : dict
            Dictionary containing:
            - 'children': The children of each non-leaf node
            - 'n_leaves': Number of leaves
            - 'n_clusters': Number of clusters found

        Examples
        --------
        >>> hc = HierarchicalClustering(n_clusters=3)
        >>> hc.fit(df)
        >>> data = hc.get_dendrogram_data()
        >>> # Use with scipy.cluster.hierarchy.dendrogram
        """
        if self.model_ is None:
            raise ValueError("Model must be fitted first")

        return {
            'children': self.children_,
            'n_leaves': self.n_leaves_,
            'n_clusters': self.n_clusters_,
            'n_connected_components': self.n_connected_components_
        }

    def __repr__(self) -> str:
        """String representation."""
        if self.n_clusters_ is not None:
            return (
                f"HierarchicalClustering(n_clusters={self.n_clusters_}, "
                f"linkage='{self.linkage}')"
            )
        return f"HierarchicalClustering(n_clusters={self.n_clusters}, linkage='{self.linkage}')"
