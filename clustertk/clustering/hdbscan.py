"""
HDBSCAN clustering algorithm implementation.

HDBSCAN (Hierarchical Density-Based Spatial Clustering of Applications with Noise)
is an advanced density-based clustering algorithm that automatically determines
the number of clusters and can handle varying density clusters.
"""

from typing import Optional
import pandas as pd
import numpy as np

from clustertk.clustering.base import BaseClusterer


class HDBSCANClustering(BaseClusterer):
    """
    HDBSCAN (Hierarchical Density-Based Spatial Clustering) wrapper.

    HDBSCAN extends DBSCAN by building a cluster hierarchy and extracting
    the most stable clusters. It handles varying density clusters better
    than DBSCAN and requires fewer parameters to tune.

    Parameters
    ----------
    min_cluster_size : int or 'auto', default='auto'
        The minimum number of samples in a cluster. If 'auto', will be
        estimated based on dataset size (typically sqrt(n_samples)).

    min_samples : int or 'auto', default='auto'
        The number of samples in a neighborhood for a point to be considered
        a core point. If 'auto', will be set equal to min_cluster_size.

    metric : str, default='euclidean'
        The metric to use for distance computation:
        - 'euclidean': Euclidean distance
        - 'manhattan': Manhattan distance
        - 'cosine': Cosine similarity
        - Any metric supported by hdbscan library

    cluster_selection_method : str, default='eom'
        Method for selecting clusters from the condensed tree:
        - 'eom': Excess of Mass (default, more stable)
        - 'leaf': Leaf node selection (more clusters)

    alpha : float, default=1.0
        Distance scaling parameter for mutual reachability distance.
        Larger values make clusters more conservative.

    cluster_selection_epsilon : float, default=0.0
        A distance threshold for cluster extraction. Clusters below this
        threshold are merged. Use 0.0 to disable.

    n_jobs : int, optional
        Number of parallel jobs to run. -1 means using all processors.

    **kwargs : dict
        Additional parameters to pass to hdbscan.HDBSCAN.

    Attributes
    ----------
    probabilities_ : np.ndarray
        Cluster membership probabilities for each point.

    outlier_scores_ : np.ndarray
        Outlier score for each point (higher = more outlier-like).

    n_noise_points_ : int
        Number of noise points (outliers) detected.

    cluster_persistence_ : dict
        Persistence values for each cluster (measure of stability).

    Notes
    -----
    - Points with label -1 are considered noise/outliers
    - HDBSCAN automatically determines the number of clusters
    - More robust to parameter choices than DBSCAN
    - Requires hdbscan library: pip install hdbscan

    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> from clustertk.clustering import HDBSCANClustering
    >>>
    >>> # Create sample data
    >>> df = pd.DataFrame(np.random.rand(100, 5))
    >>>
    >>> # Fit HDBSCAN with auto parameters
    >>> hdbscan = HDBSCANClustering(min_cluster_size='auto')
    >>> labels = hdbscan.fit_predict(df)
    >>> print(f"Found {hdbscan.n_clusters_} clusters")
    >>> print(f"Noise points: {hdbscan.n_noise_points_}")
    >>>
    >>> # Fit with manual parameters
    >>> hdbscan = HDBSCANClustering(min_cluster_size=10, min_samples=5)
    >>> labels = hdbscan.fit_predict(df)
    >>>
    >>> # Get cluster membership probabilities
    >>> probs = hdbscan.probabilities_
    >>> weak_members = probs < 0.5  # Points weakly assigned to clusters
    """

    def __init__(
        self,
        min_cluster_size: int | str = 'auto',
        min_samples: int | str = 'auto',
        metric: str = 'euclidean',
        cluster_selection_method: str = 'eom',
        alpha: float = 1.0,
        cluster_selection_epsilon: float = 0.0,
        n_jobs: Optional[int] = None,
        **kwargs
    ):
        # HDBSCAN doesn't use n_clusters, but we inherit from BaseClusterer
        # Set it to None initially, will be determined after fitting
        super().__init__(n_clusters=None, random_state=42)
        self.min_cluster_size = min_cluster_size
        self.min_samples = min_samples
        self.metric = metric
        self.cluster_selection_method = cluster_selection_method
        self.alpha = alpha
        self.cluster_selection_epsilon = cluster_selection_epsilon
        self.n_jobs = n_jobs
        self.kwargs = kwargs

        # Attributes set after fitting
        self.probabilities_: Optional[np.ndarray] = None
        self.outlier_scores_: Optional[np.ndarray] = None
        self.n_noise_points_: Optional[int] = None
        self.cluster_persistence_: Optional[dict] = None
        self._min_cluster_size_computed: Optional[int] = None
        self._min_samples_computed: Optional[int] = None

    def _check_hdbscan_available(self):
        """Check if hdbscan library is installed."""
        try:
            import hdbscan  # noqa: F401
        except ImportError:
            raise ImportError(
                "HDBSCAN requires the hdbscan library. "
                "Install it with: pip install hdbscan"
            )

    def _estimate_min_cluster_size(self, X: pd.DataFrame) -> int:
        """
        Estimate min_cluster_size parameter.

        Uses sqrt(n_samples) as a heuristic, with bounds [5, 100].

        Parameters
        ----------
        X : pd.DataFrame
            Input data.

        Returns
        -------
        min_cluster_size : int
            Estimated min_cluster_size value.
        """
        n_samples = len(X)
        # Use sqrt(n_samples) as heuristic
        size = int(np.sqrt(n_samples))
        # Bound between 5 and 100
        return max(5, min(100, size))

    def _estimate_min_samples(self, X: pd.DataFrame) -> int:
        """
        Estimate min_samples parameter.

        If not specified, defaults to min_cluster_size (common practice).

        Parameters
        ----------
        X : pd.DataFrame
            Input data.

        Returns
        -------
        min_samples : int
            Estimated min_samples value.
        """
        # Default to min_cluster_size (common practice)
        return self._min_cluster_size_computed

    def fit(self, X: pd.DataFrame) -> 'HDBSCANClustering':
        """
        Fit HDBSCAN clustering model.

        Parameters
        ----------
        X : pd.DataFrame
            Input data.

        Returns
        -------
        self : HDBSCANClustering
            Fitted clusterer.
        """
        self._check_hdbscan_available()
        import hdbscan

        if not isinstance(X, pd.DataFrame):
            raise TypeError("X must be a pandas DataFrame")

        # Estimate min_cluster_size if auto
        if self.min_cluster_size == 'auto':
            self._min_cluster_size_computed = self._estimate_min_cluster_size(X)
        else:
            self._min_cluster_size_computed = self.min_cluster_size

        # Estimate min_samples if auto
        if self.min_samples == 'auto':
            self._min_samples_computed = self._estimate_min_samples(X)
        else:
            self._min_samples_computed = self.min_samples

        # Initialize hdbscan
        # Set n_jobs to 1 if None (hdbscan doesn't accept None)
        n_jobs_value = self.n_jobs if self.n_jobs is not None else 1

        self.model_ = hdbscan.HDBSCAN(
            min_cluster_size=self._min_cluster_size_computed,
            min_samples=self._min_samples_computed,
            metric=self.metric,
            cluster_selection_method=self.cluster_selection_method,
            alpha=self.alpha,
            cluster_selection_epsilon=self.cluster_selection_epsilon,
            core_dist_n_jobs=n_jobs_value,
            **self.kwargs
        )

        # Fit the model
        self.model_.fit(X)

        # Store results
        self.labels_ = self.model_.labels_
        self.probabilities_ = self.model_.probabilities_
        self.outlier_scores_ = self.model_.outlier_scores_

        # Count actual clusters (excluding noise with label -1)
        unique_labels = np.unique(self.labels_)
        self.n_clusters_ = len(unique_labels[unique_labels >= 0])

        # Count noise points
        self.n_noise_points_ = np.sum(self.labels_ == -1)

        # Extract cluster persistence (stability measure)
        if hasattr(self.model_, 'cluster_persistence_'):
            self.cluster_persistence_ = {
                cluster_id: persistence
                for cluster_id, persistence in enumerate(self.model_.cluster_persistence_)
                if cluster_id >= 0
            }

        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict cluster labels for new data.

        Uses approximate prediction based on the trained model.

        Parameters
        ----------
        X : pd.DataFrame
            Input data.

        Returns
        -------
        labels : np.ndarray
            Predicted cluster labels (-1 for noise).
        """
        if self.model_ is None:
            raise ValueError("Model must be fitted before predict")

        if not isinstance(X, pd.DataFrame):
            raise TypeError("X must be a pandas DataFrame")

        # Use hdbscan's approximate_predict if available
        if hasattr(self.model_, 'approximate_predict'):
            labels, strengths = self.model_.approximate_predict(X.values)
            return labels
        else:
            raise NotImplementedError(
                "Prediction requires hdbscan >= 0.8.27 with approximate_predict support"
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

    def get_weak_members_mask(self, threshold: float = 0.5) -> np.ndarray:
        """
        Get boolean mask for weakly assigned cluster members.

        Points with low membership probability are less confidently
        assigned to their cluster.

        Parameters
        ----------
        threshold : float, default=0.5
            Probability threshold below which points are considered weak.

        Returns
        -------
        weak_mask : np.ndarray
            Boolean array where True indicates weak cluster membership.
        """
        if self.probabilities_ is None:
            raise ValueError("Model must be fitted first")

        return self.probabilities_ < threshold

    def get_cluster_stability(self, cluster_id: int) -> float:
        """
        Get stability (persistence) score for a specific cluster.

        Higher values indicate more stable/reliable clusters.

        Parameters
        ----------
        cluster_id : int
            ID of the cluster.

        Returns
        -------
        stability : float
            Stability score for the cluster.
        """
        if self.cluster_persistence_ is None:
            raise ValueError("Model must be fitted first")

        if cluster_id not in self.cluster_persistence_:
            raise ValueError(f"Cluster {cluster_id} not found")

        return self.cluster_persistence_[cluster_id]

    def __repr__(self) -> str:
        """String representation."""
        if self.n_clusters_ is not None:
            return (
                f"HDBSCANClustering(n_clusters={self.n_clusters_}, "
                f"min_cluster_size={self._min_cluster_size_computed}, "
                f"min_samples={self._min_samples_computed}, "
                f"noise_points={self.n_noise_points_})"
            )
        return (
            f"HDBSCANClustering(min_cluster_size={self.min_cluster_size}, "
            f"min_samples={self.min_samples})"
        )
