"""
Main pipeline class for cluster analysis.

This module provides the ClusterAnalysisPipeline class, which orchestrates
the entire clustering workflow from preprocessing to interpretation.
"""

from typing import Optional, Union, List, Dict, Any, Callable
import warnings
import pandas as pd
import numpy as np


class ClusterAnalysisPipeline:
    """
    Complete pipeline for cluster analysis.

    This class orchestrates the entire clustering workflow including:
    - Data preprocessing (missing values, outliers, scaling)
    - Feature selection (correlation, variance)
    - Dimensionality reduction (PCA)
    - Clustering (K-Means, GMM, Hierarchical, DBSCAN)
    - Evaluation (metrics, optimal k finding)
    - Interpretation (profiling, naming)
    - Visualization (optional)

    Parameters
    ----------
    handle_missing : str or callable, default='median'
        Strategy for handling missing values.
        Options: 'median', 'mean', 'drop', or a custom function.

    handle_outliers : str or None, default='robust'
        Strategy for handling outliers.
        Options: 'robust' (RobustScaler), 'clip', 'remove', None.

    scaling : str, default='robust'
        Scaling method to use.
        Options: 'standard', 'robust', 'minmax', 'auto'.

    log_transform_skewed : bool, default=False
        Whether to apply log transformation to skewed features.

    skewness_threshold : float, default=2.0
        Threshold for detecting skewed features (if log_transform_skewed=True).

    correlation_threshold : float, default=0.85
        Threshold for removing highly correlated features.
        Features with |correlation| > threshold will be removed.

    variance_threshold : float, default=0.01
        Minimum variance threshold for feature selection.
        Features with variance < threshold will be removed.

    pca_variance : float, default=0.9
        Minimum proportion of variance to explain with PCA components.

    pca_min_components : int, default=2
        Minimum number of PCA components to keep (for visualization).

    clustering_algorithm : str or object, default='kmeans'
        Clustering algorithm to use.
        Options: 'kmeans', 'gmm', 'hierarchical', 'dbscan', or a custom clusterer.

    n_clusters : int, list, or None, default=None
        Number of clusters. If None, will be automatically determined.
        If list, will try all values and select the best.

    n_clusters_range : tuple, default=(2, 10)
        Range of cluster numbers to try for automatic selection (if n_clusters=None).

    random_state : int, default=42
        Random state for reproducibility.

    verbose : bool, default=True
        Whether to print progress messages.

    Attributes
    ----------
    data_ : pd.DataFrame
        Original input data.

    data_preprocessed_ : pd.DataFrame
        Data after preprocessing (missing values, outliers handled).

    data_scaled_ : pd.DataFrame
        Data after scaling.

    selected_features_ : list
        List of selected feature names after feature selection.

    data_reduced_ : pd.DataFrame
        Data after dimensionality reduction (PCA components).

    labels_ : np.ndarray
        Cluster labels for each sample.

    n_clusters_ : int
        Final number of clusters used.

    cluster_profiles_ : pd.DataFrame
        Profile of each cluster (mean values of features).

    metrics_ : dict
        Clustering evaluation metrics.

    model_ : object
        Fitted clustering model.

    Examples
    --------
    >>> from clustertk import ClusterAnalysisPipeline
    >>> import pandas as pd
    >>>
    >>> # Create sample data
    >>> df = pd.DataFrame({'feature1': [1, 2, 3, 4, 5],
    ...                    'feature2': [2, 4, 6, 8, 10]})
    >>>
    >>> # Create and fit pipeline
    >>> pipeline = ClusterAnalysisPipeline(
    ...     handle_missing='median',
    ...     correlation_threshold=0.85,
    ...     n_clusters=2
    ... )
    >>> pipeline.fit(df)
    >>>
    >>> # Get results
    >>> labels = pipeline.labels_
    >>> profiles = pipeline.cluster_profiles_
    """

    def __init__(
        self,
        # Preprocessing parameters
        handle_missing: Union[str, Callable] = 'median',
        handle_outliers: Optional[str] = 'robust',
        scaling: str = 'robust',
        log_transform_skewed: bool = False,
        skewness_threshold: float = 2.0,
        # Feature selection parameters
        correlation_threshold: float = 0.85,
        variance_threshold: float = 0.01,
        # Dimensionality reduction parameters
        pca_variance: float = 0.9,
        pca_min_components: int = 2,
        # Clustering parameters
        clustering_algorithm: Union[str, object] = 'kmeans',
        n_clusters: Optional[Union[int, List[int]]] = None,
        n_clusters_range: tuple = (2, 10),
        # General parameters
        random_state: int = 42,
        verbose: bool = True
    ):
        # Store parameters
        self.handle_missing = handle_missing
        self.handle_outliers = handle_outliers
        self.scaling = scaling
        self.log_transform_skewed = log_transform_skewed
        self.skewness_threshold = skewness_threshold
        self.correlation_threshold = correlation_threshold
        self.variance_threshold = variance_threshold
        self.pca_variance = pca_variance
        self.pca_min_components = pca_min_components
        self.clustering_algorithm = clustering_algorithm
        self.n_clusters = n_clusters
        self.n_clusters_range = n_clusters_range
        self.random_state = random_state
        self.verbose = verbose

        # Initialize attributes that will be set during fitting
        self.data_: Optional[pd.DataFrame] = None
        self.data_preprocessed_: Optional[pd.DataFrame] = None
        self.data_scaled_: Optional[pd.DataFrame] = None
        self.selected_features_: Optional[List[str]] = None
        self.data_reduced_: Optional[pd.DataFrame] = None
        self.labels_: Optional[np.ndarray] = None
        self.n_clusters_: Optional[int] = None
        self.cluster_profiles_: Optional[pd.DataFrame] = None
        self.metrics_: Optional[Dict[str, Any]] = None
        self.model_: Optional[object] = None

        # Internal components (will be initialized during fit)
        self._missing_handler = None
        self._outlier_handler = None
        self._scaler = None
        self._transformer = None
        self._correlation_filter = None
        self._variance_filter = None
        self._pca_reducer = None
        self._clusterer = None
        self._profiler = None

    def fit(
        self,
        X: pd.DataFrame,
        feature_columns: Optional[List[str]] = None,
        category_mapping: Optional[Dict[str, List[str]]] = None
    ) -> 'ClusterAnalysisPipeline':
        """
        Run the complete clustering pipeline.

        This method executes all steps sequentially:
        1. Preprocessing
        2. Feature selection
        3. Dimensionality reduction
        4. Optimal cluster finding (if n_clusters not specified)
        5. Clustering
        6. Profiling

        Parameters
        ----------
        X : pd.DataFrame
            Input data.

        feature_columns : list of str, optional
            List of column names to use as features.
            If None, all numeric columns will be used.

        category_mapping : dict, optional
            Mapping of category names to feature names for interpretation.
            Example: {'behavioral': ['feature1', 'feature2'], 'social': ['feature3']}

        Returns
        -------
        self : ClusterAnalysisPipeline
            Fitted pipeline instance.
        """
        if self.verbose:
            print("=" * 80)
            print("ClusterTK Pipeline - Starting cluster analysis")
            print("=" * 80)

        # Run all steps
        self.preprocess(X, feature_columns)
        self.select_features()
        self.reduce_dimensions()

        if self.n_clusters is None:
            self.find_optimal_clusters()

        self.cluster()
        self.create_profiles(category_mapping)

        if self.verbose:
            print("\n" + "=" * 80)
            print("Pipeline completed successfully!")
            print(f"Final clusters: {self.n_clusters_}")
            print("=" * 80)

        return self

    def preprocess(
        self,
        X: pd.DataFrame,
        feature_columns: Optional[List[str]] = None
    ) -> 'ClusterAnalysisPipeline':
        """
        Preprocess the data (missing values, outliers, scaling).

        Parameters
        ----------
        X : pd.DataFrame
            Input data.

        feature_columns : list of str, optional
            List of column names to use as features.
            If None, all numeric columns will be used.

        Returns
        -------
        self : ClusterAnalysisPipeline
        """
        from clustertk.preprocessing import (
            MissingValueHandler,
            OutlierHandler,
            ScalerSelector,
            SkewnessTransformer
        )

        if self.verbose:
            print("\nStep 1/6: Preprocessing data...")

        # Store original data
        self.data_ = X.copy()

        # Select feature columns
        if feature_columns is None:
            feature_columns = X.select_dtypes(include=[np.number]).columns.tolist()
            if self.verbose:
                print(f"  No feature_columns specified, using {len(feature_columns)} numeric columns")

        data_working = X[feature_columns].copy()

        if self.verbose:
            print(f"  Initial data shape: {data_working.shape}")

        # Step 1.1: Handle infinity and extreme values (replace with NaN)
        # Check for infinity
        inf_mask = np.isinf(data_working.values)
        inf_count = inf_mask.sum()

        # Check for extremely large values that may cause issues
        # Use a threshold of 1e308 (close to float64 max ~1.8e308)
        extreme_mask = np.abs(data_working.values) > 1e100
        extreme_count = extreme_mask.sum() - inf_count  # Don't double count infinity

        total_problematic = inf_count + extreme_count

        if total_problematic > 0:
            if self.verbose:
                if inf_count > 0 and extreme_count > 0:
                    print(f"  Detected {inf_count} infinity and {extreme_count} extreme values, replacing with NaN...")
                elif inf_count > 0:
                    print(f"  Detected {inf_count} infinity values, replacing with NaN...")
                else:
                    print(f"  Detected {extreme_count} extreme values, replacing with NaN...")

            # Replace infinity with NaN
            data_working = data_working.replace([np.inf, -np.inf], np.nan)

            # Replace extreme values with NaN
            data_working = data_working.mask(np.abs(data_working) > 1e100)

            if self.verbose:
                print(f"    ✓ Problematic values replaced with NaN")

        # Step 1.2: Handle missing values
        if self.verbose:
            missing_count = data_working.isnull().sum().sum()
            if missing_count > 0:
                print(f"  Handling {missing_count} missing values (strategy: {self.handle_missing})...")

        self._missing_handler = MissingValueHandler(strategy=self.handle_missing)
        data_working = self._missing_handler.fit_transform(data_working)

        if self.verbose and missing_count > 0:
            print(f"    ✓ Missing values handled")

        # Step 1.3: Apply log transformation to skewed features (if enabled)
        if self.log_transform_skewed:
            if self.verbose:
                print(f"  Detecting skewed features (threshold: {self.skewness_threshold})...")

            self._transformer = SkewnessTransformer(
                threshold=self.skewness_threshold,
                method='log1p'
            )
            data_working = self._transformer.fit_transform(data_working)

            if self.verbose and len(self._transformer.skewed_features_) > 0:
                print(f"    ✓ Transformed {len(self._transformer.skewed_features_)} skewed features")

            # Check again for infinity after transformation
            inf_after_transform = np.isinf(data_working.values).sum()
            if inf_after_transform > 0:
                if self.verbose:
                    print(f"  Detected {inf_after_transform} infinity values after transformation, replacing with NaN...")
                data_working = data_working.replace([np.inf, -np.inf], np.nan)
                # Re-impute new missing values
                data_working = self._missing_handler.transform(data_working)
                if self.verbose:
                    print(f"    ✓ Post-transformation infinity handled")

        self.data_preprocessed_ = data_working.copy()

        # Step 1.4: Handle outliers and scaling
        if self.verbose:
            print(f"  Scaling data (method: {self.scaling})...")

        # Handle outliers if specified
        if self.handle_outliers == 'robust':
            # Use RobustScaler which is robust to outliers
            self._scaler = ScalerSelector(scaler_type='robust')
        elif self.handle_outliers == 'clip':
            # Clip outliers before scaling
            self._outlier_handler = OutlierHandler(method='iqr', action='clip', threshold=1.5)
            data_working = self._outlier_handler.fit_transform(data_working)
            self._scaler = ScalerSelector(scaler_type=self.scaling)
        elif self.handle_outliers == 'remove':
            # Remove outliers
            self._outlier_handler = OutlierHandler(method='iqr', action='remove', threshold=1.5)
            data_working = self._outlier_handler.fit_transform(data_working)
            self._scaler = ScalerSelector(scaler_type=self.scaling)
            if self.verbose:
                print(f"    Rows after outlier removal: {len(data_working)}")
        else:
            # No outlier handling, just scale
            self._scaler = ScalerSelector(scaler_type=self.scaling)

        # Apply scaling
        self.data_scaled_ = self._scaler.fit_transform(data_working)

        if self.verbose:
            print(f"    ✓ Data scaled using {self._scaler.selected_scaler_type_} scaler")
            print(f"  Final preprocessed shape: {self.data_scaled_.shape}")
            print("  ✓ Preprocessing completed")

        return self

    def select_features(self) -> 'ClusterAnalysisPipeline':
        """
        Select features based on correlation and variance.

        Returns
        -------
        self : ClusterAnalysisPipeline
        """
        from clustertk.feature_selection import CorrelationFilter, VarianceFilter

        if self.verbose:
            print("\nStep 2/6: Selecting features...")

        initial_features = len(self.data_scaled_.columns)
        data_working = self.data_scaled_.copy()

        # Step 2.1: Remove low-variance features
        if self.variance_threshold > 0:
            if self.verbose:
                print(f"  Removing low-variance features (threshold: {self.variance_threshold})...")

            self._variance_filter = VarianceFilter(threshold=self.variance_threshold)
            data_working = self._variance_filter.fit_transform(data_working)

            removed_count = len(self._variance_filter.features_to_drop_)
            if self.verbose and removed_count > 0:
                print(f"    ✓ Removed {removed_count} low-variance features")

        # Step 2.2: Remove highly correlated features
        if self.correlation_threshold < 1.0:
            if self.verbose:
                print(f"  Removing highly correlated features (threshold: {self.correlation_threshold})...")

            self._correlation_filter = CorrelationFilter(
                threshold=self.correlation_threshold,
                method='pearson'
            )
            data_working = self._correlation_filter.fit_transform(data_working)

            removed_count = len(self._correlation_filter.features_to_drop_)
            if self.verbose and removed_count > 0:
                print(f"    ✓ Removed {removed_count} highly correlated features")
                if self._correlation_filter.high_correlation_pairs_:
                    print(f"    Found {len(self._correlation_filter.high_correlation_pairs_)} "
                          f"high correlation pairs")

        # Store selected features and update scaled data
        self.selected_features_ = data_working.columns.tolist()
        self.data_scaled_ = data_working

        final_features = len(self.selected_features_)

        if self.verbose:
            print(f"  Feature selection: {initial_features} → {final_features} features")
            print("  ✓ Feature selection completed")

        return self

    def reduce_dimensions(self) -> 'ClusterAnalysisPipeline':
        """
        Reduce dimensionality using PCA.

        Returns
        -------
        self : ClusterAnalysisPipeline
        """
        from clustertk.dimensionality import PCAReducer

        if self.verbose:
            print("\nStep 3/6: Reducing dimensions with PCA...")

        initial_dims = len(self.selected_features_)

        # Apply PCA
        self._pca_reducer = PCAReducer(
            variance_threshold=self.pca_variance,
            min_components=self.pca_min_components,
            random_state=self.random_state
        )

        self.data_reduced_ = self._pca_reducer.fit_transform(
            self.data_scaled_[self.selected_features_]
        )

        final_dims = self.data_reduced_.shape[1]
        variance_explained = self._pca_reducer.cumulative_variance_[final_dims - 1]

        if self.verbose:
            print(f"  Dimensionality: {initial_dims} → {final_dims} components")
            print(f"  Variance explained: {variance_explained:.1%}")

            # Show top loadings for first 2 components
            if final_dims >= 2:
                print(f"  Top features in PC1:")
                loadings = self._pca_reducer.get_loadings(n_top=3)
                for feature, loading in loadings['PC1']:
                    sign = '+' if loading > 0 else ''
                    print(f"    {sign}{loading:.3f} {feature}")

            print("  ✓ Dimensionality reduction completed")

        return self

    def find_optimal_clusters(self) -> 'ClusterAnalysisPipeline':
        """
        Find optimal number of clusters if not specified.

        Returns
        -------
        self : ClusterAnalysisPipeline
        """
        from clustertk.evaluation import OptimalKFinder
        from clustertk.clustering import KMeansClustering, GMMClustering

        if self.verbose:
            print("\nStep 4/6: Finding optimal number of clusters...")
            print(f"  Testing k in range {self.n_clusters_range}...")

        # Select clusterer class based on algorithm
        if isinstance(self.clustering_algorithm, str):
            if self.clustering_algorithm == 'kmeans':
                clusterer_class = KMeansClustering
            elif self.clustering_algorithm == 'gmm':
                clusterer_class = GMMClustering
            else:
                # Default to KMeans
                clusterer_class = KMeansClustering
        else:
            # Custom clusterer provided
            clusterer_class = self.clustering_algorithm.__class__

        # Find optimal k
        finder = OptimalKFinder(
            k_range=self.n_clusters_range,
            method='voting',
            random_state=self.random_state
        )

        self.n_clusters_ = finder.find_optimal_k(self.data_reduced_, clusterer_class)

        if self.verbose:
            print(f"  Optimal k: {self.n_clusters_}")

            # Show metric votes
            if finder.metric_votes_:
                print("  Metric votes:")
                for metric, k in finder.metric_votes_.items():
                    print(f"    {metric}: k={k}")

            print("  ✓ Optimal cluster finding completed")

        return self

    def cluster(
        self,
        n_clusters: Optional[int] = None,
        algorithm: Optional[str] = None
    ) -> 'ClusterAnalysisPipeline':
        """
        Perform clustering.

        Parameters
        ----------
        n_clusters : int, optional
            Number of clusters. If None, uses self.n_clusters_.

        algorithm : str, optional
            Clustering algorithm. If None, uses self.clustering_algorithm.

        Returns
        -------
        self : ClusterAnalysisPipeline
        """
        from clustertk.clustering import KMeansClustering, GMMClustering
        from clustertk.evaluation import compute_clustering_metrics

        if self.verbose:
            print("\nStep 5/6: Performing clustering...")

        # Determine number of clusters
        if n_clusters is not None:
            self.n_clusters_ = n_clusters
        elif self.n_clusters_ is None:
            if isinstance(self.n_clusters, int):
                self.n_clusters_ = self.n_clusters
            elif isinstance(self.n_clusters, list):
                self.n_clusters_ = self.n_clusters[0]
            else:
                self.n_clusters_ = 3

        # Determine algorithm
        algo = algorithm or self.clustering_algorithm

        # Initialize clusterer
        if isinstance(algo, str):
            if algo == 'kmeans':
                self._clusterer = KMeansClustering(
                    n_clusters=self.n_clusters_,
                    random_state=self.random_state
                )
            elif algo == 'gmm':
                self._clusterer = GMMClustering(
                    n_clusters=self.n_clusters_,
                    random_state=self.random_state
                )
            else:
                raise ValueError(f"Unknown algorithm '{algo}'. Use 'kmeans' or 'gmm'.")
        else:
            # Custom clusterer provided
            self._clusterer = algo

        # Fit clustering
        self.labels_ = self._clusterer.fit_predict(self.data_reduced_)
        self.model_ = self._clusterer

        # Compute metrics
        self.metrics_ = compute_clustering_metrics(self.data_reduced_, self.labels_)

        if self.verbose:
            print(f"  Algorithm: {algo}")
            print(f"  Number of clusters: {self.n_clusters_}")
            print(f"  Cluster sizes: {dict(pd.Series(self.labels_).value_counts().sort_index())}")
            print(f"  Silhouette score: {self.metrics_['silhouette']:.3f}")
            print("  ✓ Clustering completed")

        return self

    def create_profiles(
        self,
        category_mapping: Optional[Dict[str, List[str]]] = None
    ) -> 'ClusterAnalysisPipeline':
        """
        Create cluster profiles and interpretations.

        Parameters
        ----------
        category_mapping : dict, optional
            Mapping of category names to feature names.
            Example: {'behavioral': ['feat1', 'feat2'], 'social': ['feat3']}

        Returns
        -------
        self : ClusterAnalysisPipeline
        """
        from clustertk.interpretation import ClusterProfiler

        if self.verbose:
            print("\nStep 6/6: Creating cluster profiles...")

        # Create profiler
        self._profiler = ClusterProfiler(normalize_per_feature=True)

        # Create profiles using scaled data
        self.cluster_profiles_ = self._profiler.create_profiles(
            self.data_scaled_[self.selected_features_],
            self.labels_,
            feature_names=self.selected_features_
        )

        # Get top distinguishing features
        self._profiler.get_top_features(n=5)

        if self.verbose:
            print(f"  Profiles created for {len(self.cluster_profiles_)} clusters")

            # Show cluster sizes
            cluster_sizes = pd.Series(self.labels_).value_counts().sort_index()
            print(f"  Cluster sizes:")
            for cluster, size in cluster_sizes.items():
                pct = size / len(self.labels_) * 100
                print(f"    Cluster {cluster}: {size} samples ({pct:.1f}%)")

        # Analyze by categories if provided
        if category_mapping is not None:
            if self.verbose:
                print(f"  Analyzing {len(category_mapping)} feature categories...")

            self._profiler.analyze_by_categories(
                self.data_scaled_[self.selected_features_],
                self.labels_,
                category_mapping
            )

            if self.verbose:
                print("  ✓ Category analysis completed")

        if self.verbose:
            print("  ✓ Profiling completed")

        return self

    def export_results(
        self,
        path: str,
        format: str = 'csv',
        include_original: bool = True
    ) -> None:
        """
        Export clustering results to file.

        Parameters
        ----------
        path : str
            Output file path.

        format : str, default='csv'
            Output format: 'csv', 'json', 'pickle'.

        include_original : bool, default=True
            Whether to include original data in export.
        """
        # TODO: Implement export logic
        if self.verbose:
            print(f"\nExporting results to {path}...")

        warnings.warn("Export functionality not yet implemented", UserWarning)

    def __repr__(self) -> str:
        """String representation of the pipeline."""
        return (
            f"ClusterAnalysisPipeline(\n"
            f"  clustering_algorithm={self.clustering_algorithm},\n"
            f"  n_clusters={self.n_clusters},\n"
            f"  scaling={self.scaling},\n"
            f"  pca_variance={self.pca_variance}\n"
            f")"
        )
