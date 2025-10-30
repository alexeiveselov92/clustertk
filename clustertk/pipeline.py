"""
Main pipeline class for cluster analysis.

This module provides the ClusterAnalysisPipeline class, which orchestrates
the entire clustering workflow from preprocessing to interpretation.
"""

from typing import Optional, Union, List, Dict, Any, Callable, Literal, Tuple
import warnings
import pandas as pd
import numpy as np


class ClusterAnalysisPipeline:
    """
    Complete pipeline for cluster analysis.

    This class orchestrates the entire clustering workflow including:
    - Data preprocessing (missing values, outliers, scaling)
    - Feature selection (correlation, variance)
    - Dimensionality reduction (PCA/UMAP/None with smart auto-mode)
    - Clustering (K-Means, GMM, Hierarchical, DBSCAN, HDBSCAN)
    - Evaluation (metrics, optimal k finding)
    - Interpretation (profiling, naming)
    - Visualization (optional - requires matplotlib and seaborn)
      * plot_clusters_2d() - 2D scatter plots using t-SNE/UMAP/PCA
      * plot_cluster_sizes() - cluster size distribution
      * plot_cluster_heatmap() - cluster profiles heatmap
      * plot_cluster_radar() - cluster profiles radar chart
      * plot_feature_importance() - top distinguishing features
      * plot_pca_variance() - PCA explained variance
      * plot_pca_loadings() - PCA component loadings
      * plot_correlation_matrix() - correlation matrix heatmap
      * plot_correlation_network() - correlation network graph
      * plot_feature_distributions() - feature distribution histograms

    Parameters
    ----------
    handle_missing : str or callable, default='median'
        Strategy for handling missing values.
        Options: 'median', 'mean', 'drop', or a custom function.

    handle_outliers : str or None, default='winsorize'
        Strategy for handling UNIVARIATE outliers (per-feature extremes) before scaling.
        Options:
        - 'winsorize': Clip to percentile bounds (RECOMMENDED, default since v0.13.0)
          Clips outliers to percentiles specified by outlier_percentiles.
          Best for extreme/asymmetric outliers. Works with any distribution.
        - 'robust': Use RobustScaler only (no outlier removal)
          WARNING: Outliers remain far away after scaling, may create tiny clusters!
        - 'clip': Clip outliers to IQR bounds before scaling
        - 'remove': Remove rows with outliers (data loss)
        - None: No univariate outlier handling

    outlier_percentiles : tuple of (float, float), default=(0.025, 0.975)
        Percentile limits for winsorization when handle_outliers='winsorize'.
        Format: (lower_percentile, upper_percentile).
        Default (0.025, 0.975) clips 2.5% from each tail (5% total, ~±2σ for normal dist).
        Common alternatives:
        - (0.01, 0.99): Very mild, clips 2% total (extreme outliers only)
        - (0.025, 0.975): Balanced (default), clips 5% total
        - (0.05, 0.95): Aggressive, clips 10% total (risk of information loss)
        Only used when handle_outliers='winsorize'.

    detect_multivariate_outliers : str or None, default=None
        Strategy for detecting MULTIVARIATE outliers (outliers in full feature space).
        Multivariate outliers are points far from others when considering ALL features
        together, even if they look normal per-feature. Applied AFTER scaling.
        Options:
        - 'auto': Automatically select best method (IsolationForest/LOF/EllipticEnvelope)
        - 'isolation_forest': IsolationForest (best for high-dimensional data)
        - 'lof': LocalOutlierFactor (best for low-dimensional varying density)
        - 'elliptic_envelope': Robust covariance (best for Gaussian data)
        - None: No multivariate outlier detection (default)

    multivariate_contamination : float, default=0.05
        Expected proportion of multivariate outliers (0.0 to 0.5).
        Only used if detect_multivariate_outliers is not None.

    multivariate_action : str, default='remove'
        Action to take on detected multivariate outliers:
        - 'remove': Remove outlier rows (recommended)
        - 'flag': Add '_is_outlier' column without removing
        Only used if detect_multivariate_outliers is not None.

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

    smart_correlation : bool, default=True
        Use smart correlation filter (Hopkins statistic) to select best
        feature from correlated pairs. If False, drops first feature found.

    correlation_strategy : str, default='hopkins'
        Strategy for selecting features from correlated pairs when smart_correlation=True.
        Options: 'hopkins', 'variance_ratio', 'mean_corr'.

    dim_reduction : str, default='auto'
        Dimensionality reduction method to use. **NEW in v0.15.0!**
        Options:
        - 'auto': Smart selection based on algorithm and number of features (RECOMMENDED)
          * K-Means/GMM with <50 features → 'none'
          * K-Means/GMM with ≥50 features → 'pca'
          * DBSCAN/HDBSCAN with <30 features → 'none'
          * DBSCAN/HDBSCAN with ≥30 features → 'umap'
        - 'pca': Principal Component Analysis (linear, preserves global structure)
          Best for: K-Means, GMM, Hierarchical
        - 'umap': Uniform Manifold Approximation (non-linear, preserves local density)
          Best for: DBSCAN, HDBSCAN (requires: pip install umap-learn)
        - 'none': No dimensionality reduction (work in original feature space)

    pca_variance : float, default=0.9
        Minimum proportion of variance to explain with PCA components.
        Only used if dim_reduction='pca' or auto-selected as PCA.

    pca_min_components : int, default=2
        Minimum number of PCA components to keep (for visualization).
        Only used if dim_reduction='pca' or auto-selected as PCA.

    umap_n_neighbors : int, default=30
        Size of local neighborhood for UMAP (higher = more global structure).
        Only used if dim_reduction='umap' or auto-selected as UMAP.

    umap_min_dist : float, default=0.1
        Minimum distance between points in UMAP embedding (higher = looser clusters).
        Only used if dim_reduction='umap' or auto-selected as UMAP.

    umap_n_components : int, default=10
        Number of UMAP components to keep. NOTE: For clustering use 10-20,
        NOT 2! Lower dimensions (2-3) are only for visualization.
        Only used if dim_reduction='umap' or auto-selected as UMAP.

    umap_metric : str, default='euclidean'
        Distance metric for UMAP. Options: 'euclidean', 'manhattan', 'cosine', etc.
        Only used if dim_reduction='umap' or auto-selected as UMAP.

    clustering_algorithm : str or object, default='kmeans'
        Clustering algorithm to use.
        Options: 'kmeans', 'gmm', 'hierarchical', 'dbscan', 'hdbscan', or a custom clusterer.

    clustering_params : dict or None, default=None
        Additional parameters to pass to the clustering algorithm.
        Example: {'max_iter': 500} for K-Means, {'min_samples': 5} for DBSCAN.

    n_clusters : int, list, or None, default=None
        Number of clusters. If None, will be automatically determined.
        If list, will try all values and select the best.

    n_clusters_range : tuple, default=(2, 10)
        Range of cluster numbers to try for automatic selection (if n_clusters=None).

    auto_name_clusters : bool, default=False
        Whether to automatically generate descriptive names for clusters.

    naming_max_features : int, default=2
        Maximum number of features to include in cluster names.

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
        Data after dimensionality reduction (PCA/UMAP components or original features).

    labels_ : np.ndarray
        Cluster labels for each sample.

    n_clusters_ : int
        Final number of clusters used.

    cluster_profiles_ : pd.DataFrame
        Profile of each cluster (mean values of features).

    cluster_names_ : dict or None
        Descriptive names for each cluster (if auto_name_clusters=True).

    metrics_ : dict
        Clustering evaluation metrics.

    model_ : object
        Fitted clustering model.

    Examples
    --------
    Basic usage with auto-mode (recommended):

    >>> from clustertk import ClusterAnalysisPipeline
    >>> import pandas as pd
    >>>
    >>> # Create sample data
    >>> df = pd.DataFrame({'feature1': [1, 2, 3, 4, 5],
    ...                    'feature2': [2, 4, 6, 8, 10]})
    >>>
    >>> # Create and fit pipeline (auto-mode selects best dim_reduction)
    >>> pipeline = ClusterAnalysisPipeline(
    ...     dim_reduction='auto',  # Smart selection (default)
    ...     clustering_algorithm='kmeans',
    ...     n_clusters=2
    ... )
    >>> pipeline.fit(df)
    >>>
    >>> # Get results
    >>> labels = pipeline.labels_
    >>> profiles = pipeline.cluster_profiles_

    Using HDBSCAN with UMAP for density-based clustering:

    >>> # For high-dimensional data with HDBSCAN
    >>> pipeline = ClusterAnalysisPipeline(
    ...     dim_reduction='umap',  # Preserves local density
    ...     umap_n_components=10,  # NOT 2! For clustering, not viz
    ...     clustering_algorithm='hdbscan'
    ... )
    >>> pipeline.fit(large_df)  # doctest: +SKIP

    Forcing no dimensionality reduction:

    >>> # Work in original feature space
    >>> pipeline = ClusterAnalysisPipeline(
    ...     dim_reduction='none',
    ...     clustering_algorithm='dbscan'
    ... )
    >>> pipeline.fit(df)  # doctest: +SKIP

    Custom outlier percentiles for extreme data:

    >>> # Financial data with extreme outliers - use aggressive winsorization
    >>> pipeline = ClusterAnalysisPipeline(
    ...     handle_outliers='winsorize',
    ...     outlier_percentiles=(0.05, 0.95),  # Clip 10% total (5% each tail)
    ...     clustering_algorithm='kmeans',
    ...     n_clusters=3
    ... )
    >>> pipeline.fit(financial_df)  # doctest: +SKIP
    """

    def __init__(
        self,
        # Preprocessing parameters
        handle_missing: Union[str, Callable] = 'median',
        handle_outliers: Optional[str] = 'winsorize',
        outlier_percentiles: Tuple[float, float] = (0.025, 0.975),
        detect_multivariate_outliers: Optional[str] = None,
        multivariate_contamination: float = 0.05,
        multivariate_action: Literal['remove', 'flag'] = 'remove',
        scaling: str = 'robust',
        log_transform_skewed: bool = False,
        skewness_threshold: float = 2.0,
        # Feature selection parameters
        correlation_threshold: float = 0.85,
        variance_threshold: float = 0.01,
        smart_correlation: bool = True,
        correlation_strategy: Literal['hopkins', 'variance_ratio', 'mean_corr'] = 'hopkins',
        # Dimensionality reduction parameters
        dim_reduction: Literal['auto', 'pca', 'umap', 'none'] = 'auto',
        pca_variance: float = 0.9,
        pca_min_components: int = 2,
        umap_n_neighbors: int = 30,
        umap_min_dist: float = 0.1,
        umap_n_components: int = 10,
        umap_metric: str = 'euclidean',
        # Clustering parameters
        clustering_algorithm: Union[str, object] = 'kmeans',
        n_clusters: Optional[Union[int, List[int]]] = None,
        n_clusters_range: tuple = (2, 10),
        clustering_params: Optional[Dict[str, Any]] = None,
        # Naming parameters
        auto_name_clusters: bool = False,
        naming_max_features: int = 2,
        # General parameters
        random_state: int = 42,
        verbose: bool = True
    ):
        # Store parameters
        self.handle_missing = handle_missing
        self.handle_outliers = handle_outliers
        self.outlier_percentiles = outlier_percentiles
        self.detect_multivariate_outliers = detect_multivariate_outliers
        self.multivariate_contamination = multivariate_contamination
        self.multivariate_action = multivariate_action
        self.scaling = scaling
        self.log_transform_skewed = log_transform_skewed
        self.skewness_threshold = skewness_threshold
        self.correlation_threshold = correlation_threshold
        self.variance_threshold = variance_threshold
        self.smart_correlation = smart_correlation
        self.correlation_strategy = correlation_strategy
        self.dim_reduction = dim_reduction
        self.pca_variance = pca_variance
        self.pca_min_components = pca_min_components
        self.umap_n_neighbors = umap_n_neighbors
        self.umap_min_dist = umap_min_dist
        self.umap_n_components = umap_n_components
        self.umap_metric = umap_metric
        self.clustering_algorithm = clustering_algorithm
        self.n_clusters = n_clusters
        self.n_clusters_range = n_clusters_range
        self.clustering_params = clustering_params or {}
        self.auto_name_clusters = auto_name_clusters
        self.naming_max_features = naming_max_features
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
        self.cluster_names_: Optional[Dict[int, str]] = None
        self.metrics_: Optional[Dict[str, Any]] = None
        self.model_: Optional[object] = None

        # Internal components (will be initialized during fit)
        self._missing_handler = None
        self._outlier_handler = None
        self._scaler = None
        self._transformer = None
        self._correlation_filter = None
        self._variance_filter = None
        self._reducer = None  # Can be PCAReducer, UMAP, or None
        self._clusterer = None
        self._profiler = None
        self._namer = None

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
            MultivariateOutlierDetector,
            ScalerSelector,
            SkewnessTransformer
        )

        if self.verbose:
            print("\n🔧 Preprocessing data...")

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
        elif self.handle_outliers == 'winsorize':
            # Winsorize outliers using percentile-based clipping (recommended)
            self._outlier_handler = OutlierHandler(action='winsorize', percentile_limits=self.outlier_percentiles)
            data_working = self._outlier_handler.fit_transform(data_working)
            self._scaler = ScalerSelector(scaler_type=self.scaling)
            if self.verbose:
                lower_pct = self.outlier_percentiles[0] * 100
                upper_pct = self.outlier_percentiles[1] * 100
                print(f"    Outliers winsorized to {lower_pct:.1f}%-{upper_pct:.1f}% percentile range")
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

        # Step 1.5: Detect and handle multivariate outliers (if enabled)
        if self.detect_multivariate_outliers is not None:
            if self.verbose:
                method_name = self.detect_multivariate_outliers
                if method_name == 'auto':
                    method_name = 'auto-selected method'
                print(f"  Detecting multivariate outliers ({method_name}, contamination={self.multivariate_contamination})...")

            self._multivariate_outlier_detector = MultivariateOutlierDetector(
                method=self.detect_multivariate_outliers,
                contamination=self.multivariate_contamination,
                action=self.multivariate_action,
                random_state=self.random_state
            )

            self.data_scaled_ = self._multivariate_outlier_detector.fit_transform(self.data_scaled_)

            if self.verbose:
                n_outliers = self._multivariate_outlier_detector.n_outliers_
                outlier_ratio = self._multivariate_outlier_detector.outlier_ratio_
                method_used = self._multivariate_outlier_detector.method_used_

                if self.multivariate_action == 'remove':
                    print(f"    ✓ Detected and removed {n_outliers} multivariate outliers ({outlier_ratio:.1%})")
                    print(f"    Method used: {method_used}")
                    print(f"    Rows after removal: {len(self.data_scaled_)}")
                else:
                    print(f"    ✓ Detected {n_outliers} multivariate outliers ({outlier_ratio:.1%}), flagged in data")
                    print(f"    Method used: {method_used}")

        if self.verbose:
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
        from clustertk.feature_selection import CorrelationFilter, SmartCorrelationFilter, VarianceFilter

        if self.verbose:
            print("\n🎯 Selecting features...")

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
                filter_type = "Smart" if self.smart_correlation else "Basic"
                print(f"  Removing highly correlated features (threshold: {self.correlation_threshold}, {filter_type} filter)...")

            if self.smart_correlation:
                self._correlation_filter = SmartCorrelationFilter(
                    threshold=self.correlation_threshold,
                    method='pearson',
                    selection_strategy=self.correlation_strategy,
                    verbose=self.verbose
                )
            else:
                self._correlation_filter = CorrelationFilter(
                    threshold=self.correlation_threshold,
                    method='pearson'
                )
            data_working = self._correlation_filter.fit_transform(data_working)

            removed_count = len(self._correlation_filter.features_to_drop_)
            if self.verbose and removed_count > 0:
                print(f"    ✓ Removed {removed_count} highly correlated features")
                if hasattr(self._correlation_filter, 'selection_reasons_') and self._correlation_filter.selection_reasons_:
                    print(f"    Smart selection: kept features with better clusterability ({self.correlation_strategy})")
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
        Reduce dimensionality using PCA, UMAP, or no reduction.

        The method used is determined by dim_reduction parameter:
        - 'auto': Automatic selection based on algorithm and number of features
        - 'pca': Principal Component Analysis (linear, preserves global structure)
        - 'umap': Uniform Manifold Approximation (non-linear, preserves local structure)
        - 'none': No dimensionality reduction

        Returns
        -------
        self : ClusterAnalysisPipeline
        """
        initial_dims = len(self.selected_features_)

        # Determine reduction method
        reduction_method = self._determine_reduction_method(initial_dims)

        if reduction_method == 'none':
            if self.verbose:
                print("\n📐 Skipping dimensionality reduction")
                print(f"  Keeping all {initial_dims} features")
            # Use scaled data directly
            self.data_reduced_ = self.data_scaled_[self.selected_features_].copy()
            self._reducer = None
            return self

        elif reduction_method == 'pca':
            return self._reduce_with_pca(initial_dims)

        elif reduction_method == 'umap':
            return self._reduce_with_umap(initial_dims)

        else:
            raise ValueError(f"Unknown reduction method: {reduction_method}")

    def _determine_reduction_method(self, n_features: int) -> str:
        """
        Determine which dimensionality reduction method to use.

        Parameters
        ----------
        n_features : int
            Number of features after feature selection

        Returns
        -------
        method : str
            One of 'pca', 'umap', 'none'
        """
        if self.dim_reduction != 'auto':
            return self.dim_reduction

        # Auto-mode logic
        algorithm = self.clustering_algorithm
        if isinstance(algorithm, str):
            algo_lower = algorithm.lower()
        else:
            algo_lower = algorithm.__class__.__name__.lower()

        # Density-based algorithms (DBSCAN, HDBSCAN)
        if any(name in algo_lower for name in ['dbscan', 'hdbscan']):
            if n_features > 30:
                method = 'umap'  # UMAP preserves local density
            else:
                method = 'none'  # Work in original space

        # Distance-based algorithms (K-Means, GMM, Hierarchical)
        else:
            if n_features > 50:
                method = 'pca'   # PCA for high-dimensional data
            else:
                method = 'none'  # No reduction needed

        if self.verbose:
            print(f"  Auto-selected dim_reduction='{method}' (algorithm={algorithm}, n_features={n_features})")

        return method

    def _reduce_with_pca(self, initial_dims: int) -> 'ClusterAnalysisPipeline':
        """Apply PCA reduction."""
        from clustertk.dimensionality import PCAReducer

        if self.verbose:
            print("\n📐 Reducing dimensions with PCA...")

        self._reducer = PCAReducer(
            variance_threshold=self.pca_variance,
            min_components=self.pca_min_components,
            random_state=self.random_state
        )

        self.data_reduced_ = self._reducer.fit_transform(
            self.data_scaled_[self.selected_features_]
        )

        final_dims = self.data_reduced_.shape[1]
        variance_explained = self._reducer.cumulative_variance_[final_dims - 1]

        if self.verbose:
            print(f"  Dimensionality: {initial_dims} → {final_dims} components")
            print(f"  Variance explained: {variance_explained:.1%}")

            # Show top loadings for first 2 components
            if final_dims >= 2:
                print(f"  Top features in PC1:")
                loadings = self._reducer.get_loadings(n_top=3)
                for feature, loading in loadings['PC1']:
                    sign = '+' if loading > 0 else ''
                    print(f"    {sign}{loading:.3f} {feature}")

            print("  ✓ Dimensionality reduction completed")

        return self

    def _reduce_with_umap(self, initial_dims: int) -> 'ClusterAnalysisPipeline':
        """Apply UMAP reduction."""
        try:
            import umap
        except ImportError:
            raise ImportError(
                "UMAP not installed. Install with: pip install umap-learn\n"
                "Or use dim_reduction='pca' or dim_reduction='none' instead."
            )

        if self.verbose:
            print("\n📐 Reducing dimensions with UMAP...")
            print(f"  Parameters: n_neighbors={self.umap_n_neighbors}, "
                  f"n_components={self.umap_n_components}, "
                  f"min_dist={self.umap_min_dist}")

        self._reducer = umap.UMAP(
            n_neighbors=self.umap_n_neighbors,
            min_dist=self.umap_min_dist,
            n_components=self.umap_n_components,
            metric=self.umap_metric,
            random_state=self.random_state,
            verbose=False
        )

        embedding = self._reducer.fit_transform(
            self.data_scaled_[self.selected_features_].values
        )

        # Convert to DataFrame with proper column names
        self.data_reduced_ = pd.DataFrame(
            embedding,
            columns=[f'UMAP{i+1}' for i in range(embedding.shape[1])],
            index=self.data_scaled_.index
        )

        final_dims = self.data_reduced_.shape[1]

        if self.verbose:
            print(f"  Dimensionality: {initial_dims} → {final_dims} components")
            print("  ⚠️  UMAP is non-linear - component interpretation not available")
            print("  ✓ Dimensionality reduction completed")

        return self

    def find_optimal_clusters(self) -> 'ClusterAnalysisPipeline':
        """
        Find optimal number of clusters if not specified.

        Note: This step is skipped for DBSCAN as it doesn't require
        specifying the number of clusters.

        Returns
        -------
        self : ClusterAnalysisPipeline
        """
        from clustertk.evaluation import OptimalKFinder
        from clustertk.clustering import KMeansClustering, GMMClustering, HierarchicalClustering

        # Skip for DBSCAN and HDBSCAN
        if isinstance(self.clustering_algorithm, str) and self.clustering_algorithm in ['dbscan', 'hdbscan']:
            if self.verbose:
                print("\n🔍 Finding optimal number of clusters...")
                algo_name = self.clustering_algorithm.upper()
                print(f"  Skipping for {algo_name} (density-based clustering)")
                print("  ✓ Optimal cluster finding completed")
            return self

        if self.verbose:
            print("\n🔍 Finding optimal number of clusters...")
            print(f"  Testing k in range {self.n_clusters_range}...")

        # Select clusterer class based on algorithm
        if isinstance(self.clustering_algorithm, str):
            if self.clustering_algorithm == 'kmeans':
                clusterer_class = KMeansClustering
            elif self.clustering_algorithm == 'gmm':
                clusterer_class = GMMClustering
            elif self.clustering_algorithm == 'hierarchical':
                clusterer_class = HierarchicalClustering
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
        from clustertk.clustering import (
            KMeansClustering,
            GMMClustering,
            HierarchicalClustering,
            DBSCANClustering,
            HDBSCANClustering
        )
        from clustertk.evaluation import compute_clustering_metrics

        if self.verbose:
            print("\n🔬 Performing clustering...")

        # Determine algorithm
        algo = algorithm or self.clustering_algorithm

        # Determine number of clusters (not used for DBSCAN/HDBSCAN)
        if isinstance(algo, str) and algo not in ['dbscan', 'hdbscan']:
            if n_clusters is not None:
                self.n_clusters_ = n_clusters
            elif self.n_clusters_ is None:
                if isinstance(self.n_clusters, int):
                    self.n_clusters_ = self.n_clusters
                elif isinstance(self.n_clusters, list):
                    self.n_clusters_ = self.n_clusters[0]
                else:
                    self.n_clusters_ = 3

        # Initialize clusterer
        if isinstance(algo, str):
            if algo == 'kmeans':
                # Merge default params with user params
                params = {
                    'n_clusters': self.n_clusters_,
                    'random_state': self.random_state,
                    **self.clustering_params
                }
                self._clusterer = KMeansClustering(**params)

            elif algo == 'gmm':
                params = {
                    'n_clusters': self.n_clusters_,
                    'random_state': self.random_state,
                    **self.clustering_params
                }
                self._clusterer = GMMClustering(**params)

            elif algo == 'hierarchical':
                params = {
                    'n_clusters': self.n_clusters_,
                    'linkage': 'ward',
                    'metric': 'euclidean',
                    **self.clustering_params
                }
                self._clusterer = HierarchicalClustering(**params)

            elif algo == 'dbscan':
                params = {
                    'eps': 'auto',
                    'min_samples': 'auto',
                    **self.clustering_params
                }
                self._clusterer = DBSCANClustering(**params)

            elif algo == 'hdbscan':
                params = {
                    'min_cluster_size': 'auto',
                    'min_samples': 'auto',
                    **self.clustering_params
                }
                self._clusterer = HDBSCANClustering(**params)

            else:
                raise ValueError(
                    f"Unknown algorithm '{algo}'. "
                    f"Use 'kmeans', 'gmm', 'hierarchical', 'dbscan', or 'hdbscan'."
                )
        else:
            # Custom clusterer provided
            self._clusterer = algo

        # Fit clustering
        self.labels_ = self._clusterer.fit_predict(self.data_reduced_)
        self.model_ = self._clusterer

        # Update n_clusters_ from actual result (important for DBSCAN)
        self.n_clusters_ = self._clusterer.n_clusters_

        # Compute metrics
        self.metrics_ = compute_clustering_metrics(self.data_reduced_, self.labels_)

        if self.verbose:
            print(f"  Algorithm: {algo}")
            print(f"  Number of clusters: {self.n_clusters_}")

            # Show cluster sizes (handle noise for DBSCAN)
            cluster_sizes = dict(pd.Series(self.labels_).value_counts().sort_index())
            if -1 in cluster_sizes:
                noise_count = cluster_sizes.pop(-1)
                print(f"  Cluster sizes: {cluster_sizes}")
                print(f"  Noise points: {noise_count}")
            else:
                print(f"  Cluster sizes: {cluster_sizes}")

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
            print("\n📊 Creating cluster profiles...")

        # Create profiler
        self._profiler = ClusterProfiler(normalize_per_feature=True)

        # Create profiles using scaled data
        self.cluster_profiles_ = self._profiler.create_profiles(
            self.data_scaled_[self.selected_features_],
            self.labels_,
            feature_names=self.selected_features_
        )

        # Store normalized profiles for visualization
        self.cluster_profiles_normalized_ = self._profiler.profiles_normalized_

        # Pre-calculate top distinguishing features for caching (internal optimization)
        self._profiler.get_top_features(n=10)

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

        # Generate cluster names if enabled
        if self.auto_name_clusters:
            if self.verbose:
                print(f"  Generating cluster names...")

            self.name_clusters(
                category_mapping=category_mapping,
                max_features=self.naming_max_features
            )

            if self.verbose:
                print("  ✓ Cluster naming completed")

        if self.verbose:
            print("  ✓ Profiling completed")

        return self

    def name_clusters(
        self,
        category_mapping: Optional[Dict[str, List[str]]] = None,
        naming_strategy: str = 'auto',
        max_features: Optional[int] = None,
        min_deviation: float = 0.5
    ) -> Dict[int, str]:
        """
        Generate descriptive names for clusters.

        Parameters
        ----------
        category_mapping : dict, optional
            Mapping of category names to feature names.
            Used for category-based naming if available.

        naming_strategy : str, default='auto'
            Naming strategy: 'auto', 'top_features', 'categories', or 'combined'.

        max_features : int, optional
            Maximum number of features to include in names.
            If None, uses self.naming_max_features.

        min_deviation : float, default=0.5
            Minimum deviation from mean to consider a feature significant.

        Returns
        -------
        names : dict
            Dictionary mapping cluster ID to generated name.
        """
        from clustertk.interpretation import ClusterNamer

        if self._profiler is None or self.cluster_profiles_ is None:
            raise ValueError(
                "Must call create_profiles() before generating names"
            )

        # Get category scores if category mapping was used
        category_scores = None
        if hasattr(self._profiler, 'category_scores_') and self._profiler.category_scores_ is not None:
            category_scores = self._profiler.category_scores_

        # Create namer
        self._namer = ClusterNamer(
            naming_strategy=naming_strategy,
            max_features=max_features or self.naming_max_features,
            min_deviation=min_deviation,
            use_directions=True,
            short_names=False
        )

        # Generate names
        self.cluster_names_ = self._namer.generate_names(
            profiles=self.cluster_profiles_,
            top_features=self._profiler.top_features_,
            category_scores=category_scores,
            category_mapping=category_mapping
        )

        if self.verbose:
            print("\n  Generated cluster names:")
            for cluster_id in sorted(self.cluster_names_.keys()):
                name = self.cluster_names_[cluster_id]
                print(f"    Cluster {cluster_id}: {name}")

        return self.cluster_names_

    def get_cluster_name(self, cluster_id: int) -> Optional[str]:
        """
        Get the name for a specific cluster.

        Parameters
        ----------
        cluster_id : int
            Cluster ID.

        Returns
        -------
        name : str or None
            Cluster name, or None if naming not performed or cluster not found.
        """
        if self.cluster_names_ is None:
            return None

        return self.cluster_names_.get(cluster_id)

    def analyze_feature_importance(
        self,
        method: Literal['permutation', 'shap', 'contribution', 'all'] = 'all',
        n_repeats: int = 10,
        random_state: Optional[int] = 42
    ) -> Dict[str, pd.DataFrame]:
        """
        Analyze feature importance for clustering results.

        Provides multiple methods for understanding which features
        are most important for cluster formation and separation.

        Parameters
        ----------
        method : {'permutation', 'shap', 'contribution', 'all'}, default='all'
            Which analysis method(s) to use:
            - 'permutation': Permutation importance (how much each feature affects clustering quality)
            - 'shap': SHAP values (requires shap package: pip install shap)
            - 'contribution': Feature contribution to cluster separation
            - 'all': Run all available methods
        n_repeats : int, default=10
            Number of permutation repeats (for permutation method).
        random_state : int or None, default=42
            Random state for reproducibility.

        Returns
        -------
        results : dict
            Dictionary with analysis results:
            - 'permutation': DataFrame with permutation importance
            - 'contribution': DataFrame with feature contribution
            - 'shap': SHAP analysis results (if method='shap' or 'all')

        Examples
        --------
        >>> pipeline = ClusterAnalysisPipeline(clustering_algorithm='kmeans', n_clusters=3)
        >>> pipeline.fit(df, feature_columns=['f1', 'f2', 'f3'])
        >>> importance = pipeline.analyze_feature_importance(method='permutation')
        >>> print(importance['permutation'].head())

        Notes
        -----
        - SHAP analysis requires the 'shap' package: pip install shap
        - Permutation importance uses silhouette score as the metric
        - Feature contribution uses variance ratio (between/within clusters)
        """
        from clustertk.interpretation import FeatureImportanceAnalyzer

        # Check prerequisites
        if self.data_scaled_ is None or self.labels_ is None:
            raise ValueError(
                "Pipeline must be fitted before analyzing feature importance. "
                "Call fit() first."
            )

        if self.verbose:
            print(f"\nAnalyzing feature importance (method={method})...")

        # Use scaled data with selected features
        X = self.data_scaled_[self.selected_features_]

        # Create analyzer
        analyzer = FeatureImportanceAnalyzer(verbose=self.verbose)

        # Run analysis
        results = analyzer.analyze(
            X=X,
            labels=self.labels_,
            method=method,
            n_repeats=n_repeats,
            random_state=random_state
        )

        # Store results
        self.feature_importance_results_ = results

        if self.verbose:
            print("✓ Feature importance analysis completed")

            # Print top features from each method
            if 'permutation' in results:
                print("\nTop 5 features (permutation importance):")
                top = results['permutation'].head(5)
                for idx, row in top.iterrows():
                    print(f"  {row['feature']}: {row['importance']:.4f} (±{row['std']:.4f})")

            if 'contribution' in results:
                print("\nTop 5 features (cluster separation):")
                top = results['contribution'].head(5)
                for idx, row in top.iterrows():
                    print(f"  {row['feature']}: {row['contribution']:.4f}")

        return results

    def analyze_stability(
        self,
        n_iterations: int = 100,
        sample_fraction: float = 0.8,
        random_state: Optional[int] = 42
    ) -> Dict:
        """
        Analyze clustering stability using bootstrap resampling.

        This method performs multiple clustering runs on bootstrap samples
        to assess the stability and reliability of the clustering results.

        Parameters
        ----------
        n_iterations : int, default=100
            Number of bootstrap iterations to perform.
        sample_fraction : float, default=0.8
            Fraction of samples to use in each bootstrap iteration.
        random_state : int or None, default=42
            Random state for reproducibility.

        Returns
        -------
        results : dict
            Dictionary containing:
            - 'overall_stability': Overall stability score (0-1)
            - 'cluster_stability': DataFrame with per-cluster stability
            - 'sample_confidence': Array with per-sample confidence scores
            - 'mean_ari': Mean Adjusted Rand Index
            - 'stable_clusters': List of stable cluster IDs
            - 'unstable_clusters': List of unstable cluster IDs
            - 'reference_labels': Reference cluster labels

        Raises
        ------
        ValueError
            If pipeline has not been fitted yet.

        Examples
        --------
        >>> pipeline.fit(df, feature_columns=['age', 'income'])
        >>> results = pipeline.analyze_stability(n_iterations=50)
        >>> print(f"Overall stability: {results['overall_stability']:.3f}")
        >>> print(results['cluster_stability'])

        Notes
        -----
        - Stability score ranges from 0 to 1 (higher = more stable)
        - Clusters with stability > 0.7 are considered stable
        - Clusters with stability < 0.5 are considered unstable
        - Sample confidence shows how consistently each point is assigned
          to the same cluster across bootstrap iterations
        """
        from clustertk.evaluation import ClusterStabilityAnalyzer

        if self.data_scaled_ is None or self.labels_ is None:
            raise ValueError("Pipeline must be fitted before analyzing stability")

        # Use the data that was actually used for clustering
        if self.data_reduced_ is not None:
            X = self.data_reduced_
        elif self.selected_features_ is not None:
            X = self.data_scaled_[self.selected_features_]
        else:
            X = self.data_scaled_

        # Create analyzer
        analyzer = ClusterStabilityAnalyzer(
            n_iterations=n_iterations,
            sample_fraction=sample_fraction,
            random_state=random_state,
            verbose=self.verbose
        )

        # Run stability analysis
        results = analyzer.analyze(X, self._clusterer)

        # Store results
        self.stability_results_ = results
        self.stability_analyzer_ = analyzer

        if self.verbose:
            print("\n=== Stability Analysis Results ===")
            print(f"Overall stability: {results['overall_stability']:.3f}")
            print(f"Mean ARI: {results['mean_ari']:.3f}")
            print(f"\nCluster stability:")
            print(results['cluster_stability'].to_string(index=False))

            if results['unstable_clusters']:
                print(f"\n⚠ Warning: {len(results['unstable_clusters'])} unstable clusters detected")
                print(f"  Cluster IDs: {results['unstable_clusters']}")

            # Sample confidence statistics
            conf = results['sample_confidence']
            print(f"\nSample confidence scores:")
            print(f"  Mean: {np.mean(conf):.3f}")
            print(f"  Median: {np.median(conf):.3f}")
            print(f"  Min: {np.min(conf):.3f}")
            print(f"  Max: {np.max(conf):.3f}")

            unstable_samples = np.sum(conf < 0.5)
            if unstable_samples > 0:
                pct = unstable_samples / len(conf) * 100
                print(f"  Unstable samples (conf < 0.5): {unstable_samples} ({pct:.1f}%)")

        return results

    def print_cluster_summary(self) -> None:
        """
        Print a comprehensive summary of all clusters with names and profiles.
        """
        if self.cluster_profiles_ is None:
            raise ValueError("Must call create_profiles() first")

        print("\n" + "=" * 80)
        print("CLUSTER SUMMARY")
        print("=" * 80)

        cluster_sizes = pd.Series(self.labels_).value_counts().sort_index()

        # Print noise statistics if present (v0.12.0+)
        if hasattr(self._profiler, 'n_noise_') and self._profiler.n_noise_ > 0:
            noise_count = self._profiler.n_noise_
            noise_ratio = self._profiler.noise_ratio_
            print(f"\n⚠ Noise points: {noise_count} ({noise_ratio:.2%})")
            print("  (Outliers not assigned to any cluster)")

        for cluster_id in sorted(self.cluster_profiles_.index):
            # Cluster header
            size = cluster_sizes.get(cluster_id, 0)
            pct = size / len(self.labels_) * 100

            print(f"\nCluster {cluster_id} ({size} samples, {pct:.1f}%)")

            # Name if available
            if self.cluster_names_ is not None and cluster_id in self.cluster_names_:
                print(f"  Name: {self.cluster_names_[cluster_id]}")

            # Top features
            if hasattr(self._profiler, 'top_features_') and self._profiler.top_features_:
                top_feats = self._profiler.top_features_.get(cluster_id, {})
                high_feats = top_feats.get('high', [])[:3]
                low_feats = top_feats.get('low', [])[:3]

                if high_feats:
                    print("  Top features:")
                    for feat, val in high_feats:
                        print(f"    ↑ {feat}: {val:+.3f}")

                if low_feats:
                    for feat, val in low_feats:
                        print(f"    ↓ {feat}: {val:+.3f}")

        print("\n" + "=" * 80)

    # ============================================================
    # Visualization Methods
    # ============================================================

    def plot_clusters_2d(
        self,
        method: str = 'tsne',
        title: Optional[str] = None,
        figsize: tuple = (10, 8),
        show_centers: bool = False,
        ax: Optional[Any] = None
    ):
        """
        Plot clusters in 2D space using dimensionality reduction.

        Parameters
        ----------
        method : str, default='tsne'
            Dimensionality reduction method: 'tsne', 'umap', or 'pca'.

        title : str, optional
            Plot title.

        figsize : tuple, default=(10, 8)
            Figure size.

        show_centers : bool, default=False
            Whether to show cluster centers (if available).

        ax : matplotlib.axes.Axes, optional
            Axes to plot on.

        Returns
        -------
        fig : matplotlib.figure.Figure
            The figure object.
        """
        from clustertk.visualization import check_viz_available, plot_clusters_2d

        if not check_viz_available():
            raise ImportError(
                "Visualization dependencies not installed. "
                "Install with: pip install clustertk[viz]"
            )

        if self.labels_ is None:
            raise ValueError("No clustering results available. Run fit() or cluster() first.")

        # Get cluster centers if available
        centers = None
        if show_centers and hasattr(self._clusterer, 'cluster_centers_'):
            centers = self._clusterer.cluster_centers_

        return plot_clusters_2d(
            X=self.data_reduced_,
            labels=self.labels_,
            method=method,
            title=title,
            figsize=figsize,
            show_centers=show_centers,
            centers=centers,
            ax=ax
        )

    def plot_cluster_sizes(
        self,
        title: Optional[str] = None,
        figsize: tuple = (10, 6),
        ax: Optional[Any] = None
    ):
        """
        Plot cluster size distribution as a bar chart.

        Parameters
        ----------
        title : str, optional
            Plot title.

        figsize : tuple, default=(10, 6)
            Figure size.

        ax : matplotlib.axes.Axes, optional
            Axes to plot on.

        Returns
        -------
        fig : matplotlib.figure.Figure
            The figure object.
        """
        from clustertk.visualization import check_viz_available, plot_cluster_sizes

        if not check_viz_available():
            raise ImportError(
                "Visualization dependencies not installed. "
                "Install with: pip install clustertk[viz]"
            )

        if self.labels_ is None:
            raise ValueError("No clustering results available. Run fit() or cluster() first.")

        return plot_cluster_sizes(
            labels=self.labels_,
            title=title,
            figsize=figsize,
            ax=ax
        )

    def plot_cluster_heatmap(
        self,
        title: Optional[str] = None,
        figsize: tuple = (12, 8),
        cmap: str = 'RdYlGn',
        normalize: bool = True,
        ax: Optional[Any] = None
    ):
        """
        Plot cluster profiles as a heatmap.

        Parameters
        ----------
        title : str, optional
            Plot title.

        figsize : tuple, default=(12, 8)
            Figure size.

        cmap : str, default='RdYlGn'
            Colormap name.

        normalize : bool, default=True
            Whether to normalize each feature to 0-1 scale.

        ax : matplotlib.axes.Axes, optional
            Axes to plot on.

        Returns
        -------
        fig : matplotlib.figure.Figure
            The figure object.
        """
        from clustertk.visualization import check_viz_available, plot_cluster_heatmap

        if not check_viz_available():
            raise ImportError(
                "Visualization dependencies not installed. "
                "Install with: pip install clustertk[viz]"
            )

        if self.cluster_profiles_ is None:
            raise ValueError("No cluster profiles available. Run fit() or create_profiles() first.")

        # Use pre-normalized profiles if available and normalization is requested
        if normalize and hasattr(self, 'cluster_profiles_normalized_') and self.cluster_profiles_normalized_ is not None:
            profiles_to_plot = self.cluster_profiles_normalized_
            apply_normalization = False  # Already normalized
        else:
            profiles_to_plot = self.cluster_profiles_
            apply_normalization = normalize

        return plot_cluster_heatmap(
            profiles=profiles_to_plot,
            title=title,
            figsize=figsize,
            cmap=cmap,
            normalize=apply_normalization,
            ax=ax
        )

    def plot_cluster_radar(
        self,
        cluster_ids: Optional[List[int]] = None,
        title: Optional[str] = None,
        figsize: tuple = (10, 10),
        normalize: bool = True
    ):
        """
        Plot cluster profiles as radar charts.

        Parameters
        ----------
        cluster_ids : list of int, optional
            Specific cluster IDs to plot. If None, plots all clusters.

        title : str, optional
            Plot title.

        figsize : tuple, default=(10, 10)
            Figure size.

        normalize : bool, default=True
            Whether to normalize each feature to 0-1 scale.

        Returns
        -------
        fig : matplotlib.figure.Figure
            The figure object.
        """
        from clustertk.visualization import check_viz_available, plot_cluster_radar

        if not check_viz_available():
            raise ImportError(
                "Visualization dependencies not installed. "
                "Install with: pip install clustertk[viz]"
            )

        if self.cluster_profiles_ is None:
            raise ValueError("No cluster profiles available. Run fit() or create_profiles() first.")

        # Use pre-normalized profiles if available and normalization is requested
        if normalize and hasattr(self, 'cluster_profiles_normalized_') and self.cluster_profiles_normalized_ is not None:
            profiles_to_plot = self.cluster_profiles_normalized_
            apply_normalization = False  # Already normalized
        else:
            profiles_to_plot = self.cluster_profiles_
            apply_normalization = normalize

        return plot_cluster_radar(
            profiles=profiles_to_plot,
            cluster_ids=cluster_ids,
            title=title,
            figsize=figsize,
            normalize=apply_normalization
        )

    def plot_feature_importance(
        self,
        cluster_id: int,
        n_features: int = 10,
        title: Optional[str] = None,
        figsize: tuple = (10, 6),
        ax: Optional[Any] = None
    ):
        """
        Plot top distinguishing features for a cluster.

        Parameters
        ----------
        cluster_id : int
            Cluster ID to visualize.

        n_features : int, default=10
            Number of top features to show.

        title : str, optional
            Plot title.

        figsize : tuple, default=(10, 6)
            Figure size.

        ax : matplotlib.axes.Axes, optional
            Axes to plot on.

        Returns
        -------
        fig : matplotlib.figure.Figure
            The figure object.
        """
        from clustertk.visualization import check_viz_available, plot_feature_importance

        if not check_viz_available():
            raise ImportError(
                "Visualization dependencies not installed. "
                "Install with: pip install clustertk[viz]"
            )

        if self._profiler is None or not hasattr(self._profiler, 'top_features_'):
            raise ValueError(
                "No feature importance data available. "
                "Run fit() or create_profiles() first."
            )

        return plot_feature_importance(
            top_features=self._profiler.top_features_,
            cluster_id=cluster_id,
            n_features=n_features,
            title=title,
            figsize=figsize,
            ax=ax
        )

    def plot_pca_variance(
        self,
        threshold: Optional[float] = None,
        title: Optional[str] = None,
        figsize: tuple = (12, 6),
        ax: Optional[Any] = None
    ):
        """
        Plot PCA explained variance (scree plot).

        Parameters
        ----------
        threshold : float, optional
            Variance threshold line to show (e.g., 0.9 for 90%).

        title : str, optional
            Plot title.

        figsize : tuple, default=(12, 6)
            Figure size.

        ax : matplotlib.axes.Axes, optional
            Axes to plot on.

        Returns
        -------
        fig : matplotlib.figure.Figure
            The figure object.
        """
        from clustertk.visualization import check_viz_available, plot_pca_variance

        if not check_viz_available():
            raise ImportError(
                "Visualization dependencies not installed. "
                "Install with: pip install clustertk[viz]"
            )

        if self._reducer is None or not hasattr(self._reducer, 'explained_variance_'):
            raise ValueError(
                "No PCA results available. Either:\n"
                "  1. Run fit() or reduce_dimensions() first\n"
                "  2. Use dim_reduction='pca' (current: dim_reduction='{}')".format(self.dim_reduction)
            )

        return plot_pca_variance(
            explained_variance_ratio=self._reducer.explained_variance_,
            cumulative_variance=self._reducer.cumulative_variance_,
            threshold=threshold or self.pca_variance,
            title=title,
            figsize=figsize,
            ax=ax
        )

    def plot_pca_loadings(
        self,
        components: Optional[list] = None,
        n_features: int = 10,
        title: Optional[str] = None,
        figsize: tuple = (12, 8)
    ):
        """
        Plot PCA component loadings.

        Parameters
        ----------
        components : list, optional
            Which components to plot (e.g., [0, 1, 2] for first 3 PCs).
            If None, plots all components.

        n_features : int, default=10
            Number of top features to show per component.

        title : str, optional
            Plot title.

        figsize : tuple, default=(12, 8)
            Figure size.

        Returns
        -------
        fig : matplotlib.figure.Figure
            The figure object.
        """
        from clustertk.visualization import check_viz_available, plot_pca_loadings

        if not check_viz_available():
            raise ImportError(
                "Visualization dependencies not installed. "
                "Install with: pip install clustertk[viz]"
            )

        if self._reducer is None or not hasattr(self._reducer, 'loadings_'):
            raise ValueError(
                "No PCA results available. Either:\n"
                "  1. Run fit() or reduce_dimensions() first\n"
                "  2. Use dim_reduction='pca' (current: dim_reduction='{}')".format(self.dim_reduction)
            )

        # Get loadings as DataFrame
        loadings_df = pd.DataFrame(
            self._reducer.loadings_,
            columns=[f'PC{i+1}' for i in range(self._reducer.loadings_.shape[1])],
            index=self.selected_features_
        )

        return plot_pca_loadings(
            loadings=loadings_df,
            components=components,
            n_features=n_features,
            title=title,
            figsize=figsize
        )

    def plot_correlation_matrix(
        self,
        method: str = 'pearson',
        title: Optional[str] = None,
        figsize: tuple = (12, 10),
        cmap: str = 'coolwarm',
        annot: bool = False,
        threshold: Optional[float] = None,
        ax: Optional[Any] = None
    ):
        """
        Plot correlation matrix of features.

        Parameters
        ----------
        method : str, default='pearson'
            Correlation method: 'pearson', 'spearman', or 'kendall'.

        title : str, optional
            Plot title.

        figsize : tuple, default=(12, 10)
            Figure size.

        cmap : str, default='coolwarm'
            Colormap name.

        annot : bool, default=False
            Whether to annotate cells with correlation values.

        threshold : float, optional
            If provided, only shows correlations with |r| > threshold.

        ax : matplotlib.axes.Axes, optional
            Axes to plot on.

        Returns
        -------
        fig : matplotlib.figure.Figure
            The figure object.
        """
        from clustertk.visualization import check_viz_available, plot_correlation_matrix

        if not check_viz_available():
            raise ImportError(
                "Visualization dependencies not installed. "
                "Install with: pip install clustertk[viz]"
            )

        if self.data_scaled_ is None:
            raise ValueError("No data available. Run fit() or preprocess() first.")

        return plot_correlation_matrix(
            data=self.data_scaled_[self.selected_features_] if self.selected_features_
                 else self.data_scaled_,
            method=method,
            title=title,
            figsize=figsize,
            cmap=cmap,
            annot=annot,
            threshold=threshold,
            ax=ax
        )

    def plot_correlation_network(
        self,
        threshold: float = 0.5,
        method: str = 'pearson',
        title: Optional[str] = None,
        figsize: tuple = (12, 12),
        node_size: int = 1000,
        font_size: int = 10
    ):
        """
        Plot correlation network graph.

        Parameters
        ----------
        threshold : float, default=0.5
            Minimum absolute correlation to show.

        method : str, default='pearson'
            Correlation method.

        title : str, optional
            Plot title.

        figsize : tuple, default=(12, 12)
            Figure size.

        node_size : int, default=1000
            Size of nodes.

        font_size : int, default=10
            Font size for labels.

        Returns
        -------
        fig : matplotlib.figure.Figure
            The figure object.
        """
        from clustertk.visualization import check_viz_available, plot_correlation_network

        if not check_viz_available():
            raise ImportError(
                "Visualization dependencies not installed. "
                "Install with: pip install clustertk[viz]"
            )

        if self.data_scaled_ is None:
            raise ValueError("No data available. Run fit() or preprocess() first.")

        return plot_correlation_network(
            data=self.data_scaled_[self.selected_features_] if self.selected_features_
                 else self.data_scaled_,
            threshold=threshold,
            method=method,
            title=title,
            figsize=figsize,
            node_size=node_size,
            font_size=font_size
        )

    def plot_feature_distributions(
        self,
        features: Optional[List[str]] = None,
        n_cols: int = 3,
        figsize: Optional[tuple] = None,
        bins: int = 30
    ):
        """
        Plot distribution histograms for features.

        Parameters
        ----------
        features : list of str, optional
            Features to plot. If None, plots all selected features.

        n_cols : int, default=3
            Number of columns in subplot grid.

        figsize : tuple, optional
            Figure size. If None, auto-calculated.

        bins : int, default=30
            Number of histogram bins.

        Returns
        -------
        fig : matplotlib.figure.Figure
            The figure object.
        """
        from clustertk.visualization import check_viz_available, plot_feature_distributions

        if not check_viz_available():
            raise ImportError(
                "Visualization dependencies not installed. "
                "Install with: pip install clustertk[viz]"
            )

        if self.data_preprocessed_ is None:
            raise ValueError("No data available. Run fit() or preprocess() first.")

        # Use selected features if available, otherwise all features
        data_to_plot = (self.data_preprocessed_[self.selected_features_]
                       if self.selected_features_ else self.data_preprocessed_)

        return plot_feature_distributions(
            data=data_to_plot,
            features=features,
            n_cols=n_cols,
            figsize=figsize,
            bins=bins
        )

    def export_results(
        self,
        path: str,
        format: str = 'csv',
        include_original: bool = True,
        include_profiles: bool = True,
        include_metrics: bool = True
    ) -> None:
        """
        Export clustering results to file.

        Parameters
        ----------
        path : str
            Output file path.

        format : str, default='csv'
            Output format: 'csv' or 'json'.

        include_original : bool, default=True
            Whether to include original data columns in CSV export.

        include_profiles : bool, default=True
            Whether to include cluster profiles in JSON export.

        include_metrics : bool, default=True
            Whether to include clustering metrics in JSON export.

        Examples
        --------
        >>> pipeline.export_results('results.csv', format='csv')
        >>> pipeline.export_results('results.json', format='json')
        """
        if self.labels_ is None:
            raise ValueError("No clustering results available. Run fit() or cluster() first.")

        if self.verbose:
            print(f"\nExporting results to {path} (format: {format})...")

        if format == 'csv':
            self._export_csv(path, include_original)
        elif format == 'json':
            self._export_json(path, include_profiles, include_metrics)
        else:
            raise ValueError(f"Unsupported format '{format}'. Use 'csv' or 'json'.")

        if self.verbose:
            print(f"  ✓ Results exported successfully")

    def _export_csv(self, path: str, include_original: bool = True) -> None:
        """Export results to CSV file."""
        # Create results dataframe
        if include_original and self.data_ is not None:
            # Include original data
            results_df = self.data_.copy()
        else:
            # Just create index
            results_df = pd.DataFrame(index=range(len(self.labels_)))

        # Add cluster labels
        results_df['cluster'] = self.labels_

        # Add cluster names if available
        if self.cluster_names_ is not None:
            results_df['cluster_name'] = results_df['cluster'].map(self.cluster_names_)

        # Save to CSV
        results_df.to_csv(path, index=False)

        if self.verbose:
            print(f"    Exported {len(results_df)} samples with {len(results_df.columns)} columns")

    def _export_json(
        self,
        path: str,
        include_profiles: bool = True,
        include_metrics: bool = True
    ) -> None:
        """Export results to JSON file."""
        import json

        # Build export data
        export_data: Dict[str, Any] = {
            'n_clusters': int(self.n_clusters_),
            'n_samples': int(len(self.labels_)),
            'algorithm': str(self.clustering_algorithm),
        }

        # Add cluster labels
        export_data['labels'] = self.labels_.tolist()

        # Add cluster names if available
        if self.cluster_names_ is not None:
            export_data['cluster_names'] = self.cluster_names_

        # Add cluster sizes
        cluster_sizes = pd.Series(self.labels_).value_counts().sort_index()
        export_data['cluster_sizes'] = cluster_sizes.to_dict()

        # Add profiles if requested
        if include_profiles and self.cluster_profiles_ is not None:
            # Convert profiles to dict
            profiles_dict = {}
            for cluster_id in self.cluster_profiles_.index:
                profiles_dict[int(cluster_id)] = self.cluster_profiles_.loc[cluster_id].to_dict()
            export_data['cluster_profiles'] = profiles_dict

        # Add metrics if requested
        if include_metrics and self.metrics_ is not None:
            export_data['metrics'] = self.metrics_

        # Add pipeline configuration
        export_data['config'] = {
            'handle_missing': str(self.handle_missing),
            'handle_outliers': str(self.handle_outliers),
            'scaling': str(self.scaling),
            'correlation_threshold': float(self.correlation_threshold),
            'variance_threshold': float(self.variance_threshold),
            'pca_variance': float(self.pca_variance),
            'n_clusters_range': self.n_clusters_range,
            'random_state': int(self.random_state),
        }

        # Add feature information
        if self.selected_features_ is not None:
            export_data['selected_features'] = self.selected_features_

        # Save to JSON
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)

        if self.verbose:
            print(f"    Exported {self.n_clusters_} clusters with detailed metadata")

    def export_report(
        self,
        path: str,
        include_plots: bool = True,
        plot_format: str = 'png',
        insights_top_features: int = 5,
        radar_max_features: int = 10
    ) -> None:
        """
        Generate and export an HTML report with clustering results.

        This method creates a comprehensive HTML report including:
        - Clustering summary and metrics
        - Cluster insights with top distinguishing features
        - Complete cluster profiles table (sorted by feature importance)
        - Embedded visualizations (if include_plots=True)
        - Pipeline configuration details

        Parameters
        ----------
        path : str
            Output HTML file path.

        include_plots : bool, default=True
            Whether to include embedded plots.
            Requires visualization dependencies (pip install clustertk[viz]).

        plot_format : str, default='png'
            Format for embedded plots: 'png' or 'svg'.

        insights_top_features : int, default=5
            Number of top distinguishing features to show per cluster in the
            Cluster Insights section. Shows the most different features from
            the overall mean for EACH cluster separately.

        radar_max_features : int, default=10
            Maximum number of features to show on the radar chart. Features are
            aggregated from top distinguishing features across ALL clusters.

        Examples
        --------
        >>> # Default report
        >>> pipeline.export_report('report.html')

        >>> # Concise report with fewer features
        >>> pipeline.export_report('report.html', insights_top_features=3, radar_max_features=6)

        >>> # Detailed report with more features
        >>> pipeline.export_report('report.html', insights_top_features=10, radar_max_features=15)

        >>> # Report without plots
        >>> pipeline.export_report('report.html', include_plots=False)
        """
        if self.labels_ is None:
            raise ValueError("No clustering results available. Run fit() or cluster() first.")

        if self.cluster_profiles_ is None:
            raise ValueError("No cluster profiles available. Run fit() or create_profiles() first.")

        if self.verbose:
            print(f"\nGenerating HTML report: {path}...")

        # Build HTML report
        html = self._build_html_report(include_plots, plot_format, insights_top_features, radar_max_features)

        # Save to file
        with open(path, 'w', encoding='utf-8') as f:
            f.write(html)

        if self.verbose:
            print(f"  ✓ Report generated successfully")

    def _add_algorithm_specific_metrics(self, html_parts: list) -> None:
        """Add algorithm-specific metrics to HTML report."""
        # Check for DBSCAN/HDBSCAN noise points
        if 'n_noise' in self.metrics_ and self.metrics_['n_noise'] > 0:
            html_parts.append('<div class="algo-metrics">')
            html_parts.append('<h3>🔍 Density-Based Algorithm Metrics</h3>')

            noise_count = self.metrics_['n_noise']
            noise_ratio = self.metrics_['noise_ratio']
            html_parts.append(f'<p><strong>Noise Points:</strong> {noise_count} samples ({noise_ratio:.1%} of total)</p>')
            html_parts.append('<p><em>Noise points are samples that don\'t belong to any cluster (marked as -1). '
                            'This is normal for DBSCAN/HDBSCAN algorithms.</em></p>')

            html_parts.append('</div>')

        # Check for HDBSCAN probabilities
        if hasattr(self.model_, 'probabilities_') and self.model_.probabilities_ is not None:
            avg_prob = float(np.mean(self.model_.probabilities_[self.model_.probabilities_ > 0]))
            weak_members = np.sum(self.model_.probabilities_ < 0.5)

            html_parts.append('<div class="algo-metrics">')
            html_parts.append('<h3>🎲 HDBSCAN Membership Probabilities</h3>')
            html_parts.append(f'<p><strong>Average membership probability:</strong> {avg_prob:.3f}</p>')
            html_parts.append(f'<p><strong>Weak members (prob < 0.5):</strong> {weak_members} samples</p>')
            html_parts.append('<p><em>Higher probabilities indicate more confident cluster assignments.</em></p>')
            html_parts.append('</div>')

        # Check for GMM convergence
        if hasattr(self.model_, 'converged_') and hasattr(self.model_, 'n_iter_'):
            html_parts.append('<div class="algo-metrics">')
            html_parts.append('<h3>🔄 GMM Convergence</h3>')
            html_parts.append(f'<p><strong>Converged:</strong> {"Yes" if self.model_.converged_ else "No"}</p>')
            html_parts.append(f'<p><strong>Iterations:</strong> {self.model_.n_iter_}</p>')
            html_parts.append('</div>')

    def _build_html_report(
        self,
        include_plots: bool,
        plot_format: str,
        insights_top_features: int,
        radar_max_features: int
    ) -> str:
        """Build HTML report content with enhanced features."""
        import base64
        from io import BytesIO
        from datetime import datetime

        # HTML header and CSS (enhanced styling)
        html_parts = [
            '<!DOCTYPE html>',
            '<html lang="en">',
            '<head>',
            '<meta charset="UTF-8">',
            '<meta name="viewport" content="width=device-width, initial-scale=1.0">',
            '<title>ClusterTK Analysis Report</title>',
            '<style>',
            'body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Arial, sans-serif; margin: 20px; background-color: #f5f7fa; }',
            'h1 { color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 10px; }',
            'h2 { color: #34495e; margin-top: 40px; border-bottom: 2px solid #95a5a6; padding-bottom: 5px; }',
            'h3 { color: #7f8c8d; margin-top: 25px; }',
            'h4 { color: #95a5a6; margin-top: 15px; margin-bottom: 10px; }',
            '.container { max-width: 1400px; margin: 0 auto; background-color: white; padding: 40px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); border-radius: 8px; }',
            '.summary { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 25px; border-radius: 8px; margin: 20px 0; }',
            '.metric { display: inline-block; margin: 10px 20px 10px 0; }',
            '.metric-label { font-weight: 500; opacity: 0.9; }',
            '.metric-value { font-size: 1.3em; font-weight: bold; margin-left: 5px; }',
            'table { border-collapse: collapse; width: 100%; margin: 20px 0; font-size: 0.95em; }',
            'table.compact { font-size: 0.85em; }',
            'th { background-color: #3498db; color: white; padding: 12px 8px; text-align: left; font-weight: 600; }',
            'td { padding: 10px 8px; border-bottom: 1px solid #e1e8ed; }',
            'tr:hover { background-color: #f8f9fa; }',
            '.cluster-card { background-color: #f8f9fa; border-left: 4px solid #3498db; padding: 20px; margin: 20px 0; border-radius: 4px; }',
            '.cluster-card h4 { margin-top: 0; color: #2c3e50; }',
            '.feature-list { list-style: none; padding: 0; margin: 10px 0; }',
            '.feature-list li { padding: 8px 0; border-bottom: 1px solid #e1e8ed; display: flex; justify-content: space-between; }',
            '.feature-list li:last-child { border-bottom: none; }',
            '.feature-high { color: #27ae60; font-weight: 500; }',
            '.feature-low { color: #e74c3c; font-weight: 500; }',
            '.feature-name { flex: 1; }',
            '.feature-value { font-weight: 600; margin-left: 10px; }',
            '.insight-box { background-color: #fff3cd; border-left: 4px solid #ffc107; padding: 15px; margin: 15px 0; border-radius: 4px; }',
            '.algo-metrics { background-color: #e8f5e9; border-left: 4px solid #4caf50; padding: 15px; margin: 15px 0; border-radius: 4px; }',
            '.plot-container { margin: 30px 0; text-align: center; }',
            '.plot-container img { max-width: 100%; height: auto; border: 1px solid #dee2e6; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }',
            '.config { background-color: #f8f9fa; padding: 20px; border-left: 4px solid #6c757d; margin: 20px 0; border-radius: 4px; }',
            '.config-item { margin: 8px 0; }',
            '.timestamp { color: #6c757d; font-size: 0.9em; }',
            '.badge { display: inline-block; padding: 4px 12px; border-radius: 12px; font-size: 0.85em; font-weight: 600; margin: 0 5px; }',
            '.badge-success { background-color: #d4edda; color: #155724; }',
            '.badge-warning { background-color: #fff3cd; color: #856404; }',
            '.badge-info { background-color: #d1ecf1; color: #0c5460; }',
            '.two-column { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin: 20px 0; }',
            '@media (max-width: 768px) { .two-column { grid-template-columns: 1fr; } }',
            '.importance-col { background-color: #fff9e6; border-left: 2px solid #ffc107; border-right: 2px solid #ffc107; font-weight: 600; }',
            '.importance-header { background-color: #ffa726 !important; color: white; }',
            '</style>',
            '</head>',
            '<body>',
            '<div class="container">',
        ]

        # Title
        html_parts.append('<h1>📊 ClusterTK Analysis Report</h1>')
        html_parts.append(f'<p class="timestamp">Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>')

        # Summary section (enhanced)
        html_parts.append('<h2>🎯 Clustering Summary</h2>')
        html_parts.append('<div class="summary">')

        html_parts.append(f'<div class="metric"><span class="metric-label">Algorithm:</span> <span class="metric-value">{self.clustering_algorithm}</span></div>')
        html_parts.append(f'<div class="metric"><span class="metric-label">Clusters:</span> <span class="metric-value">{self.n_clusters_}</span></div>')
        html_parts.append(f'<div class="metric"><span class="metric-label">Samples:</span> <span class="metric-value">{len(self.labels_)}</span></div>')

        if self.metrics_:
            if 'silhouette' in self.metrics_:
                sil_score = self.metrics_["silhouette"]
                if not np.isnan(sil_score):
                    quality = "Excellent" if sil_score > 0.7 else "Good" if sil_score > 0.5 else "Fair" if sil_score > 0.25 else "Weak"
                    html_parts.append(f'<div class="metric"><span class="metric-label">Silhouette:</span> <span class="metric-value">{sil_score:.3f}</span> <span class="badge badge-info">{quality}</span></div>')
            if 'cluster_balance' in self.metrics_:
                balance = self.metrics_['cluster_balance']
                if not np.isnan(balance):
                    balance_quality = "Balanced" if balance > 0.8 else "Moderate" if balance > 0.6 else "Imbalanced"
                    html_parts.append(f'<div class="metric"><span class="metric-label">Balance:</span> <span class="metric-value">{balance:.3f}</span> <span class="badge badge-success">{balance_quality}</span></div>')
            if 'calinski_harabasz' in self.metrics_:
                ch_score = self.metrics_["calinski_harabasz"]
                if not np.isnan(ch_score):
                    html_parts.append(f'<div class="metric"><span class="metric-label">Calinski-Harabasz:</span> <span class="metric-value">{ch_score:.1f}</span></div>')
            if 'davies_bouldin' in self.metrics_:
                db_score = self.metrics_["davies_bouldin"]
                if not np.isnan(db_score):
                    html_parts.append(f'<div class="metric"><span class="metric-label">Davies-Bouldin:</span> <span class="metric-value">{db_score:.3f}</span></div>')

        html_parts.append('</div>')

        # Algorithm-specific metrics (NEW!)
        self._add_algorithm_specific_metrics(html_parts)

        # Cluster sizes
        html_parts.append('<h3>Cluster Sizes</h3>')
        cluster_sizes = pd.Series(self.labels_).value_counts().sort_index()
        html_parts.append('<table>')
        html_parts.append('<tr><th>Cluster</th><th>Name</th><th>Size</th><th>Percentage</th></tr>')

        for cluster_id, size in cluster_sizes.items():
            # Skip noise points in main table (v0.12.0+)
            if cluster_id == -1:
                continue
            pct = size / len(self.labels_) * 100
            name = self.cluster_names_.get(cluster_id, '-') if self.cluster_names_ else '-'
            html_parts.append(f'<tr><td>{cluster_id}</td><td>{name}</td><td>{size}</td><td>{pct:.1f}%</td></tr>')

        # Add noise row if present (v0.12.0+)
        if -1 in cluster_sizes:
            noise_size = cluster_sizes[-1]
            noise_pct = noise_size / len(self.labels_) * 100
            html_parts.append(f'<tr style="background-color: #fff3cd;"><td><strong>Noise</strong></td><td><em>Outliers</em></td><td>{noise_size}</td><td>{noise_pct:.1f}%</td></tr>')

        html_parts.append('</table>')

        # Cluster insights with top features
        html_parts.append('<h2>💡 Cluster Insights</h2>')
        html_parts.append('<p>Key distinguishing features for each cluster (showing most different from overall average):</p>')

        # Get top features for each cluster
        if hasattr(self, '_profiler') and self._profiler is not None:
            # Safe calculation: adjust n based on available features
            n_available_features = len(self.cluster_profiles_.columns)
            n_insights = min(insights_top_features, n_available_features)

            top_features = self._profiler.get_top_features(n=n_insights)

            for cluster_id in sorted(self.cluster_profiles_.index):
                cluster_name = self.cluster_names_.get(cluster_id, f'Cluster {cluster_id}') if self.cluster_names_ else f'Cluster {cluster_id}'

                html_parts.append('<div class="cluster-card">')
                html_parts.append(f'<h4>🎯 {cluster_name}</h4>')

                if cluster_id in top_features:
                    features = top_features[cluster_id]

                    # High features - show all n_insights features
                    html_parts.append('<strong class="feature-high">⬆ Highest Features:</strong>')
                    html_parts.append('<ul class="feature-list">')
                    for feat_name, deviation in features['high'][:n_insights]:
                        profile_value = self.cluster_profiles_.loc[cluster_id, feat_name]
                        html_parts.append(f'<li><span class="feature-name">{feat_name}</span> '
                                        f'<span class="feature-value">{profile_value:.3f}</span> '
                                        f'<span class="badge badge-success">+{deviation:.3f}</span></li>')
                    html_parts.append('</ul>')

                    # Low features - show all n_insights features
                    html_parts.append('<strong class="feature-low">⬇ Lowest Features:</strong>')
                    html_parts.append('<ul class="feature-list">')
                    for feat_name, deviation in features['low'][:n_insights]:
                        profile_value = self.cluster_profiles_.loc[cluster_id, feat_name]
                        html_parts.append(f'<li><span class="feature-name">{feat_name}</span> '
                                        f'<span class="feature-value">{profile_value:.3f}</span> '
                                        f'<span class="badge badge-warning">{deviation:.3f}</span></li>')
                    html_parts.append('</ul>')

                html_parts.append('</div>')

        # Full cluster profiles (condensed table with importance)
        n_features = len(self.cluster_profiles_.columns)
        html_parts.append(f'<h2>📋 Complete Cluster Profiles</h2>')
        html_parts.append(f'<p><em>{n_features} features × {len(self.cluster_profiles_)} clusters (sorted by importance)</em></p>')
        html_parts.append('<p><strong>Note:</strong> Values are scaled/normalized for clustering. Importance shows discriminating power (max deviation across clusters).</p>')

        # Calculate feature importance for sorting
        feature_importance = {}
        if hasattr(self, '_profiler') and self._profiler is not None:
            try:
                # Get all features importance
                top_feats_all = self._profiler.get_top_features(n=n_features)
                # Calculate max deviation across all clusters for each feature
                for cluster_features in top_feats_all.values():
                    for feat_name, deviation in cluster_features['high']:
                        if feat_name not in feature_importance:
                            feature_importance[feat_name] = 0
                        feature_importance[feat_name] = max(feature_importance[feat_name], abs(deviation))
                    for feat_name, deviation in cluster_features['low']:
                        if feat_name not in feature_importance:
                            feature_importance[feat_name] = 0
                        feature_importance[feat_name] = max(feature_importance[feat_name], abs(deviation))
            except:
                # Fallback: no sorting if importance calculation fails
                feature_importance = {f: 0 for f in self.cluster_profiles_.columns}
        else:
            # No profiler: no sorting
            feature_importance = {f: 0 for f in self.cluster_profiles_.columns}

        # Sort features by importance (descending)
        sorted_features = sorted(self.cluster_profiles_.columns,
                                key=lambda f: feature_importance.get(f, 0),
                                reverse=True)

        # Transpose table if many features (better for readability)
        if n_features > 10:
            html_parts.append('<p><strong>Note:</strong> Table transposed for readability (features as rows)</p>')
            html_parts.append('<div style="overflow-x: auto;">')
            html_parts.append('<table class="compact">')

            # Header: Feature column + Importance column + cluster columns
            header_row = '<tr><th>Feature</th><th class="importance-header">Importance</th>'
            for cluster_id in sorted(self.cluster_profiles_.index):
                cluster_name = self.cluster_names_.get(cluster_id, str(cluster_id)) if self.cluster_names_ else str(cluster_id)
                header_row += f'<th>{cluster_name}</th>'
            header_row += '</tr>'
            html_parts.append(header_row)

            # Rows: one per feature (sorted by importance)
            for feature in sorted_features:
                importance = feature_importance.get(feature, 0)
                row = f'<tr><td><strong>{feature}</strong></td><td class="importance-col">{importance:.3f}</td>'
                for cluster_id in sorted(self.cluster_profiles_.index):
                    value = self.cluster_profiles_.loc[cluster_id, feature]
                    row += f'<td>{value:.3f}</td>'
                row += '</tr>'
                html_parts.append(row)

            html_parts.append('</table>')
            html_parts.append('</div>')
        else:
            # Normal table (clusters as rows, features sorted by importance)
            html_parts.append('<div style="overflow-x: auto;">')
            html_parts.append('<table>')

            # Table header (sorted features + importance row)
            header_row = '<tr><th>Cluster</th>'
            for feature in sorted_features:
                header_row += f'<th>{feature}</th>'
            header_row += '</tr>'
            html_parts.append(header_row)

            # Importance row
            importance_row = '<tr><td><strong>Importance</strong></td>'
            for feature in sorted_features:
                importance = feature_importance.get(feature, 0)
                importance_row += f'<td class="importance-col">{importance:.3f}</td>'
            importance_row += '</tr>'
            html_parts.append(importance_row)

            # Table rows (clusters)
            for cluster_id in sorted(self.cluster_profiles_.index):
                cluster_name = self.cluster_names_.get(cluster_id, str(cluster_id)) if self.cluster_names_ else str(cluster_id)
                row = f'<tr><td><strong>{cluster_name}</strong></td>'
                for feature in sorted_features:
                    value = self.cluster_profiles_.loc[cluster_id, feature]
                    row += f'<td>{value:.3f}</td>'
                row += '</tr>'
                html_parts.append(row)

            html_parts.append('</table>')
            html_parts.append('</div>')

        # Add plots if requested
        if include_plots:
            try:
                from clustertk.visualization import check_viz_available
                if check_viz_available():
                    html_parts.append('<h2>📊 Visualizations</h2>')

                    # Get top features for radar chart (aggregated from all clusters)
                    top_features_radar = None
                    if hasattr(self, '_profiler') and self._profiler is not None:
                        try:
                            # Safe calculation: adjust n based on available features
                            n_available_features = len(self.cluster_profiles_.columns)
                            n_radar_calc = min(radar_max_features, n_available_features)

                            # Get enough features to aggregate
                            top_feats = self._profiler.get_top_features(n=n_radar_calc)
                            # Collect all unique top features across clusters
                            all_top_features = set()
                            for cluster_features in top_feats.values():
                                all_top_features.update([f[0] for f in cluster_features['high']])
                                all_top_features.update([f[0] for f in cluster_features['low']])
                            top_features_radar = list(all_top_features)[:n_radar_calc]
                        except:
                            top_features_radar = None

                    # Generate and embed plots
                    plots_to_generate = [
                        ('plot_clusters_2d', {}, '2D Cluster Visualization', None),
                        ('plot_cluster_sizes', {}, 'Cluster Size Distribution', None),
                        ('plot_cluster_heatmap', {}, f'Cluster Profiles Heatmap (Top 15 Features)', sorted_features[:15]) if n_features > 15 else ('plot_cluster_heatmap', {}, 'Cluster Profiles Heatmap', None),
                    ]

                    for plot_method, kwargs, title, feature_subset in plots_to_generate:
                        try:
                            # For heatmap with many features, temporarily slice profiles
                            if feature_subset is not None and plot_method == 'plot_cluster_heatmap':
                                original_profiles = self.cluster_profiles_
                                original_profiles_norm = self.cluster_profiles_normalized_ if hasattr(self, 'cluster_profiles_normalized_') else None
                                # Slice to top features
                                self.cluster_profiles_ = self.cluster_profiles_[feature_subset]
                                if original_profiles_norm is not None:
                                    self.cluster_profiles_normalized_ = original_profiles_norm[feature_subset]

                                fig = getattr(self, plot_method)(**kwargs)

                                # Restore original profiles
                                self.cluster_profiles_ = original_profiles
                                if original_profiles_norm is not None:
                                    self.cluster_profiles_normalized_ = original_profiles_norm
                            else:
                                fig = getattr(self, plot_method)(**kwargs)

                            # Convert plot to base64
                            buffer = BytesIO()
                            fig.savefig(buffer, format=plot_format, bbox_inches='tight', dpi=120)
                            buffer.seek(0)
                            img_base64 = base64.b64encode(buffer.read()).decode()
                            buffer.close()

                            # Embed in HTML
                            html_parts.append('<div class="plot-container">')
                            html_parts.append(f'<h3>{title}</h3>')
                            html_parts.append(f'<img src="data:image/{plot_format};base64,{img_base64}" alt="{title}">')
                            html_parts.append('</div>')

                        except Exception as e:
                            if self.verbose:
                                print(f"    Warning: Could not generate {plot_method}: {e}")

                    # Add radar chart with top features only (custom plot)
                    if top_features_radar and len(top_features_radar) > 0:
                        try:
                            from clustertk.visualization import plot_cluster_radar
                            # Create filtered profiles with only top features
                            profiles_filtered = self.cluster_profiles_[top_features_radar]
                            # Use normalized if available
                            if hasattr(self, 'cluster_profiles_normalized_') and self.cluster_profiles_normalized_ is not None:
                                profiles_filtered_norm = self.cluster_profiles_normalized_[top_features_radar]
                                fig = plot_cluster_radar(profiles=profiles_filtered_norm, normalize=False,
                                                        title=f'Key Features Radar ({len(top_features_radar)} features)')
                            else:
                                fig = plot_cluster_radar(profiles=profiles_filtered, normalize=True,
                                                        title=f'Key Features Radar ({len(top_features_radar)} features)')

                            # Convert plot to base64
                            buffer = BytesIO()
                            fig.savefig(buffer, format=plot_format, bbox_inches='tight', dpi=120)
                            buffer.seek(0)
                            img_base64 = base64.b64encode(buffer.read()).decode()
                            buffer.close()

                            # Embed in HTML
                            html_parts.append('<div class="plot-container">')
                            html_parts.append(f'<h3>Key Features Radar Chart</h3>')
                            html_parts.append(f'<p><em>Showing {len(top_features_radar)} most distinguishing features across all clusters</em></p>')
                            html_parts.append(f'<img src="data:image/{plot_format};base64,{img_base64}" alt="Key Features Radar">')
                            html_parts.append('</div>')

                        except Exception as e:
                            if self.verbose:
                                print(f"    Warning: Could not generate radar chart: {e}")

                else:
                    html_parts.append('<p><em>Visualization dependencies not installed. Install with: pip install clustertk[viz]</em></p>')

            except Exception as e:
                html_parts.append(f'<p><em>Could not generate plots: {e}</em></p>')

        # Pipeline configuration
        html_parts.append('<h2>Pipeline Configuration</h2>')
        html_parts.append('<div class="config">')

        # Determine cluster configuration display
        if self.n_clusters is not None:
            if isinstance(self.n_clusters, list):
                cluster_config = f"Tested: {self.n_clusters}, Selected: {self.n_clusters_}"
            else:
                cluster_config = f"Fixed: {self.n_clusters_}"
        else:
            cluster_config = f"Auto-selected from range {self.n_clusters_range}: {self.n_clusters_}"

        config_items = [
            ('Missing Value Strategy', self.handle_missing),
            ('Outlier Handling', self.handle_outliers),
            ('Scaling Method', self.scaling),
            ('Correlation Threshold', self.correlation_threshold),
            ('Variance Threshold', self.variance_threshold),
            ('PCA Variance', self.pca_variance),
            ('Number of Clusters', cluster_config),
            ('Random State', self.random_state),
        ]

        for label, value in config_items:
            html_parts.append(f'<div class="config-item"><strong>{label}:</strong> {value}</div>')

        if self.selected_features_:
            html_parts.append(f'<div class="config-item"><strong>Selected Features:</strong> {len(self.selected_features_)} features</div>')

        html_parts.append('</div>')

        # Footer
        html_parts.append('<hr>')
        html_parts.append('<p style="text-align: center; color: #95a5a6;">Generated with <strong>ClusterTK</strong></p>')
        html_parts.append('</div>')
        html_parts.append('</body>')
        html_parts.append('</html>')

        return '\n'.join(html_parts)

    def save_pipeline(self, path: str, method: str = 'joblib') -> None:
        """
        Save the fitted pipeline to disk.

        Parameters
        ----------
        path : str
            Path where to save the pipeline.

        method : str, default='joblib'
            Serialization method: 'joblib' or 'pickle'.
            joblib is recommended for scikit-learn models.

        Examples
        --------
        >>> pipeline.save_pipeline('my_pipeline.pkl')
        >>> pipeline.save_pipeline('my_pipeline.joblib', method='joblib')
        """
        if self.labels_ is None:
            warnings.warn(
                "Pipeline has not been fitted yet. Saving unfitted pipeline.",
                UserWarning
            )

        if self.verbose:
            print(f"\nSaving pipeline to {path} (method: {method})...")

        if method == 'joblib':
            try:
                import joblib
                joblib.dump(self, path, compress=3)
            except ImportError:
                raise ImportError(
                    "joblib is not installed. Install with: pip install joblib\n"
                    "Or use method='pickle' instead."
                )
        elif method == 'pickle':
            import pickle
            with open(path, 'wb') as f:
                pickle.dump(self, f)
        else:
            raise ValueError(f"Unknown method '{method}'. Use 'joblib' or 'pickle'.")

        if self.verbose:
            print(f"  ✓ Pipeline saved successfully")

    @staticmethod
    def load_pipeline(path: str, method: str = 'joblib') -> 'ClusterAnalysisPipeline':
        """
        Load a saved pipeline from disk.

        Parameters
        ----------
        path : str
            Path to the saved pipeline file.

        method : str, default='joblib'
            Serialization method used when saving: 'joblib' or 'pickle'.

        Returns
        -------
        pipeline : ClusterAnalysisPipeline
            Loaded pipeline instance.

        Examples
        --------
        >>> pipeline = ClusterAnalysisPipeline.load_pipeline('my_pipeline.pkl')
        >>> pipeline = ClusterAnalysisPipeline.load_pipeline('my_pipeline.joblib', method='joblib')
        """
        if method == 'joblib':
            try:
                import joblib
                pipeline = joblib.load(path)
            except ImportError:
                raise ImportError(
                    "joblib is not installed. Install with: pip install joblib\n"
                    "Or use method='pickle' instead."
                )
        elif method == 'pickle':
            import pickle
            with open(path, 'rb') as f:
                pipeline = pickle.load(f)
        else:
            raise ValueError(f"Unknown method '{method}'. Use 'joblib' or 'pickle'.")

        if not isinstance(pipeline, ClusterAnalysisPipeline):
            raise ValueError(
                f"Loaded object is not a ClusterAnalysisPipeline instance. "
                f"Got {type(pipeline).__name__}"
            )

        return pipeline

    def compare_algorithms(
        self,
        X: pd.DataFrame,
        feature_columns: Optional[List[str]] = None,
        algorithms: Optional[List[str]] = None,
        n_clusters_range: Optional[tuple] = None,
        metrics: Optional[List[str]] = None,
        return_pipelines: bool = False
    ) -> Dict[str, Any]:
        """
        Compare multiple clustering algorithms and recommend the best one.

        This method runs the complete pipeline with different clustering algorithms,
        evaluates each using multiple metrics, and provides a recommendation for
        the best algorithm based on your data.

        Parameters
        ----------
        X : pd.DataFrame
            Input data to cluster.

        feature_columns : list of str, optional
            Column names to use as features. If None, uses all numeric columns.

        algorithms : list of str, optional
            Algorithms to compare. Default: ['kmeans', 'gmm', 'hierarchical', 'dbscan', 'hdbscan']
            Available: 'kmeans', 'gmm', 'hierarchical', 'dbscan', 'hdbscan'.

        n_clusters_range : tuple, optional
            Range of cluster numbers to try (min, max) for algorithms that need k.
            Default: self.n_clusters_range or (2, 8).

        metrics : list of str, optional
            Metrics to compute. Default: ['silhouette', 'calinski_harabasz', 'davies_bouldin']

        return_pipelines : bool, default=False
            Whether to return fitted pipeline objects for each algorithm.

        Returns
        -------
        results : dict
            Dictionary containing:
            - 'comparison': pd.DataFrame with metrics for each algorithm
            - 'best_algorithm': str, recommended algorithm name
            - 'best_n_clusters': int, optimal number of clusters for best algorithm
            - 'best_score': float, best silhouette score achieved
            - 'detailed_results': dict with full results per algorithm
            - 'pipelines': dict of fitted pipelines (if return_pipelines=True)

        Examples
        --------
        >>> from clustertk import ClusterAnalysisPipeline
        >>> import pandas as pd
        >>>
        >>> df = pd.read_csv('data.csv')
        >>> pipeline = ClusterAnalysisPipeline()
        >>>
        >>> # Compare all algorithms
        >>> results = pipeline.compare_algorithms(df, feature_columns=['f1', 'f2', 'f3'])
        >>>
        >>> # View comparison table
        >>> print(results['comparison'])
        >>>
        >>> # Get recommendation
        >>> print(f"Best algorithm: {results['best_algorithm']}")
        >>> print(f"Optimal clusters: {results['best_n_clusters']}")
        >>>
        >>> # Use recommended settings
        >>> pipeline.clustering_algorithm = results['best_algorithm']
        >>> pipeline.n_clusters = results['best_n_clusters']
        >>> pipeline.fit(df, feature_columns=['f1', 'f2', 'f3'])
        """
        if self.verbose:
            print("\n" + "=" * 80)
            print("ALGORITHM COMPARISON")
            print("=" * 80)

        # Set defaults
        if algorithms is None:
            algorithms = ['kmeans', 'gmm', 'hierarchical', 'dbscan', 'hdbscan']

        if n_clusters_range is None:
            n_clusters_range = self.n_clusters_range if hasattr(self, 'n_clusters_range') else (2, 8)

        if metrics is None:
            metrics = ['silhouette', 'calinski_harabasz', 'davies_bouldin', 'cluster_balance']

        # Storage for results
        comparison_data = []
        detailed_results = {}
        pipelines = {} if return_pipelines else None

        # Test each algorithm
        for algo in algorithms:
            if self.verbose:
                print(f"\n{'-' * 80}")
                print(f"Testing algorithm: {algo.upper()}")
                print(f"{'-' * 80}")

            try:
                # Create pipeline with this algorithm
                algo_pipeline = ClusterAnalysisPipeline(
                    # Copy current preprocessing settings
                    handle_missing=self.handle_missing,
                    handle_outliers=self.handle_outliers,
                    scaling=self.scaling,
                    log_transform_skewed=self.log_transform_skewed,
                    skewness_threshold=self.skewness_threshold,
                    correlation_threshold=self.correlation_threshold,
                    variance_threshold=self.variance_threshold,
                    smart_correlation=self.smart_correlation,
                    correlation_strategy=self.correlation_strategy,
                    pca_variance=self.pca_variance,
                    pca_min_components=self.pca_min_components,
                    # Algorithm-specific settings
                    clustering_algorithm=algo,
                    n_clusters=None if algo == 'dbscan' else None,  # Auto-detect
                    n_clusters_range=n_clusters_range,
                    random_state=self.random_state,
                    verbose=False  # Suppress individual pipeline output
                )

                # Fit pipeline
                algo_pipeline.fit(X, feature_columns=feature_columns)

                # Collect metrics
                algo_metrics = algo_pipeline.metrics_
                n_clusters = algo_pipeline.n_clusters_

                # Store results
                result_row = {
                    'algorithm': algo,
                    'n_clusters': n_clusters,
                }

                # Add requested metrics
                for metric in metrics:
                    if metric in algo_metrics:
                        result_row[metric] = algo_metrics[metric]

                comparison_data.append(result_row)

                # Store detailed results
                detailed_results[algo] = {
                    'n_clusters': n_clusters,
                    'metrics': algo_metrics,
                    'labels': algo_pipeline.labels_,
                    'cluster_sizes': pd.Series(algo_pipeline.labels_).value_counts().to_dict()
                }

                # Store pipeline if requested
                if return_pipelines:
                    pipelines[algo] = algo_pipeline

                if self.verbose:
                    print(f"  ✓ {algo}: {n_clusters} clusters, silhouette={algo_metrics.get('silhouette', 0):.3f}")

            except Exception as e:
                if self.verbose:
                    print(f"  ✗ {algo} failed: {str(e)}")
                comparison_data.append({
                    'algorithm': algo,
                    'n_clusters': None,
                    'error': str(e)
                })

        # Create comparison DataFrame
        comparison_df = pd.DataFrame(comparison_data)

        # Determine best algorithm
        best_algo, best_k, best_score = self._select_best_algorithm(comparison_df, detailed_results)

        if self.verbose:
            print("\n" + "=" * 80)
            print("COMPARISON RESULTS")
            print("=" * 80)
            print(comparison_df.to_string(index=False))
            print("\n" + "=" * 80)
            print(f"RECOMMENDATION: {best_algo.upper()}")
            print(f"Optimal clusters: {best_k}")
            print(f"Silhouette score: {best_score:.3f}")
            print("=" * 80)

        # Build results dictionary
        results = {
            'comparison': comparison_df,
            'best_algorithm': best_algo,
            'best_n_clusters': best_k,
            'best_score': best_score,
            'detailed_results': detailed_results
        }

        if return_pipelines:
            results['pipelines'] = pipelines

        return results

    def _select_best_algorithm(
        self,
        comparison_df: pd.DataFrame,
        detailed_results: Dict[str, Any]
    ) -> tuple:
        """
        Select best algorithm based on metrics.

        Uses weighted scoring:
        - Silhouette: higher is better (weight: 0.35)
        - Calinski-Harabasz: higher is better (weight: 0.25)
        - Davies-Bouldin: lower is better (weight: 0.25)
        - Cluster Balance: higher is better (weight: 0.15)

        Returns
        -------
        best_algo : str
            Name of best algorithm
        best_k : int
            Optimal number of clusters
        best_score : float
            Best silhouette score
        """
        # Filter out failed algorithms
        valid_df = comparison_df[comparison_df['n_clusters'].notna()].copy()

        if len(valid_df) == 0:
            raise ValueError("All algorithms failed")

        # Normalize metrics to [0, 1] range
        scores = {}

        for idx, row in valid_df.iterrows():
            algo = row['algorithm']
            score = 0.0
            weight_sum = 0.0

            # Silhouette (higher is better)
            if 'silhouette' in row and pd.notna(row['silhouette']):
                # Silhouette is already in [-1, 1], shift to [0, 1]
                normalized = (row['silhouette'] + 1) / 2
                score += normalized * 0.35
                weight_sum += 0.35

            # Calinski-Harabasz (higher is better)
            if 'calinski_harabasz' in row and pd.notna(row['calinski_harabasz']):
                # Normalize by max value
                max_ch = valid_df['calinski_harabasz'].max()
                if max_ch > 0:
                    normalized = row['calinski_harabasz'] / max_ch
                    score += normalized * 0.25
                    weight_sum += 0.25

            # Davies-Bouldin (lower is better)
            if 'davies_bouldin' in row and pd.notna(row['davies_bouldin']):
                # Invert and normalize
                max_db = valid_df['davies_bouldin'].max()
                min_db = valid_df['davies_bouldin'].min()
                if max_db > min_db:
                    normalized = 1 - (row['davies_bouldin'] - min_db) / (max_db - min_db)
                    score += normalized * 0.25
                    weight_sum += 0.25

            # Cluster Balance (higher is better)
            if 'cluster_balance' in row and pd.notna(row['cluster_balance']):
                # Balance is already in [0, 1]
                normalized = row['cluster_balance']
                score += normalized * 0.15
                weight_sum += 0.15

            # Average score
            if weight_sum > 0:
                scores[algo] = score / weight_sum
            else:
                scores[algo] = 0.0

        # Select best
        best_algo = max(scores, key=scores.get)
        best_row = valid_df[valid_df['algorithm'] == best_algo].iloc[0]
        best_k = int(best_row['n_clusters'])
        best_score = float(best_row.get('silhouette', 0))

        return best_algo, best_k, best_score

    def plot_algorithm_comparison(
        self,
        comparison_results: Dict[str, Any],
        title: Optional[str] = None,
        figsize: tuple = (14, 6)
    ):
        """
        Visualize algorithm comparison results.

        Parameters
        ----------
        comparison_results : dict
            Results from compare_algorithms() method

        title : str, optional
            Plot title

        figsize : tuple, default=(14, 6)
            Figure size

        Returns
        -------
        fig : matplotlib.figure.Figure
            Figure object

        Examples
        --------
        >>> results = pipeline.compare_algorithms(df, feature_columns=features)
        >>> pipeline.plot_algorithm_comparison(results)
        """
        from clustertk.visualization import check_viz_available, plot_algorithm_comparison

        if not check_viz_available():
            raise ImportError(
                "Visualization dependencies not installed. "
                "Install with: pip install clustertk[viz]"
            )

        return plot_algorithm_comparison(
            comparison_results['comparison'],
            title=title,
            figsize=figsize
        )

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
