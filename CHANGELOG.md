# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.16.2] - 2025-10-30

### Improved
- **Enhanced Pipeline Configuration in HTML Reports** - Complete redesign of configuration section
  - Organized into logical sections: Preprocessing, Feature Selection, Dimensionality Reduction, Clustering, Features
  - Added missing parameters:
    - `outlier_percentiles` - Shows percentile range for winsorization (e.g., "2.5%-97.5%")
    - `dim_reduction` method - Shows actual reduction method used ('auto'/'pca'/'umap'/'none')
    - `umap_n_components` - Number of UMAP components when using UMAP
    - `detect_multivariate_outliers` - Multivariate outlier detection method
    - `multivariate_contamination` - Expected contamination level for multivariate outliers
    - `clustering_algorithm` - Algorithm used (kmeans/gmm/hdbscan/etc)
    - `clustering_params` - Custom algorithm parameters if provided
    - `min_cluster_size` - Minimum cluster size enforcement
  - Auto-detection features:
    - Shows actual components used for PCA (not just variance threshold)
    - Shows auto-selected method when dim_reduction='auto'
    - Conditional display: only shows relevant parameters based on configuration
  - Better structure: clear subsections with emoji icons (⚙️ Pipeline Configuration)

### Why This Matters
- Reports now accurately reflect all pipeline settings
- Easier to reproduce results - all parameters are documented
- Better transparency for feature selection and dimensionality reduction decisions
- Critical for comparing different pipeline configurations

## [0.16.1] - 2025-10-30

### Fixed
- **KeyError in export_report() after feature selection** - Fixed crash when generating reports after `refit_with_top_features(update_pipeline=True)`
  - Problem: `export_report()` tried to access features from original clustering that no longer exist in reduced feature set
  - Solution: Added safety checks in `_build_html_report()` to only access features present in current `cluster_profiles_`
  - Updated `refit_with_top_features()` to properly update `_profiler` attribute when pipeline is updated
  - Now you can safely call `export_report()` after using feature selection

## [0.16.0] - 2025-10-30

### Added
- **Feature Selection for Clustering Optimization** - Two new methods to find optimal feature subsets
  - `get_pca_feature_importance()` - Shows which original features contribute most to PCA components
    - Returns DataFrame with total PCA loadings and relative importance
    - Helps interpret PCA-based clustering results
    - Guides feature selection decisions
  - `refit_with_top_features()` - Iterative feature selection workflow
    - Refits clustering using top N most important features
    - Compares metrics (original vs refitted) automatically
    - Supports 3 importance methods: `permutation`, `contribution`, `pca`
    - Can update pipeline if metrics improved (`update_pipeline=True`)
    - Weighted scoring: Silhouette 50%, Calinski-Harabasz 25%, Davies-Bouldin 25%

- **Customizable Outlier Winsorization** - Control percentile clipping bounds
  - Added `outlier_percentiles` parameter to Pipeline (default: `(0.025, 0.975)`)
  - Allows fine-tuning winsorization for different data distributions
  - Examples:
    - `(0.01, 0.99)` - Very mild (2% total clipping, extreme outliers only)
    - `(0.025, 0.975)` - Balanced (5% total clipping, default, ~2σ)
    - `(0.05, 0.95)` - Aggressive (10% total clipping, risk of information loss)
  - Dynamic verbose output shows actual percentile range used

### Why This Matters
- **Curse of Dimensionality:** More features ≠ better clustering
- **Feature Noise:** Irrelevant features dilute clustering signal
- **Analyst Workflow:** "I have 30 features, which ones should I use?"
- **Solution:** Iterative feature selection finds optimal subset

### Example Workflow
```python
# 1. Fit on all features
pipeline = ClusterAnalysisPipeline(dim_reduction='pca')
pipeline.fit(df)  # 30 features

# 2. Try refitting with top 10 features
comparison = pipeline.refit_with_top_features(
    n_features=10,
    importance_method='permutation',  # Best for clustering quality
    compare_metrics=True,
    update_pipeline=False  # Just compare first
)

# 3. If metrics improved, update pipeline
if comparison['metrics_improved']:
    pipeline.refit_with_top_features(n_features=10, update_pipeline=True)
```

### Performance
Test results (1000 samples, 20 features: 3 meaningful + 7 derived + 10 noise):
- **PCA importance:** +21% improvement with top 5 features
- **Permutation importance:** +105% improvement with top 8 features!
- Correctly identified meaningful features over noise

## [0.15.0] - 2025-10-30

### Added
- **Flexible Dimensionality Reduction** - Smart algorithm-specific selection
  - New `dim_reduction` parameter with options: `'auto'`, `'pca'`, `'umap'`, `'none'`
  - **Auto-mode** intelligently selects best method based on algorithm + n_features:
    - K-Means/GMM + <50 features → `none` (works well in original space)
    - K-Means/GMM + ≥50 features → `pca` (handles curse of dimensionality)
    - HDBSCAN/DBSCAN + <30 features → `none` (preserves local density)
    - HDBSCAN/DBSCAN + ≥30 features → `umap` (preserves local structure)
  - **UMAP parameters** for clustering (NOT visualization!):
    - `umap_n_components=10` (default, NOT 2!)
    - `umap_n_neighbors=30`, `umap_min_dist=0.1`, `umap_metric='euclidean'`
  - **Key insight:** UMAP CAN be used for clustering with proper settings

### Why This Matters
- **PCA problem for HDBSCAN:** PCA destroys local density → HDBSCAN finds only noise
- **UMAP solution:** Preserves local structure → HDBSCAN works correctly
- **CRITICAL:** UMAP for clustering needs `n_components=10-20`, NOT 2-3 (visualization only)

### Changed
- Renamed internal `_pca_reducer` → `_reducer` (supports PCA/UMAP/None)
- Updated all PCA-specific visualizations to check for PCA attributes

### Fixed
- HDBSCAN now works correctly on high-dimensional data via UMAP
- Visualization methods handle both PCA and UMAP reduction

## [0.14.5] - 2025-10-30

### Fixed
- **HDBSCAN/DBSCAN visualization bug** - IndexError in `plot_clusters_2d()` with noise points
  - Issue: `enumerate(unique_labels)` gave wrong indices when -1 (noise) present
  - Example: labels=[-1,0,1,2] → enumerate gives i=0,1,2,3, but colors has only 3 items
  - Fix: Separate `color_idx` counter that only increments for real clusters (not noise)
  - Noise points now correctly colored as gray

## [0.14.4] - 2025-10-30

### Fixed
- **Heatmap generation error** in `export_report()` with >15 features
  - Issue: Tried to pass `top_n_features` parameter that doesn't exist in function signature
  - Fix: Temporarily slice `cluster_profiles_` dataframe before calling method
  - Reports now correctly generate heatmaps with top 15 features when total >15

## [0.14.3] - 2025-10-30

### Improved
- **UX Enhancement:** More informative titles in HTML reports
  - Heatmap title now includes "(Top N Features)" when features are limited
  - Helps users understand what they're seeing in reports

## [0.12.0] - 2025-10-30

### Added
- **Algorithm-specific parameters** via `clustering_params` dict
  - Pass custom parameters to any clustering algorithm
  - Example: `ClusterAnalysisPipeline(clustering_algorithm='hdbscan', clustering_params={'min_cluster_size': 50})`
  - Works with all algorithms: kmeans, gmm, hierarchical, dbscan, hdbscan
  - Parameters override defaults but preserve backward compatibility

- **Noise points statistics** for DBSCAN/HDBSCAN
  - Automatically detect and count noise points (label -1)
  - New metrics: `n_noise` (count) and `noise_ratio` (percentage)
  - Shown in `get_metrics_summary()` with quality assessment
  - Metrics computed on non-noise points only (correct behavior)

### Fixed
- Clustering metrics now correctly filter out noise points before computation
- Silhouette/Calinski-Harabasz/Davies-Bouldin now use only labeled points

### Use Cases
- **HDBSCAN customization:** `clustering_params={'min_cluster_size': 100, 'min_samples': 5}`
- **DBSCAN tuning:** `clustering_params={'eps': 0.3, 'min_samples': 10}`
- **Hierarchical linkage:** `clustering_params={'linkage': 'complete'}`
- **Track noise:** See how many points couldn't be assigned to clusters

## [0.11.1] - 2025-10-30

### Fixed
- **SHAP values bug fix** - Fixed ValueError when SHAP returns multidimensional arrays
  - Issue: `mean_shap` could be 2D in multiclass scenarios, causing pandas DataFrame error
  - Fix: Added `flatten()` to ensure `mean_shap` is always 1D before DataFrame creation
  - Error: "Per-column arrays must each be 1-dimensional"
  - Now works correctly for all clustering scenarios (binary, multiclass)

## [0.11.0] - 2025-10-30

### Added
- **SmartCorrelationFilter** - Intelligent feature selection from correlated pairs
  - **Hopkins statistic strategy** (`selection_strategy='hopkins'`, default)
    - Measures clusterability of each feature
    - Keeps features that are more suited for clustering
    - Helps choose between equivalent representations (e.g., percent vs deciles)
  - **Variance ratio strategy** (`selection_strategy='variance_ratio'`)
    - Uses quick K-Means clustering to evaluate feature separation
    - Higher variance ratio = better cluster separation
  - **Backward compatible** (`selection_strategy='mean_corr'` = old behavior)
  - **New methods:**
    - `get_feature_scores()` - view clusterability scores for all features
    - `get_selection_summary()` - detailed report of selection decisions
  - Enabled by default in Pipeline (`smart_correlation=True`)

- **Cluster Balance Metric** - Measure cluster size distribution
  - `cluster_balance_score()` - normalized Shannon entropy [0, 1]
    - 1.0 = perfectly balanced (all clusters equal size)
    - >0.8 = well balanced
    - <0.5 = imbalanced (some clusters much larger)
    - ~0.0 = highly imbalanced (e.g., 99% in one cluster)
  - **Automatic handling of:**
    - DBSCAN/HDBSCAN noise points (-1 labels)
    - Single cluster edge case
    - Empty clusters after filtering

### Changed
- **Pipeline** - Enhanced feature selection
  - Added `smart_correlation` parameter (default: `True`)
  - Added `correlation_strategy` parameter (default: `'hopkins'`)
  - Verbose output now shows smart selection reasoning
  - Backward compatible: set `smart_correlation=False` for old behavior

- **Evaluation** - Cluster balance integrated throughout
  - `compute_clustering_metrics()` includes `cluster_balance` by default
  - `get_metrics_summary()` includes balance with quality thresholds
  - `OptimalKFinder` uses balance as 4th metric in voting
  - `compare_algorithms()` uses balance in weighted scoring (15% weight)
  - Weights adjusted: silhouette 35%, calinski 25%, davies-bouldin 25%, balance 15%

### Fixed
- Removed unused imports in `optimal_k.py` (basedpyright warnings)

### Tests
- Added `test_smart_correlation.py` - 9 comprehensive tests (90% coverage)
- Added `test_cluster_balance.py` - 12 comprehensive tests (100% coverage)
- All 21 new tests pass ✅

### Use Cases
This release solves the common analyst problem:
- **Problem:** "I have multiple representations of the same metric (percent, deciles, log-transformed) and don't know which is best for clustering"
- **Solution:** SmartCorrelationFilter automatically selects the representation with better clusterability using Hopkins statistic
- **Problem:** "My clustering produces one huge cluster with tiny others"
- **Solution:** cluster_balance_score detects imbalanced clusterings, and compare_algorithms() now considers balance when recommending algorithms

## [0.10.2] - 2025-10-30

### Improved
- **Feature Contribution True Vectorization**
  - Replaced pandas groupby with pure NumPy bincount vectorization
  - Performance: 0.0165s → 0.0134s (1.23x faster on 80k samples)
  - Benefits:
    - True vectorization without hidden loops
    - No pandas DataFrame creation overhead
    - Pure NumPy C-level operations
  - All 21 tests pass with 77% coverage

## [0.10.1] - 2025-10-30

### Fixed
- **CRITICAL FIX**: Feature Importance memory issue on large datasets
  - **Permutation importance** was causing OOM on datasets >10k samples
    - Silhouette score computes O(n²) pairwise distance matrix
    - 80k samples = 51+ GB memory requirement → OOM
    - **Fix:** Automatic sampling to 10k samples for silhouette computation
    - Result: 80k samples now works in ~20s instead of OOM
  - **Feature contribution** optimization
    - Replaced nested loops with vectorized groupby operations
    - 3x speedup (now <0.1s even for 80k samples)
  - Both methods now memory-efficient and fast on large datasets

### Performance
- Permutation importance on 80k samples: **OOM → 20s**
- Feature contribution on 80k samples: **0.3s → 0.03s** (10x faster)
- All 21 unit tests pass with 76% coverage

## [0.10.0] - 2025-10-30

### Changed
- **MAJOR PERFORMANCE IMPROVEMENT**: Completely rewritten `ClusterStabilityAnalyzer` for large datasets
  - **Memory optimization**: Streaming computation instead of storing all bootstrap results
    - Memory usage: O(n_samples + window) instead of O(n_samples × n_iterations)
    - 80k samples: from 32+ GB (OOM) to <500 MB (~64x reduction)
  - **Speed optimization**: Vectorized operations and adaptive sampling
    - Replaced nested Python loops with NumPy broadcasting
    - Fast lookup via `np.searchsorted()` instead of `np.where()` in loops
    - Adaptive pair sampling for large clusters (max_pairs_per_cluster parameter)
    - 80k samples, 20 iterations: ~6 seconds (previously would OOM)
  - **Sliding window approach**: Only keeps last `max_comparison_window` iterations in memory
    - Default: 10 iterations window instead of all 100
    - Provides same statistical validity with fraction of memory
  - **Algorithm complexity improvements**:
    - Sample confidence: O(n_bootstrap) per iteration (fully vectorized)
    - Cluster stability: O(n_clusters × max_pairs) instead of O(n_clusters × cluster_size²)
    - Overall stability (ARI): O(overlap × log(n)) instead of O(overlap × n)

### Added
- New `ClusterStabilityAnalyzer` parameters for performance tuning:
  - `max_comparison_window`: Sliding window size for ARI comparisons (default: 10)
  - `max_pairs_per_cluster`: Max pairs to sample per cluster (default: 5000)
- Verbose output now shows dataset size and memory optimization info
- Performance tested on 1k, 10k, and 80k sample datasets

### Technical Details
- Streaming computation methods:
  - `_update_sample_confidence_streaming()`: Vectorized incremental updates
  - `_update_cluster_stability_streaming()`: Adaptive pair sampling with vectorization
  - `_compute_ari_fast()`: Fast ARI computation using searchsorted
  - `_finalize_cluster_stability()`: Compute final scores from streaming counters
- Removed old inefficient methods:
  - `_compute_overall_stability()` (replaced with sliding window in main loop)
  - `_compute_cluster_stability()` (replaced with streaming version)
  - `_compute_sample_confidence()` (replaced with streaming version)

### Breaking Changes
- None - API remains fully backward compatible
- Users may see slightly different ARI values due to sliding window approach
  (statistical validity is maintained, but not exact reproducibility)

## [0.9.0] - 2025-10-29

### Added
- **Feature Importance Analysis** - Understand which features drive clustering results
  - Three analysis methods: Permutation importance, Feature contribution (variance ratio), SHAP values (optional)
  - `analyze_feature_importance()` pipeline method
  - `FeatureImportanceAnalyzer` class for standalone usage
  - `quick_feature_importance()` convenience function
  - Per-feature importance scores with statistical measures (mean, std)
  - Handles noise labels (-1 for DBSCAN) correctly
  - 21 comprehensive tests with 83% coverage
  - Full documentation in `docs/user_guide/interpretation.md` and `docs/examples.md`
- **Cluster Stability Analysis** - Assess clustering reliability via bootstrap resampling
  - `analyze_stability()` pipeline method with configurable iterations and sample fraction
  - `ClusterStabilityAnalyzer` class for standalone usage
  - `quick_stability_analysis()` convenience function
  - Overall stability score via pairwise Adjusted Rand Index (ARI)
  - Per-cluster stability scores (pair consistency across iterations)
  - Per-sample confidence scores (assignment consistency)
  - Automatic identification of stable (>0.7) and unstable (<0.5) clusters
  - `get_stable_samples()` and `get_unstable_samples()` helper methods
  - 20 comprehensive tests with 94% coverage
  - Full documentation in `docs/user_guide/evaluation.md` and `docs/examples.md`
  - Interpretation guidelines for decision-making

### Changed
- Updated README with feature importance and stability analysis examples
- Fixed duplicate HDBSCAN entry in README
- Added interpretation link to documentation navigation
- Expanded evaluation section to "Evaluation & Interpretation"

### Documentation
- Added feature importance section to `docs/user_guide/interpretation.md` (100+ lines)
- Added feature importance example to `docs/examples.md` with customer segmentation scenario
- Added 3 FAQ entries about feature importance (which method to use, how to reduce features)
- Added stability analysis section to `docs/user_guide/evaluation.md` (120+ lines)
- Added stability analysis example to `docs/examples.md` with decision-making workflow (100+ lines)
- All documentation includes interpretation guides and best practices

## [0.8.0] - 2025-10-29

### Added
- **HDBSCAN Clustering Algorithm** - Hierarchical DBSCAN with automatic parameter tuning
  - Auto `min_cluster_size` estimation using sqrt(n_samples), bounded [5, 100]
  - Auto `min_samples` estimation (equals min_cluster_size by default)
  - Soft clustering support via `probabilities_` attribute
  - Cluster persistence analysis via `cluster_persistence_` attribute
  - Full integration with Pipeline and `compare_algorithms()`
- **Comprehensive Test Suite** - First release with automated testing
  - 39 unit and integration tests (100% passing)
  - 39% code coverage (clustering: 66-76%, preprocessing: 61-69%)
  - pytest infrastructure with pytest.ini configuration
  - 8 reusable fixtures for various data scenarios
  - Test modules for preprocessing, clustering, evaluation, and pipeline

### Changed
- `compare_algorithms()` now includes HDBSCAN by default
- Updated documentation with HDBSCAN examples and usage guide

### Fixed
- Fixed n_jobs handling in HDBSCAN (None → 1 for library compatibility)

## [0.7.0] - 2025-10-28

### Added
- **Algorithm Comparison & Selection**
  - `compare_algorithms()` method for automatic algorithm comparison
  - Tests KMeans, GMM, Hierarchical, DBSCAN, HDBSCAN across different k values
  - Weighted scoring system (40% Silhouette, 30% Calinski-Harabasz, 30% Davies-Bouldin)
  - `plot_algorithm_comparison()` for visualization
  - Automatic recommendation of best algorithm

### Changed
- Enhanced documentation with algorithm comparison examples

## [0.6.0] - 2025-10-25

### Added
- **Complete Documentation Structure**
  - Created `docs/` directory with full documentation
  - User guides for all modules (preprocessing, clustering, evaluation, etc.)
  - Installation guide, quickstart, examples, and FAQ
  - docs/index.md as main documentation entry point

### Changed
- README.md shortened from 495 to 196 lines
- README now focuses on Quick Start with links to detailed docs

## [0.5.0] - 2025-10-22

### Added
- **Export & Reports Functionality**
  - `export_results()` - export to CSV (data + labels) and JSON (metadata + profiles + metrics)
  - `export_report()` - generate HTML reports with embedded plots (base64)
  - `save_pipeline()` / `load_pipeline()` - serialize/deserialize fitted pipelines
  - Added joblib>=1.0.0 dependency for pipeline serialization

### Changed
- Updated README with export examples

## [0.4.2] - 2025-10-20

### Changed
- Updated Quick Start example in README
- Improved navigation with Table of Contents
- Better documentation of multiple plots behavior in Jupyter

## [0.4.1] - 2025-10-19

### Fixed
- **Fixed plot duplication in Jupyter notebooks**
  - Correct solution using `plt.close(fig)` to prevent auto-display
  - All visualization functions now properly return Figure objects without duplication

## [0.4.0] - 2025-10-18

### Fixed
- Attempted fix for plot duplication (incorrect approach)

## [0.3.5] - 2025-10-17

### Changed
- Improvements in visualization module

## [0.3.4] - 2025-10-17

### Changed
- Improvements in visualization module

## [0.3.3] - 2025-10-16

### Changed
- Improvements in visualization module

## [0.3.2] - 2025-10-15

### Fixed
- Fixed normalization for small number of clusters
  - Now uses min/max from original data instead of cluster means
  - Prevents 0/1 binary values when only 2 clusters exist

## [0.3.1] - 2025-10-14

### Fixed
- README hotfix: `plot_cluster_profiles()` → `plot_cluster_heatmap()`
- Minor documentation fixes

## [0.3.0] - 2025-10-13

### Added
- **Visualization Module** (optional dependency)
  - 11 visualization functions across 4 modules
  - Integration with Pipeline via `.plot_*()` methods
  - Optional installation: `pip install clustertk[viz]`
- **Cluster Naming**
  - `ClusterNamer` with 3 strategies: top_features, categories, combined
  - Automatic meaningful cluster naming

### Changed
- Conditional imports for optional visualization dependencies

## [0.2.0] - 2025-10-10

### Added
- **Additional Clustering Algorithms**
  - `HierarchicalClustering` - Ward, Complete, Average linkage methods
  - `DBSCANClustering` - with automatic eps and min_samples selection

## [0.1.1] - 2025-10-08

### Fixed
- Documentation fixes

## [0.1.0] - 2025-10-07

### Added
- **Initial Release** - Complete clustering analysis pipeline
- **Preprocessing Module**
  - `MissingValueHandler` - median/mean/drop/custom strategies
  - `OutlierHandler` - IQR/z-score/modified z-score detection
  - `ScalerSelector` - automatic scaler selection (Standard/Robust/MinMax)
  - `SkewnessTransformer` - log/sqrt/box-cox transformations
- **Feature Selection Module**
  - `CorrelationFilter` - remove highly correlated features
  - `VarianceFilter` - remove low-variance features
- **Dimensionality Reduction Module**
  - `PCAReducer` - PCA with automatic component selection
  - `ManifoldReducer` - t-SNE/UMAP for visualization only
- **Clustering Module**
  - `BaseClusterer` - base class for all algorithms
  - `KMeansClustering` - K-Means algorithm
  - `GMMClustering` - Gaussian Mixture Model
- **Evaluation Module**
  - `compute_clustering_metrics` - Silhouette, Calinski-Harabasz, Davies-Bouldin
  - `OptimalKFinder` - automatic optimal k selection with metric voting
- **Interpretation Module**
  - `ClusterProfiler` - cluster profiling and top feature analysis
- **Pipeline Module**
  - `ClusterAnalysisPipeline` - orchestrates all steps
  - Can run as full pipeline via `fit()` or step-by-step

[Unreleased]: https://github.com/alexeiveselov92/clustertk/compare/v0.8.0...HEAD
[0.8.0]: https://github.com/alexeiveselov92/clustertk/compare/v0.7.0...v0.8.0
[0.7.0]: https://github.com/alexeiveselov92/clustertk/compare/v0.6.0...v0.7.0
[0.6.0]: https://github.com/alexeiveselov92/clustertk/compare/v0.5.0...v0.6.0
[0.5.0]: https://github.com/alexeiveselov92/clustertk/compare/v0.4.2...v0.5.0
[0.4.2]: https://github.com/alexeiveselov92/clustertk/compare/v0.4.1...v0.4.2
[0.4.1]: https://github.com/alexeiveselov92/clustertk/compare/v0.4.0...v0.4.1
[0.4.0]: https://github.com/alexeiveselov92/clustertk/compare/v0.3.5...v0.4.0
[0.3.5]: https://github.com/alexeiveselov92/clustertk/compare/v0.3.4...v0.3.5
[0.3.4]: https://github.com/alexeiveselov92/clustertk/compare/v0.3.3...v0.3.4
[0.3.3]: https://github.com/alexeiveselov92/clustertk/compare/v0.3.2...v0.3.3
[0.3.2]: https://github.com/alexeiveselov92/clustertk/compare/v0.3.1...v0.3.2
[0.3.1]: https://github.com/alexeiveselov92/clustertk/compare/v0.3.0...v0.3.1
[0.3.0]: https://github.com/alexeiveselov92/clustertk/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/alexeiveselov92/clustertk/compare/v0.1.1...v0.2.0
[0.1.1]: https://github.com/alexeiveselov92/clustertk/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/alexeiveselov92/clustertk/releases/tag/v0.1.0
