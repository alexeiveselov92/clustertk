# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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
