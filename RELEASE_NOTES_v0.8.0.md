# ClusterTK v0.8.0 Release Notes

**Release Date:** October 29, 2025

## 🎉 Major Features

### HDBSCAN Clustering Algorithm

We're excited to announce the addition of **HDBSCAN** (Hierarchical Density-Based Spatial Clustering of Applications with Noise) to ClusterTK!

**Key Features:**
- Automatic parameter tuning for `min_cluster_size` and `min_samples`
  - `min_cluster_size`: sqrt(n_samples), bounded [5, 100]
  - `min_samples`: equals min_cluster_size by default
- Soft clustering support with `probabilities_` attribute
- Cluster stability analysis via `cluster_persistence_` attribute
- Full integration with Pipeline and `compare_algorithms()`

**Usage:**
```python
from clustertk import ClusterAnalysisPipeline

# Using HDBSCAN directly
pipeline = ClusterAnalysisPipeline(
    clustering_algorithm='hdbscan',
    verbose=True
)
pipeline.fit(df, feature_columns=feature_cols)

# Compare with other algorithms
results = pipeline.compare_algorithms(
    X=df,
    feature_columns=feature_cols,
    algorithms=['kmeans', 'gmm', 'dbscan', 'hdbscan'],
    n_clusters_range=(2, 10)
)
```

**Why HDBSCAN?**
- Superior to DBSCAN for varying density clusters
- No need to specify number of clusters
- Hierarchical clustering tree provides insights
- Robust outlier detection

### Test Suite

A comprehensive test suite has been added to ensure code quality and reliability!

**Coverage:**
- **39 tests** across 5 test modules
- **39% overall coverage**
- **66-76% coverage** for clustering algorithms
- **61-69% coverage** for preprocessing modules
- **43% coverage** for pipeline

**Test Categories:**
- Unit tests for preprocessing (missing values, scaling)
- Unit tests for clustering (K-Means, HDBSCAN)
- Unit tests for evaluation (metrics)
- Integration tests for full pipeline workflows
- Edge case testing (missing data, outliers, small clusters)

**Running Tests:**
```bash
# Install dev dependencies
pip install clustertk[dev]

# Run all tests
pytest

# Run with coverage report
pytest --cov=clustertk --cov-report=html
```

## 🔧 Technical Improvements

### Bug Fixes
- Fixed n_jobs handling in HDBSCAN (None → 1 for compatibility)
- Fixed test fixtures to use correct sample sizes
- Improved error messages in preprocessing modules

### Documentation
- Added comprehensive HDBSCAN documentation in `docs/user_guide/clustering.md`
- Updated README with HDBSCAN in algorithm list
- Added pytest.ini configuration with coverage settings

### Infrastructure
- pytest framework with fixtures for reusable test data
- Coverage reporting (term, HTML, XML)
- Test markers for categorization (unit, integration, slow)

## 📊 Algorithm Comparison Update

The `compare_algorithms()` method now includes HDBSCAN by default:

```python
# Default algorithms tested
algorithms = ['kmeans', 'gmm', 'hierarchical', 'dbscan', 'hdbscan']
```

## 🚀 Installation

```bash
# Basic installation
pip install clustertk

# With visualization
pip install clustertk[viz]

# With HDBSCAN and UMAP
pip install clustertk[extras]

# Full installation (including dev tools)
pip install clustertk[all]
```

## 📈 Performance

HDBSCAN performance characteristics:
- **Time complexity:** O(n log n) on average
- **Space complexity:** O(n)
- **Recommended for:** 100-100,000 samples
- **Parallel processing:** Supported via `n_jobs` parameter

## 🔮 What's Next?

Planned for **v0.9.0**:
- Enhanced test coverage (target: >50%)
- CI/CD with GitHub Actions
- Automated testing on push/PR

Planned for **v0.10.0+**:
- Enhanced feature analysis (SHAP values, permutation importance)
- More clustering algorithms (Spectral, OPTICS)
- Full Sphinx API documentation
- GitHub Pages hosting

## 📝 Breaking Changes

None! This release is fully backward compatible with v0.7.0.

## 🙏 Acknowledgments

- HDBSCAN implementation based on the excellent `hdbscan` library by Leland McInnes
- Test infrastructure inspired by scikit-learn best practices

## 📦 Full Changelog

**New Features:**
- Add HDBSCANClustering algorithm with auto parameter tuning
- Add comprehensive test suite (39 tests, 39% coverage)
- Add pytest.ini configuration
- Add test fixtures for various data scenarios

**Improvements:**
- Update compare_algorithms() to include HDBSCAN
- Enhance documentation with HDBSCAN guide
- Add coverage reporting (HTML, XML, terminal)

**Bug Fixes:**
- Fix n_jobs=None handling in HDBSCAN
- Fix test fixture sample sizes
- Fix error message matching in tests

**Internal:**
- Add tests for preprocessing, clustering, evaluation
- Add integration tests for full pipeline
- Improve test data generation with consistent seeds

---

For full documentation, visit: https://github.com/alexeiveselov92/clustertk

For bug reports and feature requests: https://github.com/alexeiveselov92/clustertk/issues
