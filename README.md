# ClusterTK

[![PyPI version](https://badge.fury.io/py/clustertk.svg)](https://pypi.org/project/clustertk/)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Tests](https://github.com/alexeiveselov92/clustertk/workflows/Tests/badge.svg)](https://github.com/alexeiveselov92/clustertk/actions/workflows/tests.yml)
[![codecov](https://codecov.io/gh/alexeiveselov92/clustertk/branch/main/graph/badge.svg)](https://codecov.io/gh/alexeiveselov92/clustertk)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**A comprehensive Python toolkit for cluster analysis with full pipeline support.**

ClusterTK provides a complete, sklearn-style pipeline for clustering: from raw data preprocessing to cluster interpretation and export. Perfect for data analysts who want powerful clustering without writing hundreds of lines of code.

## Features

- 🔄 **Complete Pipeline** - One-line solution from raw data to insights
- 📊 **Multiple Algorithms** - K-Means, GMM, Hierarchical, DBSCAN, HDBSCAN
- 🎯 **Auto-Optimization** - Automatic optimal cluster number selection
- 🎨 **Rich Visualization** - Beautiful plots (optional dependency)
- 📁 **Export & Reports** - CSV, JSON, HTML reports with embedded plots
- 💾 **Save/Load** - Persist and reload fitted pipelines
- 🔍 **Interpretation** - Profiling, naming, and feature importance analysis

## Quick Start

### Installation

```bash
# Core functionality
pip install clustertk

# With visualization
pip install clustertk[viz]
```

### Basic Usage

```python
import pandas as pd
from clustertk import ClusterAnalysisPipeline

# Load data
df = pd.read_csv('your_data.csv')

# Create and fit pipeline
pipeline = ClusterAnalysisPipeline(
    handle_missing='median',
    correlation_threshold=0.85,
    n_clusters=None,  # Auto-detect optimal number
    verbose=True
)

pipeline.fit(df, feature_columns=['feature1', 'feature2', 'feature3'])

# Get results
labels = pipeline.labels_
profiles = pipeline.cluster_profiles_
metrics = pipeline.metrics_

print(f"Found {pipeline.n_clusters_} clusters")
print(f"Silhouette score: {metrics['silhouette']:.3f}")

# Export
pipeline.export_results('results.csv')
pipeline.export_report('report.html')

# Visualize (requires clustertk[viz])
pipeline.plot_clusters_2d()
pipeline.plot_cluster_heatmap()
```

## Documentation

- **[Installation Guide](docs/installation.md)** - Detailed installation instructions
- **[Quick Start](docs/quickstart.md)** - Get started in 5 minutes
- **[User Guide](docs/user_guide/README.md)** - Complete component documentation
  - [Preprocessing](docs/user_guide/preprocessing.md)
  - [Feature Selection](docs/user_guide/feature_selection.md)
  - [Clustering](docs/user_guide/clustering.md)
  - [Evaluation](docs/user_guide/evaluation.md)
  - [Interpretation](docs/user_guide/interpretation.md) - Profiles, naming, feature importance
  - [Visualization](docs/user_guide/visualization.md)
  - [Export](docs/user_guide/export.md)
- **[Examples](docs/examples.md)** - Real-world use cases
- **[FAQ](docs/faq.md)** - Common questions

## Pipeline Workflow

```
Raw Data → Preprocessing → Feature Selection → Dimensionality Reduction
→ Clustering → Evaluation → Interpretation → Export
```

Each step is configurable through pipeline parameters or can be run independently.

## Key Capabilities

### Preprocessing
- Missing value handling (median/mean/drop)
- Outlier detection and treatment
- Automatic scaling (robust/standard/minmax)
- Skewness transformation

### Clustering Algorithms
- **K-Means** - Fast, spherical clusters
- **GMM** - Probabilistic, elliptical clusters
- **Hierarchical** - Dendrograms, hierarchical structure
- **DBSCAN** - Density-based, arbitrary shapes
- **HDBSCAN** - Advanced density-based, varying densities (v0.8.0+)

### Evaluation & Interpretation
- Silhouette score, Calinski-Harabasz, Davies-Bouldin metrics
- Automatic optimal k selection
- Cluster profiling and automatic naming
- **Feature importance analysis** (v0.9.0+)
  - Permutation importance
  - Feature contribution (variance ratio)
  - SHAP values (optional)

### Export & Reports
- CSV export (data + labels)
- JSON export (metadata + profiles)
- HTML reports with embedded visualizations
- Pipeline serialization (save/load)

## Examples

### Feature Importance Analysis

```python
# Understand which features drive your clustering
results = pipeline.analyze_feature_importance(method='all')

# View permutation importance
print(results['permutation'].head())

# View feature contribution (variance ratio)
print(results['contribution'].head())

# Use top features for focused analysis
top_features = results['permutation'].head(5)['feature'].tolist()
```

### Algorithm Comparison

```python
# Compare multiple algorithms automatically
results = pipeline.compare_algorithms(
    X=df,
    feature_columns=['feature1', 'feature2', 'feature3'],
    algorithms=['kmeans', 'gmm', 'hierarchical', 'dbscan'],
    n_clusters_range=(2, 8)
)

print(results['comparison'])  # DataFrame with metrics
print(f"Best algorithm: {results['best_algorithm']}")

# Visualize comparison
pipeline.plot_algorithm_comparison(results)
```

### Customer Segmentation

```python
pipeline = ClusterAnalysisPipeline(
    n_clusters=None,  # Auto-detect
    auto_name_clusters=True
)

pipeline.fit(customers_df,
            feature_columns=['age', 'income', 'purchases'],
            category_mapping={
                'demographics': ['age', 'income'],
                'behavior': ['purchases']
            })

pipeline.export_report('customer_segments.html')
```

### Anomaly Detection

```python
pipeline = ClusterAnalysisPipeline(
    clustering_algorithm='dbscan'
)

pipeline.fit(transactions_df)
anomalies = transactions_df[pipeline.labels_ == -1]
```

More examples: [docs/examples.md](docs/examples.md)

## Requirements

- Python 3.8+
- numpy >= 1.20.0
- pandas >= 1.3.0
- scikit-learn >= 1.0.0
- scipy >= 1.7.0
- joblib >= 1.0.0

Optional (for visualization):
- matplotlib >= 3.4.0
- seaborn >= 0.11.0

## Contributing

Contributions are welcome! Please check:
- [GitHub Issues](https://github.com/alexeiveselov92/clustertk/issues) - Report bugs
- [GitHub Discussions](https://github.com/alexeiveselov92/clustertk/discussions) - Questions

## License

MIT License - see [LICENSE](LICENSE) file for details.

## Citation

If you use ClusterTK in your research, please cite:

```bibtex
@software{clustertk2024,
  author = {Veselov, Aleksey},
  title = {ClusterTK: A Comprehensive Python Toolkit for Cluster Analysis},
  year = {2024},
  url = {https://github.com/alexeiveselov92/clustertk}
}
```

## Links

- **PyPI**: https://pypi.org/project/clustertk/
- **GitHub**: https://github.com/alexeiveselov92/clustertk
- **Documentation**: [docs/](docs/)
- **Author**: Aleksey Veselov (alexei.veselov92@gmail.com)

---

Made with ❤️ for the data science community
