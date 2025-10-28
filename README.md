# ClusterTK

**A comprehensive toolkit for cluster analysis with full pipeline support**

[![Python Version](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

ClusterTK is a Python library designed to streamline the entire cluster analysis workflow. It provides a unified, easy-to-use interface for data preprocessing, feature selection, dimensionality reduction, clustering, evaluation, and interpretation.

## Features

- 🔄 **Complete Pipeline**: One-line solution from raw data to cluster insights
- 🛠️ **Modular Design**: Use individual components or the full pipeline
- 📊 **Multiple Algorithms**: K-Means, GMM, Hierarchical, DBSCAN
- 🎯 **Automatic Optimization**: Auto-selection of optimal cluster numbers
- 📈 **Rich Evaluation**: Comprehensive metrics (Silhouette, Calinski-Harabasz, Davies-Bouldin)
- 🎨 **Optional Visualization**: Beautiful plots without mandatory heavy dependencies
- 🔍 **Cluster Interpretation**: Automatic profiling and naming suggestions
- 📝 **Export Results**: CSV, JSON, HTML reports

## Installation

### Basic Installation (Core functionality)

```bash
pip install clustertk
```

### With Visualization Support

```bash
pip install clustertk[viz]
```

### Full Installation (All features)

```bash
pip install clustertk[all]
```

### Development Installation

```bash
git clone https://github.com/alexeiveselov92/clustertk.git
cd clustertk
pip install -e .[dev]
```

## Quick Start

```python
import pandas as pd
from clustertk import ClusterAnalysisPipeline

# Load your data
df = pd.read_csv('your_data.csv')

# Create and configure pipeline
pipeline = ClusterAnalysisPipeline(
    handle_missing='median',          # Handle missing values
    correlation_threshold=0.85,       # Remove highly correlated features
    pca_variance=0.9,                 # Keep 90% of variance
    clustering_algorithm='kmeans',    # Use K-Means
    n_clusters=None,                  # Auto-detect optimal number
    verbose=True
)

# Run complete analysis
pipeline.fit(df, feature_columns=['col1', 'col2', 'col3'])

# Get results
labels = pipeline.labels_                    # Cluster assignments
profiles = pipeline.cluster_profiles_        # Cluster profiles
metrics = pipeline.metrics_                  # Quality metrics

# Visualize (if viz dependencies installed)
pipeline.plot_clusters_2d()
pipeline.plot_cluster_heatmap()  # or plot_cluster_radar()
```

## Step-by-Step Usage

You can also run the pipeline step-by-step for more control:

```python
pipeline = ClusterAnalysisPipeline()

# Step 1: Preprocess data
pipeline.preprocess(df, feature_columns=['col1', 'col2', 'col3'])

# Step 2: Select features
pipeline.select_features()

# Step 3: Reduce dimensions
pipeline.reduce_dimensions()

# Step 4: Find optimal number of clusters
pipeline.find_optimal_clusters()

# Step 5: Perform clustering
pipeline.cluster(n_clusters=5)

# Step 6: Create cluster profiles
pipeline.create_profiles(category_mapping={
    'behavioral': ['sessions', 'duration'],
    'engagement': ['clicks', 'likes']
})

# Access intermediate results
preprocessed_data = pipeline.data_preprocessed_
pca_components = pipeline.data_reduced_
```

## Pipeline Components

### 1. Preprocessing

- **Missing Values**: Median, mean, drop, or custom imputation
- **Outliers**: IQR detection, robust scaling, clipping, or removal
- **Scaling**: StandardScaler, RobustScaler, MinMaxScaler, or auto-selection
- **Transformations**: Log transformation for skewed features

```python
pipeline = ClusterAnalysisPipeline(
    handle_missing='median',
    handle_outliers='robust',
    scaling='robust',
    log_transform_skewed=True,
    skewness_threshold=2.0
)
```

### 2. Feature Selection

- **Correlation Filtering**: Remove highly correlated features
- **Variance Filtering**: Remove low-variance features

```python
pipeline = ClusterAnalysisPipeline(
    correlation_threshold=0.85,
    variance_threshold=0.01
)
```

### 3. Dimensionality Reduction

- **PCA**: Automatic component selection based on variance threshold
- **t-SNE/UMAP**: For 2D visualization (optional)

```python
pipeline = ClusterAnalysisPipeline(
    pca_variance=0.9,
    pca_min_components=2
)
```

### 4. Clustering

Multiple algorithms supported:

```python
# K-Means
pipeline = ClusterAnalysisPipeline(clustering_algorithm='kmeans', n_clusters=5)

# Gaussian Mixture Model
pipeline = ClusterAnalysisPipeline(clustering_algorithm='gmm', n_clusters=4)

# Hierarchical
pipeline = ClusterAnalysisPipeline(clustering_algorithm='hierarchical', n_clusters=3)

# DBSCAN (auto-detects clusters)
pipeline = ClusterAnalysisPipeline(clustering_algorithm='dbscan')
```

### 5. Evaluation

- Automatic optimal cluster number detection
- Multiple metrics: Silhouette, Calinski-Harabasz, Davies-Bouldin
- Elbow method support

```python
pipeline = ClusterAnalysisPipeline(
    n_clusters=None,              # Auto-detect
    n_clusters_range=(2, 10)      # Search range
)
```

### 6. Interpretation

- Cluster profiling with feature importance
- Automatic cluster naming suggestions
- Category-based analysis

```python
pipeline.create_profiles(category_mapping={
    'behavioral': ['sessions', 'duration', 'frequency'],
    'social': ['messages', 'friends', 'shares'],
    'engagement': ['clicks', 'likes', 'comments']
})
```

## Visualization

If you installed viz dependencies (`pip install clustertk[viz]`):

```python
# Correlation matrix
pipeline.plot_correlation_matrix()

# PCA variance explained
pipeline.plot_pca_variance()

# Cluster visualization in 2D
pipeline.plot_clusters_2d(method='tsne')

# Cluster profiles heatmap
pipeline.plot_cluster_heatmap()

# Radar charts for clusters
pipeline.plot_cluster_radar()
```

**Jupyter usage:** All plot functions return matplotlib Figure objects that auto-display in Jupyter notebooks.

For multiple plots in one cell:
```python
from IPython.display import display

# Display multiple plots in one cell
display(pipeline.plot_cluster_heatmap())
display(pipeline.plot_clusters_2d())
display(pipeline.plot_cluster_radar())
```

Or use separate cells for each plot (recommended for cleaner output).

## Export Results

```python
# Export cluster labels to CSV
pipeline.export_results('results.csv', format='csv')

# Export profiles to JSON
pipeline.export_results('profiles.json', format='json')

# Generate HTML report (requires viz dependencies)
pipeline.export_report('report.html')
```

## Advanced Usage

### Custom Functions

You can provide custom functions for preprocessing:

```python
def my_custom_imputer(df):
    """Custom missing value imputation logic"""
    return df.fillna(df.median())

pipeline = ClusterAnalysisPipeline(
    handle_missing=my_custom_imputer
)
```

### Custom Clusterer

Use your own clustering implementation:

```python
from sklearn.cluster import SpectralClustering

custom_clusterer = SpectralClustering(n_clusters=4, random_state=42)

pipeline = ClusterAnalysisPipeline(
    clustering_algorithm=custom_clusterer
)
```

## Architecture

ClusterTK is built with a modular architecture:

```
clustertk/
├── preprocessing/        # Data cleaning and transformation
├── feature_selection/    # Feature filtering
├── dimensionality/       # PCA, t-SNE, UMAP
├── clustering/           # Clustering algorithms
├── evaluation/           # Metrics and optimization
├── interpretation/       # Profiling and naming
└── visualization/        # Plotting (optional)
```

Each module can be used independently:

```python
from clustertk.preprocessing import MissingValueHandler
from clustertk.clustering import KMeansClustering

# Use individual components
handler = MissingValueHandler(strategy='median')
clean_data = handler.fit_transform(df)

clusterer = KMeansClustering(n_clusters=5)
labels = clusterer.fit_predict(clean_data)
```

## Requirements

### Core Dependencies

- numpy >= 1.20.0
- pandas >= 1.3.0
- scikit-learn >= 1.0.0
- scipy >= 1.7.0

### Optional Dependencies

- matplotlib >= 3.4.0 (for visualization)
- seaborn >= 0.11.0 (for visualization)
- umap-learn >= 0.5.0 (for UMAP)
- hdbscan >= 0.8.0 (for HDBSCAN)

## Examples

Check out the [examples](examples/) directory for complete notebooks:

- `basic_usage.ipynb` - Basic clustering workflow
- `advanced_customization.ipynb` - Custom preprocessing and clustering
- `visualization_guide.ipynb` - All visualization options
- `interpretation.ipynb` - Cluster profiling and interpretation

## Documentation

Full documentation is available at: [https://clustertk.readthedocs.io](https://clustertk.readthedocs.io)

## Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use ClusterTK in your research, please cite:

```bibtex
@software{clustertk,
  author = {Aleksey Veselov},
  title = {ClusterTK: A Comprehensive Toolkit for Cluster Analysis},
  year = {2024},
  url = {https://github.com/alexeiveselov92/clustertk}
}
```

## Roadmap

- [x] Core pipeline implementation
- [x] Basic clustering algorithms
- [ ] Advanced clustering methods (HDBSCAN, Spectral)
- [ ] GPU support (cuML integration)
- [ ] Streaming/incremental clustering
- [ ] AutoML for hyperparameter tuning
- [ ] Web UI for interactive analysis
- [ ] Time series clustering support

## Acknowledgments

ClusterTK builds upon the excellent work of:

- [scikit-learn](https://scikit-learn.org/) - Machine learning algorithms
- [pandas](https://pandas.pydata.org/) - Data manipulation
- [matplotlib](https://matplotlib.org/) & [seaborn](https://seaborn.pydata.org/) - Visualization

## Support

- 📧 Email: alexei.veselov92@gmail.com
- 🐛 Issues: [GitHub Issues](https://github.com/alexeiveselov92/clustertk/issues)
- 💬 Discussions: [GitHub Discussions](https://github.com/alexeiveselov92/clustertk/discussions)
