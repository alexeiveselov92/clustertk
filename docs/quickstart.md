# Quick Start Guide

Get started with ClusterTK in 5 minutes!

## Installation

```bash
pip install clustertk
```

For visualization support:
```bash
pip install clustertk[viz]
```

## Basic Usage

### Complete Pipeline

Run the entire clustering workflow with one method:

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

print(f"Found {pipeline.n_clusters_} clusters")
print(f"Silhouette score: {metrics['silhouette']:.3f}")
```

### Export Results

```python
# Export to CSV
pipeline.export_results('results.csv')

# Export to JSON with metadata
pipeline.export_results('results.json', format='json')

# Generate HTML report with plots
pipeline.export_report('report.html')
```

### Visualization

```python
# Requires: pip install clustertk[viz]
from IPython.display import display

# 2D cluster visualization
display(pipeline.plot_clusters_2d())

# Cluster profiles heatmap
display(pipeline.plot_cluster_heatmap())

# Cluster size distribution
display(pipeline.plot_cluster_sizes())
```

## Step-by-Step Workflow

For more control, run the pipeline step-by-step:

```python
from clustertk import ClusterAnalysisPipeline

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
pipeline.create_profiles()

# Access intermediate results
preprocessed = pipeline.data_preprocessed_
pca_components = pipeline.data_reduced_
```

## Common Use Cases

### Use Case 1: Customer Segmentation

```python
from clustertk import ClusterAnalysisPipeline

# Configure for customer data
pipeline = ClusterAnalysisPipeline(
    handle_missing='median',
    scaling='robust',              # Robust to outliers
    pca_variance=0.95,
    clustering_algorithm='kmeans',
    n_clusters=None,               # Auto-detect
    auto_name_clusters=True        # Generate cluster names
)

# Fit on customer features
pipeline.fit(
    customers_df,
    feature_columns=['age', 'income', 'purchases', 'engagement'],
    category_mapping={
        'demographics': ['age', 'income'],
        'behavior': ['purchases', 'engagement']
    }
)

# Get named segments
for cluster_id in range(pipeline.n_clusters_):
    name = pipeline.get_cluster_name(cluster_id)
    size = (pipeline.labels_ == cluster_id).sum()
    print(f"Segment {cluster_id}: {name} ({size} customers)")
```

### Use Case 2: Anomaly Detection with DBSCAN

```python
from clustertk import ClusterAnalysisPipeline

# Configure for anomaly detection
pipeline = ClusterAnalysisPipeline(
    clustering_algorithm='dbscan',  # Density-based clustering
    handle_outliers='robust',
    scaling='standard'
)

pipeline.fit(df, feature_columns=numeric_columns)

# Identify anomalies (cluster -1 in DBSCAN)
anomalies = df[pipeline.labels_ == -1]
normal = df[pipeline.labels_ != -1]

print(f"Found {len(anomalies)} anomalies out of {len(df)} samples")
```

### Use Case 3: Comparing Algorithms

```python
# Try different algorithms
algorithms = ['kmeans', 'gmm', 'hierarchical', 'dbscan']

results = {}
for algo in algorithms:
    pipeline = ClusterAnalysisPipeline(
        clustering_algorithm=algo,
        n_clusters=5 if algo != 'dbscan' else None
    )
    pipeline.fit(df, feature_columns=features)
    results[algo] = {
        'silhouette': pipeline.metrics_['silhouette'],
        'n_clusters': pipeline.n_clusters_
    }

# Compare results
import pandas as pd
comparison = pd.DataFrame(results).T
print(comparison)
```

## Configuration Options

### Preprocessing

```python
pipeline = ClusterAnalysisPipeline(
    handle_missing='median',          # 'median', 'mean', 'drop', or callable
    handle_outliers='robust',         # 'robust', 'clip', 'remove', or None
    scaling='robust',                 # 'standard', 'robust', 'minmax', 'auto'
    log_transform_skewed=True,        # Apply log transform to skewed features
    skewness_threshold=2.0            # Threshold for skewness detection
)
```

### Feature Selection

```python
pipeline = ClusterAnalysisPipeline(
    correlation_threshold=0.85,       # Remove features with |corr| > 0.85
    variance_threshold=0.01           # Remove low-variance features
)
```

### Dimensionality Reduction

```python
pipeline = ClusterAnalysisPipeline(
    pca_variance=0.9,                 # Keep 90% of variance
    pca_min_components=2              # Minimum components (for visualization)
)
```

### Clustering

```python
pipeline = ClusterAnalysisPipeline(
    clustering_algorithm='kmeans',    # 'kmeans', 'gmm', 'hierarchical', 'dbscan'
    n_clusters=5,                     # Number of clusters (or None for auto)
    n_clusters_range=(2, 10),         # Range for optimal k search
    random_state=42                   # For reproducibility
)
```

### Cluster Naming

```python
pipeline = ClusterAnalysisPipeline(
    auto_name_clusters=True,          # Generate descriptive names
    naming_max_features=2             # Max features in name
)
```

## Save and Load Pipeline

```python
# Save fitted pipeline
pipeline.save_pipeline('my_pipeline.joblib')

# Load later
from clustertk import ClusterAnalysisPipeline
loaded_pipeline = ClusterAnalysisPipeline.load_pipeline('my_pipeline.joblib')

# Use loaded pipeline
new_labels = loaded_pipeline.labels_
```

## Next Steps

- **User Guide**: Detailed documentation for each component
  - [Preprocessing](user_guide/preprocessing.md)
  - [Clustering](user_guide/clustering.md)
  - [Evaluation](user_guide/evaluation.md)
  - [Visualization](user_guide/visualization.md)
  - [Export](user_guide/export.md)

- **[API Reference](api_reference.md)**: Complete API documentation

- **[Examples](examples.md)**: Real-world examples and use cases

- **[FAQ](faq.md)**: Frequently asked questions

## Getting Help

- **GitHub Issues**: [Report bugs or request features](https://github.com/alexeiveselov92/clustertk/issues)
- **GitHub Discussions**: [Ask questions](https://github.com/alexeiveselov92/clustertk/discussions)
- **Email**: alexei.veselov92@gmail.com
