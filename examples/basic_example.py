"""
Basic example of using ClusterTK.

This script demonstrates the basic workflow of cluster analysis using ClusterTK.
"""

import pandas as pd
import numpy as np
from clustertk import ClusterAnalysisPipeline

# Set random seed for reproducibility
np.random.seed(42)

# Create sample data with clear clusters
print("Creating sample dataset...")
n_samples = 300
n_features = 10

# Create 3 distinct groups
group1 = np.random.randn(100, n_features) + np.array([5, 2, 1, 3, 4, 2, 1, 3, 2, 1])
group2 = np.random.randn(100, n_features) + np.array([1, 5, 4, 1, 2, 5, 4, 1, 5, 4])
group3 = np.random.randn(100, n_features) + np.array([3, 1, 5, 4, 1, 1, 5, 4, 1, 5])

data = np.vstack([group1, group2, group3])

# Create DataFrame
df = pd.DataFrame(
    data,
    columns=[f'feature_{i}' for i in range(n_features)]
)

# Add some missing values and outliers to test preprocessing
df.loc[np.random.choice(df.index, 10), 'feature_0'] = np.nan
df.loc[np.random.choice(df.index, 5), 'feature_1'] = 100  # Outlier

print(f"Dataset shape: {df.shape}")
print(f"Missing values: {df.isnull().sum().sum()}")
print()

# Create and configure pipeline
print("=" * 80)
print("RUNNING CLUSTER ANALYSIS PIPELINE")
print("=" * 80)

pipeline = ClusterAnalysisPipeline(
    # Preprocessing
    handle_missing='median',
    handle_outliers='robust',
    scaling='robust',
    log_transform_skewed=False,

    # Feature selection
    correlation_threshold=0.85,
    variance_threshold=0.01,

    # PCA
    pca_variance=0.9,
    pca_min_components=2,

    # Clustering
    clustering_algorithm='kmeans',
    n_clusters=None,  # Auto-detect
    n_clusters_range=(2, 5),

    # General
    random_state=42,
    verbose=True
)

# Run full pipeline
pipeline.fit(df)

print()
print("=" * 80)
print("RESULTS")
print("=" * 80)

# Display results
print(f"\nNumber of clusters: {pipeline.n_clusters_}")
print(f"\nClustering metrics:")
for metric, value in pipeline.metrics_.items():
    print(f"  {metric}: {value:.3f}")

print(f"\nCluster sizes:")
cluster_sizes = pd.Series(pipeline.labels_).value_counts().sort_index()
for cluster, size in cluster_sizes.items():
    pct = size / len(pipeline.labels_) * 100
    print(f"  Cluster {cluster}: {size} samples ({pct:.1f}%)")

# Show cluster profiles (top features)
print(f"\nCluster profiles (top 3 features per cluster):")
if pipeline._profiler and pipeline._profiler.top_features_:
    for cluster in range(pipeline.n_clusters_):
        print(f"\n  Cluster {cluster}:")
        print(f"    Highest features:")
        for feat, val in pipeline._profiler.top_features_[cluster]['high'][:3]:
            print(f"      ↑ {feat}: {val:+.3f}")
        print(f"    Lowest features:")
        for feat, val in pipeline._profiler.top_features_[cluster]['low'][:3]:
            print(f"      ↓ {feat}: {val:+.3f}")

print()
print("=" * 80)
print("PIPELINE COMPLETED SUCCESSFULLY!")
print("=" * 80)
