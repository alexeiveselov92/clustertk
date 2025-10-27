"""
Test script for all clustering algorithms.

This script tests all 4 clustering algorithms:
- K-Means
- GMM (Gaussian Mixture Model)
- Hierarchical Clustering
- DBSCAN
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

# Add some missing values
df.loc[np.random.choice(df.index, 10), 'feature_0'] = np.nan

print(f"Dataset shape: {df.shape}")
print(f"Missing values: {df.isnull().sum().sum()}")
print()

# Test all algorithms
algorithms = ['kmeans', 'gmm', 'hierarchical', 'dbscan']

for algo in algorithms:
    print("=" * 80)
    print(f"TESTING {algo.upper()}")
    print("=" * 80)

    try:
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
            clustering_algorithm=algo,
            n_clusters=None if algo != 'dbscan' else None,  # Auto-detect for all
            n_clusters_range=(2, 5),

            # General
            random_state=42,
            verbose=True
        )

        # Run full pipeline
        pipeline.fit(df)

        print()
        print(f"✓ {algo.upper()} RESULTS:")
        print(f"  Number of clusters: {pipeline.n_clusters_}")
        print(f"  Metrics:")
        for metric, value in pipeline.metrics_.items():
            print(f"    {metric}: {value:.3f}")

        # Show cluster sizes
        cluster_sizes = pd.Series(pipeline.labels_).value_counts().sort_index()
        print(f"  Cluster distribution:")
        for cluster, size in cluster_sizes.items():
            pct = size / len(pipeline.labels_) * 100
            if cluster == -1:
                print(f"    Noise: {size} samples ({pct:.1f}%)")
            else:
                print(f"    Cluster {cluster}: {size} samples ({pct:.1f}%)")

        print()
        print(f"✓ {algo.upper()} COMPLETED SUCCESSFULLY!")
        print()

    except Exception as e:
        print()
        print(f"✗ {algo.upper()} FAILED!")
        print(f"  Error: {e}")
        import traceback
        traceback.print_exc()
        print()

print("=" * 80)
print("ALL TESTS COMPLETED!")
print("=" * 80)
