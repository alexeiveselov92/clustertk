"""
Test script to verify infinity handling in preprocessing.
"""

import pandas as pd
import numpy as np
from clustertk import ClusterAnalysisPipeline

# Set random seed
np.random.seed(42)

# Create sample data with infinity values
print("Creating sample dataset with infinity values...")
n_samples = 100
n_features = 5

# Create normal data
data = np.random.randn(n_samples, n_features)

# Add some infinity values
data[5, 0] = np.inf
data[10, 1] = -np.inf
data[15, 2] = 1e200  # Very large value

# Add some missing values
data[20, 3] = np.nan

# Create DataFrame
df = pd.DataFrame(
    data,
    columns=[f'feature_{i}' for i in range(n_features)]
)

print(f"Dataset shape: {df.shape}")
print(f"Infinity count: {np.isinf(df.values).sum()}")
print(f"Missing values: {df.isnull().sum().sum()}")
print()

# Create and configure pipeline (same as user's code)
print("=" * 80)
print("TESTING INFINITY HANDLING")
print("=" * 80)

pipeline = ClusterAnalysisPipeline(
    handle_missing='median',          # Handle missing values
    correlation_threshold=0.85,       # Remove highly correlated features
    pca_variance=0.9,                 # Keep 90% of variance
    clustering_algorithm='kmeans',    # Use K-Means
    n_clusters=None,                  # Auto-detect optimal number
    verbose=True
)

# Run complete analysis
try:
    pipeline.fit(df)
    print()
    print("=" * 80)
    print("SUCCESS! Pipeline handled infinity values correctly!")
    print("=" * 80)
    print(f"Number of clusters: {pipeline.n_clusters_}")
    print(f"Clustering metrics: {pipeline.metrics_}")
except Exception as e:
    print()
    print("=" * 80)
    print("FAILED! Pipeline still has issues with infinity values")
    print("=" * 80)
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
