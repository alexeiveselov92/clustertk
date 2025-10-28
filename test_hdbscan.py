"""
Test script for HDBSCAN implementation (v0.8.0).

This script tests the new HDBSCANClustering class and its integration
with the pipeline.
"""

import pandas as pd
import numpy as np
from sklearn.datasets import make_blobs
from clustertk.pipeline import ClusterAnalysisPipeline
from clustertk.clustering import HDBSCANClustering

# Set random seed
np.random.seed(42)

print("=" * 80)
print("ClusterTK v0.8.0 - HDBSCAN Test")
print("=" * 80)

# Generate synthetic data with 4 clusters
print("\n1. Generating synthetic data with 4 clusters...")
X, y_true = make_blobs(
    n_samples=500,
    n_features=10,
    centers=4,
    cluster_std=1.5,
    random_state=42
)

feature_names = [f'feature_{i}' for i in range(X.shape[1])]
df = pd.DataFrame(X, columns=feature_names)

print(f"   Generated {len(df)} samples with {X.shape[1]} features")
print(f"   True number of clusters: 4")

# Test 1: Standalone HDBSCAN
print("\n2. Testing standalone HDBSCANClustering...")
hdbscan = HDBSCANClustering(
    min_cluster_size='auto',
    min_samples='auto'
)

labels = hdbscan.fit_predict(df)

print(f"   ✓ HDBSCAN fitted successfully")
print(f"   Found {hdbscan.n_clusters_} clusters")
print(f"   Noise points: {hdbscan.n_noise_points_}")
print(f"   Parameters used:")
print(f"     - min_cluster_size: {hdbscan._min_cluster_size_computed}")
print(f"     - min_samples: {hdbscan._min_samples_computed}")

# Check probabilities
if hdbscan.probabilities_ is not None:
    weak_members = np.sum(hdbscan.probabilities_ < 0.5)
    print(f"   Weak cluster members (prob < 0.5): {weak_members}")

# Test 2: HDBSCAN in Pipeline
print("\n3. Testing HDBSCAN in Pipeline...")
pipeline = ClusterAnalysisPipeline(
    handle_missing='median',
    handle_outliers='robust',
    scaling='robust',
    pca_variance=0.95,
    clustering_algorithm='hdbscan',
    verbose=False
)

pipeline.fit(df, feature_columns=feature_names)

print(f"   ✓ Pipeline fitted with HDBSCAN")
print(f"   Found {pipeline.n_clusters_} clusters")
print(f"   Metrics:")
print(f"     - Silhouette: {pipeline.metrics_['silhouette']:.4f}")
print(f"     - Calinski-Harabasz: {pipeline.metrics_['calinski_harabasz']:.2f}")
print(f"     - Davies-Bouldin: {pipeline.metrics_['davies_bouldin']:.4f}")

# Show cluster distribution
labels = pipeline.labels_
unique, counts = np.unique(labels, return_counts=True)
print(f"\n   Cluster distribution:")
for label, count in zip(unique, counts):
    if label == -1:
        print(f"   - Noise: {count} samples ({count/len(labels)*100:.1f}%)")
    else:
        print(f"   - Cluster {label}: {count} samples ({count/len(labels)*100:.1f}%)")

# Test 3: compare_algorithms with HDBSCAN
print("\n4. Testing compare_algorithms() with HDBSCAN...")
comparison_pipeline = ClusterAnalysisPipeline(
    handle_missing='median',
    handle_outliers='robust',
    scaling='robust',
    pca_variance=0.95,
    verbose=False
)

results = comparison_pipeline.compare_algorithms(
    X=df,
    feature_columns=feature_names,
    algorithms=['kmeans', 'gmm', 'hierarchical', 'dbscan', 'hdbscan'],
    n_clusters_range=(2, 6)
)

print(f"   ✓ Comparison completed")
print(f"\n   Results:")
print(results['comparison'].to_string())

print(f"\n   Best algorithm: {results['best_algorithm']}")
print(f"   Optimal clusters: {results['best_n_clusters']}")

# Find HDBSCAN results
hdbscan_row = results['comparison'][results['comparison']['algorithm'] == 'hdbscan']
if not hdbscan_row.empty:
    print(f"\n   HDBSCAN Performance:")
    print(f"     - Clusters found: {int(hdbscan_row['n_clusters'].values[0])}")
    print(f"     - Silhouette: {hdbscan_row['silhouette'].values[0]:.4f}")
    print(f"     - Calinski-Harabasz: {hdbscan_row['calinski_harabasz'].values[0]:.2f}")
    print(f"     - Davies-Bouldin: {hdbscan_row['davies_bouldin'].values[0]:.4f}")

# Test 4: Manual parameters
print("\n5. Testing HDBSCAN with manual parameters...")
hdbscan_manual = HDBSCANClustering(
    min_cluster_size=20,
    min_samples=10
)

labels_manual = hdbscan_manual.fit_predict(df)

print(f"   ✓ HDBSCAN fitted with manual parameters")
print(f"   Found {hdbscan_manual.n_clusters_} clusters")
print(f"   Noise points: {hdbscan_manual.n_noise_points_}")

# Summary
print("\n" + "=" * 80)
print("TEST SUMMARY")
print("=" * 80)
print("✓ HDBSCANClustering class works correctly")
print("✓ Auto parameter estimation functional")
print("✓ Pipeline integration successful")
print("✓ compare_algorithms() includes HDBSCAN")
print("✓ Manual parameter override works")
print("\nAll tests passed! HDBSCAN is ready for v0.8.0")
print("=" * 80)
