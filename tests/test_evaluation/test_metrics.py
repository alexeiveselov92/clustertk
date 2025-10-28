"""
Tests for clustering evaluation metrics.
"""

import pytest
import numpy as np
from clustertk.evaluation import compute_clustering_metrics


def test_metrics_basic(sample_data_with_clusters, sample_labels):
    """Test basic metrics computation."""
    df, y_true = sample_data_with_clusters

    metrics = compute_clustering_metrics(df, sample_labels)

    # Check all metrics are present
    assert 'silhouette' in metrics
    assert 'calinski_harabasz' in metrics
    assert 'davies_bouldin' in metrics

    # Check metric ranges
    assert -1 <= metrics['silhouette'] <= 1
    assert metrics['calinski_harabasz'] > 0
    assert metrics['davies_bouldin'] > 0


def test_metrics_perfect_clustering(sample_data_with_clusters):
    """Test metrics with perfect clustering (true labels)."""
    df, y_true = sample_data_with_clusters

    metrics = compute_clustering_metrics(df, y_true)

    # Perfect clustering should have high silhouette
    assert metrics['silhouette'] > 0.5


def test_metrics_with_noise(sample_data_with_clusters, sample_labels_with_noise):
    """Test metrics computation with noise points."""
    df, y_true = sample_data_with_clusters

    # Should handle noise points (-1 labels)
    metrics = compute_clustering_metrics(df, sample_labels_with_noise)

    assert 'silhouette' in metrics
    assert isinstance(metrics['silhouette'], (int, float))


def test_metrics_single_cluster_error(sample_data_simple):
    """Test that single cluster raises appropriate error."""
    labels = np.zeros(len(sample_data_simple))

    # Single cluster should fail or return NaN
    try:
        metrics = compute_clustering_metrics(sample_data_simple, labels)
        # If it doesn't raise, silhouette should be NaN or 0
        assert np.isnan(metrics['silhouette']) or metrics['silhouette'] == 0
    except ValueError:
        # This is also acceptable
        pass


def test_metrics_two_clusters(sample_data_simple):
    """Test metrics with minimum valid clusters (2)."""
    labels = np.array([0] * 50 + [1] * 50)

    metrics = compute_clustering_metrics(sample_data_simple, labels)

    # Should work with 2 clusters
    assert 'silhouette' in metrics
    assert not np.isnan(metrics['silhouette'])
