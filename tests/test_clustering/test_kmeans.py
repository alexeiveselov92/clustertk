"""
Tests for KMeansClustering.
"""

import pytest
import numpy as np
from clustertk.clustering import KMeansClustering


def test_kmeans_basic(sample_data_with_clusters):
    """Test basic K-Means clustering."""
    df, y_true = sample_data_with_clusters

    clusterer = KMeansClustering(n_clusters=3, random_state=42)
    labels = clusterer.fit_predict(df)

    # Check correct number of clusters
    assert clusterer.n_clusters_ == 3
    assert len(np.unique(labels)) == 3

    # Check all points are assigned
    assert len(labels) == len(df)
    assert not np.any(labels < 0)


def test_kmeans_fit_then_predict(sample_data_with_clusters):
    """Test separate fit and predict."""
    df, y_true = sample_data_with_clusters

    clusterer = KMeansClustering(n_clusters=3, random_state=42)
    clusterer.fit(df)
    labels = clusterer.predict(df)

    assert len(labels) == len(df)
    assert clusterer.n_clusters_ == 3


def test_kmeans_centers(sample_data_with_clusters):
    """Test that cluster centers are computed."""
    df, y_true = sample_data_with_clusters

    clusterer = KMeansClustering(n_clusters=3, random_state=42)
    clusterer.fit(df)

    # Centers should exist
    assert hasattr(clusterer.model_, 'cluster_centers_')
    assert clusterer.model_.cluster_centers_.shape == (3, df.shape[1])


def test_kmeans_inertia(sample_data_with_clusters):
    """Test that inertia is computed."""
    df, y_true = sample_data_with_clusters

    clusterer = KMeansClustering(n_clusters=3, random_state=42)
    clusterer.fit(df)

    # Inertia should be positive
    assert clusterer.model_.inertia_ > 0


def test_kmeans_reproducibility(sample_data_with_clusters):
    """Test that results are reproducible with same random_state."""
    df, y_true = sample_data_with_clusters

    clusterer1 = KMeansClustering(n_clusters=3, random_state=42)
    labels1 = clusterer1.fit_predict(df)

    clusterer2 = KMeansClustering(n_clusters=3, random_state=42)
    labels2 = clusterer2.fit_predict(df)

    np.testing.assert_array_equal(labels1, labels2)


def test_kmeans_different_k(sample_data_with_clusters):
    """Test K-Means with different cluster numbers."""
    df, y_true = sample_data_with_clusters

    for k in [2, 3, 4, 5]:
        clusterer = KMeansClustering(n_clusters=k, random_state=42)
        labels = clusterer.fit_predict(df)

        assert len(np.unique(labels)) == k
        assert clusterer.n_clusters_ == k
