"""Tests for cluster_balance_score metric."""

import pytest
import numpy as np
from clustertk.evaluation import cluster_balance_score


class TestClusterBalanceScore:
    """Tests for cluster_balance_score function."""

    def test_perfectly_balanced(self):
        """Test perfectly balanced clusters."""
        # 3 clusters, 100 samples each
        labels = np.repeat([0, 1, 2], 100)
        score = cluster_balance_score(labels)

        # Should be very close to 1.0
        assert score > 0.99
        assert score <= 1.0

    def test_moderately_balanced(self):
        """Test moderately balanced clusters."""
        # Clusters with sizes: 100, 80, 70
        labels = np.array([0]*100 + [1]*80 + [2]*70)
        score = cluster_balance_score(labels)

        # This is actually quite well balanced (entropy ≈ 0.99)
        assert 0.95 < score < 1.0

    def test_imbalanced(self):
        """Test imbalanced clusters."""
        # Cluster 0 has 280 samples, others have 10 each
        labels = np.array([0]*280 + [1]*10 + [2]*10)
        score = cluster_balance_score(labels)

        # This is very imbalanced (93% in one cluster)
        assert 0.2 < score < 0.35

    def test_highly_imbalanced(self):
        """Test highly imbalanced clusters (99% in one cluster)."""
        # Cluster 0 has 990 samples, others have 5 each
        labels = np.array([0]*990 + [1]*5 + [2]*5)
        score = cluster_balance_score(labels)

        # Should be very low
        assert 0.0 < score < 0.3

    def test_two_clusters_balanced(self):
        """Test balanced binary clustering."""
        labels = np.repeat([0, 1], 50)
        score = cluster_balance_score(labels)

        # Should be 1.0 (perfectly balanced)
        assert score > 0.99

    def test_two_clusters_imbalanced(self):
        """Test imbalanced binary clustering."""
        labels = np.array([0]*90 + [1]*10)
        score = cluster_balance_score(labels)

        # Should be relatively low
        assert 0.2 < score < 0.6

    def test_single_cluster(self):
        """Test single cluster (edge case)."""
        labels = np.zeros(100)
        score = cluster_balance_score(labels)

        # By definition, single cluster is perfectly balanced
        assert score == 1.0

    def test_with_noise_points(self):
        """Test with DBSCAN-style noise points (-1 labels)."""
        # 3 clusters + noise points
        labels = np.array([0]*100 + [1]*100 + [2]*100 + [-1]*30)
        score = cluster_balance_score(labels)

        # Should ignore noise points and be perfectly balanced
        assert score > 0.99

    def test_empty_after_filtering_noise(self):
        """Test when all points are noise."""
        labels = np.array([-1]*100)
        score = cluster_balance_score(labels)

        # Should return 0.0 (no valid clusters)
        assert score == 0.0

    def test_known_entropy_values(self):
        """Test against known entropy values."""
        # 2 clusters: 80-20 split
        labels = np.array([0]*80 + [1]*20)
        score = cluster_balance_score(labels)

        # Shannon entropy for [0.8, 0.2] = -0.8*log(0.8) - 0.2*log(0.2) ≈ 0.5004
        # Max entropy for 2 clusters = log(2) ≈ 0.6931
        # Expected score ≈ 0.5004 / 0.6931 ≈ 0.722
        assert 0.70 < score < 0.75

    def test_many_clusters(self):
        """Test with many clusters."""
        # 10 balanced clusters
        labels = np.repeat(range(10), 100)
        score = cluster_balance_score(labels)

        # Should be perfectly balanced
        assert score > 0.99

    def test_many_clusters_imbalanced(self):
        """Test with many imbalanced clusters."""
        # 5 clusters: sizes [500, 200, 150, 100, 50]
        labels = np.array([0]*500 + [1]*200 + [2]*150 + [3]*100 + [4]*50)
        score = cluster_balance_score(labels)

        # Should be imbalanced
        assert 0.5 < score < 0.85
