"""
Tests for cluster stability analysis module.

This module tests the ClusterStabilityAnalyzer class and related functions.
"""

import pytest
import pandas as pd
import numpy as np
from sklearn.datasets import make_blobs

from clustertk.evaluation.stability import (
    ClusterStabilityAnalyzer,
    quick_stability_analysis
)
from clustertk.clustering import KMeansClustering


@pytest.fixture
def stability_data():
    """
    Generate data for stability testing.

    Returns
    -------
    df : pd.DataFrame
        DataFrame with 300 samples and 5 features, 3 clusters.
    y : np.ndarray
        True cluster labels.
    """
    np.random.seed(42)
    X, y = make_blobs(
        n_samples=300,
        n_features=5,
        centers=3,
        cluster_std=1.0,
        random_state=42
    )
    df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(5)])
    return df, y


@pytest.fixture
def stability_data_noisy():
    """
    Generate data with more noise for lower stability.

    Returns
    -------
    df : pd.DataFrame
        DataFrame with 200 samples and 4 features, noisy clusters.
    y : np.ndarray
        True cluster labels.
    """
    np.random.seed(42)
    X, y = make_blobs(
        n_samples=200,
        n_features=4,
        centers=3,
        cluster_std=2.5,  # Higher std = more overlap
        random_state=42
    )
    df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(4)])
    return df, y


class TestClusterStabilityAnalyzer:
    """Test suite for ClusterStabilityAnalyzer class."""

    def test_initialization(self):
        """Test analyzer initialization."""
        analyzer = ClusterStabilityAnalyzer()
        assert analyzer.n_iterations == 100
        assert analyzer.sample_fraction == 0.8
        assert analyzer.verbose is False
        assert analyzer.overall_stability_ is None
        assert analyzer.cluster_stability_ is None
        assert analyzer.sample_confidence_ is None

        # Test custom parameters
        analyzer_custom = ClusterStabilityAnalyzer(
            n_iterations=50,
            sample_fraction=0.7,
            verbose=True,
            random_state=123
        )
        assert analyzer_custom.n_iterations == 50
        assert analyzer_custom.sample_fraction == 0.7
        assert analyzer_custom.verbose is True
        assert analyzer_custom.random_state == 123

    def test_analyze_basic(self, stability_data):
        """Test basic stability analysis."""
        df, y_true = stability_data
        clusterer = KMeansClustering(n_clusters=3, random_state=42)

        analyzer = ClusterStabilityAnalyzer(n_iterations=10, random_state=42)
        results = analyzer.analyze(df, clusterer)

        # Check results structure
        assert 'overall_stability' in results
        assert 'cluster_stability' in results
        assert 'sample_confidence' in results
        assert 'mean_ari' in results
        assert 'stable_clusters' in results
        assert 'unstable_clusters' in results
        assert 'reference_labels' in results

        # Check overall stability
        assert 0 <= results['overall_stability'] <= 1
        assert 0 <= results['mean_ari'] <= 1

        # Check cluster stability DataFrame
        cluster_stab = results['cluster_stability']
        assert isinstance(cluster_stab, pd.DataFrame)
        assert 'cluster' in cluster_stab.columns
        assert 'stability' in cluster_stab.columns
        assert 'size' in cluster_stab.columns
        assert len(cluster_stab) == 3  # 3 clusters

        # Check sample confidence
        conf = results['sample_confidence']
        assert len(conf) == len(df)
        assert np.all((conf >= 0) & (conf <= 1))

    def test_analyze_reproducibility(self, stability_data):
        """Test that results are reproducible with same random_state."""
        df, y_true = stability_data
        clusterer = KMeansClustering(n_clusters=3, random_state=42)

        analyzer1 = ClusterStabilityAnalyzer(n_iterations=10, random_state=42)
        results1 = analyzer1.analyze(df, clusterer)

        analyzer2 = ClusterStabilityAnalyzer(n_iterations=10, random_state=42)
        results2 = analyzer2.analyze(df, clusterer)

        # Should get same results
        assert abs(results1['overall_stability'] - results2['overall_stability']) < 0.01
        assert abs(results1['mean_ari'] - results2['mean_ari']) < 0.01

    def test_analyze_different_iterations(self, stability_data):
        """Test with different number of iterations."""
        df, y_true = stability_data
        clusterer = KMeansClustering(n_clusters=3, random_state=42)

        # Test with 5 iterations
        analyzer_5 = ClusterStabilityAnalyzer(n_iterations=5, random_state=42)
        results_5 = analyzer_5.analyze(df, clusterer)

        # Test with 20 iterations
        analyzer_20 = ClusterStabilityAnalyzer(n_iterations=20, random_state=42)
        results_20 = analyzer_20.analyze(df, clusterer)

        # Both should produce valid results
        assert 0 <= results_5['overall_stability'] <= 1
        assert 0 <= results_20['overall_stability'] <= 1

        # More iterations shouldn't drastically change results (for stable data)
        # But we won't assert exact equality due to sampling variance

    def test_high_stability_data(self, stability_data):
        """Test that well-separated data has high stability."""
        df, y_true = stability_data
        clusterer = KMeansClustering(n_clusters=3, random_state=42)

        analyzer = ClusterStabilityAnalyzer(n_iterations=20, random_state=42)
        results = analyzer.analyze(df, clusterer)

        # Well-separated clusters should have high stability
        assert results['overall_stability'] > 0.7
        # Most clusters should be stable
        assert len(results['stable_clusters']) >= 2

    def test_low_stability_data(self, stability_data_noisy):
        """Test that noisy/overlapping data has lower stability."""
        df, y_true = stability_data_noisy
        clusterer = KMeansClustering(n_clusters=3, random_state=42)

        analyzer = ClusterStabilityAnalyzer(n_iterations=15, random_state=42)
        results = analyzer.analyze(df, clusterer)

        # Noisy data should have lower stability than clean data
        # (though still > 0 since there is some structure)
        assert 0 < results['overall_stability'] < 1

    def test_cluster_stability_properties(self, stability_data):
        """Test properties of per-cluster stability."""
        df, y_true = stability_data
        clusterer = KMeansClustering(n_clusters=3, random_state=42)

        analyzer = ClusterStabilityAnalyzer(n_iterations=15, random_state=42)
        results = analyzer.analyze(df, clusterer)

        cluster_stab = results['cluster_stability']

        # All stability scores should be between 0 and 1
        assert (cluster_stab['stability'] >= 0).all()
        assert (cluster_stab['stability'] <= 1).all()

        # Cluster sizes should sum to total samples
        assert cluster_stab['size'].sum() == len(df)

        # Clusters should be sorted by stability
        stabilities = cluster_stab['stability'].values
        assert all(stabilities[i] >= stabilities[i+1] for i in range(len(stabilities)-1))

    def test_sample_confidence_properties(self, stability_data):
        """Test properties of per-sample confidence scores."""
        df, y_true = stability_data
        clusterer = KMeansClustering(n_clusters=3, random_state=42)

        analyzer = ClusterStabilityAnalyzer(n_iterations=15, random_state=42)
        results = analyzer.analyze(df, clusterer)

        conf = results['sample_confidence']

        # All confidence scores between 0 and 1
        assert np.all(conf >= 0)
        assert np.all(conf <= 1)

        # Should have confidence for every sample
        assert len(conf) == len(df)

        # Mean confidence should be reasonable (> 0)
        assert np.mean(conf) > 0

    def test_get_stable_samples(self, stability_data):
        """Test get_stable_samples method."""
        df, y_true = stability_data
        clusterer = KMeansClustering(n_clusters=3, random_state=42)

        analyzer = ClusterStabilityAnalyzer(n_iterations=15, random_state=42)
        analyzer.analyze(df, clusterer)

        # Get stable samples
        stable_indices = analyzer.get_stable_samples(threshold=0.7)

        assert isinstance(stable_indices, np.ndarray)
        assert len(stable_indices) > 0
        assert len(stable_indices) <= len(df)

        # All stable samples should have confidence >= threshold
        conf = analyzer.sample_confidence_
        assert np.all(conf[stable_indices] >= 0.7)

    def test_get_unstable_samples(self, stability_data):
        """Test get_unstable_samples method."""
        df, y_true = stability_data
        clusterer = KMeansClustering(n_clusters=3, random_state=42)

        analyzer = ClusterStabilityAnalyzer(n_iterations=15, random_state=42)
        analyzer.analyze(df, clusterer)

        # Get unstable samples
        unstable_indices = analyzer.get_unstable_samples(threshold=0.5)

        assert isinstance(unstable_indices, np.ndarray)
        # May or may not have unstable samples depending on data
        if len(unstable_indices) > 0:
            conf = analyzer.sample_confidence_
            assert np.all(conf[unstable_indices] < 0.5)

    def test_get_samples_before_analyze_raises_error(self):
        """Test that accessing samples before analyze raises error."""
        analyzer = ClusterStabilityAnalyzer()

        with pytest.raises(ValueError, match="Run analyze"):
            analyzer.get_stable_samples()

        with pytest.raises(ValueError, match="Run analyze"):
            analyzer.get_unstable_samples()

    def test_two_clusters(self):
        """Test with 2 clusters."""
        np.random.seed(42)
        X, y = make_blobs(n_samples=200, n_features=4, centers=2, random_state=42)
        df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(4)])

        clusterer = KMeansClustering(n_clusters=2, random_state=42)
        analyzer = ClusterStabilityAnalyzer(n_iterations=10, random_state=42)
        results = analyzer.analyze(df, clusterer)

        # Should work with 2 clusters
        assert len(results['cluster_stability']) == 2
        assert results['overall_stability'] > 0

    def test_many_clusters(self):
        """Test with many clusters."""
        np.random.seed(42)
        X, y = make_blobs(n_samples=500, n_features=6, centers=8, random_state=42)
        df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(6)])

        clusterer = KMeansClustering(n_clusters=8, random_state=42)
        analyzer = ClusterStabilityAnalyzer(n_iterations=10, random_state=42)
        results = analyzer.analyze(df, clusterer)

        # Should work with many clusters
        assert len(results['cluster_stability']) == 8
        assert 0 <= results['overall_stability'] <= 1

    def test_small_sample_size(self):
        """Test with small number of samples."""
        np.random.seed(42)
        X, y = make_blobs(n_samples=50, n_features=3, centers=2, random_state=42)
        df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(3)])

        clusterer = KMeansClustering(n_clusters=2, random_state=42)
        analyzer = ClusterStabilityAnalyzer(
            n_iterations=5,
            sample_fraction=0.6,  # Lower fraction for small data
            random_state=42
        )
        results = analyzer.analyze(df, clusterer)

        # Should still work
        assert 'overall_stability' in results
        assert len(results['sample_confidence']) == 50

    def test_different_sample_fractions(self, stability_data):
        """Test with different sample fractions."""
        df, y_true = stability_data
        clusterer = KMeansClustering(n_clusters=3, random_state=42)

        # Test with 60% samples
        analyzer_60 = ClusterStabilityAnalyzer(
            n_iterations=10,
            sample_fraction=0.6,
            random_state=42
        )
        results_60 = analyzer_60.analyze(df, clusterer)

        # Test with 90% samples
        analyzer_90 = ClusterStabilityAnalyzer(
            n_iterations=10,
            sample_fraction=0.9,
            random_state=42
        )
        results_90 = analyzer_90.analyze(df, clusterer)

        # Both should produce valid results
        assert 0 <= results_60['overall_stability'] <= 1
        assert 0 <= results_90['overall_stability'] <= 1


class TestQuickStabilityAnalysis:
    """Test suite for quick_stability_analysis convenience function."""

    def test_quick_function_basic(self, stability_data):
        """Test basic quick function usage."""
        df, y_true = stability_data
        clusterer = KMeansClustering(n_clusters=3, random_state=42)

        results = quick_stability_analysis(
            df,
            clusterer,
            n_iterations=10,
            random_state=42
        )

        # Should return same structure as analyze()
        assert 'overall_stability' in results
        assert 'cluster_stability' in results
        assert 'sample_confidence' in results
        assert 'mean_ari' in results

    def test_quick_function_with_defaults(self, stability_data):
        """Test quick function with default parameters."""
        df, y_true = stability_data
        clusterer = KMeansClustering(n_clusters=3, random_state=42)

        # Use defaults
        results = quick_stability_analysis(df, clusterer)

        assert 'overall_stability' in results
        assert 0 <= results['overall_stability'] <= 1


class TestIntegrationWithPipeline:
    """Test stability analysis integration with ClusterAnalysisPipeline."""

    def test_pipeline_integration(self, stability_data):
        """Test analyze_stability method in pipeline."""
        from clustertk import ClusterAnalysisPipeline

        df, _ = stability_data

        # Create and fit pipeline
        pipeline = ClusterAnalysisPipeline(
            clustering_algorithm='kmeans',
            n_clusters=3,
            scaling='standard',
            random_state=42
        )
        pipeline.fit(df, feature_columns=df.columns.tolist())

        # Analyze stability
        results = pipeline.analyze_stability(n_iterations=10, random_state=42)

        # Check results
        assert 'overall_stability' in results
        assert 'cluster_stability' in results
        assert isinstance(results['cluster_stability'], pd.DataFrame)

        # Check results are stored in pipeline
        assert hasattr(pipeline, 'stability_results_')
        assert pipeline.stability_results_ is not None

    def test_pipeline_before_fit(self):
        """Test that stability analysis requires fitted pipeline."""
        from clustertk import ClusterAnalysisPipeline

        pipeline = ClusterAnalysisPipeline()

        # Should raise error before fit
        with pytest.raises(ValueError, match="Pipeline must be fitted"):
            pipeline.analyze_stability()

    def test_pipeline_stores_analyzer(self, stability_data):
        """Test that pipeline stores the analyzer object."""
        from clustertk import ClusterAnalysisPipeline

        df, _ = stability_data

        pipeline = ClusterAnalysisPipeline(
            clustering_algorithm='kmeans',
            n_clusters=3,
            random_state=42
        )
        pipeline.fit(df, feature_columns=df.columns.tolist())
        pipeline.analyze_stability(n_iterations=5, random_state=42)

        # Should have analyzer stored
        assert hasattr(pipeline, 'stability_analyzer_')
        assert pipeline.stability_analyzer_ is not None
        assert isinstance(pipeline.stability_analyzer_, ClusterStabilityAnalyzer)
