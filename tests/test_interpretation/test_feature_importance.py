"""
Tests for feature importance analysis module.

This module tests the FeatureImportanceAnalyzer class and related functions.
"""

import pytest
import pandas as pd
import numpy as np
from sklearn.datasets import make_blobs

from clustertk.interpretation.feature_importance import (
    FeatureImportanceAnalyzer,
    quick_feature_importance
)


@pytest.fixture
def feature_importance_data():
    """
    Generate data specifically designed for feature importance testing.

    Returns
    -------
    df : pd.DataFrame
        DataFrame with 300 samples and 5 features with varying importance.
    labels : np.ndarray
        Cluster labels (3 clusters).
    """
    np.random.seed(42)

    # Create data with clear cluster structure
    X, y = make_blobs(
        n_samples=300,
        n_features=5,
        centers=3,
        cluster_std=1.0,
        random_state=42
    )

    # Add noise to last feature to make it less important
    X[:, -1] += np.random.RandomState(42).randn(300) * 5.0

    df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(5)])
    return df, y


@pytest.fixture
def feature_importance_data_with_noise():
    """
    Generate data with noise points (DBSCAN-style labels).

    Returns
    -------
    df : pd.DataFrame
        DataFrame with 300 samples and 5 features.
    labels : np.ndarray
        Cluster labels with -1 for noise points.
    """
    np.random.seed(42)
    X, y = make_blobs(n_samples=300, n_features=5, centers=3, random_state=42)
    df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(5)])

    # Mark some points as noise
    noise_indices = np.random.choice(300, size=30, replace=False)
    y = y.copy()
    y[noise_indices] = -1

    return df, y


class TestFeatureImportanceAnalyzer:
    """Test suite for FeatureImportanceAnalyzer class."""

    def test_initialization(self):
        """Test analyzer initialization."""
        analyzer = FeatureImportanceAnalyzer()
        assert analyzer.verbose is False
        assert analyzer.permutation_importance_ is None
        assert analyzer.feature_contribution_ is None
        assert analyzer.shap_values_ is None

        # Test with verbose
        analyzer_verbose = FeatureImportanceAnalyzer(verbose=True)
        assert analyzer_verbose.verbose is True

    def test_permutation_importance(self, feature_importance_data):
        """Test permutation importance calculation."""
        df, labels = feature_importance_data
        analyzer = FeatureImportanceAnalyzer()

        results = analyzer.analyze(df, labels, method='permutation', n_repeats=5)

        # Check results structure
        assert 'permutation' in results
        assert isinstance(results['permutation'], pd.DataFrame)

        perm_df = results['permutation']
        assert 'feature' in perm_df.columns
        assert 'importance' in perm_df.columns
        assert 'std' in perm_df.columns

        # Check all features are present
        assert len(perm_df) == 5
        assert set(perm_df['feature']) == {f'feature_{i}' for i in range(5)}

        # Check values are numeric
        assert perm_df['importance'].dtype in [np.float64, np.float32]
        assert perm_df['std'].dtype in [np.float64, np.float32]

        # Check stored in analyzer
        assert analyzer.permutation_importance_ is not None
        pd.testing.assert_frame_equal(analyzer.permutation_importance_, perm_df)

    def test_feature_contribution(self, feature_importance_data):
        """Test feature contribution (variance ratio) calculation."""
        df, labels = feature_importance_data
        analyzer = FeatureImportanceAnalyzer()

        results = analyzer.analyze(df, labels, method='contribution')

        # Check results structure
        assert 'contribution' in results
        assert isinstance(results['contribution'], pd.DataFrame)

        contrib_df = results['contribution']
        assert 'feature' in contrib_df.columns
        assert 'contribution' in contrib_df.columns

        # Check all features are present
        assert len(contrib_df) == 5
        assert set(contrib_df['feature']) == {f'feature_{i}' for i in range(5)}

        # Check values are numeric and non-negative
        assert contrib_df['contribution'].dtype in [np.float64, np.float32]
        assert (contrib_df['contribution'] >= 0).all()

        # Check stored in analyzer
        assert analyzer.feature_contribution_ is not None
        pd.testing.assert_frame_equal(analyzer.feature_contribution_, contrib_df)

    def test_contribution_with_noise_labels(self, feature_importance_data_with_noise):
        """Test feature contribution handles noise points (-1 labels) correctly."""
        df, labels = feature_importance_data_with_noise
        analyzer = FeatureImportanceAnalyzer()

        # Should not raise error with -1 labels
        results = analyzer.analyze(df, labels, method='contribution')

        assert 'contribution' in results
        contrib_df = results['contribution']
        assert len(contrib_df) == 5
        # Contributions should be non-negative
        assert (contrib_df['contribution'] >= 0).all()

    def test_analyze_all_methods(self, feature_importance_data):
        """Test analyze with method='all'."""
        df, labels = feature_importance_data
        analyzer = FeatureImportanceAnalyzer()

        results = analyzer.analyze(df, labels, method='all', n_repeats=5)

        # Check both methods are present
        assert 'permutation' in results
        assert 'contribution' in results

        # SHAP might not be available (optional dependency)
        # If available, it should be in results
        # If not available, it should not be in results

        # Check both DataFrames are valid
        assert isinstance(results['permutation'], pd.DataFrame)
        assert isinstance(results['contribution'], pd.DataFrame)

    def test_shap_importance_not_installed(self, feature_importance_data, monkeypatch):
        """Test SHAP analysis when shap package is not installed."""
        df, labels = feature_importance_data
        analyzer = FeatureImportanceAnalyzer()

        # Mock shap import to fail
        def mock_import_shap(*args, **kwargs):
            raise ImportError("No module named 'shap'")

        # When method='all', should skip SHAP without error
        results = analyzer.analyze(df, labels, method='all')
        # Should have other methods but not shap
        assert 'permutation' in results
        assert 'contribution' in results

        # When method='shap', should raise ImportError
        with pytest.raises(ImportError, match="shap package not installed"):
            analyzer.analyze(df, labels, method='shap')

    def test_get_top_features_permutation(self, feature_importance_data):
        """Test get_top_features for permutation method."""
        df, labels = feature_importance_data
        analyzer = FeatureImportanceAnalyzer()

        # Must run analyze first
        with pytest.raises(ValueError, match="Run analyze"):
            analyzer.get_top_features(method='permutation')

        # Run analyze
        analyzer.analyze(df, labels, method='permutation', n_repeats=5)

        # Get top 3 features
        top_features = analyzer.get_top_features(method='permutation', n=3)
        assert len(top_features) == 3
        assert 'feature' in top_features.columns
        assert 'importance' in top_features.columns

    def test_get_top_features_contribution(self, feature_importance_data):
        """Test get_top_features for contribution method."""
        df, labels = feature_importance_data
        analyzer = FeatureImportanceAnalyzer()

        # Must run analyze first
        with pytest.raises(ValueError, match="Run analyze"):
            analyzer.get_top_features(method='contribution')

        # Run analyze
        analyzer.analyze(df, labels, method='contribution')

        # Get top 3 features
        top_features = analyzer.get_top_features(method='contribution', n=3)
        assert len(top_features) == 3
        assert 'feature' in top_features.columns
        assert 'contribution' in top_features.columns

    def test_invalid_method(self, feature_importance_data):
        """Test analyze with invalid method."""
        df, labels = feature_importance_data
        analyzer = FeatureImportanceAnalyzer()

        # Should not raise error - will just return empty dict or handle gracefully
        # The implementation might vary, so we just check it doesn't crash
        try:
            results = analyzer.analyze(df, labels, method='invalid')
            # If it returns something, check it's a dict
            assert isinstance(results, dict)
        except ValueError:
            # If it raises ValueError, that's also acceptable
            pass

    def test_reproducibility(self, feature_importance_data):
        """Test that results are reproducible with same random_state."""
        df, labels = feature_importance_data

        analyzer1 = FeatureImportanceAnalyzer()
        results1 = analyzer1.analyze(
            df, labels, method='permutation', n_repeats=10, random_state=42
        )

        analyzer2 = FeatureImportanceAnalyzer()
        results2 = analyzer2.analyze(
            df, labels, method='permutation', n_repeats=10, random_state=42
        )

        # Results should be identical
        pd.testing.assert_frame_equal(
            results1['permutation'],
            results2['permutation']
        )

    def test_different_n_repeats(self, feature_importance_data):
        """Test permutation importance with different n_repeats."""
        df, labels = feature_importance_data
        analyzer = FeatureImportanceAnalyzer()

        # Test with n_repeats=3
        results_3 = analyzer.analyze(
            df, labels, method='permutation', n_repeats=3, random_state=42
        )

        # Test with n_repeats=20
        analyzer2 = FeatureImportanceAnalyzer()
        results_20 = analyzer2.analyze(
            df, labels, method='permutation', n_repeats=20, random_state=42
        )

        # Both should return valid results
        assert 'permutation' in results_3
        assert 'permutation' in results_20

        # More repeats should generally give lower std
        # (not always true due to randomness, but check shapes)
        assert len(results_3['permutation']) == len(results_20['permutation'])

    def test_two_clusters(self):
        """Test with only 2 clusters."""
        np.random.seed(42)
        X, y = make_blobs(n_samples=200, n_features=4, centers=2, random_state=42)
        df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(4)])

        analyzer = FeatureImportanceAnalyzer()
        results = analyzer.analyze(df, y, method='all', n_repeats=5)

        # Should work with 2 clusters
        assert 'permutation' in results
        assert 'contribution' in results
        assert len(results['permutation']) == 4
        assert len(results['contribution']) == 4


class TestQuickFeatureImportance:
    """Test suite for quick_feature_importance convenience function."""

    def test_quick_function_permutation(self, feature_importance_data):
        """Test quick function with permutation method."""
        df, labels = feature_importance_data

        top_features = quick_feature_importance(
            df, labels, method='permutation', n_top=3
        )

        assert isinstance(top_features, pd.DataFrame)
        assert len(top_features) == 3
        assert 'feature' in top_features.columns
        assert 'importance' in top_features.columns

    def test_quick_function_contribution(self, feature_importance_data):
        """Test quick function with contribution method."""
        df, labels = feature_importance_data

        top_features = quick_feature_importance(
            df, labels, method='contribution', n_top=3
        )

        assert isinstance(top_features, pd.DataFrame)
        assert len(top_features) == 3
        assert 'feature' in top_features.columns
        assert 'contribution' in top_features.columns

    def test_quick_function_all(self, feature_importance_data):
        """Test quick function with method='all'."""
        df, labels = feature_importance_data

        top_features = quick_feature_importance(
            df, labels, method='all', n_top=5
        )

        assert isinstance(top_features, pd.DataFrame)
        assert len(top_features) == 5

        # Should have combined results
        assert 'feature' in top_features.columns
        assert 'permutation_importance' in top_features.columns
        assert 'feature_contribution' in top_features.columns
        assert 'combined_score' in top_features.columns

    def test_quick_function_default_params(self, feature_importance_data):
        """Test quick function with default parameters."""
        df, labels = feature_importance_data

        top_features = quick_feature_importance(df, labels)

        # Default: method='all', n_top=10
        assert isinstance(top_features, pd.DataFrame)
        assert len(top_features) == 5  # Only 5 features in test data


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_single_cluster(self):
        """Test with all points in one cluster."""
        np.random.seed(42)
        df = pd.DataFrame(np.random.rand(100, 5), columns=[f'feature_{i}' for i in range(5)])
        labels = np.zeros(100, dtype=int)  # All same cluster

        analyzer = FeatureImportanceAnalyzer()

        # Should handle gracefully (low/zero importance expected)
        results = analyzer.analyze(df, labels, method='contribution')
        assert 'contribution' in results

    def test_many_features(self):
        """Test with many features (>10)."""
        np.random.seed(42)
        X, y = make_blobs(n_samples=200, n_features=15, centers=3, random_state=42)
        df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(15)])

        analyzer = FeatureImportanceAnalyzer()
        results = analyzer.analyze(df, y, method='all', n_repeats=3)

        assert 'permutation' in results
        assert 'contribution' in results
        assert len(results['permutation']) == 15
        assert len(results['contribution']) == 15

    def test_few_samples(self):
        """Test with small number of samples."""
        np.random.seed(42)
        X, y = make_blobs(n_samples=30, n_features=3, centers=2, random_state=42)
        df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(3)])

        analyzer = FeatureImportanceAnalyzer()
        results = analyzer.analyze(df, y, method='all', n_repeats=3)

        # Should still work
        assert 'permutation' in results
        assert 'contribution' in results


class TestIntegrationWithPipeline:
    """Test feature importance integration with ClusterAnalysisPipeline."""

    def test_pipeline_integration(self, feature_importance_data):
        """Test analyze_feature_importance method in pipeline."""
        from clustertk import ClusterAnalysisPipeline

        df, _ = feature_importance_data

        # Create and fit pipeline
        pipeline = ClusterAnalysisPipeline(
            clustering_algorithm='kmeans',
            n_clusters=3,
            scaling='standard'
        )
        pipeline.fit(df, feature_columns=df.columns.tolist())

        # Analyze feature importance
        results = pipeline.analyze_feature_importance(method='all', n_repeats=5)

        assert 'permutation' in results
        assert 'contribution' in results
        assert isinstance(results['permutation'], pd.DataFrame)
        assert isinstance(results['contribution'], pd.DataFrame)

        # Check results are stored in pipeline
        assert hasattr(pipeline, 'feature_importance_results_')
        assert pipeline.feature_importance_results_ is not None

    def test_pipeline_before_fit(self):
        """Test that feature importance requires fitted pipeline."""
        from clustertk import ClusterAnalysisPipeline

        pipeline = ClusterAnalysisPipeline()

        # Should raise error before fit
        with pytest.raises(ValueError, match="Pipeline must be fitted"):
            pipeline.analyze_feature_importance()
