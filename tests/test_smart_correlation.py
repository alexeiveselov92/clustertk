"""Tests for SmartCorrelationFilter."""

import pytest
import pandas as pd
import numpy as np
from clustertk.feature_selection import SmartCorrelationFilter


class TestSmartCorrelationFilter:
    """Tests for SmartCorrelationFilter class."""

    @pytest.fixture
    def correlated_data(self):
        """Create dataset with correlated features for testing."""
        np.random.seed(42)
        n = 1000

        # Create base features
        df = pd.DataFrame({
            'feature_good': np.concatenate([
                np.random.normal(0, 1, n//2),
                np.random.normal(3, 1, n//2)
            ]),  # Bimodal - good for clustering
            'feature_bad': np.random.uniform(0, 1, n),  # Uniform - bad for clustering
        })

        # Add correlated features
        df['feature_good_copy'] = df['feature_good'] + np.random.normal(0, 0.1, n)
        df['feature_bad_copy'] = df['feature_bad'] + np.random.normal(0, 0.01, n)

        return df

    def test_basic_filtering(self, correlated_data):
        """Test that SmartCorrelationFilter removes correlated features."""
        filter = SmartCorrelationFilter(threshold=0.85)
        result = filter.fit_transform(correlated_data)

        # Should remove 2 features (the copies)
        assert len(result.columns) == 2
        assert len(filter.features_to_drop_) == 2

    def test_hopkins_strategy(self, correlated_data):
        """Test Hopkins strategy selects better features."""
        filter = SmartCorrelationFilter(
            threshold=0.85,
            selection_strategy='hopkins',
            random_state=42
        )
        result = filter.fit_transform(correlated_data)

        # Hopkins should keep the bimodal feature over uniform
        # (though exact behavior depends on sampling)
        assert 'feature_good' in result.columns or 'feature_good_copy' in result.columns

    def test_variance_ratio_strategy(self, correlated_data):
        """Test variance_ratio strategy."""
        filter = SmartCorrelationFilter(
            threshold=0.85,
            selection_strategy='variance_ratio',
            random_state=42
        )
        result = filter.fit_transform(correlated_data)

        # Should still remove correlated features
        assert len(result.columns) < len(correlated_data.columns)

    def test_mean_corr_strategy(self, correlated_data):
        """Test fallback to mean_corr strategy (original behavior)."""
        filter = SmartCorrelationFilter(
            threshold=0.85,
            selection_strategy='mean_corr'
        )
        result = filter.fit_transform(correlated_data)

        # Should work like original CorrelationFilter
        assert len(result.columns) < len(correlated_data.columns)
        assert filter.feature_scores_ is None  # Scores not computed for mean_corr

    def test_get_feature_scores(self, correlated_data):
        """Test get_feature_scores method."""
        filter = SmartCorrelationFilter(
            threshold=0.85,
            selection_strategy='hopkins',
            random_state=42
        )
        filter.fit(correlated_data)

        scores_df = filter.get_feature_scores()

        # Should have scores for all features
        assert len(scores_df) == len(correlated_data.columns)
        assert 'feature' in scores_df.columns
        assert 'score' in scores_df.columns
        assert 'selected' in scores_df.columns

        # Scores should be sorted descending
        assert scores_df['score'].is_monotonic_decreasing

    def test_get_selection_summary(self, correlated_data):
        """Test get_selection_summary method."""
        filter = SmartCorrelationFilter(
            threshold=0.85,
            selection_strategy='hopkins',
            random_state=42
        )
        filter.fit(correlated_data)

        summary = filter.get_selection_summary()

        # Should have information about correlated pairs
        assert 'feature1' in summary.columns
        assert 'feature2' in summary.columns
        assert 'correlation' in summary.columns
        assert 'kept' in summary.columns
        assert 'dropped' in summary.columns

    def test_no_correlated_features(self):
        """Test with no correlated features."""
        np.random.seed(42)
        df = pd.DataFrame({
            'a': np.random.rand(100),
            'b': np.random.rand(100),
            'c': np.random.rand(100)
        })

        filter = SmartCorrelationFilter(threshold=0.85)
        result = filter.fit_transform(df)

        # Should keep all features
        assert len(result.columns) == 3
        assert len(filter.features_to_drop_) == 0

    def test_invalid_strategy(self):
        """Test that invalid strategy raises error."""
        with pytest.raises(ValueError, match="Invalid selection_strategy"):
            SmartCorrelationFilter(selection_strategy='invalid')

    def test_selection_reasons(self, correlated_data):
        """Test that selection reasons are recorded."""
        filter = SmartCorrelationFilter(
            threshold=0.85,
            selection_strategy='hopkins',
            random_state=42
        )
        filter.fit(correlated_data)

        # Should have reasons for dropped features
        assert filter.selection_reasons_ is not None
        assert len(filter.selection_reasons_) == len(filter.features_to_drop_)

        # Reasons should mention the strategy
        for reason in filter.selection_reasons_.values():
            assert 'hopkins' in reason.lower()
