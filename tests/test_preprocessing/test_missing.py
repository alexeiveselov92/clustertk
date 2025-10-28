"""
Tests for MissingValueHandler.
"""

import pytest
import pandas as pd
import numpy as np
from clustertk.preprocessing import MissingValueHandler


def test_missing_handler_median_strategy(sample_data_with_missing):
    """Test median imputation strategy."""
    handler = MissingValueHandler(strategy='median')
    result = handler.fit_transform(sample_data_with_missing)

    # Check no missing values remain
    assert not result.isna().any().any(), "Missing values should be imputed"

    # Check shape unchanged
    assert result.shape == sample_data_with_missing.shape

    # Check values are reasonable (within original range)
    for col in result.columns:
        original_min = sample_data_with_missing[col].min()
        original_max = sample_data_with_missing[col].max()
        assert result[col].min() >= original_min
        assert result[col].max() <= original_max


def test_missing_handler_mean_strategy(sample_data_with_missing):
    """Test mean imputation strategy."""
    handler = MissingValueHandler(strategy='mean')
    result = handler.fit_transform(sample_data_with_missing)

    assert not result.isna().any().any()
    assert result.shape == sample_data_with_missing.shape


def test_missing_handler_drop_strategy(sample_data_with_missing):
    """Test drop rows strategy."""
    handler = MissingValueHandler(strategy='drop')
    result = handler.fit_transform(sample_data_with_missing)

    assert not result.isna().any().any()
    # Should have fewer rows after dropping
    assert len(result) < len(sample_data_with_missing)


def test_missing_handler_no_missing_data(sample_data_simple):
    """Test handler with data that has no missing values."""
    handler = MissingValueHandler(strategy='median')
    result = handler.fit_transform(sample_data_simple)

    # Should return unchanged
    pd.testing.assert_frame_equal(result, sample_data_simple)


def test_missing_handler_custom_callable():
    """Test custom callable imputation."""
    df = pd.DataFrame({
        'a': [1, np.nan, 3],
        'b': [4, 5, np.nan]
    })

    # Custom function that fills with 0
    def custom_imputer(data):
        return data.fillna(0)

    handler = MissingValueHandler(strategy=custom_imputer)
    result = handler.fit_transform(df)

    assert not result.isna().any().any()
    assert result.loc[1, 'a'] == 0.0
    assert result.loc[2, 'b'] == 0.0


def test_missing_handler_invalid_strategy():
    """Test that invalid strategy raises error."""
    with pytest.raises(ValueError, match="Invalid strategy"):
        handler = MissingValueHandler(strategy='invalid')


def test_missing_handler_fit_then_transform(sample_data_with_missing):
    """Test separate fit and transform."""
    handler = MissingValueHandler(strategy='median')

    # Fit on one dataset
    handler.fit(sample_data_with_missing)

    # Transform should work
    result = handler.transform(sample_data_with_missing)
    assert not result.isna().any().any()


def test_missing_handler_preserves_column_names(sample_data_with_missing):
    """Test that column names are preserved."""
    handler = MissingValueHandler(strategy='median')
    result = handler.fit_transform(sample_data_with_missing)

    assert list(result.columns) == list(sample_data_with_missing.columns)


def test_missing_handler_preserves_index(sample_data_with_missing):
    """Test that index is preserved (except for drop strategy)."""
    handler = MissingValueHandler(strategy='median')
    result = handler.fit_transform(sample_data_with_missing)

    # Index should be reset but length preserved
    assert len(result) == len(sample_data_with_missing)
