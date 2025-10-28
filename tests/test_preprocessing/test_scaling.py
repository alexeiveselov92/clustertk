"""
Tests for ScalerSelector.
"""

import pytest
import pandas as pd
import numpy as np
from clustertk.preprocessing import ScalerSelector


def test_scaler_standard(sample_data_simple):
    """Test StandardScaler selection."""
    scaler = ScalerSelector(scaler_type='standard')
    result = scaler.fit_transform(sample_data_simple)

    # Check mean ≈ 0, std ≈ 1 (per column)
    assert np.allclose(result.mean(axis=0), 0, atol=1e-10)
    # pandas std uses ddof=1 by default, so allow small deviation
    assert np.allclose(result.std(axis=0, ddof=0), 1, atol=0.01)


def test_scaler_robust(sample_data_with_outliers):
    """Test RobustScaler selection."""
    scaler = ScalerSelector(scaler_type='robust')
    result = scaler.fit_transform(sample_data_with_outliers)

    # Robust scaler should handle outliers better
    # Median should be close to 0
    assert np.abs(result.median().median()) < 1


def test_scaler_minmax(sample_data_simple):
    """Test MinMaxScaler selection."""
    scaler = ScalerSelector(scaler_type='minmax')
    result = scaler.fit_transform(sample_data_simple)

    # Check values in [0, 1]
    assert result.min().min() >= 0
    assert result.max().max() <= 1


def test_scaler_auto_no_outliers(sample_data_simple):
    """Test auto scaler selection with clean data."""
    scaler = ScalerSelector(scaler_type='auto')
    result = scaler.fit_transform(sample_data_simple)

    # Should select StandardScaler
    assert scaler.selected_scaler_type_ == 'standard'


def test_scaler_auto_with_outliers(sample_data_with_outliers):
    """Test auto scaler selection with outliers."""
    scaler = ScalerSelector(scaler_type='auto')
    result = scaler.fit_transform(sample_data_with_outliers)

    # Should select RobustScaler
    assert scaler.selected_scaler_type_ == 'robust'


def test_scaler_invalid_type():
    """Test invalid scaler type."""
    with pytest.raises(ValueError, match="Invalid scaler"):
        ScalerSelector(scaler_type='invalid')


def test_scaler_fit_then_transform(sample_data_simple):
    """Test separate fit and transform."""
    scaler = ScalerSelector(scaler_type='standard')

    scaler.fit(sample_data_simple)
    result = scaler.transform(sample_data_simple)

    assert np.allclose(result.mean(), 0, atol=1e-10)


def test_scaler_preserves_shape(sample_data_simple):
    """Test that scaler preserves data shape."""
    scaler = ScalerSelector(scaler_type='standard')
    result = scaler.fit_transform(sample_data_simple)

    assert result.shape == sample_data_simple.shape


def test_scaler_preserves_columns(sample_data_simple):
    """Test that column names are preserved."""
    scaler = ScalerSelector(scaler_type='standard')
    result = scaler.fit_transform(sample_data_simple)

    assert list(result.columns) == list(sample_data_simple.columns)
