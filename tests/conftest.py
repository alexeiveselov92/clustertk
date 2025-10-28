"""
Shared pytest fixtures for ClusterTK tests.

This module provides common fixtures used across all test modules.
"""

import pytest
import pandas as pd
import numpy as np
from sklearn.datasets import make_blobs, make_classification


@pytest.fixture
def sample_data_simple():
    """
    Generate simple synthetic data for basic tests.

    Returns
    -------
    df : pd.DataFrame
        DataFrame with 100 samples and 5 features.
    """
    np.random.seed(42)
    data = np.random.rand(100, 5)
    return pd.DataFrame(data, columns=[f'feature_{i}' for i in range(5)])


@pytest.fixture
def sample_data_with_clusters():
    """
    Generate synthetic data with known cluster structure.

    Returns
    -------
    df : pd.DataFrame
        DataFrame with 300 samples, 5 features, and 3 clusters.
    y_true : np.ndarray
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
    df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])])
    return df, y


@pytest.fixture
def sample_data_with_missing():
    """
    Generate data with missing values.

    Returns
    -------
    df : pd.DataFrame
        DataFrame with missing values (~10% missing).
    """
    np.random.seed(42)
    data = np.random.rand(100, 5)

    # Add missing values
    mask = np.random.rand(100, 5) < 0.1
    data = pd.DataFrame(data, columns=[f'feature_{i}' for i in range(5)])
    data[mask] = np.nan

    return data


@pytest.fixture
def sample_data_with_outliers():
    """
    Generate data with outliers.

    Returns
    -------
    df : pd.DataFrame
        DataFrame with outliers (~5% outliers).
    """
    np.random.seed(42)
    data = np.random.randn(100, 5)  # Normal distribution

    # Add outliers
    outlier_indices = np.random.choice(100, size=5, replace=False)
    data[outlier_indices] = np.random.randn(5, 5) * 10  # Extreme values

    return pd.DataFrame(data, columns=[f'feature_{i}' for i in range(5)])


@pytest.fixture
def sample_data_skewed():
    """
    Generate data with skewed distributions.

    Returns
    -------
    df : pd.DataFrame
        DataFrame with skewed features.
    """
    np.random.seed(42)
    # Exponential distribution (highly skewed)
    data = np.random.exponential(scale=2.0, size=(100, 5))
    return pd.DataFrame(data, columns=[f'feature_{i}' for i in range(5)])


@pytest.fixture
def sample_data_high_correlation():
    """
    Generate data with highly correlated features.

    Returns
    -------
    df : pd.DataFrame
        DataFrame with correlated features.
    """
    np.random.seed(42)
    base = np.random.rand(100, 2)

    # Create correlated features
    data = np.column_stack([
        base[:, 0],
        base[:, 0] + np.random.randn(100) * 0.1,  # Highly correlated
        base[:, 1],
        base[:, 1] + np.random.randn(100) * 0.1,  # Highly correlated
        np.random.rand(100)  # Independent
    ])

    return pd.DataFrame(data, columns=[f'feature_{i}' for i in range(5)])


@pytest.fixture
def sample_data_low_variance():
    """
    Generate data with low variance features.

    Returns
    -------
    df : pd.DataFrame
        DataFrame with some low-variance features.
    """
    np.random.seed(42)
    data = np.column_stack([
        np.random.rand(100),  # Normal variance
        np.random.rand(100),  # Normal variance
        np.ones(100) * 5,     # Zero variance
        np.random.randn(100) * 0.001 + 10,  # Very low variance
        np.random.rand(100)   # Normal variance
    ])

    return pd.DataFrame(data, columns=[f'feature_{i}' for i in range(5)])


@pytest.fixture
def sample_labels():
    """
    Generate sample cluster labels.

    Returns
    -------
    labels : np.ndarray
        Cluster labels for 300 samples with 3 clusters (matches sample_data_with_clusters).
    """
    np.random.seed(42)
    return np.random.randint(0, 3, size=300)


@pytest.fixture
def sample_labels_with_noise():
    """
    Generate sample cluster labels with noise points.

    Returns
    -------
    labels : np.ndarray
        Cluster labels with -1 for noise points (300 samples).
    """
    np.random.seed(42)
    labels = np.random.randint(0, 3, size=300)
    # Mark some as noise
    noise_indices = np.random.choice(300, size=30, replace=False)
    labels[noise_indices] = -1
    return labels
