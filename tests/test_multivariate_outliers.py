"""Tests for multivariate outlier detection."""

import pytest
import pandas as pd
import numpy as np
from clustertk.preprocessing import MultivariateOutlierDetector


@pytest.fixture
def normal_data_with_outliers():
    """Create dataset with multivariate outliers."""
    np.random.seed(42)

    # Normal cluster
    normal = np.random.multivariate_normal([0, 0, 0], [[1, 0, 0], [0, 1, 0], [0, 0, 1]], 100)

    # Multivariate outliers (far in combination, but not per-feature)
    outliers = np.array([
        [10, 10, 10],
        [-10, -10, -10],
        [10, -10, 10],
    ])

    all_data = np.vstack([normal, outliers])
    return pd.DataFrame(all_data, columns=['x', 'y', 'z'])


class TestMultivariateOutlierDetector:
    """Tests for MultivariateOutlierDetector class."""

    def test_init_default(self):
        """Test initialization with default parameters."""
        detector = MultivariateOutlierDetector()

        assert detector.method == 'auto'
        assert detector.contamination == 0.05
        assert detector.action == 'remove'
        assert detector.random_state is None

    def test_init_custom_params(self):
        """Test initialization with custom parameters."""
        detector = MultivariateOutlierDetector(
            method='isolation_forest',
            contamination=0.1,
            action='flag',
            random_state=42
        )

        assert detector.method == 'isolation_forest'
        assert detector.contamination == 0.1
        assert detector.action == 'flag'
        assert detector.random_state == 42

    def test_invalid_method(self, normal_data_with_outliers):
        """Test that invalid method raises error."""
        detector = MultivariateOutlierDetector(method='invalid_method')

        with pytest.raises(ValueError, match="Unknown method"):
            detector.fit(normal_data_with_outliers)

    def test_invalid_contamination(self, normal_data_with_outliers):
        """Test that invalid contamination raises error."""
        # contamination = 0.0
        detector = MultivariateOutlierDetector(contamination=0.0)
        with pytest.raises(ValueError, match="contamination must be in"):
            detector.fit(normal_data_with_outliers)

        # contamination > 0.5
        detector = MultivariateOutlierDetector(contamination=0.6)
        with pytest.raises(ValueError, match="contamination must be in"):
            detector.fit(normal_data_with_outliers)

    def test_invalid_input_type(self):
        """Test that non-DataFrame input raises error."""
        detector = MultivariateOutlierDetector()
        data_array = np.random.rand(100, 3)

        with pytest.raises(TypeError, match="X must be a pandas DataFrame"):
            detector.fit(data_array)

    def test_fit_transform_remove(self, normal_data_with_outliers):
        """Test fit_transform with action='remove'."""
        detector = MultivariateOutlierDetector(
            method='auto',
            contamination=0.05,
            action='remove',
            random_state=42
        )

        result = detector.fit_transform(normal_data_with_outliers)

        # Should remove outliers
        assert len(result) < len(normal_data_with_outliers)

        # Should detect approximately 5% outliers (103 * 0.05 ≈ 5)
        assert detector.n_outliers_ >= 3  # At least the 3 we added
        assert detector.outlier_ratio_ > 0.0

        # Result should be DataFrame
        assert isinstance(result, pd.DataFrame)

        # Columns should be preserved
        assert list(result.columns) == list(normal_data_with_outliers.columns)

    def test_fit_transform_flag(self, normal_data_with_outliers):
        """Test fit_transform with action='flag'."""
        detector = MultivariateOutlierDetector(
            method='auto',
            contamination=0.05,
            action='flag',
            random_state=42
        )

        result = detector.fit_transform(normal_data_with_outliers)

        # Should NOT remove rows
        assert len(result) == len(normal_data_with_outliers)

        # Should add _is_outlier column
        assert '_is_outlier' in result.columns

        # Number of True values should match n_outliers_
        assert result['_is_outlier'].sum() == detector.n_outliers_

    def test_fit_transform_none_action(self, normal_data_with_outliers):
        """Test fit_transform with action=None."""
        detector = MultivariateOutlierDetector(
            method='auto',
            contamination=0.05,
            action=None,
            random_state=42
        )

        result = detector.fit_transform(normal_data_with_outliers)

        # Should NOT modify data
        assert len(result) == len(normal_data_with_outliers)

        # Should NOT add column
        assert '_is_outlier' not in result.columns

        # But should still detect outliers
        assert detector.n_outliers_ > 0

    def test_isolation_forest_method(self, normal_data_with_outliers):
        """Test with IsolationForest method."""
        detector = MultivariateOutlierDetector(
            method='isolation_forest',
            contamination=0.05,
            random_state=42
        )

        detector.fit(normal_data_with_outliers)

        assert detector.method_used_ == 'isolation_forest'
        assert detector.n_outliers_ > 0

    def test_lof_method(self, normal_data_with_outliers):
        """Test with LOF method."""
        detector = MultivariateOutlierDetector(
            method='lof',
            contamination=0.05
        )

        detector.fit(normal_data_with_outliers)

        assert detector.method_used_ == 'lof'
        assert detector.n_outliers_ > 0

    def test_elliptic_envelope_method(self, normal_data_with_outliers):
        """Test with EllipticEnvelope method."""
        detector = MultivariateOutlierDetector(
            method='elliptic_envelope',
            contamination=0.05,
            random_state=42
        )

        detector.fit(normal_data_with_outliers)

        assert detector.method_used_ == 'elliptic_envelope'
        assert detector.n_outliers_ > 0

    def test_auto_method_selection(self, normal_data_with_outliers):
        """Test auto method selection."""
        detector = MultivariateOutlierDetector(
            method='auto',
            contamination=0.05,
            random_state=42
        )

        detector.fit(normal_data_with_outliers)

        # Should select one of the valid methods
        assert detector.method_used_ in ['isolation_forest', 'lof', 'elliptic_envelope']
        assert detector.n_outliers_ > 0

    def test_get_outlier_mask(self, normal_data_with_outliers):
        """Test get_outlier_mask method."""
        detector = MultivariateOutlierDetector(contamination=0.05, random_state=42)
        detector.fit(normal_data_with_outliers)

        mask = detector.get_outlier_mask()

        assert isinstance(mask, np.ndarray)
        assert mask.dtype == bool
        assert len(mask) == len(normal_data_with_outliers)
        assert mask.sum() == detector.n_outliers_

    def test_get_outlier_indices(self, normal_data_with_outliers):
        """Test get_outlier_indices method."""
        detector = MultivariateOutlierDetector(contamination=0.05, random_state=42)
        detector.fit(normal_data_with_outliers)

        indices = detector.get_outlier_indices()

        assert isinstance(indices, np.ndarray)
        assert len(indices) == detector.n_outliers_
        assert all(0 <= idx < len(normal_data_with_outliers) for idx in indices)

    def test_outlier_mask_before_fit(self):
        """Test that accessing mask before fit raises error."""
        detector = MultivariateOutlierDetector()

        with pytest.raises(ValueError, match="Must call fit"):
            detector.get_outlier_mask()

        with pytest.raises(ValueError, match="Must call fit"):
            detector.get_outlier_indices()

    def test_transform_without_fit(self, normal_data_with_outliers):
        """Test that transform without fit raises error."""
        detector = MultivariateOutlierDetector()

        with pytest.raises(ValueError, match="Must call fit"):
            detector.transform(normal_data_with_outliers)

    def test_transform_size_mismatch(self, normal_data_with_outliers):
        """Test that transform with different size data raises error."""
        detector = MultivariateOutlierDetector(random_state=42)
        detector.fit(normal_data_with_outliers)

        # Try to transform data with different size
        different_size_data = normal_data_with_outliers.iloc[:50]

        with pytest.raises(ValueError, match="X has .* rows"):
            detector.transform(different_size_data)

    def test_repr_before_fit(self):
        """Test string representation before fitting."""
        detector = MultivariateOutlierDetector(
            method='isolation_forest',
            contamination=0.1
        )

        repr_str = repr(detector)

        assert 'MultivariateOutlierDetector' in repr_str
        assert 'isolation_forest' in repr_str
        assert '0.1' in repr_str

    def test_repr_after_fit(self, normal_data_with_outliers):
        """Test string representation after fitting."""
        detector = MultivariateOutlierDetector(
            method='auto',
            contamination=0.05,
            random_state=42
        )
        detector.fit(normal_data_with_outliers)

        repr_str = repr(detector)

        assert 'MultivariateOutlierDetector' in repr_str
        assert detector.method_used_ in repr_str
        assert str(detector.n_outliers_) in repr_str

    def test_small_dataset(self):
        """Test with small dataset (should select LOF)."""
        # Create small dataset
        np.random.seed(42)
        small_data = pd.DataFrame(np.random.rand(50, 3), columns=['x', 'y', 'z'])

        detector = MultivariateOutlierDetector(method='auto', contamination=0.1)
        detector.fit(small_data)

        # Should select LOF for small datasets
        assert detector.method_used_ == 'lof'

    def test_high_dimensional_data(self):
        """Test with high-dimensional data (should select IsolationForest)."""
        # Create high-dimensional dataset
        np.random.seed(42)
        high_dim_data = pd.DataFrame(np.random.rand(200, 15), columns=[f'f{i}' for i in range(15)])

        detector = MultivariateOutlierDetector(method='auto', contamination=0.05)
        detector.fit(high_dim_data)

        # Should select IsolationForest for high dimensions
        assert detector.method_used_ == 'isolation_forest'

    def test_contamination_effect(self, normal_data_with_outliers):
        """Test that higher contamination detects more outliers."""
        detector_low = MultivariateOutlierDetector(
            method='isolation_forest',
            contamination=0.01,
            random_state=42
        )
        detector_high = MultivariateOutlierDetector(
            method='isolation_forest',
            contamination=0.10,
            random_state=42
        )

        detector_low.fit(normal_data_with_outliers)
        detector_high.fit(normal_data_with_outliers)

        # Higher contamination should detect more outliers
        assert detector_high.n_outliers_ >= detector_low.n_outliers_

    def test_reproducibility(self, normal_data_with_outliers):
        """Test that results are reproducible with same random_state."""
        detector1 = MultivariateOutlierDetector(
            method='isolation_forest',
            contamination=0.05,
            random_state=42
        )
        detector2 = MultivariateOutlierDetector(
            method='isolation_forest',
            contamination=0.05,
            random_state=42
        )

        detector1.fit(normal_data_with_outliers)
        detector2.fit(normal_data_with_outliers)

        # Should get same outliers
        assert detector1.n_outliers_ == detector2.n_outliers_
        np.testing.assert_array_equal(
            detector1.get_outlier_mask(),
            detector2.get_outlier_mask()
        )
