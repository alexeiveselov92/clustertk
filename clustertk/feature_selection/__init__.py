"""
Feature selection module for reducing feature dimensionality.

This module provides classes for:
- Correlation-based feature filtering
- Smart correlation filtering with clusterability evaluation
- Variance-based feature filtering
"""

from clustertk.feature_selection.correlation import (
    CorrelationFilter,
    find_highly_correlated_features
)
from clustertk.feature_selection.smart_correlation import (
    SmartCorrelationFilter
)
from clustertk.feature_selection.variance import (
    VarianceFilter,
    find_low_variance_features
)

__all__ = [
    'CorrelationFilter',
    'SmartCorrelationFilter',
    'VarianceFilter',
    'find_highly_correlated_features',
    'find_low_variance_features',
]
