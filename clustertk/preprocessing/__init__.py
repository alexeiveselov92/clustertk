"""
Preprocessing module for data cleaning and transformation.

This module provides classes for handling:
- Missing values
- Outliers detection and handling (univariate and multivariate)
- Scaling and normalization
- Distribution transformations (log, box-cox)
"""

from clustertk.preprocessing.missing import MissingValueHandler, detect_missing_patterns
from clustertk.preprocessing.outliers import OutlierHandler, detect_outliers_percentage
from clustertk.preprocessing.multivariate_outliers import MultivariateOutlierDetector
from clustertk.preprocessing.scaling import ScalerSelector, compare_scalers
from clustertk.preprocessing.transforms import SkewnessTransformer, detect_skewness

__all__ = [
    'MissingValueHandler',
    'OutlierHandler',
    'MultivariateOutlierDetector',
    'ScalerSelector',
    'SkewnessTransformer',
    'detect_missing_patterns',
    'detect_outliers_percentage',
    'compare_scalers',
    'detect_skewness',
]
