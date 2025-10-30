# Preprocessing

Data preprocessing is the first and most important step in cluster analysis. ClusterTK provides comprehensive preprocessing capabilities to handle common data quality issues.

## Overview

The preprocessing step handles:
1. Missing values
2. Outliers
3. Scaling/normalization
4. Skewness transformation

## Missing Value Handling

### Built-in Strategies

```python
from clustertk import ClusterAnalysisPipeline

# Median imputation (default, robust to outliers)
pipeline = ClusterAnalysisPipeline(handle_missing='median')

# Mean imputation
pipeline = ClusterAnalysisPipeline(handle_missing='mean')

# Drop rows with missing values
pipeline = ClusterAnalysisPipeline(handle_missing='drop')
```

### Custom Imputation

```python
def custom_imputer(df):
    """Custom imputation logic"""
    # Fill numeric columns with median
    numeric_cols = df.select_dtypes(include=['number']).columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
    return df

pipeline = ClusterAnalysisPipeline(handle_missing=custom_imputer)
```

## Outlier Handling

**IMPORTANT (v0.13.0+):** The default outlier handling strategy has changed from `'robust'` to `'winsorize'` for better clustering results.

### Why Outlier Handling Matters

**Common Problem:** K-Means creates 1 huge cluster (90%+) + several tiny clusters (1-2% each).

**Root Cause:** Even after scaling, outliers remain far away from normal points:
```python
# Example: revenue = [100, 150, 200, 10000]
# After RobustScaler: [0.0, 0.5, 1.0, 98.0]
# Outlier is still 98x away from normal points!
# K-Means sees: "cluster of normal points" + "isolated outlier"
```

**Solution:** Remove or clip outliers **before** scaling.

### Strategies

**Winsorize** (default, RECOMMENDED): Clips outliers to percentiles before scaling
```python
pipeline = ClusterAnalysisPipeline(handle_outliers='winsorize')  # Default since v0.13.0

# How it works:
# 1. Clip values to 2.5%-97.5% percentile range (removes 5% total)
# 2. Then apply scaling
# 3. Clustering works on properly bounded data
#
# Best for: Extreme outliers (10-50x), asymmetric distributions
```

**Robust Scaling** (old default): Uses RobustScaler only
```python
pipeline = ClusterAnalysisPipeline(handle_outliers='robust')

# WARNING: This does NOT remove outliers!
# Outliers remain far away after scaling → tiny clusters
# Use only when you WANT to detect outliers as separate clusters
```

**Clipping**: Caps outliers at IQR boundaries before scaling
```python
pipeline = ClusterAnalysisPipeline(handle_outliers='clip')

# How it works:
# Lower bound = Q1 - 1.5 × IQR
# Upper bound = Q3 + 1.5 × IQR
#
# Problem: Multiple extreme outliers clip to SAME boundary value
# Example: [10000, 12000, 15000] → all become 875
```

**Removal**: Removes rows with outliers (data loss)
```python
pipeline = ClusterAnalysisPipeline(handle_outliers='remove')

# Good when: <5% outliers, you want to discard them completely
# Bad when: >10% outliers, too much data loss
```

**None**: No outlier handling
```python
pipeline = ClusterAnalysisPipeline(handle_outliers=None)

# Only use when: Data is already clean or outliers are meaningful
```

### Execution Order

ClusterTK applies outlier handling in the correct order:
```
1. Winsorize/Clip/Remove outliers → bounds extreme values
2. Apply StandardScaler/RobustScaler → normalizes scale
3. K-Means clustering → works on clean, bounded data
```

**Why This Matters:**
- StandardScaler and RobustScaler don't remove outliers
- They only normalize the scale
- Outliers remain far away after scaling
- K-Means creates tiny clusters for these distant points

### Comparison

| Method | Outliers Removed? | Data Loss? | Best For |
|--------|------------------|-----------|----------|
| **winsorize** | ✅ Clipped to percentiles | ❌ No | **RECOMMENDED** for most cases |
| robust | ❌ Remains far away | ❌ No | When outliers = separate clusters |
| clip | ⚠️ Clipped to IQR | ❌ No | Mild outliers |
| remove | ✅ Rows deleted | ✅ Yes | <5% outliers, can discard |

### Example: Impact on Clustering

```python
import pandas as pd
import numpy as np
from clustertk import ClusterAnalysisPipeline

# Data with 3 clear clusters + 3 extreme outliers
data = pd.DataFrame({
    'revenue': [100, 150, 200] * 30 + [10000, 12000, 15000],  # 3 outliers
    'age': [25, 35, 45] * 30 + [200, 250, 300]
})

# ❌ BAD: Using 'robust' (outliers remain)
pipeline_robust = ClusterAnalysisPipeline(handle_outliers='robust', n_clusters=3)
pipeline_robust.fit(data)
# Result: [88, 1, 4] - One outlier gets own cluster, others grouped

# ✅ GOOD: Using 'winsorize' (default)
pipeline_wins = ClusterAnalysisPipeline(handle_outliers='winsorize', n_clusters=3)
pipeline_wins.fit(data)
# Result: [30, 30, 33] - Balanced clusters, outliers clipped
```

## Scaling

### Scaling Methods

**RobustScaler** (default): Uses median and IQR, robust to outliers
```python
pipeline = ClusterAnalysisPipeline(scaling='robust')
```

**StandardScaler**: Mean=0, StdDev=1
```python
pipeline = ClusterAnalysisPipeline(scaling='standard')
```

**MinMaxScaler**: Scales to [0, 1] range
```python
pipeline = ClusterAnalysisPipeline(scaling='minmax')
```

**Auto**: Selects based on data distribution
```python
pipeline = ClusterAnalysisPipeline(scaling='auto')
```

### Why Scaling Matters

Clustering algorithms (especially K-Means) are sensitive to feature scales:

```python
# Without scaling: features with larger values dominate
# Age: [20, 30, 40]     → distances ~10-20
# Income: [30k, 60k, 90k] → distances ~30,000
# Result: clustering driven by income only!

# With scaling: all features contribute equally
# Age: [-1, 0, 1]
# Income: [-1, 0, 1]
# Result: balanced clustering
```

## Skewness Transformation

Handle heavily skewed features with log transformation:

```python
pipeline = ClusterAnalysisPipeline(
    log_transform_skewed=True,
    skewness_threshold=2.0  # Features with |skewness| > 2.0 transformed
)
```

### How It Works

1. Calculate skewness for each feature
2. If `|skewness| > threshold`, apply `log1p(x)` transformation
3. Re-impute any infinity values created

### When to Use

Use skewness transformation when:
- Features have long tails (e.g., income, transaction amounts)
- Distribution is highly asymmetric
- Skewness > 2 or < -2

## Complete Preprocessing Example

```python
from clustertk import ClusterAnalysisPipeline
import pandas as pd

# Load data
df = pd.read_csv('customer_data.csv')

# Configure comprehensive preprocessing
pipeline = ClusterAnalysisPipeline(
    # Missing values
    handle_missing='median',

    # Outliers
    handle_outliers='robust',

    # Scaling
    scaling='robust',

    # Skewness
    log_transform_skewed=True,
    skewness_threshold=2.0,

    verbose=True  # See preprocessing steps
)

# Run preprocessing
pipeline.preprocess(df, feature_columns=['age', 'income', 'purchases'])

# Access preprocessed data
preprocessed = pipeline.data_preprocessed_
scaled = pipeline.data_scaled_
```

## Step-by-Step Preprocessing

For more control, use individual preprocessing modules:

```python
from clustertk.preprocessing import (
    MissingValueHandler,
    OutlierHandler,
    ScalerSelector,
    SkewnessTransformer
)

# Step 1: Handle missing values
missing_handler = MissingValueHandler(strategy='median')
data = missing_handler.fit_transform(df)

# Step 2: Transform skewed features
transformer = SkewnessTransformer(threshold=2.0)
data = transformer.fit_transform(data)

# Step 3: Handle outliers
outlier_handler = OutlierHandler(method='iqr', action='clip')
data = outlier_handler.fit_transform(data)

# Step 4: Scale features
scaler = ScalerSelector(scaler_type='robust')
data_scaled = scaler.fit_transform(data)
```

## Best Practices

### 1. Always Scale Your Data

```python
# ✅ Good
pipeline = ClusterAnalysisPipeline(scaling='robust')

# ❌ Bad (will produce poor results)
pipeline = ClusterAnalysisPipeline(scaling=None)
```

### 2. Choose Robust Methods for Real-World Data

```python
# ✅ Good for real-world data (often has outliers)
pipeline = ClusterAnalysisPipeline(
    handle_missing='median',  # Robust to outliers
    handle_outliers='robust',  # RobustScaler
    scaling='robust'
)

# ⚠️  Sensitive to outliers
pipeline = ClusterAnalysisPipeline(
    handle_missing='mean',     # Affected by outliers
    handle_outliers=None,      # Outliers not handled
    scaling='standard'         # Uses mean and std
)
```

### 3. Check Data Quality First

```python
# Before clustering, check your data
print(df.isnull().sum())  # Missing values per column
print(df.describe())       # Distribution statistics
print(df.dtypes)           # Data types
```

### 4. Use Verbose Mode for Debugging

```python
pipeline = ClusterAnalysisPipeline(verbose=True)
pipeline.preprocess(df, feature_columns=features)

# Output shows:
# - Missing values detected and handled
# - Outliers detected and handled
# - Scaling method used
# - Shape changes at each step
```

## Common Issues

### Issue: Too many missing values

```python
# Check missing percentages
missing_pct = df.isnull().sum() / len(df) * 100
print(missing_pct[missing_pct > 20])  # Columns with >20% missing

# Consider dropping columns with excessive missing values
features = [col for col in features if missing_pct[col] < 30]
```

### Issue: Infinity values after transformation

ClusterTK automatically handles this by:
1. Detecting infinity values after transformation
2. Replacing with NaN
3. Re-imputing using the selected strategy

### Issue: All values the same after scaling

This happens with zero-variance features. Use variance filtering:

```python
pipeline = ClusterAnalysisPipeline(
    variance_threshold=0.01  # Remove features with variance < 0.01
)
```

## Next Steps

- [Feature Selection](feature_selection.md) - Select relevant features
- [Dimensionality Reduction](dimensionality.md) - Reduce feature space
- [Clustering](clustering.md) - Apply clustering algorithms
