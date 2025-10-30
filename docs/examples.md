# Examples

Real-world examples and use cases for ClusterTK.

## Algorithm Comparison

Choose the best clustering algorithm automatically:

```python
import pandas as pd
from clustertk import ClusterAnalysisPipeline

# Load your data
df = pd.read_csv('data.csv')
features = ['feature1', 'feature2', 'feature3', 'feature4']

# Initialize pipeline with preprocessing settings
pipeline = ClusterAnalysisPipeline(
    handle_missing='median',
    handle_outliers='robust',
    scaling='robust',
    pca_variance=0.95,
    verbose=True
)

# Compare all available algorithms
results = pipeline.compare_algorithms(
    X=df,
    feature_columns=features,
    algorithms=['kmeans', 'gmm', 'hierarchical', 'dbscan'],
    n_clusters_range=(2, 10),
    metrics=['silhouette', 'calinski_harabasz', 'davies_bouldin']
)

# View comparison table
print("\nComparison Results:")
print(results['comparison'])
# Output:
#       algorithm  n_clusters  silhouette  calinski_harabasz  davies_bouldin
# 0        kmeans           4    0.650394        1076.898364        0.512246
# 1           gmm           4    0.650394        1076.898364        0.512246
# 2  hierarchical           4    0.650394        1076.898364        0.512246
# 3        dbscan           4    0.623707         735.818803        1.578299

# Get recommendation
print(f"\nRecommendation:")
print(f"  Best algorithm: {results['best_algorithm']}")
print(f"  Optimal n_clusters: {results['best_n_clusters']}")
print(f"  Silhouette score: {results['best_score']:.3f}")

# Visualize comparison
fig = pipeline.plot_algorithm_comparison(
    comparison_results=results,
    title='Algorithm Performance Comparison'
)
fig.savefig('algorithm_comparison.png')

# Create final pipeline with best settings
final_pipeline = ClusterAnalysisPipeline(
    handle_missing='median',
    handle_outliers='robust',
    scaling='robust',
    pca_variance=0.95,
    clustering_algorithm=results['best_algorithm'],
    n_clusters=results['best_n_clusters'],
    auto_name_clusters=True
)

# Fit and export
final_pipeline.fit(df, feature_columns=features)
final_pipeline.export_report('final_results.html')

print(f"\n✓ Analysis complete with {results['best_algorithm']}")
print(f"✓ Found {final_pipeline.n_clusters_} clusters")
print(f"✓ Results exported to final_results.html")
```

**When to use algorithm comparison:**
- You're unsure which algorithm suits your data
- You want to validate your algorithm choice
- You need to justify your algorithm selection
- You're exploring a new dataset


## Customer Segmentation

Segment customers based on behavior and demographics:

```python
import pandas as pd
from clustertk import ClusterAnalysisPipeline

# Load customer data
customers = pd.read_csv('customers.csv')

# Configure pipeline
pipeline = ClusterAnalysisPipeline(
    handle_missing='median',
    scaling='robust',
    pca_variance=0.95,
    clustering_algorithm='kmeans',
    n_clusters=None,  # Auto-detect
    auto_name_clusters=True
)

# Run analysis with category mapping
pipeline.fit(
    customers,
    feature_columns=['age', 'income', 'purchases', 'visits', 'avg_order_value'],
    category_mapping={
        'demographics': ['age', 'income'],
        'behavior': ['purchases', 'visits', 'avg_order_value']
    }
)

# Export results
pipeline.export_results('customer_segments.csv')
pipeline.export_report('customer_segments_report.html')

# Print summary
pipeline.print_cluster_summary()
```

## Anomaly Detection

Identify outliers in transaction data using DBSCAN or HDBSCAN:

```python
from clustertk import ClusterAnalysisPipeline

# Configure for anomaly detection with DBSCAN
pipeline = ClusterAnalysisPipeline(
    clustering_algorithm='dbscan',
    scaling='robust'
)

# Fit on transaction features
pipeline.fit(transactions, feature_columns=['amount', 'frequency', 'recency'])

# Check noise statistics (v0.12.0+)
print(f"Found {pipeline.n_clusters_} normal clusters")
print(f"Noise points: {pipeline.metrics_['n_noise']}")
print(f"Noise ratio: {pipeline.metrics_['noise_ratio']:.2%}")

# Extract anomalies (DBSCAN labels them as -1)
anomalies = transactions[pipeline.labels_ == -1]
normal = transactions[pipeline.labels_ != -1]

print(f"\nAnomalies detected: {len(anomalies)} / {len(transactions)}")
anomalies.to_csv('anomalies.csv', index=False)
```

**Alternative: Using HDBSCAN with custom parameters**

```python
# HDBSCAN with custom min_cluster_size for anomaly detection (v0.12.0+)
pipeline = ClusterAnalysisPipeline(
    clustering_algorithm='hdbscan',
    clustering_params={
        'min_cluster_size': 20,   # Smaller clusters are noise
        'min_samples': 5          # Lower threshold for dense areas
    },
    scaling='robust'
)

pipeline.fit(transactions, feature_columns=['amount', 'frequency', 'recency'])

# HDBSCAN provides membership probabilities
probs = pipeline.model_.probabilities_

# Identify anomalies with different confidence levels
high_confidence_anomalies = transactions[(pipeline.labels_ == -1) & (probs < 0.1)]
low_confidence_anomalies = transactions[(pipeline.labels_ == -1) & (probs >= 0.1)]

print(f"High confidence anomalies: {len(high_confidence_anomalies)}")
print(f"Low confidence anomalies: {len(low_confidence_anomalies)}")
print(f"Total noise: {pipeline.metrics_['n_noise']}")
```

## Market Basket Analysis

Cluster products based on purchase patterns:

```python
# Product purchase matrix (products × customers)
product_features = pd.read_csv('product_purchase_matrix.csv')

pipeline = ClusterAnalysisPipeline(
    clustering_algorithm='hierarchical',
    n_clusters=8,
    correlation_threshold=0.9  # Remove similar products
)

pipeline.fit(product_features)

# Visualize product clusters
pipeline.plot_clusters_2d()
pipeline.plot_cluster_heatmap()

# Export product segments
results = product_features.copy()
results['product_cluster'] = pipeline.labels_
results.to_csv('product_clusters.csv', index=False)
```

## Image Segmentation

Cluster image pixels by color:

```python
import numpy as np
from PIL import Image
from clustertk import ClusterAnalysisPipeline

# Load image
img = Image.open('photo.jpg')
pixels = np.array(img).reshape(-1, 3)  # RGB values

# Cluster colors
pipeline = ClusterAnalysisPipeline(
    clustering_algorithm='kmeans',
    n_clusters=8,
    scaling='minmax'  # RGB already in [0, 255]
)

# Create DataFrame
pixel_df = pd.DataFrame(pixels, columns=['R', 'G', 'B'])
pipeline.fit(pixel_df)

# Get dominant colors
dominant_colors = pipeline.cluster_profiles_[['R', 'G', 'B']].values
print(f"Dominant colors:\n{dominant_colors}")
```

## Time Series Clustering

Cluster time series patterns:

```python
# Extract features from time series
from scipy import stats

def extract_ts_features(series):
    return {
        'mean': series.mean(),
        'std': series.std(),
        'min': series.min(),
        'max': series.max(),
        'trend': stats.linregress(range(len(series)), series).slope,
        'skewness': stats.skew(series)
    }

# Extract features for each time series
ts_features = pd.DataFrame([
    extract_ts_features(ts) for ts in time_series_list
])

# Cluster time series
pipeline = ClusterAnalysisPipeline(
    clustering_algorithm='kmeans',
    n_clusters=5,
    scaling='standard'
)

pipeline.fit(ts_features)

# Group time series by cluster
for cluster_id in range(pipeline.n_clusters_):
    cluster_ts = [ts for ts, label in zip(time_series_list, pipeline.labels_) 
                  if label == cluster_id]
    print(f"Cluster {cluster_id}: {len(cluster_ts)} time series")
```

## Geographic Clustering

Cluster locations:

```python
# Load location data (latitude, longitude, + other features)
locations = pd.read_csv('locations.csv')

pipeline = ClusterAnalysisPipeline(
    clustering_algorithm='dbscan',  # Good for geographic data
    clustering_params={
        'eps': 0.01,        # ~1.1 km at equator
        'min_samples': 5    # At least 5 locations to form cluster
    },
    scaling='standard'
)

pipeline.fit(locations, feature_columns=['latitude', 'longitude', 'population'])

# Check results (v0.12.0+)
print(f"Found {pipeline.n_clusters_} geographic clusters")
print(f"Isolated locations (noise): {pipeline.metrics_['n_noise']}")
print(f"Noise ratio: {pipeline.metrics_['noise_ratio']:.2%}")

# Visualize on map (using folium or similar)
import folium

m = folium.Map(location=[locations['latitude'].mean(),
                         locations['longitude'].mean()],
               zoom_start=10)

colors = ['red', 'blue', 'green', 'purple', 'orange']
for idx, row in locations.iterrows():
    cluster = pipeline.labels_[idx]
    if cluster != -1:  # Skip noise points
        folium.CircleMarker(
            location=[row['latitude'], row['longitude']],
            radius=5,
            color=colors[cluster % len(colors)],
            fill=True,
            popup=f"Cluster {cluster}"
        ).add_to(m)
    else:
        # Mark noise points differently
        folium.CircleMarker(
            location=[row['latitude'], row['longitude']],
            radius=3,
            color='gray',
            fill=True,
            popup="Isolated location (noise)"
        ).add_to(m)

m.save('location_clusters.html')
print(f"\nMap saved to location_clusters.html")
```

## Text Document Clustering

Cluster documents based on TF-IDF features:

```python
from sklearn.feature_extraction.text import TfidfVectorizer

# Extract TF-IDF features
vectorizer = TfidfVectorizer(max_features=100)
tfidf_matrix = vectorizer.fit_transform(documents)
tfidf_df = pd.DataFrame(
    tfidf_matrix.toarray(),
    columns=vectorizer.get_feature_names_out()
)

# Cluster documents
pipeline = ClusterAnalysisPipeline(
    clustering_algorithm='kmeans',
    n_clusters=10,
    scaling=None,  # TF-IDF already normalized
    pca_variance=0.8  # Reduce dimensionality
)

pipeline.fit(tfidf_df)

# Get top terms per cluster
for cluster_id in range(pipeline.n_clusters_):
    profile = pipeline.cluster_profiles_.loc[cluster_id]
    top_terms = profile.nlargest(5)
    print(f"\nCluster {cluster_id} top terms:")
    print(top_terms)
```

## Feature Importance Analysis

Understand which features drive your clustering results:

```python
import pandas as pd
from clustertk import ClusterAnalysisPipeline

# Load customer data
customers = pd.read_csv('customers.csv')
features = ['age', 'income', 'purchases', 'visits', 'avg_order_value',
            'days_since_last_purchase', 'total_spent']

# Configure and fit pipeline
pipeline = ClusterAnalysisPipeline(
    handle_missing='median',
    scaling='robust',
    pca_variance=0.95,
    clustering_algorithm='kmeans',
    n_clusters=4,
    verbose=True
)

pipeline.fit(customers, feature_columns=features)

# Analyze feature importance using all methods
results = pipeline.analyze_feature_importance(
    method='all',
    n_repeats=10,
    random_state=42
)

# View permutation importance
print("\n=== Permutation Importance ===")
print("(How much each feature affects clustering quality)\n")
print(results['permutation'].head(10))
# Output:
#                     feature  importance       std
# 0              total_spent    0.234567  0.018234
# 1  days_since_last_purchase  0.198765  0.015432
# 2                 purchases    0.187654  0.012345
# 3            avg_order_value  0.156789  0.010987
# 4                    income    0.098765  0.008765

# View feature contribution (variance ratio)
print("\n=== Feature Contribution ===")
print("(Between-cluster variance / within-cluster variance)\n")
print(results['contribution'].head(10))
# Output:
#                     feature  contribution
# 0  days_since_last_purchase     15.234567
# 1              total_spent     12.456789
# 2                 purchases     10.123456
# 3            avg_order_value      8.765432
# 4                       age      5.432109

# SHAP analysis (if shap is installed)
if 'shap' in results:
    print("\n=== SHAP Importance ===")
    print("(Feature importance with interaction effects)\n")
    print(results['shap']['importance'].head(10))

# Identify most important features
# Combine permutation and contribution scores
perm = results['permutation'][['feature', 'importance']].rename(
    columns={'importance': 'perm_score'}
)
contrib = results['contribution'][['feature', 'contribution']].rename(
    columns={'contribution': 'contrib_score'}
)

combined = perm.merge(contrib, on='feature')
combined['combined_score'] = (
    combined['perm_score'] / combined['perm_score'].max() +
    combined['contrib_score'] / combined['contrib_score'].max()
)
combined = combined.sort_values('combined_score', ascending=False)

print("\n=== Combined Feature Ranking ===")
print(combined.head(10))

# Use top features for focused analysis
top_features = combined.head(5)['feature'].tolist()
print(f"\nTop 5 most important features:")
for i, feat in enumerate(top_features, 1):
    print(f"{i}. {feat}")

# Create new pipeline with only top features
print("\n=== Re-running analysis with top features only ===")
pipeline_focused = ClusterAnalysisPipeline(
    handle_missing='median',
    scaling='robust',
    clustering_algorithm='kmeans',
    n_clusters=4
)

pipeline_focused.fit(customers, feature_columns=top_features)

print(f"\nOriginal silhouette score (all features): {pipeline.metrics_['silhouette']:.3f}")
print(f"Focused silhouette score (top 5 features): {pipeline_focused.metrics_['silhouette']:.3f}")

# Export results
pipeline.export_report('customer_analysis_full.html')
pipeline_focused.export_report('customer_analysis_focused.html')

print("\n✓ Feature importance analysis complete!")
print(f"✓ Identified {len(top_features)} key features")
print("✓ Created focused analysis with top features")
```

**When to use feature importance analysis:**
- Understanding which features matter most
- Reducing dimensionality by selecting key features
- Explaining clustering results to stakeholders
- Validating domain knowledge about important variables
- Identifying redundant or irrelevant features

**Comparison of methods:**
```python
# Quick comparison of all three methods
from clustertk.interpretation import quick_feature_importance

top_features = quick_feature_importance(
    X=customers[features],
    labels=pipeline.labels_,
    method='all',
    n_top=10
)

print(top_features)
# Shows combined ranking from all methods
```

## Cluster Stability Analysis

Validate clustering reliability before making business decisions:

```python
import pandas as pd
import numpy as np
from clustertk import ClusterAnalysisPipeline

# Load data
customers = pd.read_csv('customers.csv')
features = ['age', 'income', 'purchases', 'avg_order_value']

# Create pipeline
pipeline = ClusterAnalysisPipeline(
    clustering_algorithm='kmeans',
    n_clusters=4,
    scaling='robust',
    verbose=True
)

# Fit pipeline
pipeline.fit(customers, feature_columns=features)

print(f"\nInitial clustering:")
print(f"  Silhouette score: {pipeline.metrics_['silhouette']:.3f}")
print(f"  Number of clusters: {pipeline.n_clusters_}")

# Analyze stability
print("\n" + "="*80)
print("STABILITY ANALYSIS")
print("="*80)

stability_results = pipeline.analyze_stability(
    n_iterations=100,
    sample_fraction=0.8,
    random_state=42
)

print(f"\nOverall stability: {stability_results['overall_stability']:.3f}")
print(f"Mean ARI: {stability_results['mean_ari']:.3f}")

# Per-cluster analysis
print("\nCluster stability breakdown:")
cluster_stability = stability_results['cluster_stability']
for _, row in cluster_stability.iterrows():
    cluster_id = int(row['cluster'])
    stability = row['stability']
    size = int(row['size'])

    status = "✓ STABLE" if stability > 0.7 else "⚠ UNSTABLE" if stability < 0.5 else "~ MODERATE"
    print(f"  Cluster {cluster_id} ({size} customers): {stability:.3f} {status}")

# Sample confidence analysis
confidence = stability_results['sample_confidence']
print(f"\nSample confidence:")
print(f"  Mean: {np.mean(confidence):.3f}")
print(f"  Median: {np.median(confidence):.3f}")

# Identify unstable samples
unstable_threshold = 0.5
unstable_mask = confidence < unstable_threshold
unstable_count = np.sum(unstable_mask)
unstable_pct = unstable_count / len(confidence) * 100

print(f"  Unstable samples: {unstable_count} ({unstable_pct:.1f}%)")

if unstable_count > 0:
    print("\n⚠ Warning: Some samples have low confidence scores")
    print("  Consider:")
    print("  - Reviewing features for these samples")
    print("  - Trying different clustering algorithms")
    print("  - Adjusting number of clusters")

# Decision making based on stability
print("\n" + "="*80)
print("RECOMMENDATIONS")
print("="*80)

overall_stability = stability_results['overall_stability']

if overall_stability > 0.8:
    print("✓ HIGH STABILITY - Results are reliable")
    print("  Safe to use for business decisions")
    print("  Clusters are well-defined and consistent")
elif overall_stability > 0.6:
    print("~ MODERATE STABILITY - Results are reasonable")
    print("  Consider validation with domain experts")
    print("  Monitor unstable clusters closely")
else:
    print("⚠ LOW STABILITY - Results may not be reliable")
    print("  Recommendations:")
    print("  1. Try different clustering algorithms")
    print("  2. Adjust number of clusters")
    print("  3. Improve feature engineering")
    print("  4. Collect more data or better features")

# Export results with confidence scores
customers_with_confidence = customers.copy()
customers_with_confidence['cluster'] = pipeline.labels_
customers_with_confidence['confidence'] = confidence
customers_with_confidence.to_csv('customers_with_stability.csv', index=False)

print(f"\n✓ Results exported to customers_with_stability.csv")
```

**When to use stability analysis:**
- Before presenting results to stakeholders
- When clusters will drive important decisions (pricing, marketing, etc.)
- With noisy or uncertain data
- To compare different clustering approaches
- To identify which customers/samples are reliably clustered

**Interpretation guide:**
- **Overall stability > 0.8**: Strong, trust the results
- **Overall stability 0.6-0.8**: Moderate, validate with experts
- **Overall stability < 0.6**: Weak, reconsider approach
- **Cluster stability > 0.7**: Well-defined cluster
- **Cluster stability < 0.5**: Poorly-defined, consider merging/removing
- **Sample confidence < 0.5**: Boundary points, uncertain assignment

## Comparing Multiple Algorithms

```python
from clustertk import ClusterAnalysisPipeline
import pandas as pd

# Test multiple algorithms
algorithms = ['kmeans', 'gmm', 'hierarchical']
results = []

for algo in algorithms:
    pipeline = ClusterAnalysisPipeline(
        clustering_algorithm=algo,
        n_clusters=5,
        random_state=42
    )
    pipeline.fit(df, feature_columns=features)

    results.append({
        'Algorithm': algo,
        'Silhouette': pipeline.metrics_['silhouette'],
        'Calinski-Harabasz': pipeline.metrics_['calinski_harabasz'],
        'Davies-Bouldin': pipeline.metrics_['davies_bouldin'],
        'N_Clusters': pipeline.n_clusters_
    })

# Compare results
comparison = pd.DataFrame(results)
print(comparison)
```

## More Examples

Check the `examples/` directory in the GitHub repository for Jupyter notebooks with complete examples:

- `examples/customer_segmentation.ipynb`
- `examples/anomaly_detection.ipynb`
- `examples/algorithm_comparison.ipynb`
- `examples/visualization_gallery.ipynb`

## Next Steps

- [User Guide](user_guide/README.md) - Detailed component documentation
- [API Reference](api_reference.md) - Complete API documentation
- [FAQ](faq.md) - Common questions
