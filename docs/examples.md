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

Identify outliers in transaction data:

```python
from clustertk import ClusterAnalysisPipeline

# Configure for anomaly detection
pipeline = ClusterAnalysisPipeline(
    clustering_algorithm='dbscan',
    scaling='robust'
)

# Fit on transaction features
pipeline.fit(transactions, feature_columns=['amount', 'frequency', 'recency'])

# Extract anomalies (DBSCAN labels them as -1)
anomalies = transactions[pipeline.labels_ == -1]
normal = transactions[pipeline.labels_ != -1]

print(f"Anomalies detected: {len(anomalies)} / {len(transactions)}")
anomalies.to_csv('anomalies.csv', index=False)
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
    scaling='standard'
)

pipeline.fit(locations, feature_columns=['latitude', 'longitude', 'population'])

# Visualize on map (using folium or similar)
import folium

m = folium.Map(location=[locations['latitude'].mean(), 
                         locations['longitude'].mean()], 
               zoom_start=10)

colors = ['red', 'blue', 'green', 'purple', 'orange']
for idx, row in locations.iterrows():
    cluster = pipeline.labels_[idx]
    if cluster != -1:  # Skip noise
        folium.CircleMarker(
            location=[row['latitude'], row['longitude']],
            radius=5,
            color=colors[cluster % len(colors)],
            fill=True
        ).add_to(m)

m.save('location_clusters.html')
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
