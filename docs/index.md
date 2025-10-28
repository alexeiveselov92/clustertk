# ClusterTK Documentation

Welcome to ClusterTK - a comprehensive Python toolkit for cluster analysis!

## What is ClusterTK?

ClusterTK provides a complete, sklearn-style pipeline for clustering analysis. From raw data preprocessing to cluster interpretation and export, ClusterTK makes clustering accessible and powerful for data analysts.

## Quick Links

- **[Installation](installation.md)** - Get ClusterTK installed
- **[Quick Start](quickstart.md)** - Start clustering in 5 minutes
- **[User Guide](user_guide/README.md)** - Detailed documentation
- **[Examples](examples.md)** - Real-world use cases
- **[FAQ](faq.md)** - Common questions

## Key Features

- 🔄 Complete pipeline from raw data to insights
- 📊 Multiple algorithms (K-Means, GMM, Hierarchical, DBSCAN)
- 🎯 Automatic optimal cluster number selection
- 🎨 Rich visualization capabilities
- 📁 Export to CSV, JSON, HTML reports
- 💾 Save and load fitted pipelines

## Quick Example

```python
from clustertk import ClusterAnalysisPipeline
import pandas as df

# Load data
df = pd.read_csv('data.csv')

# Create pipeline
pipeline = ClusterAnalysisPipeline(
    n_clusters=None,  # Auto-detect
    verbose=True
)

# Fit and get results
pipeline.fit(df, feature_columns=['feature1', 'feature2'])
labels = pipeline.labels_
```

## Documentation Structure

### Getting Started
1. [Installation](installation.md) - Install ClusterTK
2. [Quick Start](quickstart.md) - Your first clustering analysis

### User Guide
Detailed documentation for each component:
- [Preprocessing](user_guide/preprocessing.md) - Data cleaning
- [Feature Selection](user_guide/feature_selection.md) - Select relevant features
- [Dimensionality Reduction](user_guide/dimensionality.md) - PCA and manifold methods
- [Clustering](user_guide/clustering.md) - Clustering algorithms
- [Evaluation](user_guide/evaluation.md) - Assess quality
- [Interpretation](user_guide/interpretation.md) - Understand clusters
- [Visualization](user_guide/visualization.md) - Plot results
- [Export](user_guide/export.md) - Save and share

### Resources
- [Examples](examples.md) - Real-world applications
- [FAQ](faq.md) - Frequently asked questions

## Support

- **Issues**: [GitHub Issues](https://github.com/alexeiveselov92/clustertk/issues)
- **Discussions**: [GitHub Discussions](https://github.com/alexeiveselov92/clustertk/discussions)
- **Email**: alexei.veselov92@gmail.com

## Contributing

Contributions are welcome! Check out our GitHub repository.

## License

MIT License
