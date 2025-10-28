# Export and Persistence

Save and share your clustering results in multiple formats.

## Export to CSV

Export cluster assignments with original data:

```python
# With original data
pipeline.export_results('results.csv', format='csv')

# Only cluster assignments
pipeline.export_results('results.csv', format='csv', include_original=False)
```

The CSV includes:
- Original data columns (if include_original=True)
- `cluster` column with labels
- `cluster_name` column (if naming was performed)

## Export to JSON

Export comprehensive metadata:

```python
# Full export
pipeline.export_results('results.json', format='json')

# Without profiles
pipeline.export_results('results.json', format='json', include_profiles=False)
```

JSON includes:
- Cluster labels and sizes
- Cluster profiles (mean feature values)
- Clustering metrics
- Pipeline configuration
- Selected features list

## Generate HTML Report

Create an interactive HTML report:

```python
# With embedded plots
pipeline.export_report('report.html')

# Without plots (faster, no viz dependencies)
pipeline.export_report('report.html', include_plots=False)
```

HTML report includes:
- Clustering summary and metrics
- Cluster sizes table
- Cluster profiles table
- Embedded visualizations (base64 encoded)
- Pipeline configuration

## Save and Load Pipeline

Persist fitted pipelines:

```python
# Save pipeline
pipeline.save_pipeline('my_pipeline.joblib')

# Load pipeline
from clustertk import ClusterAnalysisPipeline
loaded = ClusterAnalysisPipeline.load_pipeline('my_pipeline.joblib')

# Use loaded pipeline
labels = loaded.labels_
profiles = loaded.cluster_profiles_
```

Uses joblib for efficient serialization.

## Best Practices

1. Save pipelines for reproducibility
2. Use JSON for programmatic access to results
3. Use HTML reports for sharing with stakeholders
4. Include original data in CSV for further analysis
