"""
Integration tests for ClusterAnalysisPipeline.
"""

import pytest
import pandas as pd
import numpy as np
from clustertk.pipeline import ClusterAnalysisPipeline


def test_pipeline_basic_workflow(sample_data_with_clusters):
    """Test complete pipeline workflow."""
    df, y_true = sample_data_with_clusters

    pipeline = ClusterAnalysisPipeline(
        clustering_algorithm='kmeans',
        n_clusters=3,
        verbose=False
    )

    pipeline.fit(df, feature_columns=df.columns.tolist())

    # Check all attributes are set
    assert pipeline.labels_ is not None
    assert pipeline.n_clusters_ == 3
    assert pipeline.metrics_ is not None
    assert pipeline.cluster_profiles_ is not None


def test_pipeline_all_algorithms(sample_data_with_clusters):
    """Test pipeline with all clustering algorithms."""
    df, y_true = sample_data_with_clusters

    algorithms = ['kmeans', 'gmm', 'hierarchical', 'dbscan', 'hdbscan']

    for algo in algorithms:
        pipeline = ClusterAnalysisPipeline(
            clustering_algorithm=algo,
            n_clusters=3 if algo not in ['dbscan', 'hdbscan'] else None,
            verbose=False
        )

        pipeline.fit(df, feature_columns=df.columns.tolist())

        assert pipeline.labels_ is not None
        assert pipeline.n_clusters_ > 0
        assert pipeline.metrics_ is not None


def test_pipeline_with_missing_values(sample_data_with_missing):
    """Test pipeline handles missing values."""
    pipeline = ClusterAnalysisPipeline(
        handle_missing='median',
        clustering_algorithm='kmeans',
        n_clusters=3,
        verbose=False
    )

    pipeline.fit(sample_data_with_missing, feature_columns=sample_data_with_missing.columns.tolist())

    # Should complete without errors
    assert pipeline.labels_ is not None
    assert not pd.isna(pipeline.labels_).any()


def test_pipeline_with_outliers(sample_data_with_outliers):
    """Test pipeline handles outliers."""
    pipeline = ClusterAnalysisPipeline(
        handle_outliers='robust',
        clustering_algorithm='kmeans',
        n_clusters=3,
        verbose=False
    )

    pipeline.fit(sample_data_with_outliers, feature_columns=sample_data_with_outliers.columns.tolist())

    assert pipeline.labels_ is not None


def test_pipeline_auto_k(sample_data_with_clusters):
    """Test pipeline with automatic k selection."""
    df, y_true = sample_data_with_clusters

    pipeline = ClusterAnalysisPipeline(
        clustering_algorithm='kmeans',
        n_clusters=None,  # Auto-detect
        n_clusters_range=(2, 5),
        verbose=False
    )

    pipeline.fit(df, feature_columns=df.columns.tolist())

    # Should select some k in range
    assert 2 <= pipeline.n_clusters_ <= 5


def test_pipeline_compare_algorithms(sample_data_with_clusters):
    """Test compare_algorithms method."""
    df, y_true = sample_data_with_clusters

    pipeline = ClusterAnalysisPipeline(verbose=False)

    results = pipeline.compare_algorithms(
        X=df,
        feature_columns=df.columns.tolist(),
        algorithms=['kmeans', 'gmm'],
        n_clusters_range=(2, 4)
    )

    # Check results structure
    assert 'comparison' in results
    assert 'best_algorithm' in results
    assert 'best_n_clusters' in results
    assert 'best_score' in results

    # Check comparison DataFrame
    assert len(results['comparison']) == 2  # 2 algorithms
    assert 'algorithm' in results['comparison'].columns
    assert 'silhouette' in results['comparison'].columns


def test_pipeline_save_load(sample_data_with_clusters, tmp_path):
    """Test pipeline save and load."""
    df, y_true = sample_data_with_clusters

    # Fit pipeline
    pipeline = ClusterAnalysisPipeline(
        clustering_algorithm='kmeans',
        n_clusters=3,
        verbose=False
    )
    pipeline.fit(df, feature_columns=df.columns.tolist())

    # Save
    save_path = tmp_path / "pipeline.pkl"
    pipeline.save_pipeline(str(save_path))

    # Load
    loaded_pipeline = ClusterAnalysisPipeline.load_pipeline(str(save_path))

    # Check loaded pipeline
    assert loaded_pipeline.n_clusters_ == pipeline.n_clusters_
    np.testing.assert_array_equal(loaded_pipeline.labels_, pipeline.labels_)


def test_pipeline_export_results(sample_data_with_clusters, tmp_path):
    """Test export_results method."""
    df, y_true = sample_data_with_clusters

    pipeline = ClusterAnalysisPipeline(
        clustering_algorithm='kmeans',
        n_clusters=3,
        verbose=False
    )
    pipeline.fit(df, feature_columns=df.columns.tolist())

    # Export CSV
    csv_path = tmp_path / "results.csv"
    pipeline.export_results(str(csv_path), format='csv')
    assert csv_path.exists()

    # Export JSON
    json_path = tmp_path / "results.json"
    pipeline.export_results(str(json_path), format='json')
    assert json_path.exists()


def test_pipeline_stepwise_execution(sample_data_with_clusters):
    """Test stepwise pipeline execution."""
    df, y_true = sample_data_with_clusters

    pipeline = ClusterAnalysisPipeline(verbose=False)

    # Step by step
    pipeline.preprocess(df, feature_columns=df.columns.tolist())
    assert pipeline.data_preprocessed_ is not None

    pipeline.select_features()
    assert pipeline.selected_features_ is not None

    pipeline.reduce_dimensions()
    assert pipeline.data_reduced_ is not None

    pipeline.find_optimal_clusters()
    assert pipeline.n_clusters_ is not None

    pipeline.cluster()
    assert pipeline.labels_ is not None

    pipeline.create_profiles()
    assert pipeline.cluster_profiles_ is not None


def test_pipeline_with_pca(sample_data_with_clusters):
    """Test pipeline with PCA dimensionality reduction."""
    df, y_true = sample_data_with_clusters

    pipeline = ClusterAnalysisPipeline(
        pca_variance=0.95,
        clustering_algorithm='kmeans',
        n_clusters=3,
        verbose=False
    )

    pipeline.fit(df, feature_columns=df.columns.tolist())

    # Check PCA was applied
    assert pipeline.data_reduced_ is not None
    # PCA should reduce dimensions
    assert pipeline.data_reduced_.shape[1] <= df.shape[1]
