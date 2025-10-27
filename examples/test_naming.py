"""
Test cluster naming functionality.

This script demonstrates automatic cluster naming in ClusterTK.
"""

import pandas as pd
import numpy as np
from sklearn.datasets import make_blobs
from clustertk import ClusterAnalysisPipeline


def test_basic_naming():
    """Test basic cluster naming without categories."""
    print("=" * 80)
    print("Test 1: Basic Cluster Naming (Feature-based)")
    print("=" * 80)

    # Generate sample data with distinct clusters
    X, true_labels = make_blobs(
        n_samples=300,
        n_features=6,
        centers=3,
        cluster_std=1.0,
        random_state=42
    )

    # Convert to DataFrame with meaningful names
    df = pd.DataFrame(
        X,
        columns=['age', 'income', 'spending', 'visits', 'loyalty_score', 'satisfaction']
    )

    # Create pipeline with auto naming enabled
    pipeline = ClusterAnalysisPipeline(
        clustering_algorithm='kmeans',
        n_clusters=3,
        auto_name_clusters=True,
        naming_max_features=2,
        verbose=True
    )

    # Fit pipeline
    pipeline.fit(df)

    # Print cluster summary
    print("\n" + "=" * 80)
    print("CLUSTER NAMES GENERATED")
    print("=" * 80)
    for cluster_id in sorted(pipeline.cluster_names_.keys()):
        name = pipeline.cluster_names_[cluster_id]
        print(f"Cluster {cluster_id}: {name}")

    # Test individual methods
    print("\n" + "=" * 80)
    print("Testing get_cluster_name()")
    print("=" * 80)
    for i in range(3):
        name = pipeline.get_cluster_name(i)
        print(f"get_cluster_name({i}) = {name}")

    # Print full summary
    pipeline.print_cluster_summary()

    print("\n✓ Test 1 completed successfully!\n")


def test_category_naming():
    """Test cluster naming with category mapping."""
    print("=" * 80)
    print("Test 2: Cluster Naming with Categories")
    print("=" * 80)

    # Generate sample data
    X, _ = make_blobs(
        n_samples=300,
        n_features=9,
        centers=4,
        cluster_std=1.2,
        random_state=42
    )

    # Create DataFrame with feature categories
    df = pd.DataFrame(
        X,
        columns=[
            'purchase_freq', 'avg_basket', 'total_spent',  # Behavioral
            'age', 'income', 'family_size',                # Demographic
            'app_usage', 'email_opens', 'social_shares'   # Engagement
        ]
    )

    # Define category mapping
    category_mapping = {
        'behavioral': ['purchase_freq', 'avg_basket', 'total_spent'],
        'demographic': ['age', 'income', 'family_size'],
        'engagement': ['app_usage', 'email_opens', 'social_shares']
    }

    # Create pipeline (no auto naming, we'll call manually)
    pipeline = ClusterAnalysisPipeline(
        clustering_algorithm='kmeans',
        n_clusters=4,
        auto_name_clusters=False,
        verbose=True
    )

    # Fit pipeline
    pipeline.fit(df, category_mapping=category_mapping)

    # Test different naming strategies
    print("\n" + "=" * 80)
    print("Testing Different Naming Strategies")
    print("=" * 80)

    # Strategy 1: Auto (will use combined since categories available)
    print("\n1. Strategy: AUTO")
    names_auto = pipeline.name_clusters(
        category_mapping=category_mapping,
        naming_strategy='auto',
        max_features=2
    )
    for cluster_id, name in sorted(names_auto.items()):
        print(f"  Cluster {cluster_id}: {name}")

    # Strategy 2: Top features only
    print("\n2. Strategy: TOP_FEATURES")
    names_features = pipeline.name_clusters(
        naming_strategy='top_features',
        max_features=2
    )
    for cluster_id, name in sorted(names_features.items()):
        print(f"  Cluster {cluster_id}: {name}")

    # Strategy 3: Categories only
    print("\n3. Strategy: CATEGORIES")
    names_categories = pipeline.name_clusters(
        category_mapping=category_mapping,
        naming_strategy='categories',
        max_features=2
    )
    for cluster_id, name in sorted(names_categories.items()):
        print(f"  Cluster {cluster_id}: {name}")

    # Strategy 4: Combined
    print("\n4. Strategy: COMBINED")
    names_combined = pipeline.name_clusters(
        category_mapping=category_mapping,
        naming_strategy='combined',
        max_features=1
    )
    for cluster_id, name in sorted(names_combined.items()):
        print(f"  Cluster {cluster_id}: {name}")

    # Print full summary
    pipeline.print_cluster_summary()

    print("\n✓ Test 2 completed successfully!\n")


def test_manual_naming():
    """Test manual cluster naming workflow."""
    print("=" * 80)
    print("Test 3: Manual Naming Workflow")
    print("=" * 80)

    # Generate data
    X, _ = make_blobs(n_samples=200, n_features=5, centers=3, random_state=42)
    df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(5)])

    # Pipeline without auto naming
    pipeline = ClusterAnalysisPipeline(
        clustering_algorithm='kmeans',
        n_clusters=3,
        auto_name_clusters=False,
        verbose=False
    )

    pipeline.fit(df)

    # Manually call naming with different parameters
    print("\nGenerating names with min_deviation=0.3...")
    names1 = pipeline.name_clusters(min_deviation=0.3, max_features=3)

    print("\nNames with low deviation threshold (more features):")
    for cluster_id, name in sorted(names1.items()):
        print(f"  Cluster {cluster_id}: {name}")

    print("\nGenerating names with min_deviation=1.0...")
    names2 = pipeline.name_clusters(min_deviation=1.0, max_features=2)

    print("\nNames with high deviation threshold (fewer features):")
    for cluster_id, name in sorted(names2.items()):
        print(f"  Cluster {cluster_id}: {name}")

    # Test namer object directly
    print("\n" + "=" * 80)
    print("Testing ClusterNamer object directly")
    print("=" * 80)

    from clustertk.interpretation import ClusterNamer

    namer = ClusterNamer(
        naming_strategy='top_features',
        max_features=1,
        use_directions=False,  # No "High/Low"
        short_names=True
    )

    names_short = namer.generate_names(
        profiles=pipeline.cluster_profiles_,
        top_features=pipeline._profiler.top_features_
    )

    print("\nShort names (no directions):")
    for cluster_id, name in sorted(names_short.items()):
        print(f"  Cluster {cluster_id}: {name}")

    # Export to dict
    print("\n" + "=" * 80)
    print("Export naming results")
    print("=" * 80)

    export_dict = namer.to_dict()
    print("\nExported naming data:")
    for cluster_id, data in sorted(export_dict.items()):
        print(f"\nCluster {cluster_id}:")
        print(f"  Name: {data['name']}")
        print(f"  Description: {data['description']}")
        print(f"  Metadata: {data['metadata']}")

    print("\n✓ Test 3 completed successfully!\n")


def main():
    """Run all naming tests."""
    print("\n" + "=" * 80)
    print("CLUSTER NAMING TESTS")
    print("=" * 80 + "\n")

    test_basic_naming()
    test_category_naming()
    test_manual_naming()

    print("=" * 80)
    print("ALL TESTS COMPLETED SUCCESSFULLY!")
    print("=" * 80)


if __name__ == '__main__':
    main()
