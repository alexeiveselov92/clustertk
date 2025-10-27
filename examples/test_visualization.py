"""
Test visualization functions.

This script demonstrates all visualization capabilities of ClusterTK.
NOTE: Requires matplotlib and seaborn: pip install clustertk[viz]
"""

import pandas as pd
import numpy as np
from sklearn.datasets import make_blobs
from clustertk import ClusterAnalysisPipeline


def test_visualizations():
    """Test all visualization methods."""

    print("=" * 80)
    print("ClusterTK Visualization Test")
    print("=" * 80)

    # Generate sample data
    print("\nGenerating sample data...")
    X, _ = make_blobs(
        n_samples=300,
        n_features=8,
        centers=4,
        cluster_std=1.5,
        random_state=42
    )

    # Convert to DataFrame
    df = pd.DataFrame(
        X,
        columns=[f'feature_{i}' for i in range(X.shape[1])]
    )

    print(f"Data shape: {df.shape}")

    # Create and fit pipeline
    print("\nFitting ClusterTK pipeline...")
    pipeline = ClusterAnalysisPipeline(
        handle_missing='median',
        correlation_threshold=0.85,
        pca_variance=0.9,
        clustering_algorithm='kmeans',
        n_clusters=None,  # Auto-detect
        verbose=True
    )

    pipeline.fit(df)

    print("\n" + "=" * 80)
    print("Testing Visualization Functions")
    print("=" * 80)

    # Check if visualization is available
    try:
        from clustertk.visualization import check_viz_available

        if not check_viz_available():
            print("\nVisualization dependencies not installed!")
            print("Install with: pip install matplotlib seaborn")
            print("Or: pip install clustertk[viz]")
            return

        import matplotlib
        matplotlib.use('Agg')  # Non-interactive backend for testing
        import matplotlib.pyplot as plt

        print("\nVisualization dependencies found!")

        # Test 1: Cluster 2D visualization
        print("\n1. Testing plot_clusters_2d()...")
        try:
            fig = pipeline.plot_clusters_2d(method='pca', title='Clusters (PCA)')
            plt.savefig('/tmp/test_clusters_2d.png')
            plt.close(fig)
            print("   ✓ plot_clusters_2d() works")
        except Exception as e:
            print(f"   ✗ plot_clusters_2d() failed: {e}")

        # Test 2: Cluster sizes
        print("\n2. Testing plot_cluster_sizes()...")
        try:
            fig = pipeline.plot_cluster_sizes()
            plt.savefig('/tmp/test_cluster_sizes.png')
            plt.close(fig)
            print("   ✓ plot_cluster_sizes() works")
        except Exception as e:
            print(f"   ✗ plot_cluster_sizes() failed: {e}")

        # Test 3: Cluster heatmap
        print("\n3. Testing plot_cluster_heatmap()...")
        try:
            fig = pipeline.plot_cluster_heatmap()
            plt.savefig('/tmp/test_cluster_heatmap.png')
            plt.close(fig)
            print("   ✓ plot_cluster_heatmap() works")
        except Exception as e:
            print(f"   ✗ plot_cluster_heatmap() failed: {e}")

        # Test 4: Cluster radar
        print("\n4. Testing plot_cluster_radar()...")
        try:
            fig = pipeline.plot_cluster_radar()
            plt.savefig('/tmp/test_cluster_radar.png')
            plt.close(fig)
            print("   ✓ plot_cluster_radar() works")
        except Exception as e:
            print(f"   ✗ plot_cluster_radar() failed: {e}")

        # Test 5: Feature importance
        print("\n5. Testing plot_feature_importance()...")
        try:
            fig = pipeline.plot_feature_importance(cluster_id=0)
            plt.savefig('/tmp/test_feature_importance.png')
            plt.close(fig)
            print("   ✓ plot_feature_importance() works")
        except Exception as e:
            print(f"   ✗ plot_feature_importance() failed: {e}")

        # Test 6: PCA variance
        print("\n6. Testing plot_pca_variance()...")
        try:
            fig = pipeline.plot_pca_variance()
            plt.savefig('/tmp/test_pca_variance.png')
            plt.close(fig)
            print("   ✓ plot_pca_variance() works")
        except Exception as e:
            print(f"   ✗ plot_pca_variance() failed: {e}")

        # Test 7: PCA loadings
        print("\n7. Testing plot_pca_loadings()...")
        try:
            fig = pipeline.plot_pca_loadings(components=[0, 1])
            plt.savefig('/tmp/test_pca_loadings.png')
            plt.close(fig)
            print("   ✓ plot_pca_loadings() works")
        except Exception as e:
            print(f"   ✗ plot_pca_loadings() failed: {e}")

        # Test 8: Correlation matrix
        print("\n8. Testing plot_correlation_matrix()...")
        try:
            fig = pipeline.plot_correlation_matrix()
            plt.savefig('/tmp/test_correlation_matrix.png')
            plt.close(fig)
            print("   ✓ plot_correlation_matrix() works")
        except Exception as e:
            print(f"   ✗ plot_correlation_matrix() failed: {e}")

        # Test 9: Correlation network (skip - requires networkx)
        print("\n9. Testing plot_correlation_network()...")
        try:
            fig = pipeline.plot_correlation_network(threshold=0.3)
            plt.savefig('/tmp/test_correlation_network.png')
            plt.close(fig)
            print("   ✓ plot_correlation_network() works")
        except ImportError:
            print("   ⚠ plot_correlation_network() skipped (networkx not installed)")
        except Exception as e:
            print(f"   ✗ plot_correlation_network() failed: {e}")

        # Test 10: Feature distributions
        print("\n10. Testing plot_feature_distributions()...")
        try:
            fig = pipeline.plot_feature_distributions()
            plt.savefig('/tmp/test_feature_distributions.png')
            plt.close(fig)
            print("   ✓ plot_feature_distributions() works")
        except Exception as e:
            print(f"   ✗ plot_feature_distributions() failed: {e}")

        print("\n" + "=" * 80)
        print("All visualization tests completed!")
        print("Plots saved to /tmp/test_*.png")
        print("=" * 80)

    except ImportError as e:
        print(f"\nVisualization dependencies not available: {e}")
        print("Install with: pip install matplotlib seaborn")
        print("Or: pip install clustertk[viz]")


if __name__ == '__main__':
    test_visualizations()
