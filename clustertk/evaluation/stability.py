"""
Cluster stability analysis using bootstrap resampling.

This module provides tools for assessing the stability and reliability
of clustering results through repeated resampling and consensus analysis.

Optimized for large datasets with streaming computation and vectorized operations.
"""

from typing import Dict, Optional, Tuple, Callable, List
from collections import deque
import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score
from sklearn.utils import resample


class ClusterStabilityAnalyzer:
    """
    Analyze clustering stability using bootstrap resampling.

    This class performs multiple clustering runs on bootstrap samples
    and computes stability metrics to assess the reliability of clusters.

    Parameters
    ----------
    n_iterations : int, default=100
        Number of bootstrap iterations to perform.
    sample_fraction : float, default=0.8
        Fraction of samples to use in each bootstrap iteration.
    random_state : int or None, default=None
        Random state for reproducibility.
    verbose : bool, default=False
        Whether to print progress messages.

    Attributes
    ----------
    overall_stability_ : float
        Overall stability score (0-1, higher = more stable).
    cluster_stability_ : pd.DataFrame
        Per-cluster stability scores.
    sample_confidence_ : np.ndarray
        Per-sample confidence scores (how consistently each sample
        is assigned to the same cluster).
    pairwise_ari_ : np.ndarray
        Pairwise Adjusted Rand Index between all iterations.

    Examples
    --------
    >>> from clustertk.evaluation import ClusterStabilityAnalyzer
    >>> from clustertk.clustering import KMeansClustering
    >>>
    >>> # Create clusterer
    >>> clusterer = KMeansClustering(n_clusters=3)
    >>>
    >>> # Analyze stability
    >>> analyzer = ClusterStabilityAnalyzer(n_iterations=50)
    >>> results = analyzer.analyze(X, clusterer)
    >>>
    >>> print(f"Overall stability: {results['overall_stability']:.3f}")
    >>> print(results['cluster_stability'])
    """

    def __init__(
        self,
        n_iterations: int = 100,
        sample_fraction: float = 0.8,
        random_state: Optional[int] = None,
        verbose: bool = False,
        max_comparison_window: int = 10,
        max_pairs_per_cluster: int = 5000
    ):
        """
        Initialize stability analyzer with optimized parameters.

        Parameters
        ----------
        n_iterations : int, default=100
            Number of bootstrap iterations.
        sample_fraction : float, default=0.8
            Fraction of samples per bootstrap.
        random_state : int or None, default=None
            Random state for reproducibility.
        verbose : bool, default=False
            Print progress messages.
        max_comparison_window : int, default=10
            Maximum number of recent iterations to keep for ARI comparison.
            Reduces memory usage from O(n_iterations) to O(max_comparison_window).
        max_pairs_per_cluster : int, default=5000
            Maximum number of sample pairs to check per cluster.
            If cluster has more pairs, randomly sample this many.
            Reduces time complexity from O(cluster_size²) to O(max_pairs).
        """
        self.n_iterations = n_iterations
        self.sample_fraction = sample_fraction
        self.random_state = random_state
        self.verbose = verbose
        self.max_comparison_window = max_comparison_window
        self.max_pairs_per_cluster = max_pairs_per_cluster

        # Results storage
        self.overall_stability_: Optional[float] = None
        self.cluster_stability_: Optional[pd.DataFrame] = None
        self.sample_confidence_: Optional[np.ndarray] = None
        self.pairwise_ari_: Optional[np.ndarray] = None
        self._bootstrap_labels_: Optional[List] = None  # Only stores last few iterations

    def analyze(
        self,
        X: pd.DataFrame,
        clusterer,
        fit_params: Optional[Dict] = None
    ) -> Dict:
        """
        Perform stability analysis on clustering algorithm.

        Optimized version with streaming computation and vectorized operations.
        Memory usage: O(n_samples + max_comparison_window) instead of O(n_samples × n_iterations).

        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix to cluster.
        clusterer : object
            Clustering algorithm with fit() and predict() or fit_predict() methods.
            Must be unfitted or will be cloned.
        fit_params : dict or None, default=None
            Additional parameters to pass to clusterer.fit().

        Returns
        -------
        results : dict
            Dictionary containing:
            - 'overall_stability': Overall stability score
            - 'cluster_stability': DataFrame with per-cluster stability
            - 'sample_confidence': Array with per-sample confidence scores
            - 'mean_ari': Mean Adjusted Rand Index across all iterations
            - 'stable_clusters': List of stable cluster IDs (stability > 0.7)
            - 'unstable_clusters': List of unstable cluster IDs (stability < 0.5)

        Examples
        --------
        >>> results = analyzer.analyze(X, KMeansClustering(n_clusters=3))
        >>> print(f"Stability: {results['overall_stability']:.3f}")
        """
        if self.verbose:
            print(f"Starting optimized stability analysis with {self.n_iterations} iterations...")
            print(f"  Dataset size: {len(X):,} samples")
            print(f"  Bootstrap size: {int(len(X) * self.sample_fraction):,} samples per iteration")
            print(f"  Memory optimization: using sliding window of {self.max_comparison_window} iterations")

        n_samples = len(X)
        n_bootstrap = int(n_samples * self.sample_fraction)

        if fit_params is None:
            fit_params = {}

        # Get reference labels by fitting on full data first
        try:
            from sklearn.base import clone
            clusterer_full = clone(clusterer)
        except:
            clusterer_full = clusterer

        try:
            clusterer_full.fit(X, **fit_params)
            reference_labels = clusterer_full.predict(X)
        except:
            reference_labels = clusterer_full.fit_predict(X, **fit_params)

        # Initialize streaming accumulators
        rng = np.random.RandomState(self.random_state)

        # For overall stability: sliding window of recent iterations
        recent_iterations = deque(maxlen=self.max_comparison_window)
        ari_scores = []

        # For sample confidence: streaming counters
        confidence_sum = np.zeros(n_samples, dtype=np.float64)
        appearance_count = np.zeros(n_samples, dtype=np.int32)

        # For cluster stability: streaming pair counters per cluster
        unique_clusters = np.unique(reference_labels)
        unique_clusters = unique_clusters[unique_clusters != -1]  # Exclude noise

        cluster_pair_stats = {}
        for cluster_id in unique_clusters:
            cluster_pair_stats[cluster_id] = {
                'together_count': 0,
                'total_pairs': 0
            }

        # Perform bootstrap iterations with streaming computation
        for i in range(self.n_iterations):
            if self.verbose and (i + 1) % 20 == 0:
                print(f"  Iteration {i + 1}/{self.n_iterations}")

            # Generate bootstrap sample
            indices = rng.choice(n_samples, size=n_bootstrap, replace=True)
            X_bootstrap = X.iloc[indices]

            # Fit clusterer on bootstrap sample
            try:
                from sklearn.base import clone
                clusterer_clone = clone(clusterer)
            except:
                clusterer_clone = clusterer

            try:
                clusterer_clone.fit(X_bootstrap, **fit_params)
                labels_bootstrap = clusterer_clone.predict(X_bootstrap)
            except:
                labels_bootstrap = clusterer_clone.fit_predict(X_bootstrap, **fit_params)

            # === STREAMING COMPUTATION: Update metrics incrementally ===

            # 1. Update sample confidence (streaming)
            self._update_sample_confidence_streaming(
                indices, labels_bootstrap, reference_labels,
                confidence_sum, appearance_count
            )

            # 2. Update cluster stability (streaming)
            self._update_cluster_stability_streaming(
                indices, labels_bootstrap, reference_labels,
                unique_clusters, cluster_pair_stats, rng
            )

            # 3. Update overall stability with sliding window
            current_iteration = (labels_bootstrap, indices)

            # Compare with recent iterations
            for prev_labels, prev_indices in recent_iterations:
                ari = self._compute_ari_fast(
                    labels_bootstrap, indices,
                    prev_labels, prev_indices
                )
                if ari is not None:
                    ari_scores.append(ari)

            # Add current to window
            recent_iterations.append(current_iteration)

        # Compute final metrics from streaming accumulators
        if self.verbose:
            print("  Finalizing stability metrics...")

        # 1. Sample confidence
        mask = appearance_count > 0
        sample_confidence = np.zeros(n_samples, dtype=np.float64)
        sample_confidence[mask] = confidence_sum[mask] / appearance_count[mask]

        # 2. Cluster stability
        cluster_stability = self._finalize_cluster_stability(
            unique_clusters, cluster_pair_stats, reference_labels
        )

        # 3. Overall stability
        overall_stability = float(np.mean(ari_scores)) if ari_scores else 0.0
        mean_ari = overall_stability

        # Store results
        self.overall_stability_ = overall_stability
        self.cluster_stability_ = cluster_stability
        self.sample_confidence_ = sample_confidence
        self.pairwise_ari_ = np.array(ari_scores)
        self._bootstrap_labels_ = list(recent_iterations)  # Only last few iterations

        # Identify stable and unstable clusters
        stable_threshold = 0.7
        unstable_threshold = 0.5

        stable_clusters = cluster_stability[
            cluster_stability['stability'] >= stable_threshold
        ]['cluster'].tolist()

        unstable_clusters = cluster_stability[
            cluster_stability['stability'] < unstable_threshold
        ]['cluster'].tolist()

        if self.verbose:
            print(f"\n✓ Stability analysis complete!")
            print(f"  Overall stability: {overall_stability:.3f}")
            print(f"  Mean ARI: {mean_ari:.3f}")
            print(f"  Stable clusters: {len(stable_clusters)}")
            print(f"  Unstable clusters: {len(unstable_clusters)}")

        return {
            'overall_stability': overall_stability,
            'cluster_stability': cluster_stability,
            'sample_confidence': sample_confidence,
            'mean_ari': mean_ari,
            'stable_clusters': stable_clusters,
            'unstable_clusters': unstable_clusters,
            'reference_labels': reference_labels
        }

    def _update_sample_confidence_streaming(
        self,
        indices: np.ndarray,
        labels_bootstrap: np.ndarray,
        reference_labels: np.ndarray,
        confidence_sum: np.ndarray,
        appearance_count: np.ndarray
    ) -> None:
        """
        Update sample confidence scores incrementally (streaming).

        This replaces the old approach of storing all bootstrap labels
        and computing confidence at the end.

        Time complexity: O(n_bootstrap) per iteration
        Space complexity: O(1) additional memory
        """
        # Vectorized update - much faster than loop
        appearance_count[indices] += 1

        # Check which samples got their reference label
        matches = labels_bootstrap == reference_labels[indices]
        confidence_sum[indices[matches]] += 1

    def _update_cluster_stability_streaming(
        self,
        indices: np.ndarray,
        labels_bootstrap: np.ndarray,
        reference_labels: np.ndarray,
        unique_clusters: np.ndarray,
        cluster_pair_stats: Dict,
        rng: np.random.RandomState
    ) -> None:
        """
        Update cluster stability metrics incrementally (streaming).

        Uses adaptive sampling: if a cluster has too many pairs,
        randomly sample max_pairs_per_cluster pairs instead of checking all.

        Time complexity: O(n_clusters × max_pairs_per_cluster) per iteration
        Space complexity: O(1) additional memory
        """
        for cluster_id in unique_clusters:
            # Get samples from this reference cluster that appear in bootstrap
            cluster_mask = reference_labels == cluster_id
            cluster_samples_in_ref = np.where(cluster_mask)[0]

            # Find which of these appear in current bootstrap
            mask = np.isin(cluster_samples_in_ref, indices)
            present_samples = cluster_samples_in_ref[mask]

            if len(present_samples) < 2:
                continue

            # Get bootstrap labels for these samples
            # Use searchsorted for fast lookup (requires sorted indices)
            sorted_idx = np.argsort(indices)
            positions = sorted_idx[np.searchsorted(indices[sorted_idx], present_samples)]
            bootstrap_labels_for_cluster = labels_bootstrap[positions]

            # Count pairs - adaptive sampling for large clusters
            n_samples_in_cluster = len(present_samples)
            total_possible_pairs = n_samples_in_cluster * (n_samples_in_cluster - 1) // 2

            if total_possible_pairs <= self.max_pairs_per_cluster:
                # Small cluster: check all pairs with vectorization
                same_label = bootstrap_labels_for_cluster[:, None] == bootstrap_labels_for_cluster[None, :]
                upper_triangle = np.triu(same_label, k=1)
                together_count = upper_triangle.sum()
                total_pairs = total_possible_pairs
            else:
                # Large cluster: sample random pairs
                n_pairs_to_sample = self.max_pairs_per_cluster

                # Generate random pairs
                idx1 = rng.randint(0, n_samples_in_cluster, size=n_pairs_to_sample)
                idx2 = rng.randint(0, n_samples_in_cluster, size=n_pairs_to_sample)

                # Ensure idx1 != idx2
                same_idx = idx1 == idx2
                idx2[same_idx] = (idx2[same_idx] + 1) % n_samples_in_cluster

                # Vectorized comparison
                together = bootstrap_labels_for_cluster[idx1] == bootstrap_labels_for_cluster[idx2]
                together_count = together.sum()
                total_pairs = n_pairs_to_sample

            # Update streaming counters
            cluster_pair_stats[cluster_id]['together_count'] += together_count
            cluster_pair_stats[cluster_id]['total_pairs'] += total_pairs

    def _finalize_cluster_stability(
        self,
        unique_clusters: np.ndarray,
        cluster_pair_stats: Dict,
        reference_labels: np.ndarray
    ) -> pd.DataFrame:
        """
        Compute final cluster stability scores from streaming counters.
        """
        cluster_stabilities = []

        for cluster_id in unique_clusters:
            stats = cluster_pair_stats[cluster_id]
            together_count = stats['together_count']
            total_pairs = stats['total_pairs']

            if total_pairs > 0:
                stability = together_count / total_pairs
            else:
                stability = 0.0

            cluster_size = np.sum(reference_labels == cluster_id)

            cluster_stabilities.append({
                'cluster': int(cluster_id),
                'stability': float(stability),
                'size': int(cluster_size)
            })

        df = pd.DataFrame(cluster_stabilities)
        df = df.sort_values('stability', ascending=False).reset_index(drop=True)
        return df

    def _compute_ari_fast(
        self,
        labels1: np.ndarray,
        indices1: np.ndarray,
        labels2: np.ndarray,
        indices2: np.ndarray
    ) -> Optional[float]:
        """
        Compute ARI between two bootstrap iterations on overlapping samples.

        Optimized version using np.searchsorted instead of np.where() in loop.

        Time complexity: O(overlap_size × log(n)) instead of O(overlap_size × n)
        """
        # Find overlapping sample indices
        overlap = np.intersect1d(indices1, indices2)

        if len(overlap) < 2:
            return None

        # Fast lookup using searchsorted
        sorted_idx1 = np.argsort(indices1)
        sorted_idx2 = np.argsort(indices2)

        pos1 = sorted_idx1[np.searchsorted(indices1[sorted_idx1], overlap)]
        pos2 = sorted_idx2[np.searchsorted(indices2[sorted_idx2], overlap)]

        labels1_overlap = labels1[pos1]
        labels2_overlap = labels2[pos2]

        # Compute ARI
        ari = adjusted_rand_score(labels1_overlap, labels2_overlap)
        return float(ari)


    def get_stable_samples(self, threshold: float = 0.7) -> np.ndarray:
        """
        Get indices of samples with high confidence scores.

        Parameters
        ----------
        threshold : float, default=0.7
            Minimum confidence score to be considered stable.

        Returns
        -------
        stable_indices : np.ndarray
            Indices of stable samples.
        """
        if self.sample_confidence_ is None:
            raise ValueError("Run analyze() first")

        return np.where(self.sample_confidence_ >= threshold)[0]

    def get_unstable_samples(self, threshold: float = 0.5) -> np.ndarray:
        """
        Get indices of samples with low confidence scores.

        Parameters
        ----------
        threshold : float, default=0.5
            Maximum confidence score to be considered unstable.

        Returns
        -------
        unstable_indices : np.ndarray
            Indices of unstable samples.
        """
        if self.sample_confidence_ is None:
            raise ValueError("Run analyze() first")

        return np.where(self.sample_confidence_ < threshold)[0]


def quick_stability_analysis(
    X: pd.DataFrame,
    clusterer,
    n_iterations: int = 50,
    sample_fraction: float = 0.8,
    random_state: Optional[int] = 42
) -> Dict:
    """
    Quick stability analysis (convenience function).

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix.
    clusterer : object
        Clustering algorithm instance.
    n_iterations : int, default=50
        Number of bootstrap iterations.
    sample_fraction : float, default=0.8
        Fraction of samples per iteration.
    random_state : int or None, default=42
        Random state for reproducibility.

    Returns
    -------
    results : dict
        Stability analysis results.

    Examples
    --------
    >>> from clustertk.evaluation import quick_stability_analysis
    >>> from clustertk.clustering import KMeansClustering
    >>>
    >>> results = quick_stability_analysis(
    ...     X=df,
    ...     clusterer=KMeansClustering(n_clusters=3),
    ...     n_iterations=50
    ... )
    >>> print(f"Stability: {results['overall_stability']:.3f}")
    """
    analyzer = ClusterStabilityAnalyzer(
        n_iterations=n_iterations,
        sample_fraction=sample_fraction,
        random_state=random_state,
        verbose=False
    )

    return analyzer.analyze(X, clusterer)
