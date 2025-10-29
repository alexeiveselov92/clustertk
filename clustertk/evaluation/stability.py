"""
Cluster stability analysis using bootstrap resampling.

This module provides tools for assessing the stability and reliability
of clustering results through repeated resampling and consensus analysis.
"""

from typing import Dict, Optional, Tuple, Callable
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
        verbose: bool = False
    ):
        self.n_iterations = n_iterations
        self.sample_fraction = sample_fraction
        self.random_state = random_state
        self.verbose = verbose

        # Results storage
        self.overall_stability_: Optional[float] = None
        self.cluster_stability_: Optional[pd.DataFrame] = None
        self.sample_confidence_: Optional[np.ndarray] = None
        self.pairwise_ari_: Optional[np.ndarray] = None
        self._bootstrap_labels_: Optional[np.ndarray] = None

    def analyze(
        self,
        X: pd.DataFrame,
        clusterer,
        fit_params: Optional[Dict] = None
    ) -> Dict:
        """
        Perform stability analysis on clustering algorithm.

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
            print(f"Starting stability analysis with {self.n_iterations} iterations...")

        n_samples = len(X)
        n_bootstrap = int(n_samples * self.sample_fraction)

        # Storage for bootstrap results
        bootstrap_labels = []
        bootstrap_indices = []

        # Perform bootstrap iterations
        rng = np.random.RandomState(self.random_state)

        for i in range(self.n_iterations):
            if self.verbose and (i + 1) % 20 == 0:
                print(f"  Iteration {i + 1}/{self.n_iterations}")

            # Bootstrap sample
            indices = rng.choice(n_samples, size=n_bootstrap, replace=True)
            X_bootstrap = X.iloc[indices]

            # Fit clusterer on bootstrap sample
            try:
                # Clone clusterer to avoid fitting the same instance
                from sklearn.base import clone
                clusterer_clone = clone(clusterer)
            except:
                # If clone fails, assume clusterer is stateless
                clusterer_clone = clusterer

            # Fit and get labels
            if fit_params is None:
                fit_params = {}

            try:
                clusterer_clone.fit(X_bootstrap, **fit_params)
                labels_bootstrap = clusterer_clone.predict(X_bootstrap)
            except:
                # If predict doesn't exist, use fit_predict
                labels_bootstrap = clusterer_clone.fit_predict(X_bootstrap, **fit_params)

            bootstrap_labels.append(labels_bootstrap)
            bootstrap_indices.append(indices)

        # Store bootstrap results
        self._bootstrap_labels_ = np.array(bootstrap_labels, dtype=object)

        # Compute stability metrics
        if self.verbose:
            print("Computing stability metrics...")

        # 1. Overall stability via pairwise ARI
        overall_stability = self._compute_overall_stability(
            bootstrap_labels, bootstrap_indices
        )
        self.overall_stability_ = overall_stability

        # 2. Per-cluster stability
        # First, fit on full data to get reference labels
        try:
            from sklearn.base import clone
            clusterer_full = clone(clusterer)
        except:
            clusterer_full = clusterer

        try:
            clusterer_full.fit(X, **fit_params if fit_params else {})
            reference_labels = clusterer_full.predict(X)
        except:
            reference_labels = clusterer_full.fit_predict(X, **fit_params if fit_params else {})

        cluster_stability = self._compute_cluster_stability(
            X, reference_labels, bootstrap_labels, bootstrap_indices
        )
        self.cluster_stability_ = cluster_stability

        # 3. Per-sample confidence scores
        sample_confidence = self._compute_sample_confidence(
            X, reference_labels, bootstrap_labels, bootstrap_indices
        )
        self.sample_confidence_ = sample_confidence

        # 4. Mean ARI
        mean_ari = self._compute_mean_ari(bootstrap_labels, bootstrap_indices)

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

    def _compute_overall_stability(
        self,
        bootstrap_labels: list,
        bootstrap_indices: list
    ) -> float:
        """
        Compute overall stability as mean pairwise ARI.

        Uses only overlapping samples between bootstrap iterations.
        """
        n_iterations = len(bootstrap_labels)
        ari_scores = []

        # Sample pairs of iterations (not all pairs for efficiency)
        max_comparisons = min(500, n_iterations * (n_iterations - 1) // 2)
        rng = np.random.RandomState(self.random_state)

        comparisons = 0
        for i in range(n_iterations):
            # Compare with a few random later iterations
            n_compare = min(10, n_iterations - i - 1)
            if n_compare > 0:
                compare_indices = rng.choice(
                    range(i + 1, n_iterations),
                    size=n_compare,
                    replace=False
                )

                for j in compare_indices:
                    if comparisons >= max_comparisons:
                        break

                    # Find overlapping samples
                    indices_i = bootstrap_indices[i]
                    indices_j = bootstrap_indices[j]

                    # Get intersection
                    overlap = np.intersect1d(indices_i, indices_j)

                    if len(overlap) > 1:  # Need at least 2 samples
                        # Get labels for overlapping samples in consistent order
                        # Map from original indices to bootstrap positions
                        labels_i_overlap = []
                        labels_j_overlap = []

                        for orig_idx in sorted(overlap):
                            # Find position in bootstrap i
                            pos_i = np.where(indices_i == orig_idx)[0][0]
                            labels_i_overlap.append(bootstrap_labels[i][pos_i])

                            # Find position in bootstrap j
                            pos_j = np.where(indices_j == orig_idx)[0][0]
                            labels_j_overlap.append(bootstrap_labels[j][pos_j])

                        # Compute ARI
                        ari = adjusted_rand_score(labels_i_overlap, labels_j_overlap)
                        ari_scores.append(ari)
                        comparisons += 1

                if comparisons >= max_comparisons:
                    break

        # Store pairwise ARI
        self.pairwise_ari_ = np.array(ari_scores)

        # Return mean ARI as overall stability
        return np.mean(ari_scores) if ari_scores else 0.0

    def _compute_cluster_stability(
        self,
        X: pd.DataFrame,
        reference_labels: np.ndarray,
        bootstrap_labels: list,
        bootstrap_indices: list
    ) -> pd.DataFrame:
        """
        Compute stability score for each cluster.

        For each cluster, measures how often its members stay together
        across bootstrap iterations.
        """
        n_samples = len(X)
        unique_clusters = np.unique(reference_labels)
        unique_clusters = unique_clusters[unique_clusters != -1]  # Exclude noise

        cluster_stabilities = []

        for cluster_id in unique_clusters:
            # Get samples in this cluster
            cluster_samples = np.where(reference_labels == cluster_id)[0]

            if len(cluster_samples) < 2:
                cluster_stabilities.append({
                    'cluster': int(cluster_id),
                    'stability': 0.0,
                    'size': len(cluster_samples)
                })
                continue

            # Count how many times pairs from this cluster appear together
            pair_counts = []

            for iter_idx, (labels, indices) in enumerate(zip(bootstrap_labels, bootstrap_indices)):
                # Check which cluster samples appear in this bootstrap
                mask = np.isin(cluster_samples, indices)
                present_samples = cluster_samples[mask]

                if len(present_samples) < 2:
                    continue

                # Find where these samples are in the bootstrap
                positions = [np.where(indices == s)[0][0] for s in present_samples]
                bootstrap_cluster_labels = labels[positions]

                # Count pairs that stay together (same label)
                for i in range(len(present_samples)):
                    for j in range(i + 1, len(present_samples)):
                        if bootstrap_cluster_labels[i] == bootstrap_cluster_labels[j]:
                            pair_counts.append(1)
                        else:
                            pair_counts.append(0)

            # Stability = proportion of pairs that stay together
            stability = np.mean(pair_counts) if pair_counts else 0.0

            cluster_stabilities.append({
                'cluster': int(cluster_id),
                'stability': stability,
                'size': len(cluster_samples)
            })

        df = pd.DataFrame(cluster_stabilities)
        df = df.sort_values('stability', ascending=False).reset_index(drop=True)

        return df

    def _compute_sample_confidence(
        self,
        X: pd.DataFrame,
        reference_labels: np.ndarray,
        bootstrap_labels: list,
        bootstrap_indices: list
    ) -> np.ndarray:
        """
        Compute confidence score for each sample.

        Confidence = proportion of bootstrap iterations where sample
        is assigned to its reference cluster.
        """
        n_samples = len(X)
        confidence_scores = np.zeros(n_samples)
        appearance_counts = np.zeros(n_samples)

        for iter_idx, (labels, indices) in enumerate(zip(bootstrap_labels, bootstrap_indices)):
            for idx, sample_idx in enumerate(indices):
                appearance_counts[sample_idx] += 1

                # Check if assigned to same cluster as reference
                if labels[idx] == reference_labels[sample_idx]:
                    confidence_scores[sample_idx] += 1

        # Avoid division by zero
        mask = appearance_counts > 0
        confidence_scores[mask] = confidence_scores[mask] / appearance_counts[mask]

        return confidence_scores

    def _compute_mean_ari(
        self,
        bootstrap_labels: list,
        bootstrap_indices: list
    ) -> float:
        """Compute mean ARI across all pairwise comparisons."""
        if self.pairwise_ari_ is not None:
            return float(np.mean(self.pairwise_ari_))
        return 0.0

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
