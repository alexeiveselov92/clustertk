"""
Smart correlation-based feature selection with clusterability evaluation.

This module extends basic correlation filtering by evaluating which features
from correlated pairs are better for clustering, using Hopkins statistic
or variance ratio analysis.
"""

from typing import Optional, List, Dict, Literal
import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors

from .correlation import CorrelationFilter


class SmartCorrelationFilter(CorrelationFilter):
    """
    Enhanced correlation filter that keeps the "best" feature from correlated pairs.

    Instead of dropping features by mean correlation alone, evaluates each feature's
    clusterability using Hopkins statistic or variance-based metrics.

    This is especially useful when you have multiple representations of the same
    metric (e.g., revenue_pct vs revenue_decile vs log_revenue) and want to
    automatically select the best one for clustering.

    Parameters
    ----------
    threshold : float, default=0.85
        Absolute correlation threshold. Feature pairs with |correlation| > threshold
        will be considered for removal.

    method : str, default='pearson'
        Correlation method to use:
        - 'pearson': Standard correlation coefficient
        - 'spearman': Spearman rank correlation
        - 'kendall': Kendall Tau correlation

    selection_strategy : {'hopkins', 'variance_ratio', 'mean_corr'}, default='hopkins'
        Strategy for selecting which feature to keep from correlated pairs:
        - 'hopkins': Keep feature with better Hopkins statistic (measures clusterability)
        - 'variance_ratio': Keep feature with higher between/within variance ratio
        - 'mean_corr': Use original behavior (drop feature with highest mean correlation)

    n_samples_hopkins : int, default=200
        Number of samples to use for Hopkins statistic calculation.
        Hopkins is expensive for large datasets, so we sample.

    random_state : int or None, default=42
        Random state for Hopkins statistic sampling.

    verbose : bool, default=False
        Whether to print progress messages.

    Attributes
    ----------
    feature_scores_ : dict
        Clusterability scores for each feature (higher = better for clustering).

    selection_reasons_ : dict
        Explanation of why each feature was kept/dropped.

    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> from clustertk.feature_selection import SmartCorrelationFilter
    >>>
    >>> # Create data with correlated features
    >>> # revenue_pct has better clusterability than revenue_decile
    >>> df = pd.DataFrame({
    ...     'revenue_pct': np.random.rand(1000) * 100,
    ...     'age': np.random.rand(1000) * 80,
    ... })
    >>> df['revenue_decile'] = pd.cut(df['revenue_pct'], bins=10, labels=False)
    >>>
    >>> # Smart filter will keep revenue_pct (better Hopkins statistic)
    >>> filter = SmartCorrelationFilter(selection_strategy='hopkins')
    >>> filtered_df = filter.fit_transform(df)
    >>> print(filter.selection_reasons_)
    """

    def __init__(
        self,
        threshold: float = 0.85,
        method: str = 'pearson',
        selection_strategy: Literal['hopkins', 'variance_ratio', 'mean_corr'] = 'hopkins',
        n_samples_hopkins: int = 200,
        random_state: Optional[int] = 42,
        verbose: bool = False
    ):
        super().__init__(threshold=threshold, method=method)

        valid_strategies = ['hopkins', 'variance_ratio', 'mean_corr']
        if selection_strategy not in valid_strategies:
            raise ValueError(
                f"Invalid selection_strategy '{selection_strategy}'. "
                f"Must be one of {valid_strategies}."
            )

        self.selection_strategy = selection_strategy
        self.n_samples_hopkins = n_samples_hopkins
        self.random_state = random_state
        self.verbose = verbose

        self.feature_scores_: Optional[Dict[str, float]] = None
        self.selection_reasons_: Optional[Dict[str, str]] = None

    def fit(self, X: pd.DataFrame) -> 'SmartCorrelationFilter':
        """
        Identify features to remove based on correlation and clusterability analysis.

        Parameters
        ----------
        X : pd.DataFrame
            Input data.

        Returns
        -------
        self : SmartCorrelationFilter
            Fitted filter.
        """
        if not isinstance(X, pd.DataFrame):
            raise TypeError("X must be a pandas DataFrame")

        if self.verbose:
            print(f"SmartCorrelationFilter: analyzing {len(X.columns)} features...")

        # Compute correlation matrix (parent class)
        self.correlation_matrix_ = X.corr(method=self.method).abs()

        # Find high correlation pairs (parent class)
        self.high_correlation_pairs_ = self._find_high_correlation_pairs()

        if self.verbose:
            print(f"  Found {len(self.high_correlation_pairs_)} high correlation pairs")

        # Compute feature scores if using smart selection
        if self.selection_strategy != 'mean_corr':
            if self.verbose:
                print(f"  Computing clusterability scores (strategy={self.selection_strategy})...")

            self.feature_scores_ = self._compute_feature_scores(X)
        else:
            self.feature_scores_ = None

        # Identify features to drop using selected strategy
        self.features_to_drop_ = self._identify_features_to_drop()

        # Determine selected features
        self.selected_features_ = [
            col for col in X.columns if col not in self.features_to_drop_
        ]

        if self.verbose:
            print(f"  Selected {len(self.selected_features_)}/{len(X.columns)} features")
            print(f"  Dropped {len(self.features_to_drop_)} features: {self.features_to_drop_}")

        return self

    def _compute_feature_scores(self, X: pd.DataFrame) -> Dict[str, float]:
        """
        Compute clusterability score for each feature.

        Returns
        -------
        scores : dict
            Dictionary mapping feature name to clusterability score.
            Higher score = better for clustering.
        """
        if self.selection_strategy == 'hopkins':
            return self._hopkins_scores(X)
        elif self.selection_strategy == 'variance_ratio':
            return self._variance_ratio_scores(X)
        else:
            raise ValueError(f"Unknown selection_strategy: {self.selection_strategy}")

    def _hopkins_scores(self, X: pd.DataFrame) -> Dict[str, float]:
        """
        Compute Hopkins statistic for each feature.

        Hopkins statistic measures the clusterability of data:
        - H close to 0.5: uniformly distributed (not clusterable)
        - H > 0.75: highly clusterable
        - H < 0.25: regularly spaced (not clusterable)

        We compute it for each feature individually to see which features
        are more clusterable.

        For computational efficiency, we sample n_samples_hopkins points.
        """
        scores = {}

        # Determine sample size
        n_samples = min(self.n_samples_hopkins, len(X))
        rng = np.random.RandomState(self.random_state)

        for col in X.columns:
            # Get feature values as 2D array (required by NearestNeighbors)
            feature_data = X[col].values.reshape(-1, 1)

            # Sample if dataset is large
            if len(X) > self.n_samples_hopkins:
                sample_indices = rng.choice(len(X), size=n_samples, replace=False)
                feature_sample = feature_data[sample_indices]
            else:
                feature_sample = feature_data

            # Compute Hopkins statistic
            hopkins = self._compute_hopkins_statistic(feature_sample, rng)
            scores[col] = hopkins

        return scores

    def _compute_hopkins_statistic(
        self,
        X: np.ndarray,
        rng: np.random.RandomState
    ) -> float:
        """
        Compute Hopkins statistic for clusterability assessment.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)
            Data (can be single feature or multiple).
        rng : np.random.RandomState
            Random number generator.

        Returns
        -------
        hopkins : float
            Hopkins statistic (0-1). Values > 0.5 indicate clusterable data.
        """
        n = len(X)
        m = min(n // 10, 50)  # Sample size for Hopkins test

        if m < 2:
            return 0.5  # Not enough data

        # Get data bounds
        data_min = X.min(axis=0)
        data_max = X.max(axis=0)

        # Generate m random samples from data distribution
        random_samples = rng.uniform(
            low=data_min,
            high=data_max,
            size=(m, X.shape[1])
        )

        # Randomly select m points from actual data
        sample_indices = rng.choice(n, size=m, replace=False)
        data_samples = X[sample_indices]

        # Fit nearest neighbors on actual data
        nbrs = NearestNeighbors(n_neighbors=1).fit(X)

        # Find nearest neighbor distances for random samples
        u_distances, _ = nbrs.kneighbors(random_samples)
        u = u_distances.sum()

        # Find nearest neighbor distances for data samples (excluding self)
        nbrs_self = NearestNeighbors(n_neighbors=2).fit(X)
        w_distances, _ = nbrs_self.kneighbors(data_samples)
        w = w_distances[:, 1].sum()  # Take second nearest (first is self)

        # Compute Hopkins statistic
        if (u + w) == 0:
            return 0.5

        hopkins = u / (u + w)

        return hopkins

    def _variance_ratio_scores(self, X: pd.DataFrame) -> Dict[str, float]:
        """
        Compute variance ratio for each feature using quick clustering.

        Uses K-Means with k=3 to quickly estimate between/within variance ratio.
        Higher ratio = better separation = better for clustering.
        """
        from sklearn.cluster import KMeans

        scores = {}

        # Quick K-Means with k=3
        kmeans = KMeans(n_clusters=3, random_state=self.random_state, n_init=3)

        for col in X.columns:
            feature_data = X[[col]]

            # Quick clustering
            labels = kmeans.fit_predict(feature_data)

            # Compute variance ratio (between / within)
            variance_ratio = self._compute_variance_ratio(
                X[col].values,
                labels
            )

            scores[col] = variance_ratio

        return scores

    def _compute_variance_ratio(
        self,
        feature_data: np.ndarray,
        labels: np.ndarray
    ) -> float:
        """
        Compute between-cluster variance / within-cluster variance.

        Higher ratio = better cluster separation.
        """
        unique_labels = np.unique(labels)

        if len(unique_labels) < 2:
            return 0.0

        # Compute cluster means and sizes
        cluster_means = []
        cluster_sizes = []

        for label in unique_labels:
            mask = labels == label
            cluster_data = feature_data[mask]

            if len(cluster_data) > 0:
                cluster_means.append(cluster_data.mean())
                cluster_sizes.append(len(cluster_data))

        cluster_means = np.array(cluster_means)
        cluster_sizes = np.array(cluster_sizes)

        # Overall mean
        overall_mean = feature_data.mean()

        # Between-cluster variance
        between_var = np.sum(
            cluster_sizes * (cluster_means - overall_mean) ** 2
        ) / cluster_sizes.sum()

        # Within-cluster variance
        within_var = 0.0
        for label in unique_labels:
            mask = labels == label
            cluster_data = feature_data[mask]

            if len(cluster_data) > 1:
                within_var += cluster_data.var() * len(cluster_data)

        within_var = within_var / len(feature_data)

        # Variance ratio
        if within_var > 0:
            return between_var / within_var
        else:
            return 0.0

    def _identify_features_to_drop(self) -> List[str]:
        """
        Identify which features to drop from correlated pairs.

        Uses the selected strategy:
        - 'hopkins' or 'variance_ratio': Drop features with LOWER scores
        - 'mean_corr': Use parent class strategy (drop by mean correlation)
        """
        if self.selection_strategy == 'mean_corr':
            # Use parent class strategy
            return super()._identify_features_to_drop()

        if not self.high_correlation_pairs_:
            self.selection_reasons_ = {}
            return []

        # Use greedy approach with feature scores
        features_to_drop = set()
        remaining_pairs = self.high_correlation_pairs_.copy()
        selection_reasons = {}

        while remaining_pairs:
            # Find features involved in remaining pairs
            involved_features = set()
            for pair in remaining_pairs:
                involved_features.add(pair['feature1'])
                involved_features.add(pair['feature2'])

            # Remove features already marked for dropping
            involved_features -= features_to_drop

            if not involved_features:
                break

            # Find feature with LOWEST score among involved features
            feature_to_drop = min(
                involved_features,
                key=lambda f: self.feature_scores_.get(f, 0)
            )

            features_to_drop.add(feature_to_drop)

            # Record reason
            # Find which feature(s) it was competing with
            competing_features = []
            for pair in remaining_pairs:
                if pair['feature1'] == feature_to_drop:
                    competing_features.append(pair['feature2'])
                elif pair['feature2'] == feature_to_drop:
                    competing_features.append(pair['feature1'])

            if competing_features:
                kept_feature = competing_features[0]
                score_dropped = self.feature_scores_.get(feature_to_drop, 0)
                score_kept = self.feature_scores_.get(kept_feature, 0)

                selection_reasons[feature_to_drop] = (
                    f"Dropped in favor of '{kept_feature}' "
                    f"(score: {score_dropped:.3f} vs {score_kept:.3f}, "
                    f"strategy: {self.selection_strategy})"
                )

            # Remove pairs involving the dropped feature
            remaining_pairs = [
                pair for pair in remaining_pairs
                if pair['feature1'] != feature_to_drop and pair['feature2'] != feature_to_drop
            ]

        self.selection_reasons_ = selection_reasons
        return list(features_to_drop)

    def get_feature_scores(self) -> pd.DataFrame:
        """
        Get clusterability scores for all features.

        Returns
        -------
        scores : pd.DataFrame
            DataFrame with feature names and their clusterability scores.
            Higher score = better for clustering.
        """
        if self.feature_scores_ is None:
            raise ValueError(
                "Feature scores not computed. "
                "Make sure selection_strategy != 'mean_corr' and filter is fitted."
            )

        df = pd.DataFrame({
            'feature': list(self.feature_scores_.keys()),
            'score': list(self.feature_scores_.values()),
            'selected': [f not in self.features_to_drop_ for f in self.feature_scores_.keys()]
        })

        return df.sort_values('score', ascending=False).reset_index(drop=True)

    def get_selection_summary(self) -> pd.DataFrame:
        """
        Get detailed summary of feature selection decisions.

        Returns
        -------
        summary : pd.DataFrame
            DataFrame with correlation pairs, scores, and selection reasons.
        """
        if self.high_correlation_pairs_ is None:
            raise ValueError("Filter must be fitted first")

        if not self.high_correlation_pairs_:
            return pd.DataFrame(columns=[
                'feature1', 'feature2', 'correlation',
                'score1', 'score2', 'kept', 'dropped', 'reason'
            ])

        summary_data = []

        for pair in self.high_correlation_pairs_:
            f1 = pair['feature1']
            f2 = pair['feature2']

            # Determine which was kept and which was dropped
            if f1 in self.features_to_drop_:
                kept, dropped = f2, f1
            elif f2 in self.features_to_drop_:
                kept, dropped = f1, f2
            else:
                # Neither dropped (shouldn't happen, but handle gracefully)
                kept, dropped = f1, None

            # Get scores
            score1 = self.feature_scores_.get(f1, None) if self.feature_scores_ else None
            score2 = self.feature_scores_.get(f2, None) if self.feature_scores_ else None

            # Get reason
            reason = self.selection_reasons_.get(dropped, '') if dropped else ''

            summary_data.append({
                'feature1': f1,
                'feature2': f2,
                'correlation': pair['correlation'],
                'score1': score1,
                'score2': score2,
                'kept': kept,
                'dropped': dropped,
                'reason': reason
            })

        df = pd.DataFrame(summary_data)
        return df.sort_values('correlation', ascending=False).reset_index(drop=True)

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"SmartCorrelationFilter(threshold={self.threshold}, "
            f"method='{self.method}', selection_strategy='{self.selection_strategy}')"
        )
