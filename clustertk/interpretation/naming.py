"""
Automatic cluster naming and interpretation.

This module provides functionality for automatically generating
descriptive names for clusters based on their profiles.
"""

from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np


class ClusterNamer:
    """
    Automatically generate descriptive names for clusters.

    This class analyzes cluster profiles and generates meaningful names
    based on distinguishing features and category patterns.

    Parameters
    ----------
    naming_strategy : str, default='auto'
        Strategy for naming clusters:
        - 'auto': Automatically select best strategy
        - 'top_features': Use top distinguishing features
        - 'categories': Use category scores (requires category_mapping)
        - 'combined': Combine features and categories

    max_features : int, default=2
        Maximum number of features to include in the name.

    min_deviation : float, default=0.5
        Minimum deviation from mean to consider a feature significant.

    use_directions : bool, default=True
        Whether to include high/low directions in names (e.g., "High X, Low Y").

    short_names : bool, default=False
        Generate short names (1-2 words) vs descriptive names.

    Attributes
    ----------
    cluster_names_ : Dict[int, str]
        Generated names for each cluster.

    cluster_descriptions_ : Dict[int, str]
        Detailed descriptions for each cluster.

    naming_metadata_ : Dict[int, Dict]
        Metadata about how each cluster was named.

    Examples
    --------
    >>> from clustertk.interpretation import ClusterNamer, ClusterProfiler
    >>>
    >>> # After creating profiles
    >>> profiler = ClusterProfiler()
    >>> profiles = profiler.create_profiles(X, labels)
    >>> top_features = profiler.get_top_features(n=5)
    >>>
    >>> # Generate names
    >>> namer = ClusterNamer(max_features=2)
    >>> names = namer.generate_names(profiles, top_features)
    >>> print(names)
    {0: 'High feature_1, Low feature_2', 1: 'High feature_3', ...}
    """

    def __init__(
        self,
        naming_strategy: str = 'auto',
        max_features: int = 2,
        min_deviation: float = 0.5,
        use_directions: bool = True,
        short_names: bool = False
    ):
        self.naming_strategy = naming_strategy
        self.max_features = max_features
        self.min_deviation = min_deviation
        self.use_directions = use_directions
        self.short_names = short_names

        # Results
        self.cluster_names_: Optional[Dict[int, str]] = None
        self.cluster_descriptions_: Optional[Dict[int, str]] = None
        self.naming_metadata_: Optional[Dict[int, Dict]] = None

    def generate_names(
        self,
        profiles: pd.DataFrame,
        top_features: Dict[int, Dict[str, List]],
        category_scores: Optional[pd.DataFrame] = None,
        category_mapping: Optional[Dict[str, List[str]]] = None
    ) -> Dict[int, str]:
        """
        Generate names for all clusters.

        Parameters
        ----------
        profiles : pd.DataFrame
            Cluster profiles (mean values per cluster).

        top_features : dict
            Top features for each cluster from ClusterProfiler.get_top_features().
            Format: {cluster_id: {'high': [...], 'low': [...]}}

        category_scores : pd.DataFrame, optional
            Category scores from ClusterProfiler.analyze_by_categories().

        category_mapping : dict, optional
            Mapping of category names to feature names.

        Returns
        -------
        names : dict
            Dictionary mapping cluster ID to generated name.
        """
        if not isinstance(profiles, pd.DataFrame):
            raise TypeError("profiles must be a DataFrame")

        if not top_features:
            raise ValueError("top_features cannot be empty")

        self.cluster_names_ = {}
        self.cluster_descriptions_ = {}
        self.naming_metadata_ = {}

        # Determine strategy
        strategy = self._select_strategy(category_scores, category_mapping)

        # Generate names for each cluster
        for cluster_id in profiles.index:
            if cluster_id not in top_features:
                continue

            if strategy == 'categories' and category_scores is not None:
                name, description, metadata = self._name_by_categories(
                    cluster_id, category_scores, top_features[cluster_id]
                )
            elif strategy == 'combined' and category_scores is not None:
                name, description, metadata = self._name_combined(
                    cluster_id, top_features[cluster_id], category_scores
                )
            else:
                # Default: top_features strategy
                name, description, metadata = self._name_by_features(
                    cluster_id, top_features[cluster_id]
                )

            self.cluster_names_[cluster_id] = name
            self.cluster_descriptions_[cluster_id] = description
            self.naming_metadata_[cluster_id] = metadata

        return self.cluster_names_

    def _select_strategy(
        self,
        category_scores: Optional[pd.DataFrame],
        category_mapping: Optional[Dict]
    ) -> str:
        """Select the best naming strategy based on available data."""
        if self.naming_strategy != 'auto':
            return self.naming_strategy

        # Auto-select strategy
        if category_scores is not None and category_mapping is not None:
            # Prefer categories if available and meaningful
            if len(category_scores.columns) >= 2:
                return 'combined'
            return 'categories'

        return 'top_features'

    def _name_by_features(
        self,
        cluster_id: int,
        cluster_top_features: Dict[str, List]
    ) -> Tuple[str, str, Dict]:
        """
        Generate name based on top distinguishing features.

        Returns
        -------
        name : str
            Short cluster name.
        description : str
            Detailed description.
        metadata : dict
            Metadata about naming decision.
        """
        high_features = cluster_top_features.get('high', [])
        low_features = cluster_top_features.get('low', [])

        # Filter by minimum deviation
        high_features = [(f, v) for f, v in high_features if abs(v) >= self.min_deviation]
        low_features = [(f, v) for f, v in low_features if abs(v) >= self.min_deviation]

        # Select features to include
        selected_high = high_features[:self.max_features]
        selected_low = low_features[:self.max_features - len(selected_high)]

        # Generate name
        name_parts = []
        description_parts = []

        if self.use_directions:
            # Include direction (High/Low)
            for feature, value in selected_high:
                clean_name = self._clean_feature_name(feature)
                name_parts.append(f"High {clean_name}")
                description_parts.append(f"high {feature} ({value:+.2f})")

            for feature, value in selected_low:
                clean_name = self._clean_feature_name(feature)
                name_parts.append(f"Low {clean_name}")
                description_parts.append(f"low {feature} ({value:+.2f})")
        else:
            # Just feature names
            for feature, _ in selected_high[:self.max_features]:
                clean_name = self._clean_feature_name(feature)
                name_parts.append(clean_name)

        if not name_parts:
            # Fallback if no significant features
            name = f"Cluster {cluster_id}"
            description = f"Cluster {cluster_id} (no strongly distinguishing features)"
        else:
            name = ", ".join(name_parts)
            if self.short_names and len(name) > 30:
                # Truncate to first feature only
                name = name_parts[0]

            description = f"Cluster {cluster_id}: {', '.join(description_parts)}"

        metadata = {
            'strategy': 'top_features',
            'n_high': len(selected_high),
            'n_low': len(selected_low),
            'features_used': [f for f, _ in selected_high + selected_low]
        }

        return name, description, metadata

    def _name_by_categories(
        self,
        cluster_id: int,
        category_scores: pd.DataFrame,
        cluster_top_features: Dict[str, List]
    ) -> Tuple[str, str, Dict]:
        """
        Generate name based on category scores.

        Returns
        -------
        name : str
            Short cluster name.
        description : str
            Detailed description.
        metadata : dict
            Metadata about naming decision.
        """
        if cluster_id not in category_scores.index:
            # Fallback to feature-based naming
            return self._name_by_features(cluster_id, cluster_top_features)

        # Get deviation from mean for each category
        global_mean = category_scores.mean(axis=0)
        cluster_scores = category_scores.loc[cluster_id]
        deviation = cluster_scores - global_mean

        # Sort by absolute deviation
        top_categories = deviation.abs().nlargest(self.max_features)

        # Generate name
        name_parts = []
        description_parts = []

        for category in top_categories.index:
            dev = deviation[category]
            if abs(dev) < self.min_deviation:
                continue

            clean_name = self._clean_feature_name(category)

            if self.use_directions:
                direction = "High" if dev > 0 else "Low"
                name_parts.append(f"{direction} {clean_name}")
                description_parts.append(f"{direction.lower()} {category} ({dev:+.2f})")
            else:
                name_parts.append(clean_name)

        if not name_parts:
            name = f"Cluster {cluster_id}"
            description = f"Cluster {cluster_id} (average across categories)"
        else:
            name = ", ".join(name_parts)
            if self.short_names and len(name) > 30:
                name = name_parts[0]

            description = f"Cluster {cluster_id}: {', '.join(description_parts)}"

        metadata = {
            'strategy': 'categories',
            'categories_used': list(top_categories.index),
            'category_deviations': {cat: deviation[cat] for cat in top_categories.index}
        }

        return name, description, metadata

    def _name_combined(
        self,
        cluster_id: int,
        cluster_top_features: Dict[str, List],
        category_scores: pd.DataFrame
    ) -> Tuple[str, str, Dict]:
        """
        Generate name combining categories and top features.

        Returns
        -------
        name : str
            Short cluster name.
        description : str
            Detailed description.
        metadata : dict
            Metadata about naming decision.
        """
        # Get category-based name (1 category)
        cat_name, cat_desc, cat_meta = self._name_by_categories(
            cluster_id, category_scores, cluster_top_features
        )

        # Get feature-based name (1 feature)
        old_max = self.max_features
        self.max_features = 1
        feat_name, feat_desc, feat_meta = self._name_by_features(
            cluster_id, cluster_top_features
        )
        self.max_features = old_max

        # Combine
        if cat_name.startswith("Cluster") and feat_name.startswith("Cluster"):
            # Both failed, use simple name
            name = f"Cluster {cluster_id}"
            description = f"Cluster {cluster_id} (mixed characteristics)"
        elif cat_name.startswith("Cluster"):
            # Only features worked
            name = feat_name
            description = feat_desc
        elif feat_name.startswith("Cluster"):
            # Only categories worked
            name = cat_name
            description = cat_desc
        else:
            # Both worked, combine
            name = f"{cat_name} / {feat_name}"
            if self.short_names and len(name) > 30:
                # Prefer category name if too long
                name = cat_name

            description = f"Cluster {cluster_id}: {cat_desc.split(': ')[1]} + {feat_desc.split(': ')[1]}"

        metadata = {
            'strategy': 'combined',
            'category_meta': cat_meta,
            'feature_meta': feat_meta
        }

        return name, description, metadata

    def _clean_feature_name(self, feature_name: str) -> str:
        """
        Clean feature name for display.

        Removes common prefixes/suffixes and formats for readability.
        """
        # Remove common prefixes
        for prefix in ['feature_', 'feat_', 'col_', 'var_']:
            if feature_name.lower().startswith(prefix):
                feature_name = feature_name[len(prefix):]

        # Replace underscores with spaces
        feature_name = feature_name.replace('_', ' ')

        # Capitalize first letter
        feature_name = feature_name.capitalize()

        return feature_name

    def get_name(self, cluster_id: int) -> Optional[str]:
        """
        Get the name for a specific cluster.

        Parameters
        ----------
        cluster_id : int
            Cluster ID.

        Returns
        -------
        name : str or None
            Cluster name, or None if not found.
        """
        if self.cluster_names_ is None:
            raise ValueError("Must call generate_names() first")

        return self.cluster_names_.get(cluster_id)

    def get_description(self, cluster_id: int) -> Optional[str]:
        """
        Get the detailed description for a specific cluster.

        Parameters
        ----------
        cluster_id : int
            Cluster ID.

        Returns
        -------
        description : str or None
            Cluster description, or None if not found.
        """
        if self.cluster_descriptions_ is None:
            raise ValueError("Must call generate_names() first")

        return self.cluster_descriptions_.get(cluster_id)

    def print_summary(self) -> None:
        """Print a summary of all cluster names and descriptions."""
        if self.cluster_names_ is None:
            raise ValueError("Must call generate_names() first")

        print("=" * 80)
        print("CLUSTER NAMES")
        print("=" * 80)

        for cluster_id in sorted(self.cluster_names_.keys()):
            name = self.cluster_names_[cluster_id]
            description = self.cluster_descriptions_[cluster_id]
            metadata = self.naming_metadata_[cluster_id]

            print(f"\nCluster {cluster_id}: {name}")
            print(f"  Description: {description}")
            print(f"  Strategy: {metadata['strategy']}")

        print("\n" + "=" * 80)

    def to_dict(self) -> Dict[int, Dict[str, str]]:
        """
        Export cluster names and descriptions as a dictionary.

        Returns
        -------
        result : dict
            Dictionary mapping cluster ID to dict with 'name', 'description', and 'metadata'.
        """
        if self.cluster_names_ is None:
            raise ValueError("Must call generate_names() first")

        return {
            cluster_id: {
                'name': self.cluster_names_[cluster_id],
                'description': self.cluster_descriptions_[cluster_id],
                'metadata': self.naming_metadata_[cluster_id]
            }
            for cluster_id in self.cluster_names_.keys()
        }
