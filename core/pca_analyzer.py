"""
PCA Analyzer Module
==================
Handles PCA transformation and variance analysis.
"""

import numpy as np
from sklearn.decomposition import PCA
from core.config import PCA_CONFIG


class PCAAnalyzer:
    """Handles PCA analysis and transformation."""

    def __init__(self):
        self.pca = None
        self.scaler = None
        self.explained_variance_ratio = None
        self.cumulative_variance = None

    def fit_transform(self, X_train, X_test, n_components=None):
        """
        Fit PCA and transform data.

        Args:
            X_train: Training features
            X_test: Test features
            n_components: Number of components (None = all)

        Returns:
            Dictionary with transformed data and analysis results
        """
        if n_components is None:
            n_components = min(X_train.shape)

        self.pca = PCA(n_components=n_components)
        X_train_pca = self.pca.fit_transform(X_train)
        X_test_pca = self.pca.transform(X_test)

        self.explained_variance_ratio = self.pca.explained_variance_ratio_
        self.cumulative_variance = np.cumsum(self.explained_variance_ratio)

        return {
            'X_train_pca': X_train_pca,
            'X_test_pca': X_test_pca,
            'explained_variance': self.explained_variance_ratio,
            'cumulative_variance': self.cumulative_variance,
            'n_components': n_components,
            'total_variance_explained': self.cumulative_variance[-1] if len(self.cumulative_variance) > 0 else 0
        }

    def get_optimal_components(self, threshold=None):
        """
        Get optimal number of components for a given variance threshold.

        Args:
            threshold: Target cumulative variance (e.g., 0.95 for 95%)

        Returns:
            Recommended number of components
        """
        if threshold is None:
            threshold = PCA_CONFIG['variance_threshold']
        if self.cumulative_variance is None:
            return None

        optimal = np.argmax(self.cumulative_variance >= threshold) + 1
        return optimal

    def get_variance_for_components(self, n_components):
        """Get total variance explained by n components."""
        if self.cumulative_variance is None or n_components > len(self.cumulative_variance):
            return None

        return self.cumulative_variance[n_components - 1]