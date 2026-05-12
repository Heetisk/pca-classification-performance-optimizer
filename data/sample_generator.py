"""
Sample Dataset Generator
=========================
Creates sample datasets for testing the PCA Performance Optimizer.
"""

import pandas as pd
import numpy as np
from sklearn.datasets import make_classification


def generate_classification_data(
    n_samples=1000,
    n_features=20,
    n_informative=None,
    n_redundant=None,
    n_classes=2,
    random_state=42
) -> pd.DataFrame:
    """
    Generate a synthetic classification dataset.

    Args:
        n_samples: Number of samples
        n_features: Total number of features
        n_informative: Number of informative features (auto if None)
        n_redundant: Number of redundant features (auto if None)
        n_classes: Number of classes
        random_state: Random seed

    Returns:
        DataFrame with features and target column
    """
    # Auto-calculate informative and redundant features
    if n_informative is None:
        n_informative = max(1, int(n_features * 0.7))
    if n_redundant is None:
        n_redundant = max(0, int(n_features * 0.2))

    # Ensure they don't exceed available features
    max_informative = n_features - 2
    n_informative = min(n_informative, max_informative)
    n_redundant = min(n_redundant, max(0, n_features - n_informative - 2))

    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_informative,
        n_redundant=n_redundant,
        n_classes=n_classes,
        random_state=random_state,
        n_clusters_per_class=2
    )

    # Create feature names
    feature_names = [f'feature_{i}' for i in range(n_features)]

    # Create DataFrame
    df = pd.DataFrame(X, columns=feature_names)
    df['target'] = y

    return df


def save_sample_data(filepath: str):
    """Generate and save a sample dataset."""
    df = generate_classification_data(n_samples=500, n_features=15, n_classes=3)
    df.to_csv(filepath, index=False)
    print(f"Sample data saved to {filepath}")
    print(f"Shape: {df.shape}")
    print(f"Classes: {df['target'].unique()}")


if __name__ == "__main__":
    save_sample_data("data/sample_classification.csv")