"""
Data Preprocessor Module
========================
Handles data loading, cleaning, and train-test splitting.
Auto-handles text targets, too many classes, and missing values.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from core.config import PREPROCESSING_CONFIG, MISSING_VALUE_STRATEGY, SPLIT_CONFIG


class DataPreprocessor:
    """Handles data preprocessing operations with auto-cleaning."""

    def __init__(self):
        self.df = None
        self.target_col = None
        self.label_encoders = {}
        self.cleaning_info = {}

    def load_data(self, file) -> pd.DataFrame:
        """Load data from CSV file."""
        self.df = pd.read_csv(file)
        return self.df

    def set_target(self, target_col: str):
        """Set the target column for prediction."""
        self.target_col = target_col

    def get_data_info(self) -> dict:
        """Return information about the loaded data."""
        if self.df is None:
            return {}

        return {
            'n_rows': len(self.df),
            'n_cols': len(self.df.columns),
            'features': [col for col in self.df.columns if col != self.target_col],
            'target': self.target_col,
            'missing_values': self.df.isnull().sum().to_dict(),
            'dtypes': self.df.dtypes.to_dict()
        }

    def get_recommended_targets(self) -> list:
        """Get list of recommended target columns (numeric/categorical with reasonable classes)."""
        if self.df is None:
            return []

        recommendations = []
        for col in self.df.columns:
            n_unique = self.df[col].nunique()

            # Good targets: 2-N classes OR numeric with <=M unique values
            if n_unique >= PREPROCESSING_CONFIG['recommended_classes_min'] and n_unique <= PREPROCESSING_CONFIG['recommended_classes_max']:
                recommendations.append({
                    'column': col,
                    'classes': n_unique,
                    'dtype': str(self.df[col].dtype),
                    'status': 'recommended'
                })
            elif n_unique > PREPROCESSING_CONFIG['recommended_classes_max'] and n_unique <= PREPROCESSING_CONFIG['will_be_binned_max']:
                recommendations.append({
                    'column': col,
                    'classes': n_unique,
                    'dtype': str(self.df[col].dtype),
                    'status': 'will_be_binned'
                })
            elif n_unique > PREPROCESSING_CONFIG['will_be_binned_max']:
                recommendations.append({
                    'column': col,
                    'classes': n_unique,
                    'dtype': str(self.df[col].dtype),
                    'status': 'too_many_classes'
                })

        return recommendations

    def auto_clean_target(self):
        """Automatically clean and prepare the target column."""
        if self.df is None or self.target_col is None:
            return

        target = self.df[self.target_col]
        self.cleaning_info = {'actions': []}

        if target.dtype == 'object' or str(target.dtype).startswith('str'):
            try:
                numeric_converted = pd.to_numeric(target, errors='coerce')
                if numeric_converted.notna().mean() > PREPROCESSING_CONFIG['numeric_convert_threshold']:
                    self.df[self.target_col] = numeric_converted
                    self.cleaning_info['actions'].append('Converted text to numbers')
                else:
                    le = LabelEncoder()
                    self.df[self.target_col] = le.fit_transform(target.astype(str))
                    self.label_encoders[self.target_col] = le
                    self.cleaning_info['actions'].append(f'Encoded {len(le.classes_)} text labels')
            except Exception:
                le = LabelEncoder()
                self.df[self.target_col] = le.fit_transform(target.astype(str))
                self.label_encoders[self.target_col] = le
                self.cleaning_info['actions'].append(f'Encoded {len(le.classes_)} text labels')

        # Handle too many unique values (regression -> classification by binning)
        n_unique = self.df[self.target_col].nunique()
        if n_unique > PREPROCESSING_CONFIG['target_nunique_threshold']:
            n_bins = PREPROCESSING_CONFIG['n_bins']
            try:
                # Bin into configured number of quantiles
                binned_col = pd.qcut(
                    self.df[self.target_col],
                    q=n_bins,
                    labels=False,
                    duplicates='drop'
                )
                actual_bins = binned_col.nunique()
                self.df[self.target_col] = binned_col
                self.cleaning_info['actions'].append(f'Binned {n_unique} values into {actual_bins} categories')
            except Exception:
                try:
                    binned_col = pd.cut(
                        self.df[self.target_col],
                        bins=n_bins,
                        labels=False,
                        duplicates='drop'
                    )
                    actual_bins = binned_col.nunique()
                    self.df[self.target_col] = binned_col
                    self.cleaning_info['actions'].append(f'Binned {n_unique} values into {actual_bins} categories')
                except Exception:
                    pass  # Keep original if binning fails

        # Fill any NaN created during conversion
        if self.df[self.target_col].isnull().any():
            self.df[self.target_col] = self.df[self.target_col].fillna(
                self.df[self.target_col].mode().iloc[0] if len(self.df[self.target_col].mode()) > 0 else 0
            )
            self.cleaning_info['actions'].append('Filled NaN in target with mode')

    def handle_missing(self, X, strategy=None):
        """Handle missing values in the dataset."""
        if X is None or X.shape[0] == 0:
            return X

        if strategy is None:
            strategy = MISSING_VALUE_STRATEGY

        X_clean = X.copy()
        for col in X_clean.columns:
            if X_clean[col].dtype in ['float64', 'int64', 'float32', 'int32', 'float16', 'int16', 'int8']:
                if X_clean[col].isnull().any():
                    if strategy == 'mean':
                        X_clean[col] = X_clean[col].fillna(X_clean[col].mean())
                    elif strategy == 'median':
                        X_clean[col] = X_clean[col].fillna(X_clean[col].median())
                    elif strategy == 'zero':
                        X_clean[col] = X_clean[col].fillna(0)

        return X_clean

    def get_missing_info(self) -> dict:
        """Return information about missing values."""
        if self.df is None:
            return {}

        missing = self.df.isnull().sum()
        return {
            'total_missing': missing.sum(),
            'columns_with_missing': missing[missing > 0].to_dict(),
            'missing_percentage': (missing / len(self.df) * 100).to_dict()
        }

    def _encode_features(self, X):
        """Encode categorical features in X."""
        if X is None or X.shape[0] == 0:
            return X

        cat_cols = X.select_dtypes(include=['object']).columns
        for col in cat_cols:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
            self.label_encoders[f'feature_{col}'] = le
        return X

    def split_data(self, test_size=None, random_state=None, handle_missing=None):
        """Split data into train and test sets with automatic cleaning."""
        if test_size is None:
            test_size = SPLIT_CONFIG['test_size']
        if random_state is None:
            random_state = SPLIT_CONFIG['random_state']

        X = self.df.drop(columns=[self.target_col])
        y = self.df[self.target_col]

        # Auto-clean target (encode text, bin too many values)
        self.auto_clean_target()
        y = self.df[self.target_col]

        # Encode categorical features in X
        X = self._encode_features(X)

        # Handle missing values in features
        X = self.handle_missing(X, strategy=handle_missing)

        # Drop any rows with NaN in target
        valid_idx = ~y.isnull()
        X = X[valid_idx]
        y = y[valid_idx]

        # If still having issues, drop remaining NaN rows
        if X.isnull().any().any():
            valid_idx = ~X.isnull().any(axis=1)
            X = X[valid_idx]
            y = y[valid_idx]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )

        # Ensure y values are integers for classification
        y_train = y_train.astype(int)
        y_test = y_test.astype(int)

        return X_train, X_test, y_train, y_test