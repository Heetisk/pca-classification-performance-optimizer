"""
Configuration Module
====================
Centralized configuration for all hardcoded values.
"""

# ============================================================================
# SHARED CONSTANTS
# ============================================================================

RANDOM_STATE = 42
DEFAULT_TEST_SIZE = 0.2
VARIANCE_THRESHOLD = 0.95
MAX_CLASSES_FOR_CLASSIFICATION = 20

# ============================================================================
# MODEL CONFIGURATION
# ============================================================================

MODEL_CONFIG = {
    'Logistic Regression': {
        'max_iter': 1000,
    },
    'Decision Tree': {},
    'Random Forest': {
        'n_estimators': 100,
    },
    'Gradient Boosting': {
        'n_estimators': 100,
    },
    'SVM': {},
    'KNN': {
        'n_neighbors': 5,
    },
    'Naive Bayes': {}
}

# ============================================================================
# PREPROCESSING CONFIGURATION
# ============================================================================

PREPROCESSING_CONFIG = {
    'target_nunique_threshold': MAX_CLASSES_FOR_CLASSIFICATION,
    'numeric_convert_threshold': 0.8,
    'n_bins': 10,
    'recommended_classes_min': 2,
    'recommended_classes_max': MAX_CLASSES_FOR_CLASSIFICATION,
    'will_be_binned_max': 50,
}

# Missing value handling
MISSING_VALUE_STRATEGY = 'mean'

# ============================================================================
# PCA CONFIGURATION
# ============================================================================

PCA_CONFIG = {
    'variance_threshold': VARIANCE_THRESHOLD,
    'variance_threshold_min': 0.80,
    'variance_threshold_max': 0.99,
    'max_components_limit': 50,
}

# ============================================================================
# UI CONFIGURATION
# ============================================================================

UI_CONFIG = {
    'test_size_default': DEFAULT_TEST_SIZE,
    'test_size_min': 0.1,
    'test_size_max': 0.4,
    'test_size_step': 0.05,
    'random_state_min': 0,
    'random_state_max': 999,
    'random_state_default': RANDOM_STATE,
    'data_preview_rows': 10,
}

# Chart colors
CHART_COLORS = {
    'before_pca': {
        'accuracy': 'lightcoral',
        'training_time': 'lightblue',
        'inference_time': 'lightsalmon',
        'memory': 'lightgreen',
    },
    'after_pca': {
        'accuracy': 'mediumseagreen',
        'training_time': 'darkblue',
        'inference_time': 'darkred',
        'memory': 'darkgreen',
    }
}

# ============================================================================
# SPLIT CONFIGURATION
# ============================================================================

SPLIT_CONFIG = {
    'test_size': DEFAULT_TEST_SIZE,
    'random_state': RANDOM_STATE,
}

# ============================================================================
# CLASSIFICATION CONFIGURATION
# ============================================================================

CLASSIFICATION_CONFIG = {
    'min_classes': 2,
    'max_classes_for_classification': MAX_CLASSES_FOR_CLASSIFICATION,
}
