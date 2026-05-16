"""
Model Trainer Module
====================
Handles ML model training, prediction, and performance measurement.
"""

import time
import psutil
import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
from core.config import MODEL_CONFIG, RANDOM_STATE


class ModelTrainer:
    """Handles ML model training and evaluation."""

    def __init__(self):
        self.models = {
            'Logistic Regression': LogisticRegression(
                max_iter=MODEL_CONFIG['Logistic Regression'].get('max_iter', 1000),
                random_state=RANDOM_STATE
            ),
            'Decision Tree': DecisionTreeClassifier(random_state=RANDOM_STATE),
            'Random Forest': RandomForestClassifier(
                n_estimators=MODEL_CONFIG['Random Forest'].get('n_estimators', 100),
                random_state=RANDOM_STATE,
                n_jobs=-1
            ),
            'Gradient Boosting': GradientBoostingClassifier(
                n_estimators=MODEL_CONFIG['Gradient Boosting'].get('n_estimators', 100),
                random_state=RANDOM_STATE
            ),
            'SVM': SVC(random_state=RANDOM_STATE),
            'KNN': KNeighborsClassifier(
                n_neighbors=MODEL_CONFIG['KNN'].get('n_neighbors', 5)
            ),
            'Naive Bayes': GaussianNB()
        }

    def get_model(self, name: str):
        """Get model by name."""
        return self.models.get(name)

    def train_model(self, model_name: str, X_train, X_test, y_train, y_test) -> dict:
        """
        Train a model and measure performance metrics.

        Args:
            model_name: Name of the model to train
            X_train, X_test: Training and test features
            y_train, y_test: Training and test labels

        Returns:
            Dictionary with all performance metrics
        """
        import gc
        gc.collect()

        model = self.get_model(model_name)
        if model is None:
            raise ValueError(f"Unknown model: {model_name}")

        # Measure training time and memory more reliably
        process = psutil.Process()

        start_time = time.time()
        model.fit(X_train, y_train)
        training_time = time.time() - start_time

        # Memory: estimate based on model size and data
        import sys
        memory_used = sys.getsizeof(model) / 1024 / 1024  # Model size in MB

        # Add data memory estimate (handle both numpy arrays and pandas DataFrames)
        try:
            if hasattr(X_train, 'values'):
                x_bytes = X_train.values.nbytes
            else:
                x_bytes = X_train.nbytes
            if hasattr(y_train, 'values'):
                y_bytes = y_train.values.nbytes
            else:
                y_bytes = y_train.nbytes
            memory_used += (x_bytes + y_bytes) / 1024 / 1024 / 100  # Rough estimate
        except:
            memory_used = 0.1  # Default fallback

        # Measure inference time
        start_time = time.time()
        y_pred = model.predict(X_test)
        inference_time = time.time() - start_time

        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)

        return {
            'model_name': model_name,
            'model': model,
            'y_pred': y_pred,
            'y_test': y_test,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'training_time': training_time,
            'inference_time': inference_time,
            'memory_used': memory_used,
            'confusion_matrix': cm,
            'classification_report': classification_report(y_test, y_pred, zero_division=0)
        }

    def save_model(self, model_name: str, filepath: str):
        """Save a trained model to disk."""
        model = self.get_model(model_name)
        if model is not None:
            joblib.dump(model, filepath)

    def load_model(self, filepath: str):
        """Load a model from disk."""
        return joblib.load(filepath)