"""
Performance Comparator Module
==============================
Compares model performance before and after PCA.
"""

import numpy as np


class PerformanceComparator:
    """Handles performance comparison between models."""

    def compare(self, results: dict) -> dict:
        """
        Compare performance of models.

        Args:
            results: Dictionary with 'before_pca' and 'after_pca' results

        Returns:
            Comparison metrics and summary
        """
        comparison = {
            'models': [],
            'metrics': {}
        }

        # Get models from after_pca results
        if 'after_pca' in results:
            comparison['models'] = list(results['after_pca'].keys())

        # Extract metrics for comparison
        metric_names = ['accuracy', 'precision', 'recall', 'f1_score', 'training_time', 'inference_time', 'memory_used']

        for metric in metric_names:
            comparison['metrics'][metric] = {
                'before_pca': {},
                'after_pca': {}
            }

            if 'before_pca' in results:
                for model_name, result in results['before_pca'].items():
                    comparison['metrics'][metric]['before_pca'][model_name] = result.get(metric, 0)

            if 'after_pca' in results:
                for model_name, result in results['after_pca'].items():
                    comparison['metrics'][metric]['after_pca'][model_name] = result.get(metric, 0)

        # Calculate deltas
        comparison['deltas'] = {}
        for metric in metric_names:
            comparison['deltas'][metric] = {}
            if metric in ['training_time', 'inference_time', 'memory_used']:
                # Lower is better for these
                for model in comparison['models']:
                    before = comparison['metrics'][metric]['before_pca'].get(model, 0)
                    after = comparison['metrics'][metric]['after_pca'].get(model, 0)
                    if before > 0:
                        delta = ((before - after) / before) * 100
                    else:
                        delta = 0
                    comparison['deltas'][metric][model] = delta
            else:
                # Higher is better for these
                for model in comparison['models']:
                    before = comparison['metrics'][metric]['before_pca'].get(model, 0)
                    after = comparison['metrics'][metric]['after_pca'].get(model, 0)
                    if before > 0:
                        delta = ((after - before) / before) * 100
                    else:
                        delta = 0
                    comparison['deltas'][metric][model] = delta

        return comparison

    def get_optimal_recommendation(self, comparison: dict, explained_variance: list) -> dict:
        """
        Get optimal PCA configuration recommendation.

        Args:
            comparison: Comparison results
            explained_variance: List of explained variance ratios

        Returns:
            Optimal recommendation dictionary
        """
        # Find best model based on accuracy
        best_model = None
        best_accuracy = 0

        if 'after_pca' in comparison.get('metrics', {}).get('accuracy', {}):
            for model, accuracy in comparison['metrics']['accuracy']['after_pca'].items():
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    best_model = model

        # Calculate recommended components (95% variance)
        recommended_components = len(explained_variance)
        if len(explained_variance) > 0:
            cumulative = np.cumsum(explained_variance)
            recommended_components = int(np.argmax(cumulative >= 0.95)) + 1

        # Total variance explained
        total_variance = cumulative[-1] if len(cumulative) > 0 else 0

        return {
            'best_model': best_model or 'N/A',
            'best_accuracy': best_accuracy,
            'recommended_components': recommended_components,
            'expected_accuracy': best_accuracy,
            'variance_explained': total_variance
        }

    def get_summary_table(self, comparison: dict) -> list:
        """Generate a summary table for display."""
        table = []
        for model in comparison['models']:
            row = {'Model': model}
            for metric, data in comparison['metrics'].items():
                before = data['before_pca'].get(model, 'N/A')
                after = data['after_pca'].get(model, 'N/A')
                delta = comparison['deltas'].get(metric, {}).get(model, 0)

                if isinstance(before, (int, float)) and isinstance(after, (int, float)):
                    row[f'{metric}_before'] = f"{before:.4f}"
                    row[f'{metric}_after'] = f"{after:.4f}"
                    row[f'{metric}_delta'] = f"{delta:+.2f}%"
                else:
                    row[f'{metric}_before'] = str(before)
                    row[f'{metric}_after'] = str(after)
                    row[f'{metric}_delta'] = 'N/A'

            table.append(row)

        return table