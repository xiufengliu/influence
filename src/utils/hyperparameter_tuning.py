import logging
import numpy as np
from itertools import product
from sklearn.model_selection import train_test_split

from src.utils.metrics import evaluate_clustering, calculate_temporal_consistency, calculate_entropy, calculate_conditional_entropy

class HyperparameterTuner:
    """
    Tunes hyperparameters for clustering models using a grid search approach.
    """

    def __init__(self, model_class, param_grid, evaluation_metrics, metric_weights=None, validation_split=0.2, random_state=42):
        self.model_class = model_class
        self.param_grid = param_grid
        self.evaluation_metrics = evaluation_metrics
        self.metric_weights = metric_weights if metric_weights is not None else {metric: 1.0 for metric in evaluation_metrics}
        self.validation_split = validation_split
        self.random_state = random_state
        self.logger = logging.getLogger(__name__)

        # Validate metric_weights
        if not all(metric in self.evaluation_metrics for metric in self.metric_weights):
            raise ValueError("Metric weights must correspond to evaluation metrics.")

    def _calculate_combined_score(self, metrics_results):
        """
        Calculates a combined score from individual metric results.
        Assumes higher values are better for all metrics.
        """
        combined_score = 0.0
        for metric_name, weight in self.metric_weights.items():
            score = metrics_results.get(metric_name, np.nan)
            if np.isnan(score):
                self.logger.warning(f"Metric {metric_name} is NaN. Skipping for combined score.")
                continue
            combined_score += score * weight
        return combined_score

    def tune(self, Z, timestamps, contexts, entity_ids):
        """
        Performs hyperparameter tuning.

        Parameters
        ----------
        Z : numpy.ndarray
            Influence space matrix.
        timestamps : numpy.ndarray
            Timestamps for each instance.
        contexts : numpy.ndarray
            Contextual attributes for each instance.
        entity_ids : numpy.ndarray
            Unique identifiers for entities.

        Returns
        -------
        dict
            Best hyperparameters found.
        float
            Best combined score.
        """
        self.logger.info("Starting hyperparameter tuning...")

        # Split data into training and validation sets
        # Need to ensure that the split preserves temporal order if necessary for time-series
        # For now, a simple train_test_split on indices, assuming data is already ordered by time
        # and that a random split is acceptable for validation of clustering quality.
        # A more robust approach for time-series would be time-series split.
        n_samples = Z.shape[0]
        indices = np.arange(n_samples)
        train_indices, val_indices = train_test_split(
            indices, test_size=self.validation_split, random_state=self.random_state, shuffle=False
        )

        Z_train, Z_val = Z[train_indices], Z[val_indices]
        timestamps_train, timestamps_val = timestamps[train_indices], timestamps[val_indices]
        contexts_train, contexts_val = contexts[train_indices], contexts[val_indices]
        entity_ids_train, entity_ids_val = entity_ids[train_indices], entity_ids[val_indices]

        best_params = None
        best_score = -float('inf')

        # Generate all combinations of hyperparameters
        keys = self.param_grid.keys()
        values = self.param_grid.values()
        
        for p_values in product(*values):
            current_params = dict(zip(keys, p_values))
            self.logger.info(f"Testing params: {current_params}")

            model = self.model_class(**current_params)
            
            try:
                model.fit(Z_train, timestamps_train, contexts_train, entity_ids_train)
                
                # Evaluate on validation set
                val_labels = model.predict(Z_val)
                
                metrics_results = evaluate_clustering(Z_val, val_labels)
                
                # Calculate Temporal Consistency on validation set
                tc_score = calculate_temporal_consistency(val_labels, timestamps_val)
                metrics_results['temporal_consistency'] = tc_score

                # Calculate Normalized Information Gain on validation set
                entropy_labels = calculate_entropy(val_labels)
                conditional_entropy_labels_context = calculate_conditional_entropy(val_labels, contexts_val)
                nig_score = (entropy_labels - conditional_entropy_labels_context) / entropy_labels if entropy_labels > 0 else 0.0
                metrics_results['normalized_information_gain'] = nig_score

                combined_score = self._calculate_combined_score(metrics_results)
                self.logger.info(f"  Metrics: {metrics_results}, Combined Score: {combined_score}")

                if combined_score > best_score:
                    best_score = combined_score
                    best_params = current_params
                    self.logger.info(f"  New best score: {best_score} with params: {best_params}")

            except Exception as e:
                self.logger.error(f"Error during tuning with params {current_params}: {e}")
                continue

        self.logger.info(f"Tuning complete. Best params: {best_params}, Best score: {best_score}")
        return best_params, best_score
