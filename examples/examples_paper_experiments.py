#!/usr/bin/env python3
"""
Comprehensive experiment runner for Dynamic Influence-Based Clustering.

This script reproduces all experiments described in the paper:
"Learning Interpretable Dynamics: Influence-Based Clustering of Energy Consumption Time Series"

Usage:
    python run_comprehensive_experiments.py --dataset energy_data --output_dir results/
    python run_comprehensive_experiments.py --dataset steel_industry --all_methods
"""

import argparse
import logging
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import ParameterGrid
from joblib import Parallel, delayed
import warnings
warnings.filterwarnings('ignore')

import config
from src.preprocessing.data_loader import DataLoader
from src.preprocessing.preprocessor import Preprocessor
from src.models.gradient_boost import GradientBoostModel
from src.models.torch_models import LSTMModel, TransformerModel
from src.influence.shap_influence import ShapInfluence
from src.influence.lime_influence import LimeInfluence
from src.influence.spearman_influence import SpearmanInfluence
from src.influence.integrated_gradients_influence import IntegratedGradientsInfluence
from src.clustering.kmeans import KMeansClustering
from src.clustering.hierarchical import HierarchicalClustering
from src.clustering.dynamic_kmeans import DynamicKMeansClustering
from src.clustering.timeseries_baselines import KShapeClustering, DTWKMedoidsClustering
from src.utils.metrics import ClusteringEvaluator
from src.utils.logger import setup_logger
from src.utils.visualization import ClusteringVisualizer


class ComprehensiveExperimentRunner:
    """
    Main experiment runner for the paper experiments.
    """
    
    def __init__(self, output_dir: str, random_seeds: list = None, n_jobs: int = -1):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.random_seeds = random_seeds if random_seeds else [42, 123, 456, 789, 101112]
        self.n_jobs = n_jobs
        
        # Setup logging
        self.logger = setup_logger("experiment_runner", 
                                 level=config.LOGGING_PARAMS["level"],
                                 log_file=self.output_dir / "experiment.log")
        
        # Initialize evaluator
        self.evaluator = ClusteringEvaluator()
        
        # Initialize visualizer
        self.visualizer = ClusteringVisualizer()
        
    def run_comprehensive_experiments(self, dataset_name: str, 
                                   n_clusters_list: list = [3, 5],
                                   methods_subset: list = None):
        """
        Run all comprehensive experiments for a dataset.
        
        Parameters
        ----------
        dataset_name : str
            Name of dataset to use
        n_clusters_list : list
            List of cluster numbers to test
        methods_subset : list, optional
            Subset of methods to run (for debugging)
        """
        self.logger.info(f"Starting comprehensive experiments for {dataset_name}")
        
        # Load and preprocess data
        data_dict = self._load_and_preprocess_data(dataset_name)
        
        # Define experimental configurations
        configurations = self._get_experimental_configurations(methods_subset)
        
        # Run experiments for each cluster number
        all_results = {}
        
        for n_clusters in n_clusters_list:
            self.logger.info(f"Running experiments with {n_clusters} clusters")
            
            cluster_results = self._run_clustering_experiments(
                data_dict, configurations, n_clusters
            )
            
            all_results[f"K_{n_clusters}"] = cluster_results
            
            # Save intermediate results
            self._save_results(all_results, dataset_name, intermediate=True)
        
        # Perform sensitivity analysis
        if 'dynamic_kmeans' in [config['clustering'] for config in configurations]:
            self.logger.info("Running sensitivity analysis")
            sensitivity_results = self._run_sensitivity_analysis(data_dict, n_clusters_list)
            all_results['sensitivity_analysis'] = sensitivity_results
        
        # Save final results
        self._save_results(all_results, dataset_name, intermediate=False)
        
        # Generate visualizations
        self._generate_visualizations(all_results, dataset_name)
        
        return all_results
        
    def _load_and_preprocess_data(self, dataset_name: str) -> dict:
        """Load and preprocess dataset."""
        self.logger.info(f"Loading dataset: {dataset_name}")
        
        data_loader = DataLoader(dataset_name=dataset_name)
        X, y, timestamps, contexts, entity_ids = data_loader.load_data()
        
        # Preprocess
        preprocessor = Preprocessor()
        X_processed = preprocessor.fit_transform(X)
        
        return {
            'X': X_processed,
            'y': y,
            'timestamps': timestamps,
            'contexts': contexts,
            'entity_ids': entity_ids,
            'feature_names': data_loader.get_feature_names()
        }
        
    def _get_experimental_configurations(self, methods_subset: list = None) -> list:
        """Get all experimental configurations from the paper."""
        
        # Base configurations from the paper
        base_configs = [
            # Baseline methods (Table 2 in paper)
            {
                'model': 'gradient_boost',
                'influence': 'raw_features',
                'clustering': 'kmeans',
                'name': 'Standard K-means (Raw Features)'
            },
            {
                'model': 'gradient_boost', 
                'influence': 'raw_features',
                'clustering': 'hierarchical',
                'name': 'Standard Hierarchical (Raw Features)'
            },
            {
                'model': 'gradient_boost',
                'influence': 'raw_features', 
                'clustering': 'k_shape',
                'name': 'k-Shape'
            },
            {
                'model': 'gradient_boost',
                'influence': 'raw_features',
                'clustering': 'dtw_kmedoids', 
                'name': 'DTW K-medoids'
            },
            
            # Proposed methods with different influence representations
            {
                'model': 'gradient_boost',
                'influence': 'spearman',
                'clustering': 'dynamic_kmeans',
                'name': 'Proposed Dynamic K-means (Spearman)'
            },
            {
                'model': 'gradient_boost',
                'influence': 'shap',
                'clustering': 'dynamic_kmeans',
                'name': 'Proposed Dynamic K-means (SHAP)'
            },
            {
                'model': 'gradient_boost',
                'influence': 'lime',
                'clustering': 'dynamic_kmeans',
                'name': 'Proposed Dynamic K-means (LIME)'
            },
            {
                'model': 'lstm',
                'influence': 'integrated_gradients',
                'clustering': 'dynamic_kmeans', 
                'name': 'Proposed Dynamic K-means (IG-LSTM)'
            },
            {
                'model': 'transformer',
                'influence': 'integrated_gradients',
                'clustering': 'dynamic_kmeans',
                'name': 'Proposed Dynamic K-means (IG-Transformer)'
            }
        ]
        
        # Filter if subset specified
        if methods_subset:
            base_configs = [c for c in base_configs if any(method in c['name'].lower() for method in methods_subset)]
            
        return base_configs
        
    def _run_clustering_experiments(self, data_dict: dict, 
                                  configurations: list, 
                                  n_clusters: int) -> dict:
        """Run clustering experiments for all configurations."""
        
        results = {}
        
        # Run experiments in parallel for different random seeds
        for config in configurations:
            self.logger.info(f"Running configuration: {config['name']}")
            
            config_results = []
            
            for seed in self.random_seeds:
                try:
                    # Run single experiment
                    result = self._run_single_experiment(
                        data_dict, config, n_clusters, seed
                    )
                    config_results.append(result)
                    
                except Exception as e:
                    self.logger.error(f"Error in {config['name']} with seed {seed}: {e}")
                    continue
                    
            # Aggregate results across seeds
            if config_results:
                aggregated = self._aggregate_results(config_results)
                results[config['name']] = aggregated
                
        return results
        
    def _run_single_experiment(self, data_dict: dict, config: dict, 
                             n_clusters: int, random_seed: int) -> dict:
        """Run a single clustering experiment."""
        
        start_time = time.time()
        
        # Set random seed
        np.random.seed(random_seed)
        
        X = data_dict['X']
        y = data_dict['y']
        timestamps = data_dict['timestamps']
        contexts = data_dict['contexts']
        entity_ids = data_dict['entity_ids']
        
        # Step 1: Train predictive model
        model = self._get_model(config['model'], random_seed)
        model.fit(X, y)
        
        # Step 2: Generate influence representation
        Z = self._generate_influence_representation(model, X, y, config['influence'], random_seed)
        
        # Step 3: Perform clustering
        clustering_alg = self._get_clustering_algorithm(config['clustering'], n_clusters, random_seed)
        
        if config['clustering'] == 'dynamic_kmeans':
            # Dynamic K-means with temporal and contextual constraints
            clustering_alg.fit(Z, timestamps=timestamps, contexts=contexts, entity_ids=entity_ids)
            transition_matrix = clustering_alg.get_transition_matrix(entity_ids, timestamps)
        else:
            # Standard clustering
            if config['influence'] == 'raw_features':
                clustering_alg.fit(X)  # Use raw features for baselines
            else:
                clustering_alg.fit(Z)  # Use influence space
            transition_matrix = None
            
        labels = clustering_alg.labels_
        
        # Step 4: Evaluate clustering
        evaluation_data = Z if config['influence'] != 'raw_features' else X
        
        metrics = self.evaluator.evaluate_comprehensive(
            Z=evaluation_data,
            labels=labels,
            timestamps=timestamps,
            contexts=contexts,
            entity_ids=entity_ids,
            transition_matrix=transition_matrix
        )
        
        # Add runtime
        metrics['runtime_seconds'] = time.time() - start_time
        metrics['random_seed'] = random_seed
        
        return {
            'metrics': metrics,
            'labels': labels,
            'transition_matrix': transition_matrix,
            'model_performance': self._evaluate_model_performance(model, X, y)
        }
        
    def _get_model(self, model_type: str, random_seed: int):
        """Get and configure predictive model."""
        
        if model_type == 'gradient_boost':
            params = config.MODEL_PARAMS['gradient_boost'].copy()
            params['random_state'] = random_seed
            return GradientBoostModel(**params)
            
        elif model_type == 'lstm':
            params = config.MODEL_PARAMS['lstm'].copy()
            params['random_state'] = random_seed
            return LSTMModel(**params)
            
        elif model_type == 'transformer':
            params = config.MODEL_PARAMS['transformer'].copy()
            params['random_state'] = random_seed
            return TransformerModel(**params)
            
        else:
            raise ValueError(f"Unknown model type: {model_type}")
            
    def _generate_influence_representation(self, model, X, y, influence_type: str, random_seed: int):
        """Generate influence representation."""
        
        if influence_type == 'raw_features':
            return X
            
        elif influence_type == 'spearman':
            influence_gen = SpearmanInfluence(random_state=random_seed)
            return influence_gen.generate_influence(model, X, y)
            
        elif influence_type == 'shap':
            influence_gen = ShapInfluence(random_state=random_seed)
            return influence_gen.generate_influence(model, X)
            
        elif influence_type == 'lime':
            influence_gen = LimeInfluence(random_state=random_seed)
            return influence_gen.generate_influence(model, X)
            
        elif influence_type == 'integrated_gradients':
            influence_gen = IntegratedGradientsInfluence(random_state=random_seed)
            return influence_gen.generate_influence(model, X)
            
        else:
            raise ValueError(f"Unknown influence type: {influence_type}")
            
    def _get_clustering_algorithm(self, clustering_type: str, n_clusters: int, random_seed: int):
        """Get and configure clustering algorithm."""
        
        if clustering_type == 'kmeans':
            return KMeansClustering(n_clusters=n_clusters, random_state=random_seed)
            
        elif clustering_type == 'hierarchical':
            return HierarchicalClustering(n_clusters=n_clusters)
            
        elif clustering_type == 'dynamic_kmeans':
            params = config.CLUSTERING_PARAMS['dynamic_kmeans'].copy()
            params['n_clusters'] = n_clusters
            params['random_state'] = random_seed
            return DynamicKMeansClustering(**params)
            
        elif clustering_type == 'k_shape':
            return KShapeClustering(n_clusters=n_clusters, random_state=random_seed)
            
        elif clustering_type == 'dtw_kmedoids':
            return DTWKMedoidsClustering(n_clusters=n_clusters, random_state=random_seed)
            
        else:
            raise ValueError(f"Unknown clustering type: {clustering_type}")
            
    def _evaluate_model_performance(self, model, X, y):
        """Evaluate predictive model performance."""
        from sklearn.metrics import mean_absolute_error, r2_score
        
        y_pred = model.predict(X)
        
        return {
            'mae': mean_absolute_error(y, y_pred),
            'r2': r2_score(y, y_pred)
        }
        
    def _aggregate_results(self, results_list: list) -> dict:
        """Aggregate results across multiple random seeds."""
        
        # Extract metrics from all runs
        all_metrics = [r['metrics'] for r in results_list]
        
        # Compute mean and confidence intervals
        aggregated = {}
        
        for metric_name in all_metrics[0].keys():
            if metric_name == 'random_seed':
                continue
                
            values = [m[metric_name] for m in all_metrics if not np.isnan(m[metric_name])]
            
            if values:
                aggregated[f"{metric_name}_mean"] = np.mean(values)
                aggregated[f"{metric_name}_std"] = np.std(values)
                aggregated[f"{metric_name}_ci_95"] = 1.96 * np.std(values) / np.sqrt(len(values))
            else:
                aggregated[f"{metric_name}_mean"] = np.nan
                aggregated[f"{metric_name}_std"] = np.nan
                aggregated[f"{metric_name}_ci_95"] = np.nan
                
        aggregated['n_runs'] = len(results_list)
        
        return aggregated
        
    def _run_sensitivity_analysis(self, data_dict: dict, n_clusters_list: list) -> dict:
        """Run sensitivity analysis for hyperparameters."""
        
        self.logger.info("Running sensitivity analysis for dynamic K-means")
        
        # Parameter ranges for sensitivity analysis
        param_grid = {
            'alpha': [0.1, 0.5, 1.0, 2.0, 5.0],
            'beta': [0.1, 0.5, 1.0, 2.0, 5.0],
            'gamma': [0.1, 0.5, 1.0, 2.0, 5.0]
        }
        
        sensitivity_results = {}
        
        for n_clusters in n_clusters_list:
            cluster_results = []
            
            for params in ParameterGrid(param_grid):
                # Run experiment with these parameters
                try:
                    result = self._run_sensitivity_experiment(
                        data_dict, params, n_clusters
                    )
                    result.update(params)
                    cluster_results.append(result)
                    
                except Exception as e:
                    self.logger.error(f"Error in sensitivity analysis with params {params}: {e}")
                    continue
                    
            sensitivity_results[f"K_{n_clusters}"] = cluster_results
            
        return sensitivity_results
        
    def _run_sensitivity_experiment(self, data_dict: dict, params: dict, n_clusters: int) -> dict:
        """Run single sensitivity analysis experiment."""
        
        X = data_dict['X']
        y = data_dict['y']
        timestamps = data_dict['timestamps']
        contexts = data_dict['contexts']
        entity_ids = data_dict['entity_ids']
        
        # Use SHAP influence with gradient boost (best performing combination)
        model = GradientBoostModel(**config.MODEL_PARAMS['gradient_boost'])
        model.fit(X, y)
        
        influence_gen = ShapInfluence()
        Z = influence_gen.generate_influence(model, X)
        
        # Dynamic K-means with specified parameters
        clustering_params = {
            'n_clusters': n_clusters,
            'alpha': params['alpha'],
            'beta': params['beta'], 
            'gamma': params['gamma'],
            'random_state': 42
        }
        
        clustering_alg = DynamicKMeansClustering(**clustering_params)
        clustering_alg.fit(Z, timestamps=timestamps, contexts=contexts, entity_ids=entity_ids)
        
        # Evaluate
        transition_matrix = clustering_alg.get_transition_matrix(entity_ids, timestamps)
        
        metrics = self.evaluator.evaluate_comprehensive(
            Z=Z,
            labels=clustering_alg.labels_,
            timestamps=timestamps,
            contexts=contexts,
            entity_ids=entity_ids,
            transition_matrix=transition_matrix
        )
        
        return metrics
        
    def _save_results(self, results: dict, dataset_name: str, intermediate: bool = False):
        """Save results to JSON file."""
        
        suffix = "_intermediate" if intermediate else ""
        output_file = self.output_dir / f"{dataset_name}_results{suffix}.json"
        
        # Convert numpy arrays to lists for JSON serialization
        serializable_results = self._make_json_serializable(results)
        
        with open(output_file, 'w') as f:
            json.dump(serializable_results, f, indent=2)
            
        self.logger.info(f"Results saved to {output_file}")
        
    def _make_json_serializable(self, obj):
        """Convert numpy arrays and other non-serializable objects to JSON-compatible format."""
        
        if isinstance(obj, dict):
            return {key: self._make_json_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._make_json_serializable(item) for item in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.float64, np.float32)):
            return float(obj)
        elif pd.isna(obj):
            return None
        else:
            return obj
            
    def _generate_visualizations(self, results: dict, dataset_name: str):
        """Generate visualizations from results."""
        
        self.logger.info("Generating visualizations")
        
        # Create visualization directory
        viz_dir = self.output_dir / "visualizations" / dataset_name
        viz_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate comparison plots, heatmaps, etc.
        try:
            self.visualizer.create_comprehensive_plots(results, viz_dir)
        except Exception as e:
            self.logger.error(f"Error generating visualizations: {e}")


def main():
    """Main execution function."""
    
    parser = argparse.ArgumentParser(description="Run comprehensive clustering experiments")
    
    parser.add_argument("--dataset", type=str, required=True,
                       choices=["energy_data", "steel_industry"],
                       help="Dataset to use")
    
    parser.add_argument("--output_dir", type=str, default="results/comprehensive",
                       help="Output directory for results")
    
    parser.add_argument("--n_clusters", type=int, nargs="+", default=[3, 5],
                       help="Number of clusters to test")
    
    parser.add_argument("--methods_subset", type=str, nargs="*", default=None,
                       help="Subset of methods to run (for debugging)")
    
    parser.add_argument("--n_jobs", type=int, default=-1,
                       help="Number of parallel jobs")
    
    parser.add_argument("--random_seeds", type=int, nargs="+", 
                       default=[42, 123, 456, 789, 101112],
                       help="Random seeds for experiments")
    
    args = parser.parse_args()
    
    # Create experiment runner
    runner = ComprehensiveExperimentRunner(
        output_dir=args.output_dir,
        random_seeds=args.random_seeds,
        n_jobs=args.n_jobs
    )
    
    # Run experiments
    results = runner.run_comprehensive_experiments(
        dataset_name=args.dataset,
        n_clusters_list=args.n_clusters,
        methods_subset=args.methods_subset
    )
    
    print(f"Experiments completed. Results saved to {args.output_dir}")


if __name__ == "__main__":
    main()
