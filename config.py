"""
Configuration settings for the Dynamic Influence-Based Clustering Framework.
"""

import os
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
RESULTS_DIR = DATA_DIR / "results"

# Create directories if they don't exist
for directory in [RAW_DATA_DIR, PROCESSED_DATA_DIR, RESULTS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Model parameters
MODEL_PARAMS = {
    "gradient_boost": {
        "n_estimators": 100,
        "max_depth": 6,
        "learning_rate": 0.1,
        "random_state": 42
    },
    "lstm": {
        "input_dim": None, # This should be set dynamically based on dataset features
        "hidden_dim": 50,
        "n_layers": 2,
        "output_dim": 1,
        "epochs": 50,
        "batch_size": 32,
        "learning_rate": 1e-3,
        "random_state": 42
    },
    "transformer": {
        "input_dim": None, # This should be set dynamically based on dataset features
        "n_heads": 8,
        "n_layers": 2,
        "output_dim": 1,
        "epochs": 50,
        "batch_size": 32,
        "learning_rate": 1e-3,
        "random_state": 42
    }
}

# Influence parameters
INFLUENCE_PARAMS = {
    "shap": {
        "n_samples": 50,
        "random_state": 42
    },
    "lime": {
        "n_samples": 100,
        "random_state": 42
    },
    "spearman": {
        "method": "spearman"
    },
    "integrated_gradients": {
        "n_steps": 50,
        "random_state": 42
    },
    "deepshap": {
        "n_samples": 100,
        "random_state": 42
    },
    "hessian": {}
}

# Clustering parameters
CLUSTERING_PARAMS = {
    "kmeans": {
        "n_clusters": 3,
        "random_state": 42,
        "n_init": 10
    },
    "hierarchical": {
        "n_clusters": 3,
        "linkage": "ward"
    },
    "spectral": {
        "n_clusters": 3,
        "random_state": 42,
        "affinity": "rbf"
    },
    "dynamic_kmeans": {
        "n_clusters": 3,
        "random_state": 42,
        "n_init": 10,
        "alpha": 1.0,
        "beta": 1.0,
        "gamma": 1.0, # Added gamma for contextual alignment
        "window_size": 24
    }
}

# Temporal parameters
TEMPORAL_PARAMS = {
    "alpha": 0.7,  # Weight for cluster cohesion
    "beta": 0.2,   # Weight for temporal smoothness
    
}

# Evaluation parameters
EVALUATION_PARAMS = {
    "test_size": 0.2,
    "random_state": 42,
    "cv_folds": 5
}

# Logging parameters
LOGGING_PARAMS = {
    "level": "DEBUG",
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
}

# Dataset parameters
DATASET_PARAMS = {
    "energy_data": {
        "target_column": "Appliances",
        "timestamp_column": "date",
        "context_columns": ["hour", "dayofweek"]
    },
    "steel_industry": {
        "target_column": "Usage_kWh",
        "timestamp_column": "date",
        "context_columns": ["WeekStatus", "Day_of_week", "Load_Type"]
    },
    "household_power_consumption": {
        "target_column": "Global_active_power",
        "timestamp_column": "datetime",
        "context_columns": []
    },
    "air_quality": {
        "target_column": "T",
        "timestamp_column": "datetime",
        "context_columns": []
    }
}

# Experiment parameters for TNNLS submission
TNNLS_EXPERIMENT_PARAMS = {
    "n_clusters_list": [3, 5, 7],
    "random_seeds": [42, 123, 456, 789, 101112],
    "n_jobs": -1,
    "verbose": True
}