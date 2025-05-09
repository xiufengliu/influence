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
    }
}

# Influence parameters
INFLUENCE_PARAMS = {
    "shap": {
        "n_samples": 50,  # Reduced from 100 to save memory
        "random_state": 42
    },
    "lime": {
        "n_samples": 1000,  # Reduced from 5000 to save memory
        "random_state": 42
    },
    "spearman": {
        "method": "spearman"
    }
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
    }
}

# Temporal parameters
TEMPORAL_PARAMS = {
    "alpha": 0.7,  # Weight for cluster cohesion
    "beta": 0.2,   # Weight for temporal smoothness
    "gamma": 0.1   # Weight for contextual coherence
}

# Evaluation parameters
EVALUATION_PARAMS = {
    "test_size": 0.2,
    "random_state": 42,
    "cv_folds": 5
}

# Logging parameters
LOGGING_PARAMS = {
    "level": "INFO",
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
}

# Dataset parameters
DATASET_PARAMS = {
    "building_genome": {
        "target_column": "energy_consumption",
        "timestamp_column": "timestamp",
        "context_columns": ["building_type", "day_type", "season"]
    },
    "industrial_site1": {
        "target_column": "energy_consumption",
        "timestamp_column": "timestamp",
        "context_columns": ["production_line", "shift"]
    },
    "industrial_site2": {
        "target_column": "energy_consumption",
        "timestamp_column": "timestamp",
        "context_columns": ["operation_mode", "product_type"]
    },
    "industrial_site3": {
        "target_column": "energy_consumption",
        "timestamp_column": "timestamp",
        "context_columns": ["facility_area", "equipment_status"]
    },
    "energy_data": {
        "target_column": "Appliances",
        "timestamp_column": "date",
        "context_columns": ["hour", "day", "month", "dayofweek"]
    },
    "steel_industry": {
        "target_column": "Usage_kWh",
        "timestamp_column": "date",
        "context_columns": ["WeekStatus", "Day_of_week", "Load_Type"]
    }
}

# Experiment parameters for TNNLS submission
TNNLS_EXPERIMENT_PARAMS = {
    "n_clusters_list": [3, 5, 7],
    "random_seeds": [42, 123, 456, 789, 101112],
    "n_jobs": -1,
    "verbose": True
}
