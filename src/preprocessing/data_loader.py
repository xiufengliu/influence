"""
Data loading utilities for the Dynamic Influence-Based Clustering Framework.
"""

import os
import logging
import pandas as pd
import numpy as np
from pathlib import Path

import config
from src.preprocessing.preprocessor import Preprocessor


class DataLoader:
    """
    Class for loading energy consumption datasets.
    """

    def __init__(self, dataset_name):
        self.dataset_name = dataset_name
        self.logger = logging.getLogger(__name__)

        # Define dataset paths
        self.dataset_paths = {
            "energy_data": config.RAW_DATA_DIR / "energydata_complete.csv",
            "steel_industry": config.RAW_DATA_DIR / "Steel_industry_data.csv",
            "household_power_consumption": config.RAW_DATA_DIR / "household_power_consumption.txt",
            "air_quality": config.RAW_DATA_DIR / "AirQualityUCI.csv"
        }

        # Check if dataset exists
        if not os.path.exists(self.dataset_paths[dataset_name]):
            self.logger.error(f"Dataset {dataset_name} not found at {self.dataset_paths[dataset_name]}")
            raise FileNotFoundError(f"Dataset {dataset_name} not found")

    def load_data(self, preprocess=True):
        """
        Load the specified dataset.
        """
        self.logger.info(f"Loading {self.dataset_name} dataset")

        try:
            # Load the dataset
            if self.dataset_name == "household_power_consumption":
                data = pd.read_csv(self.dataset_paths[self.dataset_name], sep=';', low_memory=False)
            elif self.dataset_name == "air_quality":
                data = pd.read_csv(self.dataset_paths[self.dataset_name], sep=';', decimal=',', na_values=['-200'])
            else:
                data = pd.read_csv(self.dataset_paths[self.dataset_name])
            
            self.logger.info(f"Successfully loaded {self.dataset_name} dataset with shape {data.shape}")

            # Preprocess if requested
            if preprocess:
                preprocessor = Preprocessor()
                X, y, t, c, entity_ids = preprocessor.preprocess(data, dataset_name=self.dataset_name)
                return X, y, t, c, entity_ids
            else:
                return data

        except Exception as e:
            self.logger.error(f"Error loading {self.dataset_name} dataset: {e}")
            raise