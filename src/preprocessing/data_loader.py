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

    Parameters
    ----------
    dataset_name : str
        Name of the dataset to load. Options:
        - 'building_genome', 'industrial_site1', 'industrial_site2', 'industrial_site3': Synthetic datasets
        - 'energy_data': UCI Appliances Energy Prediction dataset
        - 'steel_industry': Steel Industry Energy Consumption dataset
    """

    def __init__(self, dataset_name):
        self.dataset_name = dataset_name
        self.logger = logging.getLogger(__name__)

        # Define dataset paths
        self.dataset_paths = {
            "building_genome": config.RAW_DATA_DIR / "building_genome.csv",
            "industrial_site1": config.RAW_DATA_DIR / "industrial_site1.csv",
            "industrial_site2": config.RAW_DATA_DIR / "industrial_site2.csv",
            "industrial_site3": config.RAW_DATA_DIR / "industrial_site3.csv",
            "energy_data": config.RAW_DATA_DIR / "energydata_complete.csv",
            "steel_industry": config.RAW_DATA_DIR / "Steel_industry_data.csv"
        }

        # Check if dataset exists
        if dataset_name in ["energy_data", "steel_industry"]:
            if not os.path.exists(self.dataset_paths[dataset_name]):
                self.logger.error(f"Real dataset {dataset_name} not found at {self.dataset_paths[dataset_name]}")
                raise FileNotFoundError(f"Dataset {dataset_name} not found")
        elif not os.path.exists(self.dataset_paths[dataset_name]):
            self.logger.warning(f"Dataset {dataset_name} not found at {self.dataset_paths[dataset_name]}")
            self.logger.info(f"Creating synthetic data for {dataset_name}")
            self._create_synthetic_data()

    def load_data(self, preprocess=True):
        """
        Load the specified dataset.

        Parameters
        ----------
        preprocess : bool, default=True
            Whether to preprocess the data after loading.

        Returns
        -------
        If preprocess=True:
            X : numpy.ndarray
                Feature matrix.
            y : numpy.ndarray
                Target variable.
            t : numpy.ndarray
                Timestamps.
            c : numpy.ndarray
                Contextual attributes.
        If preprocess=False:
            pandas.DataFrame
                The loaded dataset.
        """
        self.logger.info(f"Loading {self.dataset_name} dataset")

        try:
            # Load the dataset
            if self.dataset_name in ["energy_data", "steel_industry"]:
                self.logger.info(f"Loading real dataset: {self.dataset_name}")
            else:
                self.logger.info(f"Loading synthetic dataset: {self.dataset_name}")

            data = pd.read_csv(self.dataset_paths[self.dataset_name])
            self.logger.info(f"Successfully loaded {self.dataset_name} dataset with shape {data.shape}")

            # Preprocess if requested
            if preprocess:
                preprocessor = Preprocessor()
                X, y, t, c = preprocessor.preprocess(data, dataset_name=self.dataset_name)
                return X, y, t, c
            else:
                return data

        except Exception as e:
            self.logger.error(f"Error loading {self.dataset_name} dataset: {e}")
            raise

    def _create_synthetic_data(self):
        """
        Create synthetic data for demonstration purposes when real data is not available.
        """
        self.logger.info(f"Creating synthetic data for {self.dataset_name}")

        # Create directory if it doesn't exist
        Path(self.dataset_paths[self.dataset_name]).parent.mkdir(parents=True, exist_ok=True)

        # Generate synthetic data based on dataset type
        if self.dataset_name == "building_genome":
            self._create_building_genome_data()
        elif self.dataset_name.startswith("industrial_site"):
            site_num = int(self.dataset_name[-1])
            self._create_industrial_site_data(site_num)

    def _create_building_genome_data(self):
        """
        Create synthetic data for the Building Genome dataset.
        """
        np.random.seed(42)

        # Create hourly timestamps for a year
        timestamps = pd.date_range(start='2016-01-01', end='2016-12-31 23:00:00', freq='H')
        n_samples = len(timestamps)

        # Create building features (10 buildings)
        n_buildings = 10
        building_cols = [f"building_{i}" for i in range(1, n_buildings + 1)]

        # Generate synthetic consumption data with daily and seasonal patterns
        data = pd.DataFrame(index=timestamps)

        # Add time-based features
        data['hour'] = data.index.hour
        data['day'] = data.index.day
        data['month'] = data.index.month
        data['dayofweek'] = data.index.dayofweek

        # Generate building consumption with patterns
        for i, col in enumerate(building_cols):
            # Base load
            base_load = np.random.uniform(50, 200)

            # Daily pattern (higher during working hours)
            hourly_pattern = np.sin(np.pi * data['hour'] / 12) + 1

            # Weekly pattern (lower on weekends)
            weekly_pattern = 0.7 + 0.3 * (data['dayofweek'] < 5).astype(float)

            # Seasonal pattern (higher in winter and summer)
            monthly_pattern = 0.8 + 0.4 * np.sin(np.pi * data['month'] / 6)

            # Random noise
            noise = np.random.normal(0, 0.1, n_samples)

            # Combine patterns
            data[col] = base_load * (1 + 0.3 * hourly_pattern * weekly_pattern * monthly_pattern + noise)

        # Create target variable (peak load)
        total_consumption = data[building_cols].sum(axis=1)
        peak_threshold = np.percentile(total_consumption, 90)
        data['peak_load'] = (total_consumption > peak_threshold).astype(int)

        # Add timestamp as a column
        data['timestamp'] = data.index

        # Save to CSV
        data.to_csv(self.dataset_paths[self.dataset_name], index=False)
        self.logger.info(f"Synthetic {self.dataset_name} dataset created with shape {data.shape}")

    def _create_industrial_site_data(self, site_num):
        """
        Create synthetic data for industrial sites.

        Parameters
        ----------
        site_num : int
            Site number (1, 2, or 3).
        """
        np.random.seed(42 + site_num)

        # Create quarterly timestamps for a year
        timestamps = pd.date_range(start='2022-01-01', end='2022-12-31', freq='Q')
        n_samples = len(timestamps)

        # Define features based on site number
        if site_num == 1:
            # Site 1: Eight features for textile processing
            feature_cols = [
                'electric_consumption', 'vapor_flow_rate', 'water_flow_rate',
                'dyeing_process', 'ironing_process', 'power_factor',
                'production_unit1', 'production_unit2'
            ]
            n_features = len(feature_cols)

            # Create correlation matrix for realistic dependencies
            corr_matrix = np.eye(n_features)
            corr_matrix[0, 3:] = 0.7  # Electric consumption correlated with production
            corr_matrix[3:, 0] = 0.7
            corr_matrix[1, 3:5] = 0.6  # Vapor flow correlated with dyeing and ironing
            corr_matrix[3:5, 1] = 0.6

        elif site_num == 2:
            # Site 2: Eight features emphasizing electrical consumption
            feature_cols = [
                'total_electric', 'production_area', 'uta_systems',
                'compressor_units', 'office_spaces', 'transformer1',
                'transformer2', 'power_factor'
            ]
            n_features = len(feature_cols)

            # Create correlation matrix with strong correlations (more regular patterns)
            corr_matrix = np.eye(n_features)
            corr_matrix[0, 1:5] = 0.9  # Total electric strongly correlated with areas
            corr_matrix[1:5, 0] = 0.9
            corr_matrix[5:7, 0] = 0.95  # Transformers almost perfectly correlated with total
            corr_matrix[0, 5:7] = 0.95

        else:  # site_num == 3
            # Site 3: Five features focusing on transformer readings
            feature_cols = [
                'transformer_main', 'technical_flow_rate', 'weaving_area',
                'ironing_area', 'power_factor'
            ]
            n_features = len(feature_cols)

            # Create moderate correlation matrix
            corr_matrix = np.eye(n_features)
            corr_matrix[0, 2:4] = 0.6  # Transformer correlated with production areas
            corr_matrix[2:4, 0] = 0.6

        # Generate multivariate normal data with the correlation structure
        data_raw = np.random.multivariate_normal(
            mean=np.random.uniform(50, 200, n_features),
            cov=corr_matrix * np.outer(
                np.random.uniform(10, 50, n_features),
                np.random.uniform(10, 50, n_features)
            ),
            size=n_samples
        )

        # Create DataFrame
        data = pd.DataFrame(data_raw, columns=feature_cols)

        # Add seasonal patterns
        for i, col in enumerate(feature_cols):
            seasonal_factor = 1 + 0.2 * np.sin(np.pi * np.arange(n_samples) / 2)
            data[col] *= seasonal_factor

        # Add timestamp and contextual information
        data['timestamp'] = timestamps
        data['quarter'] = data['timestamp'].dt.quarter
        data['year'] = data['timestamp'].dt.year

        # Add target variable (energy consumption)
        if site_num == 1:
            data['energy_consumption'] = data['electric_consumption'] * 0.8 + data['vapor_flow_rate'] * 0.2
        elif site_num == 2:
            data['energy_consumption'] = data['total_electric']
        else:
            data['energy_consumption'] = data['transformer_main']

        # Save to CSV
        data.to_csv(self.dataset_paths[self.dataset_name], index=False)
        self.logger.info(f"Synthetic {self.dataset_name} dataset created with shape {data.shape}")
