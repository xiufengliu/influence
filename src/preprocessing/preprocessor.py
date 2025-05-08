"""
Data preprocessing utilities for the Dynamic Influence-Based Clustering Framework.
"""

import logging
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer


class Preprocessor:
    """
    Class for preprocessing energy consumption data.

    This class handles common preprocessing tasks such as:
    - Handling missing values
    - Feature normalization
    - Temporal alignment
    - Feature extraction
    """

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.scaler = MinMaxScaler()
        self.imputer = SimpleImputer(strategy='mean')

    def preprocess(self, data, dataset_name=None):
        """
        Preprocess the input data.

        Parameters
        ----------
        data : pandas.DataFrame
            The raw data to preprocess.
        dataset_name : str, optional
            Name of the dataset being processed, used for dataset-specific preprocessing.

        Returns
        -------
        X : numpy.ndarray
            Feature matrix.
        y : numpy.ndarray
            Target variable.
        t : numpy.ndarray
            Timestamps.
        c : numpy.ndarray
            Contextual attributes.
        """
        self.logger.info("Preprocessing data...")

        # Make a copy to avoid modifying the original data
        df = data.copy()

        # Dataset-specific preprocessing
        if dataset_name == "energy_data":
            # UCI Appliances Energy Prediction dataset
            # Extract timestamps
            t = pd.to_datetime(df['date'])
            df = df.drop('date', axis=1)

            # Extract target variable
            y = df['Appliances'].values
            df = df.drop('Appliances', axis=1)

            # Add time features
            df['hour'] = t.dt.hour
            df['day'] = t.dt.day
            df['month'] = t.dt.month
            df['dayofweek'] = t.dt.dayofweek

            # Extract contextual attributes
            contextual_cols = ['hour', 'day', 'month', 'dayofweek']
            c = df[contextual_cols].values
            df = df.drop(contextual_cols, axis=1)

        elif dataset_name == "steel_industry":
            # Steel Industry Energy Consumption dataset
            # Extract timestamps - using dayfirst=True for European date format (DD/MM/YYYY)
            t = pd.to_datetime(df['date'], dayfirst=True)
            df = df.drop('date', axis=1)

            # Extract target variable
            y = df['Usage_kWh'].values
            df = df.drop('Usage_kWh', axis=1)

            # Extract contextual attributes
            contextual_cols = ['WeekStatus', 'Day_of_week', 'Load_Type']

            # Convert categorical variables to numeric
            for col in contextual_cols:
                if df[col].dtype == 'object':
                    df[col] = pd.Categorical(df[col]).codes

            c = df[contextual_cols].values
            df = df.drop(contextual_cols, axis=1)

            # Drop NSM (seconds from midnight) as it's redundant with timestamp
            if 'NSM' in df.columns:
                df = df.drop('NSM', axis=1)

        else:
            # Generic preprocessing for other datasets
            # Extract timestamps
            if 'timestamp' in df.columns:
                t = pd.to_datetime(df['timestamp'])
                df = df.drop('timestamp', axis=1)
            elif 'date' in df.columns:
                t = pd.to_datetime(df['date'])
                df = df.drop('date', axis=1)
            else:
                self.logger.warning("No timestamp column found. Creating artificial timestamps.")
                t = pd.date_range(start='2022-01-01', periods=len(df), freq='D')

            # Extract target variable
            if 'energy_consumption' in df.columns:
                y = df['energy_consumption'].values
                df = df.drop('energy_consumption', axis=1)
            elif 'peak_load' in df.columns:
                y = df['peak_load'].values
                df = df.drop('peak_load', axis=1)
            else:
                self.logger.warning("No target variable found. Using the last column as target.")
                y = df.iloc[:, -1].values
                df = df.iloc[:, :-1]

            # Extract contextual attributes
            contextual_cols = ['quarter', 'year', 'month', 'day', 'hour', 'dayofweek']
            c_cols = [col for col in contextual_cols if col in df.columns]

            if c_cols:
                c = df[c_cols].values
                df = df.drop(c_cols, axis=1)
            else:
                # Create contextual attributes from timestamps
                c = np.column_stack([
                    getattr(t.dt, attr) if hasattr(t.dt, attr) else np.zeros(len(t))
                    for attr in ['quarter', 'year', 'month', 'day', 'hour', 'dayofweek']
                ])

        # Handle missing values
        self.logger.info("Handling missing values...")
        X = self.imputer.fit_transform(df)

        # Normalize features
        self.logger.info("Normalizing features...")
        X = self.scaler.fit_transform(X)

        self.logger.info(f"Preprocessing complete. X shape: {X.shape}, y shape: {y.shape}")

        return X, y, t, c

    def add_time_features(self, df, timestamp_col='timestamp'):
        """
        Add time-based features to the dataframe.

        Parameters
        ----------
        df : pandas.DataFrame
            Input dataframe.
        timestamp_col : str, default='timestamp'
            Name of the timestamp column.

        Returns
        -------
        pandas.DataFrame
            Dataframe with added time features.
        """
        if timestamp_col not in df.columns:
            self.logger.warning(f"Timestamp column '{timestamp_col}' not found in dataframe.")
            return df

        # Convert to datetime if not already
        if not pd.api.types.is_datetime64_any_dtype(df[timestamp_col]):
            df[timestamp_col] = pd.to_datetime(df[timestamp_col])

        # Extract time features
        df['hour'] = df[timestamp_col].dt.hour
        df['day'] = df[timestamp_col].dt.day
        df['month'] = df[timestamp_col].dt.month
        df['year'] = df[timestamp_col].dt.year
        df['dayofweek'] = df[timestamp_col].dt.dayofweek
        df['quarter'] = df[timestamp_col].dt.quarter

        # Add cyclical features for hour, day, month
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)

        return df

    def align_temporal_data(self, df, timestamp_col='timestamp', freq='H'):
        """
        Align temporal data to a consistent frequency.

        Parameters
        ----------
        df : pandas.DataFrame
            Input dataframe.
        timestamp_col : str, default='timestamp'
            Name of the timestamp column.
        freq : str, default='H'
            Frequency to resample to (e.g., 'H' for hourly, 'D' for daily).

        Returns
        -------
        pandas.DataFrame
            Resampled dataframe.
        """
        if timestamp_col not in df.columns:
            self.logger.warning(f"Timestamp column '{timestamp_col}' not found in dataframe.")
            return df

        # Convert to datetime if not already
        if not pd.api.types.is_datetime64_any_dtype(df[timestamp_col]):
            df[timestamp_col] = pd.to_datetime(df[timestamp_col])

        # Set timestamp as index
        df = df.set_index(timestamp_col)

        # Resample and interpolate
        df_resampled = df.resample(freq).mean()
        df_resampled = df_resampled.interpolate(method='time')

        # Reset index to get timestamp as column again
        df_resampled = df_resampled.reset_index()

        return df_resampled
