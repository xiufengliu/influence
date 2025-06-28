import logging
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer


class Preprocessor:
    """
    Class for preprocessing energy consumption data.
    """

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.scaler = MinMaxScaler()
        self.imputer = SimpleImputer(strategy='mean')

    def preprocess(self, data, dataset_name=None):
        """
        Preprocess the input data.
        """
        self.logger.info(f"Preprocessing data for {dataset_name}...")

        df = data.copy()

        if dataset_name == "household_power_consumption":
            df['datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'])
            df = df.drop(['Date', 'Time'], axis=1)
            df = df.set_index('datetime')
            df = df.replace('?', np.nan)
            df = df.astype(float)
            y = df['Global_active_power'].values
            df = df.drop('Global_active_power', axis=1)
            t = df.index
            c = df[[]].values # No specific context columns

        elif dataset_name == "air_quality":
            df['datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'])
            df = df.drop(['Date', 'Time'], axis=1)
            df = df.set_index('datetime')
            y = df['T'].values # Using temperature as target
            df = df.drop('T', axis=1)
            t = df.index
            c = df[[]].values # No specific context columns

        elif dataset_name == "energy_data":
            t = pd.to_datetime(df['date'])
            df = df.drop('date', axis=1)
            y = df['Appliances'].values
            df = df.drop('Appliances', axis=1)
            df['hour'] = t.dt.hour
            df['dayofweek'] = t.dt.dayofweek
            contextual_cols = ['hour', 'dayofweek']
            c = df[contextual_cols].values
            df = df.drop(contextual_cols, axis=1)

        elif dataset_name == "steel_industry":
            t = pd.to_datetime(df['date'], dayfirst=True)
            df = df.drop('date', axis=1)
            y = df['Usage_kWh'].values
            df = df.drop('Usage_kWh', axis=1)
            contextual_cols = ['WeekStatus', 'Day_of_week', 'Load_Type']
            for col in contextual_cols:
                if df[col].dtype == 'object':
                    df[col] = pd.Categorical(df[col]).codes
            c = df[contextual_cols].values
            df = df.drop(contextual_cols, axis=1)
            if 'NSM' in df.columns:
                df = df.drop('NSM', axis=1)

        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")

        X = self.imputer.fit_transform(df)
        X = self.scaler.fit_transform(X)

        # For now, assume a single entity for the entire dataset
        entity_ids = np.zeros(X.shape[0], dtype=int)

        self.logger.info(f"Preprocessing complete. X shape: {X.shape}, y shape: {y.shape}")

        return X, y, t, c, entity_ids